"""Acoustic causal tracing for experimental Qwen3-TTS SonoEdit runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from .activations import ActivationCapture, get_module_by_name, validate_talker_layer_name


DEFAULT_QWEN3_TTS_1_7B_CANDIDATE_LAYERS = [
    f"talker.model.layers.{index}" for index in range(6, 18)
]


@dataclass(frozen=True)
class CausalTraceScore:
    layer: str
    clean_probability: float
    corrupted_probability: float
    restored_probability: float
    recovery: float


def _forward(model: nn.Module, inputs: Any, forward_fn: Callable[[nn.Module, Any], Any] | None) -> Any:
    if forward_fn is not None:
        return forward_fn(model, inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    raise TypeError("model output must be logits tensor or expose .logits")


def mean_target_probability(logits: torch.Tensor, target_codec0: Any) -> torch.Tensor:
    target = torch.as_tensor(target_codec0, dtype=torch.long, device=logits.device).reshape(-1)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    if target.numel() == 0:
        raise ValueError("target_codec0 cannot be empty")
    count = min(target.numel(), flat_logits.shape[0])
    if count == 0:
        raise ValueError("logits do not contain any frames")
    probs = torch.softmax(flat_logits[:count], dim=-1)
    return probs.gather(-1, target[:count, None]).mean()


def recovery_score(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    restored_logits: torch.Tensor,
    target_codec0: Any,
) -> CausalTraceScore:
    clean = mean_target_probability(clean_logits, target_codec0)
    corrupted = mean_target_probability(corrupted_logits, target_codec0)
    restored = mean_target_probability(restored_logits, target_codec0)
    denom = clean - corrupted
    recovery = torch.zeros_like(restored) if torch.isclose(denom, torch.zeros_like(denom)) else (restored - corrupted) / denom
    return CausalTraceScore(
        layer="",
        clean_probability=float(clean.detach().cpu()),
        corrupted_probability=float(corrupted.detach().cpu()),
        restored_probability=float(restored.detach().cpu()),
        recovery=float(recovery.detach().cpu()),
    )


def run_acoustic_causal_trace(
    model: nn.Module,
    clean_inputs: Any,
    corrupted_inputs: Any,
    target_codec0: Any,
    *,
    candidate_layers: list[str] | None = None,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
) -> list[CausalTraceScore]:
    """Rank layers by target codec-0 probability recovery after activation restore."""

    layers = candidate_layers or DEFAULT_QWEN3_TTS_1_7B_CANDIDATE_LAYERS
    for layer in layers:
        validate_talker_layer_name(layer)

    with torch.no_grad():
        with ActivationCapture(model, layers) as capture:
            clean_logits = _extract_logits(_forward(model, clean_inputs, forward_fn))
            clean_outputs = {name: capture.records.outputs[name] for name in layers}

        corrupted_logits = _extract_logits(_forward(model, corrupted_inputs, forward_fn))
        scores: list[CausalTraceScore] = []
        for layer in layers:
            module = get_module_by_name(model, layer)

            def restore(_module: nn.Module, _args: tuple[Any, ...], _output: Any, *, layer_name: str = layer) -> Any:
                return clean_outputs[layer_name]

            handle = module.register_forward_hook(restore)
            try:
                restored_logits = _extract_logits(_forward(model, corrupted_inputs, forward_fn))
            finally:
                handle.remove()
            score = recovery_score(clean_logits, corrupted_logits, restored_logits, target_codec0)
            scores.append(
                CausalTraceScore(
                    layer=layer,
                    clean_probability=score.clean_probability,
                    corrupted_probability=score.corrupted_probability,
                    restored_probability=score.restored_probability,
                    recovery=score.recovery,
                )
            )
    return sorted(scores, key=lambda item: item.recovery, reverse=True)

