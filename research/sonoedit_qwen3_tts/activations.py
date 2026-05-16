"""Activation capture and layer resolution helpers for Qwen3-TTS talker layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


FORBIDDEN_PREFIXES = ("speaker_encoder", "speech_tokenizer", "talker.code_predictor")
TALKER_LAYER_PREFIX = "talker.model.layers."


def validate_talker_layer_name(name: str) -> None:
    if name.startswith(FORBIDDEN_PREFIXES):
        raise ValueError(f"{name} is not editable in SonoEdit v1")
    if not name.startswith(TALKER_LAYER_PREFIX):
        raise ValueError(f"{name} is outside talker.model.layers")
    suffix = name.removeprefix(TALKER_LAYER_PREFIX)
    if not suffix.isdigit():
        raise ValueError(f"{name} must identify a numeric talker layer")


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"module not found: {name}")
    return modules[name]


def _clone_detached(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_detached(item) for item in value)
    if isinstance(value, list):
        return [_clone_detached(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_detached(item) for key, item in value.items()}
    return value


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError("no tensor found in activation")


@dataclass
class LayerCapture:
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)


class ActivationCapture:
    """Context manager that records inputs and outputs for named modules."""

    def __init__(self, model: nn.Module, layer_names: list[str], *, validate_qwen_layers: bool = True):
        self.model = model
        self.layer_names = layer_names
        self.validate_qwen_layers = validate_qwen_layers
        self.records = LayerCapture()
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationCapture":
        for name in self.layer_names:
            if self.validate_qwen_layers:
                validate_talker_layer_name(name)
            module = get_module_by_name(self.model, name)
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.records.inputs.clear()
        self.records.outputs.clear()

    def _hook(self, name: str):
        def capture(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            self.records.inputs[name] = _clone_detached(args)
            self.records.outputs[name] = _clone_detached(output)

        return capture

    def output_tensor(self, name: str) -> torch.Tensor:
        return _first_tensor(self.records.outputs[name])

