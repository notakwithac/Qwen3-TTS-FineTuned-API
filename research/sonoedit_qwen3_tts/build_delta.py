"""Build SonoEdit delta files from Qwen3-TTS activations and codec targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .activations import get_module_by_name, validate_talker_layer_name
from .nullspace_edit import compute_constrained_update, preferred_projection_weight
from .schema import SonoEditRequest
from .targets import extract_codec_target


@dataclass(frozen=True)
class DeltaBuildResult:
    weight_name: str
    output_delta_norm: float
    weight_delta_norm: float
    relative_weight_delta: float
    residual_preservation_norm: float
    target_frames: int
    preservation_examples: int
    output_delta_scale: float
    semantic_codebook_weight: float = 1.0
    residual_codebook_weight: float = 0.0


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("no tensor found")


def _slice_last_n(sequence: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    if sequence.shape[-2] < count:
        raise ValueError(f"cannot select {count} frames from sequence length {sequence.shape[-2]}")
    return sequence[..., -count:, :]


def _flatten_token_targets(target_codec0: Any, device: torch.device) -> torch.Tensor:
    target = torch.as_tensor(target_codec0, dtype=torch.long, device=device)
    if target.ndim > 1:
        target = target[..., 0]
    target = target.reshape(-1)
    if target.numel() == 0:
        raise ValueError("target codec-0 sequence cannot be empty")
    return target


def _codec_target_matrix(target_codes: Any, device: torch.device) -> torch.Tensor:
    target = torch.as_tensor(target_codes, dtype=torch.long, device=device)
    if target.ndim == 1:
        target = target[:, None]
    if target.ndim != 2:
        raise ValueError("target codec sequence must have shape [frames] or [frames, codebooks]")
    if target.shape[0] == 0:
        raise ValueError("target codec sequence cannot be empty")
    return target


def mean_projection_key(projection_input: torch.Tensor, frame_count: int | None = None) -> torch.Tensor:
    """Average projection inputs into one key vector for a rank-one edit."""

    values = _first_tensor(projection_input).float()
    if values.ndim == 1:
        return values
    if values.ndim < 2:
        raise ValueError("projection input must include a feature dimension")
    if frame_count is not None:
        values = _slice_last_n(values, frame_count)
    return values.reshape(-1, values.shape[-1]).mean(dim=0)


def desired_delta_from_codec_loss(
    codec0_logits: torch.Tensor,
    target_codec_codes: Any,
    projection_output: torch.Tensor,
    *,
    residual_logits: torch.Tensor | None = None,
    semantic_codebook_weight: float = 1.0,
    residual_codebook_weight: float = 0.0,
    output_delta_scale: float = 1.0,
) -> torch.Tensor:
    """Use the negative activation gradient as the desired output-space delta."""

    target = _codec_target_matrix(target_codec_codes, codec0_logits.device)
    frame_count = min(target.shape[0], codec0_logits.shape[-2], projection_output.shape[-2])
    if residual_logits is not None:
        frame_count = min(frame_count, residual_logits.shape[0])
    if frame_count <= 0:
        raise ValueError("no overlapping logits/target frames")

    selected_logits = _slice_last_n(codec0_logits, frame_count).reshape(-1, codec0_logits.shape[-1])
    selected_target = target[:frame_count, 0].reshape(-1)
    loss = float(semantic_codebook_weight) * F.cross_entropy(selected_logits, selected_target)
    if residual_logits is not None and target.shape[1] > 1 and float(residual_codebook_weight) != 0.0:
        residual_groups = min(target.shape[1] - 1, residual_logits.shape[1])
        residual_target = target[:frame_count, 1 : residual_groups + 1]
        residual_selected = residual_logits[:frame_count, :residual_groups, :]
        residual_loss = F.cross_entropy(
            residual_selected.reshape(-1, residual_selected.shape[-1]),
            residual_target.reshape(-1),
        )
        loss = loss + float(residual_codebook_weight) * residual_loss
    grad = torch.autograd.grad(loss, projection_output, retain_graph=True)[0]
    selected_grad = _slice_last_n(grad, frame_count)
    desired = -selected_grad.reshape(-1, selected_grad.shape[-1]).mean(dim=0)
    return desired.detach().float() * float(output_delta_scale)


def build_delta_from_captured_tensors(
    *,
    weight_name: str,
    current_weight: torch.Tensor,
    target_codec0: Any,
    codec0_logits: torch.Tensor,
    projection_input: torch.Tensor,
    projection_output: torch.Tensor,
    preservation_inputs: list[torch.Tensor],
    residual_logits: torch.Tensor | None = None,
    semantic_codebook_weight: float = 1.0,
    residual_codebook_weight: float = 0.0,
    output_delta_scale: float = 1.0,
    max_relative_delta: float | None = 1e-3,
) -> tuple[torch.Tensor, DeltaBuildResult]:
    """Compute a constrained weight delta from one differentiable forward pass."""

    target_tokens = _codec_target_matrix(target_codec0, codec0_logits.device)
    frame_count = min(target_tokens.shape[0], codec0_logits.shape[-2], projection_output.shape[-2])
    if residual_logits is not None:
        frame_count = min(frame_count, residual_logits.shape[0])
    desired_delta = desired_delta_from_codec_loss(
        codec0_logits,
        target_tokens[:frame_count, :],
        projection_output,
        residual_logits=residual_logits,
        semantic_codebook_weight=semantic_codebook_weight,
        residual_codebook_weight=residual_codebook_weight,
        output_delta_scale=output_delta_scale,
    )
    return build_delta_from_output_delta(
        weight_name=weight_name,
        current_weight=current_weight,
        projection_input=projection_input,
        desired_delta=desired_delta,
        frame_count=frame_count,
        preservation_inputs=preservation_inputs,
        semantic_codebook_weight=semantic_codebook_weight,
        residual_codebook_weight=residual_codebook_weight,
        output_delta_scale=output_delta_scale,
        max_relative_delta=max_relative_delta,
    )


def build_delta_from_output_delta(
    *,
    weight_name: str,
    current_weight: torch.Tensor,
    projection_input: torch.Tensor,
    desired_delta: torch.Tensor,
    frame_count: int,
    preservation_inputs: list[torch.Tensor],
    semantic_codebook_weight: float = 1.0,
    residual_codebook_weight: float = 0.0,
    output_delta_scale: float = 1.0,
    max_relative_delta: float | None = 1e-3,
) -> tuple[torch.Tensor, DeltaBuildResult]:
    """Compute a constrained weight delta from a precomputed output-space delta."""

    target_key = mean_projection_key(projection_input, frame_count=frame_count)
    preservation_keys = torch.stack(
        [mean_projection_key(item).to(device=target_key.device, dtype=target_key.dtype) for item in preservation_inputs],
        dim=0,
    ) if preservation_inputs else torch.empty(0, target_key.numel(), device=target_key.device, dtype=target_key.dtype)

    edit = compute_constrained_update(
        target_key=target_key,
        desired_delta=desired_delta,
        preservation_keys=preservation_keys,
        weight_name=weight_name,
    )
    delta = edit.delta.detach().cpu()
    weight_norm = float(current_weight.detach().float().norm().cpu())
    delta_norm = float(delta.float().norm())
    relative = 0.0 if weight_norm == 0.0 else delta_norm / weight_norm
    if max_relative_delta is not None and relative > float(max_relative_delta):
        scale = float(max_relative_delta) / relative
        delta = delta * scale
        delta_norm = float(delta.float().norm())
        relative = 0.0 if weight_norm == 0.0 else delta_norm / weight_norm

    result = DeltaBuildResult(
        weight_name=weight_name,
        output_delta_norm=float(desired_delta.norm().detach().cpu()),
        weight_delta_norm=delta_norm,
        relative_weight_delta=relative,
        residual_preservation_norm=edit.residual_preservation_norm,
        target_frames=int(frame_count),
        preservation_examples=len(preservation_inputs),
        output_delta_scale=float(output_delta_scale),
        semantic_codebook_weight=float(semantic_codebook_weight),
        residual_codebook_weight=float(residual_codebook_weight),
    )
    return delta, result


class ProjectionCapture:
    """Capture a projection module's input/output while keeping gradient flow."""

    def __init__(self, model: torch.nn.Module, module_name: str):
        self.model = model
        self.module_name = module_name
        self.input: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._handle: Any = None

    def __enter__(self) -> "ProjectionCapture":
        module = get_module_by_name(self.model, self.module_name)
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None

    def _hook(self, module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        self.input = _first_tensor(args)
        self.output = _first_tensor(output)
        if torch.is_tensor(self.output):
            if not self.output.requires_grad and torch.is_grad_enabled():
                self.output.requires_grad_(True)
            if self.output.requires_grad:
                self.output.retain_grad()


def _projection_module_name(weight_name: str) -> str:
    if not weight_name.endswith(".weight"):
        raise ValueError("editable weight name must end in .weight")
    return weight_name[: -len(".weight")]


def load_qwen_wrapper(checkpoint_path: str, *, device: str, dtype: str, attn_implementation: str):
    from qwen_tts import Qwen3TTSModel

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return Qwen3TTSModel.from_pretrained(
        checkpoint_path,
        device_map=device,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
        compile_enabled=False,
    )


def _tokenize_custom_voice_inputs(wrapper: Any, text: str, instruct: str | None) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])
    instruct_ids = wrapper._build_instruct_ids([instruct or ""])
    return input_ids, instruct_ids


def _build_prefill_inputs(
    qwen_model: torch.nn.Module,
    input_ids: list[torch.Tensor],
    instruct_ids: list[torch.Tensor | None],
    *,
    speaker: str | None,
    language: str,
    non_streaming_mode: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable subset of Qwen3TTSForConditionalGeneration.generate prefill setup."""

    model = qwen_model
    talker_input_embeds = [[] for _ in range(len(input_ids))]
    if instruct_ids is not None:
        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(
                    model.talker.text_projection(model.talker.get_text_embeddings()(instruct_id))
                )

    first_dtype = input_ids[0].dtype
    tts_bos_embed, tts_eos_embed, tts_pad_embed = model.talker.text_projection(
        model.talker.get_text_embeddings()(
            torch.tensor(
                [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                device=model.talker.device,
                dtype=first_dtype,
            )
        )
    ).chunk(3, dim=1)

    all_lens = [x.shape[1] for x in input_ids]
    flat_input = torch.cat(input_ids, dim=1)
    flat_embed = model.talker.text_projection(model.talker.get_text_embeddings()(flat_input))
    batched_input_embeds = torch.split(flat_embed, all_lens, dim=1)
    trailing_text_hiddens = []

    speakers = [speaker] * len(input_ids)
    languages = [language] * len(input_ids)
    for index, (input_id, sample_language, sample_speaker) in enumerate(zip(input_ids, languages, speakers)):
        if sample_speaker == "" or sample_speaker is None:
            speaker_embed = None
        else:
            speaker_key = sample_speaker.lower()
            if speaker_key not in model.config.talker_config.spk_id:
                raise NotImplementedError(f"Speaker {sample_speaker} not implemented")
            spk_id = model.config.talker_config.spk_id[speaker_key]
            speaker_embed = model.talker.get_input_embeddings()(
                torch.tensor(spk_id, device=model.talker.device, dtype=input_id.dtype)
            )
            if speaker_embed.ndim == 1:
                speaker_embed = speaker_embed.view(1, 1, -1)
            elif speaker_embed.ndim == 2:
                speaker_embed = speaker_embed.unsqueeze(0)

        if sample_language.lower() == "auto":
            language_id = None
        else:
            language_key = sample_language.lower()
            if language_key not in model.config.talker_config.codec_language_id:
                raise NotImplementedError(f"Language {sample_language} not implemented")
            language_id = model.config.talker_config.codec_language_id[language_key]

        if language_id is None:
            codec_prefill_list = [[
                model.config.talker_config.codec_nothink_id,
                model.config.talker_config.codec_think_bos_id,
                model.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                model.config.talker_config.codec_think_id,
                model.config.talker_config.codec_think_bos_id,
                language_id,
                model.config.talker_config.codec_think_eos_id,
            ]]

        codec_input_embedding_0 = model.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=model.talker.device, dtype=input_id.dtype)
        )
        codec_input_embedding_1 = model.talker.get_input_embeddings()(
            torch.tensor(
                [[model.config.talker_config.codec_pad_id, model.config.talker_config.codec_bos_id]],
                device=model.talker.device,
                dtype=input_id.dtype,
            )
        )
        if speaker_embed is None:
            codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
        else:
            codec_input_embedding = torch.cat([codec_input_embedding_0, speaker_embed, codec_input_embedding_1], dim=1)

        prefix_embed = torch.cat(
            (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
            dim=1,
        ) + codec_input_embedding[:, :-1]
        role_embed = batched_input_embeds[index][:, :3]
        talker_input_embed = torch.cat((role_embed, prefix_embed), dim=1)
        talker_input_embed = torch.cat(
            [talker_input_embed, batched_input_embeds[index][:, 3:4] + codec_input_embedding[:, -1:]],
            dim=1,
        )
        if non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    torch.cat((batched_input_embeds[index][:, 3:-5], tts_eos_embed), dim=1)
                    + model.talker.get_input_embeddings()(
                        torch.tensor(
                            [[model.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                            device=model.talker.device,
                            dtype=input_id.dtype,
                        )
                    ),
                    tts_pad_embed
                    + model.talker.get_input_embeddings()(
                        torch.tensor(
                            [[model.config.talker_config.codec_bos_id]],
                            device=model.talker.device,
                            dtype=input_id.dtype,
                        )
                    ),
                ],
                dim=1,
            )
            trailing_text_hidden = tts_pad_embed
        else:
            trailing_text_hidden = torch.cat((batched_input_embeds[index][:, 4:-5], tts_eos_embed), dim=1)
        talker_input_embeds[index].append(talker_input_embed)
        trailing_text_hiddens.append(trailing_text_hidden)

    talker_input_embeds = [torch.cat([item for item in embeds if item is not None], dim=1) for embeds in talker_input_embeds]
    original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
    sequences = [t.squeeze(0) for t in talker_input_embeds]
    padded_reversed = torch.nn.utils.rnn.pad_sequence([s.flip(dims=[0]) for s in sequences], batch_first=True, padding_value=0.0)
    talker_input_embeds_tensor = padded_reversed.flip(dims=[1])
    batch_size, max_len = talker_input_embeds_tensor.shape[0], talker_input_embeds_tensor.shape[1]
    indices = torch.arange(max_len).expand(batch_size, -1)
    num_pads = max_len - original_lengths
    talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds_tensor.device)

    pad_embedding_vector = tts_pad_embed.squeeze()
    sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
    trailing_lengths = [s.shape[0] for s in sequences_to_pad]
    padded_hiddens = torch.nn.utils.rnn.pad_sequence(sequences_to_pad, batch_first=True, padding_value=0.0)
    arange_tensor = torch.arange(max(trailing_lengths), device=padded_hiddens.device).expand(len(trailing_lengths), -1)
    lengths_tensor = torch.tensor(trailing_lengths, device=padded_hiddens.device).unsqueeze(1)
    padded_hiddens[arange_tensor >= lengths_tensor] = pad_embedding_vector
    return talker_input_embeds_tensor, talker_attention_mask, padded_hiddens, tts_pad_embed


def _run_qwen_prefill(
    wrapper: Any,
    text: str,
    *,
    speaker: str | None,
    language: str,
    instruct: str | None,
) -> Any:
    input_ids, instruct_ids = _tokenize_custom_voice_inputs(wrapper, text, instruct)
    inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed = _build_prefill_inputs(
        wrapper.model,
        input_ids,
        instruct_ids,
        speaker=speaker,
        language=language,
        non_streaming_mode=True,
    )
    return wrapper.model.talker(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        output_hidden_states=True,
        use_cache=False,
    )


def _last_hidden_states(output: Any) -> torch.Tensor:
    hidden_states = output.hidden_states
    if isinstance(hidden_states, tuple) and hidden_states and isinstance(hidden_states[0], tuple):
        return hidden_states[0][-1]
    if isinstance(hidden_states, tuple) and hidden_states and torch.is_tensor(hidden_states[-1]):
        return hidden_states[-1]
    raise ValueError("talker output did not include hidden states")


def residual_codebook_logits(
    talker: torch.nn.Module,
    output: Any,
    target_codec_codes: torch.Tensor,
    frame_count: int,
) -> torch.Tensor | None:
    """Teacher-force residual codebook logits from target codec IDs."""

    if target_codec_codes.ndim != 2 or target_codec_codes.shape[1] <= 1:
        return None
    hidden_states = _slice_last_n(_last_hidden_states(output), frame_count).reshape(-1, _last_hidden_states(output).shape[-1])
    codec_ids = target_codec_codes[:frame_count, :].to(device=hidden_states.device, dtype=torch.long)
    residual_logits, _ = talker.forward_sub_talker_finetune(codec_ids=codec_ids, talker_hidden_states=hidden_states)
    return residual_logits


def build_delta_for_request(
    request: SonoEditRequest,
    *,
    output_delta_path: str | Path,
    layer_name: str,
    speaker: str | None,
    language: str = "English",
    instruct: str | None = None,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    output_delta_scale: float = 1.0,
    semantic_codebook_weight: float = 1.0,
    residual_codebook_weight: float = 0.15,
    max_relative_delta: float | None = 1e-3,
) -> DeltaBuildResult:
    validate_talker_layer_name(layer_name)
    wrapper = load_qwen_wrapper(
        request.model_checkpoint_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    model = wrapper.model
    state_keys = {name for name, _ in model.named_parameters()}
    weight_name = preferred_projection_weight(layer_name, state_keys)
    module_name = _projection_module_name(weight_name)
    current_weight = dict(model.named_parameters())[weight_name].detach().cpu()
    frame_span = request.desired_pronunciation.codec0_frame_span or request.target_frame_span
    target_codec_codes = extract_codec_target(
        model.speech_tokenizer,
        request.desired_pronunciation.audio_path,
        frame_span,
    )

    model.train(False)
    model.requires_grad_(False)
    get_module_by_name(model, module_name).requires_grad_(True)
    with torch.enable_grad():
        with ProjectionCapture(model, module_name) as capture:
            output = _run_qwen_prefill(
                wrapper,
                request.source_sentence,
                speaker=speaker,
                language=language,
                instruct=instruct,
            )
        logits = output.logits
        if capture.input is None or capture.output is None:
            raise RuntimeError(f"projection was not executed: {module_name}")
        target_input = capture.input
        target_output = capture.output
        target_codec_tensor = _codec_target_matrix(target_codec_codes, logits.device)
        frame_count = min(target_codec_tensor.shape[0], logits.shape[-2], target_output.shape[-2])
        residual_logits = residual_codebook_logits(model.talker, output, target_codec_tensor, frame_count)

    preservation_inputs: list[torch.Tensor] = []
    for example in request.preservation_manifest:
        with torch.no_grad():
            with ProjectionCapture(model, module_name) as capture:
                _run_qwen_prefill(
                    wrapper,
                    example.sentence,
                    speaker=speaker,
                    language=language,
                    instruct=instruct,
                )
                if capture.input is None:
                    raise RuntimeError(f"projection was not executed for preservation example: {module_name}")
                preservation_inputs.append(capture.input.detach().cpu())

    delta, result = build_delta_from_captured_tensors(
        weight_name=weight_name,
        current_weight=current_weight,
        target_codec0=target_codec_codes,
        codec0_logits=logits,
        projection_input=target_input,
        projection_output=target_output,
        preservation_inputs=preservation_inputs,
        residual_logits=residual_logits,
        semantic_codebook_weight=semantic_codebook_weight,
        residual_codebook_weight=residual_codebook_weight,
        output_delta_scale=output_delta_scale,
        max_relative_delta=max_relative_delta,
    )
    output_delta_path = Path(output_delta_path)
    output_delta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({weight_name: delta}, output_delta_path)
    with output_delta_path.with_suffix(output_delta_path.suffix + ".json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return result


def build_delta_from_tensor_bundle(
    bundle_path: str | Path,
    output_delta_path: str | Path,
    *,
    weight_name: str,
    output_delta_scale: float = 1.0,
    semantic_codebook_weight: float = 1.0,
    residual_codebook_weight: float = 0.15,
    max_relative_delta: float | None = 1e-3,
) -> DeltaBuildResult:
    bundle = torch.load(bundle_path, map_location="cpu")
    required = {"current_weight", "projection_input", "preservation_inputs"}
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"tensor bundle is missing required fields: {missing}")
    if "desired_delta" in bundle:
        frame_count = int(bundle.get("target_frames", 1))
        delta, result = build_delta_from_output_delta(
            weight_name=weight_name,
            current_weight=bundle["current_weight"],
            projection_input=bundle["projection_input"],
            desired_delta=bundle["desired_delta"],
            frame_count=frame_count,
            preservation_inputs=list(bundle["preservation_inputs"]),
            semantic_codebook_weight=semantic_codebook_weight,
            residual_codebook_weight=residual_codebook_weight,
            output_delta_scale=output_delta_scale,
            max_relative_delta=max_relative_delta,
        )
    else:
        required_live = {"target_codec0", "codec0_logits", "projection_output"}
        missing_live = sorted(required_live - set(bundle))
        if missing_live:
            raise ValueError(
                "tensor bundle must include desired_delta, or include live-gradient fields "
                f"{sorted(required_live)}; missing {missing_live}"
            )
        delta, result = build_delta_from_captured_tensors(
            weight_name=weight_name,
            current_weight=bundle["current_weight"],
            target_codec0=bundle["target_codec0"],
            codec0_logits=bundle["codec0_logits"],
            projection_input=bundle["projection_input"],
            projection_output=bundle["projection_output"],
            preservation_inputs=list(bundle["preservation_inputs"]),
            residual_logits=bundle.get("residual_logits"),
            semantic_codebook_weight=semantic_codebook_weight,
            residual_codebook_weight=residual_codebook_weight,
            output_delta_scale=output_delta_scale,
            max_relative_delta=max_relative_delta,
        )
    output_delta_path = Path(output_delta_path)
    output_delta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({weight_name: delta}, output_delta_path)
    with output_delta_path.with_suffix(output_delta_path.suffix + ".json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json")
    parser.add_argument("--output-delta", required=True)
    parser.add_argument("--layer", default="talker.model.layers.8")
    parser.add_argument("--speaker")
    parser.add_argument("--language", default="English")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--output-delta-scale", type=float, default=1.0)
    parser.add_argument("--semantic-codebook-weight", type=float, default=1.0)
    parser.add_argument("--residual-codebook-weight", type=float, default=0.15)
    parser.add_argument("--max-relative-delta", type=float, default=1e-3)
    parser.add_argument("--tensor-bundle")
    parser.add_argument("--weight-name")
    args = parser.parse_args(argv)

    if args.tensor_bundle:
        if not args.weight_name:
            raise SystemExit("--weight-name is required with --tensor-bundle")
        result = build_delta_from_tensor_bundle(
            args.tensor_bundle,
            args.output_delta,
            weight_name=args.weight_name,
            output_delta_scale=args.output_delta_scale,
            semantic_codebook_weight=args.semantic_codebook_weight,
            residual_codebook_weight=args.residual_codebook_weight,
            max_relative_delta=args.max_relative_delta,
        )
    else:
        if not args.request_json:
            raise SystemExit("--request-json is required unless --tensor-bundle is used")
        request = SonoEditRequest.from_json_file(args.request_json)
        result = build_delta_for_request(
            request,
            output_delta_path=args.output_delta,
            layer_name=args.layer,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            output_delta_scale=args.output_delta_scale,
            semantic_codebook_weight=args.semantic_codebook_weight,
            residual_codebook_weight=args.residual_codebook_weight,
            max_relative_delta=args.max_relative_delta,
        )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
