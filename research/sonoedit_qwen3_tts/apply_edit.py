"""Apply experimental SonoEdit-style deltas to copied Qwen3-TTS checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .activations import validate_talker_layer_name
from .nullspace_edit import preferred_projection_weight, validate_edit_weight_name
from .schema import METADATA_FILENAME, SonoEditRequest, SonoEditResult


WEIGHT_FILENAMES = {"model.safetensors", "pytorch_model.bin"}


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


def _save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file(state_dict, str(path))


def ensure_distinct_checkpoints(input_path: Path, output_path: Path) -> None:
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        raise ValueError("--output-model-path must be different from --input-model-path")
    if resolved_output.is_relative_to(resolved_input) or resolved_input.is_relative_to(resolved_output):
        raise ValueError("--output-model-path must not overlap --input-model-path")


def resolve_planned_target_weights(request: SonoEditRequest, state_dict_keys: set[str]) -> list[str]:
    weights = []
    for layer in request.selected_edit_layers:
        validate_talker_layer_name(layer)
        weights.append(preferred_projection_weight(layer, state_dict_keys))
    return weights


def validate_requested_deltas(planned_weights: list[str], deltas: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    if not deltas:
        raise ValueError("non-dry-run SonoEdit requires a delta for each selected edit layer")
    planned = set(planned_weights)
    supplied = set(deltas)
    unexpected = sorted(supplied - planned)
    missing = sorted(planned - supplied)
    if unexpected:
        raise ValueError(f"delta targets are not selected edit weights: {unexpected}")
    if missing:
        raise ValueError(f"missing deltas for selected edit weights: {missing}")
    return deltas


def copy_checkpoint_assets(input_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    for source in input_path.iterdir():
        target = output_path / source.name
        if source.name in WEIGHT_FILENAMES:
            continue
        if source.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def load_delta_file(path: str | Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError("delta file must contain a dict of weight_name -> tensor")
    return {str(key): torch.as_tensor(value) for key, value in loaded.items()}


def apply_checkpoint_edit(
    request: SonoEditRequest,
    *,
    dry_run: bool = False,
    deltas: dict[str, torch.Tensor] | None = None,
) -> SonoEditResult:
    input_path = Path(request.model_checkpoint_path)
    output_path = Path(request.output_checkpoint_path)
    ensure_distinct_checkpoints(input_path, output_path)
    source_weights = input_path / "model.safetensors"
    if not source_weights.exists():
        raise FileNotFoundError(f"missing source weights: {source_weights}")

    state_dict = _load_safetensors(source_weights)
    planned_weights = resolve_planned_target_weights(request, set(state_dict))

    if dry_run:
        return SonoEditResult(
            request=request,
            edited_weights=planned_weights,
            selected_layers=request.selected_edit_layers,
            metadata_path=None,
            dry_run=True,
        )

    deltas = validate_requested_deltas(planned_weights, deltas)
    copy_checkpoint_assets(input_path, output_path)
    patched = dict(state_dict)
    for weight_name, delta in deltas.items():
        validate_edit_weight_name(weight_name)
        if weight_name not in patched:
            raise KeyError(f"delta target not found in checkpoint: {weight_name}")
        if tuple(delta.shape) != tuple(patched[weight_name].shape):
            raise ValueError(f"delta shape {tuple(delta.shape)} does not match {weight_name} {tuple(patched[weight_name].shape)}")
        patched[weight_name] = patched[weight_name] + delta.to(dtype=patched[weight_name].dtype)

    edited_weights = sorted(deltas)
    _save_safetensors(patched, output_path / "model.safetensors")
    result = SonoEditResult(
        request=request,
        edited_weights=edited_weights,
        selected_layers=request.selected_edit_layers,
        metadata_path=str(output_path / METADATA_FILENAME),
        dry_run=False,
    )
    with (output_path / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    return result


def result_summary(result: SonoEditResult) -> dict[str, Any]:
    return {
        "dry_run": result.dry_run,
        "selected_layers": result.selected_layers,
        "planned_or_edited_weights": result.edited_weights,
        "metadata_path": result.metadata_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--output-model-path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delta-file")
    args = parser.parse_args(argv)

    request = SonoEditRequest.from_json_file(args.request_json)
    if args.output_model_path:
        request = SonoEditRequest.from_dict({**request.to_dict(), "output_checkpoint_path": args.output_model_path})
    deltas = load_delta_file(args.delta_file) if args.delta_file else None
    result = apply_checkpoint_edit(request, dry_run=args.dry_run, deltas=deltas)
    print(json.dumps(result_summary(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
