import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from research.sonoedit_qwen3_tts.apply_edit import apply_checkpoint_edit
from research.sonoedit_qwen3_tts.schema import METADATA_FILENAME, SonoEditRequest


def _request(tmp_path: Path) -> SonoEditRequest:
    return SonoEditRequest.from_dict(
        {
            "target_term": "Qwen",
            "source_sentence": "Say Qwen.",
            "desired_pronunciation": {"audio_path": str(tmp_path / "target.wav")},
            "preservation_manifest": [{"sentence": "Keep this stable."}],
            "model_checkpoint_path": str(tmp_path / "source"),
            "output_checkpoint_path": str(tmp_path / "edited"),
            "selected_edit_layers": ["talker.model.layers.8"],
        }
    )


def _checkpoint(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text("{}", encoding="utf-8")
    save_file(
        {
            "talker.model.layers.8.mlp.down_proj.weight": torch.zeros(2, 2),
            "untouched.weight": torch.ones(1),
        },
        str(path / "model.safetensors"),
    )


def test_dry_run_reports_planned_target_weights_and_writes_no_checkpoint(tmp_path: Path):
    request = _request(tmp_path)
    _checkpoint(Path(request.model_checkpoint_path))

    result = apply_checkpoint_edit(request, dry_run=True)

    assert result.dry_run is True
    assert result.edited_weights == ["talker.model.layers.8.mlp.down_proj.weight"]
    assert not Path(request.output_checkpoint_path).exists()


def test_checkpoint_copy_patch_and_metadata(tmp_path: Path):
    request = _request(tmp_path)
    _checkpoint(Path(request.model_checkpoint_path))
    delta = {"talker.model.layers.8.mlp.down_proj.weight": torch.ones(2, 2)}

    result = apply_checkpoint_edit(request, deltas=delta)

    output = Path(request.output_checkpoint_path)
    assert (output / "config.json").exists()
    patched = load_file(str(output / "model.safetensors"))
    torch.testing.assert_close(patched["talker.model.layers.8.mlp.down_proj.weight"], torch.ones(2, 2))
    assert (output / METADATA_FILENAME).exists()
    metadata = json.loads((output / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["edited_weights"] == ["talker.model.layers.8.mlp.down_proj.weight"]
    assert result.metadata_path.endswith(METADATA_FILENAME)


def test_non_dry_run_requires_selected_weight_deltas(tmp_path: Path):
    request = _request(tmp_path)
    _checkpoint(Path(request.model_checkpoint_path))

    try:
        apply_checkpoint_edit(request)
    except ValueError as exc:
        assert "requires a delta" in str(exc)
    else:
        raise AssertionError("expected missing deltas to fail")


def test_refuses_delta_outside_selected_layers(tmp_path: Path):
    request = _request(tmp_path)
    _checkpoint(Path(request.model_checkpoint_path))
    delta = {"talker.model.layers.9.mlp.down_proj.weight": torch.ones(2, 2)}

    try:
        apply_checkpoint_edit(request, deltas=delta)
    except ValueError as exc:
        assert "not selected edit weights" in str(exc)
    else:
        raise AssertionError("expected unselected delta to fail")


def test_refuses_in_place_edit(tmp_path: Path):
    request = _request(tmp_path)
    data = request.to_dict()
    data["output_checkpoint_path"] = data["model_checkpoint_path"]
    request = SonoEditRequest.from_dict(data)
    _checkpoint(Path(request.model_checkpoint_path))

    try:
        apply_checkpoint_edit(request)
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("expected in-place edit refusal")


def test_refuses_overlapping_checkpoint_paths(tmp_path: Path):
    request = _request(tmp_path)
    data = request.to_dict()
    data["output_checkpoint_path"] = str(Path(data["model_checkpoint_path"]) / "edited")
    request = SonoEditRequest.from_dict(data)
    _checkpoint(Path(request.model_checkpoint_path))

    try:
        apply_checkpoint_edit(request, dry_run=True)
    except ValueError as exc:
        assert "must not overlap" in str(exc)
    else:
        raise AssertionError("expected overlapping path refusal")
