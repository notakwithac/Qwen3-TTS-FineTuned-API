from pathlib import Path

from research.sonoedit_qwen3_tts.schema import METADATA_FILENAME, SonoEditRequest


def test_parse_sonoedit_request_with_required_fields(tmp_path: Path):
    request = SonoEditRequest.from_dict(
        {
            "target_term": "Qwen",
            "source_sentence": "Please say Qwen clearly.",
            "desired_pronunciation": {
                "audio_path": str(tmp_path / "qwen.wav"),
                "transcript": "Qwen",
                "codec0_frame_span": [2, 5],
            },
            "preservation_manifest": [
                {"sentence": "A neutral preservation sentence.", "speaker_id": "voice-a"}
            ],
            "model_checkpoint_path": str(tmp_path / "source"),
            "output_checkpoint_path": str(tmp_path / "edited"),
            "selected_edit_layers": ["talker.model.layers.8"],
            "target_frame_span": [3, 7],
        }
    )

    assert request.target_term == "Qwen"
    assert request.source_sentence.startswith("Please")
    assert request.desired_pronunciation.audio_path.endswith("qwen.wav")
    assert request.desired_pronunciation.codec0_frame_span == (2, 5)
    assert request.preservation_manifest[0].speaker_id == "voice-a"
    assert request.model_checkpoint_path.endswith("source")
    assert request.output_checkpoint_path.endswith("edited")
    assert request.selected_edit_layers == ["talker.model.layers.8"]
    assert request.target_frame_span == (3, 7)
    assert request.metadata_path.name == METADATA_FILENAME


def test_invalid_span_fails_clearly(tmp_path: Path):
    data = {
        "target_term": "Qwen",
        "source_sentence": "Please say Qwen clearly.",
        "desired_pronunciation": {"audio_path": str(tmp_path / "qwen.wav")},
        "preservation_manifest": [],
        "model_checkpoint_path": str(tmp_path / "source"),
        "output_checkpoint_path": str(tmp_path / "edited"),
        "selected_edit_layers": [],
        "target_frame_span": [5, 5],
    }

    try:
        SonoEditRequest.from_dict(data)
    except ValueError as exc:
        assert "target_frame_span" in str(exc)
    else:
        raise AssertionError("expected invalid span to fail")

