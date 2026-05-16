import json
from pathlib import Path

from research.sonoedit_qwen3_tts.evaluate_edit import run_eval_artifact
from research.sonoedit_qwen3_tts.schema import SonoEditRequest


def test_eval_matrix_writes_source_and_edited_target_and_preservation_rows(tmp_path: Path):
    request = SonoEditRequest.from_dict(
        {
            "target_term": "Qwen",
            "source_sentence": "Say Qwen.",
            "desired_pronunciation": {"audio_path": str(tmp_path / "target.wav")},
            "preservation_manifest": [
                {"sentence": "Keep this stable."},
                {"sentence": "Keep this one stable too."},
            ],
            "model_checkpoint_path": str(tmp_path / "source"),
            "output_checkpoint_path": str(tmp_path / "edited"),
            "selected_edit_layers": ["talker.model.layers.8"],
        }
    )

    output = tmp_path / "results.jsonl"
    records = run_eval_artifact(request, "source-model", "edited-model", output)

    assert len(records) == 6
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {line["model_role"] for line in lines} == {"source", "edited"}
    assert {line["example_role"] for line in lines} == {"target", "preservation"}
    assert {"target_correctness", "asr_text", "per", "wer", "speaker_similarity", "reviewer_notes"} <= set(lines[0])

