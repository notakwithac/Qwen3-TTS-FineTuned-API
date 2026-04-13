import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINETUNING_DIR = ROOT / "finetuning"

if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

import sft_12hz  # noqa: E402


def test_copy_checkpoint_assets_restores_tokenizer_tree_from_fallback(tmp_path):
    incomplete_checkpoint = tmp_path / "checkpoint-epoch-0"
    incomplete_checkpoint.mkdir()
    (incomplete_checkpoint / "config.json").write_text(
        json.dumps({"tts_model_type": "custom_voice", "talker_config": {}}),
        encoding="utf-8",
    )

    fallback_model = tmp_path / "base-model"
    fallback_model.mkdir()
    (fallback_model / "config.json").write_text(
        json.dumps(
            {
                "tts_model_type": "base",
                "talker_config": {
                    "spk_id": {"base": 1},
                    "spk_is_dialect": {"base": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (fallback_model / "generation_config.json").write_text("{}", encoding="utf-8")
    speech_tokenizer_dir = fallback_model / "speech_tokenizer"
    speech_tokenizer_dir.mkdir()
    (speech_tokenizer_dir / "config.json").write_text("{}", encoding="utf-8")
    nested_dir = speech_tokenizer_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "vocab.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "saved"
    sft_12hz._copy_checkpoint_assets(
        source_model_path=str(incomplete_checkpoint),
        output_dir=str(output_dir),
        speaker_name="Narrator",
        fallback_model_path=str(fallback_model),
    )

    assert (output_dir / "speech_tokenizer" / "config.json").exists()
    assert (output_dir / "speech_tokenizer" / "nested" / "vocab.json").exists()
    assert (output_dir / "generation_config.json").exists()

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert config["tts_model_type"] == "custom_voice"
    assert config["talker_config"]["spk_id"] == {"Narrator": 3000}
    assert config["talker_config"]["spk_is_dialect"] == {"Narrator": False}


def test_should_save_epoch_only_from_epoch_six_onward():
    assert sft_12hz._should_save_epoch(5, save_from_epoch=6) is False
    assert sft_12hz._should_save_epoch(6, save_from_epoch=6) is True
    assert sft_12hz._should_save_epoch(12, save_from_epoch=6) is True


def test_needs_final_checkpoint_for_short_runs_before_epoch_six():
    assert sft_12hz._needs_final_checkpoint(4, 0, True, save_from_epoch=6) is True
    assert sft_12hz._needs_final_checkpoint(6, 0, True, save_from_epoch=6) is False
    assert sft_12hz._needs_final_checkpoint(4, 0, False, save_from_epoch=6) is False
