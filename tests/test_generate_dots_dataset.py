import json

import pytest

from scripts.generate_dots_dataset import DatasetPrompt, _write_train_jsonl, load_prompts, resolve_model_spec


def test_load_prompts_supplies_names_and_preserves_dialogue(tmp_path):
    manifest = tmp_path / "dialogue.jsonl"
    manifest.write_text(
        '{"text":"First line."}\n'
        '{"filename":"reply.wav","text":"Second line."}\n',
        encoding="utf-8",
    )

    assert load_prompts(manifest) == [
        DatasetPrompt(filename="sample_0001.wav", text="First line."),
        DatasetPrompt(filename="reply.wav", text="Second line."),
    ]


def test_load_prompts_rejects_path_traversal(tmp_path):
    manifest = tmp_path / "dialogue.jsonl"
    manifest.write_text('{"filename":"../escape.wav","text":"No."}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="plain file name"):
        load_prompts(manifest)


def test_qwen_manifest_uses_one_shared_reference(tmp_path):
    output = _write_train_jsonl(
        tmp_path,
        [DatasetPrompt(filename="line.wav", text="A line of dialogue.")],
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row == {
        "audio": "./data/line.wav",
        "text": "A line of dialogue.",
        "ref_audio": "./data/ref_audio.wav",
    }


def test_resolve_model_spec_accepts_hugging_face_repo_ids():
    assert resolve_model_spec("rednote-hilab/dots.tts-soar") == "rednote-hilab/dots.tts-soar"


def test_resolve_model_spec_resolves_local_directories(tmp_path):
    model_dir = tmp_path / "dots-model"
    model_dir.mkdir()

    assert resolve_model_spec(str(model_dir)) == str(model_dir.resolve())


def test_resolve_model_spec_rejects_missing_local_paths():
    with pytest.raises(FileNotFoundError):
        resolve_model_spec("./missing-model")
