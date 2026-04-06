import io
import sys
import types
import wave

import dataset_jobs


def _make_wav_bytes(duration_seconds: float = 0.5, sample_rate: int = 24000) -> bytes:
    frames = int(duration_seconds * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x01\x00" * frames)
    return buf.getvalue()


def test_validate_and_crop_audio_falls_back_without_diarization_token(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        serialization=types.SimpleNamespace(add_safe_globals=lambda _globals: None),
    )
    fake_whisperx = types.SimpleNamespace(
        load_audio=lambda _path: "audio",
        align=lambda segments, *_args, **_kwargs: {"segments": segments},
    )
    fake_pydub = types.SimpleNamespace(AudioSegment=object)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        dataset_jobs,
        "_get_whisper_model",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            transcribe=lambda _audio, batch_size=8: {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 0.4, "text": "hello"},
                    {"start": 0.4, "end": 0.8, "text": "there"},
                ],
            }
        ),
    )
    monkeypatch.setattr(dataset_jobs, "_get_align_model", lambda *_args, **_kwargs: (object(), {}))
    monkeypatch.setattr(
        dataset_jobs,
        "_get_diarization_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("HF_TOKEN is required for dataset diarization.")),
    )

    result = dataset_jobs._validate_and_crop_audio_sync(
        [("clip.wav", _make_wav_bytes(), "prompt-1")],
        "Elena",
    )

    assert len(result) == 1
    filename, _wav_bytes, prompt_id, success, transcript, reason = result[0]
    assert filename == "clip.wav"
    assert prompt_id == "prompt-1"
    assert success is True
    assert transcript == "hello there"
    assert "HF_TOKEN" in (reason or "")


def test_validate_and_crop_audio_retries_on_cpu_after_cuda_oom(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
        serialization=types.SimpleNamespace(add_safe_globals=lambda _globals: None),
    )
    fake_whisperx = types.SimpleNamespace(
        load_audio=lambda _path: "audio",
        align=lambda segments, *_args, **_kwargs: {"segments": segments},
    )
    fake_pydub = types.SimpleNamespace(AudioSegment=object)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("DATASET_PREP_DEVICE", raising=False)
    calls: list[tuple[str, str]] = []

    def fake_get_whisper_model(model_name, *, device=None, compute_type=None):
        calls.append((device or "", compute_type or ""))
        if device == "cuda":
            raise RuntimeError("CUDA failed with error out of memory")
        return types.SimpleNamespace(
            transcribe=lambda _audio, batch_size=8: {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 0.4, "text": "major"},
                    {"start": 0.4, "end": 0.8, "text": "barry"},
                ],
            }
        )

    monkeypatch.setattr(dataset_jobs, "_get_whisper_model", fake_get_whisper_model)
    monkeypatch.setattr(dataset_jobs, "_get_align_model", lambda *_args, **_kwargs: (object(), {}))
    monkeypatch.setattr(
        dataset_jobs,
        "_get_diarization_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("HF_TOKEN is required for dataset diarization.")),
    )

    result = dataset_jobs._validate_and_crop_audio_sync(
        [("clip.wav", _make_wav_bytes(), "prompt-1")],
        "Major Barry",
    )

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert result[0][3] is True
    assert result[0][4] == "major barry"
