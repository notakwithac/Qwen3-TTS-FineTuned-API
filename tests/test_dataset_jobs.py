import io
import sys
import types
import wave
import asyncio

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
    assert transcript == "Hello there."
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
    assert result[0][4] == "Major barry."


def test_validate_and_crop_audio_auto_uses_cpu_when_cuda_headroom_is_too_low(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            mem_get_info=lambda *_args, **_kwargs: (4 * 1024**3, 24 * 1024**3),
        ),
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
    monkeypatch.setenv("DATASET_WHISPER_MODEL", "large-v3")
    calls: list[tuple[str, str]] = []

    def fake_get_whisper_model(model_name, *, device=None, compute_type=None):
        calls.append((device or "", compute_type or ""))
        return types.SimpleNamespace(
            transcribe=lambda _audio, batch_size=8: {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 0.4, "text": "narrator"},
                    {"start": 0.4, "end": 0.8, "text": "line"},
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
        "Narrator",
    )

    assert calls == [("cpu", "int8")]
    assert result[0][3] is True
    assert result[0][4] == "Narrator line."


def test_validate_and_crop_audio_splits_long_dominant_speaker_segments(monkeypatch):
    class _FakeSlice:
        def __init__(self, duration_ms: int):
            self.duration_ms = duration_ms

        def __len__(self):
            return self.duration_ms

        def export(self, buf, format="wav"):
            buf.write(_make_wav_bytes(max(self.duration_ms / 1000.0, 0.1)))

    class _FakeAudioSegment:
        def __init__(self, duration_ms: int):
            self.duration_ms = duration_ms

        @classmethod
        def from_wav(cls, _stream):
            return cls(26_000)

        def __len__(self):
            return self.duration_ms

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            stop = self.duration_ms if key.stop is None else key.stop
            return _FakeSlice(max(0, stop - start))

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        serialization=types.SimpleNamespace(add_safe_globals=lambda _globals: None),
    )

    fake_whisperx = types.SimpleNamespace(
        load_audio=lambda _path: "audio",
        align=lambda segments, *_args, **_kwargs: {"segments": segments},
        assign_word_speakers=lambda _diarized, result: result,
    )
    fake_pydub = types.SimpleNamespace(AudioSegment=_FakeAudioSegment)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(
        dataset_jobs,
        "_get_whisper_model",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            transcribe=lambda _audio, batch_size=8: {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 6.0, "text": "one", "speaker": "SPEAKER_00"},
                    {"start": 6.2, "end": 12.0, "text": "two", "speaker": "SPEAKER_00"},
                    {"start": 12.2, "end": 18.0, "text": "three", "speaker": "SPEAKER_00"},
                    {"start": 18.2, "end": 26.0, "text": "four", "speaker": "SPEAKER_00"},
                ],
            }
        ),
    )
    monkeypatch.setattr(dataset_jobs, "_get_align_model", lambda *_args, **_kwargs: (object(), {}))
    monkeypatch.setattr(dataset_jobs, "_get_diarization_pipeline", lambda *_args, **_kwargs: lambda *_a, **_k: [])

    result = dataset_jobs._validate_and_crop_audio_sync(
        [("clip.wav", _make_wav_bytes(26.0), "prompt-1")],
        "Elena",
    )

    assert len(result) == 4
    assert [item[0] for item in result] == ["clip__seg_000.wav", "clip__seg_001.wav", "clip__seg_002.wav", "clip__seg_003.wav"]
    assert [item[4] for item in result] == ["One.", "Two.", "Three.", "Four."]


def test_prepare_dataset_items_preserves_reference_item(monkeypatch):
    uploaded: list[tuple[str, bytes]] = []

    async def fake_download(_storage, ref):
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_apply(audio_bytes, **_kwargs):
        return audio_bytes

    async def fake_validate(_segments, _character_name):
        return [
            ("clip-1.wav", b"clip-1-bytes", "prompt-1", True, "hello there", None),
            ("clip-2.wav", b"clip-2-bytes", "prompt-2", True, "general kenobi", None),
        ]

    async def fake_upload(_storage, payload, key, **_kwargs):
        uploaded.append((key, payload))
        return f"https://bucket/{key}"

    storage = types.SimpleNamespace(
        get_presigned_url=lambda key, expires_in=0: f"https://signed/{key}",
    )

    monkeypatch.setattr(dataset_jobs, "_download_bytes", fake_download)
    monkeypatch.setattr(dataset_jobs, "apply_audio_fx", fake_apply)
    monkeypatch.setattr(dataset_jobs, "validate_and_crop_audio", fake_validate)
    monkeypatch.setattr(dataset_jobs, "_upload_bytes", fake_upload)

    items = asyncio.run(
        dataset_jobs.prepare_dataset_items(
            storage,
            book_id="book-1",
            character_id="char-1",
            character_name="Elena",
            job_id="job-1",
            ref_audio_url="s3://bucket/ref.wav",
            ref_text="reference text",
            items=[
                {"filename": "clone-1.wav", "prompt_id": "prompt-1", "text": "hello there", "s3_url": "s3://bucket/clone-1.wav"},
                {"filename": "clone-2.wav", "prompt_id": "prompt-2", "text": "general kenobi", "s3_url": "s3://bucket/clone-2.wav"},
            ],
        )
    )

    assert items[0]["id"] == "ref_audio.wav"
    assert items[0]["is_reference"] is True
    assert items[0]["text"] == "Reference text."
    assert [item["id"] for item in items[1:]] == ["clip-1.wav", "clip-2.wav"]
    assert sum(1 for item in items if item["is_reference"]) == 1
    assert uploaded[0][0].endswith("_ref_audio.wav")
