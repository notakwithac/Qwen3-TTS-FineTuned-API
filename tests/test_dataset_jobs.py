import io
import sys
import types
import wave
import asyncio
import json
import zipfile

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
            return cls(24_000)

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
    )
    fake_pydub = types.SimpleNamespace(AudioSegment=_FakeAudioSegment)

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
                    {"start": 0.0, "end": 4.0, "text": "hello"},
                    {"start": 4.1, "end": 8.0, "text": "there"},
                    {"start": 8.1, "end": 12.0, "text": "general"},
                    {"start": 12.1, "end": 16.0, "text": "kenobi"},
                    {"start": 16.1, "end": 20.0, "text": "you"},
                    {"start": 20.1, "end": 24.0, "text": "are"},
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
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    result = dataset_jobs._validate_and_crop_audio_sync(
        [("clip.wav", _make_wav_bytes(24.0), "prompt-1")],
        "Elena",
    )

    assert len(result) == 3
    assert [item[0] for item in result] == ["clip__seg_000.wav", "clip__seg_001.wav", "clip__seg_002.wav"]
    assert [item[4] for item in result] == ["Hello there.", "General kenobi.", "You are."]


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
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    result = dataset_jobs._validate_and_crop_audio_sync(
        [("clip.wav", _make_wav_bytes(26.0), "prompt-1")],
        "Elena",
    )

    assert len(result) == 4
    assert [item[0] for item in result] == ["clip__seg_000.wav", "clip__seg_001.wav", "clip__seg_002.wav", "clip__seg_003.wav"]
    assert [item[4] for item in result] == ["One.", "Two.", "Three.", "Four."]


def test_build_split_results_keeps_time_aligned_transcripts(monkeypatch):
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
            return cls(24_000)

        def __len__(self):
            return self.duration_ms

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            stop = self.duration_ms if key.stop is None else key.stop
            return _FakeSlice(max(0, stop - start))

    fake_pydub = types.SimpleNamespace(AudioSegment=_FakeAudioSegment)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    segments = [
        {"start": 0.0, "end": 4.0, "text": "alpha", "speaker": "SPEAKER_00"},
        {"start": 4.1, "end": 8.0, "text": "beta", "speaker": "SPEAKER_00"},
        {"start": 8.1, "end": 12.0, "text": "gamma", "speaker": "SPEAKER_00"},
        {"start": 12.1, "end": 16.0, "text": "delta", "speaker": "SPEAKER_00"},
        {"start": 16.1, "end": 20.0, "text": "epsilon", "speaker": "SPEAKER_00"},
        {"start": 20.1, "end": 24.0, "text": "zeta", "speaker": "SPEAKER_00"},
    ]

    result = dataset_jobs._build_split_results(
        "clip.wav",
        _make_wav_bytes(24.0),
        "prompt-1",
        segments,
        main_speaker="SPEAKER_00",
    )

    assert [item[0] for item in result] == [
        "clip__seg_000.wav",
        "clip__seg_001.wav",
        "clip__seg_002.wav",
    ]
    assert [item[4] for item in result] == [
        "Alpha beta.",
        "Gamma delta.",
        "Epsilon zeta.",
    ]


def test_build_split_results_prefers_sentence_end_boundaries(monkeypatch):
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
            return cls(9_000)

        def __len__(self):
            return self.duration_ms

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            stop = self.duration_ms if key.stop is None else key.stop
            return _FakeSlice(max(0, stop - start))

    fake_pydub = types.SimpleNamespace(AudioSegment=_FakeAudioSegment)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    segments = [
        {"start": 0.0, "end": 3.0, "text": "First sentence.", "speaker": "SPEAKER_00"},
        {"start": 3.1, "end": 6.0, "text": "second sentence", "speaker": "SPEAKER_00"},
        {"start": 6.1, "end": 9.0, "text": "third sentence", "speaker": "SPEAKER_00"},
    ]

    result = dataset_jobs._build_split_results(
        "clip.wav",
        _make_wav_bytes(9.0),
        "prompt-1",
        segments,
        main_speaker="SPEAKER_00",
    )

    assert [item[0] for item in result] == ["clip__seg_000.wav"]
    assert [item[4] for item in result] == ["First sentence. Second sentence."]


def test_build_split_results_force_splits_single_long_segment(monkeypatch):
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
            return cls(30_000)

        def __len__(self):
            return self.duration_ms

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            stop = self.duration_ms if key.stop is None else key.stop
            return _FakeSlice(max(0, stop - start))

    fake_pydub = types.SimpleNamespace(AudioSegment=_FakeAudioSegment)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    segments = [
        {
            "start": 0.0,
            "end": 30.0,
            "text": "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence. Sixth sentence.",
            "speaker": "SPEAKER_00",
        }
    ]

    result = dataset_jobs._build_split_results(
        "clip.wav",
        _make_wav_bytes(30.0),
        "prompt-1",
        segments,
        main_speaker="SPEAKER_00",
    )

    assert result == []


def test_validate_and_crop_audio_batches_reference_with_clones_and_emits_reference_once(monkeypatch):
    class _FakeSlice:
        def __init__(self, duration_ms: int):
            self.duration_ms = duration_ms

        def __len__(self):
            return self.duration_ms

        def export(self, buf, format="wav"):
            buf.write(_make_wav_bytes(max(self.duration_ms / 1000.0, 0.1)))

    class _DurationAwareAudioSegment:
        def __init__(self, duration_ms: int):
            self.duration_ms = duration_ms

        @classmethod
        def from_wav(cls, stream):
            if hasattr(stream, "read"):
                raw = stream.read()
                stream = io.BytesIO(raw)
            with wave.open(stream, "rb") as wf:
                duration_ms = int((wf.getnframes() / wf.getframerate()) * 1000)
            return cls(duration_ms)

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

    def fake_load_audio(path):
        with open(path, "rb") as handle:
            return {"duration": dataset_jobs._wav_duration_seconds(handle.read())}

    fake_whisperx = types.SimpleNamespace(
        load_audio=fake_load_audio,
        align=lambda segments, *_args, **_kwargs: {"segments": segments},
        assign_word_speakers=lambda _diarized, result: result,
    )
    fake_pydub = types.SimpleNamespace(AudioSegment=_DurationAwareAudioSegment)
    diarize_calls: list[float] = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(
        dataset_jobs,
        "_get_whisper_model",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            transcribe=lambda _audio, batch_size=8, **_kw: {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 6.0, "text": "reference voice", "speaker": "SPEAKER_00"},
                    {"start": 6.35, "end": 12.35, "text": "keep clone line", "speaker": "SPEAKER_00"},
                    {"start": 12.35, "end": 14.35, "text": "intruder line", "speaker": "SPEAKER_01"},
                ],
            }
        ),
    )
    monkeypatch.setattr(dataset_jobs, "_get_align_model", lambda *_args, **_kwargs: (object(), {}))
    monkeypatch.setattr(
        dataset_jobs,
        "_get_diarization_pipeline",
        lambda *_args, **_kwargs: lambda audio, **_kw: diarize_calls.append(audio["duration"]) or [],
    )
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)
    monkeypatch.setattr(dataset_jobs, "DATASET_DIARIZATION_BATCH_MAX_SEC", 13.0)

    result = dataset_jobs._validate_and_crop_audio_sync(
        [
            ("ref_audio.wav", _make_wav_bytes(6.0), "ref_audio"),
            ("clone-1.wav", _make_wav_bytes(8.0), "prompt-1"),
            ("clone-2.wav", _make_wav_bytes(8.0), "prompt-2"),
        ],
        "Elena",
    )

    reference_items = [item for item in result if item[2] == "ref_audio" and item[3] is True]
    clone_items = [item for item in result if item[2] != "ref_audio" and item[3] is True]

    assert len(diarize_calls) == 2
    assert len(reference_items) == 1
    assert [item[0] for item in clone_items] == ["clone-1.wav", "clone-2.wav"]
    assert [item[4] for item in clone_items] == ["Keep clone line.", "Keep clone line."]


def test_build_reference_diarization_batches_repeats_reference_and_respects_duration_cap():
    ref_item = {
        "filename": "ref_audio.wav",
        "prompt_id": "ref_audio",
        "wav_bytes": b"ref",
        "analysis_wav_bytes": b"ref",
        "duration_sec": 20.0,
    }
    clone_items = [
        {
            "filename": "clone-1.wav",
            "prompt_id": "prompt-1",
            "wav_bytes": b"clone-1",
            "analysis_wav_bytes": b"clone-1",
            "duration_sec": 30.0,
        },
        {
            "filename": "clone-2.wav",
            "prompt_id": "prompt-2",
            "wav_bytes": b"clone-2",
            "analysis_wav_bytes": b"clone-2",
            "duration_sec": 30.0,
        },
        {
            "filename": "clone-3.wav",
            "prompt_id": "prompt-3",
            "wav_bytes": b"clone-3",
            "analysis_wav_bytes": b"clone-3",
            "duration_sec": 25.0,
        },
    ]

    batches = dataset_jobs._build_reference_diarization_batches(
        ref_item,
        clone_items,
        max_total_duration_sec=75.0,
    )

    assert [[item["filename"] for item in batch] for batch in batches] == [
        ["ref_audio.wav", "clone-1.wav"],
        ["ref_audio.wav", "clone-2.wav", "clone-3.wav"],
    ]


def test_select_target_speaker_for_interval_uses_reference_overlap():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "lead in", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 5.0, "text": "reference line", "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 9.0, "text": "other speaker dominates later", "speaker": "SPEAKER_00"},
    ]

    speaker = dataset_jobs._select_target_speaker_for_interval(segments, 0.5, 5.5)

    assert speaker == "SPEAKER_01"


def test_slice_combined_segments_to_local_interval_preserves_local_timestamps():
    segments = [
        {"start": 0.0, "end": 4.0, "text": "reference", "speaker": "SPEAKER_00"},
        {"start": 4.6, "end": 7.0, "text": "keep this", "speaker": "SPEAKER_00"},
        {"start": 7.0, "end": 8.2, "text": "drop this", "speaker": "SPEAKER_01"},
        {"start": 8.2, "end": 10.1, "text": "keep too", "speaker": "SPEAKER_00"},
    ]

    local_segments = dataset_jobs._slice_combined_segments_to_local_interval(
        segments,
        4.5,
        10.5,
        speaker="SPEAKER_00",
    )

    assert local_segments == [
        {"start": 0.1, "end": 2.5, "text": "keep this", "speaker": "SPEAKER_00"},
        {"start": 3.7, "end": 5.6, "text": "keep too", "speaker": "SPEAKER_00"},
    ]


def test_prepare_dataset_items_uploads_segmented_reference_items_only(monkeypatch):
    uploaded: list[tuple[str, bytes]] = []

    async def fake_download(_storage, ref):
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_apply(audio_bytes, **_kwargs):
        return audio_bytes

    async def fake_validate(_segments, _character_name):
        return [
            ("ref_audio__seg_000.wav", b"ref-seg-1", "ref_audio", True, "reference segment one", None, False, True),
            ("ref_audio__seg_001.wav", b"ref-seg-2", "ref_audio", True, "reference segment two", None, False, True),
            ("clip-1.wav", b"clip-1-bytes", "prompt-1", True, "hello there", None, False),
            ("clip-2.wav", b"clip-2-bytes", "prompt-2", True, "general kenobi", None, False),
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

    assert [item["id"] for item in items] == [
        "ref_audio__seg_000.wav",
        "ref_audio__seg_001.wav",
        "clip-1.wav",
        "clip-2.wav",
    ]
    assert [item["is_reference"] for item in items] == [True, True, False, False]
    assert [item["text"] for item in items[:2]] == ["Reference segment one.", "Reference segment two."]
    assert all(not key.endswith("_ref_audio.wav") for key, _payload in uploaded)
    assert uploaded[0][0].endswith("_ref_audio__seg_000.wav")


def test_prepare_dataset_items_uses_fragment_transcript_for_split_clips(monkeypatch):
    async def fake_download(_storage, ref):
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_apply(audio_bytes, **_kwargs):
        return audio_bytes

    async def fake_validate(_segments, _character_name):
        return [
            (
                "ref_audio__seg_000.wav",
                b"ref-seg",
                "ref_audio",
                True,
                "reference segment",
                None,
                False,
                True,
            ),
            (
                "clip__seg_000.wav",
                b"clip-1-bytes",
                "prompt-1",
                True,
                "first sliced line",
                None,
                False,
            ),
            (
                "clip__seg_001.wav",
                b"clip-2-bytes",
                "prompt-1",
                True,
                "second sliced line",
                None,
                False,
            ),
        ]

    async def fake_upload(_storage, payload, key, **_kwargs):
        return f"https://bucket/{key}"

    storage = types.SimpleNamespace(
        get_presigned_url=lambda key, expires_in=0: f"https://signed/{key}",
    )

    monkeypatch.setattr(dataset_jobs, "_download_bytes", fake_download)
    monkeypatch.setattr(dataset_jobs, "apply_audio_fx", fake_apply)
    monkeypatch.setattr(dataset_jobs, "validate_and_crop_audio", fake_validate)
    monkeypatch.setattr(dataset_jobs, "_upload_bytes", fake_upload)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

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
                {
                    "filename": "clone-1.wav",
                    "prompt_id": "prompt-1",
                    "text": "full original prompt that should not be reused",
                    "s3_url": "s3://bucket/clone-1.wav",
                },
            ],
        )
    )

    train_items = [item for item in items if not item["is_reference"]]
    assert [item["id"] for item in train_items] == ["clip__seg_000.wav", "clip__seg_001.wav"]
    assert [item["text"] for item in train_items] == ["First sliced line.", "Second sliced line."]


def test_prepare_dataset_items_uses_aligned_transcript_for_single_derived_clip(monkeypatch):
    async def fake_download(_storage, ref):
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_apply(audio_bytes, **_kwargs):
        return audio_bytes

    async def fake_validate(_segments, _character_name):
        return [
            (
                "ref_audio__seg_000.wav",
                b"ref-seg",
                "ref_audio",
                True,
                "reference segment",
                None,
                False,
                True,
            ),
            (
                "clip-1.wav",
                b"clip-1-bytes",
                "prompt-1",
                True,
                "cropped aligned line",
                None,
                False,
                True,
            ),
        ]

    async def fake_upload(_storage, payload, key, **_kwargs):
        return f"https://bucket/{key}"

    storage = types.SimpleNamespace(
        get_presigned_url=lambda key, expires_in=0: f"https://signed/{key}",
    )

    monkeypatch.setattr(dataset_jobs, "_download_bytes", fake_download)
    monkeypatch.setattr(dataset_jobs, "apply_audio_fx", fake_apply)
    monkeypatch.setattr(dataset_jobs, "validate_and_crop_audio", fake_validate)
    monkeypatch.setattr(dataset_jobs, "_upload_bytes", fake_upload)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

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
                {
                    "filename": "clip-1.wav",
                    "prompt_id": "prompt-1",
                    "text": "full original prompt that should not be reused",
                    "s3_url": "s3://bucket/clip-1.wav",
                },
            ],
        )
    )

    train_items = [item for item in items if not item["is_reference"]]
    assert train_items[0]["id"] == "clip-1.wav"
    assert train_items[0]["text"] == "Cropped aligned line."


def test_build_split_results_splits_on_non_main_speaker_boundaries(monkeypatch):
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
            return cls(11_000)

        def __len__(self):
            return self.duration_ms

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            stop = self.duration_ms if key.stop is None else key.stop
            return _FakeSlice(max(0, stop - start))

    fake_pydub = types.SimpleNamespace(AudioSegment=_FakeAudioSegment)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    segments = [
        {"start": 0.0, "end": 5.0, "text": "hello there", "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 6.0, "text": "intruder", "speaker": "SPEAKER_01"},
        {"start": 6.0, "end": 11.0, "text": "general kenobi", "speaker": "SPEAKER_00"},
    ]

    result = dataset_jobs._build_split_results(
        "clip.wav",
        _make_wav_bytes(11.0),
        "prompt-1",
        segments,
        main_speaker="SPEAKER_00",
    )

    assert [item[0] for item in result] == ["clip__seg_000.wav", "clip__seg_001.wav"]
    assert [item[4] for item in result] == ["Hello there.", "General kenobi."]
    assert [item[7] for item in result] == [True, True]


def test_prepare_dataset_items_normalizes_only_final_outputs(monkeypatch):
    normalize_calls: list[bytes] = []

    async def fake_download(_storage, ref):
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_apply(audio_bytes, **_kwargs):
        return audio_bytes

    async def fake_validate(_segments, _character_name):
        return [
            ("ref_audio__seg_000.wav", b"ref-seg", "ref_audio", True, "reference segment", None, False, True),
            ("clip-1.wav", b"clip-1-bytes", "prompt-1", True, "hello there", None, False),
            ("clip-2.wav", b"clip-2-bytes", "prompt-2", True, "general kenobi", None, False),
        ]

    async def fake_upload(_storage, payload, key, **_kwargs):
        return f"https://bucket/{key}"

    def fake_normalize(audio_bytes):
        normalize_calls.append(audio_bytes)
        return b"norm:" + audio_bytes

    storage = types.SimpleNamespace(
        get_presigned_url=lambda key, expires_in=0: f"https://signed/{key}",
    )

    monkeypatch.setattr(dataset_jobs, "_download_bytes", fake_download)
    monkeypatch.setattr(dataset_jobs, "apply_audio_fx", fake_apply)
    monkeypatch.setattr(dataset_jobs, "validate_and_crop_audio", fake_validate)
    monkeypatch.setattr(dataset_jobs, "_upload_bytes", fake_upload)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", fake_normalize)

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

    assert normalize_calls == [b"ref-seg", b"clip-1-bytes", b"clip-2-bytes"]
    assert items[0]["id"] == "ref_audio__seg_000.wav"


def test_prepare_dataset_items_keeps_reference_derived_clips(monkeypatch):
    async def fake_download(_storage, ref):
        return f"bytes-for:{ref}".encode("utf-8")

    async def fake_apply(audio_bytes, **_kwargs):
        return audio_bytes

    async def fake_validate(_segments, _character_name):
        return [
            ("ref_audio__seg_000.wav", b"ref-derived", "ref_audio", True, "reference fragment", None, False, True),
            ("clip-1.wav", b"clip-1-bytes", "prompt-1", True, "hello there", None, False, False),
        ]

    async def fake_upload(_storage, payload, key, **_kwargs):
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
                {"filename": "clip-1.wav", "prompt_id": "prompt-1", "text": "hello there", "s3_url": "s3://bucket/clone-1.wav"},
            ],
        )
    )

    assert [item["id"] for item in items] == ["ref_audio__seg_000.wav", "clip-1.wav"]


def test_package_dataset_chooses_longest_reference_segment(monkeypatch):
    uploaded_payloads: list[tuple[str, bytes]] = []
    wav_by_ref = {
        "s3://bucket/ref-short.wav": _make_wav_bytes(3.0),
        "s3://bucket/ref-long.wav": _make_wav_bytes(7.0),
        "s3://bucket/clip-1.wav": _make_wav_bytes(6.0),
    }

    async def fake_download(_storage, ref):
        return wav_by_ref[ref]

    async def fake_upload(_storage, payload, key, **_kwargs):
        uploaded_payloads.append((key, payload))
        return f"https://bucket/{key}"

    monkeypatch.setattr(dataset_jobs, "_download_bytes", fake_download)
    monkeypatch.setattr(dataset_jobs, "_upload_bytes", fake_upload)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda audio_bytes: audio_bytes)

    package = asyncio.run(
        dataset_jobs.package_dataset(
            object(),
            book_id="book-1",
            character_id="char-1",
            job_id="job-1",
            dataset_items=[
                {"id": "ref_audio__seg_000.wav", "s3_url": "s3://bucket/ref-short.wav", "text": "short ref", "is_reference": True, "included": True},
                {"id": "ref_audio__seg_001.wav", "s3_url": "s3://bucket/ref-long.wav", "text": "long ref", "is_reference": True, "included": True},
                {"id": "clip-1.wav", "s3_url": "s3://bucket/clip-1.wav", "text": "hello there", "is_reference": False, "included": True},
            ],
        )
    )

    assert package["dataset_s3_key"] == "datasets/book-1/dataset_char-1_job-1.zip"
    assert len(uploaded_payloads) == 1

    _key, payload = uploaded_payloads[0]
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert zf.read("data/ref_audio.wav") == wav_by_ref["s3://bucket/ref-long.wav"]
        rows = [json.loads(line) for line in zf.read("train.jsonl").decode("utf-8").splitlines()]

    assert [row["audio"] for row in rows] == ["./data/clip-1.wav"]


def test_build_split_results_discards_clip_that_becomes_too_short_after_normalization(monkeypatch):
    class _FakeSlice:
        def __init__(self, duration_ms: int):
            self.duration_ms = duration_ms

        def __len__(self):
            return self.duration_ms

        def export(self, buf, format="wav"):
            buf.write(_make_wav_bytes(max(self.duration_ms / 1000.0, 0.1)))

    class _DurationAwareAudioSegment:
        def __init__(self, duration_ms: int):
            self.duration_ms = duration_ms

        @classmethod
        def from_wav(cls, stream):
            if hasattr(stream, "read"):
                raw = stream.read()
                stream = io.BytesIO(raw)
            with wave.open(stream, "rb") as wf:
                duration_ms = int((wf.getnframes() / wf.getframerate()) * 1000)
            return cls(duration_ms)

        def __len__(self):
            return self.duration_ms

        def __getitem__(self, key):
            start = 0 if key.start is None else key.start
            stop = self.duration_ms if key.stop is None else key.stop
            return _FakeSlice(max(0, stop - start))

    fake_pydub = types.SimpleNamespace(AudioSegment=_DurationAwareAudioSegment)
    monkeypatch.setitem(sys.modules, "pydub", fake_pydub)
    monkeypatch.setattr(dataset_jobs, "_normalize_audio_for_dataset_sync", lambda _audio_bytes: _make_wav_bytes(2.3))

    segments = [
        {"start": 0.0, "end": 5.5, "text": "hello there", "speaker": "SPEAKER_00"},
    ]

    result = dataset_jobs._build_split_results(
        "clip.wav",
        _make_wav_bytes(5.5),
        "prompt-1",
        segments,
        main_speaker="SPEAKER_00",
    )

    assert result == []
