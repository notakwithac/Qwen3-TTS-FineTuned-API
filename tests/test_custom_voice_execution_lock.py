import threading
import time

from inference_manager import InferenceManager


class _StubCustomVoiceModel:
    def __init__(self, started_event, release_event, counters, calls=None):
        self._started_event = started_event
        self._release_event = release_event
        self._counters = counters
        self._calls = calls if calls is not None else []

    def generate_custom_voice(
        self,
        text,
        language=None,
        speaker=None,
        instruct=None,
        max_new_tokens=None,
        **kwargs,
    ):
        with self._counters["lock"]:
            self._counters["active"] += 1
            self._counters["max_active"] = max(
                self._counters["max_active"],
                self._counters["active"],
            )
        self._calls.append(
            {
                "text": text,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
                "max_new_tokens": max_new_tokens,
                "kwargs": kwargs,
            }
        )
        self._started_event.set()
        self._release_event.wait(timeout=5.0)
        with self._counters["lock"]:
            self._counters["active"] -= 1
        batch_size = len(text) if isinstance(text, list) else 1
        return [object() for _ in range(batch_size)], 24000


def test_custom_voice_batch_generation_is_serialized_per_checkpoint(monkeypatch):
    manager = InferenceManager(device="cpu", max_models=2)
    checkpoint_path = "/tmp/custom-voice"
    started_event = threading.Event()
    release_event = threading.Event()
    counters = {
        "active": 0,
        "max_active": 0,
        "lock": threading.Lock(),
    }
    model = _StubCustomVoiceModel(started_event, release_event, counters)

    monkeypatch.setattr(manager, "_get_model", lambda *_args, **_kwargs: (model, "Narrator"))
    monkeypatch.setattr(manager, "_encode_wav", lambda *_args, **_kwargs: b"wav")

    errors = []

    def _run_batch():
        try:
            manager.generate_batch(
                texts=["hello"],
                checkpoint_path=checkpoint_path,
                speaker_name="Narrator",
            )
        except Exception as exc:  # pragma: no cover - surfaced via assertion below
            errors.append(exc)

    first = threading.Thread(target=_run_batch)
    second = threading.Thread(target=_run_batch)

    first.start()
    assert started_event.wait(timeout=2.0)
    second.start()

    time.sleep(0.1)
    assert counters["max_active"] == 1

    release_event.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert counters["max_active"] == 1


def test_custom_voice_single_generation_is_serialized_per_checkpoint(monkeypatch):
    manager = InferenceManager(device="cpu", max_models=2)
    checkpoint_path = "/tmp/custom-voice"
    started_event = threading.Event()
    release_event = threading.Event()
    counters = {
        "active": 0,
        "max_active": 0,
        "lock": threading.Lock(),
    }
    model = _StubCustomVoiceModel(started_event, release_event, counters)

    monkeypatch.setattr(manager, "_get_model", lambda *_args, **_kwargs: (model, "Narrator"))
    monkeypatch.setattr(manager, "_encode_wav", lambda *_args, **_kwargs: b"wav")

    errors = []

    def _run_single():
        try:
            manager.generate(
                text="hello",
                checkpoint_path=checkpoint_path,
                speaker_name="Narrator",
            )
        except Exception as exc:  # pragma: no cover - surfaced via assertion below
            errors.append(exc)

    first = threading.Thread(target=_run_single)
    second = threading.Thread(target=_run_single)

    first.start()
    assert started_event.wait(timeout=2.0)
    second.start()

    time.sleep(0.1)
    assert counters["max_active"] == 1

    release_event.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert counters["max_active"] == 1


def test_custom_voice_batch_generation_does_not_force_max_new_tokens(monkeypatch):
    manager = InferenceManager(device="cpu", max_models=2)
    calls = []
    model = _StubCustomVoiceModel(
        threading.Event(),
        threading.Event(),
        {"active": 0, "max_active": 0, "lock": threading.Lock()},
        calls=calls,
    )

    monkeypatch.setattr(manager, "_get_model", lambda *_args, **_kwargs: (model, "Narrator"))
    monkeypatch.setattr(manager, "_encode_wav", lambda *_args, **_kwargs: b"wav")

    result, sr = manager.generate_batch(
        texts=["hello", "world"],
        checkpoint_path="/tmp/custom-voice",
        speaker_name="Narrator",
    )

    assert result == [b"wav", b"wav"]
    assert sr == 24000
    assert calls[-1]["max_new_tokens"] is None


def test_custom_voice_single_generation_does_not_force_max_new_tokens(monkeypatch):
    manager = InferenceManager(device="cpu", max_models=2)
    calls = []
    model = _StubCustomVoiceModel(
        threading.Event(),
        threading.Event(),
        {"active": 0, "max_active": 0, "lock": threading.Lock()},
        calls=calls,
    )

    monkeypatch.setattr(manager, "_get_model", lambda *_args, **_kwargs: (model, "Narrator"))
    monkeypatch.setattr(manager, "_encode_wav", lambda *_args, **_kwargs: b"wav")

    result, sr = manager.generate(
        text="hello",
        checkpoint_path="/tmp/custom-voice",
        speaker_name="Narrator",
    )

    assert result == b"wav"
    assert sr == 24000
    assert calls[-1]["max_new_tokens"] is None


def test_custom_voice_single_generation_passes_explicit_max_new_tokens(monkeypatch):
    manager = InferenceManager(device="cpu", max_models=2)
    calls = []
    model = _StubCustomVoiceModel(
        threading.Event(),
        threading.Event(),
        {"active": 0, "max_active": 0, "lock": threading.Lock()},
        calls=calls,
    )

    monkeypatch.setattr(manager, "_get_model", lambda *_args, **_kwargs: (model, "Narrator"))
    monkeypatch.setattr(manager, "_encode_wav", lambda *_args, **_kwargs: b"wav")

    result, sr = manager.generate(
        text="hello",
        checkpoint_path="/tmp/custom-voice",
        speaker_name="Narrator",
        max_new_tokens=3072,
    )

    assert result == b"wav"
    assert sr == 24000
    assert calls[-1]["max_new_tokens"] == 3072
