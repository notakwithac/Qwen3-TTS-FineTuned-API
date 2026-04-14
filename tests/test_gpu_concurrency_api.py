import sys
import threading
import types
from datetime import datetime, timezone

from fastapi.testclient import TestClient


class _DummyBroadcast:
    def __init__(self, *_args, **_kwargs):
        pass

    async def connect(self):
        return None

    async def disconnect(self):
        return None

    async def publish(self, *_args, **_kwargs):
        return None

    def subscribe(self, *_args, **_kwargs):
        class _Subscriber:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _Subscriber()


sse_module = types.ModuleType("sse_starlette.sse")
sse_module.EventSourceResponse = object
sys.modules.setdefault("sse_starlette", types.ModuleType("sse_starlette"))
sys.modules["sse_starlette.sse"] = sse_module

broadcast_module = types.ModuleType("broadcaster")
broadcast_module.Broadcast = _DummyBroadcast
sys.modules["broadcaster"] = broadcast_module

import api_server
from session_manager import DuplicateActiveSessionError


class StubInference:
    def __init__(self, fail_on=None):
        self._max_models = 5
        self._shared_model_replicas = {
            "voice_design": 1,
            "voice_clone": 1,
        }
        self._shared_model_min_headroom_gb = 3.0
        self._loaded_checkpoints = []
        self.fail_on = fail_on
        self.load_calls = []

    @property
    def stats(self):
        return {
            "max_models": self._max_models,
            "loaded_count": len(self._loaded_checkpoints),
            "loaded_checkpoints": list(self._loaded_checkpoints),
            "shared_model_replicas": dict(self._shared_model_replicas),
            "shared_model_min_headroom_gb": self._shared_model_min_headroom_gb,
            "inference_limiter": {
                "capacity": 5,
                "active": 0,
                "available": 5,
                "holders": {},
                "waiting": {},
            },
            "active_requests": 0,
            "last_request_started_at": None,
            "last_request_finished_at": None,
            "idle_started_at": "2026-04-11T00:00:00Z",
            "idle_seconds": 60.0,
            "gpu_memory_total_gb": 48.0,
            "gpu_memory_allocated_gb": 0.0,
            "gpu_memory_reserved_gb": 0.0,
            "gpu_memory_free_gb": 48.0,
        }

    def load_voice_design(self):
        self.load_calls.append("voice_design")
        if self.fail_on == "voice_design":
            raise RuntimeError("voice design preload failed")
        checkpoint = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign::replica-0"
        if checkpoint not in self._loaded_checkpoints:
            self._loaded_checkpoints.append(checkpoint)

    def load_voice_clone(self):
        self.load_calls.append("voice_clone")
        if self.fail_on == "voice_clone":
            raise RuntimeError("voice clone preload failed")
        checkpoint = "Qwen/Qwen3-TTS-12Hz-1.7B-Base::replica-0"
        if checkpoint not in self._loaded_checkpoints:
            self._loaded_checkpoints.append(checkpoint)

    def update_runtime_config(
        self,
        *,
        max_models=None,
        shared_model_replicas=None,
        shared_model_min_headroom_gb=None,
    ):
        if max_models is not None:
            self._max_models = max_models
        if shared_model_replicas is not None:
            self._shared_model_replicas.update(shared_model_replicas)
        if shared_model_min_headroom_gb is not None:
            self._shared_model_min_headroom_gb = shared_model_min_headroom_gb


class StubPipeline:
    def __init__(self, inference):
        self.inference = inference

    def shutdown(self):
        return None


def _patch_startup_dependencies(monkeypatch):
    monkeypatch.setattr(api_server.session_mgr, "start_cleanup_loop", lambda: None)
    monkeypatch.setattr(api_server.metrics_collector, "start", lambda: None)
    monkeypatch.setattr(api_server.metrics_collector, "stop", lambda: None)


def test_gpu_status_reports_scheduler_idle_and_cooldown_fields(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_server.session_mgr,
        "scheduler_snapshot",
        lambda: {
            "total_sessions": 1,
            "active_sessions": 0,
            "queued_session_items": 0,
            "active_workers": 0,
            "status_counts": {
                "preparing": 0,
                "ready": 0,
                "processing": 0,
                "completed": 1,
                "failed": 0,
                "cancelled": 0,
            },
        },
    )
    monkeypatch.setattr(api_server.ops_log, "get_running", lambda: [])
    monkeypatch.setattr(api_server, "GPU_COOLDOWN_SECONDS", 1200)
    monkeypatch.setattr(
        api_server.time,
        "time",
        lambda: datetime(2026, 4, 11, 0, 30, tzinfo=timezone.utc).timestamp(),
    )

    with TestClient(api_server.app) as client:
        response = client.get("/gpu/status")

    body = response.json()
    assert response.status_code == 200
    assert body["is_idle"] is True
    assert body["queued_requests"] == 0
    assert body["cooldown_seconds"] == 1200
    assert body["cooldown_ready"] is True
    assert body["active_sessions"] == 0
    assert body["running_operations"] == 0


def test_gpu_status_hides_idle_window_when_backlog_exists(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_server.session_mgr,
        "scheduler_snapshot",
        lambda: {
            "total_sessions": 1,
            "active_sessions": 1,
            "queued_session_items": 4,
            "active_workers": 1,
            "status_counts": {
                "preparing": 0,
                "ready": 0,
                "processing": 1,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
            },
        },
    )
    monkeypatch.setattr(api_server.ops_log, "get_running", lambda: [{"op_name": "session_worker_batch"}])

    with TestClient(api_server.app) as client:
        response = client.get("/gpu/status")

    body = response.json()
    assert response.status_code == 200
    assert body["is_idle"] is False
    assert body["queued_requests"] == 4
    assert body["idle_started_at"] is None
    assert body["idle_seconds"] is None
    assert body["cooldown_ready"] is False


def test_get_gpu_concurrency_returns_effective_runtime_config(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)

    with TestClient(api_server.app) as client:
        response = client.get("/gpu/concurrency")

    assert response.status_code == 200
    assert response.json() == {
        "gpu_max_models": 5,
        "voice_design_replicas": 1,
        "voice_clone_replicas": 1,
        "shared_model_min_headroom_gb": 3.0,
    }


def test_post_gpu_concurrency_updates_runtime_config(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)

    with TestClient(api_server.app) as client:
        response = client.post(
            "/gpu/concurrency",
            json={
                "gpu_max_models": 7,
                "voice_design_replicas": 2,
                "voice_clone_replicas": 2,
                "shared_model_min_headroom_gb": 4,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "gpu_max_models": 7,
        "voice_design_replicas": 2,
        "voice_clone_replicas": 2,
        "shared_model_min_headroom_gb": 4.0,
    }
    assert inference.stats["max_models"] == 7
    assert inference.stats["shared_model_replicas"] == {
        "voice_design": 2,
        "voice_clone": 2,
    }
    assert inference.stats["shared_model_min_headroom_gb"] == 4.0


def test_session_prepare_returns_conflict_for_duplicate_active_session(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)

    async def _raise_duplicate(**_kwargs):
        raise DuplicateActiveSessionError(
            "A matching session is already active for this chapter workload (session-1).",
            active_session_id="session-1",
        )

    monkeypatch.setattr(api_server.session_mgr, "prepare_session", _raise_duplicate)

    with TestClient(api_server.app) as client:
        response = client.post(
            "/session/prepare",
            json={
                "session_id": "session-2",
                "book_id": "book-1",
                "chapter_id": "chapter-1",
                "characters": [
                    {
                        "job_id": "job-1",
                        "character_name": "Narrator",
                        "line_count": 17,
                    }
                ],
            },
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": {
            "message": "A matching session is already active for this chapter workload (session-1).",
            "active_session_id": "session-1",
        }
    }


def test_post_gpu_concurrency_rejects_invalid_payload(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)

    with TestClient(api_server.app) as client:
        response = client.post(
            "/gpu/concurrency",
            json={
                "gpu_max_models": 0,
                "voice_design_replicas": 1,
                "voice_clone_replicas": 1,
                "shared_model_min_headroom_gb": -1,
            },
        )

    assert response.status_code == 422


def test_lifespan_preloads_shared_models_on_startup(monkeypatch):
    inference = StubInference()
    events = []

    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_server.ops_log,
        "log_event",
        lambda event_name, job_id=None, extra=None, level=None: events.append(
            {"event_name": event_name, "extra": extra, "level": level}
        ),
    )

    with TestClient(api_server.app):
        pass

    assert inference.load_calls == ["voice_design", "voice_clone"]
    assert inference.stats["loaded_checkpoints"] == [
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign::replica-0",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base::replica-0",
    ]
    assert [event["event_name"] for event in events] == [
        "startup_preload_started",
        "startup_preload_finished",
        "startup_preload_started",
        "startup_preload_finished",
    ]
    assert events[1]["extra"]["loaded_checkpoints"] == [
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign::replica-0",
    ]
    assert events[3]["extra"]["loaded_checkpoints"] == [
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign::replica-0",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base::replica-0",
    ]


def test_lifespan_logs_preload_failure_and_continues_boot(monkeypatch):
    inference = StubInference(fail_on="voice_design")
    events = []

    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_server.ops_log,
        "log_event",
        lambda event_name, job_id=None, extra=None, level=None: events.append(
            {"event_name": event_name, "extra": extra, "level": level}
        ),
    )

    with TestClient(api_server.app):
        pass

    assert inference.load_calls == ["voice_design", "voice_clone"]
    assert inference.stats["loaded_checkpoints"] == [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base::replica-0",
    ]
    assert [event["event_name"] for event in events] == [
        "startup_preload_started",
        "startup_preload_failed",
        "startup_preload_started",
        "startup_preload_finished",
    ]
    assert events[1]["extra"]["continue_boot"] is True
    assert events[1]["extra"]["error"] == "voice design preload failed"


def test_gpu_status_is_available_while_startup_preload_runs(monkeypatch):
    inference = StubInference()
    allow_preload_finish = threading.Event()
    preload_started = threading.Event()

    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    _patch_startup_dependencies(monkeypatch)

    async def slow_preload():
        preload_started.set()
        await api_server.asyncio.to_thread(allow_preload_finish.wait, 2.0)

    monkeypatch.setattr(api_server, "_startup_preload_shared_models", slow_preload)

    with TestClient(api_server.app) as client:
        assert preload_started.wait(timeout=1.0) is True
        response = client.get("/gpu/status")
        assert response.status_code == 200
        assert response.json()["startup_preload"]["in_progress"] is True
        allow_preload_finish.set()


def test_custom_voice_batcher_is_configured_for_serial_processing(monkeypatch):
    inference = StubInference()
    monkeypatch.setattr(api_server, "pipeline", StubPipeline(inference))
    monkeypatch.setattr(api_server, "CUSTOM_VOICE_API_BATCH_SIZE", 1)
    api_server.custom_voice_batchers.clear()

    batcher = api_server.get_custom_voice_batcher(
        "job-1",
        "/tmp/checkpoint-epoch-14",
        "Narrator",
    )

    assert batcher.batch_size == 1
    assert batcher.executor._max_workers == 1

    api_server.custom_voice_batchers.clear()
