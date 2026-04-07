import sys
import types
from types import SimpleNamespace

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
from pipeline import Job, JobStatus


def _patch_startup_dependencies(monkeypatch):
    monkeypatch.setattr(api_server.session_mgr, "start_cleanup_loop", lambda: None)
    monkeypatch.setattr(api_server.metrics_collector, "start", lambda: None)
    monkeypatch.setattr(api_server.metrics_collector, "stop", lambda: None)

    async def no_preload():
        return None

    monkeypatch.setattr(api_server, "_startup_preload_shared_models", no_preload)


def test_get_job_returns_pending_finetune_job(monkeypatch):
    _patch_startup_dependencies(monkeypatch)
    with api_server._pending_finetune_jobs_lock:
        api_server._pending_finetune_jobs.clear()

    req = SimpleNamespace(
        speaker_name="speaker_custom",
        num_epochs=15,
        batch_size=1,
        lr=1e-6,
        book_id="book-1",
        chapter_id="chapter-1",
        character_id="char-1",
    )
    pending = api_server._register_pending_finetune_job("job-pending-1", req)

    def fail_pipeline_lookup(_job_id: str):
        raise AssertionError("pending job lookups should not hit pipeline.get_job")

    monkeypatch.setattr(api_server.pipeline, "get_job", fail_pipeline_lookup)

    try:
        with TestClient(api_server.app) as client:
            response = client.get("/jobs/job-pending-1")
    finally:
        api_server._clear_pending_finetune_job("job-pending-1")

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-pending-1"
    assert response.json()["status"] == "queued"
    assert response.json()["speaker_name"] == pending["speaker_name"]
    assert response.json()["progress"]["stage"] == "queued"


def test_job_save_creates_missing_job_directory(tmp_path):
    job_dir = tmp_path / "nested" / "job-save-test"
    job = Job(
        job_id="job-save-test",
        speaker_name="speaker_custom",
        dataset_dir=str(job_dir / "dataset"),
        output_dir=str(job_dir / "output"),
        job_dir=str(job_dir),
    )
    job.status = JobStatus.QUEUED

    job.save()

    assert (job_dir / "job.json").exists()
