import asyncio
from pathlib import Path

import pytest

import session_manager
from session_manager import DuplicateActiveSessionError, SessionManager


class _StubJob:
    def __init__(self, job_id: str, checkpoint_path: Path, character_id: str = "character-1"):
        self.job_id = job_id
        self.checkpoint_path = str(checkpoint_path)
        self.character_id = character_id
        self.s3_model_key = None


class _StubPipeline:
    def __init__(self, jobs):
        self.jobs = jobs
        self.touch_calls = []

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def touch_job(self, job_id: str):
        self.touch_calls.append(job_id)


class _StubInferenceManager:
    def __init__(self, *, loaded_count: int = 2, max_models: int = 4):
        self.loaded_count = loaded_count
        self.max_models = max_models
        self.load_calls = []
        self.unpin_calls = []
        self.unload_calls = []

    @property
    def stats(self):
        return {"shared_model_min_headroom_gb": 4.0}

    def load_for_session(self, cache_key, checkpoint_path, speaker_name, session_id=""):
        self.load_calls.append((cache_key, checkpoint_path, speaker_name, session_id))

    def unpin_session(self, cache_key, session_id):
        self.unpin_calls.append((cache_key, session_id))

    def unload_specific(self, cache_key):
        self.unload_calls.append(cache_key)
        return True


@pytest.mark.asyncio
async def test_prepare_session_preloads_safe_replica_workers_and_tears_down_extra_replicas(
    monkeypatch,
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint-epoch-14"
    checkpoint_path.mkdir()
    (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")

    inference = _StubInferenceManager(loaded_count=2, max_models=4)
    pipeline = _StubPipeline({"job-1": _StubJob("job-1", checkpoint_path)})
    manager = SessionManager(
        inference_manager=inference,
        pipeline=pipeline,
        storage=None,
        replica_threshold=500,
        max_replicas=4,
        batch_size=8,
    )

    monkeypatch.setattr(manager, "_get_available_vram", lambda: 40.0)
    monkeypatch.setattr(
        session_manager.CharacterWorker,
        "start",
        lambda self: setattr(self, "_running", True),
    )

    async def _stop(self):
        self._running = False

    monkeypatch.setattr(session_manager.CharacterWorker, "stop", _stop)

    session = await manager.prepare_session(
        session_id="session-1",
        characters=[
            {
                "job_id": "job-1",
                "character_name": "Narrator",
                "line_count": 1200,
            }
        ],
    )

    plan = session.character_plans["job-1"]
    expected_replica_key = f"{checkpoint_path}::replica-1"

    assert plan.replicas == 2
    assert plan.replica_keys == [str(checkpoint_path), expected_replica_key]
    assert len(session.workers) == 2
    assert inference.load_calls == [
        (str(checkpoint_path), str(checkpoint_path), "Narrator", "session-1"),
        (expected_replica_key, str(checkpoint_path), "Narrator", "session-1"),
    ]
    assert pipeline.touch_calls == ["job-1", "job-1"]

    torn_down = await manager.teardown_session("session-1")

    assert torn_down is True
    assert inference.unpin_calls == [
        (str(checkpoint_path), "session-1"),
        (expected_replica_key, "session-1"),
    ]
    assert inference.unload_calls == [expected_replica_key]


def test_calculate_replicas_respects_available_replica_slots():
    manager = SessionManager(
        inference_manager=_StubInferenceManager(),
        pipeline=_StubPipeline({}),
        storage=None,
        replica_threshold=500,
        max_replicas=4,
    )

    replicas = manager._calculate_replicas(
        [
            {"job_id": "job-1", "line_count": 3000},
            {"job_id": "job-2", "line_count": 3000},
        ],
        available_vram_gb=40.0,
        additional_replica_slots=1,
    )

    assert replicas == {
        "job-1": 2,
        "job-2": 1,
    }


@pytest.mark.asyncio
async def test_prepare_session_rejects_duplicate_active_chapter_workload(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint-epoch-14"
    checkpoint_path.mkdir()
    (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")

    inference = _StubInferenceManager(loaded_count=2, max_models=4)
    pipeline = _StubPipeline({"job-1": _StubJob("job-1", checkpoint_path)})
    manager = SessionManager(
        inference_manager=inference,
        pipeline=pipeline,
        storage=None,
        replica_threshold=500,
        max_replicas=4,
        batch_size=1,
    )

    monkeypatch.setattr(manager, "_get_available_vram", lambda: 40.0)
    monkeypatch.setattr(
        session_manager.CharacterWorker,
        "start",
        lambda self: setattr(self, "_running", True),
    )

    async def _stop(self):
        self._running = False

    monkeypatch.setattr(session_manager.CharacterWorker, "stop", _stop)

    await manager.prepare_session(
        session_id="session-1",
        characters=[
            {
                "job_id": "job-1",
                "character_name": "Narrator",
                "line_count": 17,
            }
        ],
        book_id="book-1",
        chapter_id="chapter-1",
    )

    with pytest.raises(DuplicateActiveSessionError) as excinfo:
        await manager.prepare_session(
            session_id="session-2",
            characters=[
                {
                    "job_id": "job-1",
                    "character_name": "Narrator",
                    "line_count": 17,
                }
            ],
            book_id="book-1",
            chapter_id="chapter-1",
        )

    assert excinfo.value.active_session_id == "session-1"
