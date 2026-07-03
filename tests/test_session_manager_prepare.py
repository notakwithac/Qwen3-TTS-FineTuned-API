import asyncio
from pathlib import Path
import threading
import time

import pytest

import session_manager
from session_manager import DuplicateActiveSessionError, SessionManager, TrainingConflictError


class _StubJob:
    def __init__(
        self,
        job_id: str,
        checkpoint_path: Path,
        character_id: str = "character-1",
        speaker_name: str = "speaker_custom",
    ):
        self.job_id = job_id
        self.checkpoint_path = str(checkpoint_path)
        self.character_id = character_id
        self.speaker_name = speaker_name
        self.s3_model_key = None


class _StubPipeline:
    def __init__(self, jobs):
        self.jobs = jobs
        self.touch_calls = []
        self.model_source_calls = []

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def touch_job(self, job_id: str):
        self.touch_calls.append(job_id)

    def apply_model_source(self, job, model_source):
        self.model_source_calls.append((job.job_id, model_source))


class _StubInferenceManager:
    def __init__(self, *, loaded_count: int = 2, max_models: int = 4, training_active: bool = False):
        self.loaded_count = loaded_count
        self.max_models = max_models
        self.training_active = training_active
        self.load_calls = []
        self.unpin_calls = []
        self.unload_calls = []

    @property
    def stats(self):
        return {"shared_model_min_headroom_gb": 4.0}

    def load_for_session(self, cache_key, checkpoint_path, speaker_name, session_id=""):
        self.load_calls.append((cache_key, checkpoint_path, speaker_name, session_id))

    def is_training_active_or_requested(self):
        return self.training_active

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
    pipeline = _StubPipeline(
        {"job-1": _StubJob("job-1", checkpoint_path, speaker_name="Narrator")}
    )
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
                "model_source": {
                    "provider": "huggingface",
                    "repo_id": "notakwithac/Qwen3-Narrator-job-1",
                    "filename": "checkpoint-epoch-14.zip",
                    "checkpoint_epoch": 14,
                },
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
    assert len(pipeline.model_source_calls) == 1

    torn_down = await manager.teardown_session("session-1")

    assert torn_down is True
    assert inference.unpin_calls == [
        (str(checkpoint_path), "session-1"),
        (expected_replica_key, "session-1"),
    ]
    assert inference.unload_calls == [expected_replica_key]


@pytest.mark.asyncio
async def test_prepare_session_uses_job_speaker_name_not_display_name(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint-epoch-14"
    checkpoint_path.mkdir()
    (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")

    inference = _StubInferenceManager(loaded_count=1, max_models=2)
    pipeline = _StubPipeline(
        {
            "job-1": _StubJob(
                "job-1",
                checkpoint_path,
                speaker_name="david_copperfield___child",
            )
        }
    )
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
        session_id="session-slug",
        characters=[
            {
                "job_id": "job-1",
                "character_name": "David Copperfield - Child",
                "line_count": 1,
            }
        ],
    )

    assert session.character_plans["job-1"].character_name == "david_copperfield___child"
    assert inference.load_calls == [
        (
            str(checkpoint_path),
            str(checkpoint_path),
            "david_copperfield___child",
            "session-slug",
        )
    ]


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


@pytest.mark.asyncio
async def test_prepare_session_rejects_when_training_is_active(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint-epoch-14"
    checkpoint_path.mkdir()
    (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")

    inference = _StubInferenceManager(loaded_count=2, max_models=4, training_active=True)
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

    with pytest.raises(TrainingConflictError, match="training is active"):
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

    assert inference.load_calls == []


@pytest.mark.asyncio
async def test_prepare_session_serializes_missing_checkpoint_restores(monkeypatch, tmp_path):
    class RestorePipeline(_StubPipeline):
        def __init__(self, jobs):
            super().__init__(jobs)
            self.active_restores = 0
            self.max_active_restores = 0
            self.restore_lock = threading.Lock()

        def resolve_checkpoint_path(self, job):
            with self.restore_lock:
                self.active_restores += 1
                self.max_active_restores = max(self.max_active_restores, self.active_restores)
            time.sleep(0.05)
            checkpoint = Path(job.checkpoint_path)
            checkpoint.mkdir(parents=True, exist_ok=True)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            with self.restore_lock:
                self.active_restores -= 1
            return str(checkpoint), 14

    jobs = {
        job_id: _StubJob(job_id, tmp_path / job_id / "checkpoint-epoch-14")
        for job_id in ("job-1", "job-2")
    }
    pipeline = RestorePipeline(jobs)
    inference = _StubInferenceManager(loaded_count=0, max_models=4)
    manager = SessionManager(
        inference_manager=inference,
        pipeline=pipeline,
        storage=None,
        replica_threshold=500,
        max_replicas=1,
        batch_size=1,
    )
    monkeypatch.setattr(manager, "_get_available_vram", lambda: 40.0)
    monkeypatch.setattr(session_manager.CharacterWorker, "start", lambda self: None)

    await manager.prepare_session(
        session_id="session-restore-limit",
        characters=[
            {"job_id": "job-1", "character_name": "One", "line_count": 1},
            {"job_id": "job-2", "character_name": "Two", "line_count": 1},
        ],
    )

    assert pipeline.max_active_restores == 1


@pytest.mark.asyncio
async def test_prepare_session_preloads_only_capacity_when_unique_models_exceed_limit(
    monkeypatch,
    tmp_path,
):
    jobs = {}
    characters = []
    for index in range(3):
        job_id = f"job-{index}"
        checkpoint_path = tmp_path / job_id / "checkpoint-epoch-14"
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")
        jobs[job_id] = _StubJob(job_id, checkpoint_path, speaker_name=f"speaker-{index}")
        characters.append(
            {
                "job_id": job_id,
                "character_name": f"Character {index}",
                "line_count": 12,
            }
        )

    inference = _StubInferenceManager(loaded_count=0, max_models=1)
    manager = SessionManager(
        inference_manager=inference,
        pipeline=_StubPipeline(jobs),
        storage=None,
        max_replicas=1,
        batch_size=1,
    )
    monkeypatch.setattr(manager, "_get_available_vram", lambda: 40.0)
    monkeypatch.setattr(session_manager.CharacterWorker, "start", lambda self: None)

    session = await manager.prepare_session(
        session_id="session-oversubscribed",
        characters=characters,
    )

    assert session.status is session_manager.SessionStatus.READY
    assert len(session.character_plans) == 3
    assert len(session.workers) == 3
    assert len(inference.load_calls) == 1
    assert inference.load_calls[0][-1] == ""


@pytest.mark.asyncio
async def test_prepare_session_rolls_back_models_loaded_before_a_preload_failure(
    monkeypatch,
    tmp_path,
):
    class _RollbackInference(_StubInferenceManager):
        def __init__(self):
            super().__init__(loaded_count=0, max_models=2)
            self.loaded_paths = []

        def load_for_session(self, cache_key, checkpoint_path, speaker_name, session_id=""):
            self.load_calls.append((cache_key, checkpoint_path, speaker_name, session_id))
            if "job-2" in checkpoint_path:
                raise RuntimeError("forced second preload failure")
            self.loaded_paths.append(cache_key)
            self.loaded_count = len(self.loaded_paths)

        def unload_specific(self, cache_key):
            self.unload_calls.append(cache_key)
            if cache_key in self.loaded_paths:
                self.loaded_paths.remove(cache_key)
                self.loaded_count = len(self.loaded_paths)
                return True
            return False

    jobs = {}
    characters = []
    for job_id in ("job-1", "job-2"):
        checkpoint_path = tmp_path / job_id / "checkpoint-epoch-14"
        checkpoint_path.mkdir(parents=True)
        (checkpoint_path / "config.json").write_text("{}", encoding="utf-8")
        jobs[job_id] = _StubJob(job_id, checkpoint_path)
        characters.append({"job_id": job_id, "character_name": job_id, "line_count": 1})

    inference = _RollbackInference()
    manager = SessionManager(
        inference_manager=inference,
        pipeline=_StubPipeline(jobs),
        storage=None,
        max_replicas=1,
        batch_size=1,
    )
    monkeypatch.setattr(manager, "_get_available_vram", lambda: 40.0)

    with pytest.raises(RuntimeError, match="forced second preload failure"):
        await manager.prepare_session(
            session_id="session-rollback",
            characters=characters,
        )

    first_key = str(tmp_path / "job-1" / "checkpoint-epoch-14")
    assert inference.loaded_paths == []
    assert (first_key, "session-rollback") in inference.unpin_calls
    assert first_key in inference.unload_calls
