from pathlib import Path
import sys
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline


def _make_job(tmp_path, *, num_epochs=10):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return pipeline.Job(
        job_id="job-epoch-test",
        speaker_name="Narrator",
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(output_dir),
        num_epochs=num_epochs,
    )


def test_resolve_checkpoint_path_uses_requested_saved_epoch(tmp_path):
    job = _make_job(tmp_path, num_epochs=10)
    checkpoint_6 = Path(job.output_dir) / "checkpoint-epoch-6"
    checkpoint_9 = Path(job.output_dir) / "checkpoint-epoch-9"
    checkpoint_6.mkdir()
    checkpoint_9.mkdir()
    (checkpoint_6 / "model.safetensors").write_text("ckpt6", encoding="utf-8")
    (checkpoint_9 / "model.safetensors").write_text("ckpt9", encoding="utf-8")

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)

    resolved_path, resolved_epoch = pipeline.Pipeline.resolve_checkpoint_path(
        pipe,
        job,
        checkpoint_epoch=6,
    )

    assert resolved_epoch == 6
    assert Path(resolved_path).name == "checkpoint-epoch-6"


def test_resolve_checkpoint_path_rejects_unavailable_epoch_without_s3(tmp_path):
    job = _make_job(tmp_path, num_epochs=10)
    checkpoint_9 = Path(job.output_dir) / "checkpoint-epoch-9"
    checkpoint_9.mkdir()
    (checkpoint_9 / "model.safetensors").write_text("ckpt9", encoding="utf-8")
    job.checkpoint_path = str(checkpoint_9)

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)

    with pytest.raises(ValueError, match="not available"):
        pipeline.Pipeline.resolve_checkpoint_path(
            pipe,
            job,
            checkpoint_epoch=7,
        )


def test_upload_latest_checkpoint_to_s3_only_uploads_last_epoch(tmp_path, monkeypatch):
    job = _make_job(tmp_path, num_epochs=10)
    for epoch in (6, 7, 9):
        checkpoint_dir = Path(job.output_dir) / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_text(f"ckpt{epoch}", encoding="utf-8")

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
    uploaded_epochs = []

    def fake_upload_model_to_s3(_job, checkpoint_path, checkpoint_epoch=None):
        uploaded_epochs.append((Path(checkpoint_path).name, checkpoint_epoch))
        return f"s3://bucket/{checkpoint_epoch}.zip"

    monkeypatch.setattr(pipe, "_upload_model_to_s3", fake_upload_model_to_s3)

    checkpoint_s3_keys = pipeline.Pipeline._upload_latest_checkpoint_to_s3(pipe, job)

    assert uploaded_epochs == [("checkpoint-epoch-9", 9)]
    assert checkpoint_s3_keys == {"9": "s3://bucket/9.zip"}
    assert job.s3_model_key == "s3://bucket/9.zip"
    assert job.checkpoint_s3_keys == {"9": "s3://bucket/9.zip"}


def test_has_verified_checkpoint_backup_checks_actual_object_existence(tmp_path, monkeypatch):
    job = _make_job(tmp_path, num_epochs=10)
    job.s3_model_key = "s3://bucket/9.zip"

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)

    class _Storage:
        is_configured = True
        has_read_backend = True
        bucket = "bucket"

        @staticmethod
        def object_exists(key):
            return key == "s3://bucket/9.zip"

    monkeypatch.setattr("pipeline.storage", _Storage(), raising=False)
    import types
    monkeypatch.setitem(sys.modules, "storage", types.SimpleNamespace(storage=_Storage()))

    assert pipeline.Pipeline._has_verified_checkpoint_backup(pipe, job) is True


def test_retry_job_failed_with_only_job_json_falls_back_to_full_restart(tmp_path, monkeypatch):
    job = _make_job(tmp_path, num_epochs=10)
    job.status = pipeline.JobStatus.FAILED
    job.s3_model_key = "s3://bucket/missing.zip"

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
    pipe.jobs_dir = tmp_path / "jobs"
    pipe._lock = None
    started = []

    class _Storage:
        is_configured = True
        has_read_backend = True
        bucket = "bucket"

        @staticmethod
        def object_exists(key):
            return False

    monkeypatch.setitem(sys.modules, "storage", __import__("types").SimpleNamespace(storage=_Storage()))
    monkeypatch.setattr(pipe, "get_job", lambda job_id: job)
    monkeypatch.setattr(pipe, "start_job", lambda job_id: started.append(job_id))

    retried = pipeline.Pipeline.retry_job(pipe, job.job_id)

    assert retried is job
    assert job.status == pipeline.JobStatus.QUEUED
    assert started == [job.job_id]


def test_get_job_prefers_s3_ready_metadata_over_local_failed_job(tmp_path, monkeypatch):
    local_job = _make_job(tmp_path, num_epochs=10)
    local_job.job_dir = str(tmp_path / "jobs" / local_job.job_id)
    local_job.status = pipeline.JobStatus.FAILED
    local_job.error = "local stale failure"
    local_job.save()

    remote_data = local_job.to_dict()
    remote_data["status"] = pipeline.JobStatus.READY
    remote_data["error"] = None
    remote_data["available_checkpoint_epochs"] = [9]
    remote_data["checkpoint_s3_keys"] = {"9": "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip"}
    remote_data["s3_model_key"] = "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip"

    class _Storage:
        is_configured = True
        has_read_backend = True

        @staticmethod
        def object_exists(key):
            return key in {
                "jobs/job-epoch-test/job.json",
                "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip",
            }

        @staticmethod
        def download_bytes(key):
            assert key == "jobs/job-epoch-test/job.json"
            return json.dumps(remote_data).encode("utf-8")

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
    pipe.jobs = {}
    pipe.jobs_dir = tmp_path / "jobs"
    pipe._lock = __import__("threading").Lock()
    monkeypatch.setitem(sys.modules, "storage", __import__("types").SimpleNamespace(storage=_Storage()))

    job = pipeline.Pipeline.get_job(pipe, "job-epoch-test")

    assert job.status == pipeline.JobStatus.READY
    assert job.error is None
    assert job.s3_model_key == "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip"
    saved = json.loads((Path(local_job.job_dir) / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ready"


def test_get_job_promotes_local_failed_job_when_checkpoint_exists_on_s3(tmp_path, monkeypatch):
    job = _make_job(tmp_path, num_epochs=10)
    job.job_dir = str(tmp_path / "jobs" / job.job_id)
    job.status = pipeline.JobStatus.FAILED
    job.error = "load failed after upload"
    job.available_checkpoint_epochs = [9]
    job.checkpoint_s3_keys = {"9": "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip"}
    job.s3_model_key = "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip"
    job.save()

    class _Storage:
        is_configured = True
        has_read_backend = True

        @staticmethod
        def object_exists(key):
            return key == "s3://pathnam-ai/models/job-epoch-test/checkpoint-epoch-9.zip"

        @staticmethod
        def download_bytes(_key):
            raise AssertionError("remote job metadata should not be required")

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
    pipe.jobs = {}
    pipe.jobs_dir = tmp_path / "jobs"
    pipe._lock = __import__("threading").Lock()
    monkeypatch.setitem(sys.modules, "storage", __import__("types").SimpleNamespace(storage=_Storage()))

    resolved = pipeline.Pipeline.get_job(pipe, "job-epoch-test")

    assert resolved.status == pipeline.JobStatus.READY
    assert resolved.error is None
    assert resolved.progress["stage"] == "ready"
