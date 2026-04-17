from pathlib import Path
import sys

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
        bucket = "bucket"

        @staticmethod
        def object_exists(key):
            return key == "9.zip"

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
