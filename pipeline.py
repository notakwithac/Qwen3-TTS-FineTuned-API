# coding=utf-8
"""Pipeline orchestrator — manages fine-tuning jobs from dataset to serving."""

import gc
import json
import logging
import os
import shutil
import sys
import threading
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from cuda_cleanup import safe_cuda_cleanup
from inference_manager import InferenceManager
from model_storage import (
    get_hf_model_storage_config,
    hf_checkpoint_repo_exists,
    upload_checkpoint_to_hf,
)
from model_storage import (
    restore_checkpoint_from_hf as hf_restore_checkpoint,
)
from ops_logger import ops_log

logger = logging.getLogger(__name__)
MIN_CUSTOM_CHECKPOINT_EPOCH = 6

# Add finetuning dir to path so we can import prepare_data / sft_12hz
_finetuning_dir = str(Path(__file__).parent / "finetuning")
if _finetuning_dir not in sys.path:
    sys.path.insert(0, _finetuning_dir)


def _checkpoint_dir_name(epoch: int) -> str:
    return f"checkpoint-epoch-{epoch}"


def _checkpoint_path_for_epoch(output_dir: str, epoch: int) -> Path:
    return Path(output_dir) / _checkpoint_dir_name(epoch)


def _extract_checkpoint_epoch(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    match = Path(path).name
    if match.startswith("checkpoint-epoch-"):
        try:
            return int(match.replace("checkpoint-epoch-", "", 1))
        except ValueError:
            return None
    return None


def _list_saved_checkpoint_epochs(output_dir: str) -> list[int]:
    root = Path(output_dir)
    if not root.exists():
        return []
    epochs: list[int] = []
    for item in root.iterdir():
        if not item.is_dir():
            continue
        epoch = _extract_checkpoint_epoch(str(item))
        if epoch is not None:
            epochs.append(epoch)
    return sorted(set(epochs))


def _normalize_s3_object_key(
    value: Optional[str], bucket: Optional[str] = None
) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.startswith("s3://"):
        without_scheme = raw[5:]
        parts = without_scheme.split("/", 1)
        if len(parts) == 2:
            obj_bucket, key = parts
            if bucket and obj_bucket != bucket:
                return None
            return key
        return None
    return raw


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


class JobStatus(StrEnum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    LOADING = "loading"
    RESTORING = "restoring"  # Downloading checkpoint from S3
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


ACTIVE_JOB_STATUSES = {
    JobStatus.QUEUED,
    JobStatus.PREPARING,
    JobStatus.TRAINING,
    JobStatus.LOADING,
    JobStatus.RESTORING,
}


class Job:
    """In-memory representation of one fine-tuning job."""

    def __init__(
        self,
        job_id: str,
        speaker_name: str,
        dataset_dir: str,
        output_dir: str,
        num_epochs: int = 10,
        batch_size: int = 1,
        lr: float = 2e-6,
        flash_attn: bool = True,
        book_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        character_id: Optional[str] = None,
        job_dir: Optional[str] = None,
        base_model_path: Optional[str] = None,
        s3_model_key: Optional[str] = None,
    ):
        self.job_id = job_id
        self.speaker_name = speaker_name
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.flash_attn = flash_attn
        self.book_id = book_id
        self.chapter_id = chapter_id
        self.character_id = character_id
        self.job_dir = job_dir
        self.base_model_path = base_model_path
        self.s3_model_key = s3_model_key

        self.status = JobStatus.QUEUED
        self.message = "To force retry, send force = true"
        self.progress: Dict[str, Any] = {}
        self.checkpoint_path: Optional[str] = None
        self.available_checkpoint_epochs: list[int] = []
        self.checkpoint_s3_keys: Dict[str, str] = {}
        self.hf_model_repo: Optional[str] = None
        self.hf_model_url: Optional[str] = None
        self.hf_model_filename: Optional[str] = None
        self.checkpoint_hf_repos: Dict[str, str] = {}
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.last_accessed_at = self.created_at
        self._thread: Optional[threading.Thread] = None
        self._cancel_requested = False

    def to_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "status": self.status,
            "speaker_name": self.speaker_name,
            "progress": self.progress,
            "checkpoint_path": self.checkpoint_path,
            "available_checkpoint_epochs": self.available_checkpoint_epochs,
            "checkpoint_s3_keys": self.checkpoint_s3_keys,
            "hf_model_repo": self.hf_model_repo,
            "hf_model_url": self.hf_model_url,
            "hf_model_filename": self.hf_model_filename,
            "checkpoint_hf_repos": self.checkpoint_hf_repos,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "s3_model_key": self.s3_model_key,
            "config": {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "flash_attn": self.flash_attn,
                "book_id": self.book_id,
                "chapter_id": self.chapter_id,
                "character_id": self.character_id,
                "base_model_path": self.base_model_path,
            },
            "message": self.message,
        }
        if self.status == JobStatus.READY:
            d["inference_url"] = f"/infer/{self.job_id}"
            d["last_accessed_at"] = self.last_accessed_at
        return d

    def save(self):
        """Persist job state to disk."""
        if not self.job_dir:
            return
        job_dir = Path(self.job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "job.json"
        try:
            with open(job_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save job {self.job_id}: {e}")

    @classmethod
    def from_dict(
        cls,
        data: dict,
        job_dir: str,
        *,
        mark_stale_active: bool = True,
    ) -> "Job":
        config = data.get("config", {})
        job = cls(
            job_id=data["job_id"],
            speaker_name=data["speaker_name"],
            dataset_dir=str(Path(job_dir) / "dataset"),
            output_dir=str(Path(job_dir) / "output"),
            num_epochs=config.get("num_epochs", 10),
            batch_size=config.get("batch_size", 1),
            lr=config.get("lr", 2e-6),
            flash_attn=config.get("flash_attn", True),
            book_id=config.get("book_id"),
            chapter_id=config.get("chapter_id"),
            character_id=config.get("character_id"),
            job_dir=job_dir,
            base_model_path=config.get("base_model_path"),
            s3_model_key=data.get("s3_model_key"),
        )
        job.status = data.get("status", JobStatus.QUEUED)
        job.message = data.get("message", job.message)
        job.progress = data.get("progress", {})
        job.checkpoint_path = data.get("checkpoint_path")
        job.available_checkpoint_epochs = [
            int(epoch)
            for epoch in data.get("available_checkpoint_epochs", [])
            if str(epoch).isdigit()
        ]
        raw_checkpoint_s3_keys = data.get("checkpoint_s3_keys", {})
        job.checkpoint_s3_keys = {
            str(epoch): str(s3_key)
            for epoch, s3_key in raw_checkpoint_s3_keys.items()
            if s3_key
        }
        job.hf_model_repo = data.get("hf_model_repo")
        job.hf_model_url = data.get("hf_model_url")
        job.hf_model_filename = data.get("hf_model_filename")
        raw_checkpoint_hf_repos = data.get("checkpoint_hf_repos", {})
        job.checkpoint_hf_repos = {
            str(epoch): str(repo_id)
            for epoch, repo_id in raw_checkpoint_hf_repos.items()
            if repo_id
        }
        job.error = data.get("error")
        job.created_at = data.get("created_at", job.created_at)
        job.finished_at = data.get("finished_at")
        if "last_accessed_at" in data:
            job.last_accessed_at = data["last_accessed_at"]

        # A loaded-from-disk job cannot have its original worker thread.
        # If the API process restarted or crashed while it was active, expose
        # it as failed instead of showing a phantom in-progress training job.
        if mark_stale_active and job.status in ACTIVE_JOB_STATUSES:
            job.status = JobStatus.FAILED
            job.error = job.error or (
                "Job was active when the API process stopped. "
                "Retry with force=true to start it again."
            )
            job.finished_at = job.finished_at or datetime.now(timezone.utc).isoformat()
            job.progress = {
                "stage": "failed",
                "detail": "Recovered stale active job from disk.",
                "previous_status": str(data.get("status", "")),
            }
            job.save()

        return job

    @classmethod
    def load(cls, job_dir: str) -> Optional["Job"]:
        """Load job state from disk."""
        job_file = Path(job_dir) / "job.json"
        if not job_file.exists():
            return None
        try:
            with open(job_file, "r") as f:
                data = json.load(f)
            return cls.from_dict(data, job_dir, mark_stale_active=True)
        except Exception as e:
            logger.error(f"Failed to load job from {job_dir}: {e}")
            return None

    def touch(self):
        """Update last access time."""
        self.last_accessed_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Orchestrates fine-tuning jobs and inference."""

    def __init__(
        self,
        base_dir: str = ".",
        jobs_dir: str = "jobs",
        device: str = "cuda:0",
        use_flash_attn: bool = True,
        idle_timeout_seconds: int = 300,
        max_concurrency: int = 2,
        max_models: int = 4,
        compile: bool = False,
        gpu_controller: Any = None,
        shared_model_replicas: Optional[Dict[str, int]] = None,
        shared_model_min_headroom_gb: float = 2.0,
    ):
        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_flash_attn = use_flash_attn
        self._gpu_controller = gpu_controller

        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._training_queue = threading.Semaphore(1)
        self._upload_threads: list = []  # Track in-progress S3 upload threads
        self.inference = InferenceManager(
            device=device,
            use_flash_attn=use_flash_attn,
            idle_timeout_seconds=idle_timeout_seconds,
            max_concurrency=max_concurrency,
            max_models=max_models,
            compile=compile,
            gpu_controller=gpu_controller,
            shared_model_replicas=shared_model_replicas,
            shared_model_min_headroom_gb=shared_model_min_headroom_gb,
        )
        self._restore_locks: Dict[str, threading.Lock] = {}

        # Background S3 sync: ensure all completed jobs are uploaded
        self._stop_sync_event = threading.Event()
        self._sync_thread = threading.Thread(
            target=self._s3_sync_worker,
            daemon=True,
            name="s3-sync-monitor",
        )
        self._sync_thread.start()

    # -- Lifecycle -----------------------------------------------------------

    def shutdown(self, timeout_per_thread: float = 300.0):
        """Wait for all in-progress S3 uploads to finish before shutdown.

        Called on graceful exit (Ctrl+C / SIGTERM) so no uploads are lost.
        """
        # Stop the background sync worker
        self._stop_sync_event.set()

        with self._lock:
            pending = [t for t in self._upload_threads if t.is_alive()]

        if not pending:
            logger.info("Shutdown: No pending S3 uploads.")
            return

        logger.warning(
            f"Shutdown: Waiting for {len(pending)} S3 upload(s) to complete "
            f"(timeout {timeout_per_thread}s each). Press Ctrl+C again to force quit."
        )
        for thread in pending:
            thread.join(timeout=timeout_per_thread)
            if thread.is_alive():
                logger.error(
                    f"Shutdown: Thread {thread.name} did not finish in time — upload may be incomplete."
                )
            else:
                logger.info(f"Shutdown: Thread {thread.name} completed.")

    def _refresh_available_checkpoints(self, job: Job) -> list[int]:
        epochs = _list_saved_checkpoint_epochs(job.output_dir)
        if epochs:
            job.available_checkpoint_epochs = epochs
        elif job.checkpoint_path:
            inferred_epoch = _extract_checkpoint_epoch(job.checkpoint_path)
            if inferred_epoch is not None:
                job.available_checkpoint_epochs = [inferred_epoch]
        return list(job.available_checkpoint_epochs)

    def _latest_checkpoint_epoch(self, job: Job) -> Optional[int]:
        epochs = self._refresh_available_checkpoints(job)
        if epochs:
            return epochs[-1]
        inferred_epoch = _extract_checkpoint_epoch(job.checkpoint_path)
        if inferred_epoch is not None:
            return inferred_epoch
        if job.num_epochs > 0:
            return job.num_epochs - 1
        return None

    def _upload_latest_checkpoint_to_s3(self, job: Job) -> dict[str, str]:
        latest_epoch = self._latest_checkpoint_epoch(job)
        if latest_epoch is None:
            return {}

        checkpoint_dir = _checkpoint_path_for_epoch(job.output_dir, latest_epoch)
        if not checkpoint_dir.is_dir():
            return {}

        s3_key = self._upload_model_to_s3(
            job,
            checkpoint_dir,
            checkpoint_epoch=latest_epoch,
        )
        checkpoint_s3_keys = {str(latest_epoch): s3_key}
        job.checkpoint_s3_keys = checkpoint_s3_keys
        job.s3_model_key = s3_key
        return checkpoint_s3_keys

    def _upload_latest_checkpoint_to_hf(self, job: Job) -> dict[str, str]:
        latest_epoch = self._latest_checkpoint_epoch(job)
        if latest_epoch is None:
            return {}

        checkpoint_dir = _checkpoint_path_for_epoch(job.output_dir, latest_epoch)
        if not checkpoint_dir.is_dir():
            return {}

        repo_id, model_url = upload_checkpoint_to_hf(
            job,
            checkpoint_dir,
            latest_epoch,
            checkpoint_dir_name=_checkpoint_dir_name(latest_epoch),
        )
        checkpoint_hf_repos = {str(latest_epoch): repo_id}
        job.checkpoint_hf_repos = checkpoint_hf_repos
        job.hf_model_repo = repo_id
        job.hf_model_url = model_url
        return checkpoint_hf_repos

    def _checkpoint_hf_repo_exists(
        self,
        job: Job,
        checkpoint_epoch: Optional[int] = None,
    ) -> bool:
        repo_id = None
        if checkpoint_epoch is not None:
            repo_id = job.checkpoint_hf_repos.get(str(checkpoint_epoch))
        if not repo_id:
            repo_id = job.hf_model_repo
        return hf_checkpoint_repo_exists(repo_id)

    def _checkpoint_s3_object_exists(
        self,
        job: Job,
        checkpoint_epoch: Optional[int] = None,
    ) -> bool:
        from storage import storage

        if not getattr(storage, "has_read_backend", storage.is_configured):
            return False

        s3_ref = None
        if checkpoint_epoch is not None:
            s3_ref = job.checkpoint_s3_keys.get(str(checkpoint_epoch))
        if not s3_ref:
            s3_ref = job.s3_model_key
        s3_key = _normalize_s3_object_key(s3_ref)
        if not s3_key:
            return False
        return storage.object_exists(s3_ref)

    def _has_verified_hf_checkpoint_backup(self, job: Job) -> bool:
        latest_epoch = self._latest_checkpoint_epoch(job)
        if latest_epoch is not None and self._checkpoint_hf_repo_exists(
            job, checkpoint_epoch=latest_epoch
        ):
            return True
        return self._checkpoint_hf_repo_exists(job)

    def _has_verified_s3_checkpoint_backup(self, job: Job) -> bool:
        latest_epoch = self._latest_checkpoint_epoch(job)
        if latest_epoch is not None and self._checkpoint_s3_object_exists(
            job, checkpoint_epoch=latest_epoch
        ):
            return True
        return self._checkpoint_s3_object_exists(job)

    def _has_verified_checkpoint_backup(self, job: Job) -> bool:
        if self._has_verified_hf_checkpoint_backup(job):
            return True
        latest_epoch = self._latest_checkpoint_epoch(job)
        if latest_epoch is not None and self._checkpoint_s3_object_exists(
            job, checkpoint_epoch=latest_epoch
        ):
            return True
        return self._checkpoint_s3_object_exists(job)

    def _has_remote_checkpoint_backup(self, job: Job) -> bool:
        return bool(
            job.hf_model_repo
            or job.checkpoint_hf_repos
            or job.s3_model_key
            or job.checkpoint_s3_keys
        )

    def _has_local_checkpoint(self, job: Job) -> bool:
        checkpoint_path = job.checkpoint_path
        if checkpoint_path and os.path.exists(str(checkpoint_path)):
            return True
        latest_epoch = self._latest_checkpoint_epoch(job)
        if latest_epoch is None:
            return False
        checkpoint_dir = _checkpoint_path_for_epoch(job.output_dir, latest_epoch)
        return checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())

    def _ensure_s3_backup_for_ready_job(self, job: Job) -> bool:
        if self._has_verified_checkpoint_backup(job):
            return True
        if not self._has_local_checkpoint(job):
            return False
        if get_hf_model_storage_config() is not None:
            checkpoint_hf_repos = self._upload_latest_checkpoint_to_hf(job)
            job.save()
            return bool(
                checkpoint_hf_repos
            ) and self._has_verified_hf_checkpoint_backup(job)
        checkpoint_s3_keys = self._upload_latest_checkpoint_to_s3(job)
        self._upload_job_json_to_s3(job)
        job.save()
        return bool(checkpoint_s3_keys) and self._has_verified_checkpoint_backup(job)

    def resolve_checkpoint_path(
        self,
        job: Job,
        checkpoint_epoch: Optional[int] = None,
    ) -> tuple[str, Optional[int]]:
        latest_epoch = self._latest_checkpoint_epoch(job)

        if checkpoint_epoch is not None:
            if checkpoint_epoch < MIN_CUSTOM_CHECKPOINT_EPOCH:
                raise ValueError(
                    f"checkpoint_epoch must be >= {MIN_CUSTOM_CHECKPOINT_EPOCH} for custom checkpoints"
                )
            if job.num_epochs > 0 and checkpoint_epoch >= job.num_epochs:
                raise ValueError(
                    f"checkpoint_epoch {checkpoint_epoch} exceeds last epoch {job.num_epochs - 1}"
                )

        requested_epoch = (
            checkpoint_epoch if checkpoint_epoch is not None else latest_epoch
        )
        if requested_epoch is not None:
            requested_path = _checkpoint_path_for_epoch(job.output_dir, requested_epoch)
            if requested_path.is_dir() and any(requested_path.iterdir()):
                return str(requested_path.resolve()), requested_epoch

        if checkpoint_epoch is None and job.checkpoint_path:
            resolved_path = Path(job.checkpoint_path)
            if not resolved_path.is_absolute():
                resolved_path = resolved_path.resolve()
            if resolved_path.exists():
                return str(resolved_path), _extract_checkpoint_epoch(str(resolved_path))

        if (
            checkpoint_epoch is not None
            and requested_epoch is not None
            and latest_epoch is not None
            and requested_epoch != latest_epoch
            and str(requested_epoch) not in job.checkpoint_s3_keys
            and str(requested_epoch) not in job.checkpoint_hf_repos
        ):
            raise ValueError(
                f"checkpoint epoch {requested_epoch} is not available for job {job.job_id}"
            )

        if job.checkpoint_hf_repos or job.hf_model_repo:
            restored_path = self._restore_checkpoint_from_hf(
                job, checkpoint_epoch=requested_epoch
            )
            return str(Path(restored_path).resolve()), requested_epoch

        if job.checkpoint_s3_keys or job.s3_model_key:
            restored_path = self._restore_checkpoint_from_s3(
                job, checkpoint_epoch=requested_epoch
            )
            return str(Path(restored_path).resolve()), requested_epoch

        raise ValueError(f"Job {job.job_id} has no checkpoint available")

    def apply_model_source(self, job: Job, model_source: dict[str, Any]) -> None:
        """Apply an authoritative request-time model source and persist the repair."""
        if model_source.get("provider") != "huggingface":
            raise ValueError("Only huggingface model sources are supported")
        repo_id = str(model_source.get("repo_id") or "").strip()
        if not repo_id or repo_id.count("/") != 1:
            raise ValueError("A Hugging Face repo_id in owner/repo form is required")
        filename_value = model_source.get("filename")
        filename = str(filename_value).strip() if filename_value else None
        if filename and (Path(filename).name != filename or filename in {".", ".."}):
            raise ValueError("Hugging Face filename must be a relative basename")
        epoch_value = model_source.get("checkpoint_epoch")
        checkpoint_epoch = int(epoch_value) if epoch_value is not None else None
        if checkpoint_epoch is not None and checkpoint_epoch < 0:
            raise ValueError("checkpoint_epoch must be >= 0")

        changed = (
            job.hf_model_repo != repo_id
            or job.hf_model_filename != filename
            or (
                checkpoint_epoch is not None
                and job.checkpoint_hf_repos.get(str(checkpoint_epoch)) != repo_id
            )
        )
        job.hf_model_repo = repo_id
        job.hf_model_url = f"https://huggingface.co/{repo_id}"
        job.hf_model_filename = filename
        if checkpoint_epoch is not None:
            job.checkpoint_hf_repos[str(checkpoint_epoch)] = repo_id
            job.available_checkpoint_epochs = sorted(
                set(job.available_checkpoint_epochs + [checkpoint_epoch])
            )
        if changed:
            job.save()
            self._upload_job_json_to_s3(job)

    def _s3_sync_worker(self, interval_seconds: float = 900.0):
        """Background thread: every 15 min, upload any READY job missing remote backup."""
        from storage import storage

        hf_enabled = get_hf_model_storage_config() is not None
        logger.info("Checkpoint sync monitor started (interval: 15 min).")
        while not self._stop_sync_event.wait(timeout=interval_seconds):
            if not hf_enabled and not storage.is_configured:
                continue
            try:
                with self._lock:
                    jobs_snapshot = list(self.jobs.values())

                for job in jobs_snapshot:
                    if self._stop_sync_event.is_set():
                        break
                    if job.status != JobStatus.READY:
                        continue
                    if job.hf_model_repo or job.s3_model_key:
                        continue
                    if not job.checkpoint_path or not os.path.exists(
                        str(job.checkpoint_path)
                    ):
                        continue

                    logger.warning(
                        "Checkpoint sync: Job %s (%s) has no remote backup — uploading now.",
                        job.job_id,
                        job.speaker_name,
                    )
                    with ops_log.operation("checkpoint_sync_upload", job_id=job.job_id):
                        try:
                            if hf_enabled:
                                checkpoint_refs = self._upload_latest_checkpoint_to_hf(
                                    job
                                )
                            else:
                                checkpoint_refs = self._upload_latest_checkpoint_to_s3(
                                    job
                                )
                                self._upload_job_json_to_s3(job)
                            job.save()
                            logger.info(
                                "Checkpoint sync: Job %s uploaded %d checkpoint(s) successfully",
                                job.job_id,
                                len(checkpoint_refs),
                            )
                        except Exception as e:
                            logger.error(
                                f"Checkpoint sync: Failed to upload job {job.job_id}: {e}"
                            )
                            raise
            except Exception as e:
                logger.error(f"Checkpoint sync worker error: {e}")
        logger.info("Checkpoint sync monitor stopped.")

    # -- Job management -----------------------------------------------------

    def create_job(
        self,
        dataset_zip_path: str,
        speaker_name: str = "speaker_custom",
        num_epochs: int = 10,
        batch_size: int = 1,
        lr: float = 2e-6,
        book_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        character_id: Optional[str] = None,
        base_model_path: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Job:
        """Create a new fine-tuning job from an uploaded dataset zip.

        The zip must contain:
          - train.jsonl
          - data/ directory with .wav files
        """
        if not job_id:
            job_id = uuid.uuid4().hex[:12]

        job_dir = self.jobs_dir / job_id

        # If job_id is reused and directory exists, clean it up
        if job_dir.exists():
            logger.info(
                f"Re-creating job {job_id}: Cleaning up existing directory {job_dir}"
            )
            shutil.rmtree(job_dir, ignore_errors=True)

        dataset_dir = job_dir / "dataset"
        output_dir = job_dir / "output"

        # Cleanup old jobs if disk is full before starting new one
        self._cleanup_disk_lru()

        dataset_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the zip
        with zipfile.ZipFile(dataset_zip_path, "r") as zf:
            zf.extractall(dataset_dir)

        # Verify train.jsonl exists
        train_jsonl = dataset_dir / "train.jsonl"
        if not train_jsonl.exists():
            # Check if it's nested in a subdirectory
            candidates = list(dataset_dir.rglob("train.jsonl"))
            if candidates:
                # Move everything up from the nested dir
                nested_dir = candidates[0].parent
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(dataset_dir / item.name))
                train_jsonl = dataset_dir / "train.jsonl"

        if not train_jsonl.exists():
            raise ValueError("Dataset zip must contain a train.jsonl file")

        job = Job(
            job_id=job_id,
            speaker_name=speaker_name,
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            flash_attn=self.use_flash_attn,
            book_id=book_id,
            chapter_id=chapter_id,
            character_id=character_id,
            job_dir=str(job_dir),
            base_model_path=base_model_path,
        )

        with self._lock:
            self.jobs[job_id] = job

        job.save()
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        logger.info(f"Lookup job {job_id}")
        memory_job = self.jobs.get(job_id)

        # Try to load from disk if not in memory
        disk_job = None
        job_dir = self.jobs_dir / job_id
        if memory_job:
            logger.info(f"Job {job_id} found in memory with status {memory_job.status}")
        elif job_dir.exists():
            disk_job = Job.load(str(job_dir))
            if disk_job:
                logger.info(f"Job {job_id} found on disk at {job_dir}")

        local_job = memory_job or disk_job

        # Always check S3 metadata too. A stale local failed job.json can remain
        # after a crash even when the checkpoint/job metadata exists remotely.
        logger.info(f"Checking S3 metadata for job {job_id}...")
        s3_job = self._restore_job_from_s3(job_id, register=False)

        job = self._choose_best_job_state(local_job, s3_job)
        if job:
            job = self._reconcile_checkpoint_readiness(job)
            with self._lock:
                self.jobs[job_id] = job
            job.save()
            return job

        logger.warning(f"Job {job_id} not found anywhere (memory, disk, S3)")
        return None

    def _job_state_score(self, job: Optional[Job]) -> int:
        if job is None:
            return -1
        if self._has_local_checkpoint(job) or self._has_verified_checkpoint_backup(job):
            return 100
        if job.status == JobStatus.READY:
            return 80
        if job.status in (JobStatus.LOADING, JobStatus.RESTORING):
            return 60
        if job.status in (JobStatus.TRAINING, JobStatus.PREPARING, JobStatus.QUEUED):
            return 40
        if job.status == JobStatus.CANCELLED:
            return 10
        if job.status == JobStatus.FAILED:
            return 0
        return 20

    def _choose_best_job_state(
        self, local_job: Optional[Job], s3_job: Optional[Job]
    ) -> Optional[Job]:
        if local_job is None:
            return s3_job
        if s3_job is None:
            return local_job
        local_score = self._job_state_score(local_job)
        s3_score = self._job_state_score(s3_job)
        chosen = s3_job if s3_score > local_score else local_job
        other = local_job if chosen is s3_job else s3_job

        if not chosen.s3_model_key and other.s3_model_key:
            chosen.s3_model_key = other.s3_model_key
        if not chosen.checkpoint_s3_keys and other.checkpoint_s3_keys:
            chosen.checkpoint_s3_keys = dict(other.checkpoint_s3_keys)
        if not chosen.hf_model_repo and other.hf_model_repo:
            chosen.hf_model_repo = other.hf_model_repo
        if not chosen.hf_model_url and other.hf_model_url:
            chosen.hf_model_url = other.hf_model_url
        if not chosen.checkpoint_hf_repos and other.checkpoint_hf_repos:
            chosen.checkpoint_hf_repos = dict(other.checkpoint_hf_repos)
        if not chosen.available_checkpoint_epochs and other.available_checkpoint_epochs:
            chosen.available_checkpoint_epochs = list(other.available_checkpoint_epochs)
        if (
            (
                not chosen.checkpoint_path
                or not os.path.exists(str(chosen.checkpoint_path))
            )
            and other.checkpoint_path
            and os.path.exists(str(other.checkpoint_path))
        ):
            chosen.checkpoint_path = other.checkpoint_path
        return chosen

    def _reconcile_checkpoint_readiness(self, job: Job) -> Job:
        has_local = self._has_local_checkpoint(job)
        has_remote = self._has_verified_checkpoint_backup(job)
        if has_local or has_remote:
            if job.status != JobStatus.READY:
                job.status = JobStatus.READY
                job.error = None
                job.finished_at = (
                    job.finished_at or datetime.now(timezone.utc).isoformat()
                )
                if has_local:
                    source = "local disk"
                elif self._has_verified_hf_checkpoint_backup(job):
                    source = "Hugging Face"
                else:
                    source = "S3"
                job.progress = {
                    "stage": "ready",
                    "detail": f"Model checkpoint found on {source} and ready for inference",
                    "inference_url": f"/infer/{job.job_id}",
                }
            if (
                has_local
                and job.checkpoint_path
                and os.path.exists(str(job.checkpoint_path))
            ):
                job.checkpoint_path = str(Path(job.checkpoint_path).resolve())
        return job

    def list_jobs(self) -> list:
        return [j.to_dict() for j in self.jobs.values()]

    def touch_job(self, job_id: str):
        """Mark a job as recently used."""
        job = self.jobs.get(job_id)
        if job:
            job.touch()

    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total

    def _cleanup_disk_lru(self, threshold_gb: float = 200.0):
        """Delete oldest jobs if disk usage exceeds threshold.

        For S3-backed jobs: only deletes the heavy checkpoint files, keeping
        job.json locally. The checkpoint can be re-downloaded on demand.
        For non-S3 jobs: deletes the entire job folder (legacy behavior).
        """
        if not self.jobs_dir.exists():
            return

        current_size = self._get_dir_size(self.jobs_dir)
        threshold_bytes = threshold_gb * (1024**3)

        if current_size <= threshold_bytes:
            return

        logger.info(
            f"Disk usage ({current_size / 1024**3:.2f}GB) exceeds threshold ({threshold_gb}GB). Pruning oldest jobs..."
        )

        # Sort completed/failed jobs by last_accessed_at
        candidates = []
        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.READY, JobStatus.FAILED, JobStatus.CANCELLED):
                candidates.append(job)

        # Oldest first
        candidates.sort(key=lambda x: x.last_accessed_at)

        from storage import storage as _storage

        hf_enabled = get_hf_model_storage_config() is not None

        for job in candidates:
            if current_size <= threshold_bytes:
                break

            job_dir = self.jobs_dir / job.job_id
            if not job_dir.exists():
                continue

            if job.hf_model_repo or job.s3_model_key:
                # Remote-backed: only delete the heavy output/ and dataset/ dirs,
                # keep job.json so get_job() still works without a remote round-trip
                for subdir in ["output", "dataset"]:
                    subdir_path = job_dir / subdir
                    if subdir_path.exists():
                        size = self._get_dir_size(subdir_path)
                        try:
                            shutil.rmtree(subdir_path)
                            current_size -= size
                            logger.info(
                                f"LRU: Deleted {subdir}/ for remote-backed job {job.job_id} (freed {size / 1024**2:.1f}MB)"
                            )
                        except Exception as e:
                            logger.error(
                                f"LRU: Failed to delete {subdir}/ for {job.job_id}: {e}"
                            )
            elif hf_enabled or _storage.is_configured:
                logger.warning(
                    f"LRU: Skipping job {job.job_id} — not yet uploaded to remote storage. Cannot safely prune."
                )
            else:
                # No S3 configured: delete the entire folder (last resort)
                size = self._get_dir_size(job_dir)
                try:
                    shutil.rmtree(job_dir)
                    current_size -= size
                    logger.info(
                        f"LRU: Deleted job {job.job_id} (freed {size / 1024**2:.1f}MB)"
                    )
                except Exception as e:
                    logger.error(f"LRU: Failed to delete job {job.job_id}: {e}")

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status in (JobStatus.READY, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job._cancel_requested = True
        job.status = JobStatus.CANCELLED
        return True

    def delete_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        # Cancel if running
        if job.status in (JobStatus.QUEUED, JobStatus.PREPARING, JobStatus.TRAINING):
            job._cancel_requested = True

        # Unload model if this job's model is loaded
        if self.inference.loaded_path and job.checkpoint_path:
            if self.inference.loaded_path == job.checkpoint_path:
                self.inference.unload()

        # Clean up files
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)

        with self._lock:
            del self.jobs[job_id]
        return True

    def retry_job(self, job_id: str) -> Optional[Job]:
        """Retry a failed or cancelled job.

        Smart retry: if training already completed (checkpoint exists locally
        or can be restored from S3), skips straight to loading for inference
        instead of re-running the entire pipeline.
        """
        job = self.get_job(job_id)
        if not job:
            return None

        if job.status == JobStatus.READY:
            if self._ensure_s3_backup_for_ready_job(job):
                return job
            if not self._has_local_checkpoint(job):
                job.status = JobStatus.QUEUED
                job.error = None
                job._cancel_requested = False
                job.progress = {
                    "stage": "queued",
                    "detail": "Rebuilding missing checkpoint backup from dataset...",
                }
                self.start_job(job_id)
                return job
            return job  # Already complete, and local checkpoint still exists.

        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            return None  # Only allow retrying if it actually failed or died.

        # Check if training already completed (late-stage failure)
        checkpoint_exists = self._has_local_checkpoint(job)
        has_s3_backup = self._has_verified_checkpoint_backup(job)

        if checkpoint_exists or has_s3_backup:
            # Training succeeded — skip to Stage 3 (load for inference)
            job.status = JobStatus.LOADING
            job.error = None
            job._cancel_requested = False
            job.progress = {
                "stage": "loading",
                "detail": "Retrying: loading model for inference (training already completed)...",
            }
            job.save()

            thread = threading.Thread(
                target=self._retry_load_only, args=(job,), daemon=True
            )
            job._thread = thread
            thread.start()
        else:
            # Training never completed — full restart
            job.status = JobStatus.QUEUED
            job.error = None
            job._cancel_requested = False
            job.progress = {}
            self.start_job(job_id)

        return job

    def _retry_load_only(self, job: Job):
        """Load a model for inference (skipping training), used by smart retry."""
        op = ops_log.start("retry_load", job_id=job.job_id)
        try:
            cp, resolved_epoch = self.resolve_checkpoint_path(job)

            # Free GPU memory before loading
            safe_cuda_cleanup("before smart-retry load")

            job.status = JobStatus.LOADING
            job.progress = {
                "stage": "loading",
                "detail": "Loading fine-tuned model for inference...",
            }
            job.save()

            self.inference.load(cp, job.speaker_name)

            job.checkpoint_path = str(Path(cp).resolve())  # always absolute
            if resolved_epoch is not None:
                job.available_checkpoint_epochs = sorted(
                    set(job.available_checkpoint_epochs + [resolved_epoch])
                )
            job.status = JobStatus.READY
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.progress = {
                "stage": "ready",
                "detail": "Model loaded and ready for inference",
                "inference_url": f"/infer/{job.job_id}",
            }
            job.save()
            ops_log.end(op)
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job.save()
            ops_log.fail(op, str(e))

    # -- Pipeline execution -------------------------------------------------

    def start_job(self, job_id: str):
        """Start the fine-tuning pipeline in a background thread."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        thread = threading.Thread(target=self._run_pipeline, args=(job, 0), daemon=True)
        job._thread = thread
        thread.start()

    @staticmethod
    def _is_corrupt_model_error(error: Exception) -> bool:
        """Detect safetensors/model corruption errors."""
        err_str = str(error)
        markers = [
            "deserializing header",
            "incomplete metadata",
            "file not fully covered",
            "HeaderTooLarge",
            "invalid header",
        ]
        return any(m in err_str for m in markers)

    @staticmethod
    def _clear_hf_cache(model_name: str):
        """Clear the HuggingFace cache for a specific model."""
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # HF cache folder format: models--Org--ModelName
        folder_name = f"models--{model_name.replace('/', '--')}"
        cache_path = cache_dir / folder_name
        if cache_path.exists():
            logger.warning(f"Clearing corrupted HF cache: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
            return True
        return False

    def _run_pipeline(self, job: Job, attempt: int = 0):
        """Execute the 3-stage pipeline: prepare → train → serve."""
        pipeline_op = ops_log.start(
            "pipeline_total",
            job_id=job.job_id,
            extra={
                "speaker_name": job.speaker_name,
                "num_epochs": job.num_epochs,
                "attempt": attempt,
            },
        )
        try:
            # Stage 1: Data preparation
            # Wait for our turn in the GPU training queue
            job.status = JobStatus.QUEUED
            job.progress = {
                "stage": "queued",
                "detail": "Waiting in queue for next available slot...",
            }

            self._training_queue.acquire()

            # When training is exclusive, cached inference models should not keep
            # occupying VRAM while we wait to enter the training section.
            if self._gpu_controller and not self._gpu_controller.allow_concurrent:
                try:
                    unload_op = ops_log.start("training_pre_unload", job_id=job.job_id)
                    loaded_before = self.inference.loaded_count
                    if loaded_before > 0:
                        logger.info(
                            "Preparing exclusive training for job %s by unloading %d cached inference model(s).",
                            job.job_id,
                            loaded_before,
                        )
                        self.inference.unload()
                    ops_log.end(unload_op, extra={"unloaded_models": loaded_before})
                except Exception as unload_exc:
                    logger.warning(
                        "Failed to unload cached inference models before training job %s: %s",
                        job.job_id,
                        unload_exc,
                    )

            if self._gpu_controller:
                self._gpu_controller.begin_training(job.job_id)

            job.status = JobStatus.PREPARING
            job.progress = {
                "stage": "preparing",
                "detail": "Encoding audio to codec tokens...",
            }
            job.save()

            train_jsonl = os.path.join(job.dataset_dir, "train.jsonl")
            prepared_jsonl = os.path.join(job.dataset_dir, "train_with_codes.jsonl")

            from prepare_data import prepare_programmatic

            def on_prepare_progress(current, total):
                job.progress = {
                    "stage": "preparing",
                    "current": current,
                    "total": total,
                    "detail": f"Encoded {current}/{total} audio files",
                }
                job.save()

            prep_op = ops_log.start("prepare_data", job_id=job.job_id)
            prepare_programmatic(
                input_jsonl=train_jsonl,
                output_jsonl=prepared_jsonl,
                device=self.device,
                batch_size=int(os.environ.get("DATASET_PREP_BATCH_SIZE", "1")),
                on_progress=on_prepare_progress,
            )
            ops_log.end(prep_op)

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                ops_log.fail(pipeline_op, "cancelled")
                self._training_queue.release()
                return

            # Free tokenizer GPU memory before training
            safe_cuda_cleanup("after data preparation", synchronize=True)

            # Stage 2: Training
            job.status = JobStatus.TRAINING
            job.progress = {
                "stage": "training",
                "epoch": 0,
                "total_epochs": job.num_epochs,
            }
            job.save()

            from sft_12hz import train_programmatic

            def on_train_progress(info: dict):
                job.progress = {
                    "stage": "training",
                    **info,
                }
                job.save()

            training_config = {
                "num_epochs": job.num_epochs,
                "batch_size": job.batch_size,
                "lr": job.lr,
                "trainable_scope": os.environ.get("TRAIN_TRAINABLE_SCOPE", "full"),
                "optimizer": os.environ.get("TRAIN_OPTIMIZER", "adamw"),
                "max_total_tokens": int(os.environ.get("TRAIN_MAX_TOTAL_TOKENS", "0")),
            }
            train_op = ops_log.start(
                "training", job_id=job.job_id, extra=training_config
            )

            init_model = (
                job.base_model_path
                if job.base_model_path and os.path.exists(job.base_model_path)
                else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            )

            checkpoint_path = train_programmatic(
                config={
                    "init_model_path": init_model,
                    "output_model_path": job.output_dir,
                    "train_jsonl": prepared_jsonl,
                    "batch_size": job.batch_size,
                    "lr": job.lr,
                    "num_epochs": job.num_epochs,
                    "speaker_name": job.speaker_name,
                    "save_interval": job.num_epochs,  # Only save the last/final one by default
                    "save_from_epoch": int(
                        os.environ.get(
                            "TRAIN_SAVE_FROM_EPOCH",
                            str(max(job.num_epochs - 1, 0)),
                        )
                    ),
                    "save_last": True,
                    "flash_attn": job.flash_attn,
                    "attn_implementation": os.environ.get(
                        "TRAIN_ATTN_IMPLEMENTATION", "sdpa"
                    ),
                    "gradient_checkpointing": os.environ.get(
                        "TRAIN_GRADIENT_CHECKPOINTING", "1"
                    )
                    == "1",
                    "trainable_scope": training_config["trainable_scope"],
                    "optimizer": training_config["optimizer"],
                    "weight_decay": float(os.environ.get("TRAIN_WEIGHT_DECAY", "0.01")),
                    "max_text_tokens": int(
                        os.environ.get("TRAIN_MAX_TEXT_TOKENS", "0")
                    ),
                    "max_codec_tokens": int(
                        os.environ.get("TRAIN_MAX_CODEC_TOKENS", "0")
                    ),
                    "max_total_tokens": int(
                        os.environ.get("TRAIN_MAX_TOTAL_TOKENS", "0")
                    ),
                    "output_hidden_states": os.environ.get(
                        "TRAIN_OUTPUT_HIDDEN_STATES", "0"
                    )
                    == "1",
                    "lr_scheduler": "cosine",
                    "warmup_ratio": 0.05,
                    "resume": True,
                },
                on_progress=on_train_progress,
            )
            ops_log.end(train_op, extra={"checkpoint_path": str(checkpoint_path)})
            job.available_checkpoint_epochs = self._refresh_available_checkpoints(job)

            # --- MODEL OFFLOAD: upload checkpoint to HF or S3 (Background Thread) ---
            from storage import storage

            hf_enabled = get_hf_model_storage_config() is not None
            if hf_enabled or storage.is_configured:

                def run_checkpoint_upload(j: Job):
                    if hf_enabled:
                        j.progress = {
                            "stage": "offloading",
                            "detail": "Uploading model checkpoint to Hugging Face in background...",
                        }
                    else:
                        j.progress = {
                            "stage": "offloading",
                            "detail": "Zipping and uploading model to S3 in background...",
                        }
                    j.save()
                    offload_op = ops_log.start("model_offload", job_id=j.job_id)
                    try:
                        if hf_enabled:
                            checkpoint_refs = self._upload_latest_checkpoint_to_hf(j)
                        else:
                            checkpoint_refs = self._upload_latest_checkpoint_to_s3(j)
                            self._upload_job_json_to_s3(j)
                        j.save()
                        ops_log.end(
                            offload_op, extra={"checkpoint_count": len(checkpoint_refs)}
                        )
                    except Exception as err:
                        ops_log.fail(offload_op, f"Model upload failed: {err}")

                upload_thread = threading.Thread(
                    target=run_checkpoint_upload,
                    args=(job,),
                    daemon=False,
                    name=f"checkpoint-upload-{job.job_id[:8]}",
                )
                with self._lock:
                    self._upload_threads.append(upload_thread)
                upload_thread.start()

            # AGGRESSIVE CLEANUP: Remove intermediate runs and raw dataset
            cleanup_op = ops_log.start("cleanup", job_id=job.job_id)
            try:
                # 1. Clear dataset tokens/raw-audio (optional, but saves space)
                if os.path.exists(job.dataset_dir):
                    shutil.rmtree(job.dataset_dir)

                # 2. Clear only checkpoints outside the retained 6..last set.
                keep_epochs = set(job.available_checkpoint_epochs)
                for entry in os.listdir(job.output_dir):
                    entry_path = os.path.join(job.output_dir, entry)
                    if not os.path.isdir(entry_path):
                        continue
                    entry_epoch = _extract_checkpoint_epoch(entry_path)
                    if entry_epoch is not None and entry_epoch not in keep_epochs:
                        shutil.rmtree(entry_path)

                ops_log.end(cleanup_op)
            except Exception as e:
                ops_log.fail(cleanup_op, f"Cleanup failed: {e}")

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                ops_log.fail(pipeline_op, "cancelled")
                self._training_queue.release()
                return

            job.checkpoint_path = str(
                Path(checkpoint_path).resolve()
            )  # always absolute — prevents HF from_pretrained misinterpreting as a repo id

            # Free training GPU memory before loading for inference
            safe_cuda_cleanup("after training", synchronize=True)

            # Release GPU training lock before Stage 3 (Inference loading)
            if self._gpu_controller:
                self._gpu_controller.end_training(job.job_id)

            # Stage 3: Load for inference
            job.status = JobStatus.LOADING
            job.progress = {
                "stage": "loading",
                "detail": "Loading fine-tuned model for inference...",
            }
            job.save()

            latest_checkpoint_path, resolved_epoch = self.resolve_checkpoint_path(job)
            self.inference.load(latest_checkpoint_path, job.speaker_name)
            job.checkpoint_path = latest_checkpoint_path
            if resolved_epoch is not None:
                job.available_checkpoint_epochs = sorted(
                    set(job.available_checkpoint_epochs + [resolved_epoch])
                )

            job.status = JobStatus.READY
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.progress = {
                "stage": "ready",
                "detail": "Model loaded and ready for inference",
                "inference_url": f"/infer/{job.job_id}",
            }
            job.save()
            ops_log.end(pipeline_op)
            self._training_queue.release()

        except Exception as e:
            # Auto-retry on corrupted model cache (safetensors deserialization errors)
            if self._is_corrupt_model_error(e) and attempt < 1:
                logger.warning(
                    f"Job {job.job_id}: Detected corrupted model cache (attempt {attempt}). "
                    f"Clearing cache and retrying..."
                )
                ops_log.fail(pipeline_op, f"Corrupt cache (auto-retrying): {e}")

                # Clear the HF cache for known base models
                for model_name in [
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                ]:
                    self._clear_hf_cache(model_name)

                # Also clear if using a custom base_model_path from HuggingFace
                if (
                    job.base_model_path
                    and "/" in job.base_model_path
                    and not os.path.isabs(job.base_model_path)
                ):
                    self._clear_hf_cache(job.base_model_path)

                # Free GPU and retry
                safe_cuda_cleanup("before retry")

                # Release queue before recursive call (recursive call will re-acquire)
                self._training_queue.release()

                job.status = JobStatus.QUEUED
                job.error = None
                job.progress = {
                    "stage": "queued",
                    "detail": "Auto-retrying after clearing corrupted model cache...",
                }
                job.save()

                # Re-run pipeline with incremented attempt
                # (recursive call manages its own queue acquire/release)
                self._run_pipeline(job, attempt + 1)
                return  # Skip the finally release — already released above

            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.save()
            if self._gpu_controller:
                self._gpu_controller.end_training(job.job_id)
            ops_log.fail(pipeline_op, str(e))
            self._training_queue.release()

    # -- Inference -----------------------------------------------------------

    def generate(
        self,
        job_id: str,
        text: str,
        language: str = "English",
        instruct: str = "",
        checkpoint_path: Optional[str] = None,
        speaker_name: Optional[str] = None,
    ) -> tuple[bytes, int]:
        """Generate speech using a completed job's model."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        if job.status not in (JobStatus.READY, JobStatus.RESTORING):
            raise ValueError(f"Job {job_id} is not ready (status: {job.status})")

        cp = checkpoint_path
        if cp is None:
            cp, _ = self.resolve_checkpoint_path(job)
        spk = speaker_name or job.speaker_name

        return self.inference.generate(
            text=text,
            checkpoint_path=cp,
            speaker_name=spk,
            language=language,
            instruct=instruct,
        )

    def _upload_model_to_s3(
        self,
        job: Job,
        checkpoint_path: Path,
        checkpoint_epoch: Optional[int] = None,
    ):
        """Zips and uploads the fine-tuned model to S3."""
        import tempfile

        from storage import storage

        book_folder = job.book_id or "unsorted"
        speaker_name = job.speaker_name
        if checkpoint_epoch is None:
            zip_filename = f"{speaker_name}_{job.job_id}.zip"
            s3_key = f"models/{book_folder}/{speaker_name}/{zip_filename}"
        else:
            zip_filename = f"{_checkpoint_dir_name(checkpoint_epoch)}.zip"
            s3_key = f"models/{book_folder}/{speaker_name}/{job.job_id}/{zip_filename}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_zip_path = os.path.join(tmp_dir, f"{speaker_name}_{job.job_id}")

            archive_path = shutil.make_archive(
                base_zip_path, "zip", root_dir=str(checkpoint_path)
            )

            # Upload to S3
            storage.upload_file(archive_path, s3_key, content_type="application/zip")

        return s3_key

    def _upload_job_json_to_s3(self, job: Job):
        """Upload job.json to S3 for easy access and restoration."""
        from storage import storage

        try:
            s3_key = f"jobs/{job.job_id}/job.json"
            job_data = json.dumps(job.to_dict(), indent=2)
            storage.upload_text(job_data, s3_key)
            logger.info(f"Uploaded job.json to S3: {s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload job.json to S3 for {job.job_id}: {e}")

    def _restore_job_from_s3(
        self, job_id: str, *, register: bool = True
    ) -> Optional[Job]:
        """Try to restore a job's metadata from S3 (lightweight — no model download)."""
        from storage import storage

        if not getattr(storage, "has_read_backend", storage.is_configured):
            logger.warning(
                f"S3 restoration skipped for {job_id}: Storage not configured"
            )
            return None

        s3_key = f"jobs/{job_id}/job.json"
        try:
            describe_backends = getattr(storage, "describe_read_backends", None)
            if callable(describe_backends):
                logger.info(
                    "Looking for job metadata key %s across storage backends: %s",
                    s3_key,
                    describe_backends(),
                )
            data_bytes = storage.download_bytes(s3_key)
            data = json.loads(data_bytes.decode("utf-8"))
            job_dir = self.jobs_dir / job_id
            if register:
                job_dir.mkdir(parents=True, exist_ok=True)
                with open(job_dir / "job.json", "w") as f:
                    json.dump(data, f, indent=2)

            job = Job.from_dict(data, str(job_dir), mark_stale_active=False)
            if job:
                if register:
                    with self._lock:
                        self.jobs[job_id] = job
                logger.info(f"Restored job {job_id} metadata from S3")
                return job
        except Exception as e:
            logger.warning(
                f"Job metadata not found or unreadable on S3 at {s3_key}: {e}"
            )
        return None

    def _restore_checkpoint_from_hf(
        self,
        job: Job,
        checkpoint_epoch: Optional[int] = None,
    ) -> str:
        """Download a checkpoint snapshot from Hugging Face.

        Returns the local checkpoint path.
        Raises ValueError if restoration fails.
        """
        latest_epoch = self._latest_checkpoint_epoch(job)
        target_epoch = (
            checkpoint_epoch if checkpoint_epoch is not None else latest_epoch
        )
        repo_id = None
        if target_epoch is not None:
            repo_id = job.checkpoint_hf_repos.get(str(target_epoch))
        if not repo_id:
            repo_id = job.hf_model_repo

        if not repo_id:
            raise ValueError(f"Job {job.job_id} has no HF model repo")

        job_dir = self.jobs_dir / job.job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if target_epoch is not None:
            checkpoint_dir = _checkpoint_path_for_epoch(job.output_dir, target_epoch)
        else:
            checkpoint_dir = output_dir / f"checkpoint_{job.speaker_name}"
        checkpoint_path = str(checkpoint_dir.resolve())

        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            logger.info(
                f"Checkpoint for job {job.job_id} already exists, skipping HF restore."
            )
            job.checkpoint_path = checkpoint_path
            return checkpoint_path

        with self._lock:
            if job.job_id not in self._restore_locks:
                self._restore_locks[job.job_id] = threading.Lock()
            lock = self._restore_locks[job.job_id]

        with lock:
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                if checkpoint_epoch is None or (
                    latest_epoch is not None and target_epoch == latest_epoch
                ):
                    job.checkpoint_path = checkpoint_path
                return checkpoint_path

            prev_status = job.status
            prev_progress = dict(job.progress)
            update_default_checkpoint = checkpoint_epoch is None or (
                latest_epoch is not None and target_epoch == latest_epoch
            )
            if update_default_checkpoint:
                job.status = JobStatus.RESTORING
                job.progress = {
                    "stage": "restoring",
                    "detail": "Downloading model from Hugging Face...",
                }
                job.save()

            restore_op = ops_log.start(
                "checkpoint_restore",
                job_id=job.job_id,
                extra={
                    "hf_repo": repo_id,
                    "checkpoint_epoch": target_epoch,
                },
            )
            try:
                dir_name = (
                    _checkpoint_dir_name(target_epoch)
                    if target_epoch is not None
                    else checkpoint_dir.name
                )
                restored_path = hf_restore_checkpoint(
                    job,
                    checkpoint_dir,
                    checkpoint_dir_name=dir_name,
                    repo_id=repo_id,
                    filename=job.hf_model_filename,
                )

                if target_epoch is not None:
                    job.available_checkpoint_epochs = sorted(
                        set(job.available_checkpoint_epochs + [target_epoch])
                    )
                if update_default_checkpoint:
                    job.checkpoint_path = restored_path
                    job.status = JobStatus.READY
                    job.progress = {
                        "stage": "ready",
                        "detail": "Model restored from Hugging Face and ready for inference",
                        "inference_url": f"/infer/{job.job_id}",
                    }
                    job.save()
                else:
                    job.status = prev_status
                    job.progress = prev_progress
                ops_log.end(restore_op, extra={"checkpoint_path": restored_path})
                logger.info(
                    "Restored checkpoint for job %s from HF repo %s to %s",
                    job.job_id,
                    repo_id,
                    restored_path,
                )
                return restored_path
            except Exception as e:
                job.status = prev_status
                job.progress = prev_progress
                if update_default_checkpoint:
                    job.save()
                ops_log.fail(restore_op, str(e))
                raise ValueError(f"Failed to restore checkpoint from Hugging Face: {e}")

    def _restore_checkpoint_from_s3(
        self,
        job: Job,
        checkpoint_epoch: Optional[int] = None,
    ) -> str:
        """Download and extract the model checkpoint from S3.

        Returns the local checkpoint path.
        Raises ValueError if restoration fails.
        """
        import tempfile

        from storage import storage

        latest_epoch = self._latest_checkpoint_epoch(job)
        target_epoch = (
            checkpoint_epoch if checkpoint_epoch is not None else latest_epoch
        )
        s3_key = None
        if target_epoch is not None:
            s3_key = job.checkpoint_s3_keys.get(str(target_epoch))
        if not s3_key:
            s3_key = job.s3_model_key

        if not s3_key:
            raise ValueError(f"Job {job.job_id} has no S3 model key")
        if not getattr(storage, "has_read_backend", storage.is_configured):
            raise ValueError("Storage not configured, cannot restore checkpoint")

        # Ensure output directory exists
        job_dir = self.jobs_dir / job.job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if target_epoch is not None:
            checkpoint_dir = _checkpoint_path_for_epoch(job.output_dir, target_epoch)
        else:
            checkpoint_dir = output_dir / f"checkpoint_{job.speaker_name}"
        checkpoint_path = str(checkpoint_dir.resolve())  # always absolute

        # 1. Quick check: is it already there?
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            logger.info(
                f"Checkpoint for job {job.job_id} already exists, skipping restore."
            )
            job.checkpoint_path = checkpoint_path
            return checkpoint_path

        # 2. Acquire a lock for this specific job to serialize restoration
        with self._lock:
            if job.job_id not in self._restore_locks:
                self._restore_locks[job.job_id] = threading.Lock()
            lock = self._restore_locks[job.job_id]

        with lock:
            # 3. Double-check: did another thread just finish it?
            if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                if checkpoint_epoch is None or (
                    latest_epoch is not None and target_epoch == latest_epoch
                ):
                    job.checkpoint_path = checkpoint_path
                return checkpoint_path

            prev_status = job.status
            prev_progress = dict(job.progress)
            update_default_checkpoint = checkpoint_epoch is None or (
                latest_epoch is not None and target_epoch == latest_epoch
            )
            if update_default_checkpoint:
                job.status = JobStatus.RESTORING
                job.progress = {
                    "stage": "restoring",
                    "detail": "Downloading model from S3...",
                }
                job.save()

            restore_op = ops_log.start(
                "checkpoint_restore",
                job_id=job.job_id,
                extra={
                    "s3_key": s3_key,
                    "checkpoint_epoch": target_epoch,
                },
            )
            try:
                # Download the zip to a temp file
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = tmp.name

                storage.download_file(s3_key, tmp_path)

                # Extract into the output directory
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                import zipfile

                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(checkpoint_dir)

                os.unlink(tmp_path)

                if target_epoch is not None:
                    job.available_checkpoint_epochs = sorted(
                        set(job.available_checkpoint_epochs + [target_epoch])
                    )
                if update_default_checkpoint:
                    job.checkpoint_path = (
                        checkpoint_path  # already absolute (resolved above)
                    )
                    job.status = JobStatus.READY
                    job.progress = {
                        "stage": "ready",
                        "detail": "Model restored from S3 and ready for inference",
                        "inference_url": f"/infer/{job.job_id}",
                    }
                    job.save()
                else:
                    job.status = prev_status
                    job.progress = prev_progress
                ops_log.end(restore_op, extra={"checkpoint_path": checkpoint_path})
                logger.info(
                    f"Restored checkpoint for job {job.job_id} from S3 to {checkpoint_path}"
                )
                return checkpoint_path
            except Exception as e:
                job.status = prev_status
                job.progress = prev_progress
                if update_default_checkpoint:
                    job.save()
                ops_log.fail(restore_op, str(e))
                raise ValueError(f"Failed to restore checkpoint from S3: {e}")
