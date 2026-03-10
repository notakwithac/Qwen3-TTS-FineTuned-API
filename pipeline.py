# coding=utf-8
"""Pipeline orchestrator — manages fine-tuning jobs from dataset to serving."""

import gc
import os
import shutil
import sys
import threading
import traceback
import logging
import uuid
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from inference_manager import InferenceManager
from ops_logger import ops_log

logger = logging.getLogger(__name__)

# Add finetuning dir to path so we can import prepare_data / sft_12hz
_finetuning_dir = str(Path(__file__).parent / "finetuning")
if _finetuning_dir not in sys.path:
    sys.path.insert(0, _finetuning_dir)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

class JobStatus:
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    LOADING = "loading"
    RESTORING = "restoring"  # Downloading checkpoint from S3
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
        self.progress: Dict[str, Any] = {}
        self.checkpoint_path: Optional[str] = None
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
        }
        if self.status == JobStatus.READY:
            d["inference_url"] = f"/infer/{self.job_id}"
            d["last_accessed_at"] = self.last_accessed_at
        return d

    def save(self):
        """Persist job state to disk."""
        if not self.job_dir:
            return
        job_file = Path(self.job_dir) / "job.json"
        try:
            with open(job_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save job {self.job_id}: {e}")

    @classmethod
    def load(cls, job_dir: str) -> Optional['Job']:
        """Load job state from disk."""
        job_file = Path(job_dir) / "job.json"
        if not job_file.exists():
            return None
        try:
            with open(job_file, "r") as f:
                data = json.load(f)
            
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
            job.progress = data.get("progress", {})
            job.checkpoint_path = data.get("checkpoint_path")
            job.error = data.get("error")
            job.created_at = data.get("created_at", job.created_at)
            job.finished_at = data.get("finished_at")
            if "last_accessed_at" in data:
                job.last_accessed_at = data["last_accessed_at"]
                
            return job
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
    ):
        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_flash_attn = use_flash_attn

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
                logger.error(f"Shutdown: Thread {thread.name} did not finish in time — upload may be incomplete.")
            else:
                logger.info(f"Shutdown: Thread {thread.name} completed.")

    def _s3_sync_worker(self, interval_seconds: float = 900.0):
        """Background thread: every 15 min, upload any READY job that is missing from S3."""
        from storage import storage
        logger.info("S3 sync monitor started (interval: 15 min).")
        while not self._stop_sync_event.wait(timeout=interval_seconds):
            if not storage.is_configured:
                continue
            try:
                with self._lock:
                    jobs_snapshot = list(self.jobs.values())
                
                for job in jobs_snapshot:
                    if self._stop_sync_event.is_set():
                        break
                    # Only target READY jobs with a local checkpoint but no S3 backup
                    if job.status != JobStatus.READY:
                        continue
                    if job.s3_model_key:
                        continue  # Already uploaded
                    if not job.checkpoint_path or not os.path.exists(str(job.checkpoint_path)):
                        continue  # No local checkpoint to upload
                    
                    logger.warning(f"S3 sync: Job {job.job_id} ({job.speaker_name}) has no S3 backup — uploading now.")
                    try:
                        s3_key = self._upload_model_to_s3(job, Path(str(job.checkpoint_path)))
                        job.s3_model_key = s3_key
                        self._upload_job_json_to_s3(job)
                        job.save()
                        logger.info(f"S3 sync: Job {job.job_id} uploaded successfully → {s3_key}")
                    except Exception as e:
                        logger.error(f"S3 sync: Failed to upload job {job.job_id}: {e}")
            except Exception as e:
                logger.error(f"S3 sync worker error: {e}")
        logger.info("S3 sync monitor stopped.")

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
    ) -> Job:
        """Create a new fine-tuning job from an uploaded dataset zip.

        The zip must contain:
          - train.jsonl
          - data/ directory with .wav files
        """
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        
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
        job = self.jobs.get(job_id)
        if job:
            logger.info(f"Job {job_id} found in memory")
            return job
            
        # Try to load from disk if not in memory
        job_dir = self.jobs_dir / job_id
        if job_dir.exists():
            job = Job.load(str(job_dir))
            if job:
                logger.info(f"Job {job_id} found on disk at {job_dir}")
                with self._lock:
                    self.jobs[job_id] = job
                return job

        # Try to restore from S3
        logger.info(f"Job {job_id} not found locally, checking S3...")
        job = self._restore_job_from_s3(job_id)
        if job:
            return job
                
        logger.warning(f"Job {job_id} not found anywhere (memory, disk, S3)")
        return None

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

    def _cleanup_disk_lru(self, threshold_gb: float = 40.0):
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

        logger.info(f"Disk usage ({current_size / 1024**3:.2f}GB) exceeds threshold ({threshold_gb}GB). Pruning oldest jobs...")

        # Sort completed/failed jobs by last_accessed_at
        candidates = []
        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.READY, JobStatus.FAILED, JobStatus.CANCELLED):
                candidates.append(job)

        # Oldest first
        candidates.sort(key=lambda x: x.last_accessed_at)

        from storage import storage as _storage

        for job in candidates:
            if current_size <= threshold_bytes:
                break
            
            job_dir = self.jobs_dir / job.job_id
            if not job_dir.exists():
                continue

            if job.s3_model_key:
                # S3-backed: only delete the heavy output/ and dataset/ dirs,
                # keep job.json so get_job() still works without S3 round-trip
                for subdir in ["output", "dataset"]:
                    subdir_path = job_dir / subdir
                    if subdir_path.exists():
                        size = self._get_dir_size(subdir_path)
                        try:
                            shutil.rmtree(subdir_path)
                            current_size -= size
                            logger.info(f"LRU: Deleted {subdir}/ for S3-backed job {job.job_id} (freed {size / 1024**2:.1f}MB)")
                        except Exception as e:
                            logger.error(f"LRU: Failed to delete {subdir}/ for {job.job_id}: {e}")
            elif _storage.is_configured:
                # S3 is configured but this job hasn't been uploaded yet — skip it.
                # It may still be uploading in the background, or the upload failed.
                logger.warning(f"LRU: Skipping job {job.job_id} — not yet uploaded to S3. Cannot safely prune.")
            else:
                # No S3 configured: delete the entire folder (last resort)
                size = self._get_dir_size(job_dir)
                try:
                    shutil.rmtree(job_dir)
                    current_size -= size
                    logger.info(f"LRU: Deleted job {job.job_id} (freed {size / 1024**2:.1f}MB)")
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
        job = self.jobs.get(job_id)
        if not job:
            return None
            
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            return None  # Only allow retrying if it actually failed or died.

        # Check if training already completed (late-stage failure)
        checkpoint_exists = job.checkpoint_path and os.path.exists(str(job.checkpoint_path))
        has_s3_backup = bool(job.s3_model_key)

        if checkpoint_exists or has_s3_backup:
            # Training succeeded — skip to Stage 3 (load for inference)
            job.status = JobStatus.LOADING
            job.error = None
            job._cancel_requested = False
            job.progress = {"stage": "loading", "detail": "Retrying: loading model for inference (training already completed)..."}
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
            cp = str(job.checkpoint_path) if job.checkpoint_path else None

            # Restore checkpoint from S3 if not available locally
            if not cp or not os.path.exists(cp):
                if job.s3_model_key:
                    cp = self._restore_checkpoint_from_s3(job)
                else:
                    raise ValueError("No checkpoint and no S3 backup")

            # Free GPU memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            job.status = JobStatus.LOADING
            job.progress = {"stage": "loading", "detail": "Loading fine-tuned model for inference..."}
            job.save()

            self.inference.load(cp, job.speaker_name)

            job.checkpoint_path = cp
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
        pipeline_op = ops_log.start("pipeline_total", job_id=job.job_id, extra={
            "speaker_name": job.speaker_name,
            "num_epochs": job.num_epochs,
            "attempt": attempt,
        })
        try:
            # Stage 1: Data preparation
            # Wait for our turn in the GPU training queue
            job.status = JobStatus.QUEUED
            job.progress = {"stage": "queued", "detail": "Waiting in queue for next available slot..."}
            
            self._training_queue.acquire()
            
            job.status = JobStatus.PREPARING
            job.progress = {"stage": "preparing", "detail": "Encoding audio to codec tokens..."}
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

            prep_op = ops_log.start("prepare_data", job_id=job.job_id)
            prepare_programmatic(
                input_jsonl=train_jsonl,
                output_jsonl=prepared_jsonl,
                device=self.device,
                batch_size=16,
                on_progress=on_prepare_progress,
            )
            ops_log.end(prep_op)

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                ops_log.fail(pipeline_op, "cancelled")
                self._training_queue.release()
                return

            # Free tokenizer GPU memory before training
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 2: Training
            job.status = JobStatus.TRAINING
            job.progress = {"stage": "training", "epoch": 0, "total_epochs": job.num_epochs}
            job.save()

            from sft_12hz import train_programmatic

            def on_train_progress(info: dict):
                job.progress = {
                    "stage": "training",
                    **info,
                }

            train_op = ops_log.start("training", job_id=job.job_id, extra={
                "num_epochs": job.num_epochs, "batch_size": job.batch_size, "lr": job.lr,
            })
            
            init_model = job.base_model_path if job.base_model_path and os.path.exists(job.base_model_path) else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            
            checkpoint_path = train_programmatic(
                config={
                    "init_model_path": init_model,
                    "output_model_path": job.output_dir,
                    "train_jsonl": prepared_jsonl,
                    "batch_size": job.batch_size,
                    "lr": job.lr,
                    "num_epochs": job.num_epochs,
                    "speaker_name": job.speaker_name,
                    "save_interval": job.num_epochs, # Only save the last/final one by default
                    "save_last": True,
                    "flash_attn": job.flash_attn,
                    "lr_scheduler": "cosine",
                    "warmup_ratio": 0.05,
                    "resume": True,
                },
                on_progress=on_train_progress,
            )
            ops_log.end(train_op, extra={"checkpoint_path": str(checkpoint_path)})
            
            # --- MODEL OFFLOAD: Zip and upload to S3 (Background Thread) ---
            from storage import storage
            if storage.is_configured:
                def run_s3_upload(j: Job, cp_path: str):
                    j.progress = {"stage": "offloading", "detail": "Zipping and uploading model to S3 in background..."}
                    j.save()
                    offload_op = ops_log.start("model_offload", job_id=j.job_id)
                    try:
                        s3_k = self._upload_model_to_s3(j, Path(cp_path))
                        j.s3_model_key = s3_k
                        # Upload job.json to S3 for easy access / restoration
                        self._upload_job_json_to_s3(j)
                        j.save()  # Persist s3_model_key locally
                        ops_log.end(offload_op, extra={"s3_key": s3_k})
                    except Exception as err:
                        ops_log.fail(offload_op, f"Model upload failed: {err}")
                
                upload_thread = threading.Thread(
                    target=run_s3_upload, 
                    args=(job, str(checkpoint_path)),
                    daemon=False,  # Non-daemon: survives Ctrl+C so upload completes
                    name=f"s3-upload-{job.job_id[:8]}"
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
                
                # 2. Clear intermediate checkpoints in output_dir
                # We keep ONLY the one pointed to by checkpoint_path
                for entry in os.listdir(job.output_dir):
                    entry_path = os.path.join(job.output_dir, entry)
                    if os.path.isdir(entry_path) and str(entry_path) != str(checkpoint_path):
                        shutil.rmtree(entry_path)
                
                ops_log.end(cleanup_op)
            except Exception as e:
                ops_log.fail(cleanup_op, f"Cleanup failed: {e}")

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                ops_log.fail(pipeline_op, "cancelled")
                self._training_queue.release()
                return

            job.checkpoint_path = checkpoint_path

            # Free training GPU memory before loading for inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: Load for inference
            job.status = JobStatus.LOADING
            job.progress = {"stage": "loading", "detail": "Loading fine-tuned model for inference..."}
            job.save()

            self.inference.load(str(checkpoint_path), job.speaker_name)

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
                for model_name in ["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"]:
                    self._clear_hf_cache(model_name)
                
                # Also clear if using a custom base_model_path from HuggingFace
                if job.base_model_path and "/" in job.base_model_path and not os.path.isabs(job.base_model_path):
                    self._clear_hf_cache(job.base_model_path)
                
                # Free GPU and retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Release queue before recursive call (recursive call will re-acquire)
                self._training_queue.release()
                
                job.status = JobStatus.QUEUED
                job.error = None
                job.progress = {"stage": "queued", "detail": "Auto-retrying after clearing corrupted model cache..."}
                job.save()
                
                # Re-run pipeline with incremented attempt
                # (recursive call manages its own queue acquire/release)
                self._run_pipeline(job, attempt + 1)
                return  # Skip the finally release — already released above
            
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.save()
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

        cp = checkpoint_path or (str(job.checkpoint_path) if job.checkpoint_path else None)
        spk = speaker_name or job.speaker_name
        
        # If checkpoint is missing locally, try to restore from S3
        if not cp or not os.path.exists(cp):
            if job.s3_model_key:
                cp = self._restore_checkpoint_from_s3(job)
            else:
                raise ValueError(f"Job {job_id} has no checkpoint and no S3 backup")

        return self.inference.generate(
            text=text,
            checkpoint_path=cp,
            speaker_name=spk,
            language=language,
            instruct=instruct,
        )

    def _upload_model_to_s3(self, job: Job, checkpoint_path: Path):
        """Zips and uploads the fine-tuned model to S3."""
        from storage import storage
        import tempfile
        
        # Structure: models/{book_id}/{speaker_name}/{speaker_name}_{job_id}.zip
        book_folder = job.book_id or "unsorted"
        speaker_name = job.speaker_name
        zip_filename = f"{speaker_name}_{job.job_id}.zip"
        s3_key = f"models/{book_folder}/{speaker_name}/{zip_filename}"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # We want to zip the CONTENTS of the checkpoint_path, not the folder itself
            # Base name for zip (without extension)
            base_zip_path = os.path.join(tmp_dir, f"{speaker_name}_{job.job_id}")
            
            # shutil.make_archive(base_name, format, root_dir)
            archive_path = shutil.make_archive(
                base_zip_path, 
                'zip', 
                root_dir=str(checkpoint_path)
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

    def _restore_job_from_s3(self, job_id: str) -> Optional[Job]:
        """Try to restore a job's metadata from S3 (lightweight — no model download)."""
        from storage import storage
        if not storage.is_configured:
            logger.warning(f"S3 restoration skipped for {job_id}: Storage not configured")
            return None

        s3_key = f"jobs/{job_id}/job.json"
        try:
            if not storage.object_exists(s3_key):
                logger.warning(f"Job metadata not found on S3 at {s3_key}")
                return None

            data_bytes = storage.download_bytes(s3_key)
            data = json.loads(data_bytes.decode("utf-8"))

            # Recreate local job directory with just job.json
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            with open(job_dir / "job.json", "w") as f:
                json.dump(data, f, indent=2)

            job = Job.load(str(job_dir))
            if job:
                with self._lock:
                    self.jobs[job_id] = job
                logger.info(f"Restored job {job_id} metadata from S3")
                return job
        except Exception as e:
            logger.error(f"Failed to restore job {job_id} from S3: {e}")
        return None

    def _restore_checkpoint_from_s3(self, job: Job) -> str:
        """Download and extract the model checkpoint from S3.
        
        Returns the local checkpoint path.
        Raises ValueError if restoration fails.
        """
        from storage import storage
        import tempfile

        if not job.s3_model_key:
            raise ValueError(f"Job {job.job_id} has no S3 model key")
        if not storage.is_configured:
            raise ValueError("Storage not configured, cannot restore checkpoint")

        # Ensure output directory exists
        job_dir = self.jobs_dir / job.job_id
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = output_dir / f"checkpoint_{job.speaker_name}"
        checkpoint_path = str(checkpoint_dir)

        # 1. Quick check: is it already there?
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            logger.info(f"Checkpoint for job {job.job_id} already exists, skipping restore.")
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
                job.checkpoint_path = checkpoint_path
                return checkpoint_path

            prev_status = job.status
            job.status = JobStatus.RESTORING
            job.progress = {"stage": "restoring", "detail": "Downloading model from S3..."}
            job.save()

            restore_op = ops_log.start("checkpoint_restore", job_id=job.job_id, extra={
                "s3_key": job.s3_model_key,
            })
            try:
                # Download the zip to a temp file
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = tmp.name
                
                storage.download_file(job.s3_model_key, tmp_path)

                # Extract into the output directory
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                import zipfile
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(checkpoint_dir)

                os.unlink(tmp_path)

                job.checkpoint_path = checkpoint_path
                job.status = JobStatus.READY
                job.progress = {
                    "stage": "ready",
                    "detail": "Model restored from S3 and ready for inference",
                    "inference_url": f"/infer/{job.job_id}",
                }
                job.save()
                ops_log.end(restore_op, extra={"checkpoint_path": checkpoint_path})
                logger.info(f"Restored checkpoint for job {job.job_id} from S3 to {checkpoint_path}")
                return checkpoint_path
            except Exception as e:
                job.status = prev_status
                job.save()
                ops_log.fail(restore_op, str(e))
                raise ValueError(f"Failed to restore checkpoint from S3: {e}")
