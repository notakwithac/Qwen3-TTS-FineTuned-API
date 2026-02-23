# coding=utf-8
"""Pipeline orchestrator — manages fine-tuning jobs from dataset to serving."""

import gc
import os
import shutil
import sys
import threading
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from inference_manager import InferenceManager
from ops_logger import ops_log

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
            "config": {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "flash_attn": self.flash_attn,
                "book_id": self.book_id,
                "chapter_id": self.chapter_id,
            },
        }
        if self.status == JobStatus.READY:
            d["inference_url"] = f"/infer/{self.job_id}"
            d["last_accessed_at"] = self.last_accessed_at
        return d

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
    ):
        self.base_dir = Path(base_dir)
        self.jobs_dir = self.base_dir / jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_flash_attn = use_flash_attn

        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.inference = InferenceManager(
            device=device,
            use_flash_attn=use_flash_attn,
            idle_timeout_seconds=idle_timeout_seconds,
            max_concurrency=max_concurrency,
            max_models=max_models,
        )

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
        )

        with self._lock:
            self.jobs[job_id] = job

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

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

    def _cleanup_disk_lru(self, threshold_gb: float = 20.0):
        """Delete oldest jobs if disk usage exceeds threshold."""
        if not self.jobs_dir.exists():
            return

        current_size = self._get_dir_size(self.jobs_dir)
        threshold_bytes = threshold_gb * (1024**3)

        if current_size <= threshold_bytes:
            return

        ops_log.info(f"Disk usage ({current_size / 1024**3:.2f}GB) exceeds threshold ({threshold_gb}GB). Pruning oldest jobs...")

        # Sort completed/failed jobs by last_accessed_at
        candidates = []
        for job_id, job in self.jobs.items():
            if job.status in (JobStatus.READY, JobStatus.FAILED, JobStatus.CANCELLED):
                candidates.append(job)

        # Oldest first
        candidates.sort(key=lambda x: x.last_accessed_at)

        for job in candidates:
            if current_size <= threshold_bytes:
                break
            
            job_dir = self.jobs_dir / job.job_id
            if job_dir.exists():
                size = self._get_dir_size(job_dir)
                try:
                    shutil.rmtree(job_dir)
                    current_size -= size
                    ops_log.info(f"LRU: Deleted job {job.job_id} (retrieved {size / 1024**2:.1f}MB)")
                except Exception as e:
                    ops_log.error(f"LRU: Failed to delete job {job.job_id}: {e}")

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

    # -- Pipeline execution -------------------------------------------------

    def start_job(self, job_id: str):
        """Start the fine-tuning pipeline in a background thread."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        thread = threading.Thread(target=self._run_pipeline, args=(job,), daemon=True)
        job._thread = thread
        thread.start()

    def _run_pipeline(self, job: Job):
        """Execute the 3-stage pipeline: prepare → train → serve."""
        pipeline_op = ops_log.start("pipeline_total", job_id=job.job_id, extra={
            "speaker_name": job.speaker_name,
            "num_epochs": job.num_epochs,
        })
        try:
            # Stage 1: Data preparation
            job.status = JobStatus.PREPARING
            job.progress = {"stage": "preparing", "detail": "Encoding audio to codec tokens..."}

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
                on_progress=on_prepare_progress,
            )
            ops_log.end(prep_op)

            if job._cancel_requested:
                job.status = JobStatus.CANCELLED
                ops_log.fail(pipeline_op, "cancelled")
                return

            # Free tokenizer GPU memory before training
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 2: Training
            job.status = JobStatus.TRAINING
            job.progress = {"stage": "training", "epoch": 0, "total_epochs": job.num_epochs}

            from sft_12hz import train_programmatic

            def on_train_progress(info: dict):
                job.progress = {
                    "stage": "training",
                    **info,
                }

            train_op = ops_log.start("training", job_id=job.job_id, extra={
                "num_epochs": job.num_epochs, "batch_size": job.batch_size, "lr": job.lr,
            })
            checkpoint_path = train_programmatic(
                config={
                    "init_model_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
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
            
            # --- DISK OPTIMIZATION: Model Offloading & Cleanup ---
            from storage import storage
            if storage.is_configured:
                job.progress = {"stage": "offloading", "detail": "Uploading model weights to S3..."}
                offload_op = ops_log.start("model_offload", job_id=job.job_id)
                try:
                    # Model key structure: models/{book_id or 'unsorted'}/{speaker_name}/{job_id}/
                    model_prefix = f"models/{job.book_id or 'unsorted'}/{job.speaker_name}/{job.job_id}"
                    
                    # Upload key model files
                    weight_file = os.path.join(checkpoint_path, "model.safetensors")
                    config_file = os.path.join(checkpoint_path, "config.json")
                    
                    if os.path.exists(weight_file):
                        storage.upload_file(weight_file, f"{model_prefix}/model.safetensors")
                    if os.path.exists(config_file):
                        storage.upload_file(config_file, f"{model_prefix}/config.json")
                    
                    ops_log.end(offload_op, extra={"s3_prefix": model_prefix})
                except Exception as e:
                    ops_log.fail(offload_op, f"Model offload failed: {e}")
                    # Don't fail the whole job if S3 upload fails, but log it
            
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
                return

            job.checkpoint_path = checkpoint_path

            # Free training GPU memory before loading for inference
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Stage 3: Load for inference
            job.status = JobStatus.LOADING
            job.progress = {"stage": "loading", "detail": "Loading fine-tuned model for inference..."}

            self.inference.load(str(checkpoint_path), job.speaker_name)

            job.status = JobStatus.READY
            job.finished_at = datetime.now(timezone.utc).isoformat()
            job.progress = {
                "stage": "ready",
                "detail": "Model loaded and ready for inference",
                "inference_url": f"/infer/{job.job_id}",
            }
            ops_log.end(pipeline_op)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            ops_log.fail(pipeline_op, str(e))

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
        if job.status != JobStatus.READY:
            raise ValueError(f"Job {job_id} is not ready (status: {job.status})")

        cp = checkpoint_path or (str(job.checkpoint_path) if job.checkpoint_path else None)
        spk = speaker_name or job.speaker_name
        
        if not cp:
            raise ValueError(f"Job {job_id} has no checkpoint")

        return self.inference.generate(
            text=text,
            checkpoint_path=cp,
            speaker_name=spk,
            language=language,
            instruct=instruct,
        )
