# coding=utf-8
"""Session-based inference manager — event-driven TTS with pre-loading,
model replication, per-character queues, and parallel workers.

Usage:
    1. POST /session/prepare  → pre-load models, create queues
    2. POST /session/{id}/submit/batch  → enqueue inference messages
    3. GET  /session/{id}/status  → poll progress & collect results
    4. DELETE /session/{id}  → teardown, release GPU replicas
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch

from ops_logger import ops_log

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (overridden by env in api_server.py)
# ---------------------------------------------------------------------------
DEFAULT_REPLICA_THRESHOLD = 500   # Lines before adding a replica
DEFAULT_MAX_REPLICAS = 4          # Max replicas of one model
DEFAULT_SESSION_TIMEOUT = 3600    # Auto-cleanup after 1h idle
MODEL_VRAM_GB = 5.5               # Measured per-model VRAM (bf16 weights + compiled overhead)
DEFAULT_BATCH_TEXT_BUDGET = int(os.environ.get("SESSION_BATCH_MAX_CHARS", "4000"))
DEFAULT_BATCH_PADDED_TEXT_BUDGET = int(os.environ.get("SESSION_BATCH_MAX_PADDED_CHARS", "0"))
DEFAULT_CUSTOM_SESSION_MAX_NEW_TOKENS = int(
    os.environ.get("CUSTOM_VOICE_SESSION_MAX_NEW_TOKENS", "1536")
)
DEFAULT_CUSTOM_SESSION_MIN_NEW_TOKENS = int(
    os.environ.get("CUSTOM_VOICE_SESSION_MIN_NEW_TOKENS", "512")
)
DEFAULT_CUSTOM_SESSION_MAX_NEW_TOKENS_PER_CHAR = int(
    os.environ.get("CUSTOM_VOICE_SESSION_MAX_NEW_TOKENS_PER_CHAR", "4")
)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class SessionStatus(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DuplicateActiveSessionError(ValueError):
    """Raised when the same chapter/job workload is already active."""

    def __init__(self, message: str, active_session_id: str):
        super().__init__(message)
        self.active_session_id = active_session_id


class TrainingConflictError(RuntimeError):
    """Raised when exclusive GPU training blocks session preparation."""


@dataclass
class CharacterPlan:
    """Plan for a single character within a session."""
    job_id: str
    character_name: str
    checkpoint_path: Optional[str] = None
    line_count: int = 0
    avg_word_count: int = 20
    replicas: int = 1
    character_id: Optional[str] = None
    replica_keys: list = field(default_factory=list)  # Cache keys in InferenceManager


@dataclass
class CharacterProgress:
    """Live tracking of a character's inference progress."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    results: list = field(default_factory=list)  # List of {"s3_key": ..., "presigned_url": ...}


@dataclass
class InferenceMessage:
    """A single inference work item."""
    session_id: str
    job_id: str
    character_name: str
    text: str
    language: str = "English"
    instruct: str = ""
    s3_filename: str = ""
    book_id: str = ""
    chapter_id: str = ""
    character_id: str = ""
    speaker_name: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "job_id": self.job_id,
            "character_name": self.character_name,
            "text": self.text,
            "language": self.language,
            "instruct": self.instruct,
            "s3_filename": self.s3_filename,
            "book_id": self.book_id,
            "chapter_id": self.chapter_id,
            "character_id": self.character_id,
            "speaker_name": self.speaker_name,
        }


# ---------------------------------------------------------------------------
# Character Worker
# ---------------------------------------------------------------------------

class CharacterWorker:
    """Async worker that pulls from a per-character queue, batches items,
    and runs inference on an assigned model replica.

    Each worker is bound to one model replica (cache_key in InferenceManager).
    Multiple workers can share the same character queue for parallelism.
    """

    def __init__(
        self,
        worker_id: str,
        queue: asyncio.Queue,
        inference_manager,  # InferenceManager instance
        worker_semaphore: asyncio.Semaphore,
        cache_key: str,     # The model cache key to use
        speaker_name: str,
        batch_size: int = 32,
        batch_timeout_ms: int = 100,
        batch_text_budget: int = DEFAULT_BATCH_TEXT_BUDGET,
        batch_padded_text_budget: int = DEFAULT_BATCH_PADDED_TEXT_BUDGET,
        initial_batch_size: int = 1,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        min_new_tokens: int = DEFAULT_CUSTOM_SESSION_MIN_NEW_TOKENS,
        max_new_tokens: int = DEFAULT_CUSTOM_SESSION_MAX_NEW_TOKENS,
        max_new_tokens_per_char: int = DEFAULT_CUSTOM_SESSION_MAX_NEW_TOKENS_PER_CHAR,
        storage=None,
        progress: Optional[CharacterProgress] = None,
    ):
        self.worker_id = worker_id
        self.queue = queue
        self.inference = inference_manager
        self.worker_semaphore = worker_semaphore
        self.cache_key = cache_key
        self.speaker_name = speaker_name
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.batch_text_budget = max(0, int(batch_text_budget or 0))
        self.batch_padded_text_budget = max(0, int(batch_padded_text_budget or 0))
        self.initial_batch_size = max(1, min(int(initial_batch_size or batch_size), batch_size))
        self.generation_kwargs = dict(generation_kwargs or {})
        self.min_new_tokens = max(0, int(min_new_tokens or 0))
        self.max_new_tokens = max(0, int(max_new_tokens or 0))
        self.max_new_tokens_per_char = max(0, int(max_new_tokens_per_char or 0))
        self.storage = storage
        self.progress = progress or CharacterProgress()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._in_batch = False
        self._first_batch_pending = True

    @staticmethod
    def _message_text_cost(msg: InferenceMessage) -> int:
        return max(
            1,
            len((msg.text or "").strip()) + len((msg.instruct or "").strip()),
        )

    @staticmethod
    def _estimate_padded_batch_cost(batch_size: int, batch_max_text_cost: int) -> int:
        if batch_size <= 0 or batch_max_text_cost <= 0:
            return 0
        return batch_size * batch_max_text_cost

    def _should_defer_message(
        self,
        batch: List[InferenceMessage],
        batch_text_cost: int,
        batch_max_text_cost: int,
        msg: InferenceMessage,
        batch_size_limit: int,
    ) -> bool:
        if len(batch) >= batch_size_limit:
            return True
        msg_cost = self._message_text_cost(msg)
        # Always allow the first item in a batch, even if it exceeds the budget.
        if not batch:
            return False
        if self.batch_text_budget > 0 and (batch_text_cost + msg_cost) > self.batch_text_budget:
            return True
        if self.batch_padded_text_budget > 0:
            next_max_text_cost = max(batch_max_text_cost, msg_cost)
            next_padded_cost = self._estimate_padded_batch_cost(len(batch) + 1, next_max_text_cost)
            if next_padded_cost > self.batch_padded_text_budget:
                return True
        return False

    def _current_batch_size_limit(self) -> int:
        if self._first_batch_pending:
            return self.initial_batch_size
        return self.batch_size

    @staticmethod
    def _message_output_text_chars(msg: InferenceMessage) -> int:
        return max(1, len((msg.text or "").strip()))

    def _build_generation_kwargs(self, batch: List[InferenceMessage]) -> Dict[str, Any]:
        kwargs = dict(self.generation_kwargs)
        if batch and (self.max_new_tokens > 0 or self.max_new_tokens_per_char > 0):
            max_text_chars = max(self._message_output_text_chars(msg) for msg in batch)
            derived_limit = self.min_new_tokens
            if self.max_new_tokens_per_char > 0:
                derived_limit = max(
                    derived_limit,
                    max_text_chars * self.max_new_tokens_per_char,
                )
            if self.max_new_tokens > 0:
                derived_limit = min(derived_limit, self.max_new_tokens)
            if derived_limit > 0:
                kwargs["max_new_tokens"] = derived_limit
        return kwargs

    def start(self):
        """Start the worker coroutine."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Worker {self.worker_id} started (cache_key={self.cache_key})")

    async def stop(self):
        """Gracefully stop the worker."""
        self._running = False
        
        # Empty the queue so we don't process any more items
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
        if self._task and not self._task.done():
            # Only cancel if it's polling the queue and NOT in the middle of a GPU generation.
            # When a batch is active we must wait for inference + upload to complete; otherwise
            # an eager session teardown can race the caller and make "submit then delete" lose audio.
            if getattr(self, "_in_batch", False):
                logger.info(
                    "Worker %s waiting for active batch to finish before stopping",
                    self.worker_id,
                )
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            else:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        logger.info(f"Worker {self.worker_id} stopped")

    async def _run(self):
        """Main worker loop: pull from queue → batch → infer → upload."""
        loop = asyncio.get_running_loop()
        carryover_msg: Optional[InferenceMessage] = None

        while self._running:
            batch: List[InferenceMessage] = []
            batch_text_cost = 0
            batch_max_text_cost = 0

            try:
                # 1. Block until at least one item is available
                if carryover_msg is not None:
                    msg = carryover_msg
                    carryover_msg = None
                else:
                    msg = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                batch.append(msg)
                batch_text_cost = self._message_text_cost(msg)
                batch_max_text_cost = batch_text_cost
            except asyncio.TimeoutError:
                continue  # No work, loop back
            except asyncio.CancelledError:
                return

            # 2. Acquire a worker slot. This limits actively processing workers
            #    to max_models, preventing VRAM cache thrashing.
            async with self.worker_semaphore:
                # 3. Drain the queue as much as possible while holding the slot
                while self._running:
                    batch_size_limit = self._current_batch_size_limit()
                    # Fill the batch up to batch_size
                    deadline = time.monotonic() + self.batch_timeout
                    while len(batch) < batch_size_limit:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            msg = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                            if self._should_defer_message(
                                batch,
                                batch_text_cost,
                                batch_max_text_cost,
                                msg,
                                batch_size_limit,
                            ):
                                carryover_msg = msg
                                break
                            batch.append(msg)
                            msg_cost = self._message_text_cost(msg)
                            batch_text_cost += msg_cost
                            batch_max_text_cost = max(batch_max_text_cost, msg_cost)
                        except asyncio.TimeoutError:
                            break
                        except asyncio.CancelledError:
                            break

                    if batch:
                        await self._process_batch(batch, loop)
                        batch.clear()
                        batch_text_cost = 0
                        batch_max_text_cost = 0

                    if carryover_msg is not None:
                        batch.append(carryover_msg)
                        batch_text_cost = self._message_text_cost(carryover_msg)
                        batch_max_text_cost = batch_text_cost
                        carryover_msg = None
                        continue

                    # Check if more items are queued
                    # Use a tiny timeout to briefly wait in case orchestrator is slightly behind,
                    # but yield the slot quickly if genuinely empty.
                    try:
                        msg = await asyncio.wait_for(self.queue.get(), timeout=0.25)
                        batch.append(msg)
                        batch_text_cost = self._message_text_cost(msg)
                        batch_max_text_cost = batch_text_cost
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        break  # Queue is empty, yield the semaphore slot

    async def _process_batch(self, batch: List[InferenceMessage], loop: asyncio.AbstractEventLoop):
        """Run inference on a batch and upload results."""
        self._in_batch = True
        try:
            # Sort by text length to minimize padding waste in left-padded batching
            batch.sort(key=lambda m: len(m.text))
            texts = [m.text for m in batch]
            languages = [m.language for m in batch]
            instructs = [m.instruct for m in batch]
            generation_kwargs = self._build_generation_kwargs(batch)

            op = ops_log.start("session_worker_batch", extra={
                "worker_id": self.worker_id,
                "batch_size": len(batch),
                "batch_text_chars": sum(self._message_text_cost(msg) for msg in batch),
                "batch_max_text_chars": max(self._message_text_cost(msg) for msg in batch),
                "batch_padded_text_chars": self._estimate_padded_batch_cost(
                    len(batch),
                    max(self._message_text_cost(msg) for msg in batch),
                ),
                "cache_key": self.cache_key,
                "first_batch_mode": self._first_batch_pending,
                "max_new_tokens": generation_kwargs.get("max_new_tokens"),
                "do_sample": generation_kwargs.get("do_sample"),
            })

            try:
                # Run blocking inference in executor
                wav_bytes_list, sr = await loop.run_in_executor(
                    None,
                    lambda: self.inference.generate_batch(
                        texts=texts,
                        checkpoint_path=self.cache_key,
                        speaker_name=self.speaker_name,
                        languages=languages,
                        instructs=instructs,
                        **generation_kwargs,
                    )
                )

                # Upload results to S3 (parallel)
                upload_tasks = []
                for i, msg in enumerate(batch):
                    if self.storage and self.storage.is_configured and msg.s3_filename:
                        s3_prefix = (
                            f"audio/segments/{msg.book_id}/{msg.chapter_id}"
                            if msg.book_id and msg.chapter_id
                            else f"audio/{msg.job_id}"
                        )
                        upload_tasks.append(
                            self._upload_single(loop, wav_bytes_list[i], msg, s3_prefix, sr)
                        )
                    else:
                        # No S3 upload — just record completion
                        self.progress.completed += 1

                if upload_tasks:
                    for upload_task in asyncio.as_completed(upload_tasks):
                        await upload_task

                ops_log.end(op, extra={"completed": len(batch)})

            except Exception as e:
                logger.error(f"Worker {self.worker_id} batch failed: {e}")
                ops_log.fail(op, str(e))
                self.progress.failed += len(batch)
        finally:
            self._first_batch_pending = False
            self._in_batch = False

    async def _upload_single(
        self, loop: asyncio.AbstractEventLoop, wav_bytes: bytes,
        msg: InferenceMessage, s3_prefix: str, sr: int,
    ):
        """Upload a single WAV to S3 and update progress."""
        from functools import partial
        try:
            # Determine if session prefix is needed
            is_segment = bool(msg.book_id and msg.chapter_id)
            
            # If no filename provided, generate a default one
            filename = msg.s3_filename
            if not filename:
                ts = int(time.time())
                filename = f"audio_{ts}.wav"
            
            # Prefix with session_id for non-segment uploads
            final_filename = f"{msg.session_id}_{filename}" if not is_segment else filename

            s3_url = await loop.run_in_executor(
                None,
                partial(
                    self.storage.upload_wav,
                    wav_bytes, msg.job_id,
                    filename=final_filename,
                    prefix=s3_prefix,
                    model_id=msg.job_id,
                )
            )
            s3_key = f"{s3_prefix}/{final_filename}"
            presigned_url = self.storage.get_presigned_url(s3_key, expires_in=86400)
            self.progress.completed += 1
            self.progress.results.append({
                "s3_url": s3_url,
                "presigned_url": presigned_url,
                "s3_key": s3_key,
                "sample_rate": sr,
                "text": msg.text,
                "job_id": msg.job_id,
            })
        except Exception as e:
            logger.error(f"S3 upload failed for {msg.s3_filename}: {e}")
            self.progress.failed += 1


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """Represents one inference session — a set of characters with pre-loaded
    models, per-character queues, and worker pools."""

    def __init__(
        self,
        session_id: str,
        book_id: str = "",
        chapter_id: str = "",
        requested_job_ids: Optional[List[str]] = None,
    ):
        self.session_id = session_id
        self.book_id = book_id
        self.chapter_id = chapter_id
        self.requested_job_ids = tuple(sorted(requested_job_ids or []))
        self.status = SessionStatus.PREPARING
        self.created_at = time.time()
        self.last_active = time.time()

        # Character plans and tracking
        self.character_plans: Dict[str, CharacterPlan] = {}   # keyed by job_id
        self.character_progress: Dict[str, CharacterProgress] = {}  # keyed by job_id
        self.character_queues: Dict[str, asyncio.Queue] = {}  # keyed by job_id
        self.workers: List[CharacterWorker] = []

        # Aggregated stats
        self.total_lines = 0
        self.error: Optional[str] = None
        self.original_max_models: Optional[int] = None  # Restored on teardown

    def touch(self):
        self.last_active = time.time()

    @property
    def queued_items(self) -> int:
        return sum(q.qsize() for q in self.character_queues.values())

    @property
    def completed_lines(self) -> int:
        return sum(p.completed for p in self.character_progress.values())

    @property
    def failed_lines(self) -> int:
        return sum(p.failed for p in self.character_progress.values())

    @property
    def progress_pct(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return round((self.completed_lines / self.total_lines) * 100, 1)

    @property
    def is_complete(self) -> bool:
        return (self.completed_lines + self.failed_lines) >= self.total_lines

    def to_dict(self) -> dict:
        characters = {}
        for job_id, plan in self.character_plans.items():
            progress = self.character_progress.get(job_id, CharacterProgress())
            characters[plan.character_name] = {
                "job_id": job_id,
                "total": progress.total,
                "completed": progress.completed,
                "failed": progress.failed,
                "replicas": plan.replicas,
            }

        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "book_id": self.book_id,
            "chapter_id": self.chapter_id,
            "total_lines": self.total_lines,
            "completed_lines": self.completed_lines,
            "failed_lines": self.failed_lines,
            "progress_pct": self.progress_pct,
            "characters": characters,
            "error": self.error,
            "created_at": self.created_at,
            "requested_job_ids": list(self.requested_job_ids),
        }

    def get_all_results(self) -> list:
        """Return all completed results across all characters."""
        results = []
        for progress in self.character_progress.values():
            results.extend(progress.results)
        return results


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """Manages session lifecycle: prepare → process → teardown.

    Works with an InferenceManager to load models and an optional
    storage backend for S3 uploads.
    """

    def __init__(
        self,
        inference_manager,
        pipeline,
        storage=None,
        replica_threshold: int = DEFAULT_REPLICA_THRESHOLD,
        max_replicas: int = DEFAULT_MAX_REPLICAS,
        session_timeout: int = DEFAULT_SESSION_TIMEOUT,
        batch_size: int = 32,
        batch_text_budget: int = DEFAULT_BATCH_TEXT_BUDGET,
        batch_padded_text_budget: int = DEFAULT_BATCH_PADDED_TEXT_BUDGET,
        custom_generation_kwargs: Optional[Dict[str, Any]] = None,
        custom_min_new_tokens: int = DEFAULT_CUSTOM_SESSION_MIN_NEW_TOKENS,
        custom_max_new_tokens: int = DEFAULT_CUSTOM_SESSION_MAX_NEW_TOKENS,
        custom_max_new_tokens_per_char: int = DEFAULT_CUSTOM_SESSION_MAX_NEW_TOKENS_PER_CHAR,
    ):
        self.inference = inference_manager
        self.pipeline = pipeline
        self.storage = storage
        self.replica_threshold = replica_threshold
        self.max_replicas = max_replicas
        self.session_timeout = session_timeout
        self.batch_size = batch_size
        self.batch_text_budget = max(0, int(batch_text_budget or 0))
        self.batch_padded_text_budget = max(0, int(batch_padded_text_budget or 0))
        self.custom_generation_kwargs = dict(custom_generation_kwargs or {})
        self.custom_min_new_tokens = max(0, int(custom_min_new_tokens or 0))
        self.custom_max_new_tokens = max(0, int(custom_max_new_tokens or 0))
        self.custom_max_new_tokens_per_char = max(0, int(custom_max_new_tokens_per_char or 0))
        self.sessions: Dict[str, Session] = {}

        # Limit active workers to max_models to prevent cache thrashing
        # Allows a worker to hold its model in VRAM while it drains its queue
        self.worker_semaphore = asyncio.Semaphore(inference_manager.max_models)

        # Start cleanup timer
        self._cleanup_task: Optional[asyncio.Task] = None

    def start_cleanup_loop(self):
        """Start the background session cleanup loop (call once from lifespan)."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired = [
                sid for sid, s in self.sessions.items()
                if (now - s.last_active) > self.session_timeout
                and s.status not in (SessionStatus.PREPARING, SessionStatus.PROCESSING, SessionStatus.CANCELLED)
            ]
            for sid in expired:
                ops_log.log_event("session_auto_cleanup", extra={"session_id": sid})
                logger.info(f"Auto-cleaning expired session {sid}")
                await self.teardown_session(sid)

    # -- Replica Calculation --------------------------------------------------

    def _calculate_replicas(
        self,
        characters: list,
        available_vram_gb: float,
        additional_replica_slots: int = 0,
    ) -> Dict[str, int]:
        """Calculate how many replicas each character model needs.

        Args:
            characters: list of dicts with job_id, line_count
            available_vram_gb: VRAM budget for this session
        """
        # First pass: everyone gets 1 replica
        replicas = {}
        unique_jobs = set()
        for char in characters:
            job_id = char["job_id"]
            if job_id not in unique_jobs:
                replicas[job_id] = 1
                unique_jobs.add(job_id)

        # Calculate base VRAM needed (1 per unique model)
        base_vram = len(unique_jobs) * MODEL_VRAM_GB
        remaining_vram = available_vram_gb - base_vram

        if remaining_vram <= MODEL_VRAM_GB or additional_replica_slots <= 0:
            # No room for replicas
            return replicas

        # Sort characters by line count (descending) to prioritize high-traffic
        sorted_chars = sorted(characters, key=lambda c: c.get("line_count", 0), reverse=True)

        # Second pass: add replicas to high-traffic characters
        remaining_slots = max(0, int(additional_replica_slots))
        for char in sorted_chars:
            job_id = char["job_id"]
            line_count = char.get("line_count", 0)

            if line_count < self.replica_threshold:
                continue

            # How many replicas would be ideal?
            ideal = min(
                line_count // self.replica_threshold,
                self.max_replicas,
            )
            # How many can we actually fit?
            additional = ideal - replicas[job_id]  # How many more beyond current
            can_fit = int(remaining_vram // MODEL_VRAM_GB)
            to_add = min(additional, can_fit, remaining_slots)

            if to_add > 0:
                replicas[job_id] += to_add
                remaining_vram -= to_add * MODEL_VRAM_GB
                remaining_slots -= to_add

            if remaining_vram < MODEL_VRAM_GB or remaining_slots <= 0:
                break

        return replicas

    def _get_available_vram(self) -> float:
        """Get available VRAM in GB."""
        if not torch.cuda.is_available():
            return 40.0  # Default assumption

        mem_get_info = getattr(torch.cuda, "mem_get_info", None)
        if mem_get_info is not None:
            try:
                free_bytes, _ = mem_get_info(0)
            except TypeError:
                free_bytes, _ = mem_get_info()
            except Exception:
                free_bytes = None
            if free_bytes is not None:
                free_gb = free_bytes / 1e9
                return max(0, free_gb - 2.0)

        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        # Leave 2GB headroom for activations/KV-cache during inference
        return max(0, total - reserved - 2.0)

    @staticmethod
    def _build_replica_cache_key(checkpoint_path: str, replica_index: int) -> str:
        if replica_index <= 0:
            return checkpoint_path
        return f"{checkpoint_path}::replica-{replica_index}"

    def _shared_headroom_buffer_gb(self) -> float:
        stats = getattr(self.inference, "stats", {})
        if isinstance(stats, dict):
            try:
                return max(0.0, float(stats.get("shared_model_min_headroom_gb", 0.0) or 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _is_active_status(status: SessionStatus) -> bool:
        return status in (
            SessionStatus.PREPARING,
            SessionStatus.READY,
            SessionStatus.PROCESSING,
        )

    def _find_duplicate_active_session(
        self,
        *,
        session_id: str,
        book_id: str,
        chapter_id: str,
        requested_job_ids: tuple[str, ...],
    ) -> Optional[Session]:
        if not book_id or not chapter_id or not requested_job_ids:
            return None

        for existing in self.sessions.values():
            if existing.session_id == session_id:
                continue
            if not self._is_active_status(existing.status):
                continue
            if existing.book_id != book_id or existing.chapter_id != chapter_id:
                continue
            if existing.requested_job_ids != requested_job_ids:
                continue
            return existing
        return None

    # -- Session Lifecycle ----------------------------------------------------

    async def prepare_session(
        self,
        session_id: str,
        characters: list,
        book_id: str = "",
        chapter_id: str = "",
    ) -> Session:
        """Prepare a session: calculate replicas, pre-load models, create queues.

        Args:
            session_id: Unique session identifier
            characters: list of dicts with job_id, character_name, line_count, avg_word_count
            book_id: Optional book ID for S3 paths
            chapter_id: Optional chapter ID for S3 paths

        Returns: Session object with model plan
        """
        requested_job_ids = tuple(sorted({c["job_id"] for c in characters}))
        duplicate = self._find_duplicate_active_session(
            session_id=session_id,
            book_id=book_id,
            chapter_id=chapter_id,
            requested_job_ids=requested_job_ids,
        )
        if duplicate is not None:
            raise DuplicateActiveSessionError(
                (
                    "A matching session is already active for this chapter workload "
                    f"({duplicate.session_id})."
                ),
                active_session_id=duplicate.session_id,
            )

        if session_id in self.sessions:
            existing = self.sessions[session_id]
            if existing.status in (SessionStatus.PREPARING, SessionStatus.PROCESSING):
                raise ValueError(f"Session {session_id} is already active")
            # Teardown old session first
            await self.teardown_session(session_id)

        session = Session(
            session_id=session_id,
            book_id=book_id,
            chapter_id=chapter_id,
            requested_job_ids=list(requested_job_ids),
        )
        self.sessions[session_id] = session

        op = ops_log.start("session_prepare", extra={
            "session_id": session_id,
            "character_count": len(characters),
            "total_lines": sum(c.get("line_count", 0) for c in characters),
        })

        try:
            loop = asyncio.get_running_loop()

            # 1. Resolve unique jobs in parallel
            unique_jobs = {}
            for c in characters:
                jid = c["job_id"]
                if jid not in unique_jobs:
                    unique_jobs[jid] = c

            async def _resolve_job(char_dict):
                job_id = char_dict["job_id"]
                def _resolve_job_sync():
                    self.pipeline.touch_job(job_id)
                    job = self.pipeline.get_job(job_id)
                    if not job:
                        raise ValueError(f"Job {job_id} not found")

                    checkpoint_path = str(job.checkpoint_path) if job.checkpoint_path else None
                    if not checkpoint_path or not os.path.exists(checkpoint_path):
                        checkpoint_path, _ = self.pipeline.resolve_checkpoint_path(job)

                    self.pipeline.touch_job(job_id)
                    return job_id, checkpoint_path, job.character_id, job.speaker_name

                return await loop.run_in_executor(None, _resolve_job_sync)

            # Resolve all unique jobs
            results = await asyncio.gather(*[_resolve_job(c) for c in unique_jobs.values()])
            job_info = {r[0]: (r[1], r[2], r[3]) for r in results}

            headroom_buffer_gb = self._shared_headroom_buffer_gb()
            available_vram_gb = max(0.0, self._get_available_vram() - headroom_buffer_gb)
            loaded_count = int(getattr(self.inference, "loaded_count", 0) or 0)
            max_models = int(getattr(self.inference, "max_models", 1) or 1)
            additional_replica_slots = max(0, max_models - loaded_count - len(job_info))
            replica_counts = self._calculate_replicas(
                list(unique_jobs.values()),
                available_vram_gb=available_vram_gb,
                additional_replica_slots=additional_replica_slots,
            )

            # 2. Map all characters to their resolved plans
            for char_dict in characters:
                job_id = char_dict["job_id"]
                checkpoint_path, character_id, speaker_name = job_info[job_id]
                
                plan = CharacterPlan(
                    job_id=job_id,
                    character_name=speaker_name,
                    checkpoint_path=checkpoint_path,
                    line_count=char_dict.get("line_count", 0),
                    avg_word_count=char_dict.get("avg_word_count", 20),
                    character_id=character_id,
                )
                plan.replicas = max(1, replica_counts.get(job_id, 1))
                plan.replica_keys = [
                    self._build_replica_cache_key(checkpoint_path, replica_index)
                    for replica_index in range(plan.replicas)
                ]
                
                # Note: If multiple character names share a job_id, we keep both in the session
                # so the submission logic can find them, but they'll share a queue.
                session.character_plans[plan.job_id] = plan
                session.total_lines += plan.line_count

            # 3. Create per-job queues and progress trackers
            # If multiple character names share the same job_id, they share ONE queue.
            for job_id in job_info:
                plan = session.character_plans[job_id]
                session.character_queues[job_id] = asyncio.Queue()
                session.character_progress[job_id] = CharacterProgress(
                    total=sum(c.get("line_count", 0) for c in characters if c["job_id"] == job_id)
                )

            # 4. Pre-load and pin planned replicas before any work is queued.
            if self.inference.is_training_active_or_requested():
                raise TrainingConflictError(
                    "GPU training is active; session preparation must wait for training to finish."
                )

            preload_tasks = []
            for plan in session.character_plans.values():
                for cache_key in plan.replica_keys:
                    preload_tasks.append(
                        loop.run_in_executor(
                            None,
                            self.inference.load_for_session,
                            cache_key,
                            plan.checkpoint_path,
                            plan.character_name,
                            session_id,
                        )
                    )
            if preload_tasks:
                await asyncio.gather(*preload_tasks)

            # 5. Start workers (one per replica)
            for job_id, plan in session.character_plans.items():
                queue = session.character_queues[job_id]
                progress = session.character_progress[job_id]

                for replica_index, cache_key in enumerate(plan.replica_keys):
                    worker = CharacterWorker(
                        worker_id=f"{session_id}/{plan.character_name}/w{replica_index}",
                        queue=queue,
                        inference_manager=self.inference,
                        worker_semaphore=self.worker_semaphore,
                        cache_key=cache_key,
                        speaker_name=plan.character_name,
                        batch_size=self.batch_size,
                        batch_timeout_ms=100,
                        batch_text_budget=self.batch_text_budget,
                        batch_padded_text_budget=self.batch_padded_text_budget,
                        initial_batch_size=1,
                        generation_kwargs=self.custom_generation_kwargs,
                        min_new_tokens=self.custom_min_new_tokens,
                        max_new_tokens=self.custom_max_new_tokens,
                        max_new_tokens_per_char=self.custom_max_new_tokens_per_char,
                        storage=self.storage,
                        progress=progress,
                    )
                    worker.start()
                    session.workers.append(worker)

            session.status = SessionStatus.READY
            ops_log.end(op, extra={
                "characters": len(session.character_plans),
                "total_lines": session.total_lines,
                "max_concurrent_models": self.inference.max_models,
                "workers_started": len(session.workers),
                "replicas_planned": sum(plan.replicas for plan in session.character_plans.values()),
            })

            logger.info(
                f"Session {session_id} prepared: "
                f"{len(session.character_plans)} characters, "
                f"{session.total_lines} total lines, "
                f"max {self.inference.max_models} concurrent models"
            )
            return session

        except Exception as e:
            session.status = SessionStatus.FAILED
            session.error = str(e)
            ops_log.fail(op, str(e))
            logger.error(f"Session prepare failed: {e}")
            raise

    async def submit_messages(
        self, session_id: str, messages: List[dict],
    ) -> int:
        """Submit inference messages to the appropriate character queues.

        Args:
            session_id: Session ID
            messages: list of message dicts

        Returns: number of messages enqueued
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        if session.status == SessionStatus.READY:
            session.status = SessionStatus.PROCESSING

        session.touch()
        enqueued = 0

        for msg_dict in messages:
            job_id = msg_dict.get("job_id")
            if job_id not in session.character_queues:
                logger.warning(
                    f"Unknown job_id {job_id} in session {session_id}, skipping"
                )
                continue

            msg = InferenceMessage(
                session_id=session_id,
                job_id=job_id,
                character_name=msg_dict.get("character_name", ""),
                text=msg_dict.get("text", ""),
                language=msg_dict.get("language", "English"),
                instruct=msg_dict.get("instruct", ""),
                s3_filename=msg_dict.get("s3_filename", ""),
                book_id=msg_dict.get("book_id", session.book_id),
                chapter_id=msg_dict.get("chapter_id", session.chapter_id),
                character_id=session.character_plans[job_id].character_id or "",
                speaker_name=session.character_plans[job_id].character_name,
            )
            await session.character_queues[job_id].put(msg)
            enqueued += 1

        # Update totals if messages push past initial plan
        new_total = sum(p.total for p in session.character_progress.values())
        if enqueued > 0 and session.total_lines < new_total:
            session.total_lines = new_total

        logger.info(f"Session {session_id}: enqueued {enqueued} messages")
        return enqueued

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session:
            # Auto-complete if all work is done
            if (
                session.status == SessionStatus.PROCESSING
                and session.is_complete
                and all(q.empty() for q in session.character_queues.values())
            ):
                session.status = SessionStatus.COMPLETED
        return session

    async def teardown_session(self, session_id: str) -> bool:
        """Teardown a session: stop workers, release replicas, remove queues."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        op = ops_log.start("session_teardown", extra={"session_id": session_id})

        # 1. Stop all workers
        for worker in session.workers:
            await worker.stop()

        # 2. Unpin models from session protection (LRU can now evict them)
        for plan in session.character_plans.values():
            for cache_key in plan.replica_keys:
                self.inference.unpin_session(cache_key, session_id)
                if cache_key != plan.checkpoint_path:
                    self.inference.unload_specific(cache_key)

        session.status = SessionStatus.CANCELLED
        session.workers.clear()

        ops_log.end(op)
        logger.info(f"Session {session_id} torn down")

        # Don't remove from dict immediately — keep for status queries
        return True

    def list_sessions(self) -> list:
        """List all sessions with basic info."""
        return [
            {
                "session_id": s.session_id,
                "status": s.status.value,
                "total_lines": s.total_lines,
                "completed_lines": s.completed_lines,
                "progress_pct": s.progress_pct,
                "queued_items": s.queued_items,
                "created_at": s.created_at,
            }
            for s in self.sessions.values()
        ]

    def scheduler_snapshot(self) -> dict:
        """Aggregate session backlog and runtime state for GPU schedulers."""
        status_counts = {status.value: 0 for status in SessionStatus}
        queued_session_items = 0
        active_sessions = 0
        active_workers = 0

        for session in self.sessions.values():
            status_counts[session.status.value] = status_counts.get(session.status.value, 0) + 1
            queued_session_items += session.queued_items
            active_workers += sum(
                1 for worker in session.workers if getattr(worker, "_running", False)
            )
            if session.status in {
                SessionStatus.PREPARING,
                SessionStatus.READY,
                SessionStatus.PROCESSING,
            }:
                active_sessions += 1

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "queued_session_items": queued_session_items,
            "active_workers": active_workers,
            "status_counts": status_counts,
        }
