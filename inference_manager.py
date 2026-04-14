# coding=utf-8
"""Inference manager — loads/unloads Qwen3-TTS models with GPU idle management.

Supports both CustomVoice (fine-tuned) and VoiceDesign (generate from description).
Auto-unloads from VRAM after configurable idle timeout.
"""

import collections
import io
import logging
import re
import traceback
import threading
import time
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Dict

import soundfile as sf
import torch
from ops_logger import ops_log

logger = logging.getLogger(__name__)

# Default VoiceDesign model from HuggingFace
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Default Base model from HuggingFace
VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

_CUDA_BUSY_ERROR_PATTERNS = (
    "busy or unavailable",
    "device(s) is/are busy",
    "devices are busy",
    "device busy",
    "cuda-capable device",
)


def _iter_exception_chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_retryable_cuda_busy_error(exc: BaseException) -> bool:
    for candidate in _iter_exception_chain(exc):
        message = str(candidate).lower()
        if "cuda" not in message:
            continue
        if any(pattern in message for pattern in _CUDA_BUSY_ERROR_PATTERNS):
            return True
    return False


class RuntimeAdjustableLimiter:
    """Thread-safe limiter whose capacity can be changed at runtime."""

    def __init__(self, capacity: int):
        self._capacity = max(1, int(capacity))
        self._active = 0
        self._holders: dict[str, int] = {}
        self._waiters: dict[str, int] = {}
        self._condition = threading.Condition()
        self._wait_heartbeat_seconds = 30.0

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "capacity": self._capacity,
            "active": self._active,
            "available": max(self._capacity - self._active, 0),
            "holders": dict(self._holders),
            "waiting": dict(self._waiters),
        }

    def acquire(self, label: str = "unknown"):
        with self._condition:
            wait_started_at = None
            while self._active >= self._capacity:
                if wait_started_at is None:
                    wait_started_at = time.time()
                    self._waiters[label] = self._waiters.get(label, 0) + 1
                    ops_log.log_event(
                        "limiter_wait_started",
                        extra={
                            "label": label,
                            **self._snapshot_locked(),
                        },
                    )

                self._condition.wait(timeout=self._wait_heartbeat_seconds)

                if self._active >= self._capacity and wait_started_at is not None:
                    ops_log.log_event(
                        "limiter_wait_heartbeat",
                        extra={
                            "label": label,
                            "waited_seconds": round(time.time() - wait_started_at, 3),
                            **self._snapshot_locked(),
                        },
                    )

            if wait_started_at is not None:
                count = self._waiters.get(label, 0)
                if count <= 1:
                    self._waiters.pop(label, None)
                else:
                    self._waiters[label] = count - 1
            self._active += 1
            self._holders[label] = self._holders.get(label, 0) + 1
            if wait_started_at is not None:
                ops_log.log_event(
                    "limiter_wait_finished",
                    extra={
                        "label": label,
                        "waited_seconds": round(time.time() - wait_started_at, 3),
                        **self._snapshot_locked(),
                    },
                )

    def release(self, label: str = "unknown"):
        with self._condition:
            if self._active > 0:
                self._active -= 1
            count = self._holders.get(label, 0)
            if count <= 1:
                self._holders.pop(label, None)
            else:
                self._holders[label] = count - 1
            self._condition.notify_all()

    def update_capacity(self, capacity: int):
        with self._condition:
            self._capacity = max(1, int(capacity))
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_locked()


class InferenceManager:
    """Manages loaded TTS models with automatic GPU idle unloading.

    Handles two model types:
      - CustomVoice: fine-tuned checkpoints (loaded per job)
      - VoiceDesign: the shared VoiceDesign model (loaded on demand)

    Shared and custom models are cached in VRAM up to a configurable limit.
    Auto-unloads inactive models after the idle timeout.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        use_flash_attn: bool = True,
        idle_timeout_seconds: int = 600,
        max_concurrency: int = 16,
        max_models: int = 2,  # Safe default for keeping shared clone + design models resident
        compile: bool = False,
        gpu_controller: Any = None,
        shared_model_replicas: Optional[Dict[str, int]] = None,
        shared_model_min_headroom_gb: float = 4.0,
    ):
        self._device = device
        self._attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        self._compile = compile
        self._gpu_controller = gpu_controller
        self._lock = threading.Lock()
        # Cap concurrent inference to max_models: each inference holds one
        # model in VRAM, so we can never run more than max_models at once.
        self._inference_limiter = RuntimeAdjustableLimiter(max_models)

        # Model cache: Dict[path, (model, type, speaker_name)]
        # We use an OrderedDict to implement LRU
        self._models: Dict[str, tuple] = collections.OrderedDict()
        self._max_models = max_models

        # Model-in-use tracking: prevents LRU eviction of models that are
        # actively running inference.  Dict[cache_key, refcount].
        self._models_in_use: Dict[str, int] = {}
        self._execution_locks: Dict[str, threading.Lock] = {}
        self._shared_replica_loads: Dict[str, int] = {}
        self._shared_model_replicas = {
            "voice_design": 1,
            "voice_clone": 1,
        }
        if shared_model_replicas:
            for model_type, count in shared_model_replicas.items():
                self._shared_model_replicas[model_type] = max(1, int(count))
        self._shared_model_min_headroom_gb = float(shared_model_min_headroom_gb)
        self._estimated_model_vram_gb = 5.5

        # Session pinning: models pinned by active sessions won't be LRU evicted
        # Dict[cache_key, set[session_id]]
        self._session_pins: Dict[str, set] = {}

        # Tracking for properties (historical/last used)
        self._last_path: Optional[str] = None
        self._last_type: Optional[str] = None
        self._last_speaker: Optional[str] = None

        # Idle timeout (applies to the entire cache)
        self._idle_timeout = idle_timeout_seconds
        self._last_used: float = 0.0
        self._active_requests: int = 0
        self._last_request_started_at: Optional[float] = None
        self._last_request_finished_at: Optional[float] = None
        self._idle_started_at: Optional[float] = time.time()
        self._idle_timer: Optional[threading.Timer] = None
        self._auto_unload_enabled = idle_timeout_seconds > 0

        # Stats
        self._total_requests = 0
        self._total_loads = 0
        self._total_unloads = 0

    # -- Properties -----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return len(self._models) > 0

    @property
    def loaded_count(self) -> int:
        return len(self._models)

    @property
    def max_models(self) -> int:
        return self._max_models

    @max_models.setter
    def max_models(self, value: int):
        with self._lock:
            self._set_max_models_locked(value)

    @property
    def loaded_paths(self) -> list[str]:
        return list(self._models.keys())

    @property
    def idle_timeout(self) -> int:
        return self._idle_timeout

    @idle_timeout.setter
    def idle_timeout(self, seconds: int):
        self._idle_timeout = seconds
        self._auto_unload_enabled = seconds > 0
        if not self._auto_unload_enabled:
            self._cancel_idle_timer()
        elif self.is_loaded:
            self._reset_idle_timer()

    @property
    def stats(self) -> dict:
        gpu_info = {}
        if torch.cuda.is_available():
            snapshot = self._get_gpu_memory_snapshot()
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(snapshot["total_gb"], 2),
                "gpu_memory_allocated_gb": round(snapshot["allocated_gb"], 2),
                "gpu_memory_reserved_gb": round(snapshot["reserved_gb"], 2),
                "gpu_memory_free_gb": round(snapshot["free_gb"], 2),
            }

        idle_seconds = (
            round(time.time() - self._idle_started_at, 1)
            if self._idle_started_at is not None and self._active_requests == 0
            else None
        )

        return {
            "model_loaded": self.is_loaded,
            "loaded_count": self.loaded_count,
            "max_models": self._max_models,
            "loaded_checkpoints": self.loaded_paths,
            "shared_model_replicas": dict(self._shared_model_replicas),
            "shared_model_min_headroom_gb": self._shared_model_min_headroom_gb,
            "inference_limiter": self._inference_limiter.snapshot(),
            "auto_unload_enabled": self._auto_unload_enabled,
            "idle_timeout_seconds": self._idle_timeout,
            "active_requests": self._active_requests,
            "last_request_started_at": self._format_timestamp(self._last_request_started_at),
            "last_request_finished_at": self._format_timestamp(self._last_request_finished_at),
            "idle_started_at": self._format_timestamp(self._idle_started_at),
            "idle_seconds": idle_seconds,
            "total_requests": self._total_requests,
            "total_loads": self._total_loads,
            "total_unloads": self._total_unloads,
            **gpu_info,
        }

    # -- Idle timer management ------------------------------------------------

    def _cancel_idle_timer(self):
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _reset_idle_timer(self):
        self._cancel_idle_timer()
        if self._auto_unload_enabled and self._idle_timeout > 0:
            self._idle_timer = threading.Timer(self._idle_timeout, self._on_idle_timeout)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _on_idle_timeout(self):
        with self._lock:
            if self.is_loaded:
                if self._active_requests > 0:
                    # Delay unloading if requests are currently active
                    self._reset_idle_timer()
                    return
                elapsed = time.time() - self._last_used
                if elapsed >= self._idle_timeout:
                    logger.info(
                        f"GPU idle for {elapsed:.0f}s (timeout={self._idle_timeout}s). "
                        f"Unloading all {self.loaded_count} models."
                    )
                    ops_log.log_event("gpu_idle_timeout", extra={"elapsed": round(elapsed, 1), "loaded_count": self.loaded_count})
                    self._unload_all_unsafe()

    def _touch(self):
        self._last_used = time.time()
        self._reset_idle_timer()

    @staticmethod
    def _format_timestamp(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))

    def _get_gpu_memory_snapshot(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {
                "total_gb": 0.0,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": 0.0,
            }

        total_bytes = torch.cuda.get_device_properties(0).total_memory
        allocated_bytes = torch.cuda.memory_allocated(0)
        reserved_bytes = torch.cuda.memory_reserved(0)
        free_bytes = None
        mem_get_info = getattr(torch.cuda, "mem_get_info", None)

        if mem_get_info is not None:
            try:
                free_bytes, total_bytes = mem_get_info(0)
            except TypeError:
                free_bytes, total_bytes = mem_get_info()
            except Exception:
                free_bytes = None

        if free_bytes is None:
            free_bytes = max(total_bytes - reserved_bytes, 0)

        return {
            "total_gb": total_bytes / 1e9,
            "allocated_gb": allocated_bytes / 1e9,
            "reserved_gb": reserved_bytes / 1e9,
            "free_gb": max(free_bytes, 0) / 1e9,
        }

    def _build_runtime_diagnostics_locked(self, **extra) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "loaded_count": self.loaded_count,
            "max_models": self._max_models,
            "loaded_checkpoints": self.loaded_paths[:8],
            "models_in_use": dict(self._models_in_use),
            "shared_model_replicas": dict(self._shared_model_replicas),
            "shared_model_min_headroom_gb": self._shared_model_min_headroom_gb,
            "inference_limiter": self._inference_limiter.snapshot(),
            "gpu_memory": self._get_gpu_memory_snapshot(),
        }
        diagnostics.update(extra)
        return diagnostics

    def _set_max_models_locked(self, value: int):
        value = int(value)
        if value < 1:
            raise ValueError("max_models must be >= 1")
        self._max_models = value
        self._inference_limiter.update_capacity(value)
        ops_log.log_event("max_models_updated", extra={"new_value": value})
        self._enforce_cache_size()

    def update_runtime_config(
        self,
        *,
        max_models: Optional[int] = None,
        shared_model_replicas: Optional[Dict[str, int]] = None,
        shared_model_min_headroom_gb: Optional[float] = None,
    ) -> dict:
        with self._lock:
            if max_models is not None:
                self._set_max_models_locked(max_models)

            if shared_model_replicas is not None:
                updated_replicas = dict(self._shared_model_replicas)
                for model_type, count in shared_model_replicas.items():
                    normalized_count = int(count)
                    if normalized_count < 1:
                        raise ValueError(f"{model_type} replicas must be >= 1")
                    updated_replicas[model_type] = normalized_count
                self._shared_model_replicas = updated_replicas

            if shared_model_min_headroom_gb is not None:
                normalized_headroom = float(shared_model_min_headroom_gb)
                if normalized_headroom < 0:
                    raise ValueError("shared_model_min_headroom_gb must be >= 0")
                self._shared_model_min_headroom_gb = normalized_headroom

            return {
                "max_models": self._max_models,
                "shared_model_replicas": dict(self._shared_model_replicas),
                "shared_model_min_headroom_gb": self._shared_model_min_headroom_gb,
            }

    @contextlib.contextmanager
    def _track_active(self):
        with self._lock:
            self._cancel_idle_timer()
            self._active_requests += 1
            self._last_request_started_at = time.time()
            self._idle_started_at = None
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
                self._last_request_finished_at = time.time()
                if self._active_requests <= 0:
                    self._active_requests = 0
                    self._idle_started_at = self._last_request_finished_at
                self._touch()

    # -- Speaker name normalisation -------------------------------------------

    @staticmethod
    def _normalize_speaker_name(name: str) -> str:
        """Normalize a human-readable speaker name to the format stored in model configs.

        E.g. 'Mr. Justice Wargrave' → 'mr__justice_wargrave'
        (dot and space each become '_', adjacent underscores are preserved)
        """
        if not name:
            return name
        return name.lower().replace(" ", "_").replace(".", "_").strip("_")

    # -- Load / Unload --------------------------------------------------------

    def _load_model(self, path: str, model_type: str, speaker_name: Optional[str] = None):
        """Internal: load a model (caller must hold lock)."""
        return self._load_model_into_cache(
            cache_key=path,
            source_path=path,
            model_type=model_type,
            speaker_name=speaker_name,
        )

    def _load_model_into_cache(
        self,
        cache_key: str,
        source_path: str,
        model_type: str,
        speaker_name: Optional[str] = None,
        session_id: str = "",
    ):
        """Internal: load a model from source_path and store it under cache_key."""
        if cache_key in self._models:
            self._models.move_to_end(cache_key)
            self._touch()
            if session_id:
                self._session_pins.setdefault(cache_key, set()).add(session_id)
            return self._models[cache_key][0]

        self._enforce_cache_size(reserve=1)

        op_name = "model_load_session" if session_id else "model_load"
        op = ops_log.start(op_name, extra={
            "cache_key": cache_key,
            "path": source_path,
            "model_type": model_type,
            "speaker": speaker_name,
        })
        try:
            from qwen_tts import Qwen3TTSModel

            if cache_key == source_path:
                logger.info(f"Loading {model_type} model from {source_path}...")
            else:
                logger.info(f"Loading {model_type} model from {source_path} as {cache_key}...")

            max_retries = 5
            retry_delay = 1.0
            model = None

            for attempt in range(max_retries):
                attempt_number = attempt + 1
                attempt_context = self._build_runtime_diagnostics_locked(
                    cache_key=cache_key,
                    path=source_path,
                    model_type=model_type,
                    speaker=speaker_name,
                    session_id=session_id or None,
                    attn_impl=self._attn_impl,
                    attempt=attempt_number,
                    max_retries=max_retries,
                    stage="from_pretrained",
                )
                ops_log.log_event("model_load_attempt", extra=attempt_context)
                try:
                    try:
                        model = Qwen3TTSModel.from_pretrained(
                            source_path,
                            device_map=self._device,
                            dtype=torch.bfloat16,
                            attn_implementation=self._attn_impl,
                        )
                        break
                    except Exception as e:
                        if self._attn_impl == "flash_attention_2" and not any(x in str(e).lower() for x in ["busy", "unavailable"]):
                            err_str = str(e)
                            if any(x in err_str for x in ["FlashAttention2", "flash-attn", "flash_attn", "package f", "DLL load failed"]):
                                logger.warning(
                                    f"Flash Attention (v2) could not be loaded for {source_path}. "
                                    f"Error: {err_str}. Falling back to 'eager' implementation."
                                )
                                model = Qwen3TTSModel.from_pretrained(
                                    source_path,
                                    device_map=self._device,
                                    dtype=torch.bfloat16,
                                    attn_implementation="eager",
                                )
                                break
                            raise e
                        raise e
                except Exception as load_err:
                    if _is_retryable_cuda_busy_error(load_err) and attempt < max_retries - 1:
                        retry_context = self._build_runtime_diagnostics_locked(
                            cache_key=cache_key,
                            path=source_path,
                            model_type=model_type,
                            speaker=speaker_name,
                            session_id=session_id or None,
                            attn_impl=self._attn_impl,
                            attempt=attempt_number,
                            max_retries=max_retries,
                            stage="retry_after_cuda_busy",
                            error=str(load_err),
                        )
                        ops_log.log_event(
                            "model_load_retry",
                            extra=retry_context,
                            level=logging.WARNING,
                        )
                        logger.warning(
                            f"CUDA device busy (attempt {attempt+1}/{max_retries}). "
                            f"Retrying in {retry_delay}s..."
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise load_err

            if self._compile:
                with ops_log.operation("model_compile", extra={"path": source_path, "cache_key": cache_key}):
                    logger.info("Compiling model for faster inference (this may take a few minutes)...")
                    model.model = torch.compile(model.model, mode="reduce-overhead")

            self._models[cache_key] = (model, model_type, speaker_name)
            self._last_path = cache_key
            self._last_type = model_type
            self._last_speaker = speaker_name
            self._total_loads += 1
            if session_id:
                self._session_pins.setdefault(cache_key, set()).add(session_id)
            self._touch()

            mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            logger.info(
                f"{model_type} model loaded into cache as {cache_key}. "
                f"Counts: {self.loaded_count}/{self._max_models}. GPU: {mem:.2f} GB"
            )
            ops_log.end(op, extra={"gpu_memory_gb": round(mem, 2)})
            return model
        except Exception as e:
            failure_context = self._build_runtime_diagnostics_locked(
                cache_key=cache_key,
                path=source_path,
                model_type=model_type,
                speaker=speaker_name,
                session_id=session_id or None,
                attn_impl=self._attn_impl,
                stage="load_model_into_cache",
                traceback_tail=traceback.format_exc(limit=6).strip().splitlines()[-6:],
            )
            ops_log.fail(op, str(e), extra=failure_context)
            logger.exception(
                "Model load failed for model_type=%s cache_key=%s source_path=%s diagnostics=%s",
                model_type,
                cache_key,
                source_path,
                failure_context,
            )
            raise

    def _enforce_cache_size(self, reserve: int = 0):
        """Internal: remove LRU models if over capacity (caller must hold lock).
        Respects session-pinned AND in-use models (won't evict them)."""
        while len(self._models) > (self._max_models - reserve) and self._models:
            evicted = False
            for path in list(self._models.keys()):
                # Skip models actively running inference
                if self._models_in_use.get(path, 0) > 0:
                    continue
                # Skip session-pinned models
                if path in self._session_pins and self._session_pins[path]:
                    continue
                model_tuple = self._models.pop(path)
                model, mtype, _ = model_tuple
                logger.info(f"LRU Eviction: Unloading {mtype} model from {path}")
                ops_log.log_event("model_eviction", extra={"path": path, "type": mtype})
                del model
                self._execution_locks.pop(path, None)
                self._shared_replica_loads.pop(path, None)
                self._total_unloads += 1
                evicted = True
                break
            if not evicted:
                logger.warning("Cannot evict any models — all in-use or session-pinned")
                break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _mark_in_use(self, path: str):
        """Mark a model as actively running inference (caller holds lock)."""
        self._models_in_use[path] = self._models_in_use.get(path, 0) + 1

    def _mark_released(self, path: str):
        """Mark a model as no longer running inference (thread-safe)."""
        with self._lock:
            count = self._models_in_use.get(path, 0)
            if count <= 1:
                self._models_in_use.pop(path, None)
            else:
                self._models_in_use[path] = count - 1

    def _get_execution_lock(self, path: str) -> threading.Lock:
        with self._lock:
            if path not in self._execution_locks:
                self._execution_locks[path] = threading.Lock()
            return self._execution_locks[path]

    def _build_shared_replica_key(self, source_path: str, replica_index: int) -> str:
        return f"{source_path}::replica-{replica_index}"

    def _move_cache_entry_locked(self, old_key: str, new_key: str) -> bool:
        """Rename a cached model entry without reloading the weights."""
        if old_key == new_key or old_key not in self._models or new_key in self._models:
            return False

        model_tuple = self._models.pop(old_key)
        self._models[new_key] = model_tuple

        if old_key in self._models_in_use:
            self._models_in_use[new_key] = self._models_in_use.pop(old_key)
        if old_key in self._execution_locks:
            self._execution_locks[new_key] = self._execution_locks.pop(old_key)
        if old_key in self._shared_replica_loads:
            self._shared_replica_loads[new_key] = self._shared_replica_loads.pop(old_key)
        if old_key in self._session_pins:
            self._session_pins[new_key] = self._session_pins.pop(old_key)
        if self._last_path == old_key:
            self._last_path = new_key

        return True

    def _shared_replica_keys(self, source_path: str, model_type: str) -> list[str]:
        replica_count = max(1, int(self._shared_model_replicas.get(model_type, 1)))
        return [self._build_shared_replica_key(source_path, idx) for idx in range(replica_count)]

    def _has_shared_replica_headroom_locked(self) -> bool:
        if not torch.cuda.is_available():
            return True
        snapshot = self._get_gpu_memory_snapshot()
        return snapshot["free_gb"] >= (
            self._estimated_model_vram_gb + self._shared_model_min_headroom_gb
        )

    def _acquire_shared_replica(self, source_path: str, model_type: str) -> str:
        with self._lock:
            candidate_keys = self._shared_replica_keys(source_path, model_type)
            loaded_candidates = [key for key in candidate_keys if key in self._models]
            headroom_ok = self._has_shared_replica_headroom_locked()

            can_expand = (
                len(loaded_candidates) < len(candidate_keys)
                and self.loaded_count < self._max_models
                and headroom_ok
            )

            if can_expand:
                selectable = candidate_keys
            elif loaded_candidates:
                selectable = loaded_candidates
            else:
                selectable = [candidate_keys[0]]

            selected = min(
                selectable,
                key=lambda key: (
                    self._shared_replica_loads.get(key, 0),
                    0 if key in self._models else 1,
                    key,
                ),
            )
            selection_kind = "expand_or_load" if selected not in loaded_candidates else "reuse_loaded"
            selection_context = self._build_runtime_diagnostics_locked(
                model_type=model_type,
                source_path=source_path,
                selected_cache_key=selected,
                selection_kind=selection_kind,
                candidate_replica_count=len(candidate_keys),
                loaded_candidate_count=len(loaded_candidates),
                loaded_candidates=loaded_candidates,
                headroom_ok=headroom_ok,
                can_expand=can_expand,
            )
            ops_log.log_event("shared_replica_selection", extra=selection_context)
            self._shared_replica_loads[selected] = self._shared_replica_loads.get(selected, 0) + 1
            return selected

    def _release_shared_replica(self, cache_key: str):
        with self._lock:
            count = self._shared_replica_loads.get(cache_key, 0)
            if count <= 1:
                self._shared_replica_loads.pop(cache_key, None)
            else:
                self._shared_replica_loads[cache_key] = count - 1

    def load(self, checkpoint_path: str, speaker_name: str):
        """Load a fine-tuned CustomVoice checkpoint."""
        with self._lock:
            self._load_model(checkpoint_path, "custom_voice", speaker_name)

    def load_voice_design(self, model_path: str = VOICE_DESIGN_MODEL):
        """Load the VoiceDesign model."""
        with self._lock:
            replica_key = self._build_shared_replica_key(model_path, 0)
            if replica_key in self._models:
                self._models.move_to_end(replica_key)
                self._last_path = replica_key
                self._last_type = "voice_design"
                self._last_speaker = None
                self._touch()
                return
            if self._move_cache_entry_locked(model_path, replica_key):
                self._last_type = "voice_design"
                self._last_speaker = None
                self._touch()
                return
            self._load_model_into_cache(
                cache_key=replica_key,
                source_path=model_path,
                model_type="voice_design",
            )

    def load_voice_clone(self, model_path: str = VOICE_CLONE_MODEL):
        """Load the Base model for zero-shot voice cloning."""
        with self._lock:
            replica_key = self._build_shared_replica_key(model_path, 0)
            if replica_key in self._models:
                self._models.move_to_end(replica_key)
                self._last_path = replica_key
                self._last_type = "voice_clone"
                self._last_speaker = None
                self._touch()
                return
            if self._move_cache_entry_locked(model_path, replica_key):
                self._last_type = "voice_clone"
                self._last_speaker = None
                self._touch()
                return
            self._load_model_into_cache(
                cache_key=replica_key,
                source_path=model_path,
                model_type="voice_clone",
            )

    def _unload_all_unsafe(self):
        count = len(self._models)
        self._models.clear()
        self._execution_locks.clear()
        self._shared_replica_loads.clear()
        self._cancel_idle_timer()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if count > 0:
            self._total_unloads += count
            logger.info(f"Unloaded {count} model(s) from VRAM cache.")

    def unload(self):
        with self._lock:
            self._unload_all_unsafe()

    # -- Session-aware loading ------------------------------------------------

    def load_for_session(
        self, cache_key: str, checkpoint_path: str, speaker_name: str,
        session_id: str = "",
    ):
        """Load a model for a session, optionally as a replica.

        For replicas, cache_key differs from checkpoint_path
        (e.g. 'path/to/model::replica-1'). The actual weights are loaded
        from checkpoint_path, but stored under cache_key in the cache.

        Session-pinned models are protected from LRU eviction.
        """
        with self._lock:
            return self._load_model_into_cache(
                cache_key=cache_key,
                source_path=checkpoint_path,
                model_type="custom_voice",
                speaker_name=speaker_name,
                session_id=session_id,
            )

    def unload_specific(self, cache_key: str):
        """Unload a specific model by its cache key."""
        with self._lock:
            if cache_key in self._models:
                model, mtype, _ = self._models.pop(cache_key)
                del model
                self._total_unloads += 1
                self._session_pins.pop(cache_key, None)
                self._execution_locks.pop(cache_key, None)
                self._shared_replica_loads.pop(cache_key, None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Unloaded specific model: {cache_key}")
                return True
        return False

    def unpin_session(self, cache_key: str, session_id: str):
        """Remove a session pin from a model. If no pins remain, the model
        becomes eligible for LRU eviction again."""
        with self._lock:
            if cache_key in self._session_pins:
                self._session_pins[cache_key].discard(session_id)
                if not self._session_pins[cache_key]:
                    del self._session_pins[cache_key]
                    logger.info(f"Model {cache_key} fully unpinned")

    def get_vram_budget(self) -> dict:
        """Return VRAM budget information."""
        if not torch.cuda.is_available():
            return {"total_gb": 0, "allocated_gb": 0, "free_gb": 0}
        snapshot = self._get_gpu_memory_snapshot()
        return {
            "total_gb": round(snapshot["total_gb"], 2),
            "allocated_gb": round(snapshot["allocated_gb"], 2),
            "free_gb": round(snapshot["free_gb"], 2),
            "models_loaded": self.loaded_count,
            "session_pinned": len(self._session_pins),
        }

    def _get_model(self, path: str, model_type: str, speaker_name: Optional[str] = None):
        """Get model from cache or load it (caller holds lock)."""
        if path in self._models:
            self._models.move_to_end(path)
            self._touch()
            return self._models[path][0], self._models[path][2]
        
        # Load it
        model = self._load_model(path, model_type, speaker_name)
        return model, speaker_name

    def _get_model_by_cache_key(
        self,
        cache_key: str,
        source_path: str,
        model_type: str,
        speaker_name: Optional[str] = None,
    ):
        """Get a cached model replica or load it from its source path."""
        if cache_key in self._models:
            self._models.move_to_end(cache_key)
            self._touch()
            return self._models[cache_key][0], self._models[cache_key][2]

        model = self._load_model_into_cache(
            cache_key=cache_key,
            source_path=source_path,
            model_type=model_type,
            speaker_name=speaker_name,
        )
        return model, speaker_name

    # -- CustomVoice inference ------------------------------------------------

    # Thread pool for CPU-bound WAV encoding (shared across all generate methods)
    _wav_pool = ThreadPoolExecutor(max_workers=4)

    @staticmethod
    def _encode_wav(wav, sr) -> bytes:
        """Encode a numpy waveform to WAV bytes (CPU-bound, off GPU thread)."""
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        return buf.read()

    def generate_batch(
        self,
        texts: list[str],
        checkpoint_path: str,
        speaker_name: str,
        languages: list[str] = None,
        instructs: list[str] = None,
        max_new_tokens: Optional[int] = None,
        **generation_kwargs,
    ) -> tuple[list[bytes], int]:
        """Generate speech for multiple texts using CustomVoice model."""
        if not languages:
            languages = ["English"] * len(texts)
        if not instructs:
            instructs = [""] * len(texts)

        # Acquire semaphore FIRST to limit concurrent model usage to max_models
        with ops_log.operation("gpu_resource_wait", extra={"checkpoint": checkpoint_path}):
            if self._gpu_controller:
                self._gpu_controller.begin_inference("inference_custom_voice_batch")
            self._inference_limiter.acquire("inference_custom_voice_batch")
        try:
            with self._lock:
                model, spk = self._get_model(checkpoint_path, "custom_voice", speaker_name)
                self._mark_in_use(checkpoint_path)
                self._total_requests += len(texts)
            model_lock = self._get_execution_lock(checkpoint_path)

            try:
                with self._track_active():
                    effective_generate_kwargs = dict(generation_kwargs)
                    if max_new_tokens is not None and "max_new_tokens" not in effective_generate_kwargs:
                        effective_generate_kwargs["max_new_tokens"] = max_new_tokens
                    op = ops_log.start("inference_custom_voice_batch", extra={
                        "batch_size": len(texts),
                        "speaker": spk,
                        "max_new_tokens": effective_generate_kwargs.get("max_new_tokens"),
                        "do_sample": effective_generate_kwargs.get("do_sample"),
                    })
                    logger.info(
                        "Speaker '%s' started saying %s texts (max_new_tokens=%s, do_sample=%s).",
                        spk,
                        len(texts),
                        effective_generate_kwargs.get("max_new_tokens"),
                        effective_generate_kwargs.get("do_sample"),
                    )
                    try:
                        speakers = [self._normalize_speaker_name(spk) if spk else spk] * len(texts)

                        with model_lock:
                            wavs_list, sr = model.generate_custom_voice(
                                text=texts,
                                language=languages,
                                speaker=speakers,
                                instruct=instructs,
                                **effective_generate_kwargs,
                            )

                        # Encode WAVs in parallel on CPU threads (frees GPU thread)
                        results = list(self._wav_pool.map(
                            lambda w: self._encode_wav(w, sr), wavs_list
                        ))

                        ops_log.end(op, extra={"sample_rate": sr})
                        logger.info(f"Speaker '{spk}' finished saying {len(texts)} texts.")
                        return results, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(checkpoint_path)
        finally:
            self._inference_limiter.release("inference_custom_voice_batch")
            if self._gpu_controller:
                self._gpu_controller.end_inference()

    def generate(
        self,
        text: str,
        checkpoint_path: str,
        speaker_name: str,
        language: str = "English",
        instruct: str = "",
        max_new_tokens: Optional[int] = None,
        **generation_kwargs,
    ) -> tuple[bytes, int]:
        """Generate speech using CustomVoice model. Auto-loads if not in cache."""
        with ops_log.operation("gpu_resource_wait", extra={"checkpoint": checkpoint_path}):
            if self._gpu_controller:
                self._gpu_controller.begin_inference("inference_custom_voice")
            self._inference_limiter.acquire("inference_custom_voice")
        try:
            with self._lock:
                model, spk = self._get_model(checkpoint_path, "custom_voice", speaker_name)
                self._mark_in_use(checkpoint_path)
                self._total_requests += 1
            model_lock = self._get_execution_lock(checkpoint_path)

            try:
                with self._track_active():
                    effective_generate_kwargs = dict(generation_kwargs)
                    if max_new_tokens is not None and "max_new_tokens" not in effective_generate_kwargs:
                        effective_generate_kwargs["max_new_tokens"] = max_new_tokens
                    op = ops_log.start("inference_custom_voice", extra={
                        "text_length": len(text),
                        "language": language,
                        "speaker": spk,
                        "max_new_tokens": effective_generate_kwargs.get("max_new_tokens"),
                        "do_sample": effective_generate_kwargs.get("do_sample"),
                    })
                    logger.info(
                        "Speaker '%s' started saying text: '%s...' (max_new_tokens=%s, do_sample=%s)",
                        spk,
                        text[:50],
                        effective_generate_kwargs.get("max_new_tokens"),
                        effective_generate_kwargs.get("do_sample"),
                    )
                    try:
                        with model_lock:
                            wavs, sr = model.generate_custom_voice(
                                text=text,
                                language=language,
                                speaker=self._normalize_speaker_name(spk) if spk else spk,
                                instruct=instruct if instruct else None,
                                **effective_generate_kwargs,
                            )

                        result = self._encode_wav(wavs[0], sr)
                        ops_log.end(op, extra={"audio_bytes": len(result), "sample_rate": sr})
                        return result, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(checkpoint_path)
        finally:
            self._inference_limiter.release("inference_custom_voice")
            if self._gpu_controller:
                self._gpu_controller.end_inference()

    # -- VoiceDesign inference ------------------------------------------------

    def generate_voice_design_batch(
        self,
        texts: list[str],
        instructs: list[str],
        languages: list[str] = None,
    ) -> tuple[list[bytes], int]:
        """Generate speech for multiple texts using VoiceDesign model."""
        if not languages:
            languages = ["English"] * len(texts)

        with ops_log.operation("gpu_resource_wait", extra={"model": "voice_design"}):
            if self._gpu_controller:
                self._gpu_controller.begin_inference("inference_voice_design_batch")
            self._inference_limiter.acquire("inference_voice_design_batch")
        cache_key = self._acquire_shared_replica(VOICE_DESIGN_MODEL, "voice_design")
        try:
            try:
                with self._lock:
                    model, _ = self._get_model_by_cache_key(cache_key, VOICE_DESIGN_MODEL, "voice_design")
                    self._mark_in_use(cache_key)
                    self._total_requests += len(texts)
            except Exception as e:
                with self._lock:
                    prepare_context = self._build_runtime_diagnostics_locked(
                        model_type="voice_design",
                        cache_key=cache_key,
                        path=VOICE_DESIGN_MODEL,
                        batch_size=len(texts),
                        stage="prepare_voice_design_batch",
                        error=str(e),
                    )
                ops_log.log_event(
                    "voice_design_model_prepare_failed",
                    extra=prepare_context,
                    level=logging.ERROR,
                )
                raise
            model_lock = self._get_execution_lock(cache_key)

            try:
                with self._track_active():
                    op = ops_log.start("inference_voice_design_batch", extra={
                        "batch_size": len(texts),
                        "cache_key": cache_key,
                    })
                    logger.info(f"VoiceDesign started for {len(texts)} texts on {cache_key}.")
                    try:
                        with model_lock:
                            wavs_list, sr = model.generate_voice_design(
                                text=texts,
                                instruct=instructs,
                                language=languages,
                            )

                        # Encode WAVs in parallel on CPU threads (frees GPU thread)
                        results = list(self._wav_pool.map(
                            lambda w: self._encode_wav(w, sr), wavs_list
                        ))

                        ops_log.end(op, extra={"sample_rate": sr})
                        logger.info(f"VoiceDesign finished for {len(texts)} texts on {cache_key}.")
                        return results, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(cache_key)
        finally:
            self._release_shared_replica(cache_key)
            self._inference_limiter.release("inference_voice_design_batch")
            if self._gpu_controller:
                self._gpu_controller.end_inference()

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "English",
    ) -> tuple[bytes, int]:
        """Generate speech using VoiceDesign model."""
        with ops_log.operation("gpu_resource_wait", extra={"model": "voice_design"}):
            if self._gpu_controller:
                self._gpu_controller.begin_inference("inference_voice_design")
            self._inference_limiter.acquire("inference_voice_design")
        cache_key = self._acquire_shared_replica(VOICE_DESIGN_MODEL, "voice_design")
        try:
            try:
                with self._lock:
                    model, _ = self._get_model_by_cache_key(cache_key, VOICE_DESIGN_MODEL, "voice_design")
                    self._mark_in_use(cache_key)
                    self._total_requests += 1
            except Exception as e:
                with self._lock:
                    prepare_context = self._build_runtime_diagnostics_locked(
                        model_type="voice_design",
                        cache_key=cache_key,
                        path=VOICE_DESIGN_MODEL,
                        text_length=len(text),
                        instruct_length=len(instruct),
                        stage="prepare_voice_design",
                        error=str(e),
                    )
                ops_log.log_event(
                    "voice_design_model_prepare_failed",
                    extra=prepare_context,
                    level=logging.ERROR,
                )
                raise
            model_lock = self._get_execution_lock(cache_key)

            try:
                with self._track_active():
                    op = ops_log.start("inference_voice_design", extra={
                        "text_length": len(text),
                        "instruct_length": len(instruct),
                        "language": language,
                        "cache_key": cache_key,
                    })
                    logger.info(f"VoiceDesign started for text on {cache_key}: '{text[:50]}...'")
                    try:
                        with model_lock:
                            wavs, sr = model.generate_voice_design(
                                text=text,
                                instruct=instruct,
                                language=language,
                            )

                        result = self._encode_wav(wavs[0], sr)
                        ops_log.end(op, extra={"audio_bytes": len(result), "sample_rate": sr})
                        return result, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(cache_key)
        finally:
            self._release_shared_replica(cache_key)
            self._inference_limiter.release("inference_voice_design")
            if self._gpu_controller:
                self._gpu_controller.end_inference()

    # -- VoiceClone inference -------------------------------------------------

    def generate_voice_clone_batch(
        self,
        texts: list[str],
        ref_audio: str,
        ref_text: str,
        languages: list[str] = None,
        x_vector_only_mode: bool = False,
    ) -> tuple[list[bytes], int]:
        """Generate speech for multiple texts using zero-shot VoiceClone Base model (single reference)."""
        return self.generate_voice_clone_flexible_batch(
            texts=texts,
            ref_audios=[ref_audio] * len(texts),
            ref_texts=[ref_text] * len(texts),
            languages=languages,
            x_vector_only_modes=[x_vector_only_mode] * len(texts)
        )

    def generate_voice_clone_flexible_batch(
        self,
        texts: list[str],
        ref_audios: list[str],
        ref_texts: list[str],
        languages: list[str] = None,
        x_vector_only_modes: list[bool] = None,
    ) -> tuple[list[bytes], int]:
        """Generate speech for multiple texts using zero-shot VoiceClone Base model (flexible references)."""
        if not languages:
            languages = ["English"] * len(texts)
        if not x_vector_only_modes:
            x_vector_only_modes = [False] * len(texts)

        with ops_log.operation("gpu_resource_wait", extra={"model": "voice_clone"}):
            if self._gpu_controller:
                self._gpu_controller.begin_inference("inference_voice_clone_flexible_batch")
            self._inference_limiter.acquire("inference_voice_clone_flexible_batch")
        cache_key = self._acquire_shared_replica(VOICE_CLONE_MODEL, "voice_clone")
        try:
            with self._lock:
                model, _ = self._get_model_by_cache_key(cache_key, VOICE_CLONE_MODEL, "voice_clone")
                self._mark_in_use(cache_key)
                self._total_requests += len(texts)
            model_lock = self._get_execution_lock(cache_key)

            try:
                with self._track_active():
                    op = ops_log.start("inference_voice_clone_flexible_batch", extra={
                        "batch_size": len(texts),
                        "cache_key": cache_key,
                    })
                    try:
                        with model_lock:
                            unique_prompt_cache: Dict[tuple[str, str, bool], Any] = {}
                            prompt_items = []

                            for ref_audio, ref_text, xvec_only in zip(ref_audios, ref_texts, x_vector_only_modes):
                                prompt_cache_key = None
                                if isinstance(ref_audio, str):
                                    prompt_cache_key = (ref_audio, ref_text or "", bool(xvec_only))

                                prompt_item = unique_prompt_cache.get(prompt_cache_key) if prompt_cache_key is not None else None
                                if prompt_item is None:
                                    built_items = model.create_voice_clone_prompt(
                                        ref_audio=ref_audio,
                                        ref_text=ref_text,
                                        x_vector_only_mode=xvec_only,
                                    )
                                    prompt_item = built_items[0]
                                    if prompt_cache_key is not None:
                                        unique_prompt_cache[prompt_cache_key] = prompt_item

                                prompt_items.append(prompt_item)

                            unique_ref_count = len(unique_prompt_cache) if unique_prompt_cache else len(ref_audios)
                            logger.info(
                                "VoiceClone flexible started on %s for %s texts with %s unique prompt(s).",
                                cache_key,
                                len(texts),
                                unique_ref_count,
                            )
                            wavs_list, sr = model.generate_voice_clone(
                                text=texts,
                                language=languages,
                                voice_clone_prompt=prompt_items,
                            )

                        # Encode WAVs in parallel on CPU threads (frees GPU thread)
                        results = list(self._wav_pool.map(
                            lambda w: self._encode_wav(w, sr), wavs_list
                        ))

                        ops_log.end(op, extra={"sample_rate": sr})
                        logger.info(f"VoiceClone flexible finished for {len(texts)} texts on {cache_key}.")
                        return results, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(cache_key)
        finally:
            self._release_shared_replica(cache_key)
            self._inference_limiter.release("inference_voice_clone_flexible_batch")
            if self._gpu_controller:
                self._gpu_controller.end_inference()

    def generate_to_file(
        self,
        text: str,
        output_path: str,
        language: str = "English",
        instruct: str = "",
        speaker: Optional[str] = None,
    ) -> int:
        """Generate speech and write to a WAV file. Returns sample rate."""
        wav_bytes, sr = self.generate(text, language, instruct, speaker)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        return sr
