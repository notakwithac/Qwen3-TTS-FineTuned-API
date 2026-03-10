# coding=utf-8
"""Inference manager — loads/unloads Qwen3-TTS models with GPU idle management.

Supports both CustomVoice (fine-tuned) and VoiceDesign (generate from description).
Auto-unloads from VRAM after configurable idle timeout.
"""

import collections
import io
import logging
import re
import threading
import time
import asyncio
import contextlib
from typing import Optional, Dict

import soundfile as sf
import torch
import traceback
from qwen_tts import Qwen3TTSModel
from ops_logger import ops_log

logger = logging.getLogger(__name__)

# Default VoiceDesign model from HuggingFace
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Default Base model from HuggingFace
VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class InferenceManager:
    """Manages loaded TTS models with automatic GPU idle unloading.

    Handles two model types:
      - CustomVoice: fine-tuned checkpoints (loaded per job)
      - VoiceDesign: the shared VoiceDesign model (loaded on demand)

    Only one model is in VRAM at a time. Auto-unloads after idle timeout.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        use_flash_attn: bool = True,
        idle_timeout_seconds: int = 600,
        max_concurrency: int = 16,
        max_models: int = 4,  # Default to 4 (good for 40GB)
        compile: bool = False,
    ):
        self._device = device
        self._attn_impl = "flash_attention_2" if use_flash_attn else "eager"
        self._compile = compile
        self._lock = threading.Lock()
        # Cap concurrent inference to max_models: each inference holds one
        # model in VRAM, so we can never run more than max_models at once.
        self._inference_semaphore = threading.Semaphore(max_models)

        # Model cache: Dict[path, (model, type, speaker_name)]
        # We use an OrderedDict to implement LRU
        self._models: Dict[str, tuple] = collections.OrderedDict()
        self._max_models = max_models

        # Model-in-use tracking: prevents LRU eviction of models that are
        # actively running inference.  Dict[cache_key, refcount].
        self._models_in_use: Dict[str, int] = {}

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
            self._max_models = value
            self._enforce_cache_size()

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
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                "gpu_memory_free_gb": round(
                    (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9, 2
                ),
            }

        idle_seconds = time.time() - self._last_used if self._last_used > 0 else None

        return {
            "model_loaded": self.is_loaded,
            "loaded_count": self.loaded_count,
            "max_models": self._max_models,
            "loaded_checkpoints": self.loaded_paths,
            "auto_unload_enabled": self._auto_unload_enabled,
            "idle_timeout_seconds": self._idle_timeout,
            "idle_seconds": round(idle_seconds, 1) if idle_seconds else None,
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
                    self._unload_all_unsafe()

    def _touch(self):
        self._last_used = time.time()
        self._reset_idle_timer()

    @contextlib.contextmanager
    def _track_active(self):
        with self._lock:
            self._active_requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_requests -= 1
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
        # If already in cache, move to end (MRU)
        if path in self._models:
            self._models.move_to_end(path)
            self._touch()
            return self._models[path][0]

        # Enforce cache size before loading new
        self._enforce_cache_size(reserve=1)

        op = ops_log.start("model_load", extra={"model_type": model_type, "path": path})
        try:
            logger.info(f"Loading {model_type} model from {path}...")
            try:
                model = Qwen3TTSModel.from_pretrained(
                    path,
                    device_map=self._device,
                    dtype=torch.bfloat16,
                    attn_implementation=self._attn_impl,
                )
            except Exception as e:
                # If we tried flash_attention_2 and it failed, fallback to eager
                if self._attn_impl == "flash_attention_2":
                    err_str = str(e)
                    if any(x in err_str for x in ["FlashAttention2", "flash-attn", "flash_attn", "package f", "DLL load failed"]):
                        logger.warning(
                            f"Flash Attention (v2) could not be loaded for {path}. "
                            f"Error: {err_str}. Falling back to 'eager' implementation."
                        )
                        model = Qwen3TTSModel.from_pretrained(
                            path,
                            device_map=self._device,
                            dtype=torch.bfloat16,
                            attn_implementation="eager",
                        )
                    else:
                        raise e
                else:
                    raise e
            
            # Speed up inference using torch.compile (requires Torch 2.0+)
            if self._compile:
                logger.info("Compiling model for faster inference (this may take a few minutes)...")
                # We compile the underlying Qwen3TTSForConditionalGeneration model
                model.model = torch.compile(model.model, mode="default")
            
            self._models[path] = (model, model_type, speaker_name)
            self._last_path = path
            self._last_type = model_type
            self._last_speaker = speaker_name
            self._total_loads += 1
            self._touch()

            mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            logger.info(f"{model_type} model loaded into cache. Counts: {self.loaded_count}/{self._max_models}. GPU: {mem:.2f} GB")
            ops_log.end(op, extra={"gpu_memory_gb": round(mem, 2)})
            return model
        except Exception as e:
            ops_log.fail(op, str(e))
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
                del model
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

    def load(self, checkpoint_path: str, speaker_name: str):
        """Load a fine-tuned CustomVoice checkpoint."""
        with self._lock:
            self._load_model(checkpoint_path, "custom_voice", speaker_name)

    def load_voice_design(self, model_path: str = VOICE_DESIGN_MODEL):
        """Load the VoiceDesign model."""
        with self._lock:
            if self._last_type == "voice_design" and self._last_path == model_path:
                self._touch()
                return  # Already loaded
            self._load_model(model_path, "voice_design")

    def load_voice_clone(self, model_path: str = VOICE_CLONE_MODEL):
        """Load the Base model for zero-shot voice cloning."""
        with self._lock:
            if self._last_type == "voice_clone" and self._last_path == model_path:
                self._touch()
                return  # Already loaded
            self._load_model(model_path, "voice_clone")

    def _unload_all_unsafe(self):
        count = len(self._models)
        self._models.clear()
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
            if cache_key in self._models:
                self._models.move_to_end(cache_key)
                self._touch()
                # Add session pin
                if session_id:
                    self._session_pins.setdefault(cache_key, set()).add(session_id)
                return self._models[cache_key][0]

            # For replicas, we load from the real checkpoint_path
            # but store under the cache_key
            self._enforce_cache_size(reserve=1)

            op = ops_log.start("model_load_session", extra={
                "cache_key": cache_key,
                "checkpoint_path": checkpoint_path,
                "speaker": speaker_name,
            })
            try:
                from qwen_tts import Qwen3TTSModel
                logger.info(f"Loading model from {checkpoint_path} as {cache_key}...")
                try:
                    model = Qwen3TTSModel.from_pretrained(
                        checkpoint_path,
                        device_map=self._device,
                        dtype=torch.bfloat16,
                        attn_implementation=self._attn_impl,
                    )
                except Exception as e:
                    if self._attn_impl == "flash_attention_2":
                        err_str = str(e)
                        if any(x in err_str for x in ["FlashAttention2", "flash-attn", "flash_attn", "package f", "DLL load failed"]):
                            logger.warning(f"Flash Attention fallback for {cache_key}: {err_str}")
                            model = Qwen3TTSModel.from_pretrained(
                                checkpoint_path,
                                device_map=self._device,
                                dtype=torch.bfloat16,
                                attn_implementation="eager",
                            )
                        else:
                            raise
                    else:
                        raise

                if self._compile:
                    logger.info(f"Compiling model {cache_key}...")
                    model.model = torch.compile(model.model, mode="default")

                self._models[cache_key] = (model, "custom_voice", speaker_name)
                self._last_path = cache_key
                self._last_type = "custom_voice"
                self._last_speaker = speaker_name
                self._total_loads += 1
                self._touch()

                # Pin to session
                if session_id:
                    self._session_pins.setdefault(cache_key, set()).add(session_id)

                mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                logger.info(
                    f"Session model loaded: {cache_key}. "
                    f"Cache: {self.loaded_count}/{self._max_models}. GPU: {mem:.2f} GB"
                )
                ops_log.end(op, extra={"gpu_memory_gb": round(mem, 2)})
                return model
            except Exception as e:
                ops_log.fail(op, str(e))
                raise

    def unload_specific(self, cache_key: str):
        """Unload a specific model by its cache key."""
        with self._lock:
            if cache_key in self._models:
                model, mtype, _ = self._models.pop(cache_key)
                del model
                self._total_unloads += 1
                self._session_pins.pop(cache_key, None)
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
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        return {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(total - allocated, 2),
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

    # -- CustomVoice inference ------------------------------------------------

    def generate_batch(
        self,
        texts: list[str],
        checkpoint_path: str,
        speaker_name: str,
        languages: list[str] = None,
        instructs: list[str] = None,
    ) -> tuple[list[bytes], int]:
        """Generate speech for multiple texts using CustomVoice model."""
        if not languages:
            languages = ["English"] * len(texts)
        if not instructs:
            instructs = [""] * len(texts)

        # Acquire semaphore FIRST to limit concurrent model usage to max_models
        with self._inference_semaphore:
            with self._lock:
                model, spk = self._get_model(checkpoint_path, "custom_voice", speaker_name)
                self._mark_in_use(checkpoint_path)
                self._total_requests += len(texts)

            try:
                with self._track_active():
                    op = ops_log.start("inference_custom_voice_batch", extra={
                        "batch_size": len(texts),
                        "speaker": spk,
                    })
                    try:
                        speakers = [self._normalize_speaker_name(spk) if spk else spk] * len(texts)

                        wavs_list, sr = model.generate_custom_voice(
                            text=texts,
                            language=languages,
                            speaker=speakers,
                            instruct=instructs,
                        )

                        results = []
                        for wav in wavs_list:
                            buf = io.BytesIO()
                            sf.write(buf, wav, sr, format="WAV")
                            buf.seek(0)
                            results.append(buf.read())

                        ops_log.end(op, extra={"sample_rate": sr})
                        return results, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(checkpoint_path)

    def generate(
        self,
        text: str,
        checkpoint_path: str,
        speaker_name: str,
        language: str = "English",
        instruct: str = "",
    ) -> tuple[bytes, int]:
        """Generate speech using CustomVoice model. Auto-loads if not in cache."""
        with self._inference_semaphore:
            with self._lock:
                model, spk = self._get_model(checkpoint_path, "custom_voice", speaker_name)
                self._mark_in_use(checkpoint_path)
                self._total_requests += 1

            try:
                with self._track_active():
                    op = ops_log.start("inference_custom_voice", extra={
                        "text_length": len(text),
                        "language": language,
                        "speaker": spk,
                    })
                    try:
                        wavs, sr = model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=self._normalize_speaker_name(spk) if spk else spk,
                            instruct=instruct if instruct else None,
                        )

                        buf = io.BytesIO()
                        sf.write(buf, wavs[0], sr, format="WAV")
                        buf.seek(0)
                        result = buf.read()
                        ops_log.end(op, extra={"audio_bytes": len(result), "sample_rate": sr})
                        return result, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(checkpoint_path)

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

        with self._inference_semaphore:
            with self._lock:
                model, _ = self._get_model(VOICE_DESIGN_MODEL, "voice_design")
                self._mark_in_use(VOICE_DESIGN_MODEL)
                self._total_requests += len(texts)

            try:
                with self._track_active():
                    op = ops_log.start("inference_voice_design_batch", extra={
                        "batch_size": len(texts),
                    })
                    try:
                        wavs_list, sr = model.generate_voice_design(
                            text=texts,
                            instruct=instructs,
                            language=languages,
                        )

                        results = []
                        for wav in wavs_list:
                            buf = io.BytesIO()
                            sf.write(buf, wav, sr, format="WAV")
                            buf.seek(0)
                            results.append(buf.read())

                        ops_log.end(op, extra={"sample_rate": sr})
                        return results, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(VOICE_DESIGN_MODEL)

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "English",
    ) -> tuple[bytes, int]:
        """Generate speech using VoiceDesign model."""
        with self._inference_semaphore:
            with self._lock:
                model, _ = self._get_model(VOICE_DESIGN_MODEL, "voice_design")
                self._mark_in_use(VOICE_DESIGN_MODEL)
                self._total_requests += 1

            try:
                with self._track_active():
                    op = ops_log.start("inference_voice_design", extra={
                        "text_length": len(text),
                        "instruct_length": len(instruct),
                        "language": language,
                    })
                    try:
                        wavs, sr = model.generate_voice_design(
                            text=text,
                            instruct=instruct,
                            language=language,
                        )

                        buf = io.BytesIO()
                        sf.write(buf, wavs[0], sr, format="WAV")
                        buf.seek(0)
                        result = buf.read()
                        ops_log.end(op, extra={"audio_bytes": len(result), "sample_rate": sr})
                        return result, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(VOICE_DESIGN_MODEL)

    # -- VoiceClone inference -------------------------------------------------

    def generate_voice_clone_batch(
        self,
        texts: list[str],
        ref_audio: str,
        ref_text: str,
        languages: list[str] = None,
        x_vector_only_mode: bool = False,
    ) -> tuple[list[bytes], int]:
        """Generate speech for multiple texts using zero-shot VoiceClone Base model."""
        if not languages:
            languages = ["English"] * len(texts)

        with self._inference_semaphore:
            with self._lock:
                model, _ = self._get_model(VOICE_CLONE_MODEL, "voice_clone")
                self._mark_in_use(VOICE_CLONE_MODEL)
                self._total_requests += len(texts)

            try:
                with self._track_active():
                    op = ops_log.start("inference_voice_clone_batch", extra={
                        "batch_size": len(texts),
                    })
                    try:
                        wavs_list, sr = model.generate_voice_clone(
                            text=texts,
                            ref_audio=ref_audio,
                            ref_text=ref_text,
                            language=languages,
                            x_vector_only_mode=x_vector_only_mode,
                        )

                        results = []
                        for wav in wavs_list:
                            buf = io.BytesIO()
                            sf.write(buf, wav, sr, format="WAV")
                            buf.seek(0)
                            results.append(buf.read())

                        ops_log.end(op, extra={"sample_rate": sr})
                        return results, sr
                    except Exception as e:
                        ops_log.fail(op, str(e))
                        raise
            finally:
                self._mark_released(VOICE_CLONE_MODEL)

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
