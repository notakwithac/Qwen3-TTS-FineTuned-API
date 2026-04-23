# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import contextlib
import io
import logging
import time
import random
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import modeling_utils as hf_modeling_utils
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

logger = logging.getLogger(__name__)

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


@dataclass
class InferenceOptimizationConfig:
    matmul_precision: Optional[str] = None
    compile_enabled: bool = False
    compile_mode: str = "reduce-overhead"
    compile_fullgraph: bool = False
    enable_batched_tokenization: bool = True


def _normalize_single_device_map(load_kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[Dict[str, Union[str, torch.device, int]]]]:
    normalized_kwargs = dict(load_kwargs)
    device_map = normalized_kwargs.get("device_map")

    if isinstance(device_map, dict):
        return normalized_kwargs, None

    if isinstance(device_map, str) and device_map == "auto":
        return normalized_kwargs, None

    if device_map is None:
        return normalized_kwargs, None

    target_device_map: Optional[Dict[str, Union[str, torch.device, int]]] = None
    if isinstance(device_map, (str, torch.device, int)):
        target_device_map = {"": device_map}
    else:
        return normalized_kwargs, None

    normalized_kwargs["device_map"] = target_device_map
    return normalized_kwargs, target_device_map


@contextlib.contextmanager
def _skip_dispatch_for_single_device(single_device_map: Optional[Dict[str, Union[str, torch.device, int]]]):
    if single_device_map is None:
        yield
        return

    original_dispatch_model = hf_modeling_utils.dispatch_model

    def _dispatch_model_noop(model, **kwargs):
        return model

    hf_modeling_utils.dispatch_model = _dispatch_model_noop
    try:
        yield
    finally:
        hf_modeling_utils.dispatch_model = original_dispatch_model


def _is_meta_tensor_copy_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cannot copy out of meta tensor" in message and "to_empty()" in message


def _coerce_device(device: Union[str, torch.device, int]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def _move_single_tensor_to_device(
    model: nn.Module,
    tensor_name: str,
    tensor: torch.Tensor,
    target_device: torch.device,
    *,
    is_parameter: bool,
) -> None:
    if "." in tensor_name:
        module_name, leaf_name = tensor_name.rsplit(".", 1)
        module = model.get_submodule(module_name)
    else:
        module = model
        leaf_name = tensor_name

    moved_tensor = tensor.detach().to(target_device)
    if is_parameter:
        module._parameters[leaf_name] = nn.Parameter(moved_tensor, requires_grad=tensor.requires_grad)
    else:
        module._buffers[leaf_name] = moved_tensor


def _align_single_device_model_tensors(
    model: nn.Module,
    single_device_map: Optional[Dict[str, Union[str, torch.device, int]]],
) -> None:
    if single_device_map is None:
        return
    if not hasattr(model, "named_parameters") or not hasattr(model, "named_buffers"):
        return

    target_device = _coerce_device(single_device_map[""])
    meta_tensors: list[str] = []

    for name, param in model.named_parameters(recurse=True, remove_duplicate=False):
        if param is None:
            continue
        if param.device.type == "meta":
            meta_tensors.append(name)
            continue
        if param.device != target_device:
            _move_single_tensor_to_device(model, name, param, target_device, is_parameter=True)

    for name, buffer in model.named_buffers(recurse=True, remove_duplicate=False):
        if buffer is None:
            continue
        if buffer.device.type == "meta":
            meta_tensors.append(name)
            continue
        if buffer.device != target_device:
            _move_single_tensor_to_device(model, name, buffer, target_device, is_parameter=False)

    if meta_tensors:
        preview = ", ".join(meta_tensors[:6])
        raise RuntimeError(
            f"Single-device load left meta tensors after weight materialization: {preview}"
        )


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()
    """

    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        processor,
        generate_defaults: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[InferenceOptimizationConfig] = None,
    ):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}
        self.optimization_config = optimization_config or InferenceOptimizationConfig()
        self._compiled_generate = None
        self._compile_failure: Optional[str] = None
        self._matmul_precision_applied = False

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self._apply_runtime_optimizations()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        optimization_config = InferenceOptimizationConfig(
            matmul_precision=kwargs.pop("matmul_precision", None),
            compile_enabled=bool(kwargs.pop("compile_enabled", False)),
            compile_mode=kwargs.pop("compile_mode", "reduce-overhead"),
            compile_fullgraph=bool(kwargs.pop("compile_fullgraph", False)),
            enable_batched_tokenization=bool(kwargs.pop("enable_batched_tokenization", True)),
        )

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        load_kwargs, single_device_map = _normalize_single_device_map(kwargs)

        try:
            model = AutoModel.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
        except Exception as exc:
            if single_device_map is None or not _is_meta_tensor_copy_error(exc):
                raise
            logger.warning(
                "Single-device load hit meta-tensor dispatch path for %s; retrying with safe dispatch bypass.",
                pretrained_model_name_or_path,
            )
            with _skip_dispatch_for_single_device(single_device_map):
                model = AutoModel.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )
        if single_device_map is not None:
            _align_single_device_model_tensors(model, single_device_map)

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        generate_defaults = model.generate_config
        return cls(
            model=model,
            processor=processor,
            generate_defaults=generate_defaults,
            optimization_config=optimization_config,
        )

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _apply_runtime_optimizations(self) -> None:
        self._maybe_set_matmul_precision()
        if self.optimization_config.compile_enabled:
            self._ensure_compiled_generate()

    def _maybe_set_matmul_precision(self) -> None:
        precision = self.optimization_config.matmul_precision
        if not precision or self._matmul_precision_applied:
            return
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if setter is None:
            logger.info("torch.set_float32_matmul_precision unavailable; skipping matmul precision change.")
            return
        try:
            setter(precision)
            self._matmul_precision_applied = True
            logger.info("Enabled float32 matmul precision mode: %s", precision)
        except Exception as exc:
            logger.warning("Failed to set float32 matmul precision to %s: %s", precision, exc)

    def _ensure_compiled_generate(self) -> Any:
        if self._compiled_generate is not None:
            return self._compiled_generate
        if self._compile_failure is not None:
            return self.model.generate
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            self._compile_failure = "torch.compile unavailable"
            logger.info("torch.compile unavailable; using eager generation path.")
            return self.model.generate
        try:
            self._compiled_generate = compile_fn(
                self.model.generate,
                mode=self.optimization_config.compile_mode,
                fullgraph=self.optimization_config.compile_fullgraph,
            )
            logger.info(
                "Compiled Qwen3 TTS generate path with mode=%s fullgraph=%s",
                self.optimization_config.compile_mode,
                self.optimization_config.compile_fullgraph,
            )
        except Exception as exc:
            self._compile_failure = str(exc)
            logger.warning("Failed to compile Qwen3 TTS generate path; falling back to eager mode. %s", exc)
            return self.model.generate
        return self._compiled_generate

    def _run_generate(self, **kwargs):
        generate_fn = self.model.generate
        if self.optimization_config.compile_enabled:
            generate_fn = self._ensure_compiled_generate()
        try:
            return generate_fn(**kwargs)
        except Exception as exc:
            if (
                self.optimization_config.compile_enabled
                and generate_fn is not self.model.generate
            ):
                self._compile_failure = str(exc)
                self._compiled_generate = None
                logger.warning(
                    "Compiled Qwen3 TTS generate path failed at runtime; retrying eagerly. %s",
                    exc,
                )
                return self.model.generate(**kwargs)
            raise

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages (List[str]): Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the Instruct model.

        Args:
            speakers (List[Optional[str]]): Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        load_started_at = time.monotonic()
        if self._is_url(x):
            # Truncate URL for logging (hide sensitive tokens but show host/path)
            from urllib.parse import urlparse as _urlparse
            _parsed = _urlparse(x)
            _safe_url = f"{_parsed.scheme}://{_parsed.netloc}{_parsed.path}"
            if _parsed.query:
                _safe_url += f"?<{len(_parsed.query)} chars>"
            logger.info(f"Downloading ref audio from: {_safe_url}")
            
            max_retries = 3
            last_err = None
            for attempt in range(max_retries):
                try:
                    # Provide a browser-like User-Agent to avoid 403 Forbidden on some hosts
                    req = urllib.request.Request(
                        x, 
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        audio_bytes = resp.read()
                    logger.info(
                        "Ref audio downloaded OK: %d bytes from %s in %.3fs",
                        len(audio_bytes),
                        _safe_url,
                        time.monotonic() - load_started_at,
                    )
                    with io.BytesIO(audio_bytes) as f:
                        audio, sr = sf.read(f, dtype="float32", always_2d=False)
                    return audio.astype(np.float32), int(sr)
                except Exception as e:
                    last_err = e
                    # Log detailed error info for HTTP errors
                    if isinstance(e, urllib.error.HTTPError):
                        resp_headers = dict(e.headers) if e.headers else {}
                        logger.error(
                            f"Audio download HTTP {e.code} from {_safe_url} "
                            f"(attempt {attempt+1}/{max_retries}). "
                            f"Response headers: {resp_headers}"
                        )
                    else:
                        logger.error(f"Audio download error from {_safe_url} (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                    
                    # Retry on common transient errors: 403 (sometimes rate limit), 429 (Too Many Requests), 5xx (Server Error)
                    is_retryable = False
                    if isinstance(e, urllib.error.HTTPError):
                        if e.code in (403, 429, 500, 502, 503, 504):
                            is_retryable = True
                    elif isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError)):
                        is_retryable = True
                    
                    if is_retryable and attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + random.random()
                        logger.warning(f"Audio download attempt {attempt+1} failed: {e}. Retrying in {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                        continue
                    raise last_err
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
            logger.info(
                "Loaded ref audio from base64 payload in %.3fs",
                time.monotonic() - load_started_at,
            )
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)
            logger.info(
                "Loaded ref audio from local path in %.3fs: %s",
                time.monotonic() - load_started_at,
                x,
            )

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        _local_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        
        for a in items:
            if isinstance(a, str):
                if a not in _local_cache:
                    _local_cache[a] = self._load_audio_to_np(a)
                out.append(_local_cache[a])
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (a[0], a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    @staticmethod
    def _expand_to_batch(values: List[Any], batch_size: int, field_name: str) -> List[Any]:
        if len(values) == 1 and batch_size > 1:
            return values * batch_size
        if len(values) != batch_size:
            raise ValueError(f"Batch size mismatch: {field_name}={len(values)}, expected={batch_size}")
        return values

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        if not texts:
            return []

        if not self.optimization_config.enable_batched_tokenization:
            input_ids = []
            for text in texts:
                input = self.processor(text=text, return_tensors="pt", padding=True)
                input_id = input["input_ids"].to(self.device)
                input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
                input_ids.append(input_id)
            return input_ids

        unique_texts = list(dict.fromkeys(texts))
        batch = self.processor(text=unique_texts, return_tensors="pt", padding=True)
        input_ids_batch = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        by_text: Dict[str, torch.Tensor] = {}
        for idx, text in enumerate(unique_texts):
            row = input_ids_batch[idx]
            if attention_mask is not None:
                valid_tokens = int(attention_mask[idx].sum().item())
                row = row[:valid_tokens]
            by_text[text] = row.unsqueeze(0)
        return [by_text[text] for text in texts]

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config.json if present.
          - Otherwise, fall back to the hard defaults for common sampling args.
          - `max_new_tokens` is only forwarded when the caller or checkpoint sets it.

        Args:
            do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
                Common generation parameters.
            **kwargs:
                Other arguments forwarded to model.generate().

        Returns:
            Dict[str, Any]: Final kwargs to pass into model.generate().
        """
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
        )
        if max_new_tokens is not None:
            merged["max_new_tokens"] = max_new_tokens
        elif "max_new_tokens" in self.generate_defaults:
            merged["max_new_tokens"] = self.generate_defaults["max_new_tokens"]
        return merged

    @staticmethod
    def _generated_token_count(audio_codes: Any) -> int:
        if audio_codes is None:
            return 0
        if hasattr(audio_codes, "shape") and len(audio_codes.shape) > 0:
            return int(audio_codes.shape[0])
        try:
            return len(audio_codes)
        except TypeError:
            return 0

    @staticmethod
    def _text_preview(text: Optional[str], limit: int = 80) -> str:
        normalized = (text or "").strip().replace("\n", " ")
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    def _decode_audio_codes(self, codes_list: List[torch.Tensor]) -> Tuple[List[np.ndarray], int]:
        return self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])

    def _build_instruct_ids(self, instructs: List[Optional[str]]) -> List[Optional[torch.Tensor]]:
        normalized = [(ins or "") for ins in instructs]
        unique_non_empty = list(dict.fromkeys([ins for ins in normalized if ins]))
        tokenized = {}
        if unique_non_empty:
            tokenized_batches = self._tokenize_texts([self._build_instruct_text(ins) for ins in unique_non_empty])
            tokenized = dict(zip(unique_non_empty, tokenized_batches))
        return [None if not ins else tokenized[ins] for ins in normalized]

    def _build_ref_ids(self, ref_texts: List[Optional[str]]) -> List[Optional[torch.Tensor]]:
        normalized = [(ref or "") for ref in ref_texts]
        unique_non_empty = list(dict.fromkeys([ref for ref in normalized if ref]))
        tokenized = {}
        if unique_non_empty:
            tokenized_batches = self._tokenize_texts([self._build_ref_text(ref) for ref in unique_non_empty])
            tokenized = dict(zip(unique_non_empty, tokenized_batches))
        return [None if not ref else tokenized[ref] for ref in normalized]

    def _log_custom_voice_generation_stats(
        self,
        *,
        texts: List[str],
        speakers: List[str],
        languages: List[str],
        instructs: List[str],
        talker_codes_list: List[Any],
        gen_kwargs: Dict[str, Any],
    ) -> None:
        effective_max_new_tokens = gen_kwargs.get("max_new_tokens")
        do_sample = gen_kwargs.get("do_sample")
        subtalker_dosample = gen_kwargs.get("subtalker_dosample")

        for idx, codes in enumerate(talker_codes_list):
            actual_new_tokens = self._generated_token_count(codes)
            cap_reached = (
                effective_max_new_tokens is not None
                and actual_new_tokens >= int(effective_max_new_tokens)
            )
            logger.info(
                "CustomVoice generation stats | idx=%s | speaker=%s | language=%s | "
                "text_chars=%s | instruct_chars=%s | actual_new_tokens=%s | "
                "effective_max_new_tokens=%s | cap_reached=%s | do_sample=%s | "
                "subtalker_dosample=%s | text_preview=%r",
                idx,
                speakers[idx],
                languages[idx],
                len(texts[idx]),
                len(instructs[idx] or ""),
                actual_new_tokens,
                effective_max_new_tokens,
                cap_reached,
                do_sample,
                subtalker_dosample,
                self._text_preview(texts[idx]),
            )

    # voice clone model
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
              This is mutually exclusive with ICL.
          - x_vector_only_mode=False:
              ICL mode is enabled automatically (icl_mode=True). In this case ref_text is required,
              because the model continues/conditions on the reference text + reference speech codes.

        Batch behavior:
          - ref_audio can be a single item or a list.
          - ref_text and x_vector_only_mode can be scalars or lists.
          - If any of them are lists with length > 1, lengths must match.

        Audio input:
          - str: local wav path / URL / base64
          - (np.ndarray, sr): waveform + sampling rate

        Args:
            ref_audio:
                Reference audio(s) used to extract:
                  - ref_code via `model.speech_tokenizer.encode(...)`
                  - ref_spk_embedding via `model.extract_speaker_embedding(...)` (resampled to 24k)
            ref_text:
                Reference transcript(s). Required when x_vector_only_mode=False (ICL mode).
            x_vector_only_mode:
                Whether to use speaker embedding only. If False, ICL mode will be used.

        Returns:
            List[VoiceClonePromptItem]:
                List of prompt items that can be converted into `voice_clone_prompt` dict.

        Raises:
            ValueError:
                - If x_vector_only_mode=False but ref_text is missing.
                - If batch lengths mismatch.
        """
        prompt_started_at = time.monotonic()
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )
        
        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
            )

        normalize_started_at = time.monotonic()
        normalized = self._normalize_audio_inputs(ref_audio_list)
        normalize_seconds = time.monotonic() - normalize_started_at

        ref_wavs_for_code: List[np.ndarray] = []
        ref_sr_for_code: List[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        code_started_at = time.monotonic()
        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])
        code_seconds = time.monotonic() - code_started_at

        items: List[VoiceClonePromptItem] = []
        speaker_embed_started_at = time.monotonic()
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(y=wav_resample.astype(np.float32), 
                                           orig_sr=int(sr), 
                                           target_sr=self.model.speaker_encoder_sample_rate)

            spk_emb = self.model.extract_speaker_embedding(audio=wav_resample,
                                                           sr=self.model.speaker_encoder_sample_rate)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        speaker_embed_seconds = time.monotonic() - speaker_embed_started_at
        logger.info(
            "Voice clone prompt built: refs=%d unique_input_audio=%d normalize_audio=%.3fs code_encode=%.3fs speaker_embed=%.3fs total=%.3fs sample_rates=%s",
            len(ref_audio_list),
            len({id(wav) for wav, _ in normalized}),
            normalize_seconds,
            code_seconds,
            speaker_embed_seconds,
            time.monotonic() - prompt_started_at,
            sorted(set(ref_sr_for_code)),
        )
        return items

    @torch.inference_mode()
    def warmup_for_inference(
        self,
        *,
        mode: str,
        text: str = "Warm up.",
        language: str = "Auto",
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
        ref_audio: Optional[AudioLike] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        max_new_tokens: int = 32,
        **kwargs,
    ) -> None:
        warmup_kwargs = dict(kwargs)
        warmup_kwargs.setdefault("max_new_tokens", max_new_tokens)
        if mode == "custom_voice":
            if speaker is None:
                supported = self.get_supported_speakers()
                if not supported:
                    raise ValueError("speaker is required for custom_voice warmup when the model has no supported speaker list.")
                speaker = supported[0]
            self.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                instruct=instruct,
                **warmup_kwargs,
            )
            return
        if mode == "voice_design":
            self.generate_voice_design(
                text=text,
                instruct=instruct or "",
                language=language,
                **warmup_kwargs,
            )
            return
        if mode == "voice_clone":
            if ref_audio is None:
                raise ValueError("ref_audio is required for voice_clone warmup.")
            self.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
                **warmup_kwargs,
            )
            return
        raise ValueError(f"Unsupported warmup mode: {mode}")

    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # voice clone model
    @torch.inference_mode()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Voice clone speech using the Base model.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - `VoiceClonePromptItem` returned by `create_voice_clone_prompt`, OR
          - a list of `VoiceClonePromptItem` returned by `create_voice_clone_prompt`.
        
        `ref_audio` Supported forms:
        - str: wav path / URL / base64 audio string
        - (np.ndarray, sr): waveform + sampling rate
        - list of the above

        Input flexibility:
          - text/language can be scalar or list.
          - prompt can be single or batch.
          - If batch mode (len(text)>1), lengths must match.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            ref_audio:
                Reference audio(s) for prompt building. Required if voice_clone_prompt is not provided.
            ref_text:
                Reference text(s) used for ICL mode (required when x_vector_only_mode=False).
            x_vector_only_mode:
                If True, only speaker embedding is used (ignores ref_text/ref_code).
                If False, ICL mode is used automatically.
            voice_clone_prompt:
                list[VoiceClonePromptItem] from `create_voice_clone_prompt`.
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If batch sizes mismatch or required prompt inputs are missing.
        """
        generation_started_at = time.monotonic()
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )
        
        texts = self._ensure_list(text)
        languages = self._expand_to_batch(
            self._ensure_list(language) if isinstance(language, list) else ([language] if language is not None else ["Auto"]),
            len(texts),
            "language",
        )

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        tokenize_started_at = time.monotonic()
        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])
        tokenize_seconds = time.monotonic() - tokenize_started_at

        ref_id_started_at = time.monotonic()
        ref_ids = self._build_ref_ids(ref_texts_for_ids) if ref_texts_for_ids is not None else None
        ref_id_seconds = time.monotonic() - ref_id_started_at

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        model_generate_started_at = time.monotonic()
        talker_codes_list, _ = self._run_generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )
        model_generate_seconds = time.monotonic() - model_generate_started_at

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        decode_started_at = time.monotonic()
        wavs_all, fs = self._decode_audio_codes(codes_for_decode)
        decode_seconds = time.monotonic() - decode_started_at

        wavs_out: List[np.ndarray] = []
        CODEC_FRAME_RATE = 12  # tokens per second — Qwen3-TTS open source decoder

        trim_started_at = time.monotonic()
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                ref_duration_samples = int(ref_len / CODEC_FRAME_RATE * fs)
                logger.info(
                    "Voice clone trim diagnostic: ref_len=%d tokens implied_duration=%.3fs fs=%d output_samples=%d",
                    ref_len,
                    ref_len / CODEC_FRAME_RATE,
                    fs,
                    wav.shape[0],
                )
                cut = min(ref_duration_samples, wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        logger.info(
            "Voice clone model stages: texts=%d tokenize=%.3fs ref_ids=%.3fs generate=%.3fs decode=%.3fs trim=%.3fs total=%.3fs max_new_tokens=%s do_sample=%s",
            len(texts),
            tokenize_seconds,
            ref_id_seconds,
            model_generate_seconds,
            decode_seconds,
            time.monotonic() - trim_started_at,
            time.monotonic() - generation_started_at,
            gen_kwargs.get("max_new_tokens"),
            gen_kwargs.get("do_sample"),
        )
        return wavs_out, fs

    # voice design model
    @torch.inference_mode()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )
        
        texts = self._ensure_list(text)
        languages = self._expand_to_batch(
            self._ensure_list(language) if isinstance(language, list) else ([language] if language is not None else ["Auto"]),
            len(texts),
            "language",
        )
        instructs = self._expand_to_batch(self._ensure_list(instruct), len(texts), "instruct")

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])
        instruct_ids = self._build_instruct_ids(instructs)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self._run_generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self._decode_audio_codes(talker_codes_list)
        return wavs, fs

    # custom voice model
    @torch.inference_mode()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id, optionally controlled by instruction text.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            instruct:
                Optional instruction(s). If None, treated as empty (no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If any speaker/language is unsupported or batch sizes mismatch.
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._expand_to_batch(
            self._ensure_list(language) if isinstance(language, list) else ([language] if language is not None else ["Auto"]),
            len(texts),
            "language",
        )
        speakers = self._expand_to_batch(self._ensure_list(speaker), len(texts), "speaker")
        if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._expand_to_batch(
            self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] if instruct is not None else [""]),
            len(texts),
            "instruct",
        )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])
        instruct_ids = self._build_instruct_ids(instructs)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self._run_generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        self._log_custom_voice_generation_stats(
            texts=texts,
            speakers=speakers,
            languages=languages,
            instructs=instructs,
            talker_codes_list=talker_codes_list,
            gen_kwargs=gen_kwargs,
        )

        wavs, fs = self._decode_audio_codes(talker_codes_list)
        return wavs, fs


    def get_supported_speakers(self) -> Optional[List[str]]:
        """
        List supported speaker names for the current model.

        This is a convenience wrapper around `model.get_supported_speakers()`.
        If the underlying model does not expose speaker constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported speaker names (lowercased), if available.
                - None if the model does not provide supported speakers.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)


    def get_supported_languages(self) -> Optional[List[str]]:
        """
        List supported language names for the current model.

        This is a convenience wrapper around `model.get_supported_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
