# coding=utf-8
"""Sarvam-Translate runtime wrapper for the FastAPI server."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch

logger = logging.getLogger(__name__)


SUPPORTED_SARVAM_LANGUAGES = {
    "Assamese",
    "Bengali",
    "Bodo",
    "Dogri",
    "Gujarati",
    "English",
    "Hindi",
    "Kannada",
    "Kashmiri",
    "Konkani",
    "Maithili",
    "Malayalam",
    "Manipuri",
    "Marathi",
    "Nepali",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Santali",
    "Sindhi",
    "Tamil",
    "Telugu",
    "Urdu",
}


@dataclass(frozen=True)
class SarvamTranslationResult:
    text: str
    model_id: str
    target_language: str
    source_language: Optional[str]
    input_tokens: int
    output_tokens: int
    max_model_tokens: int
    latency_seconds: float


class SarvamTranslateService:
    """Lazy singleton loader and generator for sarvamai/sarvam-translate.

    The model card specifies an 8k-token maximum context, so this wrapper checks
    prompt tokens plus requested output tokens before running `generate`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        max_model_tokens: int = 8192,
        default_max_new_tokens: int = 1024,
        max_new_tokens_limit: int = 2048,
        dtype: str = "auto",
        device_map: Optional[str] = None,
        backend: str = "local",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout_seconds: float = 120.0,
        gpu_controller: Any = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_model_tokens = max_model_tokens
        self.default_max_new_tokens = default_max_new_tokens
        self.max_new_tokens_limit = max_new_tokens_limit
        self.dtype = dtype
        self.device_map = device_map
        self.backend = backend
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key or os.environ.get("SARVAM_TRANSLATE_API_KEY") or "EMPTY"
        self.request_timeout_seconds = request_timeout_seconds
        self.gpu_controller = gpu_controller

        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def status(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self.is_loaded,
            "device": self.device,
            "device_map": self.device_map,
            "backend": self.backend,
            "base_url": self.base_url,
            "dtype": self.dtype,
            "max_model_tokens": self.max_model_tokens,
            "default_max_new_tokens": self.default_max_new_tokens,
            "max_new_tokens_limit": self.max_new_tokens_limit,
            "supported_languages": sorted(SUPPORTED_SARVAM_LANGUAGES),
        }

    def unload(self) -> None:
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def translate(
        self,
        *,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.01,
        do_sample: bool = True,
    ) -> SarvamTranslationResult:
        if self.backend == "openai":
            return self._translate_openai(
                text=text,
                target_language=target_language,
                source_language=source_language,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        if self.backend != "local":
            raise RuntimeError(f"Unsupported Sarvam translation backend: {self.backend}")

        tokenizer, model = self._ensure_loaded()
        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        max_new_tokens = min(max_new_tokens, self.max_new_tokens_limit)

        prompt = self._build_prompt(
            tokenizer=tokenizer,
            text=text,
            target_language=target_language,
            source_language=source_language,
        )
        model_inputs = tokenizer([prompt], return_tensors="pt", truncation=False)
        input_tokens = int(model_inputs.input_ids.shape[-1])
        if input_tokens + max_new_tokens > self.max_model_tokens:
            available = max(0, self.max_model_tokens - input_tokens)
            raise ValueError(
                "translation request exceeds Sarvam-Translate context window: "
                f"input_tokens={input_tokens}, requested_max_new_tokens={max_new_tokens}, "
                f"max_model_tokens={self.max_model_tokens}, available_new_tokens={available}"
            )

        model_inputs = model_inputs.to(model.device)
        started = time.monotonic()

        if self.gpu_controller is not None:
            self.gpu_controller.begin_inference("sarvam_translate")
        try:
            with self._generate_lock, torch.inference_mode():
                generation_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "num_return_sequences": 1,
                }
                if do_sample:
                    generation_kwargs["temperature"] = temperature
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    generation_kwargs["pad_token_id"] = eos_token_id
                generated_ids = model.generate(**model_inputs, **generation_kwargs)
        finally:
            if self.gpu_controller is not None:
                self.gpu_controller.end_inference()

        output_ids = generated_ids[0][input_tokens:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return SarvamTranslationResult(
            text=output_text,
            model_id=self.model_id,
            target_language=target_language,
            source_language=source_language,
            input_tokens=input_tokens,
            output_tokens=len(output_ids),
            max_model_tokens=self.max_model_tokens,
            latency_seconds=round(time.monotonic() - started, 3),
            )

    def _ensure_loaded(self):
        if self.is_loaded:
            return self._tokenizer, self._model

        with self._load_lock:
            if self.is_loaded:
                return self._tokenizer, self._model

            from transformers import AutoModelForCausalLM, AutoTokenizer

            load_kwargs: dict[str, Any] = {
                "low_cpu_mem_usage": True,
            }
            dtype = self._resolve_torch_dtype()
            if dtype is not None:
                load_kwargs["torch_dtype"] = dtype
            if self.device_map:
                load_kwargs["device_map"] = self.device_map

            logger.info("Loading Sarvam-Translate model %s", self.model_id)
            if self.gpu_controller is not None:
                self.gpu_controller.begin_inference("sarvam_translate_load")
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
                if not self.device_map:
                    model = model.to(self.device)
                model.eval()
            finally:
                if self.gpu_controller is not None:
                    self.gpu_controller.end_inference()

            self._tokenizer = tokenizer
            self._model = model
            return tokenizer, model

    def _translate_openai(
        self,
        *,
        text: str,
        target_language: str,
        source_language: Optional[str],
        max_new_tokens: Optional[int],
        temperature: float,
    ) -> SarvamTranslationResult:
        if not self.base_url:
            raise RuntimeError(
                "SARVAM_TRANSLATE_BACKEND=openai requires SARVAM_TRANSLATE_BASE_URL"
            )

        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        max_new_tokens = min(max_new_tokens, self.max_new_tokens_limit)
        input_tokens = self._estimate_input_tokens(text, target_language, source_language)
        if input_tokens + max_new_tokens > self.max_model_tokens:
            available = max(0, self.max_model_tokens - input_tokens)
            raise ValueError(
                "translation request exceeds Sarvam-Translate context window: "
                f"estimated_input_tokens={input_tokens}, requested_max_new_tokens={max_new_tokens}, "
                f"max_model_tokens={self.max_model_tokens}, available_new_tokens={available}"
            )

        if source_language:
            system = f"Translate the text below from {source_language} to {target_language}."
        else:
            system = f"Translate the text below to {target_language}."

        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": text},
                ],
                "temperature": temperature,
                "max_tokens": max_new_tokens,
            },
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        output_text = payload["choices"][0]["message"]["content"].strip()
        usage = payload.get("usage") or {}
        return SarvamTranslationResult(
            text=output_text,
            model_id=self.model_id,
            target_language=target_language,
            source_language=source_language,
            input_tokens=int(usage.get("prompt_tokens") or input_tokens),
            output_tokens=int(usage.get("completion_tokens") or 0),
            max_model_tokens=self.max_model_tokens,
            latency_seconds=round(time.monotonic() - started, 3),
        )

    def _estimate_input_tokens(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str],
    ) -> int:
        try:
            tokenizer, _ = self._ensure_tokenizer_only()
            prompt = self._build_prompt(
                tokenizer=tokenizer,
                text=text,
                target_language=target_language,
                source_language=source_language,
            )
            return int(tokenizer([prompt], return_tensors="pt", truncation=False).input_ids.shape[-1])
        except Exception:
            return max(1, len(text) // 3 + 64)

    def _ensure_tokenizer_only(self):
        if self._tokenizer is not None:
            return self._tokenizer, self._model
        with self._load_lock:
            if self._tokenizer is not None:
                return self._tokenizer, self._model
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            return self._tokenizer, self._model

    def _resolve_torch_dtype(self):
        if self.dtype == "auto":
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                return torch.bfloat16
            return None
        normalized = self.dtype.lower()
        if normalized in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16"}:
            return torch.float16
        if normalized in {"fp32", "float32"}:
            return torch.float32
        return None

    @staticmethod
    def _build_prompt(
        *,
        tokenizer,
        text: str,
        target_language: str,
        source_language: Optional[str],
    ) -> str:
        if source_language:
            system = f"Translate the text below from {source_language} to {target_language}."
        else:
            system = f"Translate the text below to {target_language}."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
