"""Codec target extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(codes: Any) -> np.ndarray:
    if hasattr(codes, "detach"):
        return codes.detach().cpu().numpy()
    return np.asarray(codes)


def normalize_audio_codes(encoded: Any) -> Any:
    """Extract raw audio-code tensors from common Qwen tokenizer outputs."""

    if hasattr(encoded, "audio_codes"):
        encoded = encoded.audio_codes
    if isinstance(encoded, dict) and "audio_codes" in encoded:
        encoded = encoded["audio_codes"]
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
        encoded = encoded[0]
    return encoded


def select_codec0(codes: Any) -> np.ndarray:
    """Select codec group 0 from a Qwen 16-codebook audio code sequence."""

    return select_codec_codes(codes)[..., 0]


def select_codec_codes(codes: Any) -> np.ndarray:
    """Return Qwen audio codes with shape [frames, codec_groups]."""

    array = _to_numpy(normalize_audio_codes(codes))
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("expected a single encoded audio item")
        array = array[0]
    if array.ndim != 2:
        raise ValueError("expected codes with shape [16, frames], [frames, 16], or [batch, 16, frames]")
    if array.shape[0] == 16:
        return array.T
    if array.shape[1] == 16:
        return array
    raise ValueError("expected one dimension to contain 16 codec groups")


def slice_target_span(codec0: Any, frame_span: tuple[int, int] | list[int] | None) -> np.ndarray:
    values = _to_numpy(codec0)
    if frame_span is None:
        return values
    if len(frame_span) != 2:
        raise ValueError("target frame span must contain start and end")
    start, end = int(frame_span[0]), int(frame_span[1])
    length = values.shape[-1]
    if start < 0 or end <= start or end > length:
        raise ValueError(f"invalid target frame span [{start}, {end}) for {length} frames")
    return values[..., start:end]


def slice_target_frames(codes: Any, frame_span: tuple[int, int] | list[int] | None) -> np.ndarray:
    values = _to_numpy(codes)
    if frame_span is None:
        return values
    if len(frame_span) != 2:
        raise ValueError("target frame span must contain start and end")
    start, end = int(frame_span[0]), int(frame_span[1])
    length = values.shape[0]
    if start < 0 or end <= start or end > length:
        raise ValueError(f"invalid target frame span [{start}, {end}) for {length} frames")
    return values[start:end, :]


def _call_tokenizer(tokenizer: Any, audio_path: str | Path) -> Any:
    if hasattr(tokenizer, "encode"):
        return tokenizer.encode(str(audio_path))
    if hasattr(tokenizer, "encode_audio"):
        return tokenizer.encode_audio(str(audio_path))
    if hasattr(tokenizer, "audio_to_codes"):
        return tokenizer.audio_to_codes(str(audio_path))
    if hasattr(tokenizer, "tokenize_audio"):
        return tokenizer.tokenize_audio(str(audio_path))
    if callable(tokenizer):
        return tokenizer(str(audio_path))
    raise TypeError("tokenizer must provide encode, encode_audio, audio_to_codes, tokenize_audio, or be callable")


def extract_codec0_target(
    tokenizer: Any,
    audio_path: str | Path,
    frame_span: tuple[int, int] | list[int] | None = None,
) -> np.ndarray:
    """Map desired pronunciation audio through a tokenizer and return codec group 0."""

    codes = _call_tokenizer(tokenizer, audio_path)
    return slice_target_span(select_codec0(codes), frame_span)


def extract_codec_target(
    tokenizer: Any,
    audio_path: str | Path,
    frame_span: tuple[int, int] | list[int] | None = None,
) -> np.ndarray:
    """Map desired pronunciation audio through a tokenizer and return [frames, 16] codes."""

    codes = _call_tokenizer(tokenizer, audio_path)
    return slice_target_frames(select_codec_codes(codes), frame_span)
