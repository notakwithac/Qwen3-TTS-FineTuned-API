from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
import zipfile
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

_whisper_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
DATASET_SAMPLE_MIN_SEC = 1.2
DATASET_SAMPLE_TARGET_SEC = 8.0
DATASET_SAMPLE_MAX_SEC = 10.0
DATASET_MERGE_GAP_SEC = 0.75
DATASET_MAX_SENTENCES_PER_CLIP = 2
DATASET_TARGET_LUFS = -20.0
DATASET_TP_DB = -1.5

SENTENCE_PUNCT_RE = re.compile(r"[.!?]+")
DIGIT_RE = re.compile(r"\d+")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _number_to_words(number: int) -> str:
    under_20 = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

    def _inner(n: int) -> str:
        if n < 20:
            return under_20[n]
        if n < 100:
            head = tens[n // 10]
            tail = n % 10
            return head if tail == 0 else f"{head} {under_20[tail]}"
        if n < 1000:
            head = f"{under_20[n // 100]} hundred"
            tail = n % 100
            return head if tail == 0 else f"{head} {_inner(tail)}"
        if n < 1_000_000:
            head = f"{_inner(n // 1000)} thousand"
            tail = n % 1000
            return head if tail == 0 else f"{head} {_inner(tail)}"
        if n < 1_000_000_000:
            head = f"{_inner(n // 1_000_000)} million"
            tail = n % 1_000_000
            return head if tail == 0 else f"{head} {_inner(tail)}"
        head = f"{_inner(n // 1_000_000_000)} billion"
        tail = n % 1_000_000_000
        return head if tail == 0 else f"{head} {_inner(tail)}"

    return _inner(number)


def _replace_numbers(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        try:
            return _number_to_words(int(match.group(0)))
        except Exception:
            return match.group(0)

    return DIGIT_RE.sub(_repl, text)


def _normalize_transcript_text(text: str) -> str:
    cleaned = _replace_numbers((text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;!?])([^\s])", r"\1 \2", cleaned)
    cleaned = cleaned.strip(" ,")
    if not cleaned:
        return ""

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]
    if not parts:
        parts = [cleaned]
    normalized_parts: list[str] = []
    for part in parts:
        part = re.sub(r"[,;:]+$", "", part).strip()
        if not part:
            continue
        part = part[0].upper() + part[1:] if part else part
        if part[-1] not in ".!?":
            part = f"{part}."
        normalized_parts.append(part)
    return " ".join(normalized_parts).strip()


def _sentence_units(text: str) -> int:
    matches = SENTENCE_PUNCT_RE.findall(text or "")
    return len(matches)


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if not parts:
        return [text.strip()]
    out: list[str] = []
    for part in parts:
        if part[-1] not in ".!?":
            part = f"{part}."
        out.append(part)
    return out


def _group_sentences(sentences: list[str], parts: int, max_per_clip: int = DATASET_MAX_SENTENCES_PER_CLIP) -> list[str]:
    if parts <= 1:
        return [" ".join(sentences).strip()] if sentences else []
    if not sentences:
        return []

    target_parts = max(parts, math.ceil(len(sentences) / max_per_clip))
    grouped: list[list[str]] = [[] for _ in range(target_parts)]

    sentence_idx = 0
    for group_idx in range(target_parts):
        remaining_sentences = len(sentences) - sentence_idx
        remaining_groups = target_parts - group_idx
        take = max(1, math.ceil(remaining_sentences / remaining_groups))
        take = min(take, max_per_clip)
        grouped[group_idx].extend(sentences[sentence_idx:sentence_idx + take])
        sentence_idx += take
        if sentence_idx >= len(sentences):
            break

    progress_made = True
    while sentence_idx < len(sentences) and progress_made:
        progress_made = False
        for group in grouped:
            if len(group) < max_per_clip and sentence_idx < len(sentences):
                group.append(sentences[sentence_idx])
                sentence_idx += 1
                progress_made = True

    return [" ".join(group).strip() for group in grouped if group]


def _append_filename_suffix(filename: str, suffix: str) -> str:
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem = filename
        ext = ""
    return f"{stem}{suffix}.{ext}" if ext else f"{stem}{suffix}"


def _normalize_audio_for_dataset_sync(audio_bytes: bytes) -> bytes:
    filters = [
        "silenceremove=start_periods=1:start_duration=0.08:start_threshold=-45dB",
        "silenceremove=stop_periods=1:stop_duration=0.12:stop_threshold=-45dB",
        "aformat=channel_layouts=mono",
        f"loudnorm=I={DATASET_TARGET_LUFS}:LRA=7:TP={DATASET_TP_DB}",
        "aresample=24000",
    ]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-af",
        ",".join(filters),
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            cmd,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode != 0 or not process.stdout:
            if process.stderr:
                log.warning("Dataset audio normalize failed: %s", process.stderr.decode("utf-8", errors="ignore"))
            return audio_bytes
        return process.stdout
    except Exception as exc:
        log.warning("Dataset audio normalize exception: %s", exc)
        return audio_bytes


def _join_segment_text(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return ""
    parts: list[str] = []
    prev_end: float | None = None
    for seg in segments:
        raw = (seg.get("text") or "").strip()
        if not raw:
            continue
        text = re.sub(r"\s+", " ", raw).strip()
        if prev_end is not None and parts:
            gap = max(0.0, float(seg.get("start", 0.0) or 0.0) - prev_end)
            # For dataset work we strongly prefer hard sentence boundaries over commas.
            # Commas are a common failure mode for ASR: they "mash" two unrelated
            # sentences together (exactly the issue we want to avoid when trimming
            # clips to ~2 sentences). Treat moderate gaps as sentence ends.
            if gap >= 0.45:
                parts[-1] = parts[-1].rstrip(",.;!?") + "."
        parts.append(text.strip(" ,"))
        prev_end = float(seg.get("end", 0.0) or 0.0)
    return _normalize_transcript_text(" ".join(parts))


def _safe_transcribe(model: Any, audio: Any, **kwargs) -> dict[str, Any]:
    """
    Call `model.transcribe(...)` with only the kwargs that the installed whisperx
    version supports. This keeps us compatible across whisperx releases / backends.
    """
    import inspect

    try:
        sig = inspect.signature(model.transcribe)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        filtered = kwargs
    return model.transcribe(audio, **filtered)


def _segmented_filename(filename: str, segment_index: int, total_segments: int) -> str:
    if total_segments <= 1:
        return filename
    suffix = f"__seg_{segment_index:03d}"
    return _append_filename_suffix(filename, suffix)


def _chunk_segments(
    segments: list[dict[str, Any]],
    *,
    main_speaker: str,
) -> list[list[dict[str, Any]]]:
    speaker_segments: list[dict[str, Any]] = []
    for seg in segments:
        if seg.get("speaker") != main_speaker:
            continue
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        if end <= start or not text:
            continue
        speaker_segments.append({"start": start, "end": end, "text": text})

    if not speaker_segments:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_start = 0.0
    current_end = 0.0
    current_sentence_units = 0

    for seg in speaker_segments:
        if not current:
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            current_sentence_units = _sentence_units(seg["text"])
            continue

        gap = max(0.0, seg["start"] - current_end)
        proposed_duration = seg["end"] - current_start
        seg_sentence_units = _sentence_units(seg["text"])
        proposed_sentence_units = current_sentence_units + seg_sentence_units

        should_merge = False
        if (
            gap <= DATASET_MERGE_GAP_SEC
            and proposed_duration <= DATASET_SAMPLE_TARGET_SEC
            and proposed_sentence_units <= DATASET_MAX_SENTENCES_PER_CLIP
        ):
            should_merge = True
        elif (
            proposed_duration <= DATASET_SAMPLE_MAX_SEC
            and proposed_sentence_units <= DATASET_MAX_SENTENCES_PER_CLIP
            and gap <= (DATASET_MERGE_GAP_SEC * 0.5)
        ):
            should_merge = True

        if should_merge:
            current.append(seg)
            current_end = seg["end"]
            current_sentence_units = proposed_sentence_units
            continue

        chunks.append(current)
        current = [seg]
        current_start = seg["start"]
        current_end = seg["end"]
        current_sentence_units = seg_sentence_units

    if current:
        chunks.append(current)

    return chunks


def _split_chunk_by_duration(chunk: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not chunk:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_start = 0.0
    current_end = 0.0

    for seg in chunk:
        if not current:
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        proposed_duration = seg["end"] - current_start
        current_duration = current_end - current_start
        gap = max(0.0, seg["start"] - current_end)

        should_merge = False
        if proposed_duration <= DATASET_SAMPLE_TARGET_SEC:
            should_merge = True
        elif proposed_duration <= DATASET_SAMPLE_MAX_SEC and (
            current_duration < DATASET_SAMPLE_MIN_SEC or gap <= DATASET_MERGE_GAP_SEC
        ):
            should_merge = True

        if should_merge:
            current.append(seg)
            current_end = seg["end"]
            continue

        groups.append(current)
        current = [seg]
        current_start = seg["start"]
        current_end = seg["end"]

    if current:
        groups.append(current)

    return groups


def _build_split_results(
    filename: str,
    wav_bytes: bytes,
    prompt_id: str,
    segments: list[dict[str, Any]],
    *,
    main_speaker: str,
):
    from pydub import AudioSegment

    audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    chunks = _chunk_segments(segments, main_speaker=main_speaker)
    if not chunks:
        return []

    results: list[tuple[str, bytes, str, bool, str, str | None]] = []
    for idx, chunk in enumerate(chunks):
        duration_groups = _split_chunk_by_duration(chunk)
        for part_idx, part_segments in enumerate(duration_groups):
            start_ms = max(0, int(part_segments[0]["start"] * 1000) - 100)
            end_ms = min(len(audio_segment), int(part_segments[-1]["end"] * 1000) + 100)
            cropped = audio_segment[start_ms:end_ms]
            transcript = _join_segment_text(part_segments)
            if len(cropped) <= 0 or not transcript:
                continue

            out_buf = io.BytesIO()
            cropped.export(out_buf, format="wav")
            normalized_wav = _normalize_audio_for_dataset_sync(out_buf.getvalue())
            clip_name = _segmented_filename(filename, idx, len(chunks))
            if len(duration_groups) > 1:
                clip_name = _append_filename_suffix(clip_name, f"__part_{part_idx:03d}")
            results.append((clip_name, normalized_wav, prompt_id, True, transcript, None))
    return results


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda failed with error out of memory" in message


def _clear_torch_cuda_cache() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_cuda_free_memory_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        mem_get_info = getattr(torch.cuda, "mem_get_info", None)
        if mem_get_info is None:
            return None

        try:
            free_bytes, _total_bytes = mem_get_info(0)
        except TypeError:
            free_bytes, _total_bytes = mem_get_info()
        return max(float(free_bytes), 0.0) / (1024 ** 3)
    except Exception:
        return None


def _dataset_whisper_min_free_gb(model_name: str) -> float:
    override = (os.environ.get("DATASET_PREP_CUDA_MIN_FREE_GB", "") or "").strip()
    if override:
        try:
            return max(float(override), 0.0)
        except ValueError:
            log.warning("Ignoring invalid DATASET_PREP_CUDA_MIN_FREE_GB=%r", override)

    normalized = (model_name or "").strip().lower()
    thresholds = {
        "tiny": 1.0,
        "base": 1.5,
        "small": 2.5,
        "medium": 4.0,
        "turbo": 6.0,
        "large": 10.0,
        "large-v2": 10.0,
        "large-v3": 10.0,
    }
    return thresholds.get(normalized, 6.0)


def _resolve_dataset_prep_device(configured_device: str, model_name: str) -> str:
    import torch

    if configured_device in {"cpu", "cuda"}:
        return configured_device
    if not torch.cuda.is_available():
        return "cpu"

    free_gb = _get_cuda_free_memory_gb()
    min_free_gb = _dataset_whisper_min_free_gb(model_name)
    if free_gb is not None and free_gb < min_free_gb:
        log.info(
            "Dataset prep auto-selected CPU for WhisperX %s: free CUDA memory %.1f GB below %.1f GB threshold.",
            model_name,
            free_gb,
            min_free_gb,
        )
        return "cpu"
    return "cuda"


def dataset_manifest_key(book_id: str, character_id: str, job_id: str) -> str:
    safe_book = (book_id or "unknown-book").strip() or "unknown-book"
    safe_character = (character_id or "unknown-character").strip() or "unknown-character"
    return f"datasets/{safe_book}/items/{safe_character}_{job_id}_manifest.json"


def dataset_zip_key(book_id: str, character_id: str, job_id: str) -> str:
    safe_book = (book_id or "unknown-book").strip() or "unknown-book"
    safe_character = (character_id or "unknown-character").strip() or "unknown-character"
    return f"datasets/{safe_book}/dataset_{safe_character}_{job_id}.zip"


def resolve_storage_key(storage, ref: str) -> str:
    value = (ref or "").strip()
    if not value:
        raise ValueError("storage reference cannot be empty")

    if value.startswith("s3://"):
        parts = value[5:].split("/", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Invalid s3 url: {value}")
        return parts[1]

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlsplit(value)
        path = parsed.path.lstrip("/")
        bucket_prefix = f"{storage.bucket}/"
        if path.startswith(bucket_prefix):
            return path[len(bucket_prefix):]
        endpoint = storage.endpoint_url.rstrip("/")
        object_prefix = endpoint.split("://", 1)[-1]
        if parsed.netloc == object_prefix and path.startswith(bucket_prefix):
            return path[len(bucket_prefix):]
        return path

    return value


async def _download_bytes(storage, ref: str) -> bytes:
    key = resolve_storage_key(storage, ref)
    return await asyncio.to_thread(storage.download_bytes, key)


async def _upload_bytes(
    storage,
    data: bytes,
    key: str,
    *,
    content_type: str,
    metadata: dict[str, str] | None = None,
) -> str:
    return await asyncio.to_thread(
        storage.upload_bytes,
        data,
        key,
        content_type,
        metadata,
    )


async def _object_exists(storage, key: str) -> bool:
    return await asyncio.to_thread(storage.object_exists, key)


async def _load_json(storage, key: str) -> Any:
    raw = await asyncio.to_thread(storage.download_bytes, key)
    return json.loads(raw.decode("utf-8"))


async def apply_audio_fx(
    audio_bytes: bytes,
    amplitude: float = 1.0,
    speed: float = 1.0,
    pitch_shift: float = 0.0,
) -> bytes:
    if amplitude == 1.0 and speed == 1.0 and pitch_shift == 0.0:
        return audio_bytes

    filters: list[str] = []

    if amplitude != 1.0:
        filters.append(f"volume={amplitude}")

    if speed != 1.0:
        s = speed
        while s > 2.0:
            filters.append("atempo=2.0")
            s /= 2.0
        while s < 0.5:
            filters.append("atempo=0.5")
            s *= 2.0
        filters.append(f"atempo={s}")

    if pitch_shift != 0.0:
        rate_multiplier = 2 ** (pitch_shift / 12.0)
        filters.append(f"asetrate=24000*{rate_multiplier},atempo={1.0 / rate_multiplier}")

    filters.append("aresample=24000")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-af",
        ",".join(filters),
        "-f",
        "wav",
        "pipe:1",
    ]

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(input=audio_bytes)
        if process.returncode != 0:
            log.error("FFmpeg processing failed for dataset job: %s", stderr.decode("utf-8", errors="ignore"))
            return audio_bytes
        return stdout
    except Exception as exc:
        log.error("Failed to run FFmpeg for dataset job: %s", exc)
        return audio_bytes


def _get_whisper_model(model_name: str = "large-v3", device: str | None = None, compute_type: str | None = None):
    import whisperx
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    cache_key = f"model_{model_name}_{device}_{compute_type}"
    with _cache_lock:
        if cache_key not in _whisper_cache:
            log.info("Loading WhisperX model %s on %s (%s)", model_name, device, compute_type)
            _whisper_cache[cache_key] = whisperx.load_model(model_name, device, compute_type=compute_type)
        return _whisper_cache[cache_key]


def _cudnn8_available() -> bool:
    import ctypes
    import platform

    if platform.system() != "Linux":
        return True

    try:
        ctypes.CDLL("libcudnn_ops_infer.so.8")
        return True
    except OSError:
        return False


def _get_diarization_pipeline(hf_token: str | None = None, device: str | None = None):
    import inspect
    import platform
    import torch
    from whisperx.diarize import DiarizationPipeline

    if device is None:
        if platform.system() == "Windows":
            device = "cpu"
        elif torch.cuda.is_available() and _cudnn8_available():
            device = "cuda"
        else:
            device = "cpu"

    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is required for dataset diarization.")

    cache_key = f"diarize_{device}"
    with _cache_lock:
        if cache_key not in _whisper_cache:
            sig = inspect.signature(DiarizationPipeline.__init__)
            kwargs = {"device": device}
            if "token" in sig.parameters:
                kwargs["token"] = hf_token
            else:
                kwargs["use_auth_token"] = hf_token
            _whisper_cache[cache_key] = DiarizationPipeline(**kwargs)
        return _whisper_cache[cache_key]


def _get_align_model(language_code: str = "en", device: str = "cpu"):
    import whisperx

    cache_key = f"align_{language_code}_{device}"
    with _cache_lock:
        if cache_key not in _whisper_cache:
            log.info("Loading WhisperX align model lang=%s device=%s", language_code, device)
            _whisper_cache[cache_key] = whisperx.load_align_model(language_code=language_code, device=device)
        return _whisper_cache[cache_key]


def _validate_and_crop_audio_sync(
    audio_items: list[tuple[str, bytes, str]],
    char_name: str,
) -> list[tuple[str, bytes, str, bool, str, str | None]]:
    import torch
    import whisperx

    try:
        try:
            from omegaconf import DictConfig, ListConfig

            torch.serialization.add_safe_globals([ListConfig, DictConfig])
        except ImportError:
            pass

        configured_device = (os.environ.get("DATASET_PREP_DEVICE", "auto") or "auto").strip().lower()
        hf_token = os.environ.get("HF_TOKEN")
        skip_validation = _env_flag("SKIP_WHISPER_VALIDATION")
        model_name = (os.environ.get("DATASET_WHISPER_MODEL", "turbo") or "turbo").strip()
        device = _resolve_dataset_prep_device(configured_device, model_name)
        compute_type = "float16" if device == "cuda" else "int8"

        try:
            model = _get_whisper_model(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            if device == "cuda" and _is_cuda_oom_error(exc):
                log.warning(
                    "WhisperX model load hit CUDA OOM for %s; retrying dataset prep on CPU.",
                    char_name,
                    exc_info=True,
                )
                _clear_torch_cuda_cache()
                device = "cpu"
                compute_type = "int8"
                model = _get_whisper_model(model_name, device=device, compute_type=compute_type)
            else:
                raise

        diarize_model = None
        diarization_error: str | None = None
        if skip_validation:
            diarization_error = "SKIP_WHISPER_VALIDATION enabled"
            log.info("SKIP_WHISPER_VALIDATION enabled on GPU dataset job for %s.", char_name)
        else:
            try:
                diarize_model = _get_diarization_pipeline(hf_token, device=device)
            except Exception as exc:
                if device == "cuda" and _is_cuda_oom_error(exc):
                    log.warning(
                        "Diarization load hit CUDA OOM for %s; retrying diarization on CPU.",
                        char_name,
                        exc_info=True,
                    )
                    _clear_torch_cuda_cache()
                    try:
                        diarize_model = _get_diarization_pipeline(hf_token, device="cpu")
                        diarization_error = "diarization downgraded to CPU after CUDA OOM"
                    except Exception as cpu_exc:
                        diarization_error = str(cpu_exc)
                else:
                    diarization_error = str(exc)
                if diarize_model is None:
                    log.warning(
                        "Diarization unavailable for %s; falling back to aligned full-audio transcripts: %s",
                        char_name,
                        diarization_error,
                    )

        import time as _time
        import platform

        results = []
        total_clips = len(audio_items)
        for clip_idx, (filename, wav_bytes, prompt_id) in enumerate(audio_items, 1):
            clip_start = _time.monotonic()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav_path = f.name
                f.write(wav_bytes)

            try:
                audio = whisperx.load_audio(temp_wav_path)

                t0 = _time.monotonic()
                # "turbo" models are fast but can be less stable; these decoding
                # options reduce cross-sentence bleed and improve word choice.
                # We pass them through _safe_transcribe so older whisperx builds
                # won't break if they don't support a particular knob.
                result = _safe_transcribe(
                    model,
                    audio,
                    batch_size=8,
                    language=(os.environ.get("DATASET_LANGUAGE", "en") or "en"),
                    temperature=0.0,
                    beam_size=5,
                    best_of=5,
                    condition_on_previous_text=False,
                )
                log.info(
                    "[%s] Clip %d/%d: transcription done in %.1fs — lang=%s, %d segments",
                    char_name,
                    clip_idx,
                    total_clips,
                    _time.monotonic() - t0,
                    result.get("language", "?"),
                    len(result.get("segments", [])),
                )

                align_device = device
                if platform.system() == "Windows" and align_device == "cuda":
                    align_device = "cpu"

                try:
                    model_a, metadata = _get_align_model(result["language"], align_device)
                except Exception as exc:
                    if align_device == "cuda":
                        log.warning("Alignment on CUDA failed, falling back to CPU: %s", exc)
                        model_a, metadata = _get_align_model(result["language"], "cpu")
                        align_device = "cpu"
                    else:
                        raise

                t0 = _time.monotonic()
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    align_device,
                    return_char_alignments=False,
                )
                log.info(
                    "[%s] Clip %d/%d: alignment done in %.1fs (device=%s)",
                    char_name,
                    clip_idx,
                    total_clips,
                    _time.monotonic() - t0,
                    align_device,
                )

                segments = result["segments"]
                fallback_transcript = _join_segment_text(segments)

                if diarize_model is None:
                    if fallback_transcript:
                        results.append((filename, wav_bytes, prompt_id, True, fallback_transcript, diarization_error))
                        log.info(
                            "[%s] Clip %d/%d: PASS — single-speaker fallback (%s)",
                            char_name,
                            clip_idx,
                            total_clips,
                            diarization_error or "diarization unavailable",
                        )
                    else:
                        results.append(
                            (
                                filename,
                                wav_bytes,
                                prompt_id,
                                False,
                                "",
                                (diarization_error or "diarization unavailable") + "; empty transcript after alignment",
                            )
                        )
                    continue

                try:
                    t0 = _time.monotonic()
                    diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=10)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    segments = result["segments"]
                    log.info(
                        "[%s] Clip %d/%d: diarization done in %.1fs — %d segments",
                        char_name,
                        clip_idx,
                        total_clips,
                        _time.monotonic() - t0,
                        len(segments),
                    )
                except Exception as exc:
                    reason = f"diarization failed: {exc}"
                    if fallback_transcript:
                        results.append((filename, wav_bytes, prompt_id, True, fallback_transcript, reason))
                        log.warning(
                            "[%s] Clip %d/%d: diarization failed, using single-speaker fallback: %s",
                            char_name,
                            clip_idx,
                            total_clips,
                            exc,
                        )
                    else:
                        results.append((filename, wav_bytes, prompt_id, False, "", reason))
                    continue

                speaker_durations: dict[str, float] = {}
                for seg in segments:
                    speaker = seg.get("speaker")
                    if not speaker:
                        continue
                    duration = seg.get("end", 0.0) - seg.get("start", 0.0)
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration

                if not speaker_durations:
                    if fallback_transcript:
                        results.append(
                            (
                                filename,
                                wav_bytes,
                                prompt_id,
                                True,
                                fallback_transcript,
                                "diarization produced no speaker labels",
                            )
                        )
                    else:
                        results.append((filename, wav_bytes, prompt_id, False, "", "diarization produced no speaker labels"))
                    continue

                main_speaker = max(speaker_durations.items(), key=lambda item: item[1])[0]

                split_results = _build_split_results(
                    filename,
                    wav_bytes,
                    prompt_id,
                    segments,
                    main_speaker=main_speaker,
                )
                if split_results:
                    results.extend(split_results)
                    log.info(
                        "[%s] Clip %d/%d: PASS — kept speaker %s and produced %d sample(s), total %.1fs",
                        char_name,
                        clip_idx,
                        total_clips,
                        main_speaker,
                        len(split_results),
                        _time.monotonic() - clip_start,
                    )
                else:
                    if fallback_transcript:
                        results.append(
                            (
                                filename,
                                wav_bytes,
                                prompt_id,
                                True,
                                fallback_transcript,
                                "speaker crop was empty; kept original audio",
                            )
                        )
                    else:
                        results.append((filename, wav_bytes, prompt_id, False, "", "speaker crop was empty"))
            except Exception as exc:
                log.error("Dataset validation failed for %s: %s", filename, exc, exc_info=True)
                results.append((filename, wav_bytes, prompt_id, False, "", str(exc)))
            finally:
                try:
                    os.remove(temp_wav_path)
                except OSError:
                    pass

        torch.cuda.empty_cache()
        return results
    except Exception as exc:
        log.error("Dataset batch processing crashed for %s: %s", char_name, exc, exc_info=True)
        return [(filename, wav_bytes, prompt_id, False, "", str(exc)) for filename, wav_bytes, prompt_id in audio_items]


async def validate_and_crop_audio(
    audio_items: list[tuple[str, bytes, str]],
    char_name: str,
) -> list[tuple[str, bytes, str, bool, str, str | None]]:
    return await asyncio.to_thread(_validate_and_crop_audio_sync, audio_items, char_name)


async def load_existing_prepare_result(storage, *, book_id: str, character_id: str, job_id: str) -> list[dict[str, Any]] | None:
    key = dataset_manifest_key(book_id, character_id, job_id)
    if not await _object_exists(storage, key):
        return None
    data = await _load_json(storage, key)
    return data if isinstance(data, list) else None


async def load_existing_package_result(storage, *, book_id: str, character_id: str, job_id: str) -> dict[str, str] | None:
    key = dataset_zip_key(book_id, character_id, job_id)
    if not await _object_exists(storage, key):
        return None
    return {
        "dataset_s3_key": key,
        "dataset_s3_url": storage._object_url(key),
    }


async def prepare_dataset_items(
    storage,
    *,
    book_id: str,
    character_id: str,
    character_name: str,
    job_id: str,
    ref_audio_url: str,
    ref_text: str,
    items: list[dict[str, Any]],
    amplitude: float = 1.0,
    speed: float = 1.0,
    pitch_shift: float = 0.0,
) -> list[dict[str, Any]]:
    ref_audio = await _download_bytes(storage, ref_audio_url)
    ref_audio = await apply_audio_fx(ref_audio, amplitude=amplitude, speed=speed, pitch_shift=pitch_shift)
    ref_audio = _normalize_audio_for_dataset_sync(ref_audio)
    ref_text = _normalize_transcript_text(ref_text)

    raw_audio_segments: list[tuple[str, bytes, str]] = [("ref_audio.wav", ref_audio, "ref_audio")]
    prompt_id_to_text: dict[str, str] = {"ref_audio": ref_text}

    for item in items:
        ref = item.get("s3_url") or item.get("url")
        if not ref:
            continue
        filename = str(item.get("filename") or item.get("id") or "").strip()
        if not filename:
            continue
        prompt_id = str(item.get("prompt_id") or filename)
        prompt_text = str(item.get("text") or "")
        prompt_id_to_text[prompt_id] = _normalize_transcript_text(prompt_text)

        content = await _download_bytes(storage, ref)
        content = await apply_audio_fx(content, amplitude=amplitude, speed=speed, pitch_shift=pitch_shift)
        content = _normalize_audio_for_dataset_sync(content)
        raw_audio_segments.append((filename, content, prompt_id))

    validated_segments = await validate_and_crop_audio(raw_audio_segments, character_name)

    final_train_segments: list[tuple[str, bytes]] = []
    final_transcripts: list[str] = []
    failure_reasons: list[str] = []
    for filename, wav_bytes, prompt_id, success, transcript, failure_reason in validated_segments:
        if filename == "ref_audio.wav":
            continue
        if not success or not transcript:
            if failure_reason:
                failure_reasons.append(f"{filename}: {failure_reason}")
            continue
        # Important: do NOT train on ASR output when we already know the canonical
        # text for the clip (e.g., Harvard prompts). Whisper is only used here for
        # diarization and cut points.
        canonical = prompt_id_to_text.get(prompt_id) or ""
        normalized_transcript = canonical or _normalize_transcript_text(transcript)
        if not normalized_transcript:
            if failure_reason:
                failure_reasons.append(f"{filename}: transcript empty after normalization")
            continue
        final_train_segments.append((filename, _normalize_audio_for_dataset_sync(wav_bytes)))
        final_transcripts.append(normalized_transcript)

    if not final_train_segments:
        detail = "; ".join(failure_reasons[:3]) if failure_reasons else "all clips failed validation"
        raise RuntimeError(f"No valid segments available for finetuning dataset. {detail}")

    dataset_items: list[dict[str, Any]] = []
    ref_key = f"datasets/{book_id}/items/{character_id}_{job_id}_ref_audio.wav"
    ref_s3_url = await _upload_bytes(
        storage,
        ref_audio,
        ref_key,
        content_type="audio/wav",
        metadata={
            "character_id": character_id,
            "job_id": job_id,
            "type": "dataset_item_ref",
        },
    )
    dataset_items.append(
        {
            "id": "ref_audio.wav",
            "url": storage.get_presigned_url(ref_key, expires_in=86400 * 7),
            "s3_url": ref_s3_url,
            "text": ref_text,
            "is_reference": True,
            "included": True,
        }
    )

    for idx, (filename, wav_bytes) in enumerate(final_train_segments):
        transcript = final_transcripts[idx]
        item_key = f"datasets/{book_id}/items/{character_id}_{job_id}_{filename}"
        s3_url = await _upload_bytes(
            storage,
            wav_bytes,
            item_key,
            content_type="audio/wav",
            metadata={
                "character_id": character_id,
                "job_id": job_id,
                "type": "dataset_item",
            },
        )
        dataset_items.append(
            {
                "id": filename,
                "url": storage.get_presigned_url(item_key, expires_in=86400 * 7),
                "s3_url": s3_url,
                "text": transcript,
                "is_reference": False,
                "included": True,
            }
        )

    manifest_key = dataset_manifest_key(book_id, character_id, job_id)
    await _upload_bytes(
        storage,
        json.dumps(dataset_items).encode("utf-8"),
        manifest_key,
        content_type="application/json",
        metadata={
            "character_id": character_id,
            "job_id": job_id,
            "type": "dataset_manifest",
        },
    )

    return dataset_items


def _build_dataset_zip(
    audio_segments: list[tuple[str, bytes]],
    ref_audio: bytes,
    transcripts: list[str],
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data/ref_audio.wav", ref_audio)
        jsonl_lines = []
        for (filename, wav_bytes), transcript in zip(audio_segments, transcripts):
            zf.writestr(f"data/{filename}", wav_bytes)
            jsonl_lines.append(
                json.dumps(
                    {
                        "audio": f"./data/{filename}",
                        "text": transcript,
                        "ref_audio": "./data/ref_audio.wav",
                    }
                )
            )
        zf.writestr("train.jsonl", "\n".join(jsonl_lines))
    return buf.getvalue()


async def package_dataset(
    storage,
    *,
    book_id: str,
    character_id: str,
    job_id: str,
    dataset_items: list[dict[str, Any]],
) -> dict[str, str]:
    final_train_segments: list[tuple[str, bytes]] = []
    final_transcripts: list[str] = []
    chosen_ref_audio: bytes | None = None

    for item in dataset_items:
        if item.get("included") is False:
            continue
        ref = item.get("s3_url") or item.get("url")
        if not ref:
            continue
        wav_bytes = await _download_bytes(storage, str(ref))
        wav_bytes = _normalize_audio_for_dataset_sync(wav_bytes)
        filename = str(item.get("id") or item.get("filename") or "").strip()
        transcript = _normalize_transcript_text(str(item.get("text") or "").strip())
        if not filename or not transcript:
            continue
        if item.get("is_reference") is True:
            chosen_ref_audio = wav_bytes
            continue
        final_train_segments.append((filename, wav_bytes))
        final_transcripts.append(transcript)

    if not final_train_segments:
        raise RuntimeError("No dataset items available to package.")

    if chosen_ref_audio is None:
        chosen_ref_audio = final_train_segments[-1][1]

    zip_bytes = _build_dataset_zip(final_train_segments, chosen_ref_audio, final_transcripts)
    key = dataset_zip_key(book_id, character_id, job_id)
    url = await _upload_bytes(
        storage,
        zip_bytes,
        key,
        content_type="application/zip",
        metadata={
            "character_id": character_id,
            "job_id": job_id,
            "type": "dataset_zip",
        },
    )
    return {
        "dataset_s3_key": key,
        "dataset_s3_url": url,
    }
