from __future__ import annotations

import asyncio
import audioop
import io
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import threading
import time
import wave
import zipfile
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

_whisper_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
_dataset_cuda_oom_retry_after = 0.0
DATASET_SAMPLE_MIN_SEC = 4.0
DATASET_SHORT_CLIP_MAX_SEC = 5.0   # clips below this are flagged as "short" in manifest
DATASET_CLIP_FLUSH_SEC    = 5.0    # prefer to flush earlier, but never below the hard 5s floor
DATASET_SAMPLE_TARGET_SEC = 6.0
DATASET_SAMPLE_MAX_SEC = 11.0
DATASET_MERGE_GAP_SEC = 0.35
DATASET_MAX_SENTENCES_PER_CLIP = 3
DATASET_TARGET_LUFS = -20.0
DATASET_TP_DB = -1.5
DATASET_EDGE_PADDING_MS = 100
DATASET_DIARIZATION_BATCH_MAX_SEC = 90.0
DATASET_DIARIZATION_SEPARATOR_SEC = 0.35
DATASET_BATCH_ANALYSIS_SAMPLE_RATE = 16000

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


def _ends_with_sentence_punct(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", text or ""))


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


def _wav_duration_seconds(audio_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frame_rate = wf.getframerate()
            if frame_rate <= 0:
                return 0.0
            return wf.getnframes() / float(frame_rate)
    except Exception:
        return 0.0


def _write_wav_bytes(*, frames: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
    return buf.getvalue()


def _prepare_wav_for_batch_analysis(audio_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2

    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise ValueError(f"Unsupported WAV channel count for batch analysis: {channels}")

    if sample_rate != DATASET_BATCH_ANALYSIS_SAMPLE_RATE:
        frames, _state = audioop.ratecv(
            frames,
            sample_width,
            channels,
            sample_rate,
            DATASET_BATCH_ANALYSIS_SAMPLE_RATE,
            None,
        )
        sample_rate = DATASET_BATCH_ANALYSIS_SAMPLE_RATE

    return _write_wav_bytes(
        frames=frames,
        sample_rate=sample_rate,
        sample_width=sample_width,
        channels=channels,
    )


def _prepare_batch_analysis_items(
    audio_items: list[tuple[str, bytes, str]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for filename, wav_bytes, prompt_id in audio_items:
        analysis_wav_bytes = _prepare_wav_for_batch_analysis(wav_bytes)
        prepared.append(
            {
                "filename": filename,
                "prompt_id": prompt_id,
                "wav_bytes": wav_bytes,
                "analysis_wav_bytes": analysis_wav_bytes,
                "duration_sec": _wav_duration_seconds(analysis_wav_bytes),
            }
        )
    return prepared


def _build_reference_diarization_batches(
    ref_item: dict[str, Any],
    clone_items: list[dict[str, Any]],
    *,
    max_total_duration_sec: float | None = None,
) -> list[list[dict[str, Any]]]:
    if max_total_duration_sec is None:
        max_total_duration_sec = DATASET_DIARIZATION_BATCH_MAX_SEC
    if not clone_items:
        return [[ref_item]]

    ref_duration_sec = float(ref_item.get("duration_sec", 0.0) or 0.0)
    clone_budget_sec = max(0.0, max_total_duration_sec - ref_duration_sec)
    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_duration_sec = 0.0

    for item in clone_items:
        clip_duration_sec = float(item.get("duration_sec", 0.0) or 0.0)
        if current_batch and current_duration_sec + clip_duration_sec > clone_budget_sec:
            batches.append([ref_item, *current_batch])
            current_batch = []
            current_duration_sec = 0.0

        current_batch.append(item)
        current_duration_sec += clip_duration_sec

    if current_batch:
        batches.append([ref_item, *current_batch])

    return batches


def _concatenate_batch_analysis_wavs(
    batch_items: list[dict[str, Any]],
    *,
    separator_sec: float = DATASET_DIARIZATION_SEPARATOR_SEC,
) -> tuple[bytes, list[dict[str, Any]]]:
    if not batch_items:
        return b"", []

    combined_frames: list[bytes] = []
    clip_intervals: list[dict[str, Any]] = []
    sample_rate = DATASET_BATCH_ANALYSIS_SAMPLE_RATE
    sample_width = 2
    channels = 1
    silence_frames = b"\x00" * int(separator_sec * sample_rate) * sample_width * channels
    current_offset_sec = 0.0

    for index, item in enumerate(batch_items):
        analysis_wav_bytes = item["analysis_wav_bytes"]
        with wave.open(io.BytesIO(analysis_wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            duration_sec = wf.getnframes() / float(wf.getframerate() or sample_rate)

        clip_intervals.append(
            {
                **item,
                "analysis_start_sec": current_offset_sec,
                "analysis_end_sec": current_offset_sec + duration_sec,
            }
        )
        combined_frames.append(frames)
        current_offset_sec += duration_sec

        if index < len(batch_items) - 1 and silence_frames:
            combined_frames.append(silence_frames)
            current_offset_sec += separator_sec

    return (
        _write_wav_bytes(
            frames=b"".join(combined_frames),
            sample_rate=sample_rate,
            sample_width=sample_width,
            channels=channels,
        ),
        clip_intervals,
    )


def _segment_overlap_seconds(segment: dict[str, Any], start_sec: float, end_sec: float) -> float:
    segment_start = float(segment.get("start", 0.0) or 0.0)
    segment_end = float(segment.get("end", 0.0) or 0.0)
    return max(0.0, min(segment_end, end_sec) - max(segment_start, start_sec))


def _select_dominant_speaker(segments: list[dict[str, Any]]) -> str | None:
    speaker_durations: dict[str, float] = {}
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + max(
            0.0,
            float(segment.get("end", 0.0) or 0.0) - float(segment.get("start", 0.0) or 0.0),
        )
    if not speaker_durations:
        return None
    return max(speaker_durations.items(), key=lambda item: item[1])[0]


def _select_target_speaker_for_interval(
    segments: list[dict[str, Any]],
    interval_start_sec: float,
    interval_end_sec: float,
) -> str | None:
    speaker_durations: dict[str, float] = {}
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        overlap_sec = _segment_overlap_seconds(segment, interval_start_sec, interval_end_sec)
        if overlap_sec <= 0.0:
            continue
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + overlap_sec
    if speaker_durations:
        return max(speaker_durations.items(), key=lambda item: item[1])[0]
    return None


def _slice_combined_segments_to_local_interval(
    segments: list[dict[str, Any]],
    interval_start_sec: float,
    interval_end_sec: float,
    *,
    speaker: str | None = None,
) -> list[dict[str, Any]]:
    local_segments: list[dict[str, Any]] = []
    for segment in segments:
        if speaker is not None and segment.get("speaker") != speaker:
            continue
        overlap_start_sec = max(float(segment.get("start", 0.0) or 0.0), interval_start_sec)
        overlap_end_sec = min(float(segment.get("end", 0.0) or 0.0), interval_end_sec)
        text = (segment.get("text") or "").strip()
        if overlap_end_sec <= overlap_start_sec or not text:
            continue
        local_segments.append(
            {
                "start": round(overlap_start_sec - interval_start_sec, 4),
                "end": round(overlap_end_sec - interval_start_sec, 4),
                "text": text,
                "speaker": segment.get("speaker"),
            }
        )
    return local_segments


def _normalize_audio_for_dataset_sync(audio_bytes: bytes) -> bytes:
    """
    Normalize loudness and resample. For synthetic audio we trim edge silence
    more assertively than natural recordings, while still leaving a little gap
    so clips do not start or stop abruptly.
    """
    filters =[
        # 1. Remove silence at the beginning
        "silenceremove=start_periods=1:start_duration=0.2:start_threshold=-50dB",
        # 2. Reverse audio, remove silence at the "new" beginning (which is the end), reverse back
        "areverse",
        "silenceremove=start_periods=1:start_duration=0.2:start_threshold=-50dB",
        "areverse",
        # 3. Format and loudnorm
        "aformat=channel_layouts=mono",
        f"loudnorm=I={DATASET_TARGET_LUFS}:LRA=7:TP={DATASET_TP_DB}",
        "aresample=24000",
    ]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-af", ",".join(filters),
        "-f", "wav", "pipe:1",
    ]
    try:
        process = subprocess.run(
            cmd, input=audio_bytes,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
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

def _resegment_by_sentences(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Takes Whisper segments (which might contain multiple sentences) 
    and breaks them into individual sentence segments using word-level timing.
    """
    new_segments = []
    for seg in segments:
        words = seg.get("words", [])
        if not words:
            new_segments.append(seg)
            continue
            
        current_sentence_words = []
        sentence_start = words[0]["start"]
        
        for i, word in enumerate(words):
            current_sentence_words.append(word)
            # Check if this word ends with sentence punctuation
            text = word["word"].strip()
            if any(punc in text for punc in ".!?") or i == len(words) - 1:
                new_segments.append({
                    "start": sentence_start,
                    "end": word["end"],
                    "text": " ".join([w["word"] for w in current_sentence_words]).strip(),
                    "speaker": seg.get("speaker")
                })
                if i < len(words) - 1:
                    sentence_start = words[i+1]["start"]
                    current_sentence_words = []
    return new_segments

def _chunk_contiguous_segments(segments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not segments:
        return []

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_start = 0.0
    current_end = 0.0

    for seg in segments:
        if not current:
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        proposed_duration = seg["end"] - current_start

        if proposed_duration > DATASET_SAMPLE_MAX_SEC:
            chunks.append(current)
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        current_duration = current_end - current_start
        current_has_sentence_end = _ends_with_sentence_punct(current[-1].get("text", ""))
        gap_duration = max(0.0, seg["start"] - current_end)
        can_flush = current_duration >= DATASET_SAMPLE_MIN_SEC
        hit_sentence_limit = len(current) >= DATASET_MAX_SENTENCES_PER_CLIP

        if can_flush and (hit_sentence_limit or gap_duration >= DATASET_MERGE_GAP_SEC):
            chunks.append(current)
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        if can_flush and current_has_sentence_end and current_duration >= DATASET_CLIP_FLUSH_SEC:
            chunks.append(current)
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        current.append(seg)
        current_end = seg["end"]

    if current:
        chunks.append(current)

    return chunks


def _chunk_segments(
    segments: list[dict[str, Any]],
    *,
    main_speaker: str | None,
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current_run: list[dict[str, Any]] = []

    for seg in segments:
        speaker = seg.get("speaker")
        if main_speaker is not None and speaker != main_speaker:
            if current_run:
                runs.append(current_run)
                current_run = []
            continue
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0) or 0.0)
        end   = float(seg.get("end",   0.0) or 0.0)
        if end <= start or not text:
            if current_run:
                runs.append(current_run)
                current_run = []
            continue
        current_run.append({"start": start, "end": end, "text": text, "speaker": speaker})

    if current_run:
        runs.append(current_run)

    chunks: list[list[dict[str, Any]]] = []
    for run in runs:
        chunks.extend(_chunk_contiguous_segments(run))
    return chunks


def _build_split_results(
    filename: str,
    wav_bytes: bytes,
    prompt_id: str,
    segments: list[dict[str, Any]],
    *,
    main_speaker: str | None,
    canonical_text: str | None = None,       # ← NEW: pass ground truth if known
) -> list[tuple[str, bytes, str, bool, str, str | None, bool, bool]]:
    """
    Returns tuples of:
      (clip_name, wav_bytes, prompt_id, success, transcript, failure_reason, is_short, uses_aligned_transcript)
    """
    from pydub import AudioSegment

    audio_segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    audio_duration_sec = len(audio_segment) / 1000.0
    chunks = _chunk_segments(segments, main_speaker=main_speaker)
    if not chunks:
        return []

    results = []

    for idx, chunk in enumerate(chunks):
        start_ms = max(0, int(chunk[0]["start"]  * 1000) - DATASET_EDGE_PADDING_MS)
        end_ms   = min(len(audio_segment), int(chunk[-1]["end"] * 1000) + DATASET_EDGE_PADDING_MS)
        cropped  = audio_segment[start_ms:end_ms]
        asr_transcript = _join_segment_text(chunk)

        if not asr_transcript or len(cropped) <= 0:
            continue

        clip_duration_sec = max(0.0, chunk[-1]["end"] - chunk[0]["start"])

        # Hard discard: below absolute floor or above ceiling
        if clip_duration_sec < DATASET_SAMPLE_MIN_SEC or clip_duration_sec > DATASET_SAMPLE_MAX_SEC:
            log.debug(
                "Discarding %s chunk %d (%.2fs) — outside [%.1f, %.1f]s window",
                filename, idx, clip_duration_sec, DATASET_SAMPLE_MIN_SEC, DATASET_SAMPLE_MAX_SEC,
            )
            continue

        out_buf = io.BytesIO()
        cropped.export(out_buf, format="wav")
        normalized_wav = _normalize_audio_for_dataset_sync(out_buf.getvalue())
        normalized_duration_sec = _wav_duration_seconds(normalized_wav)

        max_output_duration_sec = DATASET_SAMPLE_MAX_SEC + ((2 * DATASET_EDGE_PADDING_MS) / 1000.0)
        if normalized_duration_sec < DATASET_SAMPLE_MIN_SEC or normalized_duration_sec > max_output_duration_sec:
            log.debug(
                "Discarding %s chunk %d after final normalization (%.2fs) — outside [%.1f, %.1f]s window",
                filename, idx, normalized_duration_sec, DATASET_SAMPLE_MIN_SEC, max_output_duration_sec,
            )
            continue

        clip_name = _segmented_filename(filename, idx, len(chunks))
        is_short  = normalized_duration_sec < DATASET_SHORT_CLIP_MAX_SEC
        uses_aligned_transcript = (
            main_speaker is not None
            or len(chunks) > 1
            or chunk[0]["start"] > 0.15
            or chunk[-1]["end"] < audio_duration_sec - 0.15
        )

        results.append(
            (clip_name, normalized_wav, prompt_id, True, asr_transcript, None, is_short, uses_aligned_transcript)
        )

    return results


def _try_build_split_results(
    filename: str,
    wav_bytes: bytes,
    prompt_id: str,
    segments: list[dict[str, Any]],
    *,
    main_speaker: str | None,
) -> list[tuple[str, bytes, str, bool, str, str | None, bool]]:
    try:
        return _build_split_results(
            filename,
            wav_bytes,
            prompt_id,
            segments,
            main_speaker=main_speaker,
        )
    except Exception as exc:
        log.warning("Dataset split fallback failed for %s: %s", filename, exc)
        return []


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


def _dataset_prep_cuda_wait_interval_sec() -> float:
    raw = (os.environ.get("DATASET_PREP_CUDA_WAIT_POLL_SEC", "10") or "10").strip()
    try:
        return max(float(raw), 0.1)
    except ValueError:
        log.warning("Ignoring invalid DATASET_PREP_CUDA_WAIT_POLL_SEC=%r", raw)
        return 10.0


def _dataset_prep_cuda_wait_timeout_sec() -> float | None:
    raw = (os.environ.get("DATASET_PREP_CUDA_WAIT_TIMEOUT_SEC", "0") or "0").strip()
    try:
        timeout_sec = max(float(raw), 0.0)
    except ValueError:
        log.warning("Ignoring invalid DATASET_PREP_CUDA_WAIT_TIMEOUT_SEC=%r", raw)
        timeout_sec = 0.0
    return None if timeout_sec == 0.0 else timeout_sec


def _wait_for_dataset_prep_cuda(model_name: str, *, reason: str) -> None:
    import time
    import torch

    min_free_gb = _dataset_whisper_min_free_gb(model_name)
    poll_sec = _dataset_prep_cuda_wait_interval_sec()
    timeout_sec = _dataset_prep_cuda_wait_timeout_sec()
    started_at = time.monotonic()
    last_log_at = 0.0

    while True:
        cuda_available = torch.cuda.is_available()
        free_gb = _get_cuda_free_memory_gb() if cuda_available else None
        has_headroom = free_gb is None or free_gb >= min_free_gb
        if cuda_available and has_headroom:
            if last_log_at:
                waited_sec = time.monotonic() - started_at
                if free_gb is None:
                    log.info(
                        "Dataset prep resumed on CUDA for WhisperX %s after waiting %.1fs.",
                        model_name,
                        waited_sec,
                    )
                else:
                    log.info(
                        "Dataset prep resumed on CUDA for WhisperX %s after waiting %.1fs (free %.1f GB).",
                        model_name,
                        waited_sec,
                        free_gb,
                    )
            return

        now = time.monotonic()
        if timeout_sec is not None and (now - started_at) >= timeout_sec:
            if not cuda_available:
                raise RuntimeError(
                    f"Timed out waiting for CUDA to become available for dataset prep ({reason})."
                )
            free_text = "unknown" if free_gb is None else f"{free_gb:.1f} GB"
            raise RuntimeError(
                f"Timed out waiting for CUDA headroom for dataset prep ({reason}); "
                f"free memory {free_text}, need at least {min_free_gb:.1f} GB."
            )

        if (now - last_log_at) >= poll_sec:
            if not cuda_available:
                log.info(
                    "Dataset prep waiting for CUDA availability for WhisperX %s (%s).",
                    model_name,
                    reason,
                )
            elif free_gb is None:
                log.info(
                    "Dataset prep waiting for CUDA headroom for WhisperX %s (%s).",
                    model_name,
                    reason,
                )
            else:
                log.info(
                    "Dataset prep waiting for CUDA headroom for WhisperX %s: %.1f/%.1f GB free (%s).",
                    model_name,
                    free_gb,
                    min_free_gb,
                    reason,
                )
            last_log_at = now

        time.sleep(poll_sec)


def _resolve_dataset_prep_device(configured_device: str, model_name: str) -> str:
    import torch

    if configured_device == "cpu":
        return configured_device
    if configured_device == "auto":
        if not torch.cuda.is_available():
            return "cpu"
        free_gb = _get_cuda_free_memory_gb()
        min_free_gb = _dataset_whisper_min_free_gb(model_name)
        if free_gb is not None and free_gb < min_free_gb:
            _wait_for_dataset_prep_cuda(
                model_name,
                reason=f"free CUDA memory {free_gb:.1f} GB below {min_free_gb:.1f} GB threshold",
            )
        return "cuda"
    if configured_device == "cuda":
        if not torch.cuda.is_available():
            _wait_for_dataset_prep_cuda(model_name, reason=f"configured_device={configured_device}")
        free_gb = _get_cuda_free_memory_gb()
        min_free_gb = _dataset_whisper_min_free_gb(model_name)
        if free_gb is not None and free_gb < min_free_gb:
            _wait_for_dataset_prep_cuda(
                model_name,
                reason=f"free CUDA memory {free_gb:.1f} GB below {min_free_gb:.1f} GB threshold",
            )
        return "cuda"
    if configured_device:
        log.warning("Unknown DATASET_PREP_DEVICE=%r; defaulting dataset prep to CUDA wait mode.", configured_device)
        _wait_for_dataset_prep_cuda(model_name, reason=f"configured_device={configured_device}")
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

    global _dataset_cuda_oom_retry_after
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    cache_key = f"model_{model_name}_{device}_{compute_type}"
    with _cache_lock:
        if device == "cuda" and time.time() < _dataset_cuda_oom_retry_after:
            fallback_key = f"model_{model_name}_cpu_int8"
            if fallback_key not in _whisper_cache:
                retry_in = max(0.0, _dataset_cuda_oom_retry_after - time.time())
                log.warning(
                    "Skipping WhisperX CUDA load for %.1fs after recent OOM; using CPU int8 (%s).",
                    retry_in,
                    model_name,
                )
                _whisper_cache[fallback_key] = whisperx.load_model(model_name, "cpu", compute_type="int8")
            return _whisper_cache[fallback_key]

        if cache_key not in _whisper_cache:
            log.info("Loading WhisperX model %s on %s (%s)", model_name, device, compute_type)
            try:
                _whisper_cache[cache_key] = whisperx.load_model(model_name, device, compute_type=compute_type)
            except RuntimeError as exc:
                if device != "cuda" or not _is_cuda_oom_error(exc):
                    raise

                _clear_torch_cuda_cache()
                reduced_key = f"model_{model_name}_cuda_int8_float16"
                log.warning(
                    "WhisperX CUDA OOM for %s (%s); retrying with compute_type=int8_float16.",
                    model_name,
                    compute_type,
                )
                try:
                    if reduced_key not in _whisper_cache:
                        _whisper_cache[reduced_key] = whisperx.load_model(
                            model_name,
                            "cuda",
                            compute_type="int8_float16",
                        )
                    _dataset_cuda_oom_retry_after = 0.0
                    return _whisper_cache[reduced_key]
                except RuntimeError as reduced_exc:
                    if not _is_cuda_oom_error(reduced_exc):
                        raise

                cooldown_sec = max(float(os.environ.get("DATASET_WHISPER_CUDA_OOM_COOLDOWN_SEC", "90")), 1.0)
                _dataset_cuda_oom_retry_after = time.time() + cooldown_sec
                fallback_key = f"model_{model_name}_cpu_int8"
                log.warning(
                    "WhisperX CUDA still OOM for %s; using CPU int8 for %.0fs cooldown.",
                    model_name,
                    cooldown_sec,
                )
                if fallback_key not in _whisper_cache:
                    _whisper_cache[fallback_key] = whisperx.load_model(model_name, "cpu", compute_type="int8")
                return _whisper_cache[fallback_key]
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
) -> list[tuple[str, bytes, str, bool, str, str | None, bool, bool]]:
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

        while True:
            try:
                model = _get_whisper_model(model_name, device=device, compute_type=compute_type)
                break
            except Exception as exc:
                if device == "cuda" and _is_cuda_oom_error(exc):
                    log.warning(
                        "WhisperX model load hit CUDA OOM for %s; waiting for GPU headroom before retrying.",
                        char_name,
                        exc_info=True,
                    )
                    _clear_torch_cuda_cache()
                    _wait_for_dataset_prep_cuda(model_name, reason="WhisperX model load CUDA OOM")
                    continue
                raise

        diarize_model = None
        diarization_error: str | None = None
        if skip_validation:
            diarization_error = "SKIP_WHISPER_VALIDATION enabled"
            log.info("SKIP_WHISPER_VALIDATION enabled on GPU dataset job for %s.", char_name)
        else:
            while True:
                try:
                    diarize_model = _get_diarization_pipeline(hf_token, device=device)
                    break
                except Exception as exc:
                    if device == "cuda" and _is_cuda_oom_error(exc):
                        log.warning(
                            "Diarization load hit CUDA OOM for %s; waiting for GPU headroom before retrying.",
                            char_name,
                            exc_info=True,
                        )
                        _clear_torch_cuda_cache()
                        _wait_for_dataset_prep_cuda(model_name, reason="WhisperX diarization CUDA OOM")
                        continue
                    diarization_error = str(exc)
                    log.warning(
                        "Diarization unavailable for %s; falling back to aligned full-audio transcripts: %s",
                        char_name,
                        diarization_error,
                    )
                    break

        import time as _time
        import platform

        def _single_clip_result(
            filename: str,
            wav_bytes: bytes,
            prompt_id: str,
            transcript: str,
            failure_reason: str | None,
            segments: list[dict[str, Any]],
            *,
            uses_aligned_transcript: bool,
        ) -> tuple[str, bytes, str, bool, str, str | None, bool, bool]:
            duration_sec = _wav_duration_seconds(wav_bytes)
            return (
                filename,
                wav_bytes,
                prompt_id,
                True,
                transcript,
                failure_reason,
                duration_sec < DATASET_SHORT_CLIP_MAX_SEC,
                uses_aligned_transcript,
            )

        prepared_items = _prepare_batch_analysis_items(audio_items)
        ref_item = next(
            (item for item in prepared_items if item["prompt_id"] == "ref_audio" or item["filename"] == "ref_audio.wav"),
            None,
        )
        clone_items = [item for item in prepared_items if item is not ref_item]
        if ref_item is not None and clone_items:
            batches = _build_reference_diarization_batches(
                ref_item,
                clone_items,
                max_total_duration_sec=DATASET_DIARIZATION_BATCH_MAX_SEC,
            )
        else:
            batches = [[item] for item in prepared_items]

        results: list[tuple[str, bytes, str, bool, str, str | None, bool, bool]] = []
        emitted_reference = False
        total_batches = len(batches)

        for batch_idx, batch_items in enumerate(batches, 1):
            batch_start = _time.monotonic()
            combined_wav_bytes, clip_intervals = _concatenate_batch_analysis_wavs(batch_items)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_wav_path = f.name
                f.write(combined_wav_bytes)

            try:
                audio = whisperx.load_audio(temp_wav_path)

                t0 = _time.monotonic()
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
                    "[%s] Batch %d/%d: transcription done in %.1fs — lang=%s, %d segments",
                    char_name,
                    batch_idx,
                    total_batches,
                    _time.monotonic() - t0,
                    result.get("language", "?"),
                    len(result.get("segments", [])),
                )

                align_device = device
                if platform.system() == "Windows" and align_device == "cuda":
                    align_device = "cpu"

                while True:
                    try:
                        model_a, metadata = _get_align_model(result["language"], align_device)
                        break
                    except Exception as exc:
                        if align_device == "cuda" and _is_cuda_oom_error(exc):
                            log.warning(
                                "Alignment load hit CUDA OOM for %s; waiting for GPU headroom before retrying.",
                                char_name,
                                exc_info=True,
                            )
                            _clear_torch_cuda_cache()
                            _wait_for_dataset_prep_cuda(model_name, reason="WhisperX alignment CUDA OOM")
                            continue
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
                    "[%s] Batch %d/%d: alignment done in %.1fs (device=%s)",
                    char_name,
                    batch_idx,
                    total_batches,
                    _time.monotonic() - t0,
                    align_device,
                )

                segments = result["segments"]
                target_speaker: str | None = None
                clip_reason = diarization_error

                if diarize_model is not None:
                    try:
                        t0 = _time.monotonic()
                        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=10)
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                        segments = result["segments"]
                        segments = _resegment_by_sentences(segments)
                        log.info(
                            "[%s] Batch %d/%d: diarization done in %.1fs — %d segments",
                            char_name,
                            batch_idx,
                            total_batches,
                            _time.monotonic() - t0,
                            len(segments),
                        )
                    except Exception as exc:
                        clip_reason = f"diarization failed: {exc}"
                        log.warning(
                            "[%s] Batch %d/%d: diarization failed, using aligned fallback: %s",
                            char_name,
                            batch_idx,
                            total_batches,
                            exc,
                        )

                if ref_item is not None:
                    reference_interval = next(
                        (
                            interval
                            for interval in clip_intervals
                            if interval["prompt_id"] == "ref_audio" or interval["filename"] == "ref_audio.wav"
                        ),
                        None,
                    )
                    if reference_interval is not None:
                        target_speaker = _select_target_speaker_for_interval(
                            segments,
                            reference_interval["analysis_start_sec"],
                            reference_interval["analysis_end_sec"],
                        )

                if target_speaker is None:
                    target_speaker = _select_dominant_speaker(segments)

                for clip_interval in clip_intervals:
                    filename = clip_interval["filename"]
                    wav_bytes = clip_interval["wav_bytes"]
                    prompt_id = clip_interval["prompt_id"]
                    is_reference = prompt_id == "ref_audio" or filename == "ref_audio.wav"

                    if is_reference and emitted_reference:
                        continue

                    local_segments = _slice_combined_segments_to_local_interval(
                        segments,
                        clip_interval["analysis_start_sec"],
                        clip_interval["analysis_end_sec"],
                    )
                    local_target_segments = (
                        _slice_combined_segments_to_local_interval(
                            segments,
                            clip_interval["analysis_start_sec"],
                            clip_interval["analysis_end_sec"],
                            speaker=target_speaker,
                        )
                        if target_speaker is not None
                        else []
                    )
                    fallback_transcript = _join_segment_text(local_segments)

                    split_results: list[tuple[str, bytes, str, bool, str, str | None, bool, bool]] = []
                    if local_target_segments:
                        split_results = _try_build_split_results(
                            filename,
                            wav_bytes,
                            prompt_id,
                            local_target_segments,
                            main_speaker=None,
                        )

                    if not split_results:
                        split_results = _try_build_split_results(
                            filename,
                            wav_bytes,
                            prompt_id,
                            local_segments,
                            main_speaker=None,
                        )

                    if split_results:
                        results.extend(split_results)
                        if is_reference:
                            emitted_reference = True
                        continue

                    if fallback_transcript:
                        uses_aligned_transcript = bool(local_target_segments) or bool(
                            local_segments and (
                                local_segments[0]["start"] > 0.15
                                or local_segments[-1]["end"] < _wav_duration_seconds(wav_bytes) - 0.15
                            )
                        )
                        results.append(
                            _single_clip_result(
                                filename,
                                wav_bytes,
                                prompt_id,
                                fallback_transcript,
                                clip_reason,
                                local_segments,
                                uses_aligned_transcript=uses_aligned_transcript,
                            )
                        )
                        if is_reference:
                            emitted_reference = True
                    else:
                        results.append((filename, wav_bytes, prompt_id, False, "", clip_reason or "empty transcript after alignment", False, False))

                log.info(
                    "[%s] Batch %d/%d: processed %d item(s) in %.1fs",
                    char_name,
                    batch_idx,
                    total_batches,
                    len(clip_intervals),
                    _time.monotonic() - batch_start,
                )
            except Exception as exc:
                log.error("Dataset validation batch failed for %s: %s", char_name, exc, exc_info=True)
                for clip_interval in clip_intervals:
                    results.append(
                        (
                            clip_interval["filename"],
                            clip_interval["wav_bytes"],
                            clip_interval["prompt_id"],
                            False,
                            "",
                            str(exc),
                            False,
                            False,
                        )
                    )
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
) -> list[tuple[str, bytes, str, bool, str, str | None, bool]]:
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
        raw_audio_segments.append((filename, content, prompt_id))

    validated_segments = await validate_and_crop_audio(raw_audio_segments, character_name)

    final_reference_segments: list[tuple[str, bytes, str]] = []
    final_train_segments: list[tuple[str, bytes]] = []
    final_transcripts: list[str] = []
    failure_reasons: list[str] = []
    for filename, wav_bytes, prompt_id, success, transcript, failure_reason, *extra in validated_segments:
        is_short = extra[0] if extra else False
        uses_aligned_transcript = extra[1] if len(extra) > 1 else False
        if not success or not transcript:
            if failure_reason:
                failure_reasons.append(f"{filename}: {failure_reason}")
            continue
        if prompt_id == "ref_audio":
            final_reference_segments.append(
                (
                    filename,
                    _normalize_audio_for_dataset_sync(wav_bytes),
                    _normalize_transcript_text(transcript) or ref_text,
                )
            )
            continue
        # Important: do NOT train on ASR output when we already know the canonical
        # transcript for the whole uploaded clip. But if validation split that clip
        # into `__seg_...`/`__part_...` fragments, the returned per-fragment
        # transcript is the only text that still matches the cropped audio.
        canonical = prompt_id_to_text.get(prompt_id) or ""
        is_split_fragment = "__seg_" in filename or "__part_" in filename or uses_aligned_transcript

        normalized_transcript = (
            _normalize_transcript_text(transcript)
            if is_split_fragment
            else (canonical or _normalize_transcript_text(transcript))
        )
        if not normalized_transcript:
            failure_reasons.append(f"{filename}: transcript empty after normalization")
            continue

        final_train_segments.append((filename, _normalize_audio_for_dataset_sync(wav_bytes), is_short))
        final_transcripts.append(normalized_transcript)

    if not final_train_segments:
        detail = "; ".join(failure_reasons[:3]) if failure_reasons else "all clips failed validation"
        raise RuntimeError(f"No valid segments available for finetuning dataset. {detail}")

    dataset_items: list[dict[str, Any]] = []
    for filename, wav_bytes, transcript in final_reference_segments:
        item_key = f"datasets/{book_id}/items/{character_id}_{job_id}_{filename}"
        s3_url = await _upload_bytes(
            storage,
            wav_bytes,
            item_key,
            content_type="audio/wav",
            metadata={
                "character_id": character_id,
                "job_id": job_id,
                "type": "dataset_item_ref",
            },
        )
        dataset_items.append(
            {
                "id": filename,
                "url": storage.get_presigned_url(item_key, expires_in=86400 * 7),
                "s3_url": s3_url,
                "text": transcript,
                "is_reference": True,
                "included": True,
            }
        )

    for idx, (filename, wav_bytes, _is_short) in enumerate(final_train_segments):
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
    reference_segments: list[tuple[float, bytes]] = []

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
            reference_segments.append((_wav_duration_seconds(wav_bytes), wav_bytes))
            continue
        final_train_segments.append((filename, wav_bytes))
        final_transcripts.append(transcript)

    if not final_train_segments:
        raise RuntimeError("No dataset items available to package.")

    chosen_ref_audio = max(reference_segments, key=lambda item: item[0])[1] if reference_segments else final_train_segments[-1][1]

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
