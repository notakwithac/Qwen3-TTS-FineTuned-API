from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tempfile
import threading
import zipfile
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

_whisper_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()
DATASET_SAMPLE_MIN_SEC = 10.0
DATASET_SAMPLE_TARGET_SEC = 15.0
DATASET_SAMPLE_MAX_SEC = 30.0
DATASET_MERGE_GAP_SEC = 0.75


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _join_segment_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip()).strip()


def _segmented_filename(filename: str, segment_index: int, total_segments: int) -> str:
    if total_segments <= 1:
        return filename
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem = filename
        ext = ""
    suffix = f"__seg_{segment_index:03d}"
    return f"{stem}{suffix}.{ext}" if ext else f"{stem}{suffix}"


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

    for seg in speaker_segments:
        if not current:
            current = [seg]
            current_start = seg["start"]
            current_end = seg["end"]
            continue

        gap = max(0.0, seg["start"] - current_end)
        proposed_duration = seg["end"] - current_start
        current_duration = current_end - current_start

        should_merge = False
        if gap <= DATASET_MERGE_GAP_SEC and proposed_duration <= DATASET_SAMPLE_TARGET_SEC:
            should_merge = True
        elif current_duration < DATASET_SAMPLE_MIN_SEC and proposed_duration <= DATASET_SAMPLE_MAX_SEC:
            should_merge = True

        if should_merge:
            current.append(seg)
            current_end = seg["end"]
            continue

        chunks.append(current)
        current = [seg]
        current_start = seg["start"]
        current_end = seg["end"]

    if current:
        chunks.append(current)

    return chunks


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
        start_ms = max(0, int(chunk[0]["start"] * 1000) - 100)
        end_ms = min(len(audio_segment), int(chunk[-1]["end"] * 1000) + 100)
        cropped = audio_segment[start_ms:end_ms]
        transcript = _join_segment_text(chunk)
        if len(cropped) <= 0 or not transcript:
            continue
        out_buf = io.BytesIO()
        cropped.export(out_buf, format="wav")
        results.append(
            (
                _segmented_filename(filename, idx, len(chunks)),
                out_buf.getvalue(),
                prompt_id,
                True,
                transcript,
                None,
            )
        )
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
                result = model.transcribe(audio, batch_size=8)
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
        prompt_id_to_text[prompt_id] = prompt_text

        content = await _download_bytes(storage, ref)
        content = await apply_audio_fx(content, amplitude=amplitude, speed=speed, pitch_shift=pitch_shift)
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
        final_train_segments.append((filename, wav_bytes))
        final_transcripts.append(transcript)

    if not final_train_segments:
        detail = "; ".join(failure_reasons[:3]) if failure_reasons else "all clips failed validation"
        raise RuntimeError(f"No valid segments available for finetuning dataset. {detail}")

    dataset_items: list[dict[str, Any]] = []

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
        ref = item.get("s3_url") or item.get("url")
        if not ref:
            continue
        wav_bytes = await _download_bytes(storage, str(ref))
        filename = str(item.get("id") or item.get("filename") or "").strip()
        transcript = str(item.get("text") or "").strip()
        if not filename or not transcript:
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
