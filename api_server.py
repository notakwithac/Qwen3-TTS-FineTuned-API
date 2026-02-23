# coding=utf-8
"""FastAPI server — Fine-tuning-as-a-Service for Qwen3-TTS with E2E Object Storage."""

from dotenv import load_dotenv
load_dotenv()

import logging
import os
import re
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from pipeline import Pipeline, JobStatus
from storage import storage
from ops_logger import ops_log

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Qwen3-TTS Fine-Tuning API",
    description=(
        "Upload a voice dataset, fine-tune a TTS model, generate speech, "
        "and store results in E2E Object Storage (S3-compatible)."
    ),
    version="2.0.0",
)

# GPU configuration
DEVICE = os.environ.get("DEVICE", "cuda:0")
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "1") == "1"
GPU_IDLE_TIMEOUT = int(os.environ.get("GPU_IDLE_TIMEOUT", "600"))
GPU_MAX_CONCURRENCY = int(os.environ.get("GPU_MAX_CONCURRENCY", "16"))
GPU_MAX_MODELS = int(os.environ.get("GPU_MAX_MODELS", "4"))
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "1") == "1"

pipeline = Pipeline(
    base_dir=".",
    jobs_dir="jobs",
    device=DEVICE,
    use_flash_attn=USE_FLASH_ATTN,
    idle_timeout_seconds=GPU_IDLE_TIMEOUT,
    max_concurrency=GPU_MAX_CONCURRENCY,
    max_models=GPU_MAX_MODELS,
    compile=USE_TORCH_COMPILE,
)

# ---------------------------------------------------------------------------
# Dynamic Batching
# ---------------------------------------------------------------------------

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

class DynamicBatcher:
    """Pools completely independent concurrent requests into unified GPU tensor batches."""
    def __init__(self, batch_size: int, timeout_ms: int, process_fn: Callable):
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0
        self.process_fn = process_fn
        
        self.queue = []
        self.lock = asyncio.Lock()
        self.timer_task = None
        self.executor = ThreadPoolExecutor(max_workers=1) # One batch at a time for this specific model

    async def submit(self, **kwargs) -> Any:
        future = asyncio.Future()
        
        async with self.lock:
            self.queue.append((kwargs, future))
            if len(self.queue) == 1:
                # First item in batch, start timer
                self.timer_task = asyncio.create_task(self._wait_and_process())
            
            if len(self.queue) >= self.batch_size:
                # Batch full, process immediately
                if self.timer_task:
                    self.timer_task.cancel()
                asyncio.create_task(self._process_batch(list(self.queue)))
                self.queue.clear()
                
        return await future

    async def _wait_and_process(self):
        try:
            await asyncio.sleep(self.timeout)
        except asyncio.CancelledError:
            return  # Batch was processed because it got full
            
        async with self.lock:
            if not self.queue:
                return
            batch = list(self.queue)
            self.queue.clear()
            
        await self._process_batch(batch)

    async def _process_batch(self, batch: list[tuple[dict, asyncio.Future]]):
        if not batch: return
        
        # Unzip merged kwargs
        kwargs_keys = batch[0][0].keys()
        batched_kwargs = {k: [item[0][k] for item in batch] for k in kwargs_keys}
        futures = [item[1] for item in batch]
        
        try:
            # Run the heavy blocking batch process in a thread
            loop = asyncio.get_running_loop()
            results, sr = await loop.run_in_executor(
                self.executor,
                lambda: self.process_fn(**batched_kwargs)
            )
            
            # Map results back to individual futures
            for i, future in enumerate(futures):
                if not future.done():
                    future.set_result((results[i], sr))
                    
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)


# Instantiate global batchers
voice_design_batcher = DynamicBatcher(
    batch_size=GPU_MAX_CONCURRENCY,
    timeout_ms=100,
    process_fn=pipeline.inference.generate_voice_design_batch
)

custom_voice_batchers = {}  # Map job_id -> DynamicBatcher

def get_custom_voice_batcher(job_id: str, checkpoint_path: str, speaker_name: str) -> DynamicBatcher:
    if job_id not in custom_voice_batchers:
        def process_fn(texts: list[str], languages: list[str], instructs: list[str]):
            return pipeline.inference.generate_batch(
                texts=texts,
                checkpoint_path=checkpoint_path,
                speaker_name=speaker_name,
                languages=languages,
                instructs=instructs
            )
        custom_voice_batchers[job_id] = DynamicBatcher(
            batch_size=GPU_MAX_CONCURRENCY,
            timeout_ms=100,
            process_fn=process_fn
        )
    return custom_voice_batchers[job_id]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    text: str
    language: str = "English"
    instruct: str = ""
    upload_to_s3: bool = True  # Now default to True
    s3_filename: Optional[str] = None
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None


class InferS3Response(BaseModel):
    s3_url: str
    presigned_url: Optional[str] = None
    s3_key: str
    sample_rate: int
    text: str
    job_id: str


class BatchInferRequest(BaseModel):
    """Generate multiple audio files in one call, all uploaded to S3."""
    items: list  # list of {"text": str, "language": str, "instruct": str, "filename": str}
    language: str = "English"
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    speaker_name: str
    progress: dict = {}
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None
    config: dict = {}
    inference_url: Optional[str] = None


class StorageStatus(BaseModel):
    configured: bool
    endpoint: str
    bucket: str


class VoiceDesignRequest(BaseModel):
    """Generate speech using VoiceDesign model (no fine-tuning needed)."""
    text: str
    instruct: str  # Voice description, e.g. "A warm male voice, middle-aged, calm"
    language: str = "English"
    upload_to_s3: bool = True
    s3_filename: Optional[str] = None
    character_name: Optional[str] = None
    character_uuid: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
async def root():
    return {
        "service": "qwen3-tts-finetune-api",
        "status": "ok",
        "storage_configured": storage.is_configured,
    }


@app.get("/storage/status", summary="Check storage configuration", response_model=StorageStatus)
async def storage_status():
    """Check whether E2E Object Storage is configured."""
    return {
        "configured": storage.is_configured,
        "endpoint": storage.endpoint_url,
        "bucket": storage.bucket,
    }

class FinetuneRequest(BaseModel):
    dataset_s3_key: str
    speaker_name: str
    batch_size: int = 2
    num_epochs: int = 15
    lr: float = 1e-6
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None
    character_id: Optional[str] = None
    resume_job_id: Optional[str] = None

@app.post("/finetune", summary="Start a fine-tuning job", response_model=JobSummary)
def create_finetune_job(req: FinetuneRequest):
    """Start fine-tuning using a dataset zip stored in S3.

    The dataset must be a zip file in the configured S3 bucket containing:
    - `train.jsonl` — each line: `{"audio": "./data/X.wav", "text": "...", "ref_audio": "./data/ref_audio.wav"}`
    - `data/` folder with all referenced `.wav` files

    Returns a job object with a `job_id` you can use to poll status.
    """
    if not storage.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Storage not configured. Set E2E_ACCESS_KEY and E2E_SECRET_KEY.",
        )

    # Validation from research guide: avoid underscores followed by numbers
    if re.search(r'_\d', req.speaker_name):
        raise HTTPException(
            status_code=400,
            detail="Speaker name cannot contain an underscore followed by a number (e.g. avoid 'Voice_1'). Use 'Voice1' instead."
        )

    # Download dataset from S3 to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name
    
    op = ops_log.start("finetune_job_create", extra={
        "speaker_name": req.speaker_name, "num_epochs": req.num_epochs, "dataset_s3_key": req.dataset_s3_key,
    })
    
    try:
        # Resolve resume_job_id to a base_model_path
        base_model_path = None
        if req.resume_job_id:
            previous_job = pipeline.get_job(req.resume_job_id)
            if not previous_job:
                ops_log.start("resume_job_fallback", extra={"reason": "job_not_found", "resume_job_id": req.resume_job_id})
                logger.warning(f"Job to resume ({req.resume_job_id}) not found. Falling back to default base model.")
            elif not previous_job.checkpoint_path or not os.path.exists(previous_job.checkpoint_path):
                ops_log.start("resume_job_fallback", extra={"reason": "checkpoint_not_found", "resume_job_id": req.resume_job_id})
                logger.warning(f"Job to resume ({req.resume_job_id}) does not have a valid checkpoint. Falling back to default base model.")
            else:
                base_model_path = previous_job.checkpoint_path

        # Download the file from S3
        storage.download_file(req.dataset_s3_key, tmp_path)
        
        job = pipeline.create_job(
            dataset_zip_path=tmp_path,
            speaker_name=req.speaker_name,
            num_epochs=req.num_epochs,
            batch_size=req.batch_size,
            lr=req.lr,
            book_id=req.book_id,
            chapter_id=req.chapter_id,
            character_id=req.character_id,
            base_model_path=base_model_path,
        )
        ops_log.end(op, extra={"job_id": job.job_id})
    except Exception as e:
        ops_log.fail(op, str(e))
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Failed to create job: {str(e)}")
    finally:
        # Note: pipeline.create_job extracts the zip, but we should clean up the tmp zip itself
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    pipeline.start_job(job.job_id)
    return JSONResponse(content=job.to_dict(), status_code=202)


@app.get("/gpu/cleanup", summary="Trigger Disk LRU cleanup manualy")
async def trigger_cleanup(threshold_gb: float = 30.0):
    """Manually trigger the Disk LRU cleanup process."""
    pipeline._cleanup_disk_lru(threshold_gb)
    return {"detail": "Cleanup triggered"}


@app.get("/jobs", summary="List all jobs")
async def list_jobs():
    """List all fine-tuning jobs and their statuses."""
    return pipeline.list_jobs()


@app.get("/jobs/{job_id}", summary="Get job status", response_model=JobSummary)
async def get_job(job_id: str):
    """Get the current status and progress of a fine-tuning job."""
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@app.delete("/jobs/{job_id}", summary="Cancel or delete a job")
async def delete_job(job_id: str):
    """Cancel a running job or delete a completed one."""
    if not pipeline.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"detail": f"Job {job_id} deleted"}


@app.post("/jobs/{job_id}/retry", summary="Retry a failed job", response_model=JobSummary)
async def retry_job(job_id: str):
    """Retry a job that has failed or been cancelled."""
    job = pipeline.retry_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found, or it is not in a failed/cancelled state.")
    return JSONResponse(content=job.to_dict(), status_code=202)


@app.post(
    "/infer/{job_id}",
    summary="Generate speech from a fine-tuned model",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/json": {}},
            "description": "WAV audio (default) or JSON with S3 URL (if upload_to_s3=true)",
        }
    },
)
async def infer(job_id: str, req: InferRequest):
    """Generate speech using a completed fine-tuned model.

    Set `upload_to_s3: true` to upload the audio to E2E Object Storage
    and receive a JSON response with the S3 URL instead of the raw audio.
    """
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != JobStatus.READY:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not ready (status: {job.status}). "
                   f"Poll GET /jobs/{job_id} to check progress.",
        )

    pipeline.touch_job(job_id) # Update LRU timestamp
    pipeline._cleanup_disk_lru(20.0) # Background check usage

    try:
        checkpoint_path = str(job.checkpoint_path)
        batcher = get_custom_voice_batcher(job_id, checkpoint_path, job.speaker_name)
        
        with ops_log.operation("inference_api", job_id=job_id, extra={
            "text_length": len(req.text), "upload_to_s3": req.upload_to_s3,
        }):
            wav_bytes, sr = await batcher.submit(
                texts=req.text,
                languages=req.language,
                instructs=req.instruct,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # If upload_to_s3 is True (default), return JSON URL
    if req.upload_to_s3:
        # Construct S3 prefix based on user's structure decision
        s3_prefix = f"audio/segments/{req.book_id}/{req.chapter_id}" if req.book_id and req.chapter_id else f"audio/{job_id}"
        
        with ops_log.operation("s3_upload", job_id=job_id):
            s3_url = storage.upload_wav(wav_bytes, job_id, filename=req.s3_filename, prefix=s3_prefix)
        
        s3_key = f"{s3_prefix}/{req.s3_filename}" if req.s3_filename else s3_url.split(f"{storage.bucket}/")[-1]
        
        # Generate presigned URL for private bucket access
        presigned_url = storage.get_presigned_url(s3_key, expires_in=86400) # 24h
        
        return {
            "s3_url": s3_url, # Static URL
            "presigned_url": presigned_url, # Temp access URL
            "s3_key": s3_key,
            "sample_rate": sr,
            "text": req.text,
            "job_id": job_id,
        }

    # Otherwise return raw audio (if user explicitly set upload_to_s3=False)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="tts_{job_id}.wav"',
            "X-Sample-Rate": str(sr),
        },
    )


@app.post(
    "/infer/{job_id}/batch",
    summary="Batch generate speech and upload to S3",
    response_model=list[InferS3Response],
)
async def infer_batch(job_id: str, req: BatchInferRequest):
    """Generate multiple audio files and upload all to E2E Object Storage.

    This is optimized for your other microservice: send all texts at once,
    get back S3 URLs for each.

    Request body example:
    ```json
    {
        "items": [
            {"text": "Hello world", "filename": "chapter1_001.wav"},
            {"text": "Goodbye world", "filename": "chapter1_002.wav"}
        ],
        "language": "English"
    }
    ```
    """
    if not storage.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Storage not configured. Set E2E_ACCESS_KEY and E2E_SECRET_KEY.",
        )

    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != JobStatus.READY:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is not ready (status: {job.status}).",
        )

    pipeline.touch_job(job_id)
    pipeline._cleanup_disk_lru(20.0)

    import asyncio
    from functools import partial

    async def process_item(item, index):
        text = item.get("text", "")
        language = item.get("language", req.language)
        instruct = item.get("instruct", "")
        filename = item.get("filename", f"audio_{index:04d}.wav")

        # Run generation in a thread pool (InferenceManager handles the GPU semaphore)
        loop = asyncio.get_running_loop()
        try:
            checkpoint_path = str(job.checkpoint_path)
            wav_bytes, sr = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                partial(
                    pipeline.generate,
                    job_id=job_id,
                    text=text,
                    language=language,
                    instruct=instruct,
                    checkpoint_path=checkpoint_path,
                    speaker_name=job.speaker_name,
                )
            )

            # Construct S3 prefix
            s3_prefix = f"audio/segments/{req.book_id}/{req.chapter_id}" if req.book_id and req.chapter_id else f"audio/{job_id}"

            # Parallel S3 upload
            s3_url = await loop.run_in_executor(
                None,
                partial(storage.upload_wav, wav_bytes, job_id, filename=filename, prefix=s3_prefix)
            )
            
            s3_key = f"{s3_prefix}/{filename}"
            presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)

            return {
                "s3_url": s3_url,
                "presigned_url": presigned_url,
                "s3_key": s3_key,
                "sample_rate": sr,
                "text": text,
                "job_id": job_id,
            }
        except Exception as e:
            logger.error(f"Inference failed for item {index}: {e}")
            return None

    tasks = [process_item(item, i) for i, item in enumerate(req.items)]
    results_raw = await asyncio.gather(*tasks)
    
    # Filter out failed items
    results = [r for r in results_raw if r is not None]
    return results


@app.get(
    "/storage/list/{job_id}",
    summary="List all audio files in S3 for a job",
)
def list_storage(job_id: str, book_id: Optional[str] = None, chapter_id: Optional[str] = None):
    """List all audio files stored in S3 for a given job.
    If book_id and chapter_id are provided, it lists files in the segments folder.
    """
    if not storage.is_configured:
        raise HTTPException(status_code=503, detail="Storage not configured.")
    
    prefix = f"audio/segments/{book_id}/{chapter_id}/" if book_id and chapter_id else f"audio/{job_id}/"
    objects = storage.list_objects(prefix=prefix)
    return {
        "job_id": job_id,
        "count": len(objects),
        "files": [
            {"key": key, "url": storage._object_url(key)}
            for key in objects
        ],
    }


# ---------------------------------------------------------------------------
# Voice Design
# ---------------------------------------------------------------------------

@app.post(
    "/voice-design",
    summary="Generate speech using VoiceDesign model",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/json": {}},
            "description": "WAV audio or JSON with S3 URL",
        }
    },
)
async def voice_design(req: VoiceDesignRequest):
    """Generate speech from a text description of the desired voice.

    No fine-tuning needed — uses the VoiceDesign model directly.
    Describe the voice you want and it generates speech in that style.

    Example instruct values:
    - "A warm male voice, middle-aged, calm and authoritative"
    - "A young female voice, energetic and cheerful"
    - "A deep, gravelly old man's voice, speaking slowly"
    """
    parts = []
    if req.character_name:
        safe_name = "".join(c for c in req.character_name if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
        if safe_name:
            parts.append(safe_name)
    if req.character_uuid:
        parts.append(req.character_uuid)
        
    if parts:
        prefix = "_".join(parts)
        if req.s3_filename:
            if not req.s3_filename.startswith(prefix):
                req.s3_filename = f"{prefix}_{req.s3_filename}"
        else:
            req.s3_filename = f"{prefix}.wav"

    with ops_log.operation("voice_design_api", extra={
        "text_length": len(req.text),
        "instruct_length": len(req.instruct),
        "upload_to_s3": req.upload_to_s3,
    }):
        if req.upload_to_s3 and req.s3_filename:
            if not storage.is_configured:
                raise HTTPException(status_code=503, detail="Storage not configured.")
            
            s3_key = f"audio/voice_design/{req.s3_filename}"
            if storage.object_exists(s3_key):
                presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
                return {
                    "s3_url": storage._object_url(s3_key),
                    "presigned_url": presigned_url,
                    "s3_key": s3_key,
                    "sample_rate": 24000,
                    "text": req.text,
                    "instruct": req.instruct,
                }

        try:
            wav_bytes, sr = await voice_design_batcher.submit(
                texts=req.text,
                instructs=req.instruct,
                languages=req.language,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice design failed: {str(e)}")

    if req.upload_to_s3:
        if not storage.is_configured:
            raise HTTPException(status_code=503, detail="Storage not configured.")
        with ops_log.operation("s3_upload", extra={"type": "voice_design"}):
            s3_url = storage.upload_wav(wav_bytes, "voice_design", filename=req.s3_filename)
        s3_key = f"audio/voice_design/{req.s3_filename}" if req.s3_filename else s3_url.split(f"{storage.bucket}/")[-1]
        
        presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
        
        return {
            "s3_url": s3_url,
            "presigned_url": presigned_url,
            "s3_key": s3_key,
            "sample_rate": sr,
            "text": req.text,
            "instruct": req.instruct,
        }

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="voice_design.wav"',
            "X-Sample-Rate": str(sr),
        },
    )


# ---------------------------------------------------------------------------
# GPU Management
# ---------------------------------------------------------------------------

@app.get("/gpu/status", summary="GPU and model status")
async def gpu_status():
    """Get GPU memory usage, model load state, and idle timer info."""
    return pipeline.inference.stats


@app.post("/gpu/unload", summary="Manually unload model from GPU")
async def gpu_unload():
    """Immediately unload the model from GPU to free VRAM."""
    was_loaded = pipeline.inference.is_loaded
    pipeline.inference.unload()
    return {
        "detail": "Model unloaded" if was_loaded else "No model was loaded",
        "gpu_memory_allocated_gb": 0.0,
    }


class GpuConfigRequest(BaseModel):
    idle_timeout_seconds: int = 300


@app.put("/gpu/config", summary="Update GPU idle timeout")
async def gpu_config(req: GpuConfigRequest):
    """Change idle timeout. Set to 0 to disable auto-unload."""
    pipeline.inference.idle_timeout = req.idle_timeout_seconds
    return {
        "idle_timeout_seconds": req.idle_timeout_seconds,
        "auto_unload_enabled": req.idle_timeout_seconds > 0,
    }


# ---------------------------------------------------------------------------
# Operations Logging
# ---------------------------------------------------------------------------

@app.get("/ops/averages", summary="Average duration per operation type")
async def ops_averages():
    """Get average, min, max durations grouped by operation name.

    Useful for monitoring which pipeline stages are slow.
    """
    return ops_log.get_averages()


@app.get("/ops/history", summary="Operation history")
async def ops_history(
    op_name: Optional[str] = Query(None, description="Filter by operation name"),
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
    limit: int = Query(50, description="Max results"),
):
    """Get recent operation records with timestamps and durations."""
    return ops_log.get_history(op_name=op_name, job_id=job_id, limit=limit)


@app.get("/ops/running", summary="Currently running operations")
async def ops_running():
    """Get operations currently in progress."""
    return ops_log.get_running()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
