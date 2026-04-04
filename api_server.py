# coding=utf-8
"""FastAPI server — Fine-tuning-as-a-Service for Qwen3-TTS with E2E Object Storage."""

from dotenv import load_dotenv
load_dotenv(override=True)

import logging
import os
import re
import tempfile
import time
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from pipeline import Pipeline, JobStatus
from storage import storage
from ops_logger import ops_log
from session_manager import SessionManager, SessionStatus
from metrics_collector import metrics_collector
from gpu_resource_controller import GPUResourceController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Log Streaming Setup ---
from fastapi.responses import StreamingResponse

# List of active log queues for SSE streaming
log_queues: list[asyncio.Queue] = []

class LogStreamHandler(logging.Handler):
    """Custom logging handler that pushes log records into active client queues."""
    def emit(self, record: logging.LogRecord):
        # 1. General level filter for the stream: only INFO and above
        if record.levelno < logging.DEBUG:
            return

        # 2. Filter out noisy operational and status logs from the SSE stream
        # (while keeping them in terminal/files for local debugging)
        if record.levelno < logging.WARNING:
            # Silence specific loggers known for noise
            # Silence noisy infrastructure loggers unless it's specifically for voice design
            if record.name in ("ops", "httpx", "httpcore", "urllib3"):
                if record.name == "ops" and "voice_design" in record.getMessage():
                    pass # Allow voice design operations through
                else:
                    return
            
            # Silence high-frequency polling paths and metadata requests
            msg = record.getMessage().lower()
            noise_filters = [
                "/health", "/status", "/ops/", "/gpu/status", 
                "/gpu/vram", "/sessions", "/docs", "/openapi.json", "/favicon.ico"
            ]
            if any(f in msg for f in noise_filters):
                return

        try:
            msg = self.format(record)
            # Push to all active queues
            for q in log_queues:
                asyncio.run_coroutine_threadsafe(q.put(msg), asyncio.get_event_loop())
        except Exception:
            self.handleError(record)

# Attach stream handler to root logger and 'ops' logger
stream_handler = LogStreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(stream_handler)
logging.getLogger("ops").addHandler(stream_handler)

# Suppress noisy model initialization logs
logging.getLogger("qwen_tts.core.models.configuration_qwen3_tts").setLevel(logging.ERROR)
logging.getLogger("qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app):
    # Startup: start session cleanup loop and resource monitoring
    session_mgr.start_cleanup_loop()
    metrics_collector.start()
    yield
    # Shutdown: stop metrics collector and wait for any in-progress S3 uploads
    metrics_collector.stop()
    logger.info("Server shutting down — waiting for pending S3 uploads...")
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, pipeline.shutdown)
    logger.info("Shutdown complete.")

app = FastAPI(
    title="Qwen3-TTS Fine-Tuning API",
    description=(
        "Upload a voice dataset, fine-tune a TTS model, generate speech, "
        "and store results in E2E Object Storage (S3-compatible)."
    ),
    version="2.0.0",
    lifespan=_lifespan,
)

# Global draining flag - when True, the API rejects new work but allows status checks.
IS_DRAINING = False

# ---------------------------------------------------------------------------
# Middleware: Request Logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request and its response status/duration."""
    request_id = uuid.uuid4().hex[:8]
    method = request.method
    url = str(request.url)
    client_host = request.client.host if request.client else "unknown"
    
    # We use ops_log to track the API call duration
    op = ops_log.start("api_request", extra={
        "req_id": request_id,
        "method": method,
        "url": url,
        "client": client_host,
    })
    
    # If draining, reject new work-related requests
    global IS_DRAINING
    if IS_DRAINING and method == "POST":
        # Allow /gpu/terminate to be called multiple times, but reject others
        if "/gpu/terminate" not in url:
            ops_log.fail(op, "Server is draining for termination")
            return JSONResponse(
                status_code=503,
                content={"detail": "Server is draining for termination. No new work accepted."}
            )

    start_time = time.time()
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        ops_log.end(op, extra={
            "status_code": response.status_code,
            "duration": round(duration, 3)
        })
        return response
    except Exception as e:
        duration = time.time() - start_time
        ops_log.fail(op, str(e), extra={
            "duration": round(duration, 3)
        })
        raise

# GPU configuration
DEVICE = os.environ.get("DEVICE", "cuda:0")
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "1") == "1"
GPU_IDLE_TIMEOUT = int(os.environ.get("GPU_IDLE_TIMEOUT", "600"))
GPU_MAX_CONCURRENCY = int(os.environ.get("GPU_MAX_CONCURRENCY", "16"))
GPU_MAX_MODELS = int(os.environ.get("GPU_MAX_MODELS", "4"))
GPU_BATCH_SIZE = int(os.environ.get("GPU_BATCH_SIZE", "32"))
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "1") == "1"

# Resource Isolation
_allow_concurrent_val = os.environ.get("ALLOW_CONCURRENT_TRAINING_INFERENCE")
ALLOW_CONCURRENT = None
if _allow_concurrent_val is not None:
    ALLOW_CONCURRENT = _allow_concurrent_val == "1"

gpu_controller = GPUResourceController(allow_concurrent=ALLOW_CONCURRENT)

# Session configuration
REPLICA_THRESHOLD = int(os.environ.get("REPLICA_THRESHOLD", "500"))
MAX_REPLICAS_PER_MODEL = int(os.environ.get("MAX_REPLICAS_PER_MODEL", "4"))
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))
MAX_CONCURRENT_VOICE_DESIGNS = int(os.environ.get("MAX_CONCURRENT_VOICE_DESIGNS", "4"))

logger.info(f"Loaded Configuration:")
logger.info(f"  - DEVICE: {DEVICE}")
logger.info(f"  - USE_FLASH_ATTN: {USE_FLASH_ATTN}")
logger.info(f"  - GPU_IDLE_TIMEOUT: {GPU_IDLE_TIMEOUT}s")
logger.info(f"  - GPU_MAX_CONCURRENCY: {GPU_MAX_CONCURRENCY}")
logger.info(f"  - GPU_MAX_MODELS: {GPU_MAX_MODELS}")
logger.info(f"  - GPU_BATCH_SIZE: {GPU_BATCH_SIZE}")
logger.info(f"  - USE_TORCH_COMPILE: {USE_TORCH_COMPILE}")
logger.info(f"  - REPLICA_THRESHOLD: {REPLICA_THRESHOLD}")
logger.info(f"  - MAX_CONCURRENT_VOICE_DESIGNS: {MAX_CONCURRENT_VOICE_DESIGNS}")
logger.info(f"  - MAX_REPLICAS_PER_MODEL: {MAX_REPLICAS_PER_MODEL}")
logger.info(f"  - SESSION_TIMEOUT: {SESSION_TIMEOUT}s")

pipeline = Pipeline(
    base_dir=".",
    jobs_dir="jobs",
    device=DEVICE,
    use_flash_attn=USE_FLASH_ATTN,
    idle_timeout_seconds=GPU_IDLE_TIMEOUT,
    max_concurrency=GPU_MAX_CONCURRENCY,
    max_models=GPU_MAX_MODELS,
    compile=USE_TORCH_COMPILE,
    gpu_controller=gpu_controller,
)

# Session-based inference manager
session_mgr = SessionManager(
    inference_manager=pipeline.inference,
    pipeline=pipeline,
    storage=storage,
    replica_threshold=REPLICA_THRESHOLD,
    max_replicas=MAX_REPLICAS_PER_MODEL,
    session_timeout=SESSION_TIMEOUT,
    batch_size=GPU_BATCH_SIZE,
)

# ---------------------------------------------------------------------------
# Dynamic Batching
# ---------------------------------------------------------------------------

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any

class DynamicBatcher:
    """Pools completely independent concurrent requests into unified GPU tensor batches."""
    def __init__(self, batch_size: int, timeout_ms: int, process_fn: Callable, max_workers: int = 1):
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0
        self.process_fn = process_fn
        
        self.queue = []
        self.lock = asyncio.Lock()
        self.timer_task = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

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
    batch_size=GPU_BATCH_SIZE,
    timeout_ms=100,
    process_fn=pipeline.inference.generate_voice_design_batch,
    max_workers=1  # Always 1 worker per model type to ensure thread-safety
)

voice_clone_batcher = DynamicBatcher(
    batch_size=GPU_BATCH_SIZE,
    timeout_ms=100,
    process_fn=pipeline.inference.generate_voice_clone_flexible_batch,
    max_workers=1
)

custom_voice_batchers = {}  # Map job_id -> DynamicBatcher
voice_clone_batch_jobs = {} # session_id -> {status, completed, total, results, error, ...}


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
            batch_size=GPU_BATCH_SIZE,
            timeout_ms=100,
            process_fn=process_fn,
            max_workers=1  # Always 1 worker per job/model to ensure thread-safety
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
    character_id: Optional[str] = None
    overwrite: bool = False  # If false, skips generation if file already exists on S3


class InferS3Response(BaseModel):
    s3_url: str
    presigned_url: Optional[str] = None
    s3_key: str
    sample_rate: int
    text: str
    job_id: str


class BatchInferRequest(BaseModel):
    """Generate multiple audio files in one call, all uploaded to S3."""
    items: list  # list of {"text": str, "language": str, "instruct": str, "filename": str, "overwrite": bool, "character_id": str}
    language: str = "English"
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None
    character_id: Optional[str] = None
    overwrite: bool = False  # Default overwrite flag for all items


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    speaker_name: str
    progress: dict = {}
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    finished_at: Optional[str] = None
    config: dict = {}
    inference_url: Optional[str] = None
    message: Optional[str] = None


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
    overwrite: bool = False  # If false, skips generation if file already exists on S3


class VoiceDesignBatchItem(BaseModel):
    """A single item in a voice design batch request."""
    text: str
    instruct: str  # Voice description, e.g. "A warm male voice, middle-aged, calm"
    language: str = "English"
    character_name: Optional[str] = None
    character_uuid: Optional[str] = None
    s3_filename: Optional[str] = None


class VoiceDesignBatchRequest(BaseModel):
    """Generate multiple voice designs concurrently for rapid character voice iteration."""
    items: list[VoiceDesignBatchItem]
    upload_to_s3: bool = True
    overwrite: bool = False


class VoiceCloneBatchItem(BaseModel):
    text: str
    filename: str

class VoiceCloneBatchRequest(BaseModel):
    """Batch generate zero-shot voice cloning from a reference audio and upload to S3."""
    ref_audio_url: str
    ref_text: str
    items: list[VoiceCloneBatchItem]
    language: str = "English"
    use_xvec: bool = False
    upload_to_s3: bool = True
    overwrite: bool = False

class VoiceCloneBatchStatusResponse(BaseModel):
    session_id: str
    status: str # processing, completed, failed
    progress_pct: float = 0.0
    total: int = 0
    completed: int = 0
    failed: int = 0
    results: list[dict] = []
    error: Optional[str] = None
    created_at: float


class VoiceCloneRequest(BaseModel):
    """Generate speech using zero-shot VoiceClone Base model."""
    text: str
    ref_audio_url: str
    ref_text: str
    language: str = "English"
    use_xvec: bool = False
    upload_to_s3: bool = True
    s3_filename: Optional[str] = None
    overwrite: bool = False

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
    job_id: Optional[str] = None
    force: bool = False

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
                ops_log.log_event("resume_job_fallback", extra={"reason": "job_not_found", "resume_job_id": req.resume_job_id})
                logger.warning(f"Job to resume ({req.resume_job_id}) not found. Falling back to default base model.")
            elif not previous_job.checkpoint_path or not os.path.exists(previous_job.checkpoint_path):
                ops_log.log_event("resume_job_fallback", extra={"reason": "checkpoint_not_found", "resume_job_id": req.resume_job_id})
                logger.warning(f"Job to resume ({req.resume_job_id}) does not have a valid checkpoint. Falling back to default base model.")
            else:
                base_model_path = previous_job.checkpoint_path

        # If job_id is provided, check if it already exists
        if req.job_id:
            existing_job = pipeline.get_job(req.job_id)
            if existing_job:
                # User refinement: Only return if status is QUEUED or TRAINING, 
                # unless force is true in which case we restart anyway.
                # All other states (FAILED, READY, etc) trigger a restart if force=False isn't enough.
                if not req.force and existing_job.status in (JobStatus.QUEUED, JobStatus.TRAINING):
                    logger.info(f"Job {req.job_id} already exists with active status {existing_job.status}. Returning existing.")
                    return JSONResponse(content=existing_job.to_dict(), status_code=200)
                else:
                    logger.warning(f"Re-creating/Retrying job {req.job_id} (Status: {existing_job.status}, Force={req.force})")

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
            job_id=req.job_id,
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
async def trigger_cleanup(threshold_gb: float = 40.0):
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
    loop = asyncio.get_running_loop()
    job = await loop.run_in_executor(None, pipeline.get_job, job_id)
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
async def retry_job(job_id: str, req: Optional[FinetuneRequest] = None):
    """Retry a job that has failed or been cancelled.
    
    If the job is not found in memory, disk, or S3, and a request body is provided,
    it will attempt to create a new job with the specified `job_id`.
    """
    job = pipeline.retry_job(job_id)
    if job:
        # If it was already READY or safely transitioned (smart retry)
        return JSONResponse(content=job.to_dict(), status_code=202 if job.status != JobStatus.READY else 200)

    # Job not found OR not in a failed/cancelled state (e.g. QUEUED, TRAINING).
    if req:
        # User refinement: Only skip retry if it's currently active (QUEUED/TRAINING)
        existing = pipeline.get_job(job_id)
        if existing and not req.force and existing.status in (JobStatus.QUEUED, JobStatus.TRAINING):
            logger.info(f"Job {job_id} already active with status {existing.status}. Skipping restart.")
            return JSONResponse(content=existing.to_dict(), status_code=200)
            
        logger.info(f"Retrying/Resurrecting job {job_id}. Status={existing.status if existing else 'None'}, Force={req.force}")
        # Re-use the creation logic but with our specific job_id
        # We need to handle the S3 download again here or refactor it.
        # For simplicity, we'll repeat the download logic since it's short.
        
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            storage.download_file(req.dataset_s3_key, tmp_path)
            
            # Resolve resume_job_id if any (unlikely for a fresh creation, but for completeness)
            base_model_path = None
            if req.resume_job_id:
                prev = pipeline.get_job(req.resume_job_id)
                if prev and prev.checkpoint_path and os.path.exists(prev.checkpoint_path):
                    base_model_path = prev.checkpoint_path

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
                job_id=job_id,
            )
            pipeline.start_job(job.job_id)
            return JSONResponse(content=job.to_dict(), status_code=202)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail=f"Failed to resurrect job {job_id}: {str(e)}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    raise HTTPException(status_code=404, detail=f"Job {job_id} not found, or it is not in a failed/cancelled state.")


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
    loop = asyncio.get_running_loop()
    job = await loop.run_in_executor(None, pipeline.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status not in (JobStatus.READY, JobStatus.RESTORING):
        # Auto-recover: if job FAILED but training completed (has S3 backup or local checkpoint),
        # transparently recover instead of returning an error
        _cp_check = str(Path(job.checkpoint_path).resolve()) if job.checkpoint_path else None
        can_recover = job.status == JobStatus.FAILED and (
            job.s3_model_key or (_cp_check and os.path.exists(_cp_check))
        )
        if not can_recover:
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} is not ready (status: {job.status}). "
                       f"Poll GET /jobs/{job_id} to check progress.",
            )

    pipeline.touch_job(job_id) # Update LRU timestamp
    pipeline._cleanup_disk_lru(30.0) # Background check usage

    # Enhanced Fast-path check
    s3_key_found = None
    if req.upload_to_s3 and not req.overwrite and req.s3_filename and storage.is_configured:
        if req.book_id and req.chapter_id:
            proper_key = f"audio/segments/{req.book_id}/{req.chapter_id}/{req.s3_filename}"
            if storage.object_exists(proper_key):
                s3_key_found = proper_key
        else:
            # Check for existing session-based audio if filename provided
            # Note: without a session_code in the request, we can't perfectly fast-path 
            # prefixed files, but we can check the legacy path for backwards compatibility
            legacy_key = f"audio/{job_id}/{req.s3_filename}"
            if storage.object_exists(legacy_key):
                s3_key_found = legacy_key

    if s3_key_found:
        presigned_url = storage.get_presigned_url(s3_key_found, expires_in=86400)
        return {
            "s3_url": storage._object_url(s3_key_found),
            "presigned_url": presigned_url,
            "s3_key": s3_key_found,
            "sample_rate": 24000,
            "text": req.text,
            "job_id": job_id,
        }

    try:
        checkpoint_path = str(job.checkpoint_path) if job.checkpoint_path else None
        # Resolve to absolute path — relative paths (e.g. "jobs/id/output/ckpt") cause
        # HuggingFace from_pretrained to misinterpret them as Hub repo IDs.
        if checkpoint_path and not os.path.isabs(checkpoint_path):
            checkpoint_path = str(Path(checkpoint_path).resolve())
        # If checkpoint is missing locally, restore from S3
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            if job.s3_model_key:
                checkpoint_path = pipeline._restore_checkpoint_from_s3(job)
            else:
                raise HTTPException(status_code=500, detail=f"Job {job_id} has no checkpoint and no S3 backup")
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
        # Construct S3 prefix based on user's structure decision (audio/{job_id}/{session_code}_{filename})
        session_code = uuid.uuid4().hex[:8]
        is_segment = bool(req.book_id and req.chapter_id)
        
        s3_prefix = f"audio/segments/{req.book_id}/{req.chapter_id}" if is_segment else f"audio/{job_id}"
        
        # Determine final filename
        final_filename = req.s3_filename
        if not is_segment:
            if not final_filename:
                # Generate a default filename if none provided
                ts = int(time.time())
                final_filename = f"audio_{ts}.wav"
            # Prefix with session code for non-segment uploads
            final_filename = f"{session_code}_{final_filename}"

        with ops_log.operation("s3_upload", job_id=job_id):
            s3_url = storage.upload_wav(wav_bytes, job_id, filename=final_filename, prefix=s3_prefix, model_id=job_id)
        
        s3_key = f"{s3_prefix}/{final_filename}"
        
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

    loop = asyncio.get_running_loop()
    job = await loop.run_in_executor(None, pipeline.get_job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status not in (JobStatus.READY, JobStatus.RESTORING):
        _cp_check = str(Path(job.checkpoint_path).resolve()) if job.checkpoint_path else None
        can_recover = job.status == JobStatus.FAILED and (
            job.s3_model_key or (_cp_check and os.path.exists(_cp_check))
        )
        if not can_recover:
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} is not ready (status: {job.status}).",
            )

    pipeline.touch_job(job_id)
    pipeline._cleanup_disk_lru(30.0)

    from functools import partial

    # Create a Semaphore to limit concurrent S3 checks and generation launching
    # Allowing up to GPU_BATCH_SIZE concurrent submissions ensures we fill our fusion batches efficiently.
    concurrency_limit = asyncio.Semaphore(GPU_BATCH_SIZE)
    
    # Generate one session code for the entire batch request so they are grouped together
    batch_session_code = uuid.uuid4().hex[:8]
    is_segment_batch = bool(req.book_id and req.chapter_id)
    s3_prefix_base = f"audio/segments/{req.book_id}/{req.chapter_id}" if is_segment_batch else f"audio/{job_id}"

    async def process_item(item, index):
        async with concurrency_limit:
            text = item.get("text", "")
            language = item.get("language", req.language)
            instruct = item.get("instruct", "")
            filename = item.get("filename", f"audio_{index:04d}.wav")
            overwrite = item.get("overwrite", req.overwrite)
            
            # Construct S3 prefix for the upload phase
            s3_prefix = s3_prefix_base

            # Enhanced Fast-path check
            s3_key_found = None
            if not overwrite and storage.is_configured:
                # We must use thread executor since boto3 is blocking
                loop = asyncio.get_running_loop()
                
                if req.book_id and req.chapter_id:
                    proper_key = f"audio/segments/{req.book_id}/{req.chapter_id}/{filename}"
                    if await loop.run_in_executor(None, storage.object_exists, proper_key):
                        s3_key_found = proper_key
                else:
                    legacy_key = f"audio/{job_id}/{filename}"
                    if await loop.run_in_executor(None, storage.object_exists, legacy_key):
                        s3_key_found = legacy_key

            if s3_key_found:
                loop = asyncio.get_running_loop()
                return {
                    "s3_url": storage._object_url(s3_key_found),
                    "presigned_url": storage.get_presigned_url(s3_key_found, expires_in=86400),
                    "s3_key": s3_key_found,
                    "sample_rate": 24000,
                    "text": text,
                    "job_id": job_id,
                }

            # Run generation using the local batcher (fuses multiple requests into one GPU pass)
            try:
                loop = asyncio.get_running_loop()

                # Resolve to absolute path — prevents HF from_pretrained treating a relative
                # path like 'jobs/id/output/checkpoint-epoch-14' as a Hub repo ID.
                checkpoint_path = str(job.checkpoint_path) if job.checkpoint_path else None
                if checkpoint_path and not os.path.isabs(checkpoint_path):
                    checkpoint_path = str(Path(checkpoint_path).resolve())

                # If the local epoch folder was LRU-evicted or never restored, fall back to S3.
                if not checkpoint_path or not os.path.exists(checkpoint_path):
                    if job.s3_model_key:
                        logger.info(f"Batch infer: checkpoint missing locally for job {job_id}, restoring from S3...")
                        checkpoint_path = await loop.run_in_executor(
                            None, pipeline._restore_checkpoint_from_s3, job
                        )
                    else:
                        raise RuntimeError(
                            f"Job {job_id} has no local checkpoint and no S3 backup to restore from."
                        )

                batcher = get_custom_voice_batcher(job_id, checkpoint_path, job.speaker_name)
                
                wav_bytes, sr = await batcher.submit(
                    texts=text,
                    languages=language,
                    instructs=instruct
                )

                # Determine final filename with session prefix
                final_filename = f"{batch_session_code}_{filename}" if not is_segment_batch else filename

                # Parallel S3 upload
                s3_url = await loop.run_in_executor(
                    None,
                    partial(storage.upload_wav, wav_bytes, job_id, filename=final_filename, prefix=s3_prefix, model_id=job_id)
                )
                
                s3_key = f"{s3_prefix}/{final_filename}"
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
# Voice Design & Voice Clone
# ---------------------------------------------------------------------------

@app.post(
    "/voice-clone",
    summary="Generate speech using zero-shot VoiceClone model",
    responses={
        200: {
            "content": {"audio/wav": {}, "application/json": {}},
            "description": "WAV audio or JSON with S3 URL",
        }
    },
)
async def voice_clone(req: VoiceCloneRequest):
    """Generate speech using zero-shot VoiceClone Base model.
    Provide a reference audio and its transcript to clone the voice.
    """
    if req.upload_to_s3 and not storage.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Storage not configured. Set E2E_ACCESS_KEY and E2E_SECRET_KEY.",
        )

    pipeline.inference._touch()
    logger.info(f"Voice clone request: ref_audio_url={req.ref_audio_url[:120]}... text='{req.text[:50]}...'")
    
    filename = req.s3_filename or f"clone_{uuid.uuid4().hex[:8]}.wav"
    s3_prefix = "audio/voice_clone"
    s3_key = f"{s3_prefix}/{filename}"

    # Fast-path check
    if not req.overwrite and req.upload_to_s3 and storage.is_configured:
        loop = asyncio.get_running_loop()
        if await loop.run_in_executor(None, storage.object_exists, s3_key):
            presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
            return {
                "s3_url": storage._object_url(s3_key),
                "presigned_url": presigned_url,
                "s3_key": s3_key,
                "sample_rate": 24000,
                "text": req.text,
                "job_id": "voice_clone",
            }

    try:
        wav_bytes, sr = await voice_clone_batcher.submit(
            texts=req.text,
            ref_audios=req.ref_audio_url,
            ref_texts=req.ref_text,
            languages=req.language,
            x_vector_only_modes=req.use_xvec,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {str(e)}")

    if req.upload_to_s3:
        loop = asyncio.get_running_loop()
        from functools import partial
        s3_url = await loop.run_in_executor(
            None,
            partial(storage.upload_wav, wav_bytes, "voice_clone", filename=filename, prefix=s3_prefix, model_id="voice_clone_qwen")
        )
        presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
        return {
            "s3_url": s3_url,
            "presigned_url": presigned_url,
            "s3_key": s3_key,
            "sample_rate": sr,
            "text": req.text,
            "job_id": "voice_clone",
        }

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Sample-Rate": str(sr),
        },
    )

@app.post(
    "/voice-clone/batch",
    summary="Batch generate zero-shot voice cloning (ASYNCHRONOUS)",
    response_model=dict,
)
async def voice_clone_batch(req: VoiceCloneBatchRequest):
    """Generate multiple audio files in parallel using zero-shot VoiceClone Base model and upload to S3.
    This is asynchronous: returns a session_id immediately, poll status via GET /voice-clone/batch/{id}.
    """
    if req.upload_to_s3 and not storage.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Storage not configured. Set E2E_ACCESS_KEY and E2E_SECRET_KEY.",
        )

    pipeline.inference._touch()
    session_id = uuid.uuid4().hex[:8]
    logger.info(f"Voice clone BATCH request: {len(req.items)} items, session_id={session_id}")

    # Initialize status
    voice_clone_batch_jobs[session_id] = {
        "session_id": session_id,
        "status": "processing",
        "total": len(req.items),
        "completed": 0,
        "failed": 0,
        "results": [],
        "error": None,
        "created_at": time.time(),
    }

    async def run_batch_task():
        concurrency_limit = asyncio.Semaphore(10)
        s3_prefix = f"audio/voice_clone/{session_id}"
        
        async def process_item(item: VoiceCloneBatchItem, index: int):
            async with concurrency_limit:
                filename = item.filename or f"clone_batch_{index:04d}.wav"
                s3_key = f"{s3_prefix}/{filename}"
                
                # S3 fast-path
                if not req.overwrite and req.upload_to_s3 and storage.is_configured:
                    loop = asyncio.get_running_loop()
                    if await loop.run_in_executor(None, storage.object_exists, s3_key):
                        presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
                        res = {
                            "s3_url": storage._object_url(s3_key),
                            "presigned_url": presigned_url,
                            "s3_key": s3_key,
                            "sample_rate": 24000,
                            "text": item.text,
                            "job_id": "voice_clone",
                        }
                        voice_clone_batch_jobs[session_id]["completed"] += 1
                        voice_clone_batch_jobs[session_id]["results"].append(res)
                        return

                # Generate via batcher
                try:
                    wav_bytes, sr = await voice_clone_batcher.submit(
                        texts=item.text,
                        ref_audios=req.ref_audio_url,
                        ref_texts=req.ref_text,
                        languages=req.language,
                        x_vector_only_modes=req.use_xvec,
                    )
                except Exception as e:
                    logger.error(f"Voice clone batch item {index} failed: {e}")
                    voice_clone_batch_jobs[session_id]["failed"] += 1
                    return

                # Upload to S3
                if req.upload_to_s3:
                    try:
                        loop = asyncio.get_running_loop()
                        from functools import partial
                        s3_url = await loop.run_in_executor(
                            None,
                            partial(storage.upload_wav, wav_bytes, "voice_clone", filename=filename, prefix=s3_prefix, model_id="voice_clone_qwen")
                        )
                        presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
                        res = {
                            "s3_url": s3_url,
                            "presigned_url": presigned_url,
                            "s3_key": s3_key,
                            "sample_rate": sr,
                            "text": item.text,
                            "job_id": "voice_clone",
                        }
                        voice_clone_batch_jobs[session_id]["completed"] += 1
                        voice_clone_batch_jobs[session_id]["results"].append(res)
                    except Exception as e:
                        logger.error(f"Voice clone S3 upload item {index} failed: {e}")
                        voice_clone_batch_jobs[session_id]["failed"] += 1
                else:
                    import base64
                    res = {
                        "s3_url": "",
                        "presigned_url": f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}",
                        "s3_key": filename,
                        "sample_rate": sr,
                        "text": item.text,
                        "job_id": "voice_clone_local",
                    }
                    voice_clone_batch_jobs[session_id]["completed"] += 1
                    voice_clone_batch_jobs[session_id]["results"].append(res)

        tasks = [process_item(item, i) for i, item in enumerate(req.items)]
        await asyncio.gather(*tasks)
        
        job = voice_clone_batch_jobs[session_id]
        if job["completed"] + job["failed"] >= job["total"]:
            job["status"] = "completed" if job["failed"] == 0 else "completed_with_errors"

    # Start background task WITHOUT waiting
    asyncio.create_task(run_batch_task())
    
    return {
        "session_id": session_id,
        "finetune_job_id": session_id,
        "status": "processing",
        "total": len(req.items)
    }

@app.get(
    "/voice-clone/batch/{session_id}",
    summary="Get status of an asynchronous voice clone batch",
    response_model=VoiceCloneBatchStatusResponse,
)
async def get_voice_clone_batch_status(session_id: str):
    """Poll for the status and results of a voice clone batch initiated via POST /voice-clone/batch."""
    job = voice_clone_batch_jobs.get(session_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Voice clone batch {session_id} not found")
    
    # Calculate progress percentage
    total = job["total"]
    completed = job["completed"]
    failed = job["failed"]
    
    progress_pct = 0.0
    if total > 0:
        progress_pct = round(((completed + failed) / total) * 100, 1)
    
    return {
        **job,
        "progress_pct": progress_pct
    }


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
    logger.info(f"Voice design request: character='{req.character_name}' text='{req.text[:50]}...' instruct='{req.instruct[:50]}...'")
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
            if not req.overwrite and storage.object_exists(s3_key):
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
            s3_url = storage.upload_wav(wav_bytes, "voice_design", filename=req.s3_filename, model_id="voice_design_qwen")
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


@app.post(
    "/voice-design/batch",
    summary="Batch generate multiple voice designs concurrently",
)
async def voice_design_batch(req: VoiceDesignBatchRequest):
    """Generate multiple voice designs in parallel for rapid character voice iteration.

    Submit up to MAX_CONCURRENT_VOICE_DESIGNS items at once. Each item gets a
    different voice description (instruct) and text, and they are processed
    concurrently through the GPU batcher for maximum throughput.

    Example request:
    ```json
    {
        "items": [
            {"text": "Hello world", "instruct": "A warm male voice", "character_name": "Hero"},
            {"text": "Hello world", "instruct": "A young female voice", "character_name": "Heroine"},
            {"text": "Hello world", "instruct": "A deep gravelly voice", "character_name": "Villain"}
        ],
        "upload_to_s3": true
    }
    ```
    """
    logger.info(f"Voice design BATCH request: {len(req.items)} items")
    if not req.items:
        raise HTTPException(status_code=400, detail="No items provided.")

    if len(req.items) > MAX_CONCURRENT_VOICE_DESIGNS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many items ({len(req.items)}). Maximum is {MAX_CONCURRENT_VOICE_DESIGNS}.",
        )

    if req.upload_to_s3 and not storage.is_configured:
        raise HTTPException(status_code=503, detail="Storage not configured.")

    from functools import partial

    concurrency_limit = asyncio.Semaphore(MAX_CONCURRENT_VOICE_DESIGNS)

    async def process_item(item: VoiceDesignBatchItem, index: int):
        async with concurrency_limit:
            # Build S3 filename with character prefix (same logic as single endpoint)
            s3_filename = item.s3_filename
            parts = []
            if item.character_name:
                safe_name = "".join(
                    c for c in item.character_name if c.isalnum() or c in ("-", "_", " ")
                ).strip().replace(" ", "_")
                if safe_name:
                    parts.append(safe_name)
            if item.character_uuid:
                parts.append(item.character_uuid)

            if parts:
                prefix = "_".join(parts)
                if s3_filename:
                    if not s3_filename.startswith(prefix):
                        s3_filename = f"{prefix}_{s3_filename}"
                else:
                    s3_filename = f"{prefix}.wav"
            elif not s3_filename:
                import uuid as _uuid
                s3_filename = f"voice_design_{_uuid.uuid4().hex[:8]}.wav"

            # S3 fast-path: skip if file already exists
            if req.upload_to_s3 and not req.overwrite and storage.is_configured:
                s3_key = f"audio/voice_design/{s3_filename}"
                loop = asyncio.get_running_loop()
                exists = await loop.run_in_executor(None, storage.object_exists, s3_key)
                if exists:
                    presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
                    return {
                        "index": index,
                        "status": "skipped",
                        "s3_url": storage._object_url(s3_key),
                        "presigned_url": presigned_url,
                        "s3_key": s3_key,
                        "sample_rate": 24000,
                        "text": item.text,
                        "instruct": item.instruct,
                        "character_name": item.character_name,
                    }

            # Generate via the existing batcher
            try:
                wav_bytes, sr = await voice_design_batcher.submit(
                    texts=item.text,
                    instructs=item.instruct,
                    languages=item.language,
                )
            except Exception as e:
                logger.error(f"Voice design batch item {index} failed: {e}")
                return {
                    "index": index,
                    "status": "failed",
                    "error": str(e),
                    "text": item.text,
                    "instruct": item.instruct,
                    "character_name": item.character_name,
                }

            # Upload to S3
            if req.upload_to_s3:
                loop = asyncio.get_running_loop()
                with ops_log.operation("s3_upload", extra={"type": "voice_design_batch"}):
                    s3_url = await loop.run_in_executor(
                        None,
                        partial(storage.upload_wav, wav_bytes, "voice_design", filename=s3_filename, model_id="voice_design_qwen"),
                    )
                s3_key = f"audio/voice_design/{s3_filename}"
                presigned_url = storage.get_presigned_url(s3_key, expires_in=86400)
                return {
                    "index": index,
                    "status": "success",
                    "s3_url": s3_url,
                    "presigned_url": presigned_url,
                    "s3_key": s3_key,
                    "sample_rate": sr,
                    "text": item.text,
                    "instruct": item.instruct,
                    "character_name": item.character_name,
                }

            # No S3 upload — encode WAV bytes as base64 for JSON response
            import base64
            return {
                "index": index,
                "status": "success",
                "audio_base64": base64.b64encode(wav_bytes).decode(),
                "sample_rate": sr,
                "text": item.text,
                "instruct": item.instruct,
                "character_name": item.character_name,
            }

    with ops_log.operation("voice_design_batch_api", extra={
        "item_count": len(req.items),
        "upload_to_s3": req.upload_to_s3,
    }):
        tasks = [process_item(item, i) for i, item in enumerate(req.items)]
        results = await asyncio.gather(*tasks)

    return {
        "total": len(results),
        "succeeded": sum(1 for r in results if r.get("status") == "success"),
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }


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


@app.post("/gpu/terminate", summary="Request instance termination")
async def gpu_terminate():
    """Request that the instance be terminated once all background tasks are done.
    
    This creates a signal file that the GPU watchdog monitors. The watchdog will
    trigger a full instance termination via the Massed Compute API as soon as 
    all active operations (S3 uploads, etc.) are finished.
    """
    global IS_DRAINING
    IS_DRAINING = True
    
    signal_file = "terminate_signal.tmp"
    with open(signal_file, "w") as f:
        f.write(str(time.time()))
    
    ops_log.log_event("termination_requested")
    logger.warning("Instance termination requested via API — server entered DRAINING mode.")
    
    return {
        "status": "termination_scheduled",
        "detail": "Server is now DRAINING. New work will be rejected. Instance will terminate once current tasks complete.",
        "signal_file": signal_file,
        "instance_uuid": (
            os.getenv("GPU_INSTANCE_ID", "").strip() or 
            os.getenv("MASSED_COMPUTE_INSTANCE_UUID", "").strip() or 
            os.getenv("GPU_INSTANCE_UUID", "").strip() or 
            "not_set_in_env"
        ),
    }


@app.post("/gpu/cancel-terminate", summary="Cancel pending instance termination")
async def gpu_cancel_terminate():
    """Cancel a pending termination request.
    
    This resets the draining mode, allowing the API to accept new work again,
    and deletes the signal file monitored by the GPU watchdog.
    """
    global IS_DRAINING
    IS_DRAINING = False
    
    signal_file = "terminate_signal.tmp"
    removed = False
    if os.path.exists(signal_file):
        try:
            os.remove(signal_file)
            removed = True
        except Exception as e:
            logger.error(f"Failed to remove signal file {signal_file}: {e}")
    
    ops_log.log_event("termination_cancelled")
    logger.info("Instance termination CANCELLED — server resumed normal operation.")
    
    return {
        "status": "termination_cancelled",
        "detail": "Server is now ACTIVE. New work is accepted. Watchdog termination halted.",
        "signal_file_removed": removed,
    }


# ---------------------------------------------------------------------------
# Session-Based Inference (Event-Driven)
# ---------------------------------------------------------------------------

class SessionCharacterInfo(BaseModel):
    job_id: str
    character_name: str
    line_count: int = 0
    avg_word_count: int = 20

class SessionPrepareRequest(BaseModel):
    """Prepare a session: pre-load models, create per-character queues."""
    session_id: Optional[str] = None  # Auto-generated if not provided
    characters: list[SessionCharacterInfo]
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None

class SessionSubmitItem(BaseModel):
    job_id: str
    character_name: str
    text: str
    language: str = "English"
    instruct: str = ""
    s3_filename: str = ""
    book_id: Optional[str] = None
    chapter_id: Optional[str] = None

class SessionSubmitBatchRequest(BaseModel):
    """Submit multiple inference messages to session queues."""
    items: list[SessionSubmitItem]


@app.post("/session/prepare", summary="Prepare a session for event-driven inference")
async def session_prepare(req: SessionPrepareRequest):
    """Pre-load models into VRAM, create per-character queues, allocate replicas.

    Call this BEFORE submitting any inference messages. The API will:
    1. Calculate how many GPU replicas each character needs (based on line_count)
    2. Pre-load all model checkpoints (restoring from S3 if needed)
    3. Create per-character worker pools

    Returns the session plan with replica allocations and estimated timing.
    """
    import uuid as _uuid
    session_id = req.session_id or str(_uuid.uuid4())

    try:
        session = await session_mgr.prepare_session(
            session_id=session_id,
            characters=[c.model_dump() for c in req.characters],
            book_id=req.book_id or "",
            chapter_id=req.chapter_id or "",
        )

        model_plan = {}
        for job_id, plan in session.character_plans.items():
            model_plan[job_id] = {
                "character_name": plan.character_name,
                "replicas": plan.replicas,
                "replica_keys": plan.replica_keys,
                "status": "loaded",
            }

        return {
            "session_id": session_id,
            "status": session.status.value,
            "model_plan": model_plan,
            "total_lines": session.total_lines,
            "workers_started": len(session.workers),
            "vram": pipeline.inference.get_vram_budget(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session prepare failed: {str(e)}")


@app.post("/session/{session_id}/submit", summary="Submit a single inference message")
async def session_submit_single(session_id: str, item: SessionSubmitItem):
    """Submit a single inference message to the session's character queue."""
    try:
        enqueued = await session_mgr.submit_messages(
            session_id, [item.model_dump()]
        )
        return {"enqueued": enqueued, "session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/session/{session_id}/submit/batch", summary="Submit multiple inference messages")
async def session_submit_batch(session_id: str, req: SessionSubmitBatchRequest):
    """Submit multiple inference messages to the session's character queues.

    Messages are automatically routed to the correct character's queue based on job_id.
    Workers will batch them for GPU inference automatically.
    """
    try:
        enqueued = await session_mgr.submit_messages(
            session_id, [item.model_dump() for item in req.items]
        )
        session = session_mgr.get_session(session_id)
        return {
            "enqueued": enqueued,
            "session_id": session_id,
            "queue_depths": {
                plan.character_name: session.character_queues[jid].qsize()
                for jid, plan in session.character_plans.items()
            } if session else {},
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/session/{session_id}/status", summary="Get session progress")
async def session_status(session_id: str, include_results: bool = False):
    """Poll session progress. Returns per-character completion stats.

    Set include_results=true to also get the list of completed S3 URLs.
    """
    session = session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    response = session.to_dict()

    if include_results:
        response["results"] = session.get_all_results()

    # Add queue depths
    response["queue_depths"] = {
        session.character_plans[jid].character_name: q.qsize()
        for jid, q in session.character_queues.items()
    }

    return response


@app.delete("/session/{session_id}", summary="Teardown a session")
async def session_teardown(session_id: str):
    """Stop workers, release GPU replicas, and clean up queues.

    Primary models may remain cached (subject to normal LRU eviction)
    but replicas are immediately freed.
    """
    success = await session_mgr.teardown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {
        "detail": f"Session {session_id} torn down",
        "vram": pipeline.inference.get_vram_budget(),
    }


@app.get("/sessions", summary="List all sessions")
async def list_sessions():
    """List all sessions with basic progress info."""
    return session_mgr.list_sessions()


@app.get("/gpu/vram", summary="VRAM budget info")
async def gpu_vram():
    """Get detailed VRAM budget including session-pinned models."""
    return pipeline.inference.get_vram_budget()


@app.get("/gpu/metrics", summary="Current resource utilization (CPU, GPU, RAM, VRAM)")
async def gpu_metrics():
    """Get real-time snapshot of system resources."""
    return metrics_collector.get_latest()


@app.get("/gpu/metrics/history", summary="Historical resource utilization")
async def gpu_metrics_history(
    limit: Optional[int] = Query(60, description="Max records to return"),
    start_ts: Optional[str] = Query(None, description="Start ISO timestamp filter"),
    end_ts: Optional[str] = Query(None, description="End ISO timestamp filter"),
):
    """Get historical resource utilization records with optional time filtering."""
    return metrics_collector.get_history(limit=limit, start_ts=start_ts, end_ts=end_ts)


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


@app.get("/logs/stream", summary="Stream server logs in real-time")
async def stream_logs(request: Request):
    """Stream server logs using Server-Sent Events (SSE).
    
    This endpoint provides a real-time feed of all server logs, including 
    structured operations from ops_logger.
    """
    queue = asyncio.Queue(maxsize=100)
    log_queues.append(queue)
    
    async def log_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                
                log_entry = await queue.get()
                yield f"data: {log_entry}\n\n"
        finally:
            if queue in log_queues:
                log_queues.remove(queue)

    return StreamingResponse(log_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
