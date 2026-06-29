# Qwen3-TTS Fine-Tuning API — Documentation

> **Base URL**: `http://<TIR_GPU_IP>:8000`
> **Interactive docs**: `http://<TIR_GPU_IP>:8000/docs` (Swagger UI)

---

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Quick Start](#quick-start)
  - [Disk Management (LRU)](#disk-management-lru)
  - [Authentication](#authentication)
  - [Lazy vLLM Proxies](#lazy-vllm-proxies)
  - [External GPU Leases](#external-gpu-leases)
- [Job Lifecycle (Finetuning)](#job-lifecycle-finetuning)
  - [POST /finetune](#post-finetune)
  - [GET /jobs/{job_id}](#get-jobsjob_id)
  - [GET /jobs](#get-jobs)
  - [POST /jobs/{job_id}/retry](#post-jobsjob_idretry)
  - [DELETE /jobs/{job_id}](#delete-jobsjob_id)
- [Speech Generation (Inference)](#speech-generation-inference)
  - [POST /infer/{job_id}](#post-inferjob_id)
  - [POST /infer/{job_id}/batch](#post-inferjob_idbatch)
  - [POST /voice-design](#post-voice-design)
  - [POST /voice-clone/batch](#post-voice-clonebatch)
- [Resource Monitoring & Diagnostics](#resource-monitoring--diagnostics)
  - [GET /gpu/metrics](#get-gpumetrics)
  - [GET /gpu/metrics/history](#get-gpumetricshistory)
  - [Operations Logging (/ops/*)](#operations-logging-ops)
- [Storage & Infrastructure](#storage--infrastructure)
  - [Storage Configuration](#storage-configuration)
  - [GPU Management & VRAM](#gpu-management--vram)
- [Technical Reference](#technical-reference)
  - [Environment Variables](#environment-variables)
  - [Error Codes](#error-codes)

---

## Core Concepts

### Quick Start

```bash
# 1. Setup on TIR (one-time — downloads models + installs deps)
bash setup_tir.sh

# 2. Configure storage (optional, for S3 features)
export E2E_ACCESS_KEY=your_key
export E2E_SECRET_KEY=your_secret

# 3. Start the API
bash start_api.sh
```

### Disk Management (LRU)

The API is optimized for instances with limited storage (e.g., 50GB SSD).

1. **Automated Cleanup**:
   - Raw datasets and intermediate checkpoints (optimizer states) are **purged immediately** after training to save space.
   - Local model weights (~3.5GB) are kept in a **Disk LRU** cache. 
   - When the `jobs/` directory exceeds **20GB**, the oldest models (by last access time) are automatically removed.
   - All models are safely backed up to S3 before local deletion.
   - For S3-backed jobs, only the heavy `output/` directory is deleted — `job.json` is kept locally and on S3 for fast lookups.

2. **Manual Cleanup**:
   ```
   GET /gpu/cleanup?threshold_gb=20.0
   ```
   Triggers the LRU pruning process manually.

### Authentication

No authentication is required by default. For production, add API key auth at the reverse proxy level (nginx/caddy) or add FastAPI middleware.

### Lazy vLLM Proxies

Qwen3-TTS can front Gemma e4b and Sarvam Translate through lazy vLLM
subprocesses. Plain `docker compose up --build` starts the API. Startup can
pre-download model files into the Hugging Face cache, but Gemma/Sarvam load into
GPU only when requested.

GPU residency rule:

- Starting Gemma stops Sarvam and unloads resident Qwen models.
- Starting Sarvam stops Gemma and unloads resident Qwen models.
- Loading any Qwen TTS model stops Gemma/Sarvam first.

Startup cache knobs:

- `PREFETCH_MODELS_ON_START=1`
- `PREFETCH_MODEL_SET=base voice_design tokenizer gemma sarvam_translate`

Gemma OpenAI-compatible routes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /gemma/status`
- `GET /vllm/status`

Generation calls hold the same GPU limiter used by TTS inference. This lets
Pathnam use one backend without bypassing Qwen3-TTS admission control.

```bash
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "e4b",
    "messages": [
      {"role": "user", "content": "Return one short narration note."}
    ],
    "max_tokens": 128
  }'
```

Sarvam translation route:

```bash
curl -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Every voice carries a place with it.",
    "source_language": "English",
    "target_language": "Hindi",
    "max_new_tokens": 256
  }'
```

### External GPU Leases

An authenticated lease temporarily reserves every inference permit for one
external GPU worker. Acquiring a lease stops managed vLLM processes and unloads
resident Qwen models before returning the opaque lease token. Requests must send
the service key in `X-GPU-Lease-Key`; keys are never accepted in query strings.

```http
POST   /gpu/leases
POST   /gpu/leases/{token}/heartbeat
DELETE /gpu/leases/{token}
GET    /gpu/leases/status
```

Acquire with `{"owner":"dataset-lab","ttl_seconds":120}`. Heartbeats may send
`{"ttl_seconds":120}` to extend expiry. TTL values must be between 30 and 900
seconds. Status responses expose ownership and expiry metadata but never the
lease token. Release is idempotent after a lease is no longer active.

---

## Job Lifecycle (Finetuning)

### POST /finetune
**Start a fine-tuning job.**

```http
POST /finetune
Content-Type: application/json
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_s3_key` | string | *required* | S3 key to the dataset .zip |
| `speaker_name` | string | `speaker_custom` | Name for the fine-tuned voice |
| `num_epochs` | int | `15` | Training epochs |
| `batch_size` | int | `2` | Batch size |
| `lr` | float | `1e-6` | Learning rate |
| `job_id` | string | null | Optional: reuse a specific job ID |
| `force` | bool | `false` | If `true`, restarts the job even if it already exists |

**Response** (HTTP 202):
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "speaker_name": "hero_voice"
}
```

### GET /jobs/{job_id}
**Get current status and progress of a job.**

**Status transitions:**
`queued → preparing → training → loading → ready`

**Response (during training):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "training",
  "progress": {
    "epoch": 5,
    "total_epochs": 10,
    "loss": 0.0234
  }
}
```

### GET /jobs
**List all jobs and their current statuses.**

### POST /jobs/{job_id}/retry
**Retry or restart a job.**

**Smart behavior**: 
- If `force=false` (default): Only retries if the job failed or was cancelled. If training finished, it skips to loading.
- If `force=true`: Restarts the full pipeline from scratch even if the job is active or completed.

### DELETE /jobs/{job_id}
**Cancel a running job or delete a completed one (including files).**

---

## Speech Generation (Inference)

### POST /infer/{job_id}
**Generate single audio using a fine-tuned model.**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize |
| `upload_to_s3` | bool | `true` | Upload result to S3 |
| `s3_filename` | string | auto | Custom filename |
| `overwrite` | bool | `false` | Skip Generation if file exists on S3 |

### POST /infer/{job_id}/batch
**Generate multiple audio files in parallel and upload all to S3.**

### POST /voice-design
**Generate speech from a voice description (e.g. "Warm male voice") without needing a dataset.**

### POST /voice-clone/batch
**Batch generate speech from a single reference audio (Zero-shot cloning).**

### POST /dataset/prepare
**Prepare raw finetuning dataset items from validated clone audio.**

This endpoint now stops after creating raw `dataset_items` for review. It does not auto-package the final Qwen dataset, even if `approval_mode` is `"auto"`.

Notes:
- `approval_mode` is accepted for backward compatibility.
- The response job reaches `phase="awaiting_approval"` when raw dataset items are ready.
- `dataset_items` are candidate clips for approval/editing before packaging.

### POST /dataset/package
**Package approved dataset items into the final Qwen finetune zip.**

This endpoint expects already-approved `dataset_items`.

Notes:
- The service trusts the incoming `dataset_items`.
- If one item has `is_reference=true`, that clip becomes `data/ref_audio.wav`.
- If none is marked, the service falls back to the last included clip.

### GET /dataset/status/{job_id}
**Get status for dataset preparation or packaging jobs.**

---

## Resource Monitoring & Diagnostics

### GET /gpu/metrics
**Get a real-time snapshot of system resource utilization.**

**Response:**
```json
{
  "timestamp": "2024-04-03T18:12:48.521078Z",
  "cpu": {
    "percent": 29.8,
    "count": 28
  },
  "ram": {
    "total_gb": 31.85,
    "used_gb": 19.51,
    "percent": 61.3
  },
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA A100-SXM4-40GB",
      "utilization_percent": 45,
      "vram_total_gb": 40.0,
      "vram_used_gb": 12.5,
      "vram_percent": 31.25
    }
  ]
}
```

### GET /gpu/metrics/history
**Get historical resource utilization records.**

**Query Parameters:**
- `limit` (int, default 60): Max records to return.
- `start_ts` (string): ISO timestamp to filter start.
- `end_ts` (string): ISO timestamp to filter end.

### Operations Logging (/ops/*)
- `GET /ops/averages`: Grouped timing metrics for all pipeline stages.
- `GET /ops/history`: Detailed log of individual operations.
- `GET /ops/running`: Currently active background tasks.

---

## Storage & Infrastructure

### Storage Configuration
- `GET /storage/status`: Verify S3 endpoint and credentials.
- `GET /storage/list/{job_id}`: List all audio generated for a specific job.

### GPU Management & VRAM
- `GET /gpu/status`: VRAM usage and model loading state.
- `GET /gpu/vram`: Direct access to VRAM budget/allocations.
- `POST /gpu/unload`: Manually free GPU memory.
- `PUT /gpu/config`: Update idle timeout for auto-unload.
- `GET /gpu/concurrency`: Show the live runtime GPU concurrency and shared-replica targets.
- `POST /gpu/concurrency`: Update runtime GPU concurrency in memory without redeploying.

**Startup preload behavior:**
- On boot, the server attempts to warm one shared `VoiceDesign` model and one shared `Base` model before serving requests.
- These preloaded shared models are not pinned. They remain normal cache entries and can still be evicted later if custom checkpoints need VRAM.
- If a preload step fails, startup continues and the server falls back to lazy loading for that model type.

**Runtime concurrency payload example:**
```json
{
  "gpu_max_models": 7,
  "voice_design_replicas": 2,
  "voice_clone_replicas": 2,
  "shared_model_min_headroom_gb": 4
}
```

**Notes:**
- `.env` values are startup defaults only.
- `POST /gpu/concurrency` overrides the live process immediately.
- Aggressive values are allowed, but actual replica admission is still gated by runtime VRAM headroom.
- Configured shared replica targets can be higher than the effective loaded replica count. The runtime will only materialize additional shared replicas when true free VRAM still satisfies the headroom rule.
- For current throughput tuning, hot resident models plus large batch requests are usually more effective than pushing per-model replica counts upward until the GPU fragments or stalls.

**Suggested tuning workflow:**
1. Boot the server with conservative `.env` defaults.
2. Call `GET /gpu/concurrency` to inspect the live baseline.
3. Call `POST /gpu/concurrency` with your test values.
4. Run `python stress_shared_model_replicas.py --base-url http://<TIR_GPU_IP>:8000`.
5. Run `python stress_logs_stream.py --base-url http://<TIR_GPU_IP>:8000`.
6. Inspect `GET /gpu/vram`, `GET /gpu/metrics`, and `GET /ops/history?limit=100`.

---

## Technical Reference

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_MAX_MODELS` | `auto` / `7` | Startup capacity budget for loaded models, not an automatic shared-replica count. Keep this high when custom checkpoints also need to share the GPU |
| `VOICE_DESIGN_REPLICAS` | `1` | Throughput-mode startup target for shared VoiceDesign replicas. The server preloads `replica-0` at boot and may load more later only if headroom allows |
| `VOICE_CLONE_REPLICAS` | `1` | Throughput-mode startup target for shared VoiceClone replicas. Prefer batching before increasing this target |
| `SHARED_MODEL_MIN_HEADROOM_GB` | `4` | Minimum free VRAM headroom required before loading another shared replica |
| `GPU_IDLE_TIMEOUT` | `0` | Throughput-mode recommendation for keeping the hot shared models resident. Use a larger value if you still want eventual idle unload |
| `GPU_LEASE_API_KEY` | unset | Service key enabling exclusive external GPU lease endpoints |
| `GPU_LEASE_DEFAULT_TTL_SECONDS` | `120` | Default lease TTL; requests are restricted to 30..900 seconds |
| `E2E_BUCKET` | `qwen3-tts` | Default S3 bucket |

**Throughput-mode guidance:**
- Keep `GPU_MAX_MODELS=7` so custom fine-tuned checkpoints can still load when needed.
- Start with `VOICE_DESIGN_REPLICAS=1` and `VOICE_CLONE_REPLICAS=1`; these are warm defaults, not guarantees that more replicas will fit.
- Keep `GPU_IDLE_TIMEOUT=0` or very large if you want startup preload and `torch.compile` warm-up costs to be paid once per process boot instead of repeatedly after idle unload.
- Use `/voice-clone/batch` and other batch paths to increase throughput. A single loaded replica still serializes the actual generate call, so batching is the main throughput lever on one GPU.

### Manual Stress Utilities
- `python stress_shared_model_replicas.py --base-url http://<TIR_GPU_IP>:8000`
  - Seeds real VoiceDesign audio in S3, then runs clone-only and mixed design+clone waves while polling clone batch completion.
- `python stress_logs_stream.py --base-url http://<TIR_GPU_IP>:8000`
  - Keeps `/logs/stream` open during live GPU work and asserts that design/clone activity appears in the stream.

### Error Codes
| Status | Meaning |
|--------|---------|
| `400` | Invalid request or dataset |
| `404` | Job/Session not found |
| `409` | Job not ready (still training) |
| `503` | S3 Storage unavailable |
