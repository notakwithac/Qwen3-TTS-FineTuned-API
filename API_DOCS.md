# Qwen3-TTS Fine-Tuning API — Documentation

> **Base URL**: `http://<TIR_GPU_IP>:8000`
> **Interactive docs**: `http://<TIR_GPU_IP>:8000/docs` (Swagger UI)

---

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Quick Start](#quick-start)
  - [Disk Management (LRU)](#disk-management-lru)
  - [Authentication](#authentication)
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

---

## Technical Reference

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_MAX_MODELS` | `4` | Max voices kept hot in VRAM |
| `GPU_IDLE_TIMEOUT` | `300` | Seconds before auto-unload |
| `E2E_BUCKET` | `qwen3-tts` | Default S3 bucket |

### Error Codes
| Status | Meaning |
|--------|---------|
| `400` | Invalid request or dataset |
| `404` | Job/Session not found |
| `409` | Job not ready (still training) |
| `503` | S3 Storage unavailable |
