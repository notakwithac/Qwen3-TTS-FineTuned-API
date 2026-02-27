# Qwen3-TTS Fine-Tuning API — Documentation

> **Base URL**: `http://<TIR_GPU_IP>:8000`
> **Interactive docs**: `http://<TIR_GPU_IP>:8000/docs` (Swagger UI)

---

## Quick Start

```bash
# 1. Setup on TIR (one-time — downloads models + installs deps)
bash setup_tir.sh

# 2. Configure storage (optional, for S3 features)
export E2E_ACCESS_KEY=your_key
export E2E_SECRET_KEY=your_secret

# 3. Start the API
bash start_api.sh
```

---

## Disk Management (LRU)

The API is optimized for instances with limited storage (e.g., 50GB SSD).

1. **Automated Cleanup**:
   - Raw datasets and intermediate checkpoints (optimizer states) are **purged immediately** after training to save space.
   - Local model weights (~3.5GB) are kept in a **Disk LRU** cache. 
   - When the `jobs/` directory exceeds **20GB**, the oldest models (by last access time) are automatically removed.
   - All models are safely backed up to S3 before local deletion.

2. **Manual Cleanup**:
   ```
   GET /gpu/cleanup?threshold_gb=20.0
   ```
   Triggers the LRU pruning process manually.

---

## Authentication

No authentication is required by default. If you need to add API key auth
for production, set it at the reverse proxy level (nginx/caddy) or
add FastAPI middleware.

---

## Endpoints

### Health Check

```
GET /
```

**Response:**
```json
{
  "service": "qwen3-tts-finetune-api",
  "status": "ok",
  "storage_configured": true
}
```

---

### Fine-Tuning

#### Start a fine-tuning job

```
POST /finetune
Content-Type: application/json
```

**Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_s3_key` | string | *required* | S3 key to the dataset .zip (e.g. `datasets/Hero1/dataset.zip`) |
| `speaker_name` | string | `speaker_custom` | Name for the fine-tuned voice (Avoid `_` + number) |
| `num_epochs` | int | `15` | Training epochs |
| `batch_size` | int | `2` | Batch size (increase on larger GPUs) |
| `lr` | float | `1e-6` | Learning rate |
| `book_id` | string | `null` | Optional book ID for S3 organization |
| `chapter_id` | string | `null` | Optional chapter ID for S3 organization |
| `character_id` | string | `null` | Optional character ID for local directory organization (`jobs/{job_id}`) |
| `resume_job_id` | string | `null` | Optional ID of a previous job to continue training from. Falls back to base model if not found. |

**Dataset zip format:**
```
dataset.zip
├── train.jsonl
└── data/
    ├── seg_00000001.wav
    ├── seg_00000002.wav
    ├── ...
    └── ref_audio.wav
```

**train.jsonl format** (one JSON object per line):
```json
{"audio": "./data/seg_00000001.wav", "text": "Hello world.", "ref_audio": "./data/ref_audio.wav"}
{"audio": "./data/seg_00000002.wav", "text": "Goodbye world.", "ref_audio": "./data/ref_audio.wav"}
```

**Example:**
```bash
curl -X POST http://localhost:8000/finetune \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_s3_key": "datasets/Hero1/dataset.zip",
    "speaker_name": "hero_voice",
    "num_epochs": 15,
    "batch_size": 2,
    "lr": 1e-6
  }'
```

**Response** (HTTP 202):
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "queued",
  "speaker_name": "hero_voice",
  "config": { "num_epochs": 15, "batch_size": 2, "lr": 1e-6, "flash_attn": true }
}
```

---

#### Check job status

```
GET /jobs/{job_id}
```

**Status transitions:**
```
queued → preparing → training → loading → ready
                                         ↘ failed
```

**Example:**
```bash
curl http://localhost:8000/jobs/a1b2c3d4e5f6
```

**Response (during training):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "training",
  "progress": {
    "stage": "training",
    "epoch": 5,
    "total_epochs": 10,
    "step": 12,
    "loss": 0.0234,
    "global_step": 42
  }
}
```

**Response (when ready):**
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "ready",
  "inference_url": "/infer/a1b2c3d4e5f6",
  "checkpoint_path": "jobs/a1b2c3d4e5f6/output/checkpoint-epoch-9"
}
```

---

#### List all jobs

```
GET /jobs
```

Returns an array of all job objects.

---

#### Delete a job

```
DELETE /jobs/{job_id}
```

Cancels a running job or deletes a completed one (including files).

---

### Speech Generation (Inference)

### Secure Private Storage

Since your bucket is configured with **blocked public access** and **ACLs disabled** (the recommended security posture), the API provides a `presigned_url` for every generated audio file.

- `s3_url`: The permanent static URL (Reference/Internal use).
- `presigned_url`: A temporary access URL (Valid for 24 hours). Useful for web players.
- `s3_key`: The unique path in your bucket.

#### Example Response
```json
{
  "s3_url": "https://pathnam-ai.s3.amazonaws.com/audio/job_abc/tts_123.wav",
  "presigned_url": "https://pathnam-ai.s3.amazonaws.com/audio/job_abc/tts_123.wav?X-Amz-Signature=...",
  "s3_key": "audio/job_abc/tts_123.wav",
  "sample_rate": 24000,
  "text": "Hello world",
  "job_id": "job_abc"
}
```

#### Generate single audio

```
POST /infer/{job_id}
Content-Type: application/json
```

**Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize |
| `language` | string | `English` | Language name |
| `instruct` | string | `""` | Style/emotion instruction |
| `upload_to_s3` | bool | `true` | Upload result to S3 (Default) |
| `s3_filename` | string | auto | Custom filename for S3 upload |
| `book_id` | string | null | Optional: Triggers path `audio/segments/{book_id}/` |
| `chapter_id` | string | null | Optional: Used with book_id for chapter path |

**Example — get WAV directly:**
```bash
curl -X POST http://localhost:8000/infer/a1b2c3d4e5f6 \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, welcome to our audiobook.", "language": "English"}' \
  --output hello.wav
```

**Example — upload to S3:**
```bash
curl -X POST http://localhost:8000/infer/a1b2c3d4e5f6 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, welcome to our audiobook.",
    "language": "English",
    "upload_to_s3": true,
    "book_id": "book_001",
    "chapter_id": "chapter_1",
    "s3_filename": "intro.wav"
  }'
```

**Response (Integrated Structure):**
```json
{
  "s3_url": "https://pathnam-ai.s3.amazonaws.com/audio/segments/book_001/chapter_1/intro.wav",
  "presigned_url": "...",
  "s3_key": "audio/segments/book_001/chapter_1/intro.wav",
  "sample_rate": 24000,
  "text": "Hello, welcome to our audiobook.",
  "job_id": "a1b2c3d4e5f6"
}
```

---

#### Batch generate + S3 upload

```
POST /infer/{job_id}/batch
Content-Type: application/json
```

Generate multiple audio files in parallel and upload all to S3 in one call.
**Designed for high-throughput microservice integration.**

The GPU tasks are internally queued via a semaphore, while the S3 uploads happen in parallel.
**Batch processing speed is significantly improved.**

**Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `items` | array | *required* | List of `{text, filename, language?, instruct?}` |
| `language` | string | `English` | Default language for all items |
| `book_id` | string | null | Triggers `audio/segments/{book_id}/` structure |
| `chapter_id` | string | null | Triggers `audio/segments/.../{chapter_id}/` |

**Example:**
```bash
curl -X POST http://localhost:8000/infer/a1b2c3d4e5f6/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"text": "Chapter one. The adventure begins.", "filename": "ch1_001.wav"},
      {"text": "Our hero stood at the crossroads.", "filename": "ch1_002.wav"},
      {"text": "A cold wind blew from the north.", "filename": "ch1_003.wav"}
    ],
    "language": "English"
  }'
```

**Response:**
```json
[
  {
    "s3_url": "https://objectstore.e2enetworks.net/qwen3-tts/audio/a1b2c3d4e5f6/ch1_001.wav",
    "s3_key": "audio/a1b2c3d4e5f6/ch1_001.wav",
    "sample_rate": 24000,
    "text": "Chapter one. The adventure begins.",
    "job_id": "a1b2c3d4e5f6"
  },
  ...
]
```

---

### Voice Design (no fine-tuning needed)

#### Generate speech from a voice description

```
POST /voice-design
Content-Type: application/json
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize |
| `instruct` | string | *required* | Voice description |
| `language` | string | `English` | Language |
| `upload_to_s3` | bool | `false` | Upload to S3 |
| `s3_filename` | string | auto | Custom filename |

**Example:**
```bash
curl -X POST http://localhost:8000/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to this audiobook.",
    "instruct": "A warm male voice, middle-aged, calm and authoritative",
    "language": "English"
  }' --output voice_design.wav
```

> **Note**: The VoiceDesign model auto-loads on first call (~10-15s). Only one model
> (VoiceDesign or CustomVoice) can be in VRAM at a time — they swap automatically.

---

### Voice Cloning (Zero-shot)

#### Batch Voice Cloning
Generate multiple speech audio files in parallel from a single reference audio and text. Uses the Qwen3-TTS Base model.

```
POST /voice-clone/batch
Content-Type: application/json
```

**Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ref_audio_url` | string | *required* | URL or base64 of the reference audio for voice cloning |
| `ref_text` | string | *required* | Transcript of the reference audio (used for ICL mode) |
| `items` | array | *required* | List of `{text, filename?}` to generate |
| `language` | string | `English` | Target language |
| `use_xvec` | bool | `false` | If true, uses only speaker embedding (ignores `ref_text`) |
| `upload_to_s3` | bool | `true` | Upload outputs to S3 |
| `overwrite` | bool | `false` | Overwrite existing files on S3 |

**Example:**
```bash
curl -X POST http://localhost:8000/voice-clone/batch \
  -H "Content-Type: application/json" \
  -d '{
    "ref_audio_url": "https://example.com/audio/sample.wav",
    "ref_text": "This is the reference spoken text.",
    "items": [
        {"text": "First sentence to generate in this voice.", "filename": "clone_001.wav"},
        {"text": "Second sentence to generate concurrently.", "filename": "clone_002.wav"}
    ],
    "language": "English",
    "upload_to_s3": true
  }'
```

**Response:**
Returns a JSON array containing objects similar to the standard `InferS3Response`, with `s3_url` and `presigned_url` for each item.

---

### Storage (E2E Object Storage)

#### Check storage configuration

```
GET /storage/status
```

```json
{
  "configured": true,
  "endpoint": "https://objectstore.e2enetworks.net",
  "bucket": "qwen3-tts"
}
```

#### List files for a job

```
GET /storage/list/{job_id}
```

```json
{
  "job_id": "a1b2c3d4e5f6",
  "count": 3,
  "files": [
    {"key": "audio/a1b2c3d4e5f6/ch1_001.wav", "url": "https://..."},
    {"key": "audio/a1b2c3d4e5f6/ch1_002.wav", "url": "https://..."}
  ]
}
```

---

### GPU Management

#### GPU status

```
GET /gpu/status
```

**Response:**
```json
{
  "model_loaded": true,
  "loaded_checkpoint": "jobs/a1b2c3d4e5f6/output/checkpoint-epoch-9",
  "speaker_name": "hero_voice",
  "auto_unload_enabled": true,
  "idle_timeout_seconds": 300,
  "idle_seconds": 45.2,
  "total_requests": 150,
  "total_loads": 3,
  "total_unloads": 2,
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "gpu_memory_total_gb": 40.0,
  "gpu_memory_allocated_gb": 4.2,
  "gpu_memory_reserved_gb": 4.8,
  "gpu_memory_free_gb": 35.8
}
```

#### Manually unload model

```
POST /gpu/unload
```

Immediately frees GPU VRAM. Model auto-reloads on next `/infer` request.

```bash
curl -X POST http://localhost:8000/gpu/unload
```

#### Configure idle timeout

```
PUT /gpu/config
Content-Type: application/json
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `idle_timeout_seconds` | int | `300` | Seconds of inactivity before auto-unload. Set to `0` to disable. |

**Example — set to 2 minutes:**
```bash
curl -X PUT http://localhost:8000/gpu/config \
  -H "Content-Type: application/json" \
  -d '{"idle_timeout_seconds": 120}'
```

**Example — disable auto-unload (keep model in VRAM permanently):**
```bash
curl -X PUT http://localhost:8000/gpu/config \
  -H "Content-Type: application/json" \
  -d '{"idle_timeout_seconds": 0}'
```

---

### Operations Logging

#### Average durations

```bash
curl http://localhost:8000/ops/averages
```

```json
{
  "prepare_data": {"avg_seconds": 45.2, "count": 5},
  "training": {"avg_seconds": 1200.0, "count": 3},
  "inference_custom_voice": {"avg_seconds": 2.1, "count": 150},
  "inference_voice_design": {"avg_seconds": 3.0, "count": 12},
  "model_load": {"avg_seconds": 12.5, "count": 6},
  "s3_upload": {"avg_seconds": 0.8, "count": 100}
}
```

#### Operation history

```bash
# All history
curl http://localhost:8000/ops/history

# Filter by job
curl "http://localhost:8000/ops/history?job_id=a1b2c3d4e5f6"

# Filter by operation type
curl "http://localhost:8000/ops/history?op_name=inference_custom_voice&limit=10"
```

#### Currently running operations

```bash
curl http://localhost:8000/ops/running
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda:0` | GPU device |
| `USE_FLASH_ATTN` | `1` | Enable flash attention (`0` to disable) |
| `GPU_IDLE_TIMEOUT` | `300` | Seconds before auto-unloading model from VRAM |
| `GPU_MAX_CONCURRENCY` | `16` | Max concurrent inference tasks on GPU |
| `GPU_MAX_MODELS` | `4` | Max models kept in VRAM cache (increase for A100) |
| `E2E_ACCESS_KEY` | *(unset)* | E2E Object Storage access key |
| `E2E_SECRET_KEY` | *(unset)* | E2E Object Storage secret key |
| `E2E_BUCKET` | `qwen3-tts` | S3 bucket name |
| `E2E_ENDPOINT_URL` | `https://objectstore.e2enetworks.net` | S3 endpoint |

---

### GPU Memory Management (LRU Cache)

The service implements an **LRU (Least Recently Used) VRAM cache**:
- On **12GB GPUs** (like T4), keep `GPU_MAX_MODELS=1` (Default). The system will swap checkpoints in ~10s when switching speakers.
- On **40GB/80GB GPUs** (like A100), set `GPU_MAX_MODELS=4` or higher. The system will keep multiple voices "hot" in VRAM, allowing instant switching between dialogues.
- All models are automatically unloaded if the GPU is idle for `GPU_IDLE_TIMEOUT` seconds.

**Recommendations for A100 40GB:**

| Setting | Value | Description |
|---------|-------|-------------|
| `GPU_MAX_MODELS` | `4` | Keep up to 4 voices hot in VRAM |
| `GPU_IDLE_TIMEOUT` | `600` | 10 min idle before clearing all |
| `GPU_MAX_CONCURRENCY` | `16` | Handle 16 parallel inference requests |

---

## Local Windows Setup (Notes)

If running locally on Windows:
1.  **Flash Attention**: Standard `pip install` may fail. Use a pre-compiled `.whl` (e.g., from `jmica/flash_attention` on HuggingFace).
2.  **Audio Processing**: Ensure `ffmpeg` is in your system `PATH`.
3.  **Speaker Names**: Avoid underscores followed by numbers (e.g., use `Hero1` instead of `Hero_1`) to prevent potential library parsing errors.
4.  **Dataset Maker**: The API is 100% compatible with datasets exported from the [dataset-maker](https://github.com/JarodMica/dataset-maker) tool. Just zip the exported folder and upload.

---

## Microservice Integration Pattern

Your other microservice should follow this pattern:

```python
import requests

GPU_API = "http://<TIR_GPU_IP>:8000"

# 0. (Optional) Generate reference audio via Voice Design
r = requests.post(f"{GPU_API}/voice-design",
    json={
        "text": "Test sentence for voice reference.",
        "instruct": "A warm male voice, middle-aged, calm",
        "s3_filename": "ref_voice.wav",
    }
)
print(f"Voice design: {r.json()['s3_url']}")

# 1. Fine-tune (one-time per voice)
with open("voice_dataset.zip", "rb") as f:
    r = requests.post(f"{GPU_API}/finetune",
        files={"dataset": f},
        data={"speaker_name": "hero_voice", "num_epochs": 10}
    )
job_id = r.json()["job_id"]

# 2. Wait for training
import time
while True:
    r = requests.get(f"{GPU_API}/jobs/{job_id}")
    status = r.json()["status"]
    if status == "ready":
        break
    elif status == "failed":
        raise Exception(r.json()["error"])
    time.sleep(30)

# 3. Generate audio → S3 (your main loop)
texts = [
    {"text": "Chapter one begins.", "filename": "ch1_001.wav"},
    {"text": "The hero stood tall.", "filename": "ch1_002.wav"},
]
r = requests.post(f"{GPU_API}/infer/{job_id}/batch",
    json={"items": texts, "language": "English"}
)
for item in r.json():
    print(f"Audio: {item['s3_url']}")

# 4. Check operation timing
r = requests.get(f"{GPU_API}/ops/averages")
print(f"Avg inference time: {r.json()['inference_custom_voice']['avg_seconds']}s")
```

---

## Errors

| Status | Meaning |
|--------|---------|
| `400` | Invalid dataset zip (missing train.jsonl) |
| `404` | Job not found |
| `409` | Job not ready for inference yet |
| `500` | Inference or training error |
| `503` | Storage not configured (S3 features unavailable) |
