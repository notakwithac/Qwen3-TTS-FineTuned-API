# Event-Driven TTS Inference — Developer Guide

This document describes the high-performance, session-based inference architecture in Qwen3-TTS. This system is designed for high-traffic scenarios (e.g., generating audiobooks with multiple characters) where latency and GPU utilization are critical.

---

## 🚀 Key Advantages

| Feature | Event-Driven (Session-Based) | Classic Synchronous API |
|---------|----------------------------|------------------------|
| **Model Pre-loading** | Models stay hot in VRAM for the entire session. | Models may swap/reload frequently (LRU). |
| **Parallelism** | Multiple characters processed in parallel by dedicated workers. | Sequential or limited by global semaphore. |
| **Model Replication** | Automatically duplicates high-traffic models across VRAM. | Single instance per model. |
| **Latency** | ~70% lower for multi-character batches. | Variable based on cache hits/swaps. |
| **Pipelining** | Inference and S3 uploads happen concurrently. | Uploads usually happen after inference finishes. |

---

## ⚙️ Configuration

Set these in your `.env` file to tune performance:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPLICA_THRESHOLD` | `500` | Lines threshold to trigger model replication for a character. |
| `MAX_REPLICAS_PER_MODEL` | `4` | Max number of GPU clones of the same model weights. |
| `SESSION_TIMEOUT` | `3600` | Seconds of inactivity before a session is auto-torn down. |
| `GPU_BATCH_SIZE` | `32` | Max items a worker pulls from its queue for a single GPU step. |

---

## 📡 API Reference

### 1. Prepare Session
Initiate a production workflow by pre-loading all required character models.

```bash
POST /session/prepare
```

**Request Body:**
```json
{
  "session_id": "my-chapter-1",
  "characters": [
    { "job_id": "job_narrator", "character_name": "Narrator", "line_count": 1200 },
    { "job_id": "job_hero", "character_name": "Hero", "line_count": 150 }
  ],
  "book_id": "book_42",
  "chapter_id": "chapter_1"
}
```

**What happens:**
- The system checks available GPU VRAM.
- It determines the "Narrator" needs **2 replicas** because `line_count > REPLICA_THRESHOLD`.
- Models are loaded into VRAM and **pinned** (protected from LRU eviction).
- Dedicated worker loops are started for each model instance.

---

### 2. Submit Inference Messages
Push content into the character queues for processing.

```bash
POST /session/{session_id}/submit/batch
```

**Request Body:**
```json
{
  "items": [
    {
      "job_id": "job_narrator",
      "character_name": "Narrator",
      "text": "Once upon a time in a land far away...",
      "s3_filename": "segment_001.wav"
    },
    {
      "job_id": "job_hero",
      "character_name": "Hero",
      "text": "Where am I?",
      "s3_filename": "segment_002.wav"
    }
  ]
}
```

---

### 3. Check Progress
Poll the status of your session.

```bash
GET /session/{session_id}/status?include_results=true
```

**Response:**
```json
{
  "session_id": "my-chapter-1",
  "status": "processing",
  "total_lines": 1350,
  "completed_lines": 420,
  "progress_pct": 31.1,
  "results": {
    "job_narrator": ["https://.../segment_001.wav"],
    "job_hero": ["https://.../segment_002.wav"]
  }
}
```

---

### 4. Teardown
Release replicas and unpin models.

```bash
DELETE /session/{session_id}
```
*Note: Primary models remain in the LRU cache for potential reuse, but unique replicas are immediately purged to free VRAM.*

---

## 🐍 Integration Example (Python)

This example shows how an external service (e.g., a Book Orchestrator) should implement this flow.

```python
import requests
import time

API_URL = "http://qwen-gpu-server:8000"
SESSION_ID = "production_chapter_v1"

# 1. Prepare: Load characters Narrator and Hero
# We tell the server Narrator has many lines so it can replicate the model
manifest = {
    "session_id": SESSION_ID,
    "characters": [
        {"job_id": "65d83a...", "character_name": "Narrator", "line_count": 800},
        {"job_id": "65d91b...", "character_name": "Hero", "line_count": 50}
    ],
    "book_id": "fantasy_book_01",
    "chapter_id": "1"
}

print("Preparing GPU resources...")
prepare_resp = requests.post(f"{API_URL}/session/prepare", json=manifest)
if prepare_resp.status_code != 200:
    print(f"Error: {prepare_resp.text}")
    exit(1)

# 2. Submit: Send actual text lines
# Messages are processed in the background as soon as they are submitted
lines = [
    {"job_id": "65d83a...", "character_name": "Narrator", "text": "The wind howled.", "s3_filename": "001.wav"},
    {"job_id": "65d91b...", "character_name": "Hero", "text": "I must find shelter.", "s3_filename": "002.wav"},
    # ... submit hundreds more lines here ...
]

print("Submitting lines to per-character queues...")
requests.post(f"{API_URL}/session/{SESSION_ID}/submit/batch", json={"items": lines})

# 3. Monitor: Poll for completion
while True:
    status = requests.get(f"{API_URL}/session/{SESSION_ID}/status").json()
    done = status["completed_lines"]
    total = status["total_lines"]
    print(f"Progress: {done}/{total} ({status['progress_pct']}%)")
    
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# 4. Cleanup: Free GPU memory
requests.delete(f"{API_URL}/session/{SESSION_ID}")
print("Session complete and resources released.")
```
