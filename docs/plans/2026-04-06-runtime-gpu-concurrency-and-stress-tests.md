# Runtime GPU Concurrency And Stress Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add runtime API controls for GPU concurrency/shared replica counts, then add real S3-backed GPU stress tests plus `/logs/stream` verification so the server can be tuned and validated without redeploying.

**Architecture:** Keep `.env` as startup defaults, but add a live configuration layer on top of `InferenceManager` that can update `GPU_MAX_MODELS`, shared design/clone replica targets, and shared headroom at runtime. Add manual stress scripts that hit `http://<ip>:8000`, submit real `voice-design` and `voice-clone` work, poll for completion, and verify mixed-load behavior and log streaming. Preserve the existing custom-model/session concurrency path and do not revert any current uncommitted edits in the repo.

**Tech Stack:** FastAPI, Pydantic, Python threading/asyncio, existing `InferenceManager`/`Pipeline`/`SessionManager`, `requests`, SSE (`sse-starlette` + `Broadcast`), S3-backed API flow.

---

### Task 1: Snapshot current in-progress state and protect existing edits

**Files:**
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/docs/plans/2026-04-06-runtime-gpu-concurrency-and-stress-tests.md`
- Reference only: `e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`
- Reference only: `e:/Projects/Qwen-Finetune/Qwen3-TTS/inference_manager.py`
- Reference only: `e:/Projects/Qwen-Finetune/Qwen3-TTS/pipeline.py`
- Reference only: `e:/Projects/Qwen-Finetune/Qwen3-TTS/.env.example`
- Reference only: `e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_shared_model_replicas.py`

**Step 1: Record current worktree state**

Run:

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS status --short
```

Expected: existing modified files include `.env.example`, `api_server.py`, `inference_manager.py`, `pipeline.py`, and `tests/test_shared_model_replicas.py`.

**Step 2: Confirm no unrelated edits are reverted**

Read the current diffs before implementing anything:

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS diff -- .env.example api_server.py inference_manager.py pipeline.py tests/test_shared_model_replicas.py
```

Expected: understand current shared-replica work and preserve it.

**Step 3: Commit**

Do not commit yet. This task is a guardrail task only.

### Task 2: Add an adjustable runtime inference limiter

**Files:**
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/inference_manager.py`
- Test: `e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_shared_model_replicas.py`

**Step 1: Write the failing test**

Add tests for a runtime-adjustable limiter:

```python
def test_runtime_max_models_update_changes_effective_limit():
    manager = InferenceManager(device="cpu", max_models=2)
    manager.update_runtime_config(max_models=4)
    assert manager.max_models == 4
```

```python
def test_runtime_shared_replica_update_changes_targets():
    manager = InferenceManager(device="cpu", shared_model_replicas={"voice_design": 1})
    manager.update_runtime_config(shared_model_replicas={"voice_design": 3})
    assert manager.stats["shared_model_replicas"]["voice_design"] == 3
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_shared_model_replicas.py -v
```

Expected: FAIL because `update_runtime_config` does not exist yet.

**Step 3: Write minimal implementation**

Add to `InferenceManager`:
- an adjustable semaphore/limiter abstraction for `_inference_semaphore`
- `update_runtime_config(...)`
- validation for:
  - `max_models >= 1`
  - `voice_design` replicas >= 1
  - `voice_clone` replicas >= 1
  - `shared_model_min_headroom_gb >= 0`

The update rules should be:
- increases apply immediately
- decreases apply to new work without killing in-flight requests
- shared replica target updates affect new scheduling immediately

**Step 4: Expose updated config in stats**

Ensure `InferenceManager.stats` includes:
- `max_models`
- `shared_model_replicas`
- `shared_model_min_headroom_gb`
- limiter state if helpful (`active`, `available`, or equivalent)

**Step 5: Run tests**

Run:

```bash
python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_shared_model_replicas.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS add inference_manager.py tests/test_shared_model_replicas.py
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS commit -m "feat: add runtime adjustable inference limits"
```

### Task 3: Add runtime GPU concurrency API endpoints
c
**Files:**
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/pipeline.py`
- Test: `e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_gpu_concurrency_api.py`

**Step 1: Write the failing test**

Add API-level tests for:

```python
def test_get_gpu_concurrency_returns_effective_runtime_config():
    ...
```

```python
def test_post_gpu_concurrency_updates_runtime_config():
    ...
```

Expected payload:

```json
{
  "gpu_max_models": 7,
  "voice_design_replicas": 2,
  "voice_clone_replicas": 2,
  "shared_model_min_headroom_gb": 4
}
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_gpu_concurrency_api.py -v
```

Expected: FAIL because endpoints do not exist yet.

**Step 3: Write minimal implementation**

Add to `api_server.py`:
- `GPUConcurrencyConfigRequest`
- `GPUConcurrencyConfigResponse`
- `GET /gpu/concurrency`
- `POST /gpu/concurrency`

Behavior:
- `GET` returns current live values from `pipeline.inference`
- `POST` validates payload and applies it in memory
- response returns effective values after update

Add to `Pipeline` a thin helper if needed, but keep updates centered in `InferenceManager`.

**Step 4: Validation behavior**

Reject invalid payloads with 422/400 for:
- zero or negative model counts
- absurdly low/negative headroom
- malformed JSON

Do not reject just because requested settings are aggressive; let runtime admission rules gate actual replica creation.

**Step 5: Run tests**

Run:

```bash
python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_gpu_concurrency_api.py -v
```

Expected: PASS

**Step 6: Commit**

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS add api_server.py pipeline.py tests/test_gpu_concurrency_api.py
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS commit -m "feat: add runtime gpu concurrency api"
```

### Task 4: Add a real mixed-load GPU stress script

**Files:**
- Create: `e:/Projects/Qwen-Finetune/Qwen3-TTS/stress_shared_model_replicas.py`
- Reference: `e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`

**Step 1: Write the script skeleton**

The script must accept:

```bash
python stress_shared_model_replicas.py --base-url http://<ip>:8000
```

Optional flags:
- `--design-count`
- `--mixed-clone-count`
- `--clone-only-count`
- `--poll-interval`
- `--overwrite`

**Step 2: Implement phase 1 (voice design)**

Submit a few `POST /voice-design` requests concurrently with `upload_to_s3=true`.

Use distinct filenames so S3 results are easy to inspect:

```json
{
  "text": "...",
  "instruct": "...",
  "character_name": "stress_designer_01",
  "s3_filename": "stress_design_01.wav",
  "upload_to_s3": true,
  "overwrite": true
}
```

Collect returned `presigned_url` and/or `s3_url`.

**Step 3: Implement phase 2 (clone from designed audio)**

Use the returned audio URLs as `ref_audio_url` in clone requests.

Submit:
- one wave of clone requests
- one wave of mixed design + clone requests
- one wave of clone-only requests

For clone batch flow, use:
- `POST /voice-clone/batch`
- then poll `GET /voice-clone/batch/{session_id}` until:
  - `completed`
  - or timeout/failure

**Step 4: Add assertions / expected behavior**

The script should fail fast if:
- any design request returns non-200
- any clone batch never leaves `processing`
- any final result has `status=failed`
- returned clone result count does not match submitted item count

**Step 5: Print a usable summary**

At the end print:
- total requests
- phase durations
- success/failure counts
- any session ids used

**Step 6: Manual verification**

Run:

```bash
python e:/Projects/Qwen-Finetune/Qwen3-TTS/stress_shared_model_replicas.py --base-url http://<ip>:8000
```

Expected: all phases complete and output includes timings and zero failures.

**Step 7: Commit**

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS add stress_shared_model_replicas.py
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS commit -m "test: add shared model gpu stress script"
```

### Task 5: Add a dedicated log-stream stress verifier

**Files:**
- Create: `e:/Projects/Qwen-Finetune/Qwen3-TTS/stress_logs_stream.py`
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`
- Test: `e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_logs_stream.py`

**Step 1: Write the failing test**

Add at least one focused test for `LogStreamHandler` behavior:

```python
def test_log_stream_handler_emits_background_thread_logs():
    ...
```

If direct integration testing is hard, test the handler/formatter path in isolation.

**Step 2: Investigate the current gap before fixing**

Read carefully:
- `LogStreamHandler.emit(...)`
- stdout redirection setup
- any background thread logging sites in training/inference/session code

Capture one concrete missing-log example before changing code.

**Step 3: Implement `stress_logs_stream.py`**

The script should:
- open `GET /logs/stream`
- keep the SSE connection alive
- trigger background work on the GPU API
- record streamed messages to memory/file
- assert that expected events appear during background processing

Expected events to look for:
- design request started/finished
- clone batch started/finished
- session worker batch logs if relevant

**Step 4: Fix the root cause if logs are missing**

Likely candidates:
- only stdout is redirected, not stderr
- logs from background threads are filtered out
- `loop.call_soon_threadsafe(...)` is skipped when no main loop reference exists
- noisy log filters accidentally exclude useful ops

Fix only after confirming the actual cause.

**Step 5: Manual verification**

Run:

```bash
python e:/Projects/Qwen-Finetune/Qwen3-TTS/stress_logs_stream.py --base-url http://<ip>:8000
```

Expected: stream remains connected and captures logs from background or threadpool-driven work.

**Step 6: Commit**

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS add stress_logs_stream.py api_server.py tests/test_logs_stream.py
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS commit -m "test: verify gpu log streaming under stress"
```

### Task 6: Add a convenience runtime tuning + stress workflow

**Files:**
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/.env.example`
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/API_DOCS.md`

**Step 1: Document runtime tuning**

Add docs for:
- `GET /gpu/concurrency`
- `POST /gpu/concurrency`
- example payloads
- note that `.env` values are startup defaults only

**Step 2: Document the manual stress workflow**

Recommended sequence:

1. Boot server with safe defaults
2. `GET /gpu/concurrency`
3. `POST /gpu/concurrency` with test values
4. run `stress_shared_model_replicas.py`
5. run `stress_logs_stream.py`
6. inspect `/gpu/vram`, `/gpu/metrics`, `/ops/history`

**Step 3: Keep `.env.example` honest**

Clarify:
- `GPU_MAX_MODELS` is a capacity budget
- `VOICE_DESIGN_REPLICAS` and `VOICE_CLONE_REPLICAS` are startup defaults
- runtime API can override them without redeploy

**Step 4: Commit**

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS add .env.example API_DOCS.md
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS commit -m "docs: add runtime gpu concurrency workflow"
```

### Task 7: Final validation under real server conditions

**Files:**
- No code changes required unless failures are found

**Step 1: Start with conservative live settings**

Example:

```json
{
  "gpu_max_models": 7,
  "voice_design_replicas": 2,
  "voice_clone_replicas": 2,
  "shared_model_min_headroom_gb": 4
}
```

**Step 2: Exercise the runtime API**

Run:

```bash
curl http://<ip>:8000/gpu/concurrency
curl -X POST http://<ip>:8000/gpu/concurrency -H "Content-Type: application/json" -d "{\"gpu_max_models\":7,\"voice_design_replicas\":2,\"voice_clone_replicas\":2,\"shared_model_min_headroom_gb\":4}"
```

Expected: updated settings are reflected immediately.

**Step 3: Run both stress scripts**

Run:

```bash
python e:/Projects/Qwen-Finetune/Qwen3-TTS/stress_shared_model_replicas.py --base-url http://<ip>:8000
python e:/Projects/Qwen-Finetune/Qwen3-TTS/stress_logs_stream.py --base-url http://<ip>:8000
```

Expected:
- no stuck clone sessions
- no unexpected 5xx
- logs stream captures background activity
- mixed design + clone load completes successfully

**Step 4: Inspect runtime telemetry**

Check:
- `GET /gpu/vram`
- `GET /gpu/metrics`
- `GET /ops/history?limit=100`

**Step 5: Commit only after clean run**

```bash
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS add api_server.py inference_manager.py pipeline.py .env.example API_DOCS.md tests/test_shared_model_replicas.py tests/test_gpu_concurrency_api.py tests/test_logs_stream.py stress_shared_model_replicas.py stress_logs_stream.py
git -C e:/Projects/Qwen-Finetune/Qwen3-TTS commit -m "feat: add runtime gpu concurrency controls and stress validation"
```
