# Custom Voice Batch Sizing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce custom-voice session batch pressure and make batch admission reflect padded batch cost instead of raw item count alone.

**Architecture:** Keep the existing session worker model, but give custom-voice workers a lower default max item count and a second size gate that approximates VRAM pressure from left-padded batches. Preserve oversized single-dialogue handling while improving ops visibility with per-batch size diagnostics.

**Tech Stack:** Python, asyncio, FastAPI config plumbing, pytest

---

### Task 1: Add custom-voice-specific session batch limits

**Files:**
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/session_manager.py`

**Step 1: Write the config defaults**

Add env-backed custom-voice session defaults that derive from existing global batch settings:
- halve the current `GPU_BATCH_SIZE` for custom-voice session workers
- derive a padded-size budget from `SESSION_BATCH_MAX_CHARS`

**Step 2: Wire the values into `SessionManager`**

Pass the custom-voice item cap and padded-size budget into the worker constructor without changing unrelated inference paths.

**Step 3: Commit**

```bash
git add e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py e:/Projects/Qwen-Finetune/Qwen3-TTS/session_manager.py
git commit -m "feat: tighten custom voice session batch limits"
```

### Task 2: Make session batching padding-aware

**Files:**
- Modify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/session_manager.py`
- Test: `e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_session_manager_batching.py`

**Step 1: Write the failing tests**

Add a test showing:
- the worker splits when padded batch cost would exceed budget even if raw char sum still fits
- oversized single messages still run as one-item batches

**Step 2: Run the targeted test file**

Run: `python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_session_manager_batching.py -q`

Expected: FAIL before the worker becomes padding-aware.

**Step 3: Write the minimal implementation**

In `session_manager.py`:
- add helpers for per-message cost, padded batch cost, and next-batch estimation
- gate batching on item count, raw char budget, and padded batch budget
- include the new size diagnostics in `session_worker_batch` ops logging

**Step 4: Run the targeted test file again**

Run: `python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_session_manager_batching.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add e:/Projects/Qwen-Finetune/Qwen3-TTS/session_manager.py e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_session_manager_batching.py
git commit -m "feat: make custom voice session batching size aware"
```

### Task 3: Verify syntax and behavior

**Files:**
- Verify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`
- Verify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/session_manager.py`
- Verify: `e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_session_manager_batching.py`

**Step 1: Run compile checks**

Run:
- `python -m py_compile e:/Projects/Qwen-Finetune/Qwen3-TTS/api_server.py`
- `python -m py_compile e:/Projects/Qwen-Finetune/Qwen3-TTS/session_manager.py`

Expected: no output

**Step 2: Run focused tests**

Run: `python -m pytest e:/Projects/Qwen-Finetune/Qwen3-TTS/tests/test_session_manager_batching.py -q`

Expected: PASS
