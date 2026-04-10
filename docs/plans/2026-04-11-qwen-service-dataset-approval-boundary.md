# Qwen Service Dataset Approval Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Qwen service produce raw dataset items during `/dataset/prepare` and trust orchestrator-approved `dataset_items` during `/dataset/package`, while preserving the existing package-time fallback if no reference item is marked.

**Architecture:** Keep the current two-step dataset API surface, but simplify responsibilities. The prepare step will only build the unified raw candidate items and persist them for status polling. The package step will remain responsible for converting approved items into the final Qwen dataset zip with `train.jsonl` and `ref_audio.wav`, using the marked `is_reference` item when present and the existing fallback otherwise.

**Tech Stack:** FastAPI, Pydantic v2, asyncio background tasks, zipfile, pytest

---

### Task 1: Remove approval behavior from prepare-time service logic

**Files:**
- Modify: `api_server.py`
- Modify: `dataset_jobs.py`

**Step 1: Keep `/dataset/prepare` compatibility but stop auto-packaging**

Update the endpoint flow so `approval_mode` is accepted but not used to package automatically.

**Step 2: Stop prepare-time reference selection**

Update `prepare_dataset_items()` to produce raw dataset candidates without injecting a chosen reference sample or pretending approval has happened.

### Task 2: Preserve package-time trust boundary

**Files:**
- Modify: `dataset_jobs.py`

**Step 1: Keep current package fallback**

Ensure `package_dataset()` still:
- trusts incoming `dataset_items`
- prefers the item flagged `is_reference`
- falls back to the last included clip if none is marked

### Task 3: Update API docs/tests

**Files:**
- Modify: `API_DOCS.md`
- Modify: `tests/test_dataset_status_api.py`

**Step 1: Adjust docs**

Document that `/dataset/prepare` returns raw `dataset_items` for approval, and `/dataset/package` expects already-approved items.

**Step 2: Update/add tests**

Cover:
- prepare no longer auto-packages on `approval_mode="auto"`
- prepare returns raw items without forcing a reference selection
- package still succeeds with explicit `is_reference`
- package still falls back when no reference is marked

### Task 4: Verify

**Files:**
- Test: `tests/test_dataset_status_api.py`

**Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_dataset_status_api.py -q
```

**Step 2: Fix and rerun**

Patch any regressions and rerun until green.
