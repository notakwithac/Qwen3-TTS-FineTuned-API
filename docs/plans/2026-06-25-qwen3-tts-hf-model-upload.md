# Qwen3-TTS Hugging Face Model Storage Migration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Move Qwen3-TTS fine-tuned model checkpoint upload and restore from S3 to public Hugging Face Hub repos authenticated by `HF_TOKEN`.

**Architecture:** Keep the current S3 flow for audio outputs and dataset artifacts unless explicitly changed later. Introduce a model-storage layer that uploads the latest checkpoint directory to a public HF repo, stores the repo id in job metadata, and restores missing local checkpoints via `snapshot_download`. Preserve legacy S3 reads only for older jobs so existing checkpoints remain usable during migration.

**Tech Stack:** Python, FastAPI, `huggingface_hub`, job metadata JSON, existing Qwen3-TTS pipeline/session manager.

---

### Task 1: Map the current S3 model durability path

**Files:**
- Reference: `pipeline.py`
- Reference: `session_manager.py`
- Reference: `storage.py`
- Reference: `api_server.py`
- Reference: `.env.example`
- Reference: `tests/test_checkpoint_epoch_resolution.py`
- Reference: `tests/test_session_manager_prepare.py`

**Step 1:** Treat `pipeline.py` as the source of truth for model upload/restore behavior.

Focus on:
- `Job.s3_model_key`
- `Job.checkpoint_s3_keys`
- `_upload_latest_checkpoint_to_s3(...)`
- `_restore_checkpoint_from_s3(...)`
- `resolve_checkpoint_path(...)`
- `_ensure_s3_backup_for_ready_job(...)`

**Step 2:** Confirm where the session manager depends on S3-backed checkpoints.

The main call site is the path that restores missing checkpoints before inference/session preparation.

**Step 3:** Identify the minimal API surface that needs HF-backed equivalents.

The expected boundary is:
- upload after training completes
- restore when a checkpoint is missing locally
- metadata persistence in `job.json`
- readiness checks for old vs new jobs

---

### Task 2: Add Hugging Face-backed checkpoint upload

**Files:**
- Modify: `pipeline.py`
- Modify: `.env.example`
- Reference: `scripts/upload_kept_finetunes_to_hf.py`

**Step 1:** Add HF config sourced from environment.

Use `HF_TOKEN` for auth and add a namespace/prefix scheme for repo naming, such as:
- `HF_MODEL_NAMESPACE`
- `HF_MODEL_REPO_PREFIX`

**Step 2:** Create a model upload helper that targets a public HF repo.

The helper should:
- create the repo if needed
- upload the latest checkpoint directory
- upload a small `README.md` / model card
- upload `job.json` or equivalent metadata needed for restore
- return a stable HF repo id / URL

**Step 3:** Replace the S3 offload path for new jobs.

When training finishes:
- upload the latest checkpoint to HF instead of zipping to S3
- persist HF metadata on the job
- keep the local checkpoint until cleanup logic decides what to prune

**Step 4:** Keep legacy S3 uploads readable only as fallback.

Do not remove the old S3 restore path yet. Existing jobs should still load if they only have S3 metadata.

---

### Task 3: Restore checkpoints from Hugging Face

**Files:**
- Modify: `pipeline.py`
- Modify: `session_manager.py`
- Test: `tests/test_checkpoint_epoch_resolution.py`

**Step 1:** Add an HF restore helper that downloads a repo snapshot.

Use `snapshot_download(...)` to hydrate the repo locally before inference.

**Step 2:** Update checkpoint resolution logic.

Preferred order:
1. existing local checkpoint
2. HF repo snapshot
3. legacy S3 restore for historical jobs

**Step 3:** Keep epoch-specific behavior working.

If multiple checkpoint epochs are still tracked, preserve the ability to resolve the latest available checkpoint and any explicitly requested epoch that exists in metadata.

**Step 4:** Update session preparation to use the new HF restore path transparently.

The session manager should not need to know whether a checkpoint came from local disk, HF, or legacy S3. It should only receive a usable local path.

---

### Task 4: Update metadata and API responses

**Files:**
- Modify: `pipeline.py`
- Modify: `api_server.py`
- Test: `tests/test_job_status_api.py`
- Test: `tests/test_checkpoint_epoch_resolution.py`

**Step 1:** Add HF model fields to `Job`.

Likely fields:
- `hf_model_repo`
- `hf_model_url`
- `checkpoint_hf_repos`

**Step 2:** Persist the new fields in `job.json`.

Keep legacy S3 fields in place for backward compatibility, but make HF the primary durable source for new jobs.

**Step 3:** Update status payloads and readiness checks.

Surface enough metadata for operators and the API to tell:
- where the checkpoint is stored
- whether it is already hydrated locally
- whether it can be re-downloaded from HF

---

### Task 5: Refresh docs and tests

**Files:**
- Modify: `README.md` or the most relevant operator docs
- Modify: `.env.example`
- Modify: `API_DOCS.md` only if the public API shape changes
- Test: `tests/test_checkpoint_epoch_resolution.py`
- Test: `tests/test_session_manager_prepare.py`

**Step 1:** Document the HF upload and restore flow.

Call out:
- `HF_TOKEN`
- public repo naming
- where model checkpoints live now
- the fallback behavior for old S3-backed jobs

**Step 2:** Add or update tests for the new primary path.

Verify:
- upload creates the expected HF repo metadata
- restore downloads from HF when local files are missing
- session prep still resolves a usable checkpoint path

**Step 3:** Keep the scope tight.

Do not change audio upload, dataset packaging, or unrelated S3 storage unless a test proves those paths are coupled to model checkpoint handling.

---

### Assumptions

- “Model upload no longer happens to S3” means fine-tuned model checkpoints only.
- Existing S3-backed checkpoints remain restorable during migration.
- Public HF repos are acceptable for these fine-tuned models.
- The implementation should prefer HF for new jobs and use S3 only as a compatibility fallback.
