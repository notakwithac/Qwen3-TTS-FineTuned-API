# Batch Reference Diarization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run diarization against temporary batches that include reference audio plus clones, emit only segmented reference/clone items, and choose the packaged `ref_audio.wav` from the segmented reference set.

**Architecture:** Build bounded temporary WAV batches that always include the reference audio plus a subset of clone clips. Run WhisperX alignment and diarization once per batch, infer the target speaker from the reference interval, map target-speaker segments back to each clip, then upload only segmented items. Packaging selects the best reference segment and writes it as `data/ref_audio.wav`.

**Tech Stack:** Python, WhisperX, pyannote via `whisperx.diarize`, stdlib `wave`/`audioop`, pytest

---

### Task 1: Add batch-diarization helpers

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\dataset_jobs.py`
- Test: `e:\Projects\Qwen-Finetune\Qwen3-TTS\tests\test_dataset_jobs.py`

**Step 1: Write the failing tests**

Add tests that require:
- batching reference plus clones into one analysis pass
- selecting the target speaker from the reference interval
- mapping diarized segments back to per-clip local timestamps

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: FAIL on missing batch helper behavior

**Step 3: Write minimal implementation**

Add helpers in `dataset_jobs.py` for:
- measuring WAV duration
- preparing WAV bytes for batch analysis
- grouping clone clips into batches with a total duration cap
- concatenating analysis WAVs with separator silence
- selecting the reference speaker from the combined diarized output
- slicing combined aligned segments back to local clip intervals

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: PASS for the new helper coverage

**Step 5: Commit**

```bash
git add dataset_jobs.py tests/test_dataset_jobs.py
git commit -m "feat: add batched reference-aware diarization helpers"
```

### Task 2: Replace per-clip diarization with batch processing

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\dataset_jobs.py`
- Test: `e:\Projects\Qwen-Finetune\Qwen3-TTS\tests\test_dataset_jobs.py`

**Step 1: Write the failing test**

Add coverage that ensures:
- diarization sees reference + clones together
- clone outputs keep only the target speaker inferred from the reference interval
- reference outputs are emitted once, not once per batch

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: FAIL on current per-clip diarization behavior

**Step 3: Write minimal implementation**

Refactor `_validate_and_crop_audio_sync` to:
- build temporary batches capped at roughly 90 seconds including reference
- run WhisperX transcription/alignment/diarization once per batch
- identify the target speaker from the reference interval
- emit split results for reference and clone intervals using only target-speaker segments

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: PASS for batched diarization behavior

**Step 5: Commit**

```bash
git add dataset_jobs.py tests/test_dataset_jobs.py
git commit -m "feat: batch diarization with reference speaker selection"
```

### Task 3: Emit segmented reference items only

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\dataset_jobs.py`
- Test: `e:\Projects\Qwen-Finetune\Qwen3-TTS\tests\test_dataset_jobs.py`

**Step 1: Write the failing test**

Add coverage that ensures `prepare_dataset_items`:
- does not upload the full raw `ref_audio.wav`
- uploads segmented reference items only
- marks those items as reference items

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: FAIL because the current prepare flow still uploads full reference audio

**Step 3: Write minimal implementation**

Update `prepare_dataset_items` to:
- keep validated reference segments
- upload them as dataset items with `is_reference=True`
- exclude the unsplit raw reference upload path

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: PASS for reference item output

**Step 5: Commit**

```bash
git add dataset_jobs.py tests/test_dataset_jobs.py
git commit -m "feat: emit segmented reference items only"
```

### Task 4: Select packaged reference audio from segmented reference items

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\dataset_jobs.py`
- Test: `e:\Projects\Qwen-Finetune\Qwen3-TTS\tests\test_dataset_jobs.py`

**Step 1: Write the failing test**

Add coverage that ensures `package_dataset`:
- chooses a reference segment from `is_reference=True` items
- writes that selected segment to `data/ref_audio.wav`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: FAIL because packaging currently expects one uploaded full reference clip

**Step 3: Write minimal implementation**

Update `package_dataset` to:
- collect all included reference items
- pick the best reference segment, preferably the longest valid one
- write that chosen segment as `data/ref_audio.wav`

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dataset_jobs.py -q`
Expected: PASS for package-time reference selection

**Step 5: Commit**

```bash
git add dataset_jobs.py tests/test_dataset_jobs.py
git commit -m "feat: choose packaged reference audio from reference segments"
```
