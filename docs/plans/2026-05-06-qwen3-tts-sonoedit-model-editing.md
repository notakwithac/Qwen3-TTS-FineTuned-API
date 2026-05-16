# Qwen3-TTS SonoEdit Model Editing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a research-grade SonoEdit-style pipeline that applies targeted one-shot pronunciation edits to copied Qwen3-TTS checkpoints while preserving unrelated speech behavior.

**Architecture:** Adapt SonoEdit to Qwen3-TTS by editing the Transformer speech-token planner, not the tokenizer or vocoder. Use Qwen's codec-0 prediction path as the coarse pronunciation target, localize sensitive `talker.model.layers` with causal tracing, compute a null-space constrained update on selected projection weights, and save an edited checkpoint with metadata.

**Tech Stack:** PyTorch, Qwen3-TTS model internals, safetensors, NumPy/SciPy linear algebra, pytest

---

### Task 1: Add edit request and artifact schemas

**Files:**
- Create: `research/sonoedit_qwen3_tts/schema.py`
- Create: `tests/test_sonoedit_schema.py`

**Step 1: Write schema tests**

Test parsing for:
- target term
- source sentence
- desired pronunciation exemplar audio
- preservation manifest
- model checkpoint path
- output checkpoint path
- selected edit layers

**Step 2: Implement schema dataclasses**

Define:
- `SonoEditRequest`
- `TargetPronunciationExample`
- `PreservationExample`
- `SonoEditResult`

Require all model-editing runs to write `sonoedit_metadata.json` beside the edited checkpoint.

### Task 2: Build Qwen3-TTS activation capture utilities

**Files:**
- Create: `research/sonoedit_qwen3_tts/activations.py`
- Test: `tests/test_sonoedit_activation_hooks.py`

**Step 1: Write hook tests with small fake modules**

Test that hooks can:
- capture inputs and outputs for named layers
- remove themselves cleanly
- avoid retaining tensors after context exit

**Step 2: Implement hooks for Qwen talker layers**

Support layer names like:
- `talker.model.layers.0`
- `talker.model.layers.10`
- `talker.model.layers.19`

Do not hook `speaker_encoder`, `speech_tokenizer`, or `talker.code_predictor` for v1.

### Task 3: Implement codec-0 target extraction

**Files:**
- Create: `research/sonoedit_qwen3_tts/targets.py`
- Test: `tests/test_sonoedit_targets.py`

**Step 1: Write target extraction tests**

Use synthetic codec arrays to verify:
- codec-0 is selected from Qwen's 16-codebook audio code sequence
- target frame spans can be manually specified
- invalid spans fail clearly

**Step 2: Implement v1 target selection**

V1 accepts explicit target frame spans in the edit request. Do not attempt automatic phoneme alignment in v1.

Map desired pronunciation audio through the Qwen3-TTS tokenizer and extract codec group 0 as the target acoustic sequence.

### Task 4: Add acoustic causal tracing

**Files:**
- Create: `research/sonoedit_qwen3_tts/causal_trace.py`
- Test: `tests/test_sonoedit_causal_trace.py`

**Step 1: Write tests with fake logits**

Test that impact scores rank layers by recovery of target codec-0 probability after restoring clean activations.

**Step 2: Implement tracing runner**

For each candidate layer:
- run clean forward pass
- run corrupted text/input pass
- restore that layer's activation from clean pass
- compute target codec-0 probability recovery

Default candidate layers for Qwen3-TTS 1.7B:
- `talker.model.layers.6` through `talker.model.layers.17`

The default is based on Qwen's 20 talker layers and SonoEdit's mid-to-late layer finding. It must remain overrideable.

### Task 5: Implement null-space constrained edit computation

**Files:**
- Create: `research/sonoedit_qwen3_tts/nullspace_edit.py`
- Test: `tests/test_sonoedit_nullspace_edit.py`

**Step 1: Write linear algebra tests**

Use small matrices to verify:
- preservation keys define a null-space projection
- the computed update changes the target association
- the first-order change on preservation keys is near zero

**Step 2: Implement constrained update**

Compute updates for selected projection weights only:
- preferred v1 target: `talker.model.layers.{layer}.mlp.down_proj.weight`
- fallback target: `talker.model.layers.{layer}.self_attn.o_proj.weight`

Do not edit embeddings, codec heads, speaker embeddings, or tokenizer weights in v1.

### Task 6: Save edited checkpoint safely

**Files:**
- Create: `research/sonoedit_qwen3_tts/apply_edit.py`
- Test: `tests/test_sonoedit_apply_edit.py`

**Step 1: Implement dry-run mode**

Dry-run must:
- load and validate request JSON
- resolve layer names
- report planned target weights
- write no checkpoint

**Step 2: Implement checkpoint copy and patch**

The edit command must:
- require `--output-model-path` different from `--input-model-path`
- copy non-weight model assets
- write edited `model.safetensors`
- write `sonoedit_metadata.json`

Never mutate the source checkpoint in place.

### Task 7: Add regression eval

**Files:**
- Create: `research/sonoedit_qwen3_tts/evaluate_edit.py`
- Test: `tests/test_sonoedit_evaluate_edit.py`
- Modify: `research/qwen_finetuning_guide.md`

**Step 1: Implement eval matrix**

Evaluate:
- source model target pronunciation
- edited model target pronunciation
- source model preservation set
- edited model preservation set

V1 outputs `results.jsonl` with fields for target correctness, ASR text, PER, WER, speaker similarity, and reviewer notes. Automated scoring may be optional, but the schema must support it.

**Step 2: Document promotion rule**

An edited model can be used only if:
- target examples improve in manual or automated review
- preservation samples do not regress materially
- speaker identity is unchanged for the intended voice mode

### Task 8: Verify

**Files:**
- Test: `tests/test_sonoedit_schema.py`
- Test: `tests/test_sonoedit_activation_hooks.py`
- Test: `tests/test_sonoedit_targets.py`
- Test: `tests/test_sonoedit_causal_trace.py`
- Test: `tests/test_sonoedit_nullspace_edit.py`
- Test: `tests/test_sonoedit_apply_edit.py`
- Test: `tests/test_sonoedit_evaluate_edit.py`

**Step 1: Run unit tests**

```bash
python -m pytest tests/test_sonoedit_schema.py tests/test_sonoedit_activation_hooks.py tests/test_sonoedit_targets.py tests/test_sonoedit_causal_trace.py tests/test_sonoedit_nullspace_edit.py tests/test_sonoedit_apply_edit.py tests/test_sonoedit_evaluate_edit.py -q
```

Expected: PASS.

**Step 2: Compile research modules**

```bash
python -m py_compile research/sonoedit_qwen3_tts/schema.py research/sonoedit_qwen3_tts/activations.py research/sonoedit_qwen3_tts/targets.py research/sonoedit_qwen3_tts/causal_trace.py research/sonoedit_qwen3_tts/nullspace_edit.py research/sonoedit_qwen3_tts/apply_edit.py research/sonoedit_qwen3_tts/evaluate_edit.py
```

Expected: no output.

### How to use this plan

**Base model**

Use the base model first for SonoEdit experiments. Copy `Qwen/Qwen3-TTS-12Hz-1.7B-Base` to a local checkpoint directory, apply the edit to a separate output directory, and compare source vs edited results on both target and preservation sets.

**Existing finetuned models**

Only edit a copied checkpoint. Existing finetunes may already encode speaker/domain behavior in the same layers that SonoEdit touches, so every edited checkpoint must pass the preservation eval before it is used by the API.

**Upcoming finetune models**

Apply SonoEdit after the finetune is complete, not before. Finetuning after a model edit can overwrite or distort the targeted update. Store `sonoedit_metadata.json` with the final checkpoint so the exact edited terms, layers, and preservation set are auditable.

**Acceptance criteria**

- Source checkpoints are never edited in place.
- V1 edits only `talker.model.layers` projection weights.
- Causal tracing layer choices are saved with the edit metadata.
- Every edited checkpoint has a source-vs-edited eval artifact before promotion.
