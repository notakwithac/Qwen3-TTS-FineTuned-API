# Product-First README Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the inherited upstream README with a concise, repository-specific guide to the production fine-tuning and inference service.

**Architecture:** Keep `README.md` as the product landing page and route detailed contracts to the existing specialist documentation. Derive all operational claims and examples from the checked-in API, environment, container, and fine-tuning docs.

**Tech Stack:** Markdown, FastAPI, Docker Compose, Qwen3-TTS, PyTorch/CUDA, S3-compatible object storage

---

### Task 1: Rewrite the product landing page

**Files:**
- Modify: `README.md`

**Step 1:** Replace the upstream-heavy introduction and model reference material with the approved product-first structure.

**Step 2:** Add Docker and script-based quick starts using commands already supported by the repository.

**Step 3:** Add representative fine-tuning, inference, voice design, voice cloning, and translation examples that defer full schemas to `API_DOCS.md`.

**Step 4:** Add architecture, operations, documentation, attribution, citation, and Apache 2.0 license sections.

### Task 2: Validate the documentation

**Files:**
- Verify: `README.md`
- Reference: `API_DOCS.md`
- Reference: `EVENT_DRIVEN_INFERENCE_DOCS.md`
- Reference: `finetuning/README.md`
- Reference: `.env.example`
- Reference: `docker-compose.yml`
- Reference: `LICENSE`

**Step 1:** Check all relative Markdown links resolve to tracked files.

**Step 2:** Compare documented endpoints, ports, commands, and configuration names against their source files.

**Step 3:** Inspect the Markdown heading outline and final Git diff for clarity and accidental unrelated edits.
