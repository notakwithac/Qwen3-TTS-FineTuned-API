# TIR Image Slimming Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce GPU setup time by shrinking the TIR Docker image while keeping finetuning and API serving functional.

**Architecture:** Keep the existing single-image flow, but remove optional and Docker-only dependencies from the image. Make TensorBoard optional in package metadata, stop creating a container-only virtualenv, and run the startup script against the base Python environment.

**Tech Stack:** Docker, PyTorch base image, pip/setuptools packaging, Bash startup scripts

---

### Task 1: Make TensorBoard optional in packaging

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\pyproject.toml`

**Step 1: Update core dependencies**

Remove `tensorboard>=2.20.0` from the main `dependencies` list so it is not always installed.

**Step 2: Add optional dependency group**

Add a new optional dependency group for training logs so TensorBoard can be added back later if needed.

**Step 3: Verify training code behavior**

Confirm `finetuning/sft_12hz.py` already treats TensorBoard as optional and therefore does not need code changes.

### Task 2: Slim the Docker image while preserving finetuning

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\Dockerfile.tir`

**Step 1: Remove Docker-only tooling**

Delete `curl`, `git`, `build-essential`, `dos2unix`, and the `uv` install step because they are not required at runtime.

**Step 2: Remove always-on optional installs**

Update the pip install layers so TensorBoard and Gradio are not installed in the image by default.

**Step 3: Remove container virtualenv creation**

Delete the editable install plus `.venv` creation and rely on the source checkout plus the container's default Python environment.

**Step 4: Preserve required runtime deps**

Keep packages needed by both serving and finetuning, including audio, FastAPI, boto3, psutil, nvidia-ml-py, and flash-attn.

### Task 3: Update startup flow to work without `.venv`

**Files:**
- Modify: `e:\Projects\Qwen-Finetune\Qwen3-TTS\start_tir.sh`

**Step 1: Make virtualenv activation optional**

If `.venv/bin/activate` exists, activate it. Otherwise continue with the container's default Python environment instead of exiting.

**Step 2: Keep runtime behavior unchanged**

Preserve watchdog startup, env loading, and uvicorn launch behavior.

### Task 4: Review the diff

**Files:**
- Review: `e:\Projects\Qwen-Finetune\Qwen3-TTS\pyproject.toml`
- Review: `e:\Projects\Qwen-Finetune\Qwen3-TTS\Dockerfile.tir`
- Review: `e:\Projects\Qwen-Finetune\Qwen3-TTS\start_tir.sh`

**Step 1: Inspect diff**

Run `git -C e:\Projects\Qwen-Finetune\Qwen3-TTS diff -- pyproject.toml Dockerfile.tir start_tir.sh`.

**Step 2: Sanity-check behavior**

Confirm the image still contains all dependencies needed for finetuning and API startup, except optional TensorBoard logging and the Gradio demo UI.
