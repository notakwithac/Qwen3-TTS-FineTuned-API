# Qwen3-TTS Setup and Fine-Tuning Guide (Windows)

This document provides a comprehensive guide to installing, preparing data, fine-tuning, and serving Qwen3-TTS (1.7B) in a Windows environment, with advanced sections for production scaling.

---

## 🛠 Prerequisites

Before starting, ensure your host machine meets the following requirements:

| Component | Requirement |
| :--- | :--- |
| **OS** | Windows 10/11 |
| **GPU** | NVIDIA GPU (Min 12GB VRAM; 30GB+ recommended for speed) |
| **Storage** | 50GB+ SSD (Optimized via LRU) |

### System Tools
1.  **Git**: For repository cloning.
2.  **uv**: extremely fast Python package manager.
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
3.  **FFmpeg**: Required for audio processing.
    *   Download `ffmpeg-git-full.7z` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
    *   Add the `bin/` folder to your System **PATH**.

---

## 🚀 Phase 1: Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/notakwithac/Qwen3-TTS-FineTuned-API.git
cd Qwen3-TTS-FineTuned-API
```

### 2. Install Flash Attention 2 (Windows Fix)
> [!IMPORTANT]
> Compiling Flash Attention on Windows manually usually fails. You **must** use a pre-compiled wheel.

1.  Download the `.whl` file for Python 3.11 from the [jmica/flash_attention](https://huggingface.co/jmica/flash_attention) repo.
2.  **Rename the file**: Browsers often escape the `+` sign as `%2B`. Rename it back to `+` (e.g., `flash_attn-2.8.3+cu128...`).
3.  Place the `.whl` in the root `Qwen3-TTS` folder.

### 3. Initialize Python Environment
```bash
uv sync --python 3.11
```

### 4. Download Base Models
```bash
uv run huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
uv run huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
```

---

## 📊 Phase 2: Dataset Preparation

We use a separate "Dataset Maker" tool to slice and transcribe raw audio into the Qwen3 format.

### 1. Setup Dataset Maker
```bash
git clone https://github.com/JarodMica/dataset-maker.git
cd dataset-maker
uv sync --python 3.11
```

### 2. WhisperX DLL Fix
Move `cudnn_cnn_infer64.dll` and `cudnn_ops_infer64.dll` into:
`dataset-maker\.venv\Lib\site-packages\ctranslate2\`

### 3. Generate the Dataset
1.  **Launch UI**: `uv run gradio_interface.py` -> Open `http://localhost:7860`.
2.  **Settings**:
    *   **Slicing**: Use `WhisperX Timestamps`.
    *   **Max Length**: 20 seconds.
    *   **Export Format**: `Qwen 3 TTS`.
3.  **Transfer**: Copy the project folder from `dataset-maker\datasets_folder\` to the root of your `Qwen3-TTS` repo.

---

## 🏗 Phase 3: Fine-Tuning

### 1. Prepare Data
Edit `prepare_data.bat`:
*   Change tokenizer path to `Qwen3-TTS-Tokenizer-12Hz`.
*   Point `--input_jsonl` to your dataset folder.

```cmd
prepare_data.bat
```

### 2. Training
Edit `train17.bat`:
*   **Batch Size**: 1-2 for 12GB VRAM; 8+ for 24GB+.
*   **Speaker Name**: 
    > [!CAUTION]
    > Do not use underscores followed by numbers (e.g., `Speaker_1`). This breaks UI parsing. Use `SpeakerOne`.

```cmd
train17.bat
```

---

## 🖥 Phase 4: Serving & Inference

### Local Gradio UI
```bash
uv run qwen-tts-demo runs/run1/checkpoint-epoch-10 --ip 0.0.0.0 --port 8001
```

---

## ⚡ Phase 5: Production API (Recommended)

For automated or high-volume workflows, use the **FastAPI Production Server**.

### 🌟 Key Enhancements
*   **S3 Integration**: Automatically uploads models and serves segments via presigned URLs.
*   **Disk LRU**: Manages 50GB instances by purging oldest models when usage hits **20GB**.
*   **VRAM Caching**: Instant voice switching (~10ms) by keeping multiple checkpoints in VRAM.

### 1. Start Service
```bash
# Set VRAM limits
export GPU_MAX_MODELS=4
bash start_api.sh
```

### 2. Trigger S3 Fine-tuning
```bash
curl -X POST http://localhost:8000/finetune \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_s3_key": "datasets/Project1/data.zip",
    "speaker_name": "HeroVoice",
    "num_epochs": 10
  }'
```

> [!TIP]
> Refer to `API_DOCS.md` for the full technical specification of the production endpoints.
