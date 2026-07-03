# Qwen3-TTS Production Fine-Tuning & Inference API

<p align="center">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" alt="Qwen3-TTS" width="380">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache 2.0 License"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/API-FastAPI-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/GPU-CUDA-76B900.svg" alt="CUDA">
</p>

A production-oriented service for fine-tuning, cloning, designing, and serving voices with [Alibaba's Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). It wraps the model stack in a concurrent FastAPI service with job management, S3-compatible storage, GPU lifecycle controls, batching, monitoring, and failure recovery.

This is not the upstream Qwen3-TTS repository. It is an independent production harness built around the upstream models and training code.

## Latest integrations

- **Gemma 4 / e4b LLM proxy is done.** The service now exposes OpenAI-compatible `/v1` routes backed by a lazy vLLM Gemma process, so Pathnam can run chapter extraction and dialogue attribution through the same local GPU stack that serves TTS.
- **Hugging Face checkpoint storage is integrated.** When `HF_TOKEN` is configured, fine-tuned checkpoints are uploaded to Hugging Face repos, job metadata records the repo URL, and inference can restore checkpoints from Hugging Face when local disk or S3 copies are not present.
- **Model prefetching uses the shared Hugging Face cache.** Qwen base models, VoiceDesign, tokenizer assets, Gemma, and Sarvam Translate can be pre-downloaded at startup without keeping every model resident in VRAM.
- **Pathnam integration is first-class.** Point Pathnam at this API with `FINETUNE_API_URL` for voice work and `LLM_PROVIDER=vllm`, `LLM_MODEL=e4b`, `LLM_BASE_URL=http://<qwen-api-host>:8000/v1` for local LLM extraction.

## Why this repository exists

Running a speech model once is different from operating it as a service. This project handles the machinery around the model:

- Fine-tuning jobs with persistent status, progress, retry, cancellation, and checkpoint recovery
- Dataset preparation with an explicit review boundary before final packaging
- Fine-tuned inference, zero-shot voice cloning, and natural-language voice design
- Batch and event-driven generation for narration and multi-character workloads
- GPU-safe model caching, runtime concurrency controls, idle unloading, and VRAM-aware replica admission
- S3-compatible model backup and audio delivery through presigned URLs
- Disk LRU cleanup for storage-constrained GPU instances
- Live CPU, RAM, GPU, VRAM, operation timing, and server-log visibility
- Optional Indian-language translation through `sarvamai/sarvam-translate`
- Idle GPU instance shutdown and periodic checkpoint/log uploads for supported cloud environments

The result is a service that can train a voice, generate speech, recover from failures, and manage limited GPU and disk capacity without constant manual supervision.

## Core workflows

| Workflow | What it does | Primary API |
|---|---|---|
| Fine-tune a voice | Trains Qwen3-TTS from a packaged dataset and persists the resulting checkpoint | `POST /finetune` |
| Fine-tuned inference | Generates one or many clips from a completed fine-tuning job | `POST /infer/{job_id}` |
| Voice design | Creates speech from a natural-language voice description | `POST /voice-design` |
| Zero-shot clone | Clones a voice from reference audio and its transcript | `POST /voice-clone` |
| Dataset preparation | Transcribes and prepares candidate clips for human review and packaging | `POST /dataset/prepare` |
| Local dataset synthesis | Expands an authorized reference voice into Qwen-ready dialogue clips with dots.tts-mf | [docs/dots-tts-dataset.md](docs/dots-tts-dataset.md) |
| Event-driven narration | Preloads character models and processes queued lines in the background | `POST /session/prepare` |
| Translation | Translates text across English and 22 Indian languages | `POST /translate` |

Interactive OpenAPI documentation is available at `http://localhost:8000/docs` while the service is running.

## Architecture

```text
Client / application
        |
        v
  FastAPI service  <---->  Job and session managers
        |                         |
        |                         +---- queues, retries, progress, logs
        v
  GPU resource controller
        |
        +---- Qwen3-TTS Base / VoiceDesign / fine-tuned checkpoints
        +---- VRAM-aware cache, batching, replicas, idle unloading
        |
        v
  S3-compatible storage  <---->  local disk LRU
        |
        +---- datasets, checkpoints, generated audio, logs, metrics
```

Fine-tuning follows a persistent asynchronous lifecycle:

```text
queued -> preparing -> training -> loading -> ready
                       |                    |
                       +---- failed <-------+
                              |
                              +---- retry / resume
```

## Quick start with Docker Compose

### Requirements

- Linux with an NVIDIA CUDA-capable GPU
- NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker Engine with Compose support
- Enough GPU memory for the selected model and workload; concurrency is admitted according to live VRAM headroom

Clone and configure the service:

```bash
git clone https://github.com/notakwithac/Qwen3-TTS-FineTuned-API.git
cd Qwen3-TTS-FineTuned-API
cp .env.example .env
```

The default environment starts a local MinIO instance, so the S3-backed workflows work without external object storage. Change the credentials and endpoints in `.env` before exposing the stack.

Start the API, MinIO, and bucket initializer:

```bash
docker compose up --build
```

Then verify the service:

```bash
curl http://localhost:8000/
curl http://localhost:8000/storage/status
```

Open:

- API documentation: <http://localhost:8000/docs>
- MinIO console: <http://localhost:9001>
- MinIO S3 endpoint: <http://localhost:9000>

Gemma e4b and Sarvam Translate run as lazy vLLM subprocesses inside the API
container. Plain compose starts the API and storage; the first Gemma/Sarvam
request loads its model into GPU, and the idle timeout stops it again:

```bash
docker compose up --build
```

By default startup pre-downloads model files into the shared Hugging Face cache
volume without loading them into GPU:

```env
PREFETCH_MODELS_ON_START=1
PREFETCH_MODEL_SET=base voice_design tokenizer gemma sarvam_translate
```

Set `PREFETCH_MODELS_ON_START=0` for faster API boot with first-request
downloads. Gemma may require `HF_TOKEN` accepted for the selected Hugging Face
repo.

Qwen3-TTS exposes OpenAI-compatible Gemma routes on its own API port:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /gemma/status`
- `GET /vllm/status`

Point Pathnam at Qwen3-TTS with `LLM_PROVIDER=vllm`, `LLM_MODEL=e4b`, and `LLM_BASE_URL=http://<qwen-api-host>:8000/v1`. The proxy unloads resident Qwen models before starting Gemma, stops any active Sarvam vLLM process first, and holds the same GPU limiter used by voice design, voice clone, and custom voice inference.

## TIR or direct Linux setup

The included scripts target a Python 3.11 and CUDA 12.8 GPU environment:

```bash
bash setup_tir.sh
cp .env.example .env
bash start_api.sh
```

`setup_tir.sh` creates `.venv`, installs the CUDA dependencies, installs FlashAttention, and verifies the GPU. `start_api.sh` launches the GPU idle watchdog and serves FastAPI on port `8000`.

For package-level model usage without this service, follow the [upstream Qwen3-TTS quick start](https://github.com/QwenLM/Qwen3-TTS).

## End-to-end API example

Set a base URL for the examples:

```bash
export BASE_URL=http://localhost:8000
```

### 1. Start fine-tuning

The dataset zip must already exist in the configured S3-compatible bucket. See [Fine-tuning data format](finetuning/README.md) and the dataset preparation endpoints in [API documentation](API_DOCS.md).

```bash
curl -X POST "$BASE_URL/finetune" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_s3_key": "datasets/narrator_voice.zip",
    "speaker_name": "narrator_voice",
    "num_epochs": 15,
    "batch_size": 1,
    "lr": 0.000001
  }'
```

The response includes a `job_id`. Follow its progress:

```bash
curl "$BASE_URL/jobs/<job_id>"
```

### 2. Generate with the trained voice

Once the job reaches `ready`:

```bash
curl -X POST "$BASE_URL/infer/<job_id>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "A production voice should survive more than the happy path.",
    "language": "English",
    "upload_to_s3": true,
    "s3_filename": "demo/narrator.wav"
  }'
```

With `upload_to_s3: true`, the API returns JSON containing the stored audio location and a presigned URL. Set it to `false` to receive WAV audio directly.

### 3. Design a voice without training

```bash
curl -X POST "$BASE_URL/voice-design" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The old observatory woke beneath a sky full of static.",
    "instruct": "A calm, weathered storyteller with restrained wonder",
    "language": "English",
    "upload_to_s3": true
  }'
```

### 4. Clone a voice from reference audio

Use only audio you have permission to process and clone.

```bash
curl -X POST "$BASE_URL/voice-clone" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This line uses the reference speaker.",
    "ref_audio_url": "s3://qwen3-tts/references/speaker.wav",
    "ref_text": "This is the exact transcript of the reference audio.",
    "language": "English",
    "upload_to_s3": true
  }'
```

Batch clone requests return immediately with a session ID. Poll `GET /voice-clone/batch/{session_id}` for completion.

## Dataset approval boundary

Dataset construction is deliberately split into two steps:

1. `POST /dataset/prepare` creates candidate `dataset_items` from validated audio.
2. A person or upstream service reviews, edits, and approves those items.
3. `POST /dataset/package` builds the final Qwen fine-tuning archive from the approved items.

Preparation never silently auto-packages a training dataset. This keeps transcript, speaker, and reference-audio decisions visible before GPU training begins.

## GPU and throughput controls

The service serializes or overlaps training and inference according to GPU capacity and configuration. Shared VoiceDesign and Base models can be preloaded at startup, while custom checkpoints use the same VRAM-aware cache.

Important defaults live in [.env.example](.env.example):

| Variable | Purpose |
|---|---|
| `ALLOW_CONCURRENT_TRAINING_INFERENCE` | Allows training and inference to overlap when the GPU can support it |
| `GPU_MAX_MODELS` | Capacity budget for resident models, including custom checkpoints |
| `GPU_IDLE_TIMEOUT` | Unloads models after inactivity; `0` keeps hot models resident |
| `VOICE_DESIGN_REPLICAS` | Startup target for shared VoiceDesign replicas |
| `VOICE_CLONE_REPLICAS` | Startup target for shared Base/clone replicas |
| `SHARED_MODEL_MIN_HEADROOM_GB` | Free VRAM required before admitting another shared replica |
| `GPU_BATCH_SIZE` | Upper bound for supported inference batching paths |
| `MANAGED_VLLM_ENABLED` | Starts Gemma/Sarvam vLLM lazily and keeps them mutually exclusive with Qwen |
| `MANAGED_VLLM_IDLE_TIMEOUT_SECONDS` | Stops idle Gemma/Sarvam vLLM processes after inactivity |
| `PREFETCH_MODELS_ON_START` | Downloads model files at startup without loading GPU |
| `PREFETCH_MODEL_SET` | Space-separated model keys to pre-cache |
| `GEMMA_VLLM_MODEL` | Model id exposed through Qwen3-TTS `/v1` proxy |
| `SARVAM_VLLM_MODEL` | Model id used by the `/translate` proxy |

Runtime controls let you tune a live process without redeploying:

```bash
curl "$BASE_URL/gpu/concurrency"

curl -X POST "$BASE_URL/gpu/concurrency" \
  -H "Content-Type: application/json" \
  -d '{
    "gpu_max_models": 7,
    "voice_design_replicas": 1,
    "voice_clone_replicas": 1,
    "shared_model_min_headroom_gb": 4
  }'
```

Batching is usually the first throughput lever on a single GPU. Raising replica targets does not guarantee that replicas will load; admission still depends on actual free VRAM.

## Storage and disk management

The service supports AWS S3, E2E Object Storage, MinIO, and other S3-compatible backends.

- Generated audio is organized and uploaded to object storage.
- Private objects can be returned through time-limited presigned URLs.
- Fine-tuned model checkpoints are uploaded to public Hugging Face repos when `HF_TOKEN` is set; older jobs with only S3 metadata remain restorable from object storage.
- Raw datasets and heavy intermediate training artifacts are cleaned up after use.
- The local `jobs/` model cache is pruned by least-recently-used access when it exceeds its configured threshold.
- An optional fallback S3 backend can restore custom models that are not present in the primary store.

Model checkpoint env vars:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Authenticates Hugging Face upload/download for fine-tuned checkpoints and dataset diarization |
| `HF_MODEL_NAMESPACE` | Hugging Face org/user namespace for checkpoint repos (default: `vaaniqo-finetuned-models`) |
| `HF_MODEL_REPO_PREFIX` | Repo name prefix, e.g. `Qwen3-Book-Speaker-jobid` (default: `Qwen3`) |

Check the active backend with `GET /storage/status` and trigger disk cleanup with `GET /gpu/cleanup?threshold_gb=20`.

## Observability and operations

No separate monitoring stack is required for basic diagnosis:

| Endpoint | Signal |
|---|---|
| `GET /gpu/status` | Loaded models and GPU state |
| `GET /gpu/vram` | VRAM capacity and allocations |
| `GET /gpu/metrics` | Current CPU, RAM, GPU, and VRAM utilization |
| `GET /gpu/metrics/history` | Historical resource samples |
| `GET /ops/running` | Operations currently in flight |
| `GET /ops/history` | Per-operation timing history |
| `GET /ops/averages` | Aggregated stage durations |
| `GET /logs/stream` | Live server-sent log stream |

Stress utilities are included for shared-model concurrency and log-stream verification:

```bash
python stress_shared_model_replicas.py --base-url http://localhost:8000
python stress_logs_stream.py --base-url http://localhost:8000
```

## Event-driven multi-character inference

Long-form narration can preload character voices, submit lines as they arrive, process work in the background, and tear the session down when complete:

```text
POST /session/prepare
        |
        v
POST /session/{id}/submit or /submit/batch
        |
        v
GET /session/{id}/status
        |
        v
DELETE /session/{id}
```

See the [event-driven inference guide](EVENT_DRIVEN_INFERENCE_DOCS.md) for request schemas and a complete Python integration.

## Translation

The optional Sarvam service supports English plus Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, and Urdu.

```bash
curl -X POST "$BASE_URL/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Every voice carries a place with it.",
    "source_language": "English",
    "target_language": "Hindi",
    "max_new_tokens": 256
  }'
```

In Docker, translation uses the same lazy vLLM manager as Gemma. Sarvam, Gemma, and Qwen TTS models are not intentionally kept resident together: starting one external vLLM model stops the other, and Qwen model loading stops external vLLM first.

## Repository guide

| Path | Purpose |
|---|---|
| `api_server.py` | FastAPI routes, request validation, and service orchestration |
| `pipeline.py` | Fine-tuning job lifecycle and checkpoint management |
| `inference_manager.py` | Model loading, caching, batching, and inference execution |
| `dataset_jobs.py` | Dataset preparation and packaging jobs |
| `session_manager.py` | Event-driven inference sessions and worker queues |
| `gpu_resource_controller.py` | Training/inference resource isolation |
| `gpu_idle_watchdog.py` | Idle detection, heartbeat uploads, and cloud termination |
| `storage.py` | S3-compatible storage and presigned URL handling |
| `metrics_collector.py` | CPU, RAM, GPU, and VRAM metrics |
| `finetuning/` | Qwen3-TTS SFT scripts and data-format guide |
| `tests/` | Unit and integration tests |

## Documentation

- [Complete API reference](API_DOCS.md)
- [Event-driven inference guide](EVENT_DRIVEN_INFERENCE_DOCS.md)
- [Fine-tuning data and CLI guide](finetuning/README.md)
- [Postman collection](Qwen3-TTS_Postman_Collection.json)
- [Environment variable reference](.env.example)
- [Upstream Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)

## Production notes

- Authentication is not enabled by default. Put the API behind an authenticated reverse proxy or add FastAPI authentication middleware before exposing it publicly.
- Replace the example MinIO credentials in `.env` for every non-local deployment.
- Voice cloning can carry consent, impersonation, privacy, and jurisdiction-specific obligations. Only process voices and datasets you are authorized to use.
- GPU memory requirements vary with model size, training settings, batching, compilation, and concurrent workloads. Begin with conservative settings and observe `/gpu/vram` under realistic load.
- S3 uploads and presigned URLs should use TLS-backed endpoints outside local development.

## Upstream attribution

Qwen3-TTS and its core model implementation were developed by the Alibaba Qwen team. This repository adds the production fine-tuning, inference, storage, GPU management, monitoring, and service orchestration layer.

If you use the underlying model in research, cite the Qwen3-TTS technical report:

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

Review the upstream model cards and repository for model-specific terms, limitations, and release information.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

Copyright and attribution notices for upstream and third-party components remain governed by their respective files and licenses.
