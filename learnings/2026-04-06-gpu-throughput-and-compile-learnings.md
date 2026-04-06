# GPU Throughput And Compile Learnings

## What We Learned
- Actual GPU generation is serialized per loaded replica because the active generate path still runs under a per-replica `model_lock`. That means one loaded replica can handle concurrent waiting, downloads, and uploads, but only one real generation at a time.
- Configured shared replica targets are only targets. A setting like `VOICE_DESIGN_REPLICAS=2` or a live `/gpu/concurrency` update does not guarantee that `replica-1` will ever load if true free VRAM never clears the runtime headroom gate.
- `torch.compile` is reusable only while the exact loaded model object stays resident. Restarting the process, unloading on idle timeout, evicting a model through LRU, or loading a second replica all trigger another compile path.
- `reserved_gb` matters as much as `allocated_gb` for practical admission. We repeatedly saw states where `allocated_gb` looked moderate but `reserved_gb` had consumed most of the card, leaving too little true free VRAM for new shared replicas.
- On the current A6000 setup, max throughput is better served by one hot `Base` model plus one hot `VoiceDesign` model with batching than by chasing extra shared replicas that do not fit reliably.

## Operational Guidance
- Keep `GPU_MAX_MODELS=7` so custom fine-tuned checkpoints can still be admitted when needed.
- Preload `VoiceDesign::replica-0` and `Base::replica-0` at startup so the first request does not pay the full load and compile path.
- Let those shared models remain evictable. They are warm defaults, not pinned reservations, so custom checkpoints can still reclaim VRAM through the normal cache policy.
- Use `GPU_IDLE_TIMEOUT=0` or a very large timeout in throughput mode if you want startup preload and compile costs to be paid once per process boot rather than repeatedly after idle unload.
- Prefer batch endpoints, especially `/voice-clone/batch`, as the main throughput lever. Larger batches usually outperform raising replica targets on the same GPU.

## Why Models Fight Each Other
- Clone and design traffic share the same GPU capacity budget, so one side can consume limiter slots while the other is still waiting to start.
- Shared models and custom models also compete for the same VRAM cache. If preloaded shared models are pinned forever, custom checkpoints lose flexibility; if they are always unloaded, the service pays repeated warm-up costs.
- `torch.compile` can improve steady-state speed for a resident model but can also increase memory pressure enough to block extra replicas, reducing overall concurrency.

## Next Investigations
- Benchmark `USE_TORCH_COMPILE=1` versus `0` on the same box with idle unload disabled so the comparison isolates steady-state throughput versus VRAM pressure.
- Add per-model-type quotas or separate admission pools so large clone waves cannot fully starve design traffic.
- Consider splitting clone and design traffic into separate workers or services if predictable latency matters more than single-process simplicity.
- Continue logging limiter holders, waiters, and startup preload events so future incidents show whether the bottleneck is admission, model load, compile warm-up, or steady-state execution.
