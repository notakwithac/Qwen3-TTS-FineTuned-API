# Startup Shared-Model Preload And Throughput-Mode Plan

## Summary
Implement a warm-start throughput mode that preloads one shared `VoiceDesign` model and one shared `Base` model during FastAPI startup, keeps `GPU_MAX_MODELS=7` available for future custom models, and allows the preloaded shared models to be evicted later if custom checkpoints need VRAM. Save the implementation plan under `docs/plans/` and the operational learnings from this debugging session under `learnings/`.

## Key Changes
- **Startup preload in `api_server.py`**
  - Add startup preload logic inside `_lifespan()` after broadcast/session/metrics startup but before yielding control.
  - Preload `pipeline.inference.load_voice_design()`.
  - Preload `pipeline.inference.load_voice_clone()`.
  - Log `startup_preload_started`, `startup_preload_finished`, and `startup_preload_failed` events with post-load `pipeline.inference.stats`.
  - On preload failure, log a warning and continue boot so the server falls back to lazy loading later.
  - Do not pin these models; let existing eviction rules reclaim them if custom models need space.

- **Warm-throughput operating defaults and docs**
  - Update `.env.example` to document the recommended throughput mode:
    - keep `GPU_MAX_MODELS=7`
    - recommend `VOICE_DESIGN_REPLICAS=1`
    - recommend `VOICE_CLONE_REPLICAS=1`
    - recommend `GPU_IDLE_TIMEOUT=0` or a very large value when compile/warm-start benefits should persist
  - Update `API_DOCS.md` to explain:
    - startup preload warms only the first shared replica of each shared model type
    - extra runtime replica targets may still not materialize if real free VRAM is insufficient
    - throughput is currently best achieved through hot resident models plus batch requests, not by expecting one replica to serve many GPU generations in parallel

- **No new public endpoints required**
  - Keep existing APIs unchanged.
  - Reuse `/gpu/status` and existing ops logs to verify preload success and ongoing residency/eviction behavior.
  - Preserve `GPU_MAX_MODELS=7` semantics so custom session/custom-voice models can still load on demand.

- **Separate learnings note**
  - Create `learnings/2026-04-06-gpu-throughput-and-compile-learnings.md`.
  - Capture:
    - current concurrency reality: per-replica `model_lock` serializes actual generation
    - why configured shared replica counts can be higher than effective loaded replicas
    - why `torch.compile` is one-time per resident model object, not globally once
    - why allocator reservation (`reserved_gb`) can suppress replica expansion even when `allocated_gb` looks moderate
    - why max throughput currently favors one hot `Base` plus one hot `VoiceDesign` model with batching
    - next-step ideas for reducing shared-model contention, including per-model quotas/service split and compile-vs-memory tradeoff benchmarking

## Implementation Tasks
1. Save this implementation plan to `docs/plans/2026-04-06-startup-shared-model-preload-throughput.md`.
2. Add startup preload logging and preload execution in `_lifespan()` using the existing `InferenceManager.load_voice_design()` and `load_voice_clone()` helpers.
3. Ensure preload events record enough context for operations review: loaded checkpoints, limiter snapshot, GPU memory snapshot, and whether boot is continuing after failure.
4. Keep eviction behavior unchanged for the preloaded shared models so custom models can reclaim VRAM through normal LRU behavior.
5. Update operator-facing docs in `.env.example` and `API_DOCS.md` to describe the intended throughput mode and recommended settings.
6. Write the separate learnings markdown note at `learnings/2026-04-06-gpu-throughput-and-compile-learnings.md`.

## Test Plan
- Add or extend tests around startup preload behavior:
  - preload success logs completion and results in both shared checkpoints appearing in `pipeline.inference.stats`
  - preload failure logs a warning/error event and does not abort startup
- Add a focused test for eviction policy:
  - startup-preloaded shared models remain ordinary cached entries and are still evictable when cache pressure requires room for other models
- Add or update docs-facing verification:
  - recommended throughput env settings appear in `.env.example`
  - startup preload behavior and throughput guidance appear in `API_DOCS.md`
- Manual validation on a live box:
  - boot server with throughput settings
  - confirm `/gpu/status` shows both shared models loaded shortly after startup
  - confirm first request latency is reduced versus lazy-load startup
  - confirm a later custom-model load can still evict shared models if VRAM pressure requires it

## Assumptions And Defaults
- Default startup failure policy: boot with warning and continue
- Preload only `replica-0` for `VoiceDesign` and `Base`; do not attempt to materialize additional shared replicas at startup
- Keep `GPU_MAX_MODELS=7` to preserve headroom for custom-model loading
- Use batching plus hot resident shared models as the throughput strategy for now; do not redesign replica architecture in this change
- Save implementation plans under `docs/plans/` and long-lived debugging/operational notes under `learnings/`
