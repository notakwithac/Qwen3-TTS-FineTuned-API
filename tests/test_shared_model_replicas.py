import inference_manager
import logging
import sys
import types

import pytest
from inference_manager import InferenceManager, VOICE_DESIGN_MODEL


def test_shared_replica_keys_are_distinct():
    manager = InferenceManager(
        device="cpu",
        shared_model_replicas={"voice_design": 3},
    )

    keys = manager._shared_replica_keys(VOICE_DESIGN_MODEL, "voice_design")

    assert keys == [
        f"{VOICE_DESIGN_MODEL}::replica-0",
        f"{VOICE_DESIGN_MODEL}::replica-1",
        f"{VOICE_DESIGN_MODEL}::replica-2",
    ]


def test_acquire_shared_replica_expands_when_loaded_replica_is_busy():
    manager = InferenceManager(
        device="cpu",
        max_models=4,
        shared_model_replicas={"voice_design": 2},
    )
    first = manager._build_shared_replica_key(VOICE_DESIGN_MODEL, 0)
    second = manager._build_shared_replica_key(VOICE_DESIGN_MODEL, 1)
    manager._models[first] = (object(), "voice_design", None)
    manager._shared_replica_loads[first] = 1
    manager._has_shared_replica_headroom_locked = lambda: True

    selected = manager._acquire_shared_replica(VOICE_DESIGN_MODEL, "voice_design")

    assert selected == second


def test_acquire_shared_replica_reuses_loaded_replica_when_headroom_is_tight():
    manager = InferenceManager(
        device="cpu",
        max_models=4,
        shared_model_replicas={"voice_design": 2},
    )
    first = manager._build_shared_replica_key(VOICE_DESIGN_MODEL, 0)
    manager._models[first] = (object(), "voice_design", None)
    manager._shared_replica_loads[first] = 1
    manager._has_shared_replica_headroom_locked = lambda: False

    selected = manager._acquire_shared_replica(VOICE_DESIGN_MODEL, "voice_design")

    assert selected == first


def test_runtime_max_models_update_changes_effective_limit():
    manager = InferenceManager(device="cpu", max_models=2)

    manager.update_runtime_config(max_models=4)

    assert manager.max_models == 4
    assert manager.stats["inference_limiter"] == {
        "capacity": 4,
        "active": 0,
        "available": 4,
    }


def test_runtime_shared_replica_update_changes_targets():
    manager = InferenceManager(device="cpu", shared_model_replicas={"voice_design": 1})

    manager.update_runtime_config(shared_model_replicas={"voice_design": 3})

    assert manager.stats["shared_model_replicas"]["voice_design"] == 3
    assert manager._shared_replica_keys(VOICE_DESIGN_MODEL, "voice_design") == [
        f"{VOICE_DESIGN_MODEL}::replica-0",
        f"{VOICE_DESIGN_MODEL}::replica-1",
        f"{VOICE_DESIGN_MODEL}::replica-2",
    ]


def test_shared_replica_headroom_uses_driver_free_memory(monkeypatch):
    manager = InferenceManager(device="cuda:0")

    class _Props:
        total_memory = int(48e9)

    monkeypatch.setattr(inference_manager.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        inference_manager.torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: _Props(),
    )
    monkeypatch.setattr(inference_manager.torch.cuda, "memory_allocated", lambda *_args, **_kwargs: int(1e9))
    monkeypatch.setattr(inference_manager.torch.cuda, "memory_reserved", lambda *_args, **_kwargs: int(2e9))
    monkeypatch.setattr(
        inference_manager.torch.cuda,
        "mem_get_info",
        lambda *_args, **_kwargs: (int(8e9), int(48e9)),
    )

    assert manager._has_shared_replica_headroom_locked() is False


def test_get_vram_budget_reports_driver_free_memory(monkeypatch):
    manager = InferenceManager(device="cuda:0")

    class _Props:
        total_memory = int(48e9)

    monkeypatch.setattr(inference_manager.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        inference_manager.torch.cuda,
        "get_device_properties",
        lambda *_args, **_kwargs: _Props(),
    )
    monkeypatch.setattr(inference_manager.torch.cuda, "memory_allocated", lambda *_args, **_kwargs: int(1e9))
    monkeypatch.setattr(inference_manager.torch.cuda, "memory_reserved", lambda *_args, **_kwargs: int(2e9))
    monkeypatch.setattr(
        inference_manager.torch.cuda,
        "mem_get_info",
        lambda *_args, **_kwargs: (int(8e9), int(48e9)),
    )

    assert manager.get_vram_budget() == {
        "total_gb": 48.0,
        "allocated_gb": 1.0,
        "free_gb": 8.0,
        "models_loaded": 0,
        "session_pinned": 0,
    }


def test_acquire_shared_replica_logs_selection_context(monkeypatch):
    manager = InferenceManager(
        device="cpu",
        max_models=4,
        shared_model_replicas={"voice_design": 2},
    )
    first = manager._build_shared_replica_key(VOICE_DESIGN_MODEL, 0)
    manager._models[first] = (object(), "voice_design", None)
    manager._shared_replica_loads[first] = 1
    manager._has_shared_replica_headroom_locked = lambda: False

    events = []

    def _capture_event(event_name, job_id=None, extra=None, level=logging.INFO):
        events.append((event_name, extra, level))

    monkeypatch.setattr(inference_manager.ops_log, "log_event", _capture_event)

    selected = manager._acquire_shared_replica(VOICE_DESIGN_MODEL, "voice_design")

    assert selected == first
    assert events
    event_name, extra, level = events[-1]
    assert event_name == "shared_replica_selection"
    assert level == logging.INFO
    assert extra["model_type"] == "voice_design"
    assert extra["selected_cache_key"] == first
    assert extra["selection_kind"] == "reuse_loaded"
    assert extra["headroom_ok"] is False
    assert extra["can_expand"] is False


def test_load_model_into_cache_logs_failure_context(monkeypatch):
    manager = InferenceManager(device="cpu")
    failure_events = []

    class _StubQwen3TTSModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")

    stub_module = types.ModuleType("qwen_tts")
    stub_module.Qwen3TTSModel = _StubQwen3TTSModel

    monkeypatch.setitem(sys.modules, "qwen_tts", stub_module)
    monkeypatch.setattr(inference_manager.ops_log, "log_event", lambda *args, **kwargs: None)

    def _capture_fail(record, error, extra=None):
        failure_events.append((record.op_name, error, extra))

    monkeypatch.setattr(inference_manager.ops_log, "fail", _capture_fail)

    with pytest.raises(RuntimeError, match="busy or unavailable"):
        manager._load_model_into_cache(
            cache_key=f"{VOICE_DESIGN_MODEL}::replica-0",
            source_path=VOICE_DESIGN_MODEL,
            model_type="voice_design",
        )

    assert failure_events
    op_name, error, extra = failure_events[-1]
    assert op_name == "model_load"
    assert "busy or unavailable" in error
    assert extra["model_type"] == "voice_design"
    assert extra["cache_key"].endswith("::replica-0")
    assert extra["stage"] == "load_model_into_cache"
    assert "gpu_memory" in extra
