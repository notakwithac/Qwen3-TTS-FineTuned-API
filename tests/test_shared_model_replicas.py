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
