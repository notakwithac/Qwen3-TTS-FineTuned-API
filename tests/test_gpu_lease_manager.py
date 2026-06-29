import threading

import pytest

from gpu_lease_manager import (
    GpuLeaseManager,
    InvalidLeaseTokenError,
    LeaseConflictError,
)


class FakeClock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def build_manager(clock, *, acquired=2, prepared=None, released=None):
    prepared = prepared if prepared is not None else []
    released = released if released is not None else []
    manager = GpuLeaseManager(
        acquire_permits=lambda label: acquired,
        release_permits=lambda label, count: released.append((label, count)),
        prepare_gpu=lambda: prepared.append(True),
        clock=clock,
    )
    return manager, prepared, released


def test_acquire_returns_opaque_grant_and_sanitized_status():
    manager, prepared, _released = build_manager(FakeClock(100.0))

    lease = manager.acquire(owner="dataset-lab", ttl_seconds=30)

    assert lease.owner == "dataset-lab"
    assert lease.permit_count == 2
    assert len(lease.token) >= 32
    assert prepared == [True]
    status = manager.status()
    assert status["active"] is True
    assert status["owner"] == "dataset-lab"
    assert "token" not in status
    assert lease.token not in repr(status)


def test_active_or_pending_acquisition_conflicts():
    permit_started = threading.Event()
    allow_permit = threading.Event()

    def acquire_permits(_label):
        permit_started.set()
        assert allow_permit.wait(1)
        return 1

    manager = GpuLeaseManager(
        acquire_permits=acquire_permits,
        release_permits=lambda _label, _count: None,
        prepare_gpu=lambda: None,
    )
    worker = threading.Thread(
        target=lambda: manager.acquire(owner="first", ttl_seconds=30)
    )
    worker.start()
    assert permit_started.wait(1)

    with pytest.raises(LeaseConflictError):
        manager.acquire(owner="second", ttl_seconds=30)

    allow_permit.set()
    worker.join(1)
    manager.close()


def test_heartbeat_extends_lease_and_rejects_wrong_token():
    clock = FakeClock(100.0)
    manager, _prepared, _released = build_manager(clock)
    lease = manager.acquire(owner="dataset-lab", ttl_seconds=30)
    clock.advance(10)

    renewed = manager.heartbeat(lease.token, ttl_seconds=60)

    assert renewed.expires_at == 170.0
    assert renewed.last_heartbeat_at == 110.0
    with pytest.raises(InvalidLeaseTokenError):
        manager.heartbeat("wrong-token")


def test_release_is_idempotent_and_releases_exact_reservation():
    manager, _prepared, released = build_manager(FakeClock(100.0))
    lease = manager.acquire(owner="dataset-lab", ttl_seconds=30)

    assert manager.release(lease.token) is True
    assert manager.release(lease.token) is False
    assert released == [(lease.label, 2)]
    assert manager.status()["active"] is False


def test_expired_lease_releases_exact_reservation():
    clock = FakeClock(100.0)
    manager, _prepared, released = build_manager(clock)
    lease = manager.acquire(owner="dataset-lab", ttl_seconds=30)
    clock.advance(31)

    assert manager.reap_expired() is True

    assert released == [(lease.label, 2)]
    assert manager.reap_expired() is False
    assert manager.status()["active"] is False


def test_prepare_failure_releases_permits_and_clears_pending_state():
    released = []
    manager = GpuLeaseManager(
        acquire_permits=lambda _label: 3,
        release_permits=lambda label, count: released.append((label, count)),
        prepare_gpu=lambda: (_ for _ in ()).throw(RuntimeError("prepare failed")),
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        manager.acquire(owner="dataset-lab", ttl_seconds=30)

    assert released == [("external:dataset-lab", 3)]
    assert manager.status() == {"active": False, "pending": False}


@pytest.mark.parametrize("owner", ["", "x" * 129])
def test_owner_length_is_validated(owner):
    manager, _prepared, _released = build_manager(FakeClock(100.0))
    with pytest.raises(ValueError, match="owner"):
        manager.acquire(owner=owner, ttl_seconds=30)


@pytest.mark.parametrize("ttl", [29, 901])
def test_ttl_bounds_are_validated(ttl):
    manager, _prepared, _released = build_manager(FakeClock(100.0))
    with pytest.raises(ValueError, match="ttl_seconds"):
        manager.acquire(owner="dataset-lab", ttl_seconds=ttl)
