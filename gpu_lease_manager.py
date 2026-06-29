"""Thread-safe, expiring reservations for exclusive external GPU work."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable


MIN_TTL_SECONDS = 30
MAX_TTL_SECONDS = 900
MAX_OWNER_LENGTH = 128


class LeaseConflictError(RuntimeError):
    """Raised when another lease acquisition is pending or active."""


class InvalidLeaseTokenError(RuntimeError):
    """Raised when a token does not own the active lease."""


@dataclass(frozen=True)
class LeaseGrant:
    owner: str
    label: str
    permit_count: int
    acquired_at: float
    expires_at: float
    last_heartbeat_at: float
    token: str


class GpuLeaseManager:
    def __init__(
        self,
        *,
        acquire_permits: Callable[[str], int],
        release_permits: Callable[[str, int], None],
        prepare_gpu: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._acquire_permits = acquire_permits
        self._release_permits = release_permits
        self._prepare_gpu = prepare_gpu
        self._clock = clock
        self._lock = threading.RLock()
        self._lease: LeaseGrant | None = None
        self._pending = False
        self._closed = False

    @staticmethod
    def _validate_owner(owner: str) -> str:
        normalized = owner.strip() if isinstance(owner, str) else ""
        if not normalized or len(normalized) > MAX_OWNER_LENGTH:
            raise ValueError(
                f"owner must contain 1..{MAX_OWNER_LENGTH} non-whitespace characters"
            )
        return normalized

    @staticmethod
    def _validate_ttl(ttl_seconds: int) -> int:
        if isinstance(ttl_seconds, bool):
            raise ValueError(
                f"ttl_seconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}"
            )
        try:
            ttl = int(ttl_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ttl_seconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}"
            ) from exc
        if ttl != ttl_seconds or not MIN_TTL_SECONDS <= ttl <= MAX_TTL_SECONDS:
            raise ValueError(
                f"ttl_seconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}"
            )
        return ttl

    def acquire(self, owner: str, ttl_seconds: int) -> LeaseGrant:
        owner = self._validate_owner(owner)
        ttl = self._validate_ttl(ttl_seconds)
        self.reap_expired()
        label = f"external:{owner}"

        with self._lock:
            if self._closed:
                raise RuntimeError("GPU lease manager is closed")
            if self._pending or self._lease is not None:
                raise LeaseConflictError("An external GPU lease is already pending or active")
            self._pending = True

        permit_count = 0
        try:
            permit_count = int(self._acquire_permits(label))
            if permit_count < 1:
                raise RuntimeError("GPU permit reservation returned no permits")
            self._prepare_gpu()
            now = self._clock()
            lease = LeaseGrant(
                owner=owner,
                label=label,
                permit_count=permit_count,
                acquired_at=now,
                expires_at=now + ttl,
                last_heartbeat_at=now,
                token=secrets.token_urlsafe(32),
            )
            with self._lock:
                if self._closed:
                    raise RuntimeError("GPU lease manager is closed")
                self._lease = lease
                self._pending = False
            return lease
        except BaseException:
            if permit_count > 0:
                self._release_permits(label, permit_count)
            with self._lock:
                self._pending = False
            raise

    def heartbeat(self, token: str, ttl_seconds: int | None = None) -> LeaseGrant:
        ttl = self._validate_ttl(ttl_seconds) if ttl_seconds is not None else None
        self.reap_expired()
        with self._lock:
            lease = self._lease
            if lease is None or not secrets.compare_digest(token, lease.token):
                raise InvalidLeaseTokenError("Invalid or expired GPU lease token")
            now = self._clock()
            effective_ttl = ttl if ttl is not None else int(lease.expires_at - lease.last_heartbeat_at)
            renewed = replace(
                lease,
                expires_at=now + effective_ttl,
                last_heartbeat_at=now,
            )
            self._lease = renewed
            return renewed

    def release(self, token: str) -> bool:
        with self._lock:
            lease = self._lease
            if lease is None:
                return False
            if not secrets.compare_digest(token, lease.token):
                raise InvalidLeaseTokenError("Invalid GPU lease token")
            self._lease = None
        self._release_permits(lease.label, lease.permit_count)
        return True

    def reap_expired(self) -> bool:
        with self._lock:
            lease = self._lease
            if lease is None or self._clock() < lease.expires_at:
                return False
            self._lease = None
        self._release_permits(lease.label, lease.permit_count)
        return True

    def status(self) -> dict[str, object]:
        self.reap_expired()
        with self._lock:
            lease = self._lease
            if lease is None:
                return {"active": False, "pending": self._pending}
            return {
                "active": True,
                "pending": False,
                "owner": lease.owner,
                "label": lease.label,
                "permit_count": lease.permit_count,
                "acquired_at": lease.acquired_at,
                "expires_at": lease.expires_at,
                "last_heartbeat_at": lease.last_heartbeat_at,
            }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            lease = self._lease
            self._lease = None
        if lease is not None:
            self._release_permits(lease.label, lease.permit_count)
