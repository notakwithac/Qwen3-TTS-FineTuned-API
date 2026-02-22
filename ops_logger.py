# coding=utf-8
"""Structured operations logger — timestamps, durations, and averages for all pipeline stages.

Usage:
    from ops_logger import ops_log

    with ops_log.operation("prepare_data", job_id="abc123"):
        # ... do work ...
        pass

    # Or manual:
    op = ops_log.start("training", job_id="abc123", extra={"epoch": 0})
    # ... do work ...
    ops_log.end(op, extra={"loss": 0.02})

    # Get averages:
    ops_log.get_averages()  # {"prepare_data": {"avg_seconds": 12.5, "count": 4}, ...}
"""

import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ops")


class OperationRecord:
    """One logged operation."""

    def __init__(
        self,
        op_name: str,
        op_id: str,
        job_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.op_name = op_name
        self.op_id = op_id
        self.job_id = job_id
        self.start_time = time.time()
        self.start_ts = datetime.now(timezone.utc).isoformat()
        self.end_time: Optional[float] = None
        self.end_ts: Optional[str] = None
        self.duration_seconds: Optional[float] = None
        self.status: str = "running"  # running | completed | failed
        self.error: Optional[str] = None
        self.extra: Dict[str, Any] = extra or {}

    def complete(self, extra: Optional[Dict[str, Any]] = None):
        self.end_time = time.time()
        self.end_ts = datetime.now(timezone.utc).isoformat()
        self.duration_seconds = round(self.end_time - self.start_time, 3)
        self.status = "completed"
        if extra:
            self.extra.update(extra)

    def fail(self, error: str, extra: Optional[Dict[str, Any]] = None):
        self.end_time = time.time()
        self.end_ts = datetime.now(timezone.utc).isoformat()
        self.duration_seconds = round(self.end_time - self.start_time, 3)
        self.status = "failed"
        self.error = error
        if extra:
            self.extra.update(extra)

    def to_dict(self) -> dict:
        d = {
            "op_id": self.op_id,
            "op_name": self.op_name,
            "job_id": self.job_id,
            "status": self.status,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_seconds": self.duration_seconds,
        }
        if self.error:
            d["error"] = self.error
        if self.extra:
            d["extra"] = self.extra
        return d

    def _log_line(self, event: str) -> str:
        parts = [
            f"[{event}]",
            f"op={self.op_name}",
            f"op_id={self.op_id}",
        ]
        if self.job_id:
            parts.append(f"job_id={self.job_id}")
        if self.duration_seconds is not None:
            parts.append(f"duration={self.duration_seconds}s")
        if self.error:
            parts.append(f"error={self.error[:100]}")
        for k, v in self.extra.items():
            parts.append(f"{k}={v}")
        return " | ".join(parts)


class OpsLogger:
    """Central operations logger with timing and averages."""

    def __init__(self, max_history: int = 1000):
        self._lock = threading.Lock()
        self._history: List[OperationRecord] = []
        self._max_history = max_history
        self._running: Dict[str, OperationRecord] = {}

    def start(
        self,
        op_name: str,
        job_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> OperationRecord:
        """Start tracking an operation. Returns the record to pass to end()."""
        op_id = uuid.uuid4().hex[:8]
        record = OperationRecord(op_name, op_id, job_id, extra)

        with self._lock:
            self._running[op_id] = record

        logger.info(record._log_line("START"))
        return record

    def end(
        self,
        record: OperationRecord,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Complete an operation successfully."""
        record.complete(extra)
        with self._lock:
            self._running.pop(record.op_id, None)
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        logger.info(record._log_line("END"))

    def fail(
        self,
        record: OperationRecord,
        error: str,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Mark an operation as failed."""
        record.fail(error, extra)
        with self._lock:
            self._running.pop(record.op_id, None)
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        logger.error(record._log_line("FAIL"))

    @contextmanager
    def operation(
        self,
        op_name: str,
        job_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracking an operation.

        Usage:
            with ops_log.operation("inference", job_id="abc"):
                generate_speech(...)
        """
        record = self.start(op_name, job_id, extra)
        try:
            yield record
            self.end(record)
        except Exception as e:
            self.fail(record, str(e))
            raise

    def get_averages(self) -> Dict[str, dict]:
        """Get average durations grouped by operation name."""
        with self._lock:
            by_op: Dict[str, List[float]] = {}
            for r in self._history:
                if r.status == "completed" and r.duration_seconds is not None:
                    by_op.setdefault(r.op_name, []).append(r.duration_seconds)

        result = {}
        for op_name, durations in by_op.items():
            result[op_name] = {
                "avg_seconds": round(sum(durations) / len(durations), 3),
                "min_seconds": round(min(durations), 3),
                "max_seconds": round(max(durations), 3),
                "count": len(durations),
                "total_seconds": round(sum(durations), 3),
            }
        return result

    def get_history(
        self,
        op_name: Optional[str] = None,
        job_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        """Get recent operation history, optionally filtered."""
        with self._lock:
            records = list(self._history)

        if op_name:
            records = [r for r in records if r.op_name == op_name]
        if job_id:
            records = [r for r in records if r.job_id == job_id]

        return [r.to_dict() for r in records[-limit:]]

    def get_running(self) -> List[dict]:
        """Get currently running operations."""
        with self._lock:
            return [r.to_dict() for r in self._running.values()]


# Singleton
ops_log = OpsLogger()
