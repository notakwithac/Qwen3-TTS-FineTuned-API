"""Lazy vLLM process management for GPU-exclusive auxiliary models."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class ManagedVllmService:
    name: str
    model: str
    served_model_name: str
    port: int
    gpu_memory_utilization: str = "0.45"
    max_model_len: str = "8192"
    extra_args: list[str] = field(default_factory=list)
    api_key: str = "EMPTY"
    startup_timeout_seconds: float = 900.0
    log_path: str = ""
    use_hf_token: bool = True

    process: subprocess.Popen | None = None
    log_handle: Any = None
    last_used_at: float = 0.0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class VllmRuntimeManager:
    """Keeps at most one vLLM model process alive on the GPU."""

    def __init__(self, services: dict[str, ManagedVllmService], idle_timeout_seconds: float = 120.0):
        self._services = services
        self._idle_timeout_seconds = max(0.0, float(idle_timeout_seconds))
        self._lock = threading.RLock()

    def get(self, name: str) -> ManagedVllmService:
        return self._services[name]

    def base_url(self, name: str) -> str:
        return self.get(name).base_url

    def stop_all(self, except_name: str | None = None) -> None:
        with self._lock:
            for name, service in self._services.items():
                if name == except_name:
                    continue
                self._stop_locked(service)

    def stop_idle(self) -> None:
        if self._idle_timeout_seconds <= 0:
            return
        now = time.time()
        with self._lock:
            for service in self._services.values():
                if service.is_running() and now - service.last_used_at >= self._idle_timeout_seconds:
                    self._stop_locked(service)

    def ensure_running(self, name: str) -> ManagedVllmService:
        with self._lock:
            service = self._services[name]
            self.stop_all(except_name=name)
            if service.is_running():
                service.last_used_at = time.time()
                return service

            command = [
                "vllm",
                "serve",
                service.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(service.port),
                "--served-model-name",
                service.served_model_name,
                "--gpu-memory-utilization",
                str(service.gpu_memory_utilization),
                "--max-model-len",
                str(service.max_model_len),
                *service.extra_args,
            ]
            env = os.environ.copy()
            if not service.use_hf_token:
                for token_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
                    env.pop(token_key, None)
                env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            elif not env.get("HF_TOKEN") and not env.get("HUGGINGFACE_HUB_TOKEN") and not env.get("HF_HUB_TOKEN"):
                # Avoid failing public model startup because of an expired token
                # persisted in the shared Hugging Face cache volume.
                env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            if service.api_key:
                env["VLLM_API_KEY"] = service.api_key
            logger.info("Starting managed vLLM service %s: %s", name, " ".join(command))
            self._close_log_handle(service)
            log_path = service.log_path or f"/tmp/pathnam-vllm-{service.name}.log"
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            service.log_path = log_path
            service.log_handle = open(log_path, "a", buffering=1)
            service.log_handle.write(f"\n--- starting {name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            service.process = subprocess.Popen(
                command,
                env=env,
                stdout=service.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self._wait_until_ready(service)
            service.last_used_at = time.time()
            return service

    def mark_used(self, name: str) -> None:
        with self._lock:
            self._services[name].last_used_at = time.time()

    def status(self) -> dict[str, Any]:
        self.stop_idle()
        with self._lock:
            return {
                name: {
                    "model": service.model,
                    "served_model_name": service.served_model_name,
                    "base_url": service.base_url,
                    "running": service.is_running(),
                    "pid": service.process.pid if service.process and service.is_running() else None,
                    "last_used_at": service.last_used_at or None,
                }
                for name, service in self._services.items()
            }

    def tail_log(self, name: str, *, max_chars: int = 2000) -> str:
        with self._lock:
            return self._tail_log(self._services[name], max_chars=max_chars)

    def _wait_until_ready(self, service: ManagedVllmService) -> None:
        deadline = time.time() + service.startup_timeout_seconds
        headers = {"Authorization": f"Bearer {service.api_key}"} if service.api_key else {}
        last_error = ""
        while time.time() < deadline:
            if service.process is not None and service.process.poll() is not None:
                log_tail = self._tail_log(service)
                raise RuntimeError(
                    f"vLLM service {service.name} exited during startup with code "
                    f"{service.process.returncode}. Recent log: {log_tail}"
                )
            try:
                response = requests.get(f"{service.base_url}/models", headers=headers, timeout=5)
                if response.ok:
                    return
                last_error = response.text[:500]
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2)
        self._stop_locked(service)
        raise TimeoutError(
            f"vLLM service {service.name} did not become ready: {last_error}. "
            f"Recent log: {self._tail_log(service)}"
        )

    def _stop_locked(self, service: ManagedVllmService) -> None:
        process = service.process
        if process is None:
            return
        if process.poll() is None:
            logger.info("Stopping managed vLLM service %s pid=%s", service.name, process.pid)
            self._terminate_process_group(process)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Killing managed vLLM service %s pid=%s", service.name, process.pid)
                self._kill_process_group(process)
                process.wait(timeout=30)
        service.process = None
        self._close_log_handle(service)

    def _terminate_process_group(self, process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (AttributeError, OSError):
            process.terminate()

    def _kill_process_group(self, process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, OSError):
            process.kill()

    def _close_log_handle(self, service: ManagedVllmService) -> None:
        handle = service.log_handle
        if handle is None:
            return
        try:
            handle.close()
        finally:
            service.log_handle = None

    def _tail_log(self, service: ManagedVllmService, *, max_chars: int = 2000) -> str:
        if service.log_handle is not None:
            service.log_handle.flush()
        log_path = service.log_path
        if not log_path:
            return ""
        try:
            with open(log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_chars))
                return handle.read().decode("utf-8", errors="replace").strip()
        except OSError as exc:
            return f"could not read {log_path}: {exc}"
