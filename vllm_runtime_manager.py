"""Lazy vLLM process management for GPU-exclusive auxiliary models."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
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

    process: subprocess.Popen | None = None
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
            if service.api_key:
                env["VLLM_API_KEY"] = service.api_key
            logger.info("Starting managed vLLM service %s: %s", name, " ".join(command))
            service.process = subprocess.Popen(command, env=env)
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

    def _wait_until_ready(self, service: ManagedVllmService) -> None:
        deadline = time.time() + service.startup_timeout_seconds
        headers = {"Authorization": f"Bearer {service.api_key}"} if service.api_key else {}
        last_error = ""
        while time.time() < deadline:
            if service.process is not None and service.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM service {service.name} exited during startup with code {service.process.returncode}"
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
        raise TimeoutError(f"vLLM service {service.name} did not become ready: {last_error}")

    def _stop_locked(self, service: ManagedVllmService) -> None:
        process = service.process
        if process is None:
            return
        if process.poll() is None:
            logger.info("Stopping managed vLLM service %s pid=%s", service.name, process.pid)
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Killing managed vLLM service %s pid=%s", service.name, process.pid)
                process.kill()
                process.wait(timeout=30)
        service.process = None
