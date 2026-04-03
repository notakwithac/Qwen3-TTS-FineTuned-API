# coding=utf-8
"""
Resource metrics collector for CPU, GPU, RAM, and VRAM.
Uses psutil and nvidia-ml-py (pynvml).
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psutil

# Optional GPU support
try:
    from pynvml import (
        nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName, nvmlShutdown, NVMLError
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

logger = logging.getLogger("metrics")

class MetricsCollector:
    def __init__(self, storage_path: str = "metrics/resource_metrics.jsonl", interval: int = 60):
        self.storage_path = storage_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # Initialize NVML if available
        self.nvml_initialized = False
        if HAS_GPU:
            try:
                nvmlInit()
                self.nvml_initialized = True
                logger.info("NVML initialized successfully. GPU monitoring enabled.")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}. GPU monitoring disabled.")

    def collect_now(self) -> Dict[str, Any]:
        """Manually trigger a collection and return the snapshot."""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=None),
                "count": psutil.cpu_count(),
                "load": os.getloadavg() if hasattr(os, "getloadavg") else None
            },
            "ram": {
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "percent": psutil.virtual_memory().percent
            },
            "gpus": []
        }
        
        if self.nvml_initialized:
            try:
                device_count = nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    name = nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                        
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    
                    metrics["gpus"].append({
                        "index": i,
                        "name": name,
                        "utilization_percent": util.gpu,
                        "memory_utilization_percent": util.memory,
                        "vram_total_gb": round(mem.total / (1024**3), 2),
                        "vram_used_gb": round(mem.used / (1024**3), 2),
                        "vram_percent": round((mem.used / mem.total) * 100, 2)
                    })
            except Exception as e:
                logger.error(f"Error collecting GPU metrics: {e}")
        
        with self._lock:
            self._latest_metrics = metrics
            
        return metrics

    def _save_to_file(self, metrics: Dict[str, Any]):
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Failed to save metrics to {self.storage_path}: {e}")

    def _loop(self):
        # Initial call to cpu_percent to initialize
        psutil.cpu_percent(interval=None)
        
        while not self._stop_event.is_set():
            metrics = self.collect_now()
            self._save_to_file(metrics)
            
            # Sleep in small chunks to remain responsive to stop event
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"Metrics collection started (interval={self.interval}s, file={self.storage_path})")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.nvml_initialized:
            try:
                nvmlShutdown()
            except:
                pass
        logger.info("Metrics collection stopped.")

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return self._latest_metrics

    def get_history(self, limit: int = 100, start_ts: Optional[str] = None, end_ts: Optional[str] = None) -> List[Dict[str, Any]]:
        history = []
        if not os.path.exists(self.storage_path):
            return history
            
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                try:
                    record = json.loads(line.strip())
                    ts = record.get("timestamp")
                    
                    if start_ts and ts < start_ts:
                        continue
                    if end_ts and ts > end_ts:
                        continue
                        
                    history.append(record)
                except:
                    continue
                    
            if limit:
                history = history[-limit:]
        except Exception as e:
            logger.error(f"Error reading metrics history: {e}")
            
        return history

# Singleton
metrics_collector = MetricsCollector()
