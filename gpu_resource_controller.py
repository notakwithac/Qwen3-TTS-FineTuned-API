# coding=utf-8
import threading
import logging
import os
import torch

logger = logging.getLogger(__name__)

class GPUResourceController:
    """Manages GPU resource synchronization between training and inference.
    
    Implements a Readers-Writers lock:
    - Many Inferences (Readers) can run concurrently.
    - Training (Writer) is exclusive both to other trainings and to all inferences.
    """
    
    def __init__(self, allow_concurrent: bool = None):
        # If not explicitly provided, determine default based on VRAM
        if allow_concurrent is None:
            self.allow_concurrent = self._get_smart_default()
        else:
            self.allow_concurrent = allow_concurrent
            
        self._lock = threading.Condition(threading.Lock())
        self._training_active = False
        self._training_requested = False
        self._inference_count = 0
        
        logger.info(f"GPUResourceController initialized. allow_concurrent={self.allow_concurrent}")

    def _get_smart_default(self) -> bool:
        """Default to True (allow concurrent) only on 40GB+ GPUs (A100, H100, A6000)."""
        if not torch.cuda.is_available():
            return True
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # 39GB as threshold to catch 40GB cards
        return total_vram_gb >= 39.0

    def begin_training(self, job_id: str = "?"):
        if self.allow_concurrent:
            return
        with self._lock:
            self._training_requested = True
            logger.info(
                "GPU: Training requested for job %s. Waiting for active inferences to drain... "
                "(active_inferences=%d, training_active=%s)",
                job_id,
                self._inference_count,
                self._training_active,
            )
            while self._inference_count > 0 or self._training_active:
                self._lock.wait()
            self._training_active = True
            self._training_requested = False
            logger.info(f"GPU: Training lock acquired for job {job_id}.")

    def end_training(self, job_id: str = "?"):
        if self.allow_concurrent:
            return
        with self._lock:
            self._training_active = False
            self._lock.notify_all()
            logger.info(f"GPU: Training lock released for job {job_id}.")

    def begin_inference(self, op_name: str = "inference"):
        if self.allow_concurrent:
            return
        with self._lock:
            if self._training_active or self._training_requested:
                logger.info(f"GPU: {op_name} waiting for training to complete...")
            while self._training_active or self._training_requested:
                self._lock.wait()
            self._inference_count += 1

    def end_inference(self):
        if self.allow_concurrent:
            return
        with self._lock:
            self._inference_count -= 1
            if self._inference_count == 0:
                self._lock.notify_all()
