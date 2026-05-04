# coding=utf-8
import gc
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def safe_cuda_cleanup(reason: str, *, synchronize: bool = False) -> None:
    """Best-effort CUDA allocator cleanup that never fails caller control flow."""
    gc.collect()
    if not torch.cuda.is_available():
        return

    try:
        if synchronize:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("CUDA cleanup skipped after allocator error (%s): %s", reason, exc)

