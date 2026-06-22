from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def test_builds_custom_voice_generation_stats_log_fields(monkeypatch):
    model = object.__new__(Qwen3TTSModel)

    logged = []

    def _capture(message, *args):
        logged.append(message % args)

    monkeypatch.setattr(
        "qwen_tts.inference.qwen3_tts_model.logger.info",
        _capture,
    )

    model._log_custom_voice_generation_stats(
        texts=["A" * 325],
        speakers=["narrator"],
        languages=["English"],
        instructs=[""],
        talker_codes_list=[torch.zeros((1300, 2), dtype=torch.int64)],
        gen_kwargs={
            "max_new_tokens": 1300,
            "do_sample": False,
            "subtalker_dosample": False,
        },
    )

    assert len(logged) == 1
    assert "actual_new_tokens=1300" in logged[0]
    assert "effective_max_new_tokens=1300" in logged[0]
    assert "cap_reached=True" in logged[0]
    assert "do_sample=False" in logged[0]
    assert "subtalker_dosample=False" in logged[0]
    assert "text_chars=325" in logged[0]
