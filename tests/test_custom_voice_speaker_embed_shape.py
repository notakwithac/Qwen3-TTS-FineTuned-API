from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_tts.core.models.modeling_qwen3_tts import _reshape_speaker_embed_for_prefix


def test_single_token_speaker_embed_becomes_one_prefix_token():
    speaker_embed = torch.randn(1024)

    reshaped = _reshape_speaker_embed_for_prefix(speaker_embed)

    assert reshaped.shape == (1, 1, 1024)


def test_multi_token_speaker_embed_preserves_token_axis():
    speaker_embed = torch.randn(4, 1024)

    reshaped = _reshape_speaker_embed_for_prefix(speaker_embed)

    assert reshaped.shape == (1, 4, 1024)


def test_prefix_concat_accepts_multi_token_speaker_embed():
    speaker_embed = _reshape_speaker_embed_for_prefix(torch.randn(4, 8))
    prefix_embed = torch.randn(1, 4, 8)
    suffix_embed = torch.randn(1, 2, 8)

    combined = torch.cat([prefix_embed, speaker_embed, suffix_embed], dim=1)

    assert combined.shape == (1, 10, 8)
