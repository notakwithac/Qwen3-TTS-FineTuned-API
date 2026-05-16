import torch
from torch import nn

from research.sonoedit_qwen3_tts.causal_trace import run_acoustic_causal_trace


class FakeTraceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.talker = nn.Module()
        self.talker.model = nn.Module()
        self.talker.model.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))

    def forward(self, x):
        x = self.talker.model.layers[0](x)
        x = self.talker.model.layers[1](x)
        return self.head(x)


def test_impact_scores_rank_layer_by_probability_recovery():
    model = FakeTraceModel()
    clean = torch.tensor([[0.0, 4.0]])
    corrupted = torch.tensor([[4.0, 0.0]])

    scores = run_acoustic_causal_trace(
        model,
        clean,
        corrupted,
        [1],
        candidate_layers=["talker.model.layers.0", "talker.model.layers.1"],
    )

    assert scores[0].recovery > 0.99
    assert scores[0].layer in {"talker.model.layers.0", "talker.model.layers.1"}
    assert scores[0].restored_probability > scores[0].corrupted_probability

