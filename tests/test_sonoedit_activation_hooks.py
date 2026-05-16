import torch
from torch import nn

from research.sonoedit_qwen3_tts.activations import ActivationCapture, validate_talker_layer_name


class FakeTalkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.talker = nn.Module()
        self.talker.model = nn.Module()
        self.talker.model.layers = nn.ModuleList([nn.Linear(3, 3), nn.ReLU()])
        self.speaker_encoder = nn.Linear(3, 3)

    def forward(self, x):
        for layer in self.talker.model.layers:
            x = layer(x)
        return x


def test_capture_inputs_and_outputs_for_named_layers():
    model = FakeTalkerModel()
    x = torch.randn(2, 3)

    with ActivationCapture(model, ["talker.model.layers.0"]) as capture:
        y = model(x)
        assert "talker.model.layers.0" in capture.records.inputs
        assert "talker.model.layers.0" in capture.records.outputs
        assert capture.output_tensor("talker.model.layers.0").shape == y.shape


def test_hooks_remove_cleanly_and_release_tensors():
    model = FakeTalkerModel()
    layer = model.talker.model.layers[0]

    with ActivationCapture(model, ["talker.model.layers.0"]) as capture:
        model(torch.randn(1, 3))
        assert len(layer._forward_hooks) == 1

    assert len(layer._forward_hooks) == 0
    assert capture.records.inputs == {}
    assert capture.records.outputs == {}


def test_rejects_non_talker_or_forbidden_layers():
    for name in ["speaker_encoder", "speech_tokenizer", "talker.code_predictor", "model.layers.0"]:
        try:
            validate_talker_layer_name(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {name} to be rejected")

