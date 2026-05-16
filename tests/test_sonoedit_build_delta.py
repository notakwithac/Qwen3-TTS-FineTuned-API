import json

import torch

from research.sonoedit_qwen3_tts.build_delta import (
    ProjectionCapture,
    build_delta_from_captured_tensors,
    build_delta_from_tensor_bundle,
)


WEIGHT_NAME = "talker.model.layers.8.mlp.down_proj.weight"


def _toy_bundle():
    projection_output = torch.zeros(1, 1, 2, requires_grad=True)
    head = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return {
        "current_weight": torch.ones(2, 3),
        "target_codec0": torch.tensor([1]),
        "codec0_logits": projection_output @ head.T,
        "projection_input": torch.tensor([[[0.0, 1.0, 0.0]]]),
        "projection_output": projection_output,
        "preservation_inputs": [torch.tensor([[[1.0, 0.0, 0.0]]])],
    }


def test_build_delta_from_captured_tensors_preserves_nullspace_key():
    bundle = _toy_bundle()
    residual_logits = bundle["projection_output"].reshape(1, 1, 2)

    delta, result = build_delta_from_captured_tensors(
        weight_name=WEIGHT_NAME,
        current_weight=bundle["current_weight"],
        target_codec0=torch.tensor([[1, 0]]),
        codec0_logits=bundle["codec0_logits"],
        projection_input=bundle["projection_input"],
        projection_output=bundle["projection_output"],
        preservation_inputs=bundle["preservation_inputs"],
        residual_logits=residual_logits,
        residual_codebook_weight=0.1,
        max_relative_delta=None,
    )

    assert delta.shape == bundle["current_weight"].shape
    assert result.target_frames == 1
    assert result.preservation_examples == 1
    assert result.residual_codebook_weight == 0.1
    assert torch.linalg.norm(delta).item() > 0
    preservation_key = bundle["preservation_inputs"][0].reshape(-1)
    assert torch.allclose(delta @ preservation_key, torch.zeros(2), atol=1e-6)


def test_build_delta_from_tensor_bundle_writes_delta_and_metadata(tmp_path):
    bundle_path = tmp_path / "bundle.pt"
    output_path = tmp_path / "delta.pt"
    bundle = _toy_bundle()
    bundle["desired_delta"] = torch.tensor([-0.5, 0.5])
    bundle["target_frames"] = 1
    del bundle["codec0_logits"]
    del bundle["projection_output"]
    del bundle["target_codec0"]
    torch.save(bundle, bundle_path)

    result = build_delta_from_tensor_bundle(
        bundle_path,
        output_path,
        weight_name=WEIGHT_NAME,
        max_relative_delta=None,
    )

    saved = torch.load(output_path, map_location="cpu")
    assert set(saved) == {WEIGHT_NAME}
    assert saved[WEIGHT_NAME].shape == (2, 3)
    metadata = json.loads(output_path.with_suffix(output_path.suffix + ".json").read_text(encoding="utf-8"))
    assert metadata["weight_name"] == WEIGHT_NAME
    assert metadata["target_frames"] == result.target_frames


def test_projection_capture_promotes_activation_to_gradient_source():
    module = torch.nn.Linear(3, 2)
    module.requires_grad_(False)

    with torch.enable_grad():
        with ProjectionCapture(module, "") as capture:
            output = module(torch.ones(1, 3))
            loss = output.square().sum()

    assert capture.output is not None
    assert capture.output.requires_grad
    assert torch.autograd.grad(loss, capture.output)[0].shape == output.shape
