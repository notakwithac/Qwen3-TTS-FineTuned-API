import torch

from research.sonoedit_qwen3_tts.nullspace_edit import compute_constrained_update, preferred_projection_weight


def test_preservation_keys_define_nullspace_projection():
    target_key = torch.tensor([0.0, 1.0])
    desired_delta = torch.tensor([2.0, -1.0])
    preservation_keys = torch.tensor([[1.0, 0.0]])

    edit = compute_constrained_update(target_key, desired_delta, preservation_keys)

    torch.testing.assert_close(edit.delta @ target_key, desired_delta)
    torch.testing.assert_close(edit.delta @ preservation_keys[0], torch.zeros(2), atol=1e-6, rtol=0)
    assert edit.residual_preservation_norm < 1e-6


def test_computed_update_changes_target_association():
    weight = torch.zeros(2, 2)
    target_key = torch.tensor([0.0, 1.0])
    desired_delta = torch.tensor([0.5, 1.5])
    preservation_keys = torch.tensor([[1.0, 0.0]])

    edit = compute_constrained_update(target_key, desired_delta, preservation_keys)
    changed = (weight + edit.delta) @ target_key

    torch.testing.assert_close(changed, desired_delta)


def test_preferred_projection_weight_uses_down_proj_then_o_proj():
    keys = {"talker.model.layers.3.mlp.down_proj.weight", "talker.model.layers.3.self_attn.o_proj.weight"}
    assert preferred_projection_weight("talker.model.layers.3", keys).endswith("mlp.down_proj.weight")

    keys = {"talker.model.layers.3.self_attn.o_proj.weight"}
    assert preferred_projection_weight("talker.model.layers.3", keys).endswith("self_attn.o_proj.weight")

