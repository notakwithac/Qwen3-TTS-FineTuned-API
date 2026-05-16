"""Null-space constrained projection-weight edits."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NullspaceEdit:
    weight_name: str
    delta: torch.Tensor
    residual_preservation_norm: float


def preferred_projection_weight(layer_name: str, state_dict_keys: set[str]) -> str:
    down = f"{layer_name}.mlp.down_proj.weight"
    if down in state_dict_keys:
        return down
    fallback = f"{layer_name}.self_attn.o_proj.weight"
    if fallback in state_dict_keys:
        return fallback
    raise KeyError(f"no supported editable projection weight found for {layer_name}")


def validate_edit_weight_name(name: str) -> None:
    if not name.startswith("talker.model.layers."):
        raise ValueError(f"{name} is outside talker.model.layers")
    if not (name.endswith(".mlp.down_proj.weight") or name.endswith(".self_attn.o_proj.weight")):
        raise ValueError(f"{name} is not an allowed SonoEdit v1 projection weight")


def nullspace_projector(preservation_keys: torch.Tensor) -> torch.Tensor:
    keys = preservation_keys.float()
    if keys.ndim != 2:
        raise ValueError("preservation_keys must have shape [examples, hidden_dim]")
    hidden = keys.shape[1]
    if keys.shape[0] == 0:
        return torch.eye(hidden, dtype=keys.dtype, device=keys.device)
    return torch.eye(hidden, dtype=keys.dtype, device=keys.device) - keys.T @ torch.linalg.pinv(keys @ keys.T) @ keys


def compute_constrained_update(
    target_key: torch.Tensor,
    desired_delta: torch.Tensor,
    preservation_keys: torch.Tensor,
    *,
    weight_name: str = "talker.model.layers.0.mlp.down_proj.weight",
    eps: float = 1e-8,
) -> NullspaceEdit:
    validate_edit_weight_name(weight_name)
    key = target_key.float().reshape(-1)
    delta_out = desired_delta.float().reshape(-1)
    projector = nullspace_projector(preservation_keys.to(device=key.device))
    projected_key = projector @ key
    denom = torch.dot(projected_key, key)
    if torch.abs(denom) < eps:
        raise ValueError("target key lies in preservation span; constrained edit is underdetermined")
    delta = delta_out[:, None] @ (projected_key[None, :] / denom)
    residual = delta @ preservation_keys.to(device=delta.device, dtype=delta.dtype).T
    return NullspaceEdit(
        weight_name=weight_name,
        delta=delta,
        residual_preservation_norm=float(torch.linalg.norm(residual).detach().cpu()),
    )


def apply_weight_delta(module: nn.Module, edit: NullspaceEdit, *, alpha: float = 1.0) -> None:
    validate_edit_weight_name(edit.weight_name)
    params = dict(module.named_parameters())
    if edit.weight_name not in params:
        raise KeyError(f"parameter not found: {edit.weight_name}")
    weight = params[edit.weight_name]
    if tuple(weight.shape) != tuple(edit.delta.shape):
        raise ValueError(f"delta shape {tuple(edit.delta.shape)} does not match {edit.weight_name} {tuple(weight.shape)}")
    with torch.no_grad():
        weight.add_(edit.delta.to(device=weight.device, dtype=weight.dtype) * alpha)

