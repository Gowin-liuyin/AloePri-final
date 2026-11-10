from __future__ import annotations

import torch
from torch import nn

from src.hidden_keys import (
    HiddenTransform,
    build_hidden_transform,
    generate_hidden_permutation,
    generate_hidden_scaling,
)
from src.transforms import apply_hidden_transform, apply_inverse_hidden_transform

FFNTransform = HiddenTransform
generate_ffn_permutation = generate_hidden_permutation
generate_ffn_scaling = generate_hidden_scaling
build_ffn_transform = build_hidden_transform


def apply_ffn_permutation(hidden: torch.Tensor, ffn_transform: FFNTransform) -> torch.Tensor:
    perm = ffn_transform.perm.to(hidden.device)
    transformed = torch.empty_like(hidden)
    transformed[..., perm] = hidden
    return transformed


def apply_ffn_up_transform(hidden: torch.Tensor, ffn_transform: FFNTransform) -> torch.Tensor:
    permuted = apply_ffn_permutation(hidden, ffn_transform)
    scale = ffn_transform.scale.to(device=hidden.device, dtype=hidden.dtype)
    return permuted * scale


def invert_ffn_product_transform(hidden: torch.Tensor, ffn_transform: FFNTransform) -> torch.Tensor:
    inv_scale = ffn_transform.inv_scale.to(device=hidden.device, dtype=hidden.dtype)
    scaled = hidden * inv_scale
    perm = ffn_transform.perm.to(hidden.device)
    return scaled.index_select(dim=-1, index=perm)


def fuse_ffn_intermediate_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    transform: FFNTransform,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Write the intermediate permutation/scaling into PyTorch rows/columns offline.

    New channel j is old channel inv_perm[j]. Only the up branch is scaled,
    so an arbitrary elementwise gate activation remains valid. The down columns
    undo this same scale. No inverse permutation is needed during inference.
    """
    order = transform.inv_perm.to(gate_weight.device)
    scale = transform.scale.to(device=up_weight.device, dtype=up_weight.dtype)
    gate = gate_weight.index_select(0, order)
    up = up_weight.index_select(0, order.to(up_weight.device)) * scale.unsqueeze(1)
    down = down_weight.index_select(1, order.to(down_weight.device)) / scale.to(down_weight).unsqueeze(0)
    gate_bias = None if gate_bias is None else gate_bias.index_select(0, order.to(gate_bias.device))
    up_bias = None if up_bias is None else up_bias.index_select(0, order.to(up_bias.device)) * scale.to(up_bias)
    return gate, up, down, gate_bias, up_bias


class ObfuscatedQwen2MLP(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        hidden_transform: HiddenTransform,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = mlp_module.gate_proj
        self.up_proj = mlp_module.up_proj
        self.down_proj = mlp_module.down_proj
        self.act_fn = mlp_module.act_fn
        self.hidden_transform = hidden_transform
        self.ffn_transform = ffn_transform
        self.recorder = recorder
        self.record_name = record_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_hidden = apply_inverse_hidden_transform(hidden_states, self.hidden_transform)
        gate_base = self.gate_proj(base_hidden)
        up_base = self.up_proj(base_hidden)

        gate_mid = apply_ffn_permutation(gate_base, self.ffn_transform)
        up_mid = apply_ffn_up_transform(up_base, self.ffn_transform)
        product_mid = self.act_fn(gate_mid) * up_mid
        product_base = invert_ffn_product_transform(product_mid, self.ffn_transform)
        down_base = self.down_proj(product_base)
        output = apply_hidden_transform(down_base, self.hidden_transform)
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, output)
        return output


def obfuscate_ffn_block(
    mlp_module: nn.Module,
    hidden_transform: HiddenTransform,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> ObfuscatedQwen2MLP:
    return ObfuscatedQwen2MLP(
        mlp_module=mlp_module,
        hidden_transform=hidden_transform,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )
