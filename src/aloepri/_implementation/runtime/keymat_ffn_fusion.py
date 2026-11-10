from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.hidden_keys import build_identity_hidden_transform
from src.keymat import KeyMatTransform, apply_inverse_keymat_transform, sample_compatible_inverse
from src.obfuscate_ffn import (
    FFNTransform,
    obfuscate_ffn_block,
    fuse_ffn_intermediate_weights,
)


class KeyMatFFNBridgeNormFused(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.keymat_transform = keymat_transform
        self.register_buffer("input_norm_weight", input_norm_weight.detach().to(device="cpu", dtype=torch.float32), persistent=False)
        self.register_buffer("key_matrix", keymat_transform.key.detach().to(device="cpu", dtype=torch.float32), persistent=False)
        self.inner = obfuscate_ffn_block(
            mlp_module=mlp_module,
            hidden_transform=build_identity_hidden_transform(keymat_transform.hidden_size),
            ffn_transform=ffn_transform,
            recorder=recorder,
            record_name=record_name,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        base_hidden = base_hidden * self.input_norm_weight.to(device=hidden_states.device, dtype=base_hidden.dtype)
        mlp_base = self.inner(base_hidden)
        return torch.matmul(mlp_base, self.key_matrix.to(device=hidden_states.device, dtype=mlp_base.dtype))


class KeyMatFusedQwen2MLP(nn.Module):
    def __init__(
        self,
        mlp_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.act_fn = mlp_module.act_fn
        self.ffn_transform = ffn_transform
        self.recorder = recorder
        self.record_name = record_name

        q = keymat_transform.inverse.detach().to(device="cpu", dtype=torch.float32)
        p = keymat_transform.key.detach().to(device="cpu", dtype=torch.float32)
        norm_weight = input_norm_weight.detach().to(device="cpu", dtype=torch.float32)
        gate_inverse = sample_compatible_inverse(keymat_transform, seed + 1).to(q)
        up_inverse = sample_compatible_inverse(keymat_transform, seed + 2).to(q)
        gate_bridge = gate_inverse * norm_weight.unsqueeze(0)
        up_bridge = up_inverse * norm_weight.unsqueeze(0)

        gate_weight = mlp_module.gate_proj.weight.detach().to(device="cpu", dtype=torch.float32) @ gate_bridge.T
        up_weight = mlp_module.up_proj.weight.detach().to(device="cpu", dtype=torch.float32) @ up_bridge.T
        down_weight = p.T @ mlp_module.down_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        gate_bias = None if mlp_module.gate_proj.bias is None else mlp_module.gate_proj.bias.detach().to(device="cpu", dtype=torch.float32)
        up_bias = None if mlp_module.up_proj.bias is None else mlp_module.up_proj.bias.detach().to(device="cpu", dtype=torch.float32)
        gate_weight, up_weight, down_weight, gate_bias, up_bias = fuse_ffn_intermediate_weights(
            gate_weight, up_weight, down_weight, ffn_transform, gate_bias, up_bias,
        )

        self.register_buffer("gate_weight", gate_weight, persistent=False)
        self.register_buffer("up_weight", up_weight, persistent=False)
        self.register_buffer("down_weight", down_weight, persistent=False)
        self.register_buffer("restore_matrix", q, persistent=False)
        self.register_buffer("gate_input_inverse", gate_inverse, persistent=False)
        self.register_buffer("up_input_inverse", up_inverse, persistent=False)

        if mlp_module.gate_proj.bias is not None:
            self.register_buffer("gate_bias", gate_bias, persistent=False)
        else:
            self.gate_bias = None
        if mlp_module.up_proj.bias is not None:
            self.register_buffer("up_bias", up_bias, persistent=False)
        else:
            self.up_bias = None
        if mlp_module.down_proj.bias is not None:
            down_bias = torch.matmul(
                mlp_module.down_proj.bias.detach().to(device="cpu", dtype=torch.float32),
                p,
            )
            self.register_buffer("down_bias", down_bias, persistent=False)
        else:
            self.down_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_linear = F.linear(
            hidden_states,
            self.gate_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.gate_bias is None else self.gate_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        up_linear = F.linear(
            hidden_states,
            self.up_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.up_bias is None else self.up_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

        down_input = self.act_fn(gate_linear) * up_linear
        output = F.linear(
            down_input,
            self.down_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.down_bias is None else self.down_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        if self.recorder is not None and self.record_name is not None:
            restored = torch.matmul(
                output.to(torch.float32),
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(self.record_name, restored)
        return output


def build_keymat_ffn_bridge_norm_fused(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFFNBridgeNormFused:
    return KeyMatFFNBridgeNormFused(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )


def build_keymat_fused_ffn(
    mlp_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
    seed: int = 0,
) -> KeyMatFusedQwen2MLP:
    return KeyMatFusedQwen2MLP(
        mlp_module=mlp_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
        seed=seed,
    )
