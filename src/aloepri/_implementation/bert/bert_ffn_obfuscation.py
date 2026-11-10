from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.keymat import (
    KeyMatTransform,
    apply_inverse_keymat_transform,
    apply_keymat_transform,
)
from src.obfuscate_ffn import FFNTransform, apply_ffn_up_transform


def _resolve_dense(module: nn.Module) -> nn.Module:
    return module.dense if hasattr(module, "dense") else module


def _resolve_activation(intermediate_module: nn.Module):
    intermediate_act_fn = getattr(intermediate_module, "intermediate_act_fn", None)
    if intermediate_act_fn is not None:
        return intermediate_act_fn

    config = getattr(intermediate_module, "config", None)
    hidden_act = getattr(config, "hidden_act", None)
    if isinstance(hidden_act, str):
        activation_name = hidden_act.lower()
        if activation_name == "gelu":
            return nn.GELU()
        if activation_name == "gelu_new":
            return nn.GELU(approximate="tanh")
        if activation_name == "relu":
            return nn.ReLU()

    return nn.GELU()


def _adapt_output_weight_for_ffn_transform(
    weight: torch.Tensor,
    ffn_transform: FFNTransform,
) -> torch.Tensor:
    inv_perm = ffn_transform.inv_perm.to(device=weight.device)
    inv_scale = ffn_transform.inv_scale.to(device=weight.device, dtype=weight.dtype)
    return weight.index_select(dim=1, index=inv_perm) * inv_scale.unsqueeze(0)


class KeyMatBertFFNBridge(nn.Module):
    def __init__(
        self,
        intermediate_module: nn.Module,
        output_module: nn.Module,
        keymat_transform: KeyMatTransform,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.intermediate_dense = _resolve_dense(intermediate_module)
        self.output_dense = _resolve_dense(output_module)
        self.activation = _resolve_activation(intermediate_module)
        self.keymat_transform = keymat_transform
        self.ffn_transform = ffn_transform
        self.recorder = recorder
        self.record_name = record_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        intermediate_linear = self.intermediate_dense(base_hidden)
        intermediate_activated = self.activation(intermediate_linear)
        intermediate_obfuscated = apply_ffn_up_transform(intermediate_activated, self.ffn_transform)
        output_weight = _adapt_output_weight_for_ffn_transform(
            self.output_dense.weight,
            self.ffn_transform,
        )
        output_plain = F.linear(intermediate_obfuscated, output_weight, self.output_dense.bias)
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, output_plain)
        return apply_keymat_transform(output_plain, self.keymat_transform)


def build_keymat_bert_ffn_bridge(
    intermediate_module: nn.Module,
    output_module: nn.Module,
    keymat_transform: KeyMatTransform,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatBertFFNBridge:
    return KeyMatBertFFNBridge(
        intermediate_module=intermediate_module,
        output_module=output_module,
        keymat_transform=keymat_transform,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )


class KeyMatFusedBertFFN(nn.Module):
    def __init__(
        self,
        intermediate_module: nn.Module,
        output_module: nn.Module,
        keymat_transform: KeyMatTransform,
        ffn_transform: FFNTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.ffn_transform = ffn_transform
        self.activation = _resolve_activation(intermediate_module)
        self.recorder = recorder
        self.record_name = record_name

        intermediate_dense = _resolve_dense(intermediate_module)
        output_dense = _resolve_dense(output_module)

        inverse_matrix = keymat_transform.inverse.detach().to(torch.float32)
        key_matrix = keymat_transform.key.detach().to(torch.float32)
        w1_adapted = intermediate_dense.weight.detach().to(torch.float32)
        w2_adapted = _adapt_output_weight_for_ffn_transform(
            output_dense.weight.detach().to(torch.float32),
            ffn_transform,
        )

        intermediate_weight = w1_adapted @ inverse_matrix.T
        output_weight = key_matrix.T @ w2_adapted

        self.register_buffer("intermediate_weight", intermediate_weight, persistent=False)
        self.register_buffer("output_weight", output_weight, persistent=False)
        self.register_buffer("restore_matrix", inverse_matrix, persistent=False)

        if intermediate_dense.bias is not None:
            self.register_buffer(
                "intermediate_bias",
                intermediate_dense.bias.detach().to(torch.float32),
                persistent=False,
            )
        else:
            self.intermediate_bias = None

        if output_dense.bias is not None:
            output_bias = torch.matmul(output_dense.bias.detach().to(torch.float32), key_matrix)
            self.register_buffer("output_bias", output_bias, persistent=False)
        else:
            self.output_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_linear = F.linear(
            hidden_states,
            self.intermediate_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None
            if self.intermediate_bias is None
            else self.intermediate_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        intermediate_activated = self.activation(intermediate_linear)
        intermediate_obfuscated = apply_ffn_up_transform(intermediate_activated, self.ffn_transform)
        output_obfuscated = F.linear(
            intermediate_obfuscated,
            self.output_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.output_bias is None else self.output_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        if self.recorder is not None and self.record_name is not None:
            restored = torch.matmul(
                output_obfuscated.to(torch.float32),
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(self.record_name, restored)
        return output_obfuscated


def build_keymat_fused_bert_ffn(
    intermediate_module: nn.Module,
    output_module: nn.Module,
    keymat_transform: KeyMatTransform,
    ffn_transform: FFNTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFusedBertFFN:
    return KeyMatFusedBertFFN(
        intermediate_module=intermediate_module,
        output_module=output_module,
        keymat_transform=keymat_transform,
        ffn_transform=ffn_transform,
        recorder=recorder,
        record_name=record_name,
    )
