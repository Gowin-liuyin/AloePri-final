from __future__ import annotations

import torch
from torch import nn

from src.keymat import KeyMatTransform, apply_inverse_keymat_transform, apply_keymat_transform


def _extract_layernorm_affine(
    norm_layer: nn.Module,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = getattr(norm_layer, "weight", None)
    bias = getattr(norm_layer, "bias", None)
    if weight is None:
        weight = torch.ones(hidden_size, dtype=torch.float32)
    else:
        weight = weight.detach().to(torch.float32)
    if bias is None:
        bias = torch.zeros(hidden_size, dtype=torch.float32)
    else:
        bias = bias.detach().to(torch.float32)
    return weight, bias


def _resolve_layernorm_epsilon(norm_layer: nn.Module) -> float:
    epsilon = getattr(norm_layer, "variance_epsilon", None)
    if epsilon is None:
        epsilon = getattr(norm_layer, "eps", None)
    if epsilon is None:
        raise AttributeError("LayerNorm module must define `eps` or `variance_epsilon`.")
    return float(epsilon)


class KeyMatBertLayerNormBridge(nn.Module):
    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer
        self.keymat_transform = keymat_transform
        self.recorder = recorder
        self.record_name = record_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        plain_hidden = apply_inverse_keymat_transform(hidden_states, self.keymat_transform)
        normalized = self.norm_layer(plain_hidden)
        if self.recorder is not None and self.record_name is not None:
            self.recorder.record(self.record_name, normalized)
        return apply_keymat_transform(normalized, self.keymat_transform)


def build_keymat_bert_layernorm_bridge(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatBertLayerNormBridge:
    return KeyMatBertLayerNormBridge(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        recorder=recorder,
        record_name=record_name,
    )


class KeyMatFusedBertLayerNorm(nn.Module):
    """
    WARNING: This fused implementation computes mean/variance in obfuscated space.
    LayerNorm is NOT equivariant to signed permutations.
    The Bridge version (KeyMatBertLayerNormBridge) is the ONLY correct path.
    Use this fused class only for approximate benchmarks.
    """

    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        recorder=None,
        record_name: str | None = None,
    ) -> None:
        super().__init__()
        self.eps = _resolve_layernorm_epsilon(norm_layer)
        self.recorder = recorder
        self.record_name = record_name

        key_matrix = keymat_transform.key.detach().to(torch.float32)
        restore_matrix = keymat_transform.inverse.detach().to(torch.float32)
        norm_weight, norm_bias = _extract_layernorm_affine(norm_layer, keymat_transform.hidden_size)
        affine_matrix = (restore_matrix * norm_weight.unsqueeze(0)) @ key_matrix
        obfuscated_bias = torch.matmul(norm_bias.unsqueeze(0), key_matrix).squeeze(0)

        self.register_buffer("norm_weight", norm_weight)
        self.register_buffer("norm_bias", norm_bias)
        self.register_buffer("affine_matrix", affine_matrix)
        self.register_buffer("obfuscated_bias", obfuscated_bias)
        self.register_buffer("restore_matrix", restore_matrix)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.to(torch.float32)
        mean = hidden_fp32.mean(dim=-1, keepdim=True)
        centered = hidden_fp32 - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.eps)
        output = torch.matmul(
            normalized,
            self.affine_matrix.to(device=hidden_states.device, dtype=torch.float32),
        )
        output = output + self.obfuscated_bias.to(device=hidden_states.device, dtype=torch.float32)
        if self.recorder is not None and self.record_name is not None:
            restored = torch.matmul(
                output,
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(self.record_name, restored)
        return output.to(input_dtype)


def build_keymat_fused_bert_layernorm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatFusedBertLayerNorm:
    return KeyMatFusedBertLayerNorm(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        recorder=recorder,
        record_name=record_name,
    )


def wrap_bert_norm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    recorder=None,
    record_name: str | None = None,
) -> KeyMatBertLayerNormBridge:
    return build_keymat_bert_layernorm_bridge(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        recorder=recorder,
        record_name=record_name,
    )


__all__ = [
    "KeyMatBertLayerNormBridge",
    "KeyMatFusedBertLayerNorm",
    "build_keymat_bert_layernorm_bridge",
    "build_keymat_fused_bert_layernorm",
    "wrap_bert_norm",
]
