from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

from src.attention_keys import AttentionComplexConfig, build_attention_complex_config
from src.gqa_layout import GQALayout
from src.keymat import KeyMatTransform, sample_compatible_inverse
from src.aloepri._implementation.runtime.token_space_runtime import TraceRecorder


def _block_diag_repeat(matrix: torch.Tensor, repeats: int) -> torch.Tensor:
    return torch.block_diag(*[matrix for _ in range(repeats)])


def _sample_value_matrices(head_dim: int, num_heads: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw Gaussian U_vo per KV head, rejecting matrices too ill-conditioned for float32.

    This rejection threshold is an explicit numerical choice of this implementation;
    the paper does not specify how to handle nearly singular Gaussian samples.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    from src.secure_random import random_stream
    stream=random_stream("attention.value_matrices",seed)
    matrices, inverses = [], []
    for _ in range(num_heads):
        for _attempt in range(64):
            shape=(head_dim,head_dim)
            matrix=(stream.normal(shape) if stream is not None else torch.randn(shape,generator=generator,dtype=torch.float64))/head_dim**0.5
            if float(torch.linalg.cond(matrix)) <= 100.0:
                matrices.append(matrix.to(torch.float32))
                inverses.append(torch.linalg.inv(matrix).to(torch.float32))
                break
        else:
            raise RuntimeError("Could not sample a sufficiently conditioned U_vo matrix.")
    return torch.stack(matrices), torch.stack(inverses)


def _inverse_index(order: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), dtype=order.dtype)
    return inv


def _query_head_order(layout: GQALayout, tau_kv: torch.Tensor | None, tau_group: torch.Tensor | None) -> torch.Tensor:
    heads = torch.arange(layout.num_query_heads, dtype=torch.long).view(1, layout.num_query_heads, 1, 1)
    grouped = layout.reshape_query_groups(heads)
    grouped = layout.permute_query_groups(grouped, tau_kv, tau_group)
    merged = layout.merge_query_groups(grouped)
    return merged.view(-1).to(torch.long)


def _kv_head_order(layout: GQALayout, tau_kv: torch.Tensor | None) -> torch.Tensor:
    heads = torch.arange(layout.num_kv_heads, dtype=torch.long).view(1, layout.num_kv_heads, 1, 1)
    if tau_kv is None:
        return heads.view(-1).to(torch.long)
    return layout.permute_kv_heads(heads, tau_kv).view(-1).to(torch.long)


def _feature_order_from_head_order(head_order: torch.Tensor, head_dim: int) -> torch.Tensor:
    blocks = [
        torch.arange(int(head_idx) * head_dim, (int(head_idx) + 1) * head_dim, dtype=torch.long)
        for head_idx in head_order.tolist()
    ]
    return torch.cat(blocks, dim=0)


def _apply_output_transform(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dense_transform: torch.Tensor | None = None,
    feature_order: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    transformed_weight = weight
    transformed_bias = bias
    if dense_transform is not None:
        dense_transform = dense_transform.to(device=weight.device, dtype=weight.dtype)
        transformed_weight = dense_transform.T @ transformed_weight
        if transformed_bias is not None:
            transformed_bias = dense_transform.T @ transformed_bias
    if feature_order is not None:
        feature_order = feature_order.to(weight.device)
        transformed_weight = transformed_weight.index_select(0, feature_order)
        if transformed_bias is not None:
            transformed_bias = transformed_bias.index_select(0, feature_order)
    return transformed_weight, transformed_bias


@dataclass(frozen=True)
class StaticAttentionMetadata:
    q_feature_order: torch.Tensor
    q_feature_inv_order: torch.Tensor
    kv_feature_order: torch.Tensor
    kv_feature_inv_order: torch.Tensor
    q_dense: torch.Tensor
    q_dense_inverse: torch.Tensor
    k_dense: torch.Tensor
    k_dense_inverse: torch.Tensor
    v_dense: torch.Tensor
    v_dense_inverse: torch.Tensor


class StaticizedQwen2Attention(nn.Module):
    def __init__(
        self,
        attention_module: nn.Module,
        keymat_transform: KeyMatTransform,
        input_norm_weight: torch.Tensor,
        recorder: TraceRecorder | None,
        layer_idx: int,
        attention_profile: str = "rqk_hqk_block_taukv_taugroup",
        seed: int = 0,
        qk_scale_range: tuple[float, float] = (0.95, 1.05),
        beta: int = 4,
        gamma: float = 1e3,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self.layer_idx = layer_idx
        self.prefix = f"layer_{layer_idx}"
        self.config = attention_module.config
        self.layer_idx_attr = attention_module.layer_idx
        self.head_dim = attention_module.head_dim
        self.scaling = attention_module.scaling
        self.sliding_window = getattr(attention_module,"sliding_window",None)
        self.attention_dropout = attention_module.attention_dropout
        self.is_causal = getattr(attention_module, "is_causal", True)
        if hasattr(attention_module, "num_key_value_groups"):
            self.num_key_value_groups = attention_module.num_key_value_groups

        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        if getattr(attention_module, "q_norm", None) is not None or getattr(attention_module, "k_norm", None) is not None:
            raise ValueError("This Qwen2 attention implementation does not support additional q_norm/k_norm layers.")
        rope_parameters = getattr(self.config, "rope_parameters", None) or getattr(self.config, "rope_scaling", None) or {}
        if any(isinstance(value, dict) for value in rope_parameters.values()):
            raise ValueError("Nested/layer-specific RoPE parameters require a separate verified implementation.")
        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
        partial = rope_parameters.get("partial_rotary_factor", getattr(self.config, "partial_rotary_factor", 1.0))
        if rope_type not in {"default", "llama3"} or partial != 1.0:
            raise ValueError("Only default or Llama 3 full-head RoPE is supported by this implementation.")
        if rope_type == "llama3" and "block" in attention_profile:
            raise ValueError("Llama 3 scaled RoPE requires a profile without frequency-block permutation; use rqk_hqk.")
        rope_base = float(rope_parameters.get("rope_theta", getattr(self.config, "rope_theta", rope_base)))
        if attention_module.q_proj.out_features != self.num_heads * self.head_dim or attention_module.k_proj.out_features != self.num_kv_heads * self.head_dim:
            raise ValueError("Attention projection shapes do not match the configured head counts/dimension.")
        self.layout = GQALayout(self.num_heads, self.num_kv_heads)
        self.attention_config: AttentionComplexConfig = build_attention_complex_config(
            profile=attention_profile,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            num_groups=self.layout.num_groups,
            seed=seed,
            qk_scale_range=qk_scale_range,
            beta=beta,
            gamma=gamma,
            rope_base=rope_base,
        )

        q = keymat_transform.inverse.detach().to(device="cpu", dtype=torch.float32)
        p = keymat_transform.key.detach().to(device="cpu", dtype=torch.float32)
        norm_weight = input_norm_weight.detach().to(device="cpu", dtype=torch.float32)
        # Separate inverse samples cancel the same residual key P. They need not
        # equal one another; all attention outputs still use the shared P.
        query_inverse = sample_compatible_inverse(keymat_transform, seed + 101).to(q)
        key_inverse = sample_compatible_inverse(keymat_transform, seed + 102).to(q)
        value_inverse = sample_compatible_inverse(keymat_transform, seed + 103).to(q)
        base_q_weight = attention_module.q_proj.weight.detach().to(device="cpu", dtype=torch.float32) @ (query_inverse * norm_weight).T
        base_k_weight = attention_module.k_proj.weight.detach().to(device="cpu", dtype=torch.float32) @ (key_inverse * norm_weight).T
        base_v_weight = attention_module.v_proj.weight.detach().to(device="cpu", dtype=torch.float32) @ (value_inverse * norm_weight).T
        base_o_weight = p.T @ attention_module.o_proj.weight.detach().to(device="cpu", dtype=torch.float32)
        value_matrices, value_inverses = _sample_value_matrices(self.head_dim, self.num_kv_heads, seed + 104)
        v_dense = torch.block_diag(*list(value_matrices))
        v_dense_inverse = torch.block_diag(*list(value_inverses))
        # Each query head in a GQA group consumes the corresponding KV head's U_vo.
        output_inverse = torch.block_diag(*[
            value_inverses[head // self.layout.num_groups] for head in range(self.num_heads)
        ])
        base_o_weight = base_o_weight @ output_inverse.T

        q_bias = attention_module.q_proj.bias.detach().to(device="cpu", dtype=torch.float32) if attention_module.q_proj.bias is not None else None
        k_bias = attention_module.k_proj.bias.detach().to(device="cpu", dtype=torch.float32) if attention_module.k_proj.bias is not None else None
        v_bias = attention_module.v_proj.bias.detach().to(device="cpu", dtype=torch.float32) if attention_module.v_proj.bias is not None else None
        o_bias = (
            torch.matmul(attention_module.o_proj.bias.detach().to(device="cpu", dtype=torch.float32), p)
            if attention_module.o_proj.bias is not None
            else None
        )

        q_head_order = _query_head_order(
            self.layout,
            self.attention_config.inter_head.tau_kv,
            self.attention_config.inter_head.tau_group,
        )
        kv_head_order = _kv_head_order(
            self.layout,
            self.attention_config.inter_head.tau_kv,
        )
        q_feature_order = _feature_order_from_head_order(q_head_order, self.head_dim)
        kv_feature_order = _feature_order_from_head_order(kv_head_order, self.head_dim)
        q_feature_inv_order = _inverse_index(q_feature_order)
        kv_feature_inv_order = _inverse_index(kv_feature_order)

        q_dense = _block_diag_repeat(self.attention_config.intra_head.q_matrix, self.num_heads)
        k_dense = _block_diag_repeat(self.attention_config.intra_head.k_matrix, self.num_kv_heads)
        q_dense_inverse = _block_diag_repeat(self.attention_config.intra_head.q_inverse, self.num_heads)
        k_dense_inverse = _block_diag_repeat(self.attention_config.intra_head.k_inverse, self.num_kv_heads)

        q_weight, q_bias = _apply_output_transform(base_q_weight, q_bias, dense_transform=q_dense, feature_order=q_feature_order)
        k_weight, k_bias = _apply_output_transform(base_k_weight, k_bias, dense_transform=k_dense, feature_order=kv_feature_order)
        v_weight, v_bias = _apply_output_transform(base_v_weight, v_bias, dense_transform=v_dense, feature_order=kv_feature_order)
        o_weight = base_o_weight.index_select(1, q_feature_order)

        self.register_buffer("q_weight", q_weight, persistent=False)
        self.register_buffer("k_weight", k_weight, persistent=False)
        self.register_buffer("v_weight", v_weight, persistent=False)
        self.register_buffer("o_weight", o_weight, persistent=False)
        self.register_buffer("restore_matrix", q, persistent=False)
        self.register_buffer("q_feature_inv_order", q_feature_inv_order, persistent=False)
        self.register_buffer("kv_feature_inv_order", kv_feature_inv_order, persistent=False)
        self.register_buffer("q_dense_inverse", q_dense_inverse, persistent=False)
        self.register_buffer("k_dense_inverse", k_dense_inverse, persistent=False)
        self.register_buffer("v_dense_inverse", v_dense_inverse, persistent=False)
        self.register_buffer("query_input_inverse", query_inverse, persistent=False)
        self.register_buffer("key_input_inverse", key_inverse, persistent=False)
        self.register_buffer("value_input_inverse", value_inverse, persistent=False)
        self.static_metadata = StaticAttentionMetadata(
            q_feature_order=q_feature_order,
            q_feature_inv_order=q_feature_inv_order,
            kv_feature_order=kv_feature_order,
            kv_feature_inv_order=kv_feature_inv_order,
            q_dense=q_dense,
            q_dense_inverse=q_dense_inverse,
            k_dense=k_dense,
            k_dense_inverse=k_dense_inverse,
            v_dense=v_dense,
            v_dense_inverse=v_dense_inverse,
        )

        if q_bias is not None:
            self.register_buffer("q_bias", q_bias, persistent=False)
        else:
            self.q_bias = None
        if k_bias is not None:
            self.register_buffer("k_bias", k_bias, persistent=False)
        else:
            self.k_bias = None
        if v_bias is not None:
            self.register_buffer("v_bias", v_bias, persistent=False)
        else:
            self.v_bias = None
        if o_bias is not None:
            self.register_buffer("o_bias", o_bias, persistent=False)
        else:
            self.o_bias = None

    def _restore_qkv_for_compare(
        self,
        query_linear: torch.Tensor,
        key_linear: torch.Tensor,
        value_linear: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_restored = query_linear.index_select(dim=-1, index=self.q_feature_inv_order.to(query_linear.device))
        q_restored = torch.matmul(
            q_restored,
            self.q_dense_inverse.to(device=query_linear.device, dtype=query_linear.dtype),
        )
        k_restored = key_linear.index_select(dim=-1, index=self.kv_feature_inv_order.to(key_linear.device))
        k_restored = torch.matmul(
            k_restored,
            self.k_dense_inverse.to(device=key_linear.device, dtype=key_linear.dtype),
        )
        v_restored = value_linear.index_select(dim=-1, index=self.kv_feature_inv_order.to(value_linear.device))
        v_restored = torch.matmul(v_restored, self.v_dense_inverse.to(device=value_linear.device, dtype=value_linear.dtype))
        return q_restored, k_restored, v_restored

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_linear = F.linear(
            hidden_states,
            self.q_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.q_bias is None else self.q_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        key_linear = F.linear(
            hidden_states,
            self.k_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.k_bias is None else self.k_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        value_linear = F.linear(
            hidden_states,
            self.v_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.v_bias is None else self.v_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

        if self.recorder is not None:
            q_restored, k_restored, v_restored = self._restore_qkv_for_compare(query_linear, key_linear, value_linear)
            self.recorder.record(f"{self.prefix}_q_proj_out", q_restored)
            self.recorder.record(f"{self.prefix}_k_proj_out", k_restored)
            self.recorder.record(f"{self.prefix}_v_proj_out", v_restored)

        query_states = query_linear.view(hidden_shape).transpose(1, 2)
        key_states = key_linear.view(hidden_shape).transpose(1, 2)
        value_states = value_linear.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx_attr, cache_kwargs)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(
            attn_output,
            self.o_weight.to(device=hidden_states.device, dtype=hidden_states.dtype),
            None if self.o_bias is None else self.o_bias.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )
        if self.recorder is not None:
            restored_attn = torch.matmul(
                attn_output.to(torch.float32),
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(f"{self.prefix}_attn_out", restored_attn)
        return attn_output, attn_weights


def build_staticized_attention(
    attention_module: nn.Module,
    keymat_transform: KeyMatTransform,
    input_norm_weight: torch.Tensor,
    recorder: TraceRecorder | None,
    layer_idx: int,
    attention_profile: str,
    seed: int,
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 4,
    gamma: float = 1e3,
    rope_base: float = 10000.0,
) -> StaticizedQwen2Attention:
    return StaticizedQwen2Attention(
        attention_module=attention_module,
        keymat_transform=keymat_transform,
        input_norm_weight=input_norm_weight,
        recorder=recorder,
        layer_idx=layer_idx,
        attention_profile=attention_profile,
        seed=seed,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
        rope_base=rope_base,
    )
