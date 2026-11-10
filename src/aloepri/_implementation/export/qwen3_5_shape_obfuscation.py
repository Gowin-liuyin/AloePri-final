"""Square-monomial obfuscation for Qwen3.5/3.8 hybrid decoders.

``standard_shape_obfuscation`` assumes every decoder layer exposes ``q_proj`` /
``k_proj`` / ``v_proj`` / ``o_proj``, which holds for Llama and Qwen2 but not for the
Qwen3.5/3.8 text tower: 48 of its 64 layers are gated-delta-net (``linear_attn``)
blocks whose projections are ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` /
``in_proj_a`` / ``out_proj``.

The monomial transform ``K`` acts on the residual width only, so the rule is
positional, not architectural:

* a linear layer that *reads* the residual stream gets ``K^{-T}`` on its weight
  columns (``adapt_input_linear_weight_for_square``), which leaves its output in the
  original coordinates;
* a linear layer that *writes* back into the residual stream gets ``K`` on its rows
  (``adapt_output_linear_weight_for_square``);
* RMSNorm weights are permuted without signs, because the sign of ``x' = xK`` cancels
  between its input and its output.

Everything a gated-delta-net block computes after its input projections -- ``conv1d``,
``A_log``, ``dt_bias`` and the gated RMSNorm -- lives in head space rather than the
residual stream, so those tensors are not rewritten.
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from src.aloepri._implementation.export.square_transform import (
    SquareMonomialTransform,
    build_square_monomial_transform,
)
from src.aloepri._implementation.export.standard_shape_obfuscation import (
    adapt_input_linear_weight_for_square,
    adapt_output_bias_for_square,
    adapt_output_linear_weight_for_square,
    obfuscate_embedding_with_square_transform_stage_a,
    obfuscate_head_with_square_transform_stage_a,
    permute_rmsnorm_weight_for_square,
)
from src.aloepri._implementation.runtime.token_space_runtime import prepare_stage_a_model

#: Projections that read the residual stream in a gated-delta-net block.
_DELTA_INPUT_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
#: Projections that read the residual stream in a full-attention block.
_ATTENTION_INPUT_PROJECTIONS = ("q_proj", "k_proj", "v_proj")
_SPLIT_PREFIX = "model.language_model."


def is_split_text_checkpoint(model_path) -> bool:
    """True when the text tower is stored under ``model.language_model.*``."""
    import json

    root = Path(model_path)
    config_path = root / "config.json"
    if not config_path.is_file() or not (root / "model.safetensors.index.json").is_file():
        return False
    config = json.loads(config_path.read_text())
    return config.get("model_type") == "qwen3_5" and any(
        "ConditionalGeneration" in arch for arch in config.get("architectures") or ())


def load_qwen3_5_text_model(model_path, *, dtype=torch.bfloat16, device="cpu"):
    """Load only the text tower, dropping the vision tower and the MTP head.

    ``dtype`` defaults to bfloat16 because every step of the square-monomial transform
    is an index selection or a sign flip, which is exact at any width; bf16 halves both
    the resident footprint and the exported checkpoint.
    """
    import json

    from safetensors.torch import load_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    root = Path(model_path)
    raw = json.loads((root / "config.json").read_text())
    config = Qwen3_5TextConfig(**raw["text_config"])
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = Qwen3_5ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous)

    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    known = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    remaining = {n for n, _ in model.named_parameters()}
    for shard in sorted(set(weight_map.values())):
        tensors = load_file(str(root / shard), device="cpu")
        remapped = {}
        for key, value in tensors.items():
            if key.startswith(_SPLIT_PREFIX):
                name = "model." + key[len(_SPLIT_PREFIX):]
            elif key.startswith("lm_head."):
                name = key
            else:
                continue
            if name in known:
                remapped[name] = value.to(dtype)
        remaining.difference_update(remapped)
        model.load_state_dict(remapped, strict=False, assign=True)
        del tensors, remapped
    if remaining:
        raise RuntimeError(f"text backbone is missing {len(remaining)} tensors, e.g. {sorted(remaining)[:3]}")
    return model.to(device=device).eval()


def _adapt_input_projection(module, transform: SquareMonomialTransform) -> None:
    with torch.no_grad():
        module.weight.copy_(
            adapt_input_linear_weight_for_square(module.weight.detach(), transform).to(module.weight.dtype)
        )
        if getattr(module, "bias", None) is not None:
            module.bias.copy_(adapt_input_linear_weight_for_square(module.bias.detach(), transform).to(module.bias.dtype))


def _adapt_output_projection(module, transform: SquareMonomialTransform) -> None:
    with torch.no_grad():
        module.weight.copy_(
            adapt_output_linear_weight_for_square(module.weight.detach(), transform).to(module.weight.dtype)
        )
        if getattr(module, "bias", None) is not None:
            module.bias.copy_(adapt_output_bias_for_square(module.bias.detach(), transform).to(module.bias.dtype))


def adapt_qwen3_5_layer_inplace(layer, transform: SquareMonomialTransform) -> str:
    """Rewrite one decoder layer in place; returns ``"linear"`` or ``"full"``."""
    with torch.no_grad():
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, norm_name, None)
            if norm is not None and getattr(norm, "weight", None) is not None:
                norm.weight.copy_(
                    permute_rmsnorm_weight_for_square(norm.weight.detach(), transform).to(norm.weight.dtype)
                )

    if getattr(layer, "self_attn", None) is not None:
        for name in _ATTENTION_INPUT_PROJECTIONS:
            _adapt_input_projection(getattr(layer.self_attn, name), transform)
        _adapt_output_projection(layer.self_attn.o_proj, transform)
        kind = "full"
    elif getattr(layer, "linear_attn", None) is not None:
        for name in _DELTA_INPUT_PROJECTIONS:
            _adapt_input_projection(getattr(layer.linear_attn, name), transform)
        _adapt_output_projection(layer.linear_attn.out_proj, transform)
        kind = "linear"
    else:
        raise ValueError(f"unrecognised decoder layer {type(layer).__name__}: neither self_attn nor linear_attn")

    for name in ("gate_proj", "up_proj"):
        _adapt_input_projection(getattr(layer.mlp, name), transform)
    _adapt_output_projection(layer.mlp.down_proj, transform)
    return kind


def build_qwen3_5_square_model(
    baseline_model,
    tokenizer,
    *,
    seed: int,
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
):
    """Return ``(obfuscated_model, perm_vocab, inv_perm_vocab, transform, layer_kinds)``."""
    from src.key_manager import ordinary_token_ids

    movable_ids = ordinary_token_ids(tokenizer)
    stage_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(
        baseline_model, tokenizer, seed=seed, copy_model=False
    )
    transform = build_square_monomial_transform(
        hidden_size=baseline_model.config.hidden_size, seed=seed + 5000, global_scale=1.0
    )

    original_head_weight = stage_model.lm_head.weight.detach().cpu().clone()
    original_head_bias = None
    if getattr(stage_model.lm_head, "bias", None) is not None:
        original_head_bias = stage_model.lm_head.bias.detach().cpu().float().clone()

    layer_kinds: list[str] = []
    with torch.no_grad():
        embed_weight = stage_model.model.embed_tokens.weight.detach().cpu()
        embed_obf = obfuscate_embedding_with_square_transform_stage_a(
            embed_weight, transform, alpha_e=alpha_e, seed=seed + 101, movable_ids=movable_ids
        )
        stage_model.model.embed_tokens.weight.copy_(
            embed_obf.to(stage_model.model.embed_tokens.weight.dtype)
        )
        del embed_weight, embed_obf
        for layer in stage_model.model.layers:
            layer_kinds.append(adapt_qwen3_5_layer_inplace(layer, transform))

    with torch.no_grad():
        head_obf = obfuscate_head_with_square_transform_stage_a(
            original_head_weight, transform, alpha_h=alpha_h, seed=seed + 202, movable_ids=movable_ids
        )
        stage_model.lm_head.weight.copy_(head_obf.to(stage_model.lm_head.weight.dtype))
        if original_head_bias is not None:
            stage_model.lm_head.bias.copy_(original_head_bias.to(stage_model.lm_head.bias.dtype))
    del original_head_weight
    if hasattr(stage_model.config, "tie_word_embeddings"):
        stage_model.config.tie_word_embeddings = False

    with torch.no_grad():
        stage_model.model.norm.weight.copy_(
            permute_rmsnorm_weight_for_square(stage_model.model.norm.weight.detach(), transform).to(
                stage_model.model.norm.weight.dtype
            )
        )

    stage_model.eval()
    gc.collect()
    return stage_model, perm_vocab, inv_perm_vocab, transform, layer_kinds
