from __future__ import annotations

from dataclasses import dataclass, replace
import secrets
import torch
from typing import Dict

from src.keymat import KeyMatTransform
from src.attention_keys import build_attention_complex_config, AttentionComplexConfig
from src.obfuscate_ffn import FFNTransform
from src.aloepri.config import AloePriConfig
from src.key_manager import generate_vocab_permutation, invert_permutation

@dataclass(frozen=True)
class AloePriKeys:
    config: AloePriConfig
    token_perm: torch.Tensor
    inv_token_perm: torch.Tensor
    keymat_transform: KeyMatTransform
    # Layer-specific keys (per-layer)
    attention_configs: Dict[int, AttentionComplexConfig]
    ffn_transforms: Dict[int, FFNTransform]
    # Normalization kappa values
    kappas: Dict[int, Dict[str, float]] # e.g., {layer_idx: {"input_norm": 1.0, "post_attn_norm": 1.0}}
    final_norm_kappa: float
    seed_source: str = "explicit_reproducible"

def build_aloepri_keys(
    config: AloePriConfig,
    tokenizer_vocab_size: int,
    movable_ids: torch.Tensor | None = None,
) -> AloePriKeys:
    seed_source = "system_random" if config.seed is None else "explicit_reproducible"
    if config.seed is None:
        config = replace(config, seed=secrets.randbits(63))
    # 1. Token permutation
    model_vocab_size = config.model_vocab_size or tokenizer_vocab_size
    if model_vocab_size < tokenizer_vocab_size:
        raise ValueError("The model embedding has fewer rows than tokenizer.vocab_size.")
    if movable_ids is None:
        movable_ids = torch.arange(tokenizer_vocab_size, dtype=torch.long)
    token_perm = generate_vocab_permutation(model_vocab_size, config.seed, movable_ids=movable_ids)
    inv_token_perm = invert_permutation(token_perm)

    # 2. KeyMat transform
    keymat_transform = build_keymat_transform_from_config(config)

    # 3. Layer-specific keys
    attention_configs = {}

    num_groups = config.num_attention_heads // config.num_key_value_heads

    for layer_idx in config.adapted_layers:
        # Attention keys
        attn_cfg = build_attention_complex_config(
            profile=config.attention_profile,
            head_dim=config.head_dim,
            num_kv_heads=config.num_key_value_heads,
            num_groups=num_groups,
            seed=config.seed + 1000 + layer_idx,
            qk_scale_range=config.qk_scale_range,
            beta=config.beta,
            gamma=config.gamma,
            rope_base=config.rope_theta,
        )
        attention_configs[layer_idx] = attn_cfg

        # FFN widths and normalization weights come from the actual model.
        # AloePriEngine fills the corresponding dictionaries when constructing it.

    return AloePriKeys(
        config=config,
        token_perm=token_perm,
        inv_token_perm=inv_token_perm,
        keymat_transform=keymat_transform,
        attention_configs=attention_configs,
        ffn_transforms={}, # Filled later
        kappas={},         # Filled later
        final_norm_kappa=1.0, # Filled later
        seed_source=seed_source,
    )

def build_keymat_transform_from_config(config: AloePriConfig) -> KeyMatTransform:
    from src.keymat import build_keymat_transform
    return build_keymat_transform(
        d=config.hidden_size,
        h=config.expansion_size,
        lam=config.lam,
        init_seed=config.seed + 100,
        dtype=getattr(torch, config.dtype)
    )
