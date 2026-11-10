"""交付文件的公开元数据白名单；种子和恢复变换只保留在客户端文件。"""
from __future__ import annotations

from typing import Any
import json
from pathlib import Path
import shutil


PUBLIC_METADATA_FIELDS = {
    "stage", "track", "model_line", "variant", "goal", "architecture_family",
    "dtype", "alpha_e", "alpha_h", "num_layers", "total_layers", "adapted_layers",
    "transform_family", "transform_dim", "movable_token_count", "perm_is_valid",
    "tokenizer_vocab_size", "model_vocab_size", "special_token_count", "special_ids_fixed",
    "tail_row_count", "tail_rows_fixed", "requires_custom_runtime", "paper_algorithm1_reproduced",
    "pooler_output_space", "seed_source", "implementation_variant", "security_scope",
    "residual_matrix_construction", "normalization_mode", "kappa_source", "attention_profile",
    "original_hidden_size", "expanded_hidden_size", "attention_head_dim",
    "random_source", "randomness_mode",
    "router_normalization",
}


def public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key in PUBLIC_METADATA_FIELDS}


def require_research_debug_opt_in(allow_unsafe_debug_export: bool) -> None:
    if not allow_unsafe_debug_export:
        raise ValueError(
            "This historical debug format includes recovery information and is unsafe for a server. "
            "For local research only, explicitly pass allow_unsafe_debug_export=True. "
            "It cannot be used as a standard release checkpoint."
        )


def copy_standard_server(source: str | Path, target: str | Path) -> None:
    """只复制推理文件，不把客户端秘密、历史调试配置一起发给服务器。"""
    from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
    source, target = Path(source).resolve(), Path(target)
    if source == target.resolve() or source in target.resolve().parents or target.resolve() in source.parents:
        raise ValueError("Source and target directories must not overlap.")
    proof = build_stage_j_standard_weight_proof(source)
    if not proof["is_standard_weight_export"]:
        raise ValueError(f"Source is not a complete standard checkpoint: {proof}")
    if (source / "obfuscation_config.json").exists() or list(source.rglob("*client_secret*")):
        raise ValueError("Server source contains historical debug configuration or client secrets.")
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"Target is not empty: {target}; choose a new export directory.")
    target.mkdir(parents=True, exist_ok=True)
    names = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
             "vocab.json", "merges.txt", "tokenizer.model", "tokenizer_config.json"}
    index = source / "model.safetensors.index.json"
    if index.exists():
        names.add(index.name)
        names.update(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    else:
        names.add("model.safetensors")
    for name in sorted(names):
        if (source / name).is_file():
            shutil.copy2(source / name, target / name)
