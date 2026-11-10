"""Obfuscated-checkpoint export for Qwen3.5/3.8 text towers.

The paper's export path (``checkpoints.qwen``) only accepts ``model_type=qwen2``. This
entry point mirrors ``export_standard_square_checkpoint`` but loads the language tower
out of a multimodal Qwen3.5/3.8 checkpoint and applies the hybrid-attention variant of
the square-monomial transform implemented in ``qwen3_5_shape_obfuscation``.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.aloepri._implementation.export.checkpoint_io import export_stage_i_vllm_checkpoint
from src.aloepri._implementation.export.qwen3_5_shape_obfuscation import (
    build_qwen3_5_square_model,
    load_qwen3_5_text_model,
)
from src.model_loader import set_global_seed
from src.secure_random import random_metadata, secure_export


@secure_export
def export_qwen3_5_square_checkpoint(
    export_dir: str | Path,
    *,
    model_dir: str | Path,
    seed: int | None = None,
    dtype: str = "bfloat16",
    device: str = "cpu",
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
) -> dict[str, Path]:
    """Write ``server/`` (obfuscated weights) and ``client/`` (permutation) under ``export_dir``."""
    source, target = Path(model_dir).resolve(), Path(export_dir).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Original model and export directories must not overlap.")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Export directory must be empty; existing files will not be deleted.")

    actual_seed = secrets.randbits(63) if seed is None else int(seed)
    set_global_seed(actual_seed)

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    baseline = load_qwen3_5_text_model(source, dtype=getattr(torch, dtype), device=device)
    if model_dir_type(baseline) != "qwen3_5_text":
        raise ValueError(f"expected a qwen3_5 text tower, got model_type={model_dir_type(baseline)!r}")

    model, perm, inverse, transform, layer_kinds = build_qwen3_5_square_model(
        baseline, tokenizer, seed=actual_seed, alpha_e=alpha_e, alpha_h=alpha_h
    )
    total = len(layer_kinds)
    metadata: dict[str, Any] = {
        **random_metadata(),
        "stage": "J",
        "track": "standard_square_engineering_variant",
        "model_line": "qwen3_5",
        "variant": "square_monomial_full_layer",
        "goal": "standard_hf_checkpoint_with_square_transform",
        "implementation_variant": "square_monomial",
        "paper_algorithm1_reproduced": False,
        "requires_custom_runtime": False,
        "security_scope": "No paper-level privacy or task-quality claim; evaluate separately.",
        "model_dir": str(source),
        "seed": actual_seed,
        "seed_source": "system_random" if seed is None else "explicit_reproducible",
        "dtype": dtype,
        "alpha_e": alpha_e,
        "alpha_h": alpha_h,
        "num_layers": total,
        "total_layers": total,
        "adapted_layers": list(range(total)),
        "layer_kinds": {"linear_attention": layer_kinds.count("linear"), "full_attention": layer_kinds.count("full")},
        "transform_family": "square_monomial_standard_shape",
        "transform_dim": transform.dim,
        "architecture_note": (
            "vision tower and MTP head dropped; gated-delta-net blocks keep conv1d, A_log, "
            "dt_bias and the gated RMSNorm untouched because they act in head space"
        ),
    }
    paths = export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=model,
        perm_vocab=perm,
        inv_perm_vocab=inverse,
        metadata=metadata,
    )
    del model, baseline
    import gc

    gc.collect()

    manifest_path = Path(export_dir) / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "track": metadata["track"],
                "goal": metadata["goal"],
                "implementation_variant": "square_monomial",
                "paper_algorithm1_reproduced": False,
                "requires_custom_runtime": False,
                "layer_kinds": metadata["layer_kinds"],
            }
        )
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**paths, "manifest_path": manifest_path}


def model_dir_type(model) -> str:
    return str(getattr(model.config, "model_type", ""))
