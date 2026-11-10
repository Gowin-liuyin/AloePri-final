"""从原始 Qwen2/Llama 生成同宽方阵变体；这不是论文算法 1 的扩维实现。"""
from __future__ import annotations

import json
import gc
from pathlib import Path
import secrets
import math
from typing import Any

from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
from src.aloepri._implementation.export.checkpoint_io import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.aloepri._implementation.export.standard_shape_obfuscation import build_stage_j_square_model
from src.secure_random import secure_export,random_metadata


@secure_export
def export_standard_square_checkpoint(
    export_dir: str | Path, *, model_dir: str | Path, expected_model_type: str,
    seed: int | None = None, dtype: str = "float32", device: str = "cpu",
    alpha_e: float = 0.0, alpha_h: float = 0.0, num_layers: int | None = None,
) -> dict[str, Path]:
    """全层改写后保存普通 HF 权重；完整随机种子只写到客户端秘密文件。"""
    if expected_model_type not in {"llama", "qwen2"}:
        raise ValueError("Only standard Llama and Qwen2 decoder architectures are supported.")
    source, target = Path(model_dir).resolve(), Path(export_dir).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Original model and export directories must not overlap.")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Export directory must be empty; existing files will not be deleted.")
    if any(not math.isfinite(value) or value < 0 for value in (alpha_e, alpha_h)):
        raise ValueError("Noise scales must be finite and nonnegative.")
    actual_seed = secrets.randbits(63) if seed is None else int(seed)
    set_global_seed(actual_seed)
    tokenizer, baseline = load_model_and_tokenizer(str(model_dir), device=device, dtype=dtype)
    if str(baseline.config.model_type) != expected_model_type:
        raise ValueError(f"Expected model_type={expected_model_type!r}, got {baseline.config.model_type!r}.")
    total = int(baseline.config.num_hidden_layers)
    if num_layers is not None and num_layers != total:
        raise ValueError("Standard checkpoint export requires all decoder layers; partial layers require a custom runtime.")
    model, perm, inverse, transform = build_stage_j_square_model(
        baseline, tokenizer, adapted_layers=list(range(total)), seed=actual_seed,
        alpha_e=alpha_e, alpha_h=alpha_h,
        copy_model=False,
    )
    metadata: dict[str, Any] = {
        **random_metadata(),
        "stage": "J", "track": "standard_square_engineering_variant",
        "model_line": expected_model_type, "variant": "square_monomial_full_layer",
        "goal": "standard_hf_checkpoint_with_square_transform",
        "implementation_variant": "square_monomial", "paper_algorithm1_reproduced": False,
        "requires_custom_runtime": False,
        "security_scope": "No paper-level privacy or task-quality claim; evaluate separately.",
        "model_dir": str(Path(model_dir).resolve()), "seed": actual_seed,
        "seed_source": "system_random" if seed is None else "explicit_reproducible",
        "dtype": dtype, "alpha_e": alpha_e, "alpha_h": alpha_h,
        "num_layers": total, "total_layers": total, "adapted_layers": list(range(total)),
        "transform_family": "square_monomial_standard_shape", "transform_dim": transform.dim,
        **summarize_token_partitions(tokenizer=tokenizer, model_vocab_size=model.config.vocab_size, perm_vocab=perm),
    }
    paths = export_stage_i_vllm_checkpoint(
        export_dir, tokenizer=tokenizer, stage_a_model=model, perm_vocab=perm,
        inv_perm_vocab=inverse, metadata=metadata,
    )
    del model,baseline
    gc.collect()
    proof = build_stage_j_standard_weight_proof(paths["server_dir"])
    if not proof["is_standard_weight_export"]:
        raise ValueError(f"Exported checkpoint failed structural validation: {proof}")
    manifest_path = Path(export_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"track": metadata["track"], "goal": metadata["goal"],
                     "implementation_variant": "square_monomial", "paper_algorithm1_reproduced": False,
                     "requires_custom_runtime": False, "standard_weight_proof": proof})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**paths, "manifest_path": manifest_path}
