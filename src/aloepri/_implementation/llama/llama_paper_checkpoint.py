"""Llama 同宽方阵工程变体的兼容入口；未实现论文算法 1 的扩维方案。"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from src.aloepri._implementation.export.standard_square_checkpoint import export_standard_square_checkpoint
from src.defaults import DEFAULT_LLAMA_MODEL_DIR, DEFAULT_LLAMA_CHECKPOINT_DIR, DEFAULT_LLAMA_BUNDLE_DIR

DEFAULT_LLAMA_INSTRUCT_MODEL_DIR = DEFAULT_LLAMA_MODEL_DIR
DEFAULT_LLAMA_PAPER_CONSISTENT_DIR = DEFAULT_LLAMA_CHECKPOINT_DIR

def build_stage_j_llama_paper_consistent_target() -> dict[str, Any]:
    return {"stage": "J", "model_line": "llama_3_2_1b_instruct",
            "goal": "standard_hf_checkpoint_with_square_transform",
            "standard_graph_required": True, "standard_visible_keys_required": True,
            "paper_algorithm1_reproduced": False,
            "canonical_candidate_dir": DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
            "release_surface": DEFAULT_LLAMA_BUNDLE_DIR, "active_profiles": ["default", "reference"]}

def export_stage_j_llama_paper_consistent_checkpoint(
    export_dir: str | Path = DEFAULT_LLAMA_PAPER_CONSISTENT_DIR, *,
    model_dir: str | Path = DEFAULT_LLAMA_INSTRUCT_MODEL_DIR, seed: int | None = None,
    dtype: str = "bfloat16", device: str = "cpu", alpha_e: float = 0.0,
    alpha_h: float = 0.0, num_layers: int | None = None,
) -> dict[str, Path]:
    return export_standard_square_checkpoint(
        export_dir, model_dir=model_dir, expected_model_type="llama", seed=seed,
        dtype=dtype, device=device, alpha_e=alpha_e, alpha_h=alpha_h, num_layers=num_layers,
    )

export_llama_paper_checkpoint = export_stage_j_llama_paper_consistent_checkpoint
