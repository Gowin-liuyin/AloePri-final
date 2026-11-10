from __future__ import annotations

from pathlib import Path

from src.aloepri._implementation.qwen.qwen_paper_checkpoint import export_stage_j_paper_consistent_candidate
from src.aloepri._implementation.export.standard_square_checkpoint import export_standard_square_checkpoint


def export_qwen_paper_checkpoint(
    export_dir: str | Path,
    *,
    source_dir: str | Path | None = None,
    materialize: bool = False,
    norm_strategy: str = "kappa_fused",
    model_dir: str | Path | None = None,
    seed: int | None = None,
    dtype: str = "float32",
    device: str = "cpu",
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
) -> dict:
    if (model_dir is None) == (source_dir is None):
        raise ValueError("Specify exactly one of model_dir (square engineering variant) or source_dir (existing artifact).")
    if model_dir is not None:
        return export_standard_square_checkpoint(
            export_dir, model_dir=model_dir, expected_model_type="qwen2", seed=seed,
            dtype=dtype, device=device, alpha_e=alpha_e, alpha_h=alpha_h,
        )
    return export_stage_j_paper_consistent_candidate(
        export_dir,
        source_dir=source_dir,
        materialize=materialize,
        norm_strategy=norm_strategy,
    )
