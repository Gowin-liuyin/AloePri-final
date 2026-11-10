from __future__ import annotations

from pathlib import Path

from src.aloepri._implementation.llama.llama_paper_checkpoint import (
    DEFAULT_LLAMA_INSTRUCT_MODEL_DIR,
    export_stage_j_llama_paper_consistent_checkpoint,
)


def export_llama_paper_checkpoint(
    export_dir: str | Path,
    *,
    model_dir: str | Path = DEFAULT_LLAMA_INSTRUCT_MODEL_DIR,
    seed: int | None = None,
    dtype: str = "bfloat16",
    device: str = "cpu",
    alpha_e: float = 0.0,
    alpha_h: float = 0.0,
    num_layers: int | None = None,
) -> dict:
    return export_stage_j_llama_paper_consistent_checkpoint(
        export_dir,
        model_dir=model_dir,
        seed=seed,
        dtype=dtype,
        device=device,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        num_layers=num_layers,
    )
