from __future__ import annotations

from pathlib import Path

from src.aloepri._implementation.bert.bert_paper_checkpoint import (
    DEFAULT_BERT_MODEL_DIR,
    build_stage_j_bert_paper_consistent_target,
    export_stage_j_bert_paper_consistent_checkpoint,
)


def export_bert_paper_checkpoint(
    export_dir: str | Path,
    *,
    model_dir: str | Path = DEFAULT_BERT_MODEL_DIR,
    seed: int | None = None,
    dtype: str = "bfloat16",
    device: str = "cpu",
    alpha_e: float = 0.0,
    num_layers: int | None = None,
) -> dict:
    # alpha_e > 0 adds embedding noise for security and breaks 1e-4 correctness checks.
    return export_stage_j_bert_paper_consistent_checkpoint(
        export_dir,
        model_dir=model_dir,
        seed=seed,
        dtype=dtype,
        device=device,
        alpha_e=alpha_e,
        num_layers=num_layers,
    )
