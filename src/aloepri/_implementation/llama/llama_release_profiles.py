from __future__ import annotations

from pathlib import Path

from src.aloepri._implementation.export.release_bundle_impl import StageKProfile, export_stage_k_release
from src.defaults import DEFAULT_LLAMA_CHECKPOINT_DIR, DEFAULT_LLAMA_CORRECTNESS_DIR


DEFAULT_LLAMA_PAPER_CONSISTENT_DIR = DEFAULT_LLAMA_CHECKPOINT_DIR


def default_stage_k_llama_profiles() -> list[StageKProfile]:
    return [
        StageKProfile(
            name="default",
            source_dir=DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
            description="Default Llama-3.2-1B-Instruct square engineering profile.",
            recommended_use="Default delivery entry for the paper-consistent Llama Instruct deployment line.",
            correctness_evidence_file=f"{DEFAULT_LLAMA_CORRECTNESS_DIR}/default.json",
        ),
        StageKProfile(
            name="reference",
            source_dir=DEFAULT_LLAMA_PAPER_CONSISTENT_DIR,
            description="Reference Llama-3.2-1B-Instruct square engineering profile.",
            recommended_use="Audit and evidence entry for the same paper-consistent Llama Instruct deployment line.",
            correctness_evidence_file=f"{DEFAULT_LLAMA_CORRECTNESS_DIR}/reference.json",
        ),
    ]


def export_stage_k_llama_release(
    export_dir: str | Path,
    *,
    materialize: bool = False,
) -> dict:
    return export_stage_k_release(
        export_dir,
        profiles=default_stage_k_llama_profiles(),
        materialize=materialize,
        recommended_profile="default",
        reference_profile="reference",
        title="Llama-3.2-1B-Instruct Local Evaluation Bundle",
    )
