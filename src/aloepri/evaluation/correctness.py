from __future__ import annotations

from pathlib import Path

from src.aloepri._implementation.qwen.qwen_correctness import run_stage_k_release_correctness
from src.aloepri._implementation.llama.llama_correctness import run_stage_k_llama_release_correctness


def run_qwen_correctness(
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    profiles: tuple[str, ...],
    buffered_source_dir: str | None,
    dtype: str,
    device: str,
    seed: int,
    max_new_tokens: int,
    correctness_atol: float = 1e-4,
    baseline_model_dir: str | Path | None = None,
) -> dict:
    if baseline_model_dir is not None:
        return run_stage_k_llama_release_correctness(
            release_dir=bundle_dir, output_dir=output_dir, profiles=profiles,
            baseline_model_dir=baseline_model_dir, dtype=dtype, device=device, seed=seed,
            max_new_tokens=max_new_tokens, correctness_atol=correctness_atol,
        )
    return run_stage_k_release_correctness(
        release_dir=bundle_dir,
        output_dir=output_dir,
        profiles=profiles,
        buffered_source_dir=buffered_source_dir,
        dtype=dtype,
        device=device,
        seed=seed,
        max_new_tokens=max_new_tokens,
        correctness_atol=correctness_atol,
    )


def run_llama_correctness(
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    profiles: tuple[str, ...],
    baseline_model_dir: str | Path,
    dtype: str,
    device: str,
    seed: int,
    max_new_tokens: int,
    correctness_atol: float = 1e-4,
) -> dict:
    return run_stage_k_llama_release_correctness(
        release_dir=bundle_dir,
        output_dir=output_dir,
        profiles=profiles,
        baseline_model_dir=baseline_model_dir,
        dtype=dtype,
        device=device,
        seed=seed,
        max_new_tokens=max_new_tokens,
        correctness_atol=correctness_atol,
    )


def run_bert_correctness(
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    profiles: tuple[str, ...],
    baseline_model_dir: str | Path,
    dtype: str,
    device: str,
    seed: int,
    max_new_tokens: int,
) -> dict:
    from src.aloepri._implementation.bert.bert_correctness import run_stage_k_bert_release_correctness

    del max_new_tokens
    return run_stage_k_bert_release_correctness(
        release_dir=bundle_dir,
        output_dir=output_dir,
        profiles=profiles,
        baseline_model_dir=baseline_model_dir,
        dtype=dtype,
        device=device,
        seed=seed,
    )
