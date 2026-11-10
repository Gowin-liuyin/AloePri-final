from __future__ import annotations

from pathlib import Path
from typing import Any

from src.aloepri._implementation.runtime.token_space_runtime import prepare_stage_a_model
from src.aloepri._implementation.export.checkpoint_io import export_stage_i_vllm_checkpoint, load_stage_i_hf_bundle, summarize_token_partitions


def build_token_space_bundle(
    *,
    baseline_model,
    tokenizer: Any,
    seed: int,
) -> dict[str, Any]:
    obfuscated_model, perm_vocab, inv_perm_vocab = prepare_stage_a_model(baseline_model, tokenizer, seed=seed)
    metadata = {
        "variant": "token_space_vocab_permutation",
        "seed": seed,
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=obfuscated_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }
    return {
        "model": obfuscated_model,
        "perm_vocab": perm_vocab,
        "inv_perm_vocab": inv_perm_vocab,
        "metadata": metadata,
    }


def export_token_space_checkpoint(
    export_dir: str | Path,
    *,
    tokenizer: Any,
    baseline_model,
    seed: int,
) -> dict[str, Any]:
    bundle = build_token_space_bundle(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        seed=seed,
    )
    paths = export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=bundle["model"],
        perm_vocab=bundle["perm_vocab"],
        inv_perm_vocab=bundle["inv_perm_vocab"],
        metadata=bundle["metadata"],
    )
    return {
        **bundle,
        **paths,
    }


def load_token_space_checkpoint(
    server_dir: str | Path,
    *,
    client_secret_path: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "auto",
) -> dict[str, Any]:
    return load_stage_i_hf_bundle(
        server_dir,
        client_secret_path=client_secret_path,
        device=device,
        dtype=dtype,
    )
