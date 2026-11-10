from __future__ import annotations

import json
import secrets
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.defaults import DEFAULT_SEED
from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids
from src.model_loader import resolve_torch_dtype, set_global_seed
from src.obfuscate_embed_head import permute_embedding_weight
from src.aloepri._implementation.export.checkpoint_io import export_stage_i_vllm_checkpoint, summarize_token_partitions
from src.aloepri._implementation.export.square_transform import build_square_monomial_transform
from src.aloepri._implementation.export.standard_shape_obfuscation import (
    adapt_input_linear_weight_for_square,
    adapt_output_bias_for_square,
    adapt_output_linear_weight_for_square,
    obfuscate_embedding_with_square_transform_stage_a,
)
from src.aloepri.adapters.bert import BertArchitectureAdapter


DEFAULT_BERT_MODEL_DIR = "model/bert-base-uncased"
DEFAULT_BERT_PAPER_CONSISTENT_DIR = "artifacts/stage_j_bert_base_uncased_paper_consistent"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_square_transform_metadata(transform: Any) -> dict[str, Any]:
    return {
        "perm": transform.perm.tolist(),
        "inv_perm": transform.inv_perm.tolist(),
        "signs": transform.signs.tolist(),
        "global_scale": float(transform.global_scale),
    }


def _prepare_stage_a_bert_model(model, tokenizer, seed: int):
    """Vocab-permuted BERT model. BertModel has no lm_head, so only permute embeddings."""
    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    movable_ids = ordinary_token_ids(tokenizer)
    perm_vocab = generate_vocab_permutation(vocab_size=model_vocab_size, seed=seed, movable_ids=movable_ids)
    inv_perm_vocab = invert_permutation(perm_vocab)

    stage_model = deepcopy(model)
    input_embeddings = stage_model.get_input_embeddings()
    permuted_weight = permute_embedding_weight(input_embeddings.weight.detach(), perm_vocab)
    with torch.no_grad():
        input_embeddings.weight.copy_(permuted_weight)

    return stage_model, perm_vocab, inv_perm_vocab


def build_stage_j_bert_paper_consistent_target(export_dir: str | Path) -> dict[str, Any]:
    return {
        "stage": "J",
        "model_line": "bert_base_uncased",
        "goal": "bert_local_research_with_custom_layernorm_bridge",
        "standard_graph_required": False,
        "requires_custom_runtime": True,
        "paper_algorithm1_reproduced": False,
        "standard_visible_keys_required": True,
        "canonical_candidate_dir": str(Path(export_dir)),
        "release_surface": "artifacts/stage_k_bert_release",
        "active_profiles": ["default", "reference"],
    }


def _adapt_square_bert_layer_inplace(layer, transform) -> None:
    """Adapt one BERT encoder layer. LayerNorm weights are left UNCHANGED —
    the client applies KeyMatBertLayerNormBridge at load time using the stored transform."""
    with torch.no_grad():
        for proj_name in ("query", "key", "value"):
            proj = getattr(layer.attention.self, proj_name)
            proj.weight.copy_(adapt_input_linear_weight_for_square(proj.weight.detach(), transform).to(proj.weight.dtype))

        attention_output = layer.attention.output.dense
        attention_output.weight.copy_(
            adapt_output_linear_weight_for_square(attention_output.weight.detach(), transform).to(
                attention_output.weight.dtype
            )
        )
        if attention_output.bias is not None:
            attention_output.bias.copy_(
                adapt_output_bias_for_square(attention_output.bias.detach(), transform).to(attention_output.bias.dtype)
            )

        intermediate = layer.intermediate.dense
        intermediate.weight.copy_(
            adapt_input_linear_weight_for_square(intermediate.weight.detach(), transform).to(intermediate.weight.dtype)
        )

        output_dense = layer.output.dense
        output_dense.weight.copy_(
            adapt_output_linear_weight_for_square(output_dense.weight.detach(), transform).to(output_dense.weight.dtype)
        )
        if output_dense.bias is not None:
            output_dense.bias.copy_(
                adapt_output_bias_for_square(output_dense.bias.detach(), transform).to(output_dense.bias.dtype)
            )


def export_stage_j_bert_paper_consistent_checkpoint(
    export_dir: str | Path = DEFAULT_BERT_PAPER_CONSISTENT_DIR,
    *,
    model_dir: str | Path = DEFAULT_BERT_MODEL_DIR,
    seed: int | None = None,
    dtype: str = "bfloat16",
    device: str = "cpu",
    alpha_e: float = 0.02,
    num_layers: int | None = None,
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    model_dir = Path(model_dir)
    seed_source = "system_random" if seed is None else "explicit_reproducible"
    seed = secrets.randbits(63) if seed is None else int(seed)
    set_global_seed(seed)

    torch_dtype = resolve_torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    baseline_model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True, torch_dtype=torch_dtype)
    baseline_model.eval()
    if device != "cpu":
        baseline_model.to(device)

    adapter = BertArchitectureAdapter.from_model(baseline_model)

    total_layers = adapter.num_hidden_layers
    if num_layers is None:
        num_layers = total_layers
    if num_layers < 1 or num_layers > total_layers:
        raise ValueError(f"num_layers must be between 1 and {total_layers}, got {num_layers}")
    adapted_layers = list(range(num_layers))
    if num_layers < total_layers:
        raise NotImplementedError(
            "Partial-layer BERT adaptation (num_layers < total) is not yet supported. All layers must be adapted."
        )

    stage_model, perm_vocab, inv_perm_vocab = _prepare_stage_a_bert_model(baseline_model, tokenizer, seed=seed)
    transform = build_square_monomial_transform(
        hidden_size=adapter.hidden_size,
        seed=seed + 5000,
        global_scale=1.0,
    )
    movable_ids = ordinary_token_ids(tokenizer)
    square_transform_metadata = _build_square_transform_metadata(transform)

    with torch.no_grad():
        word_embeddings = stage_model.embeddings.word_embeddings
        word_weight = word_embeddings.weight.detach().cpu().to(torch.float32)
        word_obfuscated = obfuscate_embedding_with_square_transform_stage_a(
            word_weight,
            transform,
            alpha_e=alpha_e,
            seed=seed + 101,
            movable_ids=movable_ids,
        )
        word_embeddings.weight.copy_(word_obfuscated.to(word_embeddings.weight.dtype))

        square_key = transform.key(dtype=torch.float32)

        position_embeddings = getattr(stage_model.embeddings, "position_embeddings", None)
        if position_embeddings is not None:
            position_weight = position_embeddings.weight.detach().cpu().to(torch.float32)
            position_embeddings.weight.copy_((position_weight @ square_key).to(position_embeddings.weight.dtype))

        token_type_embeddings = getattr(stage_model.embeddings, "token_type_embeddings", None)
        if token_type_embeddings is not None:
            token_type_weight = token_type_embeddings.weight.detach().cpu().to(torch.float32)
            token_type_embeddings.weight.copy_((token_type_weight @ square_key).to(token_type_embeddings.weight.dtype))

        # Embeddings LayerNorm is left unchanged — client applies bridge at load time

        for layer_idx in adapted_layers:
            _adapt_square_bert_layer_inplace(stage_model.encoder.layer[layer_idx], transform)

        pooler = getattr(stage_model, "pooler", None)
        if pooler is not None and getattr(pooler, "dense", None) is not None:
            pooler.dense.weight.copy_(
                adapt_input_linear_weight_for_square(pooler.dense.weight.detach(), transform).to(
                    pooler.dense.weight.dtype
                )
            )

    has_pooler = getattr(stage_model, "pooler", None) is not None and getattr(stage_model.pooler, "dense", None) is not None
    metadata = {
        "stage": "J",
        "track": "bert_local_research",
        "model_line": "bert_base_uncased",
        "variant": "square_transform_with_layernorm_bridge",
        "goal": "bert_local_research_with_custom_layernorm_bridge",
        "requires_custom_runtime": True,
        "paper_algorithm1_reproduced": False,
        "architecture_family": "bert_encoder",
        "architecture": adapter.describe(),
        "model_dir": str(model_dir),
        "seed": seed,
        "seed_source": seed_source,
        "dtype": dtype,
        "device": device,
        "alpha_e": alpha_e,
        "num_layers": num_layers,
        "total_layers": total_layers,
        "adapted_layers": adapted_layers,
        "transform_family": "square_monomial_standard_shape",
        "transform_dim": int(transform.dim),
        "square_transform": square_transform_metadata,
        "pooler_output_space": "plain" if has_pooler else "absent",
        "movable_token_count": int(movable_ids.numel()),
        **summarize_token_partitions(
            tokenizer=tokenizer,
            model_vocab_size=stage_model.get_input_embeddings().weight.shape[0],
            perm_vocab=perm_vocab,
        ),
    }

    paths = export_stage_i_vllm_checkpoint(
        export_dir,
        tokenizer=tokenizer,
        stage_a_model=stage_model,
        perm_vocab=perm_vocab,
        inv_perm_vocab=inv_perm_vocab,
        metadata=metadata,
    )

    manifest_path = export_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    manifest.update(
        {
            **build_stage_j_bert_paper_consistent_target(export_dir),
            "materialized": True,
            "server_dir": "server",
            "client_dir": "client",
            "model_dir": str(model_dir),
            "track": "bert_local_research",
            "candidate_role": "canonical_stage_j_acceptance_target",
            "transform_family": "square_monomial_standard_shape",
            "export_visible_components": {
                "embeddings": {
                    "word_embeddings_obfuscated": True,
                    "position_embeddings_obfuscated": getattr(stage_model.embeddings, "position_embeddings", None) is not None,
                    "token_type_embeddings_obfuscated": getattr(
                        stage_model.embeddings, "token_type_embeddings", None
                    )
                    is not None,
                    "layernorm_strategy": "client_side_keymat_bridge",
                    "alpha_e": alpha_e,
                },
                "encoder": {
                    "standard_graph_preserved": False,
                    "transform_family": "square_monomial_standard_shape",
                    "adapted_layers_count": len(adapted_layers),
                },
                "pooler": {
                    "present": has_pooler,
                    "adapted": has_pooler,
                    "output_space": "plain" if has_pooler else "absent",
                },
            },
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(export_dir / "paper_consistent_target.json", build_stage_j_bert_paper_consistent_target(export_dir))
    return paths


__all__ = [
    "DEFAULT_BERT_MODEL_DIR",
    "DEFAULT_BERT_PAPER_CONSISTENT_DIR",
    "build_stage_j_bert_paper_consistent_target",
    "export_stage_j_bert_paper_consistent_checkpoint",
]
