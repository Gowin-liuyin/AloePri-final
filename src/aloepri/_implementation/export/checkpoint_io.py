from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.key_manager import validate_permutation
from src.model_loader import resolve_torch_dtype, load_causal_model
from src.aloepri._implementation.export.deployable_obfuscation import build_stage_h_deployable_inventory
from src.aloepri._implementation.analysis.expression_audit import build_redesigned_expression_audit
from src.aloepri._implementation.export.export_safety import public_metadata


def _normalize_saved_tokenizer_config(server_dir: Path) -> None:
    tokenizer_config_path = server_dir / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        return
    payload = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    extra_special_tokens = payload.get("extra_special_tokens")
    if isinstance(extra_special_tokens, list):
        payload.pop("extra_special_tokens", None)
        tokenizer_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_stage_i_manifest() -> dict[str, Any]:
    return {
        "format": "stage_i_vllm_v1",
        "server_dir": "server",
        "client_dir": "client",
        "server_files": [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        ],
        "client_files": [
            "client_secret.pt",
        ],
    }


def summarize_token_partitions(
    *,
    tokenizer: Any,
    model_vocab_size: int,
    perm_vocab: torch.Tensor,
) -> dict[str, Any]:
    perm_vocab = torch.as_tensor(perm_vocab, dtype=torch.long).cpu()
    special_ids = sorted(token_id for token_id in tokenizer.all_special_ids if token_id < tokenizer.vocab_size)
    tail_start = int(tokenizer.vocab_size)
    tail_ids = list(range(tail_start, model_vocab_size))
    tail_tensor = torch.arange(tail_start, model_vocab_size, dtype=torch.long)

    special_fixed = all(int(perm_vocab[token_id].item()) == token_id for token_id in special_ids)
    tail_fixed = True
    if tail_ids:
        tail_fixed = bool(torch.equal(perm_vocab[tail_start:model_vocab_size], tail_tensor))

    return {
        "perm_is_valid": validate_permutation(perm_vocab),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "model_vocab_size": int(model_vocab_size),
        "special_token_count": len(special_ids),
        "special_ids_fixed": special_fixed,
        "tail_row_count": max(model_vocab_size - int(tokenizer.vocab_size), 0),
        "tail_rows_fixed": tail_fixed,
    }


def export_stage_i_vllm_checkpoint(
    export_dir: str | Path,
    *,
    tokenizer: Any,
    stage_a_model: Any,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    # 先检查整个交付目录，不能只检查 server/，否则会覆盖先前保留的客户端密钥。
    if export_dir.exists() and (not export_dir.is_dir() or any(export_dir.iterdir())):
        raise ValueError(f"Export directory must be empty: {export_dir}; existing files will not be overwritten.")
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    perm_vocab = torch.as_tensor(perm_vocab, dtype=torch.long).cpu()
    inv_perm_vocab = torch.as_tensor(inv_perm_vocab, dtype=torch.long).cpu()
    if server_dir.exists() and any(server_dir.iterdir()):
        raise ValueError(f"Export server directory is not empty: {server_dir}; choose a new export directory.")
    if not validate_permutation(perm_vocab) or inv_perm_vocab.shape != perm_vocab.shape or not torch.equal(
        inv_perm_vocab[perm_vocab], torch.arange(len(perm_vocab))
    ):
        raise ValueError("Client permutation and inverse are invalid or inconsistent.")
    if len(perm_vocab) != int(stage_a_model.get_input_embeddings().weight.shape[0]):
        raise ValueError("Client permutation size must match the model vocabulary size.")
    server_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(server_dir)
    _normalize_saved_tokenizer_config(server_dir)
    stage_a_model.config.save_pretrained(server_dir)
    try:
        stage_a_model.generation_config.save_pretrained(server_dir)
    except Exception:
        GenerationConfig.from_model_config(stage_a_model.config).save_pretrained(server_dir)
    # 分片限制也控制Windows验收时单次文件映射所需的提交内存。
    stage_a_model.save_pretrained(server_dir, safe_serialization=True,max_shard_size="512MB")

    torch.save(
        {
            "perm_vocab": torch.as_tensor(perm_vocab, dtype=torch.long).cpu(),
            "inv_perm_vocab": torch.as_tensor(inv_perm_vocab, dtype=torch.long).cpu(),
            "metadata": metadata,
        },
        client_dir / "client_secret.pt",
    )
    manifest = build_stage_i_manifest()
    manifest["format"] = "standard_hf_local_v2"
    manifest["server_files"] = sorted(path.name for path in server_dir.iterdir() if path.is_file())
    manifest["requires_custom_runtime"] = bool(metadata.get("requires_custom_runtime", False))
    manifest["task_quality_tested"] = False
    manifest["privacy_tested"] = False
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "stage_i_metadata.json").write_text(
        json.dumps(public_metadata(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "export_dir": export_dir,
        "server_dir": server_dir,
        "client_secret_path": client_dir / "client_secret.pt",
    }


def load_stage_i_hf_bundle(
    server_dir: str | Path,
    *,
    client_secret_path: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "auto",
) -> dict[str, Any]:
    server_dir = Path(server_dir)
    if os.name == "nt":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    tokenizer = AutoTokenizer.from_pretrained(server_dir, trust_remote_code=False)
    model = load_causal_model(server_dir, device=device, dtype=dtype)
    model.eval()

    if client_secret_path is None:
        default_secret = server_dir.parent / "client" / "client_secret.pt"
        if default_secret.exists():
            client_secret_path = default_secret

    secret: dict[str, Any] | None = None
    perm_vocab = None
    inv_perm_vocab = None
    if client_secret_path is not None and Path(client_secret_path).exists():
        secret = torch.load(client_secret_path, map_location="cpu", weights_only=True)
        perm_vocab = torch.as_tensor(secret["perm_vocab"], dtype=torch.long)
        inv_perm_vocab = torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long)
        if not validate_permutation(perm_vocab) or inv_perm_vocab.shape != perm_vocab.shape or not torch.equal(inv_perm_vocab[perm_vocab], torch.arange(len(perm_vocab))):
            raise ValueError("Client permutation and inverse are invalid or inconsistent.")
        if len(perm_vocab) != int(model.get_input_embeddings().weight.shape[0]):
            raise ValueError("Client permutation size does not match the model vocabulary.")

    metadata_path = server_dir.parent / "stage_i_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if secret is not None:
        metadata = {**metadata, **secret.get("metadata", {})}

    return {
        "tokenizer": tokenizer,
        "model": model,
        "perm_vocab": perm_vocab,
        "inv_perm_vocab": inv_perm_vocab,
        "secret": secret,
        "metadata": metadata,
    }


def build_phase2_feasibility_summary() -> dict[str, Any]:
    return {
        "phase": "stage_i_phase2_feasibility",
        "scope": "Assess what can be materialized into a standard HF/vLLM-compatible checkpoint without custom kernels.",
        "components": [
            {
                "component": "Stage A vocab permutation",
                "status": "feasible_now",
                "requires_runtime_custom_logic": False,
                "affects_standard_kernels": False,
                "rationale": "Pure embedding/lm_head row permutation with client-side token/logit remapping.",
            },
            {
                "component": "Embedding/head noise + KeyMat",
                "status": "blocked_by_expanded_hidden_dim",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Current Algorithm-1 KeyMat expands hidden size, so obfuscated embed/head weights no longer match the original HF architecture shapes.",
            },
            {
                "component": "FFN fused path",
                "status": "blocked_by_expanded_hidden_dim",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Fused FFN preserves MLP semantics but its input/output dimensions follow expanded KeyMat hidden states, so standard Qwen MLP shapes no longer match.",
            },
            {
                "component": "Attention staticized path",
                "status": "blocked_by_expanded_hidden_dim",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Stage H reduced runtime structure, but q/k/v/o weights are still expressed on expanded hidden states and cannot be copied into the original Qwen attention parameter shapes.",
            },
            {
                "component": "RMSNorm fused path",
                "status": "not_vllm_ready",
                "requires_runtime_custom_logic": True,
                "affects_standard_kernels": True,
                "rationale": "Metric-matrix style fused norm changes standard RMSNorm kernel semantics and would need rollback or custom kernel support.",
            },
        ],
        "redesigned_stage_i": build_stage_i_deployability_matrix(),
        "redesigned_boundary_audit": build_stage_i_boundary_audit(),
    }


def build_stage_i_deployability_matrix() -> dict[str, Any]:
    inventory = build_stage_h_deployable_inventory()
    return {
        "stage": "I",
        "source_stage": "H",
        "inventory_goal": inventory["goal"],
        "runtime_boundary": {
            "standard_transformer_graph": False,
            "custom_online_operator_required": True,
            "compatible_target_surfaces": [],
            "scope": "Historical expanded research runtime; standard square export is a separate variant.",
        },
        "validated_components": {
            "embedding_head": "standard_weight_rewrite",
            "attention_diversity": "needs_materialization_check",
            "ffn_component_transform": "needs_materialization_check",
            "norm_kappa_correction": "supported_if_fused_offline",
        },
        "legacy_reference": {
            "legacy_stage_i_scope": "stage_a_standard_entry_and_phase2_probe",
            "legacy_report": "docs/阶段I_vLLM复现报告.md",
        },
    }


def build_stage_i_boundary_audit() -> dict[str, Any]:
    expression_audit = build_redesigned_expression_audit()
    return {
        "runtime_graph_is_standard": False,
        "custom_online_operator_required": True,
        "compatible_target_surfaces": [],
        "exported_artifact_has_attention_manifest": bool(expression_audit["stage_j"]["has_component_level_expression_manifest"]),
        "exported_artifact_proves_attention_expression": bool(
            expression_audit["stage_j"]["completion_status"] == "export_visible_complete"
        ),
        "exported_artifact_proves_ffn_expression": bool(
            expression_audit["stage_j"]["completion_status"] == "export_visible_complete"
        ),
        "exported_artifact_proves_norm_expression": bool(
            expression_audit["stage_j"]["completion_status"] == "export_visible_complete"
        ),
        "release_catalog_carries_paper_consistent_lineage": bool(
            expression_audit["stage_k"]["points_to_paper_consistent_stage_j"]
        ),
    }
