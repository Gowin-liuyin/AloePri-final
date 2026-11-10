from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from src.model_loader import load_model_and_tokenizer, set_global_seed
from src.aloepri._implementation.runtime.keymat_bridge_runtime import build_default_stage_f_keymat, calibrate_keymat_kappas
from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
from src.aloepri._implementation.export.tensor_io import read_safetensors_state, safetensors_shapes
from src.aloepri._implementation.export.export_safety import copy_standard_server, require_research_debug_opt_in


def _ensure_link_or_copy(source_dir: Path, target_dir: Path, *, materialize: bool) -> None:
    source, target = source_dir.resolve(), target_dir.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target directories must not overlap.")
    if target_dir.exists() or target_dir.is_symlink():
        raise ValueError("Target already exists; existing files will not be deleted.")
    if materialize:
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)


def build_stage_j_standard_bridge_manifest() -> dict[str, Any]:
    return {
        "stage": "J",
        "track": "standard_visible_bridge",
        "goal": "standard_weight_visible_bridge_for_redesigned_stage_j",
        "buffered_reference": "artifacts/stage_j_qwen_redesign",
        "standard_visible_source": "artifacts/stage_j_qwen_redesign",
        "equivalence_to_buffered_redesign": False,
        "notes": "Bridge artifact: standard-weight-visible, but not yet proven equivalent to the buffered redesign line.",
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_file_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _resolve_norm_strategy(source_server: Path, norm_strategy: str) -> str:
    if norm_strategy != "auto":
        return norm_strategy
    obf_cfg = _load_json(source_server / "obfuscation_config.json")
    family = obf_cfg.get("keymat_family", "algorithm1")
    if family == "diag_friendly":
        return "metric_diag_sqrt"
    return "kappa_fused"


def _is_buffered_stage_state(state: dict[str, torch.Tensor]) -> bool:
    return "buffer::stage_a_model.model.embed_tokens.weight" in state


def _infer_buffered_layer_indices(state: dict[str, torch.Tensor]) -> list[int]:
    prefix = "buffer::stage_a_model.model.layers."
    indices: set[int] = set()
    for key in state:
        if key.startswith(prefix):
            remain = key[len(prefix) :]
            layer_idx = remain.split(".", 1)[0]
            if layer_idx.isdigit():
                indices.add(int(layer_idx))
    return sorted(indices)


def _build_modified_config_for_buffered_source(
    config_payload: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    updated = dict(config_payload)
    hidden_size = int(state["buffer::stage_a_model.model.embed_tokens.weight"].shape[1])
    q_out = int(state["buffer::stage_a_model.model.layers.0.self_attn.q_weight"].shape[0])
    gate_out = int(state["buffer::stage_a_model.model.layers.0.mlp.gate_weight"].shape[0])
    num_attention_heads = int(updated["num_attention_heads"])
    head_dim = q_out // num_attention_heads
    updated["hidden_size"] = hidden_size
    updated["intermediate_size"] = gate_out
    updated["head_dim"] = head_dim
    updated["tie_word_embeddings"] = False
    return updated


def _build_standard_state_from_buffered(
    state: dict[str, torch.Tensor],
    *,
    hidden_size: int,
    norm_strategy: str,
    kappa_values: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    standard: dict[str, torch.Tensor] = {}
    standard["model.embed_tokens.weight"] = state["buffer::stage_a_model.model.embed_tokens.weight"].contiguous()
    standard["lm_head.weight"] = state["buffer::stage_a_model.lm_head.weight"].contiguous()

    layer_indices = _infer_buffered_layer_indices(state)
    ones = torch.ones(hidden_size, dtype=standard["model.embed_tokens.weight"].dtype)
    kappa_values = kappa_values or {"layers": {}, "final": 1.0}

    def norm_weight_from_metric(key: str) -> torch.Tensor:
        if norm_strategy in {"ones", "kappa_fused"} or key not in state:
            return ones.clone()
        if norm_strategy == "metric_diag_sqrt":
            metric = state[key].to(torch.float32)
            diag = torch.diagonal(metric).clamp_min(0.0).sqrt().to(dtype=ones.dtype)
            return diag.contiguous()
        raise ValueError(f"Unsupported norm_strategy: {norm_strategy}")

    def layer_kappa(layer_idx: int, kind: str) -> float:
        return float(kappa_values.get("layers", {}).get(str(layer_idx), {}).get(kind, 1.0))

    for layer_idx in layer_indices:
        prefix = f"buffer::stage_a_model.model.layers.{layer_idx}"
        standard[f"model.layers.{layer_idx}.input_layernorm.weight"] = norm_weight_from_metric(
            f"{prefix}.input_layernorm.metric_matrix"
        )
        standard[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = norm_weight_from_metric(
            f"{prefix}.post_attention_layernorm.metric_matrix"
        )
        input_scale = layer_kappa(layer_idx, "input") if norm_strategy == "kappa_fused" else 1.0
        post_attn_scale = layer_kappa(layer_idx, "post_attn") if norm_strategy == "kappa_fused" else 1.0
        standard[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (state[f"{prefix}.self_attn.q_weight"] * input_scale).contiguous()
        standard[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (state[f"{prefix}.self_attn.k_weight"] * input_scale).contiguous()
        standard[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (state[f"{prefix}.self_attn.v_weight"] * input_scale).contiguous()
        standard[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = state[f"{prefix}.self_attn.o_weight"].contiguous()
        if f"{prefix}.self_attn.q_bias" in state:
            standard[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = state[f"{prefix}.self_attn.q_bias"].contiguous()
        if f"{prefix}.self_attn.k_bias" in state:
            standard[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = state[f"{prefix}.self_attn.k_bias"].contiguous()
        if f"{prefix}.self_attn.v_bias" in state:
            standard[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = state[f"{prefix}.self_attn.v_bias"].contiguous()
        standard[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (state[f"{prefix}.mlp.gate_weight"] * post_attn_scale).contiguous()
        standard[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (state[f"{prefix}.mlp.up_weight"] * post_attn_scale).contiguous()
        standard[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = state[f"{prefix}.mlp.down_weight"].contiguous()

    standard["model.norm.weight"] = norm_weight_from_metric("buffer::stage_a_model.model.norm.metric_matrix")
    if norm_strategy == "kappa_fused":
        final_scale = float(kappa_values.get("final", 1.0))
        standard["lm_head.weight"] = (standard["lm_head.weight"] * final_scale).contiguous()
    return standard


def _load_or_estimate_kappas(source_server: Path, config_payload: dict[str, Any]) -> dict[str, Any]:
    obf_cfg = _load_json(source_server / "obfuscation_config.json")
    if "kappa_overrides" in obf_cfg:
        return obf_cfg["kappa_overrides"]

    required = ["model_dir", "seed", "lambda", "h", "prompts_for_kappa", "adapted_layers"]
    if not all(key in obf_cfg for key in required):
        return {"layers": {}, "final": 1.0}

    set_global_seed(int(obf_cfg["seed"]))
    tokenizer, baseline_model = load_model_and_tokenizer(
        obf_cfg["model_dir"],
        device="cpu",
        dtype=obf_cfg.get("dtype", "float32"),
    )
    keymat_transform = build_default_stage_f_keymat(
        baseline_model,
        lam=float(obf_cfg["lambda"]),
        h=int(obf_cfg["h"]),
        seed=int(obf_cfg["seed"]),
    )
    layer_kappas = calibrate_keymat_kappas(
        baseline_model=baseline_model,
        tokenizer=tokenizer,
        prompts=list(obf_cfg["prompts_for_kappa"]),
        keymat_transform=keymat_transform,
        trace_layers=list(obf_cfg["adapted_layers"]),
    )
    return {
        "layers": {str(idx): {"input": values["input"], "post_attn": values["post_attn"]} for idx, values in layer_kappas.items()},
        "final": 1.0,
    }


def _materialize_buffered_source_to_standard_visible(
    *,
    source_server: Path,
    target_server: Path,
    norm_strategy: str,
) -> None:
    state = read_safetensors_state(source_server)
    config_payload = _load_json(source_server / "config.json")
    modified_config = _build_modified_config_for_buffered_source(config_payload, state)
    kappa_values = _load_or_estimate_kappas(source_server, modified_config) if norm_strategy == "kappa_fused" else None
    standard_state = _build_standard_state_from_buffered(
        state,
        hidden_size=int(modified_config["hidden_size"]),
        norm_strategy=norm_strategy,
        kappa_values=kappa_values,
    )

    target_server.mkdir(parents=True, exist_ok=True)
    save_file(standard_state, str(target_server / "model.safetensors"))
    (target_server / "config.json").write_text(json.dumps(modified_config, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ]:
        _copy_file_if_exists(source_server / filename, target_server / filename)


def export_stage_j_redesign_standard_bridge(
    export_dir: str | Path,
    *,
    source_dir: str | Path = "artifacts/stage_j_qwen_redesign",
    materialize: bool = False,
    norm_strategy: str = "auto",
    allow_unsafe_debug_export: bool = False,
) -> dict[str, Path]:
    export_dir = Path(export_dir)
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Stage-J standard bridge source does not exist: {source_dir}")

    server_source = source_dir / "server"
    client_source = source_dir / "client"
    if not server_source.exists() or not client_source.exists():
        raise FileNotFoundError("Stage-J standard bridge source must contain server/ and client/ directories.")

    export_dir.mkdir(parents=True, exist_ok=True)
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    shapes = safetensors_shapes(server_source)
    is_buffered = _is_buffered_stage_state(shapes)
    resolved_norm_strategy = _resolve_norm_strategy(server_source, norm_strategy)
    if is_buffered:
        require_research_debug_opt_in(allow_unsafe_debug_export)
        if server_dir.exists() and any(server_dir.iterdir()):
            raise ValueError("Choose a new empty research export directory.")
        _materialize_buffered_source_to_standard_visible(
            source_server=server_source,
            target_server=server_dir,
            norm_strategy=resolved_norm_strategy,
        )
    else:
        copy_standard_server(server_source, server_dir)
    if client_dir.exists():
        raise ValueError("Client target already exists; choose a new export directory.")
    shutil.copytree(client_source, client_dir)

    manifest = build_stage_j_standard_bridge_manifest()
    manifest["materialized"] = True
    manifest["resolved_source_dir"] = str(source_dir)
    manifest["server_dir"] = "server"
    manifest["client_dir"] = "client"
    manifest["bridge_source_layout"] = "buffered_stage_style" if is_buffered else "standard_weight_visible"
    manifest["research_only"] = is_buffered
    manifest["paper_algorithm1_reproduced"] = False
    manifest["norm_strategy"] = resolved_norm_strategy
    manifest["standard_weight_proof"] = build_stage_j_standard_weight_proof(server_dir)
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "export_dir": export_dir,
        "server_dir": server_dir,
        "client_dir": client_dir,
        "manifest_path": export_dir / "manifest.json",
    }
