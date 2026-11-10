from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
from src.aloepri._implementation.export.export_safety import require_research_debug_opt_in

def build_stage_j_redesign_manifest() -> dict[str, Any]:
    return {
        "stage": "J",
        "goal": "qwen_full_model_deployment_materialization",
        "source_stages": ["H", "I"],
        "bootstrap_source": "artifacts/stage_h_pretrained",
        "legacy_reference": "artifacts/stage_j_full_square",
        "redesign_export_dir": "artifacts/stage_j_qwen_redesign",
    }


def _build_component_expression(source_dir: Path) -> dict[str, Any]:
    config_path = source_dir / "server" / "obfuscation_config.json"
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "embedding_head": {
            "noise_terms_present": all(key in payload for key in ["alpha_e", "alpha_h"]),
            "alpha_e": payload.get("alpha_e"),
            "alpha_h": payload.get("alpha_h"),
        },
        "attention": {
            "attention_profile": payload.get("attention_profile"),
            "profile_present": bool(payload.get("attention_profile")),
        },
        "keymat": {
            "present": all(key in payload for key in ["lambda", "h"]),
            "lambda": payload.get("lambda"),
            "h": payload.get("h"),
        },
        "norm_ffn": {
            "beta": payload.get("beta"),
            "gamma": payload.get("gamma"),
            "adapted_layers_count": len(payload.get("adapted_layers", [])),
        },
    }


def _ensure_link_or_copy(source_dir: Path, target_dir: Path, *, materialize: bool) -> None:
    source, target = source_dir.resolve(), target_dir.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Research artifact source and target directories must not overlap.")
    if target_dir.exists() or target_dir.is_symlink():
        raise ValueError("Research target already exists; existing files will not be deleted.")
    if materialize:
        shutil.copytree(source_dir, target_dir)
    else:
        target_dir.symlink_to(source_dir.resolve(), target_is_directory=True)


def export_stage_j_redesign_checkpoint(
    export_dir: str | Path,
    *,
    source_dir: str | Path = "artifacts/stage_h_pretrained",
    materialize: bool = False,
    allow_unsafe_debug_export: bool = False,
) -> dict[str, Path]:
    require_research_debug_opt_in(allow_unsafe_debug_export)
    export_dir = Path(export_dir)
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Stage-J redesign bootstrap source does not exist: {source_dir}")

    server_source = source_dir / "server"
    client_source = source_dir / "client"
    if not server_source.exists() or not client_source.exists():
        raise FileNotFoundError("Stage-J redesign bootstrap source must contain server/ and client/ directories.")

    export_dir.mkdir(parents=True, exist_ok=True)
    server_dir = export_dir / "server"
    client_dir = export_dir / "client"
    _ensure_link_or_copy(server_source, server_dir, materialize=materialize)
    _ensure_link_or_copy(client_source, client_dir, materialize=materialize)

    manifest = build_stage_j_redesign_manifest()
    manifest["materialized"] = bool(materialize)
    manifest["research_only"] = True
    manifest["contains_recovery_information"] = True
    manifest["resolved_bootstrap_source"] = str(source_dir)
    manifest["server_dir"] = "server"
    manifest["client_dir"] = "client"
    manifest["component_expression"] = _build_component_expression(source_dir)
    manifest["standard_weight_proof"] = build_stage_j_standard_weight_proof(server_dir)
    (export_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "export_dir": export_dir,
        "server_dir": server_dir,
        "client_dir": client_dir,
        "manifest_path": export_dir / "manifest.json",
    }


def build_stage_j_redesign_regression(export_dir: str | Path = "artifacts/stage_j_qwen_redesign") -> dict[str, Any]:
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "stage": "J",
        "artifact_dir": str(export_dir),
        "checks": {
            "manifest_exists": manifest_path.exists(),
            "server_exists": (export_dir / "server").exists(),
            "client_exists": (export_dir / "client").exists(),
            "source_stages_match": manifest.get("source_stages") == ["H", "I"],
            "is_standard_weight_export": manifest.get("standard_weight_proof", {}).get("is_standard_weight_export"),
            "weight_layout": manifest.get("standard_weight_proof", {}).get("layout"),
        },
        "summary": {
            "status": "ready" if manifest_path.exists() and (export_dir / "server").exists() and (export_dir / "client").exists() else "missing_artifact",
        },
    }
