"""创建本地交付包；服务器目录与客户端秘密分开，包整体不能上传服务器。"""
from __future__ import annotations
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from src.key_manager import validate_permutation
from src.aloepri._implementation.export.export_safety import copy_standard_server, public_metadata
from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof

@dataclass(frozen=True)
class StageKProfile:
    name: str
    source_dir: str
    description: str
    recommended_use: str
    correctness_evidence_file: str | None = None

def default_stage_k_profiles() -> list[StageKProfile]:
    return [StageKProfile(name=name, source_dir="artifacts/stage_j_qwen_paper_consistent",
                         description="Same-width square engineering variant; inspect the manifest.",
                         recommended_use="Local evaluation; only its server directory may be uploaded.")
            for name in ("default", "reference")]

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

def _read_correctness_summary(path: str | None) -> dict[str, Any]:
    payload = _load_json(Path(path)) if path else {}
    summary = payload.get("summary", payload)
    fields = {"status", "prompt_count", "correctness_atol", "validation_scope", "task_quality_tested",
              "generated_ids_exact_match_rate", "generated_text_exact_match_rate", "greedy_first_token_match_rate",
              "baseline_has_nan_or_inf", "stage_k_has_nan_or_inf", "max_restored_full_logits_max_abs_error",
              "avg_restored_full_logits_max_abs_error", "avg_restored_full_logits_mean_abs_error"}
    return {key: value for key, value in summary.items() if key in fields}

def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    fields = {"format", "goal", "track", "variant", "implementation_variant", "paper_algorithm1_reproduced",
              "requires_custom_runtime", "research_only", "standard_weight_proof", "buffered_source_of_truth",
              "resolved_source_dir", "bridge_source_layout", "norm_strategy"}
    return {key: value for key, value in manifest.items() if key in fields}

def export_stage_k_release(
    export_dir: str | Path, *, profiles: list[StageKProfile] | None = None,
    materialize: bool = False, recommended_profile: str = "default",
    reference_profile: str = "reference", title: str = "AloePri Local Evaluation Bundle",
) -> dict[str, Any]:
    export_dir = Path(export_dir).resolve()
    active = default_stage_k_profiles() if profiles is None else profiles
    names = [profile.name for profile in active]
    if not active or len(names) != len({name.casefold() for name in names}):
        raise ValueError("Profiles must be nonempty and their names must be unique.")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) for name in names):
        raise ValueError("Profile names may contain only ASCII letters, numbers, '_' and '-'.")
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if any(name.casefold() in reserved for name in names):
        raise ValueError("Profile names must not use Windows reserved device names.")
    if recommended_profile not in names or reference_profile not in names:
        raise ValueError("Recommended and reference profiles must exist in the bundle.")
    if export_dir.exists() and any(export_dir.iterdir()):
        raise ValueError("Release target must be empty; existing files will not be deleted.")
    # 写入前先检查所有来源和目录关系，避免半个包和源文件被覆盖。
    for profile in active:
        source = Path(profile.source_dir).resolve()
        if source == export_dir or source in export_dir.parents or export_dir in source.parents:
            raise ValueError("Release source and output directories must not overlap.")
        if not (source / "server").is_dir() or not (source / "client" / "client_secret.pt").is_file():
            raise FileNotFoundError(f"Source must contain server/ and client/client_secret.pt: {source}")
        model_config = _load_json(source / "server" / "config.json")
        if str(model_config.get("model_type", "")) not in {"qwen2", "llama","qwen2_moe","mixtral"}:
            raise ValueError("Standard release supports Qwen2/Llama/Qwen2-MoE/Mixtral; BERT remains local research.")
        manifest = _load_json(source / "manifest.json")
        if manifest.get("research_only") or manifest.get("requires_custom_runtime"):
            raise ValueError("Research/debug artifacts and custom-runtime BERT artifacts cannot be standard releases.")
        proof = build_stage_j_standard_weight_proof(source / "server")
        if not proof["is_standard_weight_export"]:
            raise ValueError(f"Source has invalid standard checkpoint structure: {proof}")
        secret = torch.load(source / "client" / "client_secret.pt", weights_only=True, map_location="cpu")
        perm = torch.as_tensor(secret["perm_vocab"], dtype=torch.long)
        inverse = torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long)
        vocab_size = int(model_config["vocab_size"])
        if (not validate_permutation(perm) or perm.numel() != vocab_size or inverse.shape != perm.shape
                or not torch.equal(inverse[perm], torch.arange(perm.numel()))):
            raise ValueError("Source client permutation/inverse does not match its model vocabulary.")
    export_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for profile in active:
        source = Path(profile.source_dir).resolve()
        target = export_dir / "profiles" / profile.name
        copy_standard_server(source / "server", target / "server")
        (target / "client").mkdir(parents=True)
        shutil.copy2(source / "client" / "client_secret.pt", target / "client" / "client_secret.pt")
        metadata = public_metadata(_load_json(source / "stage_i_metadata.json"))
        manifest = _public_manifest(_load_json(source / "manifest.json"))
        (target / "stage_i_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({"name": profile.name, "description": profile.description,
                          "recommended_use": profile.recommended_use,
                          "source_dir": str(source), "server_dir": f"profiles/{profile.name}/server",
                          "client_secret": f"profiles/{profile.name}/client/client_secret.pt",
                          "metadata": metadata, "manifest": manifest,
                          "correctness_evidence_file": profile.correctness_evidence_file,
                          "correctness_summary": _read_correctness_summary(profile.correctness_evidence_file),
                          "evidence_is_not_a_task_quality_certificate": True})
    catalog = {"format": "aloepri_local_bundle_v2", "materialized": True,
               "materialize_requested": bool(materialize), "profiles_dir": "profiles", "profiles": summaries,
               "recommended_profile": recommended_profile, "reference_profile": reference_profile,
               "upload_whole_bundle_to_server": False}
    (export_dir / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    contract = {"client_holds": ["client_secret.pt"],
                "server_holds": ["only files inside profiles/<name>/server"],
                "never_upload": ["profiles/<name>/client", "the whole local bundle"],
                "transport": "token IDs; no claim that detokenize/re-tokenize preserves arbitrary mapped IDs",
                "input_mapping_function": "src/transforms.py::map_input_ids",
                "token_id_restore_function": "src/transforms.py::unmap_output_ids"}
    (export_dir / "deployment_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    text = [f"# {title}", "", "This is a LOCAL owner bundle and contains client secrets.",
            "Upload ONLY profiles/<name>/server to the server. Keep profiles/<name>/client on the client.",
            "The same-width square variant is not a full reproduction of paper Algorithm 1.",
            "Structural checks do not establish task quality or privacy.", "",
            "Run scripts/release/infer_bundle.py with --bundle-dir and --profile for a local simulation."]
    (export_dir / "README.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return catalog
