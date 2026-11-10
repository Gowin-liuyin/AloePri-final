from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.model_loader import resolve_torch_dtype, set_global_seed
from src.aloepri._implementation.bert.bert_layernorm_bridge import build_keymat_bert_layernorm_bridge
from src.aloepri._implementation.bert.bert_paper_checkpoint import DEFAULT_BERT_MODEL_DIR
from src.aloepri._implementation.export.square_transform import SquareMonomialTransform
from src.transforms import map_input_ids


DEFAULT_BERT_TEST_SENTENCES = (
    "Hello world.",
    "Private inference needs careful evaluation.",
    "BERT encodes sentences into contextual hidden states.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models should preserve output fidelity.",
)
BERT_CORRECTNESS_ATOL = 1e-4


def _read_catalog_profile(release_dir: str | Path, profile: str) -> dict[str, Any]:
    release_dir = Path(release_dir)
    catalog = json.loads((release_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    if profile not in profile_map:
        raise ValueError(f"Unknown Stage-K BERT profile: {profile}")
    return profile_map[profile]


def resolve_stage_k_bert_profile_paths(release_dir: str | Path, profile: str) -> dict[str, str]:
    release_dir = Path(release_dir)
    selected = _read_catalog_profile(release_dir, profile)
    return {
        "server_dir": str(release_dir / selected["server_dir"]),
        "client_secret": str(release_dir / selected["client_secret"]),
        "correctness_evidence_file": str(selected.get("correctness_evidence_file", "")),
    }


def summarize_bert_sentence_results(items: list[dict[str, Any]]) -> dict[str, float | bool | None]:
    count = max(len(items), 1)
    pooler_items = [item for item in items if item["pooler_output_max_abs_error"] is not None]
    pooler_count = max(len(pooler_items), 1)
    return {
        "sentence_count": len(items),
        "avg_last_hidden_state_max_abs_error": sum(float(item["last_hidden_state_max_abs_error"]) for item in items) / count,
        "avg_last_hidden_state_mean_abs_error": sum(float(item["last_hidden_state_mean_abs_error"]) for item in items) / count,
        "max_last_hidden_state_max_abs_error": max(float(item["last_hidden_state_max_abs_error"]) for item in items),
        "avg_cls_max_abs_error": sum(float(item["cls_max_abs_error"]) for item in items) / count,
        "avg_cls_mean_abs_error": sum(float(item["cls_mean_abs_error"]) for item in items) / count,
        "max_cls_max_abs_error": max(float(item["cls_max_abs_error"]) for item in items),
        "avg_pooler_output_max_abs_error": (
            sum(float(item["pooler_output_max_abs_error"]) for item in pooler_items) / pooler_count
            if pooler_items
            else None
        ),
        "avg_pooler_output_mean_abs_error": (
            sum(float(item["pooler_output_mean_abs_error"]) for item in pooler_items) / pooler_count
            if pooler_items
            else None
        ),
        "max_pooler_output_max_abs_error": (
            max(float(item["pooler_output_max_abs_error"]) for item in pooler_items)
            if pooler_items
            else None
        ),
        "baseline_has_nan_or_inf": any(bool(item["baseline_has_nan_or_inf"]) for item in items),
        "stage_k_has_nan_or_inf": any(bool(item["stage_k_has_nan_or_inf"]) for item in items),
        "restored_has_nan_or_inf": any(bool(item["restored_has_nan_or_inf"]) for item in items),
    }


def _select_bert_test_sentences() -> tuple[str, ...]:
    ascii_prompts = tuple(prompt for prompt in DEFAULT_PROMPTS if prompt.isascii())
    if ascii_prompts:
        return ascii_prompts
    return DEFAULT_BERT_TEST_SENTENCES


def _load_bert_tokenizer(model_dir: str | Path):
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def _load_bert_model(model_dir: str | Path, *, device: str, dtype: str):
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(dtype),
    )
    model.eval()
    if device != "cpu":
        model.to(device)
    return model


@dataclass(frozen=True)
class _SquareKeyMatProxy:
    key: torch.Tensor
    inverse: torch.Tensor
    hidden_size: int = 0


def _load_square_transform(metadata: dict[str, Any]) -> SquareMonomialTransform:
    square_transform = metadata.get("square_transform")
    if not isinstance(square_transform, dict):
        raise ValueError("square_transform metadata is required for Stage-K BERT correctness.")
    return SquareMonomialTransform(
        perm=torch.as_tensor(square_transform["perm"], dtype=torch.long),
        inv_perm=torch.as_tensor(square_transform["inv_perm"], dtype=torch.long),
        signs=torch.as_tensor(square_transform["signs"], dtype=torch.float32),
        global_scale=float(square_transform.get("global_scale", 1.0)),
    )


def _wrap_square_bert_layernorms_inplace(model: Any, transform: SquareMonomialTransform) -> None:
    proxy = _SquareKeyMatProxy(
        key=transform.key(dtype=torch.float32),
        inverse=transform.inverse(dtype=torch.float32),
        hidden_size=transform.dim,
    )
    model.embeddings.LayerNorm = build_keymat_bert_layernorm_bridge(model.embeddings.LayerNorm, proxy)
    for layer in model.encoder.layer:
        layer.attention.output.LayerNorm = build_keymat_bert_layernorm_bridge(
            layer.attention.output.LayerNorm,
            proxy,
        )
        layer.output.LayerNorm = build_keymat_bert_layernorm_bridge(layer.output.LayerNorm, proxy)


def _load_stage_k_bert_profile_bundle(
    *,
    release_dir: str | Path,
    profile: str,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    release_dir = Path(release_dir)
    selected = _read_catalog_profile(release_dir, profile)
    profile_paths = resolve_stage_k_bert_profile_paths(release_dir, profile)
    server_dir = Path(profile_paths["server_dir"])
    client_secret_path = Path(profile_paths["client_secret"])

    tokenizer = _load_bert_tokenizer(server_dir)
    model = _load_bert_model(server_dir, device=device, dtype=dtype)
    secret = torch.load(client_secret_path, map_location="cpu", weights_only=True)

    metadata_path = server_dir.parent / "stage_i_metadata.json"
    stage_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    selected_metadata = selected.get("metadata", {}) if isinstance(selected.get("metadata"), dict) else {}
    secret_metadata = secret.get("metadata", {}) if isinstance(secret.get("metadata"), dict) else {}
    metadata = {**selected_metadata, **secret_metadata, **stage_metadata}
    square_transform = _load_square_transform(metadata)
    _wrap_square_bert_layernorms_inplace(model, square_transform)

    return {
        "tokenizer": tokenizer,
        "model": model,
        "perm_vocab": torch.as_tensor(secret["perm_vocab"], dtype=torch.long),
        "inv_perm_vocab": torch.as_tensor(secret["inv_perm_vocab"], dtype=torch.long),
        "metadata": metadata,
        "square_transform": square_transform,
        "server_dir": str(server_dir),
        "client_secret": str(client_secret_path),
    }


def _restore_hidden_states(hidden: torch.Tensor, transform: SquareMonomialTransform) -> torch.Tensor:
    inverse = transform.inverse(dtype=hidden.dtype).to(hidden.device)
    return torch.matmul(hidden, inverse)


def _restore_pooler_output(
    pooler_output: torch.Tensor | None,
    transform: SquareMonomialTransform,
    metadata: dict[str, Any],
) -> torch.Tensor | None:
    if pooler_output is None:
        return None
    if str(metadata.get("pooler_output_space", "obfuscated")).lower() == "plain":
        return pooler_output
    return _restore_hidden_states(pooler_output, transform)


def _summarize_bert_status(summary: dict[str, Any]) -> str:
    max_pooler_error = summary["max_pooler_output_max_abs_error"]
    passes_hidden = float(summary["max_last_hidden_state_max_abs_error"]) <= BERT_CORRECTNESS_ATOL
    passes_cls = float(summary["max_cls_max_abs_error"]) <= BERT_CORRECTNESS_ATOL
    passes_pooler = True if max_pooler_error is None else float(max_pooler_error) <= BERT_CORRECTNESS_ATOL
    all_finite = not (
        bool(summary["baseline_has_nan_or_inf"])
        or bool(summary["stage_k_has_nan_or_inf"])
        or bool(summary["restored_has_nan_or_inf"])
    )
    return "pass" if passes_hidden and passes_cls and passes_pooler and all_finite else "fail"


@torch.inference_mode()
def _collect_bert_sentence_result(
    *,
    sentence_id: int,
    sentence: str,
    baseline_tokenizer: Any,
    baseline_model: Any,
    stage_bundle: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    encoded = baseline_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    baseline_outputs = baseline_model(**encoded, return_dict=True)
    perm_vocab = stage_bundle["perm_vocab"].to(encoded["input_ids"].device)
    mapped_input_ids = map_input_ids(encoded["input_ids"], perm_vocab)
    stage_inputs = dict(encoded)
    stage_inputs["input_ids"] = mapped_input_ids
    stage_outputs = stage_bundle["model"](**stage_inputs, return_dict=True)

    baseline_last_hidden_state = baseline_outputs.last_hidden_state.detach().cpu().to(torch.float32)
    stage_last_hidden_state = stage_outputs.last_hidden_state.detach().cpu().to(torch.float32)
    restored_last_hidden_state = _restore_hidden_states(stage_last_hidden_state, stage_bundle["square_transform"])

    baseline_pooler_output = getattr(baseline_outputs, "pooler_output", None)
    stage_pooler_output = getattr(stage_outputs, "pooler_output", None)
    baseline_pooler_cpu = (
        baseline_pooler_output.detach().cpu().to(torch.float32)
        if baseline_pooler_output is not None
        else None
    )
    stage_pooler_cpu = (
        stage_pooler_output.detach().cpu().to(torch.float32)
        if stage_pooler_output is not None
        else None
    )
    restored_pooler_output = _restore_pooler_output(
        stage_pooler_cpu,
        stage_bundle["square_transform"],
        stage_bundle["metadata"],
    )

    baseline_cls = baseline_last_hidden_state[:, 0, :]
    restored_cls = restored_last_hidden_state[:, 0, :]
    result = {
        "sentence_id": sentence_id,
        "sentence": sentence,
        "token_count": int(encoded["input_ids"].shape[1]),
        "mapped_input_ids": mapped_input_ids[0].detach().cpu().tolist(),
        "last_hidden_state_max_abs_error": max_abs_error(
            baseline_last_hidden_state,
            restored_last_hidden_state,
        ),
        "last_hidden_state_mean_abs_error": mean_abs_error(
            baseline_last_hidden_state,
            restored_last_hidden_state,
        ),
        "cls_max_abs_error": max_abs_error(baseline_cls, restored_cls),
        "cls_mean_abs_error": mean_abs_error(baseline_cls, restored_cls),
        "baseline_has_nan_or_inf": not bool(torch.isfinite(baseline_last_hidden_state).all().item()),
        "stage_k_has_nan_or_inf": not bool(torch.isfinite(stage_last_hidden_state).all().item()),
        "restored_has_nan_or_inf": not bool(torch.isfinite(restored_last_hidden_state).all().item()),
    }
    if baseline_pooler_cpu is not None and restored_pooler_output is not None:
        result["pooler_output_max_abs_error"] = max_abs_error(baseline_pooler_cpu, restored_pooler_output)
        result["pooler_output_mean_abs_error"] = mean_abs_error(baseline_pooler_cpu, restored_pooler_output)
    else:
        result["pooler_output_max_abs_error"] = None
        result["pooler_output_mean_abs_error"] = None
    return result


def run_stage_k_bert_profile_correctness(
    *,
    release_dir: str | Path,
    profile: str,
    baseline_model_dir: str | Path = DEFAULT_BERT_MODEL_DIR,
    dtype: str = "float32",
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    set_global_seed(seed)
    profile_paths = resolve_stage_k_bert_profile_paths(release_dir, profile)
    baseline_tokenizer = _load_bert_tokenizer(baseline_model_dir)
    baseline_model = _load_bert_model(baseline_model_dir, device=device, dtype=dtype)
    stage_bundle = _load_stage_k_bert_profile_bundle(
        release_dir=release_dir,
        profile=profile,
        device=device,
        dtype=dtype,
    )

    sentence_results: list[dict[str, Any]] = []
    for index, sentence in enumerate(_select_bert_test_sentences(), start=1):
        sentence_results.append(
            _collect_bert_sentence_result(
                sentence_id=index,
                sentence=sentence,
                baseline_tokenizer=baseline_tokenizer,
                baseline_model=baseline_model,
                stage_bundle=stage_bundle,
                device=device,
            )
        )

    summary = summarize_bert_sentence_results(sentence_results)
    status = _summarize_bert_status(summary)
    return {
        "stage": "K",
        "phase": "bert_release_hidden_state_correctness",
        "release_dir": str(release_dir),
        "profile": profile,
        "baseline_model_dir": str(baseline_model_dir),
        "server_dir": profile_paths["server_dir"],
        "client_secret": profile_paths["client_secret"],
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "correctness_atol": BERT_CORRECTNESS_ATOL,
        "transform_family": stage_bundle["metadata"].get("transform_family"),
        "pooler_output_space": stage_bundle["metadata"].get("pooler_output_space"),
        "status": status,
        **summary,
        "summary": {"status": status, **summary},
        "sentences": sentence_results,
    }


def run_stage_k_bert_release_correctness(
    *,
    release_dir: str | Path = "artifacts/stage_k_bert_release",
    output_dir: str | Path = f"{DEFAULT_OUTPUT_DIR}/stage_k_bert_release/correctness",
    profiles: tuple[str, ...] = ("default", "reference"),
    baseline_model_dir: str | Path = DEFAULT_BERT_MODEL_DIR,
    dtype: str = "float32",
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_results: dict[str, dict[str, Any]] = {}
    del max_new_tokens
    for profile in profiles:
        payload = run_stage_k_bert_profile_correctness(
            release_dir=release_dir,
            profile=profile,
            baseline_model_dir=baseline_model_dir,
            dtype=dtype,
            device=device,
            seed=seed,
        )
        profile_results[profile] = payload
        write_json(output_dir / f"{profile}.json", payload)

    blocking = [name for name, payload in profile_results.items() if payload.get("status") != "pass"]
    summary = {
        "stage": "K",
        "phase": "bert_release_hidden_state_correctness",
        "release_dir": str(release_dir),
        "output_dir": str(output_dir),
        "baseline_model_dir": str(baseline_model_dir),
        "profiles": list(profile_results.keys()),
        "profile_evidence_files": {
            name: str(output_dir / f"{name}.json")
            for name in profile_results
        },
        "profile_summaries": {
            name: payload.get("summary", payload)
            for name, payload in profile_results.items()
        },
        "completion_status": "complete" if not blocking else "not_complete",
        "blocking_profiles": blocking,
    }
    write_json(output_dir.parent / "correctness_summary.json", summary)
    return summary


__all__ = [
    "resolve_stage_k_bert_profile_paths",
    "summarize_bert_sentence_results",
    "run_stage_k_bert_profile_correctness",
    "run_stage_k_bert_release_correctness",
]
