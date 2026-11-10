from __future__ import annotations

import json
import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_OUTPUT_DIR, DEFAULT_PROMPTS, DEFAULT_SEED
from src.evaluator import max_abs_error, mean_abs_error, write_json
from src.llama_local_dev import tokenize_llama_prompt
from src.model_loader import load_model_and_tokenizer, set_global_seed, tokenize_prompt
from src.aloepri._implementation.analysis.equivalence import equivalence_status
from src.aloepri._implementation.export.checkpoint_io import load_stage_i_hf_bundle
from src.aloepri._implementation.llama.llama_paper_checkpoint import DEFAULT_LLAMA_INSTRUCT_MODEL_DIR
from src.transforms import map_input_ids, restore_logits


def resolve_stage_k_llama_profile_paths(release_dir: str | Path, profile: str) -> dict[str, str]:
    release_dir = Path(release_dir)
    catalog = json.loads((release_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    if profile not in profile_map:
        raise ValueError(f"Unknown Stage-K Llama profile: {profile}")
    selected = profile_map[profile]
    return {
        "server_dir": str(release_dir / selected["server_dir"]),
        "client_secret": str(release_dir / selected["client_secret"]),
        "correctness_evidence_file": str(selected.get("correctness_evidence_file", "")),
    }


def summarize_llama_prompt_results(items: list[dict[str, Any]]) -> dict[str, float | bool]:
    count = max(len(items), 1)
    return {
        "prompt_count": len(items),
        "max_restored_full_logits_max_abs_error": max((float(item["full_logits_max_abs_error"]) for item in items), default=float("inf")),
        "avg_restored_full_logits_max_abs_error": sum(float(item["full_logits_max_abs_error"]) for item in items) / count,
        "avg_restored_full_logits_mean_abs_error": sum(float(item["full_logits_mean_abs_error"]) for item in items) / count,
        "avg_restored_last_token_max_abs_error": sum(float(item["last_token_logits_max_abs_error"]) for item in items) / count,
        "avg_restored_last_token_mean_abs_error": sum(float(item["last_token_logits_mean_abs_error"]) for item in items) / count,
        "greedy_first_token_match_rate": sum(1.0 for item in items if item["greedy_first_token_match"]) / count,
        "generated_ids_exact_match_rate": sum(1.0 for item in items if item["generated_ids_exact_match"]) / count,
        "generated_text_exact_match_rate": sum(1.0 for item in items if item["generated_text_exact_match"]) / count,
        "baseline_has_nan_or_inf": any(bool(item["baseline_has_nan_or_inf"]) for item in items),
        "stage_k_has_nan_or_inf": any(bool(item["stage_k_has_nan_or_inf"]) for item in items),
    }


@torch.inference_mode()
def _greedy_generate_plain(model: Any, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    current_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(input_ids=current_ids, attention_mask=torch.ones_like(current_ids)).logits.detach()
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("Non-finite logits during baseline generation; equivalence cannot pass.")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        current_ids = torch.cat([current_ids, next_token.to(current_ids.device)], dim=1)
    return current_ids[:, input_ids.shape[1] :]


@torch.inference_mode()
def _greedy_generate_stage_k(
    model: Any,
    input_ids: torch.Tensor,
    perm_vocab: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    current_plain_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        mapped_ids = map_input_ids(current_plain_ids.cpu(), perm_vocab.cpu()).to(current_plain_ids.device)
        logits_perm = model(input_ids=mapped_ids, attention_mask=torch.ones_like(mapped_ids)).logits.detach().to(torch.float32)
        if not bool(torch.isfinite(logits_perm).all()):
            raise ValueError("Non-finite logits during mapped generation; equivalence cannot pass.")
        restored_logits = restore_logits(logits_perm[:, -1, :].cpu(), perm_vocab.cpu())
        next_token = torch.argmax(restored_logits, dim=-1, keepdim=True)
        current_plain_ids = torch.cat([current_plain_ids, next_token.to(current_plain_ids.device)], dim=1)
    return current_plain_ids[:, input_ids.shape[1] :]


@torch.inference_mode()
def _collect_baseline_results(*, baseline_model_dir, dtype, device, seed, max_new_tokens, prompts):
    set_global_seed(seed)
    print("Loading original model...", flush=True)
    tokenizer, baseline_model = load_model_and_tokenizer(str(baseline_model_dir), device=device, dtype=dtype)
    results = []
    for index, prompt in enumerate(prompts, 1):
        print(f"Original model prompt {index}/{len(prompts)}", flush=True)
        encoded = (tokenize_prompt(tokenizer, prompt, device=device)
                   if baseline_model.config.model_type == "qwen2" and getattr(tokenizer, "chat_template", None)
                   else tokenize_llama_prompt(tokenizer, prompt, device=device))
        logits = baseline_model(**encoded).logits.detach().cpu().to(torch.float32)
        generated = _greedy_generate_plain(baseline_model, encoded["input_ids"], max_new_tokens)[0].cpu()
        results.append(({key: value.cpu() for key, value in encoded.items()}, logits, generated))
    return results


def _baseline_observations(**kwargs):
    if os.name != "nt":
        return _collect_baseline_results(**kwargs)
    # A separate process returns Windows commit memory, including native allocator caches.
    descriptor, filename = tempfile.mkstemp(prefix="aloepri-baseline-", suffix=".pt")
    os.close(descriptor)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.aloepri._implementation.llama.llama_correctness", filename],
            input=json.dumps(kwargs), text=True, capture_output=True,
            cwd=Path(__file__).resolve().parents[4],
        )
        if result.returncode:
            raise RuntimeError(f"Baseline worker failed ({result.returncode}): {result.stderr}")
        return torch.load(filename, map_location="cpu", weights_only=True)
    finally:
        Path(filename).unlink(missing_ok=True)


@torch.inference_mode()
def run_stage_k_llama_profile_correctness(
    *,
    release_dir: str | Path,
    profile: str,
    baseline_model_dir: str | Path = DEFAULT_LLAMA_INSTRUCT_MODEL_DIR,
    dtype: str = "bfloat16",
    device: str = "cuda",
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    correctness_atol: float = 1e-4,
    baseline_observations_path: str | Path | None = None,
) -> dict[str, Any]:
    set_global_seed(seed)
    profile_paths = resolve_stage_k_llama_profile_paths(release_dir, profile)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_results = (torch.load(baseline_observations_path, map_location="cpu", weights_only=True)
                        if baseline_observations_path is not None else _baseline_observations(
                            baseline_model_dir=str(baseline_model_dir), dtype=dtype, device=device, seed=seed,
                            max_new_tokens=max_new_tokens, prompts=list(DEFAULT_PROMPTS)))
    tokenizer = AutoTokenizer.from_pretrained(str(baseline_model_dir), trust_remote_code=False)
    release_bundle = load_stage_i_hf_bundle(
        profile_paths["server_dir"],
        client_secret_path=profile_paths["client_secret"],
        device=device,
        dtype=dtype,
    )
    stage_model = release_bundle["model"]
    exported_tokenizer = release_bundle["tokenizer"]
    perm_vocab = release_bundle["perm_vocab"]
    if perm_vocab is None:
        raise ValueError(f"client secret is required for Stage-K Llama correctness: {profile}")

    prompt_results: list[dict[str, Any]] = []
    for index, (prompt, baseline_result) in enumerate(zip(DEFAULT_PROMPTS, baseline_results), start=1):
        encoded_cpu, baseline_logits, baseline_generated_ids = baseline_result
        encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
        mapped_ids = map_input_ids(encoded["input_ids"].cpu(), perm_vocab.cpu()).to(encoded["input_ids"].device)
        stage_logits_perm = stage_model(
            input_ids=mapped_ids,
            attention_mask=encoded["attention_mask"],
        ).logits.detach().cpu().to(torch.float32)
        restored_logits = restore_logits(stage_logits_perm, perm_vocab.cpu())
        stage_generated_ids = _greedy_generate_stage_k(
            stage_model,
            encoded["input_ids"],
            perm_vocab,
            max_new_tokens,
        )[0].cpu()
        prompt_results.append(
            {
                "prompt_id": index,
                "prompt": prompt,
                "mapped_input_ids": mapped_ids[0].detach().cpu().tolist(),
                "full_logits_max_abs_error": max_abs_error(baseline_logits, restored_logits),
                "full_logits_mean_abs_error": mean_abs_error(baseline_logits, restored_logits),
                "last_token_logits_max_abs_error": max_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "last_token_logits_mean_abs_error": mean_abs_error(baseline_logits[0, -1], restored_logits[0, -1]),
                "greedy_first_token_match": int(torch.argmax(baseline_logits[0, -1]).item()) == int(torch.argmax(restored_logits[0, -1]).item()),
                "generated_ids_exact_match": baseline_generated_ids.tolist() == stage_generated_ids.tolist(),
                "generated_text_exact_match": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True)
                == exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_generated_ids": baseline_generated_ids.tolist(),
                "stage_k_generated_ids": stage_generated_ids.tolist(),
                "baseline_generated_text": tokenizer.decode(baseline_generated_ids, skip_special_tokens=True),
                "stage_k_generated_text": exported_tokenizer.decode(stage_generated_ids, skip_special_tokens=True),
                "baseline_has_nan_or_inf": not bool(torch.isfinite(baseline_logits).all().item()),
                "stage_k_has_nan_or_inf": not bool(torch.isfinite(stage_logits_perm).all().item()),
            }
        )

    summary = summarize_llama_prompt_results(prompt_results)
    summary = equivalence_status(summary, atol=correctness_atol)
    status = summary["status"]
    return {
        "stage": "K",
        "phase": "release_original_model_equivalence",
        "release_dir": str(release_dir),
        "profile": profile,
        "baseline_model_dir": str(baseline_model_dir),
        "server_dir": profile_paths["server_dir"],
        "client_secret": profile_paths["client_secret"],
        "dtype": dtype,
        "device": device,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "status": status,
        **summary,
        "summary": {"status": status, **summary},
        "prompts": prompt_results,
    }


def run_stage_k_llama_release_correctness(
    *,
    release_dir: str | Path = "artifacts/stage_k_llama_release",
    output_dir: str | Path = f"{DEFAULT_OUTPUT_DIR}/stage_k_llama_release/correctness",
    profiles: tuple[str, ...] = ("default", "reference"),
    baseline_model_dir: str | Path = DEFAULT_LLAMA_INSTRUCT_MODEL_DIR,
    dtype: str = "bfloat16",
    device: str = "cuda",
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    correctness_atol: float = 1e-4,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_results: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        payload = run_stage_k_llama_profile_correctness(
            release_dir=release_dir,
            profile=profile,
            baseline_model_dir=baseline_model_dir,
            dtype=dtype,
            device=device,
            seed=seed,
            max_new_tokens=max_new_tokens,
            correctness_atol=correctness_atol,
        )
        profile_results[profile] = payload
        write_json(output_dir / f"{profile}.json", payload)

    blocking = [name for name, payload in profile_results.items() if payload.get("status") != "pass"]
    summary = {
        "stage": "K",
        "phase": "llama_release_surface_correctness",
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


if __name__ == "__main__":
    arguments = (json.loads(Path(sys.argv[-1]).read_text(encoding="utf-8"))
                 if sys.argv[1] in {"--baseline", "--compare"} else json.loads(sys.stdin.read()))
    if sys.argv[1] == "--compare":
        payload = run_stage_k_llama_profile_correctness(
            **arguments, baseline_observations_path=sys.argv[2])
        write_json(Path(sys.argv[3]), payload)
    elif sys.argv[1] == "--baseline":
        torch.save(_collect_baseline_results(**arguments), sys.argv[2])
    else:
        torch.save(_collect_baseline_results(**arguments), sys.argv[1])
