from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED, DEFAULT_LLAMA_MODEL_DIR,
                          DEFAULT_LLAMA_BUNDLE_DIR, DEFAULT_LLAMA_CORRECTNESS_DIR)


DEFAULT_LLAMA_INSTRUCT_MODEL_DIR = DEFAULT_LLAMA_MODEL_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Llama release-bundle correctness.")
    parser.add_argument("--bundle-dir", default=DEFAULT_LLAMA_BUNDLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_LLAMA_CORRECTNESS_DIR)
    parser.add_argument("--profiles", nargs="+", default=["default"],
                        help="Default validates the delivery profile; reference is a copy of the same checkpoint.")
    parser.add_argument("--baseline-model-dir", default=DEFAULT_LLAMA_INSTRUCT_MODEL_DIR)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--correctness-atol", type=float, default=2e-4,
                        help="Explicit Llama 1B FP32 tolerance; use 0.0001 for the stricter comparison.")
    parser.add_argument("--isolate-model-processes", action="store_true",
                        help="Run baseline and each exported model in separate processes to bound GPU memory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if os.name == "nt" or args.isolate_model_processes:
        run_isolated_windows_check(args)
        return
    from src.aloepri.evaluation import run_llama_correctness
    summary = run_llama_correctness(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        profiles=tuple(args.profiles),
        baseline_model_dir=args.baseline_model_dir,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        correctness_atol=args.correctness_atol,
    )
    print(f"Saved Llama correctness evidence under {Path(args.output_dir)}")
    print(f"completion_status: {summary.get('completion_status')}")
    if summary.get("completion_status") != "complete":
        raise SystemExit(1)


def run_isolated_windows_check(args) -> None:
    # Keep the coordinator free of Torch/CUDA allocations while each model runs.
    from src.defaults import DEFAULT_PROMPTS
    worker = "src.aloepri._implementation.llama.llama_correctness"
    descriptor, observation_file = tempfile.mkstemp(prefix="aloepri-baseline-", suffix=".pt")
    os.close(descriptor)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = dict(baseline_model_dir=args.baseline_model_dir, dtype=args.dtype,
                  device=args.device, seed=args.seed, max_new_tokens=args.max_new_tokens)
    arguments_file = Path(observation_file + ".json")
    profiles = {}
    try:
        print("Computing original-model observations in an isolated process...", flush=True)
        arguments_file.write_text(json.dumps({**common, "prompts": list(DEFAULT_PROMPTS)}), encoding="utf-8")
        subprocess.run([sys.executable, "-m", worker, "--baseline", observation_file, str(arguments_file)],
                       check=True, cwd=ROOT)
        for profile in args.profiles:
            if not profile.isascii() or not profile.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Profile names must contain only ASCII letters, numbers, '_' and '-'.")
            result_file = output_dir / f"{profile}.json"
            print(f"Comparing exported profile {profile} in an isolated process...", flush=True)
            arguments_file.write_text(json.dumps({**common, "release_dir": args.bundle_dir, "profile": profile,
                                                   "correctness_atol": args.correctness_atol}), encoding="utf-8")
            subprocess.run([sys.executable, "-m", worker, "--compare", observation_file, str(result_file), str(arguments_file)],
                           check=True, cwd=ROOT)
            profiles[profile] = json.loads(result_file.read_text(encoding="utf-8"))
    finally:
        Path(observation_file).unlink(missing_ok=True)
        arguments_file.unlink(missing_ok=True)
    blocking = [name for name, payload in profiles.items() if payload["status"] != "pass"]
    summary = dict(stage="K", phase="llama_release_surface_correctness", release_dir=args.bundle_dir,
                   output_dir=str(output_dir), baseline_model_dir=args.baseline_model_dir,
                   profiles=list(profiles), profile_evidence_files={name: str(output_dir / f"{name}.json") for name in profiles},
                   profile_summaries={name: payload["summary"] for name, payload in profiles.items()},
                   completion_status="not_complete" if blocking else "complete", blocking_profiles=blocking,
                   model_processes_isolated=True)
    (output_dir.parent / "correctness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Llama correctness evidence under {output_dir}")
    print(f"completion_status: {summary['completion_status']}")
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
