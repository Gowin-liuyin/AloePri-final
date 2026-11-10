from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED
from src.aloepri.evaluation import run_qwen_correctness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen release-bundle correctness.")
    parser.add_argument("--bundle-dir", default="artifacts/qwen/release_bundle")
    parser.add_argument("--output-dir", default="outputs/evaluation/qwen/correctness")
    parser.add_argument("--profiles", nargs="+", default=["default", "reference"])
    parser.add_argument("--buffered-source-dir", default=None)
    parser.add_argument("--baseline-model-dir", default=None, help="Original Qwen2 checkpoint for square-variant equivalence.")
    parser.add_argument("--correctness-atol", type=float, default=1e-4)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_qwen_correctness(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        profiles=tuple(args.profiles),
        buffered_source_dir=args.buffered_source_dir,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        correctness_atol=args.correctness_atol, baseline_model_dir=args.baseline_model_dir,
    )
    print(f"Saved Qwen correctness evidence under {Path(args.output_dir)}")
    print(f"completion_status: {summary.get('completion_status')}")


if __name__ == "__main__":
    main()
