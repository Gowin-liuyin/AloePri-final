from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri._implementation.bert.bert_correctness import run_stage_k_bert_release_correctness
from src.aloepri._implementation.bert.bert_paper_checkpoint import DEFAULT_BERT_MODEL_DIR
from src.defaults import DEFAULT_MAX_NEW_TOKENS, DEFAULT_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BERT release-bundle correctness.")
    parser.add_argument("--bundle-dir", default="artifacts/bert/release_bundle")
    parser.add_argument("--baseline-model-dir", default=DEFAULT_BERT_MODEL_DIR)
    parser.add_argument("--output-dir", default="outputs/evaluation/bert/correctness")
    parser.add_argument("--profiles", nargs="+", default=["default", "reference"])
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_stage_k_bert_release_correctness(
        release_dir=args.bundle_dir,
        output_dir=args.output_dir,
        profiles=tuple(args.profiles),
        baseline_model_dir=args.baseline_model_dir,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    print(f"Saved BERT correctness evidence under {Path(args.output_dir)}")
    print(f"completion_status: {summary.get('completion_status')}")


if __name__ == "__main__":
    main()
