from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri.checkpoints import export_qwen_paper_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a Qwen2 square engineering variant or convert an existing artifact.")
    parser.add_argument("--export-dir", default="artifacts/qwen/paper_consistent_checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-dir", help="Existing artifact; historical buffered formats remain research-only.")
    inputs.add_argument("--model-dir", help="Original local Qwen2 checkpoint; creates a same-width square variant.")
    parser.add_argument("--seed", type=int, default=None, help="Explicit reproducible experiment seed; default uses system randomness.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--alpha-h", type=float, default=0.0)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--norm-strategy", default="kappa_fused", choices=["kappa_fused", "metric_diag_sqrt"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_qwen_paper_checkpoint(
        args.export_dir,
        source_dir=args.source_dir,
        materialize=args.materialize,
        norm_strategy=args.norm_strategy,
        model_dir=args.model_dir, seed=args.seed, dtype=args.dtype, device=args.device,
        alpha_e=args.alpha_e, alpha_h=args.alpha_h,
    )
    print(f"Exported Qwen checkpoint to {Path(args.export_dir)}; inspect manifest.json for the implementation variant.")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
