from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_LLAMA_MODEL_DIR, DEFAULT_LLAMA_CHECKPOINT_DIR
from src.aloepri.checkpoints import export_llama_paper_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a full-layer same-width square Llama engineering variant.")
    parser.add_argument("--model-dir", default=DEFAULT_LLAMA_MODEL_DIR)
    parser.add_argument("--export-dir", default=DEFAULT_LLAMA_CHECKPOINT_DIR)
    parser.add_argument("--seed", type=int, default=None, help="Explicit experiment seed; omitted means system randomness.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--alpha-h", type=float, default=0.0)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Must equal all decoder layers for standard export; omitted means all layers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_llama_paper_checkpoint(
        args.export_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
        num_layers=args.num_layers,
    )
    print(f"Exported Llama same-width square checkpoint to {Path(args.export_dir)}")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
