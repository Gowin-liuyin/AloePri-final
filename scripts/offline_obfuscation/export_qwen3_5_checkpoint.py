from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aloepri.checkpoints.qwen3_5 import export_qwen3_5_square_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obfuscate a Qwen3.5/3.8 text tower into a standard HF checkpoint (square_monomial)."
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="original multimodal Qwen3.5/3.8 checkpoint")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha-e", type=float, default=0.0)
    parser.add_argument("--alpha-h", type=float, default=0.0)
    args = parser.parse_args()

    result = export_qwen3_5_square_checkpoint(
        args.export_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        alpha_e=args.alpha_e,
        alpha_h=args.alpha_h,
    )
    for name, path in result.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
