"""Export a local BERT research checkpoint requiring a custom LayerNorm bridge.

Usage:
  python scripts/offline_obfuscation/export_bert_checkpoint.py \
    --model-dir model/bert-base-uncased \
    --export-dir artifacts/bert/paper_consistent_checkpoint \
    --device cpu \
    --dtype float32 \
    --alpha-e 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.defaults import DEFAULT_SEED
from src.aloepri.checkpoints import export_bert_paper_checkpoint


DEFAULT_BERT_MODEL_DIR = "model/bert-base-uncased"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a local BERT research checkpoint requiring a custom LayerNorm bridge.")
    parser.add_argument("--model-dir", default=DEFAULT_BERT_MODEL_DIR)
    parser.add_argument("--export-dir", default="artifacts/bert/paper_consistent_checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Explicit experiment seed; omitted means system randomness.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha-e", type=float, default=0.02)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Adapt the first N encoder layers. Defaults to all layers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_bert_paper_checkpoint(
        args.export_dir,
        model_dir=args.model_dir,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        alpha_e=args.alpha_e,
        num_layers=args.num_layers,
    )
    print(f"Exported BERT paper-consistent checkpoint to {Path(args.export_dir)}")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
