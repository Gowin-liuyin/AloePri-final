from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri.private_inference import generate_private_text
from src.defaults import DEFAULT_LLAMA_BUNDLE_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run private inference from a release bundle profile.")
    parser.add_argument("--bundle-dir", default=DEFAULT_LLAMA_BUNDLE_DIR)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        generate_private_text(
            bundle_dir=Path(args.bundle_dir),
            profile=args.profile,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            device=args.device, do_sample=args.do_sample, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
