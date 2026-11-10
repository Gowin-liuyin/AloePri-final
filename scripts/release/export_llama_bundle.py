from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri.release import ReleaseProfile, export_release_bundle
from src.defaults import DEFAULT_LLAMA_CHECKPOINT_DIR, DEFAULT_LLAMA_BUNDLE_DIR, DEFAULT_LLAMA_CORRECTNESS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a local Llama evaluation bundle containing separate client secrets.")
    parser.add_argument("--source-dir", default=DEFAULT_LLAMA_CHECKPOINT_DIR)
    parser.add_argument("--export-dir", default=DEFAULT_LLAMA_BUNDLE_DIR)
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--default-correctness-file", default=f"{DEFAULT_LLAMA_CORRECTNESS_DIR}/default.json")
    parser.add_argument("--reference-correctness-file", default=f"{DEFAULT_LLAMA_CORRECTNESS_DIR}/reference.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profiles = [
        ReleaseProfile(
            name="default",
            source_dir=args.source_dir,
            description="Default Llama square engineering variant.",
            recommended_use="Local evaluation; upload only its server directory.",
            correctness_evidence_file=args.default_correctness_file,
        ),
        ReleaseProfile(
            name="reference",
            source_dir=args.source_dir,
            description="Reference Llama profile of the same source checkpoint.",
            recommended_use="Local audit and evidence; keep client secrets private.",
            correctness_evidence_file=args.reference_correctness_file,
        ),
    ]
    export_release_bundle(
        args.export_dir,
        profiles=profiles,
        materialize=args.materialize,
        title="Llama Local Evaluation Bundle",
    )
    print(f"Exported Llama release bundle to {Path(args.export_dir)}")


if __name__ == "__main__":
    main()
