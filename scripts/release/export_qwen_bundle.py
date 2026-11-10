from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aloepri.release import ReleaseProfile, export_release_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a local Qwen evaluation bundle containing separate client secrets.")
    parser.add_argument("--source-dir", default="artifacts/qwen/paper_consistent_checkpoint")
    parser.add_argument("--export-dir", default="artifacts/qwen/release_bundle")
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--default-correctness-file", default="outputs/evaluation/qwen/correctness/default.json")
    parser.add_argument("--reference-correctness-file", default="outputs/evaluation/qwen/correctness/reference.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profiles = [
        ReleaseProfile(
            name="default",
            source_dir=args.source_dir,
            description="Default Qwen checkpoint profile; inspect its implementation variant.",
            recommended_use="Local evaluation; upload only its server directory.",
            correctness_evidence_file=args.default_correctness_file,
        ),
        ReleaseProfile(
            name="reference",
            source_dir=args.source_dir,
            description="Reference Qwen profile of the same source checkpoint.",
            recommended_use="Local audit and evidence; keep client secrets private.",
            correctness_evidence_file=args.reference_correctness_file,
        ),
    ]
    export_release_bundle(
        args.export_dir,
        profiles=profiles,
        materialize=args.materialize,
        title="Qwen Local Evaluation Bundle",
    )
    print(f"Exported Qwen release bundle to {Path(args.export_dir)}")


if __name__ == "__main__":
    main()
