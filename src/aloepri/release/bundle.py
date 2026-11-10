from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.aloepri._implementation.export.release_bundle_impl import StageKProfile, export_stage_k_release


@dataclass(frozen=True)
class ReleaseProfile:
    name: str
    source_dir: str
    description: str
    recommended_use: str
    correctness_evidence_file: str | None = None


def _to_compat_profile(profile: ReleaseProfile) -> StageKProfile:
    return StageKProfile(
        name=profile.name,
        source_dir=profile.source_dir,
        description=profile.description,
        recommended_use=profile.recommended_use,
        correctness_evidence_file=profile.correctness_evidence_file,
    )


def export_release_bundle(
    export_dir: str | Path,
    *,
    profiles: list[ReleaseProfile],
    materialize: bool = False,
    recommended_profile: str = "default",
    reference_profile: str = "reference",
    title: str = "Paper-Consistent Release Bundle",
) -> dict:
    return export_stage_k_release(
        export_dir,
        profiles=[_to_compat_profile(profile) for profile in profiles],
        materialize=materialize,
        recommended_profile=recommended_profile,
        reference_profile=reference_profile,
        title=title,
    )
