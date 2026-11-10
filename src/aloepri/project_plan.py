from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectLayer:
    name: str
    status: str
    responsibility: str
    modules: list[str]


def paper_aligned_project_layers() -> list[ProjectLayer]:
    return [
        ProjectLayer(
            name="model_adaptation",
            status="active",
            responsibility="Normalize Qwen and Llama model structure behind one adapter interface.",
            modules=["src/aloepri/adapters"],
        ),
        ProjectLayer(
            name="obfuscation_keys",
            status="active",
            responsibility="Generate token permutations, KeyMat transforms, attention keys, FFN transforms, and norm metadata.",
            modules=["src/aloepri/keys.py", "src/keymat.py", "src/attention_keys.py"],
        ),
        ProjectLayer(
            name="component_obfuscation",
            status="active",
            responsibility="Apply paper-level obfuscation to embedding, attention, FFN, RMSNorm, and lm head components.",
            modules=["src/aloepri/layers"],
        ),
        ProjectLayer(
            name="offline_obfuscation",
            status="active",
            responsibility="Materialize paper-consistent checkpoints from baseline Hugging Face models.",
            modules=["src/aloepri/engine.py", "src/aloepri/checkpoints"],
        ),
        ProjectLayer(
            name="private_inference",
            status="active",
            responsibility="Map prompt token ids into the private token space and restore output ids or logits.",
            modules=["src/aloepri/token_ops.py", "scripts/release/infer_bundle.py"],
        ),
        ProjectLayer(
            name="release_bundle",
            status="active",
            responsibility="Package server weights, client secrets, profiles, catalog metadata, and deployment contract.",
            modules=["src/aloepri/release", "scripts/release"],
        ),
        ProjectLayer(
            name="evaluation",
            status="active",
            responsibility="Run correctness checks and privacy attacks independently from checkpoint construction.",
            modules=["src/aloepri/evaluation", "src/security_qwen", "scripts/evaluation"],
        ),
    ]


def project_plan_payload() -> dict[str, Any]:
    return {
        "format": "aloepri_paper_aligned_project_plan_v1",
        "layers": [asdict(layer) for layer in paper_aligned_project_layers()],
    }
