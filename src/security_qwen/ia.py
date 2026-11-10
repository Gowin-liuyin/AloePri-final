"""IA 的未实现结果模板；空指标不表示通过 Gate-IA 或 Attn-IA 安全测试。"""

from __future__ import annotations

from src.security_qwen.weight_attacks import recover_weight_mapping, recover_from_directories
from src.security_qwen.metrics import classify_risk_level
from src.security_qwen.schema import SecurityEvalTarget, build_security_eval_payload


def build_ia_template(target: SecurityEvalTarget) -> dict:
    primary_metric = None
    return build_security_eval_payload(
        attack="ia",
        target=target,
        config={
            "phase": "phase0",
            "dataset_name": "phase0_smoke_prompts",
            "seed": 20260323,
            "variants": ["gate_ia", "attn_ia"],
        },
        metrics={
            "token_top1_recovery_rate": primary_metric,
            "invariant_separability_score": None,
        },
        summary={
            "status": "planned",
            "primary_metric_name": "token_top1_recovery_rate",
            "primary_metric_value": primary_metric,
            "risk_level": classify_risk_level(primary_metric),
            "notes": "Unexecuted template only. Use weight_attacks.recover_from_directories for Gate-IA or the documented Attn-IA interpretation; no measured attack result is stored here.",
        },
        artifacts={},
    )
