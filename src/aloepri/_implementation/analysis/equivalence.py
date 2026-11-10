"""判断小样本决定性等价；这不能代替真实任务准确率评测。"""
from __future__ import annotations
import math
from typing import Any


def equivalence_status(summary: dict[str, Any], *, atol: float = 1e-4) -> dict[str, Any]:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("Equivalence tolerance must be finite and nonnegative.")
    try:
        error = float(summary.get("max_restored_full_logits_max_abs_error", math.inf))
        passed = (
            int(summary.get("prompt_count", 0)) > 0
            and {"baseline_has_nan_or_inf", "stage_k_has_nan_or_inf"}.issubset(summary)
            and float(summary.get("generated_ids_exact_match_rate", 0)) == 1.0
            and float(summary.get("generated_text_exact_match_rate", 0)) == 1.0
            and not any(bool(value) for key, value in summary.items() if "has_nan_or_inf" in key)
            and math.isfinite(error) and 0 <= error <= atol
        )
    except (TypeError, ValueError, OverflowError):
        passed = False
    # status 最后写入，调用者传入的同名字段不能覆盖验收结论。
    return {**summary, "status": "pass" if passed else "fail", "correctness_atol": atol,
            "validation_scope": "small_sample_deterministic_equivalence", "task_quality_tested": False}
