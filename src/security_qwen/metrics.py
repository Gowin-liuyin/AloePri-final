from __future__ import annotations


def classify_risk_level(primary_metric_value: float | None) -> str:
    """按项目自定的 10% 和 30% 分档便于查看，不是论文给出的隐私安全等级。"""
    if primary_metric_value is None:
        return "unknown"
    if primary_metric_value >= 0.3:
        return "high"
    if primary_metric_value >= 0.1:
        return "medium"
    return "low"
