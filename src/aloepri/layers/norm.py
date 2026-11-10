from __future__ import annotations

import torch
from torch import nn
from typing import Optional

from src.keymat import KeyMatTransform
from src.aloepri.layers.base import ObfuscatedLayer
from src.aloepri._implementation.bert.bert_layernorm_bridge import (
    KeyMatBertLayerNormBridge,
    build_keymat_bert_layernorm_bridge,
)
from src.aloepri._implementation.runtime.keymat_norm_fusion import KeyMatFusedRMSNorm


class AloePriRMSNorm(ObfuscatedLayer):
    """
    Standardized AloePri RMSNorm.
    Wraps KeyMatFusedRMSNorm from stage_g_norm.
    """

    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        kappa: float,
        recorder: Optional[object] = None,
        record_name: Optional[str] = None,
        mode: str = "exact_metric_reference",
    ) -> None:
        super().__init__(recorder=recorder, record_name=record_name)
        self.inner = KeyMatFusedRMSNorm(
            norm_layer=norm_layer,
            keymat_transform=keymat_transform,
            kappa=kappa,
            mode=mode,
            recorder=recorder,
            record_name=record_name,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # KeyMatFusedRMSNorm handles recording inside its forward
        return self.inner(hidden_states)


def wrap_norm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    kappa: float,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
    mode: str = "exact_metric_reference",
) -> AloePriRMSNorm:
    return AloePriRMSNorm(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        kappa=kappa,
        mode=mode,
        recorder=recorder,
        record_name=record_name,
    )


def wrap_bert_norm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    recorder: Optional[object] = None,
    record_name: Optional[str] = None,
) -> KeyMatBertLayerNormBridge:
    return build_keymat_bert_layernorm_bridge(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        recorder=recorder,
        record_name=record_name,
    )


__all__ = [
    "AloePriRMSNorm",
    "KeyMatBertLayerNormBridge",
    "build_keymat_bert_layernorm_bridge",
    "wrap_norm",
    "wrap_bert_norm",
]
