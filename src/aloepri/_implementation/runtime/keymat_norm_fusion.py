from __future__ import annotations

import math
import torch
from torch import nn

from src.keymat import KeyMatTransform


class KeyMatFusedRMSNorm(nn.Module):
    def __init__(
        self,
        norm_layer: nn.Module,
        keymat_transform: KeyMatTransform,
        kappa: float,
        recorder=None,
        record_name: str | None = None,
        mode: str = "exact_metric_reference",
    ) -> None:
        super().__init__()
        self.variance_epsilon = float(norm_layer.variance_epsilon)
        self.kappa = float(kappa)
        if mode not in {"exact_metric_reference", "scalar_rms_approximation"}:
            raise ValueError(f"Unknown normalization mode: {mode}")
        if not math.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be finite and positive.")
        self.mode = mode
        self.recorder = recorder
        self.record_name = record_name

        q = keymat_transform.inverse.detach().to(device="cpu", dtype=torch.float32)
        norm_weight = norm_layer.weight.detach().to(device="cpu", dtype=torch.float32)
        restore = q * norm_weight.unsqueeze(0) if recorder is not None else None
        # The scalar mode must not allocate an unused D-by-D custom norm matrix.
        metric_matrix = q @ q.T if mode == "exact_metric_reference" else None
        self.register_buffer("restore_matrix", restore, persistent=False)
        self.register_buffer("metric_matrix", metric_matrix, persistent=False)
        self.hidden_size = int(keymat_transform.hidden_size)
        self.expanded_size = int(keymat_transform.expanded_size)
        self.scalar_coefficient = self.kappa * math.sqrt(self.hidden_size / self.expanded_size)
        self.scalar_epsilon = self.scalar_coefficient**2 * self.variance_epsilon

    def _apply(self, fn, recurse=True):
        # Keep normalization statistics in float32 even when the model's linear
        # weights are cast to fp16/bfloat16. Converting back after rounding loses
        # precision, so transfer the original float32 buffers to the new device.
        original = {name: getattr(self, name) for name in ("metric_matrix", "restore_matrix")}
        result = super()._apply(fn, recurse=recurse)
        for name, tensor in original.items():
            converted = getattr(self, name)
            if tensor is not None:
                setattr(self, name, tensor.to(device=converted.device, dtype=torch.float32))
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_fp32 = hidden_states.to(torch.float32)
        if self.mode == "exact_metric_reference":
            # Custom reference: this computes the norm of x_tilde Q exactly.
            metric_matrix = self.metric_matrix.to(device=hidden_states.device, dtype=torch.float32)
            quadratic = torch.matmul(hidden_fp32, metric_matrix)
            variance = (quadratic * hidden_fp32).sum(dim=-1, keepdim=True) / float(self.hidden_size)
            output = hidden_fp32 * torch.rsqrt(variance.clamp_min(0) + self.variance_epsilon)
        else:
            # Scalar approximation with the missing dimension ratio restored.
            # Scaling epsilon too is exact when P preserves every norm up to kappa.
            variance = hidden_fp32.square().mean(dim=-1, keepdim=True)
            output = hidden_fp32 * torch.rsqrt(variance + self.scalar_epsilon) * self.scalar_coefficient
        if self.recorder is not None and self.record_name is not None:
            restored = torch.matmul(
                output,
                self.restore_matrix.to(device=hidden_states.device, dtype=torch.float32),
            )
            self.recorder.record(self.record_name, restored)
        return output.to(input_dtype)


def build_keymat_fused_rmsnorm(
    norm_layer: nn.Module,
    keymat_transform: KeyMatTransform,
    kappa: float,
    recorder=None,
    record_name: str | None = None,
    mode: str = "exact_metric_reference",
) -> KeyMatFusedRMSNorm:
    return KeyMatFusedRMSNorm(
        norm_layer=norm_layer,
        keymat_transform=keymat_transform,
        kappa=kappa,
        recorder=recorder,
        record_name=record_name,
        mode=mode,
    )
