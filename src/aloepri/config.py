from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class AloePriConfig:
    # Model parameters
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float = 10000.0
    architecture_family: str = "qwen_decoder"
    model_type: str = "qwen"

    # Obfuscation parameters
    expansion_size: int = 128
    lam: float = 0.3
    alpha_e: float = 1.0
    alpha_h: float = 0.2

    # Attention specific
    attention_profile: str = "rqk_hqk_block_taukv_taugroup"
    qk_scale_range: tuple[float, float] = (0.95, 1.05)
    beta: int = 8
    gamma: float = 1e3

    # FFN specific
    ffn_scale_range: tuple[float, float] = (0.9, 1.1)

    # Execution parameters
    # None draws a secret seed once; an explicit seed is for repeatable experiments.
    seed: int | None = None
    device: str = "cpu"
    dtype: str = "float32"
    # The exact mode is a custom reference, not the paper's scalar approximation.
    normalization_mode: str = "exact_metric_reference"
    model_vocab_size: int | None = None

    # Adapted layers (list of layer indices)
    adapted_layers: list[int] | None = None

    def __post_init__(self):
        if self.adapted_layers is None:
            # Default to all layers if not specified
            object.__setattr__(self, 'adapted_layers', list(range(self.num_hidden_layers)))
        layers = self.adapted_layers
        if any(type(index) is not int for index in layers):
            raise ValueError("adapted_layers must contain integer layer indices.")
        if layers != list(range(len(layers))) or len(layers) > self.num_hidden_layers:
            raise ValueError("Only a contiguous prefix [0, ..., k-1] is supported for adapted_layers.")
        if self.num_hidden_layers < 1 or self.hidden_size < 1:
            raise ValueError("hidden_size and num_hidden_layers must be positive.")
        if self.num_attention_heads < 1 or self.num_key_value_heads < 1 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.head_dim < 1 or (self.architecture_family != "bert_encoder" and self.head_dim % 2):
            raise ValueError("head_dim must be positive and even for RoPE.")
        if self.normalization_mode not in {"exact_metric_reference", "scalar_rms_approximation"}:
            raise ValueError("Unknown normalization_mode; choose exact_metric_reference or scalar_rms_approximation.")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be None or an integer in [0, 2**63).")

    @classmethod
    def from_model(cls, model, **overrides) -> "AloePriConfig":
        from src.aloepri.adapters import build_architecture_config

        return build_architecture_config(model, **overrides)
