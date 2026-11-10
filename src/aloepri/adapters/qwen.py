from __future__ import annotations

from typing import Any

from src.aloepri.adapters._base import ArchitectureAdapter


class QwenArchitectureAdapter(ArchitectureAdapter):
    """
    Structural adapter for Qwen-style decoder-only causal LM models.

    This adapter intentionally validates by module layout instead of by a fixed
    `model_type` string, so it can cover Qwen checkpoints of different sizes as
    long as they expose the same decoder architecture contract.
    """

    @classmethod
    def from_model(cls, model: Any) -> "QwenArchitectureAdapter":
        config = cls._require_attr(model, "config", "Qwen")
        model_type = str(getattr(config, "model_type", "")).lower()
        if "qwen" not in model_type:
            raise ValueError(f"Expected a Qwen-family model_type, got {model_type!r}")

        cls._require_attr(model, "config.hidden_size", "Qwen")
        cls._require_attr(model, "config.num_hidden_layers", "Qwen")
        cls._require_attr(model, "config.num_attention_heads", "Qwen")
        cls._require_attr(model, "model.embed_tokens", "Qwen")
        cls._require_attr(model, "model.layers", "Qwen")
        cls._require_attr(model, "model.norm", "Qwen")
        cls._require_attr(model, "lm_head", "Qwen")
        layers = cls._require_attr(model, "model.layers", "Qwen")
        if len(layers) == 0:
            raise ValueError("Qwen-architecture model must have at least one decoder layer.")
        layer0 = layers[0]
        cls._require_attr(layer0, "self_attn.q_proj", "Qwen")
        cls._require_attr(layer0, "self_attn.k_proj", "Qwen")
        cls._require_attr(layer0, "self_attn.v_proj", "Qwen")
        cls._require_attr(layer0, "self_attn.o_proj", "Qwen")
        cls._require_attr(layer0, "input_layernorm", "Qwen")
        cls._require_attr(layer0, "post_attention_layernorm", "Qwen")
        cls._require_attr(layer0, "mlp.gate_proj", "Qwen")
        cls._require_attr(layer0, "mlp.up_proj", "Qwen")
        cls._require_attr(layer0, "mlp.down_proj", "Qwen")
        return cls(model=model)

    @property
    def layers(self):
        return self.model.model.layers

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description["is_tied_word_embeddings"] = bool(
            self.model.get_input_embeddings().weight.data_ptr() == self.model.get_output_embeddings().weight.data_ptr()
        )
        return description


def is_qwen_compatible_model(model: Any) -> bool:
    return QwenArchitectureAdapter.is_compatible_model(model)


def build_qwen_config(
    model: Any,
    *,
    expansion_size: int = 128,
    lam: float = 0.3,
    alpha_e: float = 1.0,
    alpha_h: float = 0.2,
    attention_profile: str = "rqk_hqk_block_taukv_taugroup",
    qk_scale_range: tuple[float, float] = (0.95, 1.05),
    beta: int = 8,
    gamma: float = 1e3,
    ffn_scale_range: tuple[float, float] = (0.9, 1.1),
    seed: int | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    normalization_mode: str = "exact_metric_reference",
    adapted_layers: list[int] | None = None,
):
    return QwenArchitectureAdapter._build_config(
        model,
        architecture_family="qwen_decoder",
        expansion_size=expansion_size,
        lam=lam,
        alpha_e=alpha_e,
        alpha_h=alpha_h,
        attention_profile=attention_profile,
        qk_scale_range=qk_scale_range,
        beta=beta,
        gamma=gamma,
        ffn_scale_range=ffn_scale_range,
        seed=seed,
        device=device,
        dtype=dtype,
        normalization_mode=normalization_mode,
        adapted_layers=adapted_layers,
    )
