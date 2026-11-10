from __future__ import annotations

from typing import Any

from src.aloepri.adapters._base import ArchitectureAdapter


class LlamaArchitectureAdapter(ArchitectureAdapter):
    """
    Structural adapter for Llama-style decoder-only causal LM models.

    Unlike the Qwen adapter, this adapter intentionally validates that the
    checkpoint identifies itself as the Llama family, so that similarly-shaped
    decoder architectures do not get misclassified.
    """

    @classmethod
    def from_model(cls, model: Any) -> "LlamaArchitectureAdapter":
        config = cls._require_attr(model, "config", "Llama")
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type != "llama":
            raise ValueError(f"Expected model_type='llama', got {model_type!r}")

        cls._require_attr(model, "config.hidden_size", "Llama")
        cls._require_attr(model, "config.num_hidden_layers", "Llama")
        cls._require_attr(model, "config.num_attention_heads", "Llama")
        cls._require_attr(model, "model.embed_tokens", "Llama")
        cls._require_attr(model, "model.layers", "Llama")
        cls._require_attr(model, "model.norm", "Llama")
        cls._require_attr(model, "lm_head", "Llama")
        layers = cls._require_attr(model, "model.layers", "Llama")
        if len(layers) == 0:
            raise ValueError("Llama-architecture model must have at least one decoder layer.")
        layer0 = layers[0]
        cls._require_attr(layer0, "self_attn.q_proj", "Llama")
        cls._require_attr(layer0, "self_attn.k_proj", "Llama")
        cls._require_attr(layer0, "self_attn.v_proj", "Llama")
        cls._require_attr(layer0, "self_attn.o_proj", "Llama")
        cls._require_attr(layer0, "input_layernorm", "Llama")
        cls._require_attr(layer0, "post_attention_layernorm", "Llama")
        cls._require_attr(layer0, "mlp.gate_proj", "Llama")
        cls._require_attr(layer0, "mlp.up_proj", "Llama")
        cls._require_attr(layer0, "mlp.down_proj", "Llama")
        return cls(model=model)

    @property
    def layers(self):
        return self.model.model.layers

    @property
    def head_dim(self) -> int:
        explicit = getattr(self.config, "head_dim", None)
        if explicit is not None:
            return int(explicit)
        return super().head_dim

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description["is_tied_word_embeddings"] = bool(
            self.model.get_input_embeddings().weight.data_ptr() == self.model.get_output_embeddings().weight.data_ptr()
        )
        return description


def is_llama_compatible_model(model: Any) -> bool:
    return LlamaArchitectureAdapter.is_compatible_model(model)


def build_llama_config(
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
    return LlamaArchitectureAdapter._build_config(
        model,
        architecture_family="llama_decoder",
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
