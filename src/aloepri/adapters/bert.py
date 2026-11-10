from __future__ import annotations

from typing import Any

from src.aloepri.adapters._base import ArchitectureAdapter


class BertArchitectureAdapter(ArchitectureAdapter):
    """
    Structural adapter for BERT-style encoder-only models.

    This adapter validates by module layout and accepts models whose
    `model_type` contains `bert`, as long as they expose the expected encoder
    architecture contract.
    """

    @classmethod
    def from_model(cls, model: Any) -> "BertArchitectureAdapter":
        config = cls._require_attr(model, "config", "BERT")
        model_type = str(getattr(config, "model_type", "")).lower()
        if "bert" not in model_type:
            raise ValueError(f"Expected a BERT-family model_type, got {model_type!r}")

        cls._require_attr(model, "config.hidden_size", "BERT")
        cls._require_attr(model, "config.num_hidden_layers", "BERT")
        cls._require_attr(model, "config.num_attention_heads", "BERT")
        cls._require_attr(model, "config.intermediate_size", "BERT")
        cls._require_attr(model, "config.vocab_size", "BERT")
        cls._require_attr(model, "config.max_position_embeddings", "BERT")
        cls._require_attr(model, "config.type_vocab_size", "BERT")
        cls._require_attr(model, "embeddings.word_embeddings", "BERT")
        cls._require_attr(model, "encoder.layer", "BERT")
        pooler = getattr(model, "pooler", None)
        if pooler is not None:
            cls._require_attr(model, "pooler.dense", "BERT")
        layers = cls._require_attr(model, "encoder.layer", "BERT")
        if len(layers) == 0:
            raise ValueError("BERT-architecture model must have at least one encoder layer.")
        layer0 = layers[0]
        cls._require_attr(layer0, "attention.self.query", "BERT")
        cls._require_attr(layer0, "attention.self.key", "BERT")
        cls._require_attr(layer0, "attention.self.value", "BERT")
        cls._require_attr(layer0, "attention.output.dense", "BERT")
        cls._require_attr(layer0, "attention.output.LayerNorm", "BERT")
        cls._require_attr(layer0, "intermediate.dense", "BERT")
        cls._require_attr(layer0, "output.dense", "BERT")
        cls._require_attr(layer0, "output.LayerNorm", "BERT")
        return cls(model=model)

    @property
    def layers(self):
        return self.model.encoder.layer

    @property
    def num_key_value_heads(self) -> int:
        return self.num_attention_heads

    @property
    def intermediate_size(self) -> int:
        return int(self.config.intermediate_size)

    @property
    def vocab_size(self) -> int:
        return int(self.config.vocab_size)

    @property
    def max_position_embeddings(self) -> int:
        return int(self.config.max_position_embeddings)

    @property
    def type_vocab_size(self) -> int:
        return int(self.config.type_vocab_size)

    @property
    def rope_theta(self) -> float | None:
        return None

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "intermediate_size": self.intermediate_size,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
                "type_vocab_size": self.type_vocab_size,
                "has_pooler": getattr(self.model, "pooler", None) is not None,
                "has_token_type_embeddings": getattr(
                    getattr(self.model, "embeddings", object()), "token_type_embeddings", None
                )
                is not None,
                "has_position_embeddings": getattr(
                    getattr(self.model, "embeddings", object()), "position_embeddings", None
                )
                is not None,
                "norm_type": "layernorm",
                "ffn_activation": "gelu",
                "attention_type": "standard_mha",
            }
        )
        return description


def is_bert_compatible_model(model: Any) -> bool:
    return BertArchitectureAdapter.is_compatible_model(model)


def build_bert_config(
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
    return BertArchitectureAdapter._build_config(
        model,
        architecture_family="bert_encoder",
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
