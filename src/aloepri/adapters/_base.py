from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArchitectureAdapter:
    model: Any

    @classmethod
    def _require_attr(cls, obj: Any, path: str, family_name: str):
        current = obj
        for part in path.split("."):
            if not hasattr(current, part):
                raise ValueError(f"Missing required {family_name}-architecture attribute: {path}")
            current = getattr(current, part)
        return current

    @property
    def config(self):
        return self.model.config

    @property
    def hidden_size(self) -> int:
        return int(self.config.hidden_size)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.config.num_hidden_layers)

    @property
    def num_attention_heads(self) -> int:
        return int(self.config.num_attention_heads)

    @property
    def num_key_value_heads(self) -> int:
        return int(getattr(self.config, "num_key_value_heads", self.config.num_attention_heads))

    @property
    def head_dim(self) -> int:
        explicit = getattr(self.config, "head_dim", None)
        if explicit is not None:
            return int(explicit)
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by attention heads when head_dim is absent.")
        return self.hidden_size // self.num_attention_heads

    @property
    def rope_theta(self) -> float | None:
        parameters = getattr(self.config, "rope_parameters", None)
        if isinstance(parameters, dict) and "rope_theta" in parameters:
            return float(parameters["rope_theta"])
        rope_theta = getattr(self.config, "rope_theta", None)
        if rope_theta is None:
            return 10000.0
        return float(rope_theta)

    @property
    def model_type(self) -> str:
        return str(getattr(self.config, "model_type", ""))

    def describe(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rope_theta": self.rope_theta,
        }

    @classmethod
    def is_compatible_model(cls, model: Any) -> bool:
        try:
            cls.from_model(model)
            return True
        except Exception:
            return False

    @classmethod
    def _build_config(cls, model: Any, architecture_family: str, **overrides):
        from src.aloepri.config import AloePriConfig

        adapter = cls.from_model(model)
        adapted_layers = overrides.pop("adapted_layers", None)
        get_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_embeddings):
            embeddings = get_embeddings()
        else:
            # Structural BERT adapters do not require the Hugging Face getter.
            embeddings = getattr(getattr(model, "embeddings", None), "word_embeddings", None)
        embedding_weight = getattr(embeddings, "weight", None)
        model_vocab_size = int(embedding_weight.shape[0]) if hasattr(embedding_weight, "shape") else getattr(adapter.config, "vocab_size", None)
        return AloePriConfig(
            hidden_size=adapter.hidden_size,
            num_hidden_layers=adapter.num_hidden_layers,
            num_attention_heads=adapter.num_attention_heads,
            num_key_value_heads=adapter.num_key_value_heads,
            head_dim=adapter.head_dim,
            rope_theta=adapter.rope_theta,
            adapted_layers=list(range(adapter.num_hidden_layers)) if adapted_layers is None else adapted_layers,
            architecture_family=architecture_family,
            model_type=adapter.model_type,
            model_vocab_size=model_vocab_size,
            **overrides,
        )
