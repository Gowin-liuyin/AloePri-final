from types import SimpleNamespace

from src.aloepri.adapters import (
    BertArchitectureAdapter,
    build_architecture_config,
    build_bert_config,
    get_architecture_adapter,
    is_bert_compatible_model,
    is_llama_compatible_model,
    is_qwen_compatible_model,
)
from src.aloepri.config import AloePriConfig


class _Linear:
    def __init__(self):
        self.weight = object()
        self.bias = object()


class _LayerNorm:
    def __init__(self):
        self.weight = object()
        self.bias = object()


class _Embedding:
    def __init__(self):
        self.weight = object()


class _BertLayer:
    def __init__(self):
        self.attention = SimpleNamespace(
            self=SimpleNamespace(
                query=_Linear(),
                key=_Linear(),
                value=_Linear(),
            ),
            output=SimpleNamespace(
                dense=_Linear(),
                LayerNorm=_LayerNorm(),
            ),
        )
        self.intermediate = SimpleNamespace(dense=_Linear())
        self.output = SimpleNamespace(
            dense=_Linear(),
            LayerNorm=_LayerNorm(),
        )


class _BertModel:
    def __init__(self):
        self.config = SimpleNamespace(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            vocab_size=30522,
            max_position_embeddings=512,
            type_vocab_size=2,
            model_type="bert",
        )
        self.embeddings = SimpleNamespace(
            word_embeddings=_Embedding(),
            position_embeddings=_Embedding(),
            token_type_embeddings=_Embedding(),
        )
        self.encoder = SimpleNamespace(
            layer=[_BertLayer()],
        )
        self.pooler = SimpleNamespace(dense=_Linear())


def test_bert_adapter_accepts_structurally_compatible_model() -> None:
    model = _BertModel()
    adapter = BertArchitectureAdapter.from_model(model)
    assert adapter.hidden_size == 768
    assert adapter.num_hidden_layers == 12
    assert adapter.num_attention_heads == 12
    assert adapter.num_key_value_heads == 12
    assert adapter.head_dim == 64
    assert adapter.intermediate_size == 3072
    assert adapter.vocab_size == 30522
    assert adapter.model_type == "bert"
    assert adapter.max_position_embeddings == 512
    assert adapter.type_vocab_size == 2
    assert is_bert_compatible_model(model) is True
    assert is_qwen_compatible_model(model) is False
    assert is_llama_compatible_model(model) is False


def test_build_bert_config_uses_model_structure() -> None:
    model = _BertModel()
    config = build_bert_config(model, adapted_layers=[0, 1], alpha_e=0.0, alpha_h=0.0)
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.head_dim == 64
    assert config.adapted_layers == [0, 1]
    assert config.architecture_family == "bert_encoder"
    assert config.model_type == "bert"


def test_architecture_dispatch_selects_bert() -> None:
    model = _BertModel()
    adapter = get_architecture_adapter(model)
    config = build_architecture_config(model, alpha_e=0.0, alpha_h=0.0)
    assert isinstance(adapter, BertArchitectureAdapter)
    assert config.architecture_family == "bert_encoder"


def test_aloepri_config_accepts_bert() -> None:
    model = _BertModel()
    config = AloePriConfig.from_model(model, alpha_e=0.0, alpha_h=0.0)
    assert config.architecture_family == "bert_encoder"


def test_bert_adapter_rejects_non_bert_model_type() -> None:
    model = _BertModel()
    model.config.model_type = "qwen2"
    assert is_bert_compatible_model(model) is False


def test_bert_adapter_describe_includes_encoder_metadata() -> None:
    model = _BertModel()
    description = BertArchitectureAdapter.from_model(model).describe()
    assert description["model_type"] == "bert"
    assert description["hidden_size"] == 768
    assert description["num_hidden_layers"] == 12
    assert description["num_attention_heads"] == 12
    assert description["num_key_value_heads"] == 12
    assert description["head_dim"] == 64
    assert description["intermediate_size"] == 3072
    assert description["vocab_size"] == 30522
    assert description["max_position_embeddings"] == 512
    assert description["type_vocab_size"] == 2
    assert description["has_pooler"] is True
    assert description["has_token_type_embeddings"] is True
    assert description["has_position_embeddings"] is True
    assert description["norm_type"] == "layernorm"
    assert description["ffn_activation"] == "gelu"
    assert description["attention_type"] == "standard_mha"


def test_bert_adapter_accepts_missing_optional_pooler() -> None:
    model = _BertModel()
    delattr(model, "pooler")
    adapter = BertArchitectureAdapter.from_model(model)
    assert adapter.describe()["has_pooler"] is False


def test_bert_adapter_exposes_shared_compatibility_classmethod() -> None:
    model = _BertModel()
    assert BertArchitectureAdapter.is_compatible_model(model) is True
