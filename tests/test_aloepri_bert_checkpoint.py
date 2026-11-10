from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch


def test_bert_checkpoint_api_is_exported() -> None:
    checkpoints = importlib.import_module("src.aloepri.checkpoints")
    aloepri = importlib.import_module("src.aloepri")

    assert hasattr(checkpoints, "export_bert_paper_checkpoint")
    assert hasattr(aloepri, "export_bert_paper_checkpoint")


def test_build_stage_j_bert_paper_consistent_target_metadata(tmp_path) -> None:
    module = importlib.import_module("src.aloepri.checkpoints.bert")
    payload = module.build_stage_j_bert_paper_consistent_target(tmp_path)

    assert payload["stage"] == "J"
    assert payload["model_line"] == "bert_base_uncased"
    assert payload["goal"] == "bert_local_research_with_custom_layernorm_bridge"
    assert payload["standard_graph_required"] is False
    assert payload["requires_custom_runtime"] is True
    assert payload["standard_visible_keys_required"] is True
    assert payload["canonical_candidate_dir"] == str(tmp_path)


class _DummyTokenizer:
    vocab_size = 4
    all_special_ids = [0]


class _DummyEmbeddings:
    def __init__(self, hidden_size: int) -> None:
        self.word_embeddings = SimpleNamespace(weight=torch.zeros(4, hidden_size, dtype=torch.float32))
        self.position_embeddings = SimpleNamespace(weight=torch.zeros(4, hidden_size, dtype=torch.float32))
        self.token_type_embeddings = SimpleNamespace(weight=torch.zeros(2, hidden_size, dtype=torch.float32))


class _DummyModel:
    def __init__(self, hidden_size: int = 4, num_hidden_layers: int = 2) -> None:
        self.config = SimpleNamespace(
            model_type="bert",
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=2,
            intermediate_size=8,
            vocab_size=4,
            max_position_embeddings=4,
            type_vocab_size=2,
        )
        self.embeddings = _DummyEmbeddings(hidden_size)
        self.encoder = SimpleNamespace(layer=[object() for _ in range(num_hidden_layers)])
        self.pooler = None

    def eval(self):
        return self

    def to(self, _device: str):
        return self

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings


class _DummyTransform:
    def __init__(self, hidden_size: int) -> None:
        self.dim = hidden_size
        self.perm = torch.arange(hidden_size, dtype=torch.long)
        self.inv_perm = torch.arange(hidden_size, dtype=torch.long)
        self.signs = torch.ones(hidden_size, dtype=torch.float32)
        self.global_scale = 1.0

    def key(self, dtype=torch.float32):
        return torch.eye(self.dim, dtype=dtype)


def _install_minimal_export_stubs(monkeypatch: pytest.MonkeyPatch, tmp_path) -> dict[str, object]:
    impl = importlib.import_module("src.aloepri._implementation.bert.bert_paper_checkpoint")

    captured: dict[str, object] = {}

    monkeypatch.setattr(impl, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: _DummyTokenizer()))
    monkeypatch.setattr(impl, "AutoModel", SimpleNamespace(from_pretrained=lambda *args, **kwargs: _DummyModel()))
    monkeypatch.setattr(impl, "set_global_seed", lambda seed: None)
    monkeypatch.setattr(impl, "resolve_torch_dtype", lambda dtype: torch.float32)
    monkeypatch.setattr(
        impl,
        "BertArchitectureAdapter",
        SimpleNamespace(
            from_model=lambda model: SimpleNamespace(
                hidden_size=model.config.hidden_size,
                num_hidden_layers=model.config.num_hidden_layers,
                describe=lambda: {"model_type": "bert", "num_hidden_layers": model.config.num_hidden_layers},
            )
        ),
    )
    monkeypatch.setattr(
        impl,
        "_prepare_stage_a_bert_model",
        lambda baseline_model, tokenizer, seed: (
            baseline_model,
            torch.arange(baseline_model.config.vocab_size, dtype=torch.long),
            torch.arange(baseline_model.config.vocab_size, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(impl, "build_square_monomial_transform", lambda hidden_size, seed, global_scale: _DummyTransform(hidden_size))
    monkeypatch.setattr(impl, "ordinary_token_ids", lambda tokenizer: torch.tensor([1, 2, 3], dtype=torch.long))
    monkeypatch.setattr(
        impl,
        "obfuscate_embedding_with_square_transform_stage_a",
        lambda weight, transform, alpha_e, seed, movable_ids: weight.clone(),
    )
    monkeypatch.setattr(impl, "adapt_input_linear_weight_for_square", lambda weight, transform: weight.clone())
    monkeypatch.setattr(impl, "adapt_output_linear_weight_for_square", lambda weight, transform: weight.clone())
    monkeypatch.setattr(impl, "adapt_output_bias_for_square", lambda bias, transform: bias.clone())
    monkeypatch.setattr(impl, "_adapt_square_bert_layer_inplace", lambda layer, transform: None)
    monkeypatch.setattr(
        impl,
        "summarize_token_partitions",
        lambda tokenizer, model_vocab_size, perm_vocab: {
            "perm_is_valid": True,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "model_vocab_size": model_vocab_size,
            "special_token_count": len(tokenizer.all_special_ids),
            "special_ids_fixed": True,
            "tail_row_count": 0,
            "tail_rows_fixed": True,
        },
    )

    def _fake_export_stage_i(export_dir, *, tokenizer, stage_a_model, perm_vocab, inv_perm_vocab, metadata):
        export_dir.mkdir(parents=True, exist_ok=True)
        captured["metadata"] = metadata
        (export_dir / "manifest.json").write_text(json.dumps({"format": "stage_i_vllm_v1"}), encoding="utf-8")
        return {
            "export_dir": export_dir,
            "server_dir": export_dir / "server",
            "client_secret_path": export_dir / "client" / "client_secret.pt",
        }

    monkeypatch.setattr(impl, "export_stage_i_vllm_checkpoint", _fake_export_stage_i)
    return captured


def test_export_stage_j_bert_paper_checkpoint_rejects_partial_layer_adaptation(monkeypatch, tmp_path) -> None:
    impl = importlib.import_module("src.aloepri._implementation.bert.bert_paper_checkpoint")
    _install_minimal_export_stubs(monkeypatch, tmp_path)

    with pytest.raises(NotImplementedError, match="Partial-layer BERT adaptation"):
        impl.export_stage_j_bert_paper_consistent_checkpoint(tmp_path, seed=1, alpha_e=0.0, num_layers=1)


def test_export_stage_j_bert_paper_checkpoint_keeps_square_transform_only_in_client_metadata(monkeypatch, tmp_path) -> None:
    impl = importlib.import_module("src.aloepri._implementation.bert.bert_paper_checkpoint")
    captured = _install_minimal_export_stubs(monkeypatch, tmp_path)

    impl.export_stage_j_bert_paper_consistent_checkpoint(tmp_path, seed=1, alpha_e=0.0)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "square_transform" not in manifest
    assert "square_transform" in captured["metadata"]


def test_export_bert_paper_checkpoint_defaults_alpha_e_to_zero(monkeypatch, tmp_path) -> None:
    module = importlib.import_module("src.aloepri.checkpoints.bert")
    captured: dict[str, object] = {}

    def _fake_export(*args, **kwargs):
        captured["alpha_e"] = kwargs["alpha_e"]
        return {}

    monkeypatch.setattr(module, "export_stage_j_bert_paper_consistent_checkpoint", _fake_export)

    module.export_bert_paper_checkpoint(tmp_path, seed=1)

    assert captured["alpha_e"] == 0.0
