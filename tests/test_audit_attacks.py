"""用本地小矩阵检验攻击评测边界；这些测试不代表论文的大模型攻击率。"""

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.security_qwen import ima, isa, sda, tfma, vma
from src.security_qwen.protocol import public_candidate_ids


class TinyTokenizer:
    vocab_size = 32
    all_special_ids = []

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3, 4]}


@pytest.mark.parametrize("setting", ["zero_knowledge", "domain_aware", "distribution_aware"])
def test_reference_and_private_texts_are_disjoint(setting):
    corpora = tfma.build_frequency_corpora(setting, seed=7)
    assert not set(corpora["reference_texts"]) & set(corpora["private_texts"])
    assert corpora == tfma.build_frequency_corpora(setting, seed=7)
    assert len(corpora["private_texts"]) == 180


def test_tfma_uses_proportions_not_raw_counts():
    args = dict(observed_counter=Counter({10: 50, 11: 10}), candidate_plain_ids=[1, 2], observed_token_ids=[10, 11])
    first = tfma._candidate_rank_lists(reference_counter=Counter({1: 10, 2: 2}), **args)
    repeated = tfma._candidate_rank_lists(reference_counter=Counter({1: 100, 2: 20}), **args)
    assert first == repeated == {10: [1, 2], 11: [2, 1]}


@pytest.mark.parametrize("topk", [1, 5, 10, 100])
def test_frequency_metrics_keep_named_k_and_full_occurrence_denominator(topk):
    metrics = tfma._evaluate_rank_lists(
        candidate_lists={0: [0], 1: [0]}, inv_perm_vocab=torch.arange(3),
        sensitive_plain_ids=set(), topk_values=(1, min(topk, 10), min(topk, 100)),
        observed_counter=Counter({0: 9, 1: 1, 2: 10}),
    )
    assert metrics["token_top1_recovery_rate"] == 0.5
    assert metrics["token_occurrence_top1_recovery_rate"] == 0.45
    assert metrics["evaluated_token_occurrence_count"] == 20
    assert metrics["token_top100_recovery_rate"] == 0.5


def test_unknown_predictions_stay_unknown_and_bleu_never_joins_sentences():
    assert sda._reconstruct_sequences([[1, 2, 3, 4]], {1: [7]}) == [[7, -1, -1, -1]]
    assert sda._token_bleu4([[1, 2, 3, 4]], [[1, 2, 3, 4]]) == 1.0
    # 将前后两句对换，拼接评分可偶然计入跨句四元组；逐句计分不允许。
    score = sda._token_bleu4([[1, 2], [3, 4]], [[3, 4], [1, 2]])
    assert score < 1e-6
    with pytest.raises(ValueError, match="counts"):
        sda._token_bleu4([[1]], [])


@pytest.mark.parametrize("topk", [1, 5, 10, 100])
@pytest.mark.parametrize("module", [tfma, sda])
def test_frequency_runners_do_not_receive_truth_candidates_or_fill_answers(monkeypatch, topk, module):
    tokenizer = TinyTokenizer()
    monkeypatch.setattr(module, "load_frequency_bundle", lambda **kwargs: {
        "tokenizer": tokenizer, "perm_vocab": torch.arange(32), "inv_perm_vocab": torch.arange(32),
        "resolved_target": {"stage": "J", "artifact_dir": "fixture", "profile": None, "variant": "fixture"},
    })
    monkeypatch.setattr(module, "build_frequency_corpora", lambda *args, **kwargs: {
        "reference_texts": ["public"], "private_texts": ["private"],
    })
    monkeypatch.setattr(module, "_tokenize_texts", lambda tokenizer, texts: [[1] * 8] if texts == ["public"] else [[2, 2, 2, 3]])
    runner = module.run_tfma_baseline if module is tfma else module.run_sda_baseline
    result = runner(target_name="fixture", candidate_pool_size=1, topk=topk)
    assert result["metrics"]["candidate_coverage_rate"] == 0.0
    assert result["metrics"]["token_top100_recovery_rate"] == 0.0
    if module is sda:
        assert result["artifacts"]["reconstructed_sequences_sample"] == [[1, 1, 1, -1]]
        assert result["config"]["paper_attack_reproduced"] is False


def test_candidate_subsets_do_not_follow_test_answers_or_inverse_permutation():
    tokenizer = TinyTokenizer()
    args = dict(tokenizer=tokenizer, train_size=4, val_size=2, candidate_pool_size=1, seed=3,
                sensitive_plain_ids_override=torch.empty(0, dtype=torch.long))
    first = ima._sample_ima_splits(test_size=2, **args)
    second = ima._sample_ima_splits(test_size=4, **args)
    assert torch.equal(first["candidate_plain_ids"], second["candidate_plain_ids"])
    expected = public_candidate_ids(tokenizer, size=1, seed=3)
    assert torch.equal(first["candidate_plain_ids"], expected)
    assert not set(first["train_plain_ids"].tolist()) & set(first["test_plain_ids"].tolist())
    vma_args = dict(tokenizer=tokenizer, perm_vocab=torch.arange(32), eval_size=4,
                    candidate_pool_size=1, seed=3, sensitive_plain_ids_override=torch.empty(0, dtype=torch.long))
    _, candidates_a, _ = vma._sample_eval_and_candidate_sets(inv_perm_vocab=torch.arange(32), **vma_args)
    _, candidates_b, _ = vma._sample_eval_and_candidate_sets(inv_perm_vocab=torch.arange(31, -1, -1), **vma_args)
    assert torch.equal(candidates_a, candidates_b)
    assert torch.equal(candidates_a, expected)
    first_isa = isa._sample_plain_sequences(tokenizer=tokenizer, sequence_length=2, train_sequences=2,
        val_sequences=1, test_sequences=1, candidate_pool_size=1, seed=3)
    second_isa = isa._sample_plain_sequences(tokenizer=tokenizer, sequence_length=2, train_sequences=2,
        val_sequences=1, test_sequences=5, candidate_pool_size=1, seed=3)
    assert torch.equal(first_isa["candidate_plain_ids"], second_isa["candidate_plain_ids"])


@pytest.mark.parametrize("runner", [ima.run_ima_baseline, ima.run_ima_paper_like, isa.run_isa_baseline])
def test_simulated_known_pairs_require_explicit_opt_in_before_model_loading(runner):
    with pytest.raises(ValueError, match="allow_simulated_known_pairs"):
        runner(target_name="does_not_exist")


@pytest.mark.parametrize("runner", [tfma.run_tfma_baseline, sda.run_sda_baseline, vma.run_vma_baseline])
def test_invalid_topk_is_rejected_before_model_loading(runner):
    with pytest.raises(ValueError, match="topk"):
        runner(target_name="does_not_exist", topk=0)


@pytest.mark.parametrize("topk", [1, 5, 10, 100])
def test_inversion_top10_is_not_relabelled_top1(topk):
    embedding = torch.eye(12)
    predicted = torch.tensor([[1.0, 0.8] + [0.0] * 10])
    result = ima._evaluate_inversion_predictions(predicted_embeddings=predicted, true_plain_ids=torch.tensor([1]),
        candidate_plain_ids=torch.arange(12), baseline_embed=embedding, topk=topk)
    assert result["token_top1_recovery_rate"] == 0.0
    assert result["token_top10_recovery_rate"] == 1.0
    ranked = ima._rank_candidate_plain_ids(predicted_embeddings=predicted, candidate_plain_ids=torch.arange(12),
        baseline_embed=embedding, topk=topk, chunk_size=2)
    assert ranked.shape == (1, min(topk, 12))


def test_qwen_inverter_updates_head_width_and_runs_backward(monkeypatch):
    monkeypatch.setattr(ima.AutoConfig, "from_pretrained", lambda *a, **kw: Qwen2Config(
        hidden_size=32, head_dim=8, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=32))
    config = ima.build_paper_like_inverter_config(observed_hidden_size=16, vocab_size=32)
    assert config.head_dim == 2 and config.num_attention_heads == 8
    model = ima._PaperLikeIMAInverter(backbone_config=config, target_embedding_dim=12)
    output = model(torch.randn(2, 3, 16))
    assert output.shape == (2, 3, 12)
    output.square().mean().backward()
    assert model.output_proj.weight.grad is not None
    assert model.backbone.get_input_embeddings().weight.grad is None


def test_vma_norm_fusion_and_positive_scalar_handling():
    embed = torch.tensor([[1., 2., 3., 4.], [2., 0., 1., 3.]])
    gate = torch.tensor([[1., 0., 2., 0.], [0., 1., 0., 2.], [2., 1., 1., 0.]])
    norm = torch.tensor([2., 3., 4., 5.])
    fixed = vma._fuse_public_reference_norms({"model.layers.0.post_attention_layernorm.weight": norm}, {"layer_0_gate": gate})
    assert torch.equal(embed @ fixed["layer_0_gate"].T, (embed * norm) @ gate.T)
    assert torch.allclose(vma._sorted_quantile_features(embed @ fixed["layer_0_gate"].T, 3),
                          vma._sorted_quantile_features(2.5 * embed @ fixed["layer_0_gate"].T, 3), atol=1e-6)
    assert torch.isfinite(vma._rowwise_standardize(torch.ones(2, 1))).all()


def _save_shards(directory: Path, state):
    directory.mkdir()
    keys = list(state)
    parts = [keys[::2], keys[1::2]]
    weight_map = {}
    for i, names in enumerate(parts):
        filename = f"model-{i}.safetensors"
        save_file({name: state[name].clone().contiguous() for name in names}, str(directory / filename))
        weight_map.update({name: filename for name in names})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")


def test_vma_and_ima_load_shards_and_use_real_untied_head(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(4)
    embed = torch.randn(32, 4, generator=generator)
    head = torch.randn(32, 4, generator=generator) * 2
    gate = torch.randn(3, 4, generator=generator)
    norm = torch.tensor([1., 2., 3., 4.])
    baseline = tmp_path / "baseline"
    server = tmp_path / "server"
    _save_shards(baseline, {"model.embed_tokens.weight": embed, "lm_head.weight": head,
        "model.layers.0.mlp.gate_proj.weight": gate, "model.layers.0.post_attention_layernorm.weight": norm})
    _save_shards(server, {"buffer::stage_a_model.model.embed_tokens.weight": embed,
        "buffer::stage_a_model.lm_head.weight": head,
        "buffer::stage_a_model.model.layers.0.mlp.gate_weight": gate * norm})
    secret = tmp_path / "client_secret.pt"
    torch.save({"perm_vocab": torch.arange(32), "inv_perm_vocab": torch.arange(32)}, secret)
    target = {"name": "fixture", "stage": "J", "artifact_dir": str(tmp_path), "profile": None,
              "variant": "fixture", "server_dir": str(server), "metadata_path": None,
              "manifest_path": None, "client_secret_path": str(secret), "resolved_root_dir": str(tmp_path)}
    resolved = SimpleNamespace(**target, to_dict=lambda: dict(target))
    monkeypatch.setattr(vma, "resolve_security_target", lambda *a: resolved)
    monkeypatch.setattr(ima, "resolve_security_target", lambda *a: resolved)
    weights, observed, projections, _, aux = vma.load_vma_weight_sources(target_name="fixture", baseline_model_dir=str(baseline))
    assert torch.equal(weights["head"], head)
    assert not torch.equal(weights["head"], embed)
    assert torch.equal(observed["head"], head)
    assert torch.equal(projections["layer_0_gate"], gate * norm)
    assert aux["reference_norm_fused"] is True
    original_embed, observed_embed, _ = ima.load_ima_embedding_sources(target_name="fixture", baseline_model_dir=str(baseline))
    assert torch.equal(original_embed, embed) and torch.equal(observed_embed, embed)
    monkeypatch.setattr(vma.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    result = vma.run_vma_baseline(target_name="fixture", baseline_model_dir=str(baseline), eval_size=4,
        candidate_pool_size=32, feature_bins=3, topk=1, sensitive_plain_ids_override=torch.empty(0, dtype=torch.long))
    assert result["metrics"]["token_top1_recovery_rate"] == 1.0
    assert result["config"]["paper_attack_reproduced"] is False


def test_public_inversion_windows_are_unique_across_splits(tmp_path):
    class RepeatedTokenizer(TinyTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 1, 2, 3, 4, 5, 6]}
    path = tmp_path / "public.txt"
    path.write_text("public", encoding="utf-8")
    splits = ima._collect_public_token_windows(tokenizer=RepeatedTokenizer(), corpus_paths=[str(path)], sequence_length=2,
        train_sequence_count=1, val_sequence_count=1, test_sequence_count=1, seed=1)
    windows = [tuple(row[0].tolist()) for row in [splits["train_plain_ids"], splits["val_plain_ids"], splits["test_plain_ids"]]]
    assert len(set(windows)) == 3


def _tiny_embedding_bundle():
    embedding = torch.randn(32, 16, generator=torch.Generator().manual_seed(12))
    target = {"name": "fixture", "stage": "J", "artifact_dir": "fixture", "profile": None, "variant": "fixture"}
    return embedding, embedding.clone(), {"resolved_target": target, "perm_vocab": torch.arange(32), "inv_perm_vocab": torch.arange(32)}


@pytest.mark.parametrize("topk", [1, 5, 10, 100])
def test_ridge_ima_and_isa_runners_label_known_pair_experiments(monkeypatch, topk):
    torch.set_num_threads(1)
    baseline, observed, aux = _tiny_embedding_bundle()
    monkeypatch.setattr(ima, "load_ima_embedding_sources", lambda **kw: (baseline, observed, aux))
    monkeypatch.setattr(ima.AutoTokenizer, "from_pretrained", lambda *a, **kw: TinyTokenizer())
    result = ima.run_ima_baseline(target_name="fixture", train_size=4, val_size=2, test_size=2,
        candidate_pool_size=2, topk=topk, sensitive_plain_ids_override=torch.empty(0, dtype=torch.long),
        allow_simulated_known_pairs=True)
    assert result["config"]["deployment_only"] is False
    assert result["config"]["training_pair_source"] == "client_secret_simulator"
    assert result["config"]["candidate_pool_size"] == 2
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)).eval()
    monkeypatch.setattr(isa, "load_isa_model_bundle", lambda **kw: {
        "baseline_embed": baseline, "perm_vocab": torch.arange(32), "tokenizer": TinyTokenizer(),
        "model": model, "extractor": isa._get_standard_observable, "resolved_target": aux["resolved_target"],
    })
    result = isa.run_isa_baseline(target_name="fixture", observable_layer=0, train_sequences=3,
        val_sequences=2, test_sequences=2, sequence_length=2, candidate_pool_size=2, topk=topk,
        allow_simulated_known_pairs=True)
    assert result["config"]["deployment_only"] is False
    assert result["config"]["paper_attack_reproduced"] is False
    assert result["metrics"]["test_sample_count"] == 4
    assert 0.0 <= result["metrics"]["candidate_coverage_rate"] <= 1.0


def test_paper_like_ima_trains_on_local_unique_windows(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    baseline, observed, aux = _tiny_embedding_bundle()
    monkeypatch.setattr(ima, "load_ima_embedding_sources", lambda **kw: (baseline, observed, aux))
    class PublicTokenizer(TinyTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(1, 9))}
    monkeypatch.setattr(ima.AutoTokenizer, "from_pretrained", lambda *a, **kw: PublicTokenizer())
    monkeypatch.setattr(ima.AutoConfig, "from_pretrained", lambda *a, **kw: Qwen2Config(vocab_size=32,
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
    path = tmp_path / "public.txt"
    path.write_text("local public data", encoding="utf-8")
    result = ima.run_ima_paper_like(target_name="fixture", sequence_length=2, train_sequence_count=2,
        val_sequence_count=1, test_sequence_count=1, epochs=1, batch_size=1, topk=1, device="cpu",
        public_corpus_paths=[str(path)], allow_simulated_known_pairs=True)
    assert result["config"]["best_epoch"] == 1
    assert result["config"]["attack_head_dim"] == 2
    assert result["config"]["paper_attack_reproduced"] is False
    assert result["config"]["candidate_pool_strategy"] == "full_movable_vocab"
    assert result["metrics"]["evaluated_token_count"] == 2


def test_single_ridge_training_pair_remains_finite():
    x = torch.tensor([[1., 2.]])
    y = torch.tensor([[3., 4., 5.]])
    model = ima._fit_ridge_regressor(x, y, ridge_alpha=1.0)
    assert torch.isfinite(ima._predict_ridge(model, x)).all()
    assert torch.allclose(ima._predict_ridge(model, x), y)
