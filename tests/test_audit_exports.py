"""覆盖真实权重、分片、秘密落盘和错误验收；不下载预训练大模型。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

from src.aloepri._implementation.analysis.equivalence import equivalence_status
from src.aloepri._implementation.analysis.standard_weight_proof import build_stage_j_standard_weight_proof
from src.aloepri._implementation.export.checkpoint_io import export_stage_i_vllm_checkpoint, load_stage_i_hf_bundle
from src.aloepri._implementation.export.release_bundle_impl import StageKProfile, export_stage_k_release
from src.aloepri._implementation.export.standard_shape_obfuscation import build_stage_j_square_model
from src.aloepri._implementation.export.tensor_io import read_safetensors_state, safetensors_shapes
from src.aloepri.checkpoints.qwen import export_qwen_paper_checkpoint
from src.aloepri.private_inference import generate_mapped_ids
from src.transforms import map_input_ids, restore_logits


def _tokenizer():
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"word{i}": i + 3 for i in range(30)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")


def _qwen_tokenizer():
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
    vocab.update({char: index + 3 for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    return Qwen2Tokenizer(vocab=vocab, merges=[], unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]")


def _model(family="llama", tied=False, vocab_size=33):
    config_type, model_type = (LlamaConfig, LlamaForCausalLM) if family == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    config = config_type(vocab_size=vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=tied,
                         bos_token_id=1, eos_token_id=2, pad_token_id=2)
    return model_type(config).eval()


@pytest.mark.parametrize("family", ["llama", "qwen2"])
@pytest.mark.parametrize("tied", [False, True])
def test_square_head_both_weight_sharing_modes_preserve_logits(family, tied):
    torch.manual_seed(7)
    baseline = _model(family, tied)
    stage, perm, _, _ = build_stage_j_square_model(baseline, _tokenizer(), adapted_layers=[0, 1], seed=9)
    ids = torch.tensor([[3, 5, 7, 4]])
    with torch.no_grad():
        expected = baseline(ids).logits
        actual = restore_logits(stage(map_input_ids(ids, perm)).logits, perm)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert stage.lm_head.weight.data_ptr() != stage.model.embed_tokens.weight.data_ptr()


def test_untied_head_noise_is_applied_and_partial_indices_are_validated():
    baseline, tokenizer = _model(), _tokenizer()
    a = build_stage_j_square_model(baseline, tokenizer, adapted_layers=[0, 1], seed=9, alpha_h=0.0)[0]
    b = build_stage_j_square_model(baseline, tokenizer, adapted_layers=[0, 1], seed=9, alpha_h=0.4)[0]
    assert not torch.equal(a.lm_head.weight, b.lm_head.weight)
    for layers in ([], [1], [0, 0], [0, 2]):
        with pytest.raises(ValueError, match="contiguous prefix"):
            build_stage_j_square_model(baseline, tokenizer, adapted_layers=layers, seed=9)


def test_standard_structure_rejects_two_keys_and_wrong_shapes(tmp_path):
    model = _model()
    model.config.save_pretrained(tmp_path)
    save_file({"model.embed_tokens.weight": torch.zeros(33, 16), "lm_head.weight": torch.zeros(33, 16)}, tmp_path / "model.safetensors")
    proof = build_stage_j_standard_weight_proof(tmp_path)
    assert not proof["is_standard_weight_export"] and proof["missing_standard_keys"]
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    state["model.layers.0.mlp.down_proj.weight"] = torch.zeros(17, 32)
    save_file(state, tmp_path / "model.safetensors")
    proof = build_stage_j_standard_weight_proof(tmp_path)
    assert "model.layers.0.mlp.down_proj.weight" in proof["shape_mismatches"]


def test_sharded_hf_checkpoint_structure_and_real_loading(tmp_path):
    model = _model()
    model.save_pretrained(tmp_path, max_shard_size="2KB")
    assert (tmp_path / "model.safetensors.index.json").exists()
    shapes = safetensors_shapes(tmp_path)
    state = read_safetensors_state(str(tmp_path))
    assert set(shapes) == set(state) == set(model.state_dict())
    assert build_stage_j_standard_weight_proof(tmp_path)["is_standard_weight_export"]
    loaded = LlamaForCausalLM.from_pretrained(tmp_path, local_files_only=True).eval()
    torch.testing.assert_close(loaded(torch.tensor([[3, 4]])).logits, model(torch.tensor([[3, 4]])).logits)


def test_shard_index_rejects_wrong_key_and_external_path(tmp_path):
    save_file({"x": torch.zeros(2)}, tmp_path / "a.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"not_x": "a.safetensors"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        read_safetensors_state(tmp_path)
    index.write_text(json.dumps({"weight_map": {"x": "../a.safetensors"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid shard"):
        safetensors_shapes(tmp_path)


def test_private_metadata_only_in_client_file(tmp_path):
    model, tokenizer = _model(), _tokenizer()
    perm = torch.arange(33)
    exported = export_stage_i_vllm_checkpoint(tmp_path / "source", tokenizer=tokenizer, stage_a_model=model,
        perm_vocab=perm, inv_perm_vocab=perm, metadata={"seed": 999, "square_transform": {"inverse": [1]},
        "alpha_e": 0.0, "model_line": "llama", "requires_custom_runtime": False})
    public = json.loads((tmp_path / "source" / "stage_i_metadata.json").read_text())
    assert "seed" not in public and "square_transform" not in public
    secret = torch.load(exported["client_secret_path"], weights_only=True)
    assert secret["metadata"]["seed"] == 999
    assert load_stage_i_hf_bundle(exported["server_dir"])["metadata"]["seed"] == 999
    profile = StageKProfile("default", str(tmp_path / "source"), "test", "local")
    catalog = export_stage_k_release(tmp_path / "release", profiles=[profile], reference_profile="default")
    catalog_text = json.dumps(catalog)
    assert '"seed"' not in catalog_text and "square_transform" not in catalog_text
    assert not list((tmp_path / "release" / "profiles" / "default" / "server").rglob("*secret*"))


@pytest.mark.parametrize("existing_file", ["client/client_secret.pt", "manifest.json", "server/old_weights.bin"])
def test_low_level_export_does_not_overwrite_existing_client_or_metadata(tmp_path, existing_file):
    destination = tmp_path / "existing_export"
    sentinel = destination / existing_file
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"existing-owner-data-must-survive")
    before = sorted(str(path.relative_to(destination)) for path in destination.rglob("*"))
    with pytest.raises(ValueError, match="Export directory must be empty"):
        export_stage_i_vllm_checkpoint(destination, tokenizer=_tokenizer(), stage_a_model=_model(),
                                      perm_vocab=torch.arange(33), inv_perm_vocab=torch.arange(33), metadata={})
    assert sentinel.read_bytes() == b"existing-owner-data-must-survive"
    assert sorted(str(path.relative_to(destination)) for path in destination.rglob("*")) == before


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "C:evil", "a\\b", ""])
def test_release_rejects_invalid_profile_names_before_writing(tmp_path, name):
    with pytest.raises(ValueError, match="Profile names"):
        export_stage_k_release(tmp_path / "release", profiles=[StageKProfile(name, "missing", "", "")])
    assert not (tmp_path / "release").exists()


def test_release_rejects_duplicate_names_and_overlapping_paths(tmp_path):
    profile = StageKProfile("default", str(tmp_path), "", "")
    with pytest.raises(ValueError, match="unique"):
        export_stage_k_release(tmp_path / "release", profiles=[profile, profile], reference_profile="default")
    with pytest.raises(ValueError, match="overlap"):
        export_stage_k_release(tmp_path / "release", profiles=[profile], reference_profile="default")
    with pytest.raises(ValueError, match="unique"):
        export_stage_k_release(tmp_path / "case_release", profiles=[profile, StageKProfile("DEFAULT", "missing", "", "")])
    for name in ("CON", "nul", "COM1", "Lpt9"):
        with pytest.raises(ValueError, match="reserved device names"):
            export_stage_k_release(tmp_path / "device_release", profiles=[StageKProfile(name, "missing", "", "")])


def test_equivalence_rejects_partial_success_nonfinite_and_status_override():
    valid = {"prompt_count": 5, "generated_ids_exact_match_rate": 1.0, "generated_text_exact_match_rate": 1.0,
             "max_restored_full_logits_max_abs_error": 1e-7, "baseline_has_nan_or_inf": False,
             "stage_k_has_nan_or_inf": False}
    assert equivalence_status(valid)["status"] == "pass"
    for changes in ({"generated_ids_exact_match_rate": 0.2}, {"stage_k_has_nan_or_inf": True},
                    {"max_restored_full_logits_max_abs_error": float("nan")},
                    {"max_restored_full_logits_max_abs_error": float("inf")},
                    {"max_restored_full_logits_max_abs_error": 0.1}, {"prompt_count": 0}):
        assert equivalence_status({**valid, **changes, "status": "pass"})["status"] == "fail"


def test_debug_formats_need_explicit_research_opt_in(tmp_path):
    from src.aloepri._implementation.export.deployable_artifact import save_stage_h_artifact
    from src.aloepri._implementation.export.pretrained_deployable import export_stage_h_pretrained
    perm = torch.arange(3)
    with pytest.raises(ValueError, match="local research"):
        save_stage_h_artifact(tmp_path, stage_model=None, perm_vocab=perm, inv_perm_vocab=perm, metadata={})
    with pytest.raises(ValueError, match="local research"):
        export_stage_h_pretrained(tmp_path, tokenizer=None, baseline_model=None, stage_model=None,
                                 perm_vocab=perm, inv_perm_vocab=perm, metadata={})
    assert not list(tmp_path.iterdir())


def test_generation_maps_once_and_restores_only_returned_ids():
    captured = {}
    class Model:
        generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=2)
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[4, 2]])], dim=1)
    perm = torch.tensor([0, 1, 2, 4, 3])
    plain = generate_mapped_ids(Model(), torch.tensor([[3]]), attention_mask=torch.ones(1, 1),
                               perm_vocab=perm, inv_perm_vocab=perm, max_new_tokens=8)
    assert captured["input_ids"].tolist() == [[4]] and captured["use_cache"]
    assert captured["eos_token_id"] == [2] and plain.tolist() == [[3, 2]]
    with pytest.raises(ValueError, match="nonempty token IDs"):
        generate_mapped_ids(Model(), torch.empty((1, 0), dtype=torch.long), attention_mask=None,
                            perm_vocab=perm, inv_perm_vocab=perm, max_new_tokens=8)


def test_raw_qwen_entry_explicit_variant_and_partial_llama_export_rejected(tmp_path):
    original = tmp_path / "original"
    tokenizer = _qwen_tokenizer()
    _model("qwen2", vocab_size=len(tokenizer)).save_pretrained(original)
    tokenizer.save_pretrained(original)
    paths = export_qwen_paper_checkpoint(tmp_path / "export", model_dir=original, seed=123)
    manifest = json.loads(paths["manifest_path"].read_text())
    assert manifest["implementation_variant"] == "square_monomial"
    assert manifest["paper_algorithm1_reproduced"] is False
    assert build_stage_j_standard_weight_proof(paths["server_dir"])["is_standard_weight_export"]
    loaded = load_stage_i_hf_bundle(paths["server_dir"])
    assert loaded["tokenizer"]("word2 word4")["input_ids"] == tokenizer("word2 word4")["input_ids"]
    from src.aloepri._implementation.export.standard_square_checkpoint import export_standard_square_checkpoint
    with pytest.raises(ValueError, match="all decoder layers"):
        export_standard_square_checkpoint(tmp_path / "partial", model_dir=original, expected_model_type="qwen2", num_layers=1)


def test_bert_real_disk_export_keeps_recovery_transform_in_client_and_rejects_standard_release(tmp_path):
    from transformers import BertConfig, BertModel
    from src.aloepri._implementation.bert.bert_paper_checkpoint import export_stage_j_bert_paper_consistent_checkpoint
    original = tmp_path / "bert_original"
    config = BertConfig(vocab_size=33, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=4, max_position_embeddings=32)
    BertModel(config).save_pretrained(original)
    _tokenizer().save_pretrained(original)
    paths = export_stage_j_bert_paper_consistent_checkpoint(tmp_path / "bert_export", model_dir=original,
                                                          seed=123, alpha_e=0.0, dtype="float32")
    public = json.loads((tmp_path / "bert_export" / "stage_i_metadata.json").read_text())
    assert "square_transform" not in public and "seed" not in public
    secret = torch.load(paths["client_secret_path"], weights_only=True)
    assert "square_transform" in secret["metadata"]
    profile = StageKProfile("default", str(tmp_path / "bert_export"), "BERT", "local research")
    with pytest.raises(ValueError, match="BERT remains local research"):
        export_stage_k_release(tmp_path / "bert_release", profiles=[profile], reference_profile="default")
    # 丢失说明文件也不能绕过架构限制；无需信任调用者填写的运行模块声明。
    (tmp_path / "bert_export" / "manifest.json").unlink()
    with pytest.raises(ValueError, match="BERT remains local research"):
        export_stage_k_release(tmp_path / "bert_without_manifest", profiles=[profile], reference_profile="default")


def test_real_generation_reuses_cache_stops_at_mapped_eos_and_rejects_nan():
    model = _model("qwen2")
    model.generation_config.eos_token_id = None
    ids = torch.tensor([[3, 4, 5]])
    perm = torch.arange(33)
    lengths = []
    def record_input(_module, _args, kwargs):
        lengths.append(int(kwargs["input_ids"].shape[1]))
    handle = model.register_forward_pre_hook(record_input, with_kwargs=True)
    generated = generate_mapped_ids(model, ids, attention_mask=torch.ones_like(ids),
                                   perm_vocab=perm, inv_perm_vocab=perm, max_new_tokens=4)
    handle.remove()
    assert generated.shape[1] == 4 and lengths == [3, 1, 1, 1]
    with torch.no_grad():
        expected_eos = int(model(ids).logits[0, -1].argmax())
    model.generation_config.eos_token_id = expected_eos
    result = generate_mapped_ids(model, ids, attention_mask=torch.ones_like(ids),
                                perm_vocab=perm, inv_perm_vocab=perm, max_new_tokens=8)
    assert result.tolist() == [[expected_eos]]
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite raw logits"):
        generate_mapped_ids(model, ids, attention_mask=torch.ones_like(ids),
                            perm_vocab=perm, inv_perm_vocab=perm, max_new_tokens=2)
