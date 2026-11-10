"""Regression checks for the 1B checkpoint's tied head and scaled RoPE."""
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from tests.test_round3_models import llama_setup
from src.aloepri._implementation.export.standard_shape_obfuscation import build_stage_j_square_model
from src.aloepri._implementation.export.expanded_llama_checkpoint import build_expanded_llama_model
from src.llama_local_dev import tokenize_llama_prompt


def llama3_setup():
    tokenizer, _ = llama_setup()
    torch.manual_seed(321)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=1, head_dim=4,
        max_position_embeddings=16384, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=[2, 31], pad_token_id=2,
        rope_parameters=dict(rope_type="llama3", rope_theta=500000., factor=32.,
                             high_freq_factor=4., low_freq_factor=1.,
                             original_max_position_embeddings=8192))).eval()
    return tokenizer, model


def test_tied_llama3_square_export_reload_and_cached_generation(tmp_path):
    tokenizer, baseline = llama3_setup()
    assert baseline.lm_head.weight is baseline.model.embed_tokens.weight
    model, perm, inverse, _ = build_stage_j_square_model(
        baseline, tokenizer, adapted_layers=[0, 1], seed=321)
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    ids = torch.tensor([[3, 7, 9]])
    with torch.inference_mode():
        expected = baseline(ids).logits
        actual = loaded(perm[ids]).logits[..., perm]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        plain = baseline.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False)
        from src.aloepri.private_inference import generate_mapped_ids
        generated = generate_mapped_ids(loaded, ids, attention_mask=torch.ones_like(ids),
                                       perm_vocab=perm, inv_perm_vocab=inverse, max_new_tokens=4)
    assert torch.equal(generated, plain[:, ids.shape[1]:])
    assert loaded.config.tie_word_embeddings is False
    assert loaded.generation_config.eos_token_id == [2, 31]
    assert loaded.config.rope_parameters == baseline.config.rope_parameters


def test_expansion_preserves_llama3_frequencies_and_scalar_reference(tmp_path):
    from copy import deepcopy
    from torch import nn
    tokenizer, baseline = llama3_setup()
    expanded, perm, _, key, details = build_expanded_llama_model(baseline, tokenizer, expansion_size=2, seed=321)
    torch.testing.assert_close(expanded.model.rotary_emb.inv_freq, baseline.model.rotary_emb.inv_freq, atol=0, rtol=0)
    reference = deepcopy(baseline)
    class ScalarNorm(nn.Module):
        def __init__(self, norm):
            super().__init__()
            self.weight = nn.Parameter(norm.weight.detach().clone())
        def forward(self, value):
            return (value * details["coefficient"]
                    * torch.rsqrt((value @ key.key).square().mean(-1, keepdim=True) + details["scalar_epsilon"])
                    * self.weight)
    for layer in reference.model.layers:
        layer.input_layernorm = ScalarNorm(layer.input_layernorm)
        layer.post_attention_layernorm = ScalarNorm(layer.post_attention_layernorm)
    reference.model.norm = ScalarNorm(reference.model.norm)
    ids = torch.tensor([[3, 7, 9]])
    positions = torch.tensor([[8191, 8192, 8193]])
    expanded.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(perm[ids], position_ids=positions).logits[..., perm],
                                   reference(ids, position_ids=positions).logits, atol=3e-6, rtol=3e-5)
    with pytest.raises(ValueError, match="without frequency-block permutation"):
        build_expanded_llama_model(baseline, tokenizer, expansion_size=2,
                                   attention_profile="rqk_hqk_block_taukv_taugroup")


def test_chat_prompt_has_exactly_one_bos():
    class Tokenizer:
        chat_template = "llama3"
        def apply_chat_template(self, *args, **kwargs):
            return "<bos> user question assistant"
        def __call__(self, text, *, return_tensors, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": torch.tensor([[1, 3, 4]])}
    assert tokenize_llama_prompt(Tokenizer(), "question")["input_ids"].tolist() == [[1, 3, 4]]


def test_inplace_bf16_export_matches_copy_without_changing_default_copy_behavior():
    from copy import deepcopy
    tokenizer, baseline = llama3_setup()
    baseline = baseline.to(torch.bfloat16)
    original = {name: value.clone() for name, value in baseline.state_dict().items()}
    copied, *expected_keys = build_stage_j_square_model(baseline, tokenizer, adapted_layers=[0, 1], seed=321)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    consumed = deepcopy(baseline)
    result, *actual_keys = build_stage_j_square_model(consumed, tokenizer, adapted_layers=[0, 1], seed=321, copy_model=False)
    assert result is consumed
    for name, value in result.state_dict().items():
        torch.testing.assert_close(value, copied.state_dict()[name], rtol=0, atol=0)
    for expected, actual in zip(expected_keys[:2], actual_keys[:2]):
        assert torch.equal(expected, actual)


def test_large_bf16_column_transform_matches_independent_rows():
    from src.aloepri._implementation.export.square_transform import build_square_monomial_transform
    from src.aloepri._implementation.export.standard_shape_obfuscation import adapt_input_linear_weight_for_square
    torch.manual_seed(321)
    weight = torch.randn(8193, 2048, dtype=torch.bfloat16)
    key = build_square_monomial_transform(2048, 321)
    result = adapt_input_linear_weight_for_square(weight, key)
    for row in (0, 2047, 2048, 8192):
        expected = weight[row].float() @ key.key()
        torch.testing.assert_close(result[row].float(), expected, rtol=0, atol=0)


def test_source_sharding_preserves_weights_and_single_file(tmp_path):
    from safetensors.torch import save_file
    from scripts.setup.prepare_llama_1b import shard_checkpoint
    from src.aloepri._implementation.export.tensor_io import read_safetensors_state
    values = {"embedding": torch.randn(9, 8, dtype=torch.bfloat16),
              "linear": torch.randn(8, 8), "norm": torch.randn(8)}
    save_file(values, str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    source_bytes = (tmp_path / "model.safetensors").read_bytes()
    shard_checkpoint(tmp_path, max_bytes=128)
    actual = read_safetensors_state(tmp_path)
    assert set(actual) == set(values)
    for name in values:
        torch.testing.assert_close(actual[name], values[name], rtol=0, atol=0)
    assert (tmp_path / "source_single_file/model.safetensors").read_bytes() == source_bytes
    assert not (tmp_path / "model.safetensors").exists()


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_streaming_loader_matches_native_reload_and_preserves_llama3(tmp_path, tied, dtype):
    from src.model_loader import load_local_model_streaming
    _, source = llama3_setup()
    if not tied:
        source.lm_head = torch.nn.Linear(16, 32, bias=False)
        source.config.tie_word_embeddings = False
    source.to(torch.bfloat16).save_pretrained(tmp_path, max_shard_size="2KB")
    target_dtype = getattr(torch, dtype)
    native = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=target_dtype).eval()
    streamed = load_local_model_streaming(tmp_path, dtype=dtype)
    assert (streamed.lm_head.weight is streamed.model.embed_tokens.weight) == tied
    assert streamed.generation_config.eos_token_id == [2, 31]
    assert not any(value.device.type == "meta" for value in streamed.buffers())
    for name, value in native.state_dict().items():
        torch.testing.assert_close(streamed.state_dict()[name], value, rtol=0, atol=0)
    ids = torch.tensor([[3, 7, 9]])
    with torch.inference_mode():
        torch.testing.assert_close(streamed(ids, position_ids=torch.tensor([[8191, 8192, 8193]])).logits,
                                   native(ids, position_ids=torch.tensor([[8191, 8192, 8193]])).logits,
                                   rtol=0, atol=0)
