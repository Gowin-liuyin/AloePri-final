"""Regression checks for the October audit: algebra, runtime boundaries and key consistency."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

import src.attention_keys as attention_keys
from src.aloepri import AloePriConfig, AloePriEngine
from src.aloepri.adapters.qwen import QwenArchitectureAdapter
from src.aloepri._implementation.runtime.keymat_ffn_fusion import KeyMatFusedQwen2MLP
from src.aloepri._implementation.runtime.keymat_norm_fusion import KeyMatFusedRMSNorm
from src.aloepri._implementation.runtime.token_space_runtime import TraceRecorder
from src.hidden_keys import build_hidden_transform
from src.keymat import build_keymat_transform, sample_compatible_inverse
from src.obfuscate_ffn import fuse_ffn_intermediate_weights


class Tokenizer:
    vocab_size = 32
    all_special_ids = [0, 1, 2]


def tiny_model():
    torch.manual_seed(91)
    config = Qwen2Config(
        vocab_size=35, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


def engine_for(model, layers=None, **overrides):
    return AloePriEngine.from_model(
        model, Tokenizer(), expansion_size=4, seed=81,
        alpha_e=0.0, alpha_h=0.0, attention_profile="simplified",
        adapted_layers=layers, **overrides,
    )


@pytest.mark.parametrize("layers", [None, [0], []])
def test_qwen_full_prefix_and_empty_prefix_have_consistent_logits(layers):
    model = tiny_model()
    engine = engine_for(model, layers)
    obfuscated = engine.obfuscate_model(model).eval()  # No recorder is required.
    tokens = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]])
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
        actual = obfuscated(engine.keys.token_perm[tokens], use_cache=False).logits
    restored = actual.index_select(-1, engine.keys.token_perm)
    assert torch.allclose(restored, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(obfuscated.perm_vocab, engine.keys.token_perm)
    assert torch.equal(obfuscated.inv_perm_vocab, engine.keys.inv_token_perm)
    assert torch.equal(engine.keys.token_perm[:3], torch.arange(3))
    assert torch.equal(engine.keys.token_perm[32:], torch.arange(32, 35))
    assert len(engine.keys.ffn_transforms) == len(engine.config.adapted_layers)
    assert len(engine.keys.kappas) == len(engine.config.adapted_layers)
    assert obfuscated.aloepri_runtime_metadata["shared_residual_key"] is True
    assert obfuscated.aloepri_runtime_metadata["normalization_mode"] == "exact_metric_reference"


def test_qwen_cache_matches_full_sequence_and_original():
    model = tiny_model()
    engine = engine_for(model)
    obfuscated = engine.obfuscate_model(model).eval()
    tokens = torch.tensor([[3, 4, 5, 6]])
    encoded = engine.keys.token_perm[tokens]
    with torch.no_grad():
        prefix = obfuscated(encoded[:, :3], use_cache=True)
        step = obfuscated(encoded[:, 3:], past_key_values=prefix.past_key_values, use_cache=True).logits
        full = obfuscated(encoded, use_cache=False).logits[:, -1:]
        plain = model(tokens, use_cache=False).logits[:, -1:]
    assert torch.allclose(step, full, atol=2e-5, rtol=2e-5)
    assert torch.allclose(step.index_select(-1, engine.keys.token_perm), plain, atol=2e-5, rtol=2e-5)


def test_every_attention_and_ffn_residual_branch_uses_the_same_output_key():
    model = tiny_model()
    engine = engine_for(model)
    obfuscated = engine.obfuscate_model(model).eval()
    plain_values, encoded_values, handles = {}, {}, []
    def capture(target, name):
        def hook(_module, _inputs, output):
            target[name] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook
    for index, (plain_layer, encoded_layer) in enumerate(zip(model.model.layers, obfuscated.model.layers)):
        for branch in ("self_attn", "mlp"):
            name = f"{index}_{branch}"
            handles.append(getattr(plain_layer, branch).register_forward_hook(capture(plain_values, name)))
            handles.append(getattr(encoded_layer, branch).register_forward_hook(capture(encoded_values, name)))
    try:
        tokens = torch.tensor([[3, 4, 5, 6]])
        with torch.no_grad():
            model(tokens, use_cache=False)
            obfuscated(engine.keys.token_perm[tokens], use_cache=False)
        for name in plain_values:
            assert torch.allclose(encoded_values[name], plain_values[name] @ engine.keys.keymat_transform.key, atol=2e-5, rtol=2e-5)
    finally:
        for handle in handles:
            handle.remove()


def test_engine_scalar_mode_is_explicit_and_uses_scaled_epsilon():
    model = tiny_model()
    engine = engine_for(model, normalization_mode="scalar_rms_approximation")
    obfuscated = engine.obfuscate_model(model).eval()
    with torch.no_grad():
        logits = obfuscated(engine.keys.token_perm[torch.tensor([[3, 4, 5]])], use_cache=False).logits
    assert torch.isfinite(logits).all()
    assert obfuscated.aloepri_runtime_metadata["normalization_mode"] == "scalar_rms_approximation"
    norm = obfuscated.model.layers[0].input_layernorm.inner
    assert norm.scalar_epsilon == pytest.approx(norm.scalar_coefficient**2 * norm.variance_epsilon)
    assert norm.metric_matrix is None


@pytest.mark.parametrize("layers", [[1], [0, 2], [0, 0], [-1], [0, 1, 2]])
def test_nonprefix_duplicate_and_out_of_range_layers_are_rejected(layers):
    with pytest.raises(ValueError, match="contiguous prefix"):
        engine_for(tiny_model(), layers)


def test_adapter_uses_explicit_head_dimension():
    fake = SimpleNamespace(config=SimpleNamespace(hidden_size=16, num_attention_heads=4, head_dim=6))
    assert QwenArchitectureAdapter(fake).head_dim == 6


def test_adapter_reads_modern_rope_parameters_instead_of_assuming_10000():
    fake = SimpleNamespace(config=SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}))
    assert QwenArchitectureAdapter(fake).rope_theta == 1000000.0


def test_extra_qk_normalization_is_rejected_instead_of_silently_skipped():
    model = tiny_model()
    model.model.layers[0].self_attn.q_norm = nn.Identity()
    with pytest.raises(ValueError, match="q_norm/k_norm"):
        engine_for(model).obfuscate_model(model)


def test_non_qwen_engine_runtime_is_rejected_before_using_qwen_modules():
    model = tiny_model()
    engine = engine_for(model)
    engine.config = replace(engine.config, architecture_family="bert_encoder")
    with pytest.raises(ValueError, match="Qwen decoders only"):
        engine.obfuscate_model(model)


def test_default_seeds_are_distinct_and_explicit_seed_is_repeatable():
    model = tiny_model()
    config = AloePriConfig.from_model(model, expansion_size=4, alpha_e=0, alpha_h=0)
    first, second = AloePriEngine(config, Tokenizer()), AloePriEngine(config, Tokenizer())
    assert first.config.seed != second.config.seed
    assert first.keys.seed_source == second.keys.seed_source == "system_random"
    fixed = replace(config, seed=17)
    one, two = AloePriEngine(fixed, Tokenizer()), AloePriEngine(fixed, Tokenizer())
    assert torch.equal(one.keys.token_perm, two.keys.token_perm)
    assert torch.equal(one.keys.keymat_transform.key, two.keys.keymat_transform.key)


def test_three_cycle_qk_permutation_preserves_same_position_scores(monkeypatch):
    permutation = torch.tensor([1, 2, 0])
    matrix = torch.zeros(6, 6)
    for old, new in enumerate(permutation.tolist()):
        matrix[new, old] = matrix[new + 3, old + 3] = 1
    monkeypatch.setattr(attention_keys, "generate_block_perm", lambda **_: (permutation, matrix))
    config = attention_keys.build_attention_complex_config(
        profile="rqk_hqk_block", head_dim=6, num_kv_heads=1, num_groups=1, seed=3,
    )
    query, key = torch.randn(5, 6), torch.randn(5, 6)
    actual = (query @ config.intra_head.q_matrix) @ (key @ config.intra_head.k_matrix).T
    assert torch.allclose(actual, query @ key.T, atol=2e-5)


def test_default_dynamic_windows_actually_permute_some_blocks():
    changed = 0
    for seed in range(20):
        permutation, _ = attention_keys.generate_block_perm(64, 8, 1000.0, 10000.0, seed)
        assert torch.equal(permutation.sort().values, torch.arange(64))
        changed += int(not torch.equal(permutation, torch.arange(64)))
    assert changed >= 15


def test_ffn_offline_formula_with_bias_and_nontrivial_scaling():
    torch.manual_seed(10)
    gate, up, down = nn.Linear(4, 6), nn.Linear(4, 6), nn.Linear(6, 4)
    transform = build_hidden_transform(torch.tensor([2, 0, 4, 1, 5, 3]), torch.tensor([.5, 2., .8, 1.2, 3., .9]))
    g, u, d, gb, ub = fuse_ffn_intermediate_weights(
        gate.weight, up.weight, down.weight, transform, gate.bias, up.bias,
    )
    x = torch.randn(7, 4)
    expected = down(F.silu(gate(x)) * up(x))
    actual = F.linear(F.silu(F.linear(x, g, gb)) * F.linear(x, u, ub), d, down.bias)
    assert torch.allclose(actual, expected, atol=1e-6)
    order, scale = transform.inv_perm, transform.scale
    assert torch.equal(g, gate.weight[order])
    assert torch.equal(u, up.weight[order] * scale[:, None])
    assert torch.equal(d, down.weight[:, order] / scale[None])


def test_ffn_static_weights_depend_on_intermediate_keys_and_keep_shared_residual_key():
    torch.manual_seed(11)
    mlp = SimpleNamespace(gate_proj=nn.Linear(4, 6), up_proj=nn.Linear(4, 6), down_proj=nn.Linear(6, 4), act_fn=F.silu)
    key = build_keymat_transform(4, 2, .1, 5)
    first = build_hidden_transform(torch.tensor([2, 0, 4, 1, 5, 3]), torch.ones(6))
    second = build_hidden_transform(torch.arange(6), torch.full((6,), 2.0))
    a = KeyMatFusedQwen2MLP(mlp, key, torch.ones(4), first, seed=2)
    b = KeyMatFusedQwen2MLP(mlp, key, torch.ones(4), second, seed=2)
    assert not torch.equal(a.gate_weight, b.gate_weight)
    assert not torch.equal(a.up_weight, b.up_weight)
    assert not torch.equal(a.down_weight, b.down_weight)
    assert not torch.equal(a.gate_input_inverse, a.up_input_inverse)
    assert torch.allclose(key.key @ a.gate_input_inverse, torch.eye(4), atol=1e-5)
    x = torch.randn(3, 4)
    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x)) @ key.key
    assert torch.allclose(a(x @ key.key), expected, atol=2e-5, rtol=2e-5)


def test_attention_uses_distinct_compatible_input_inverses_and_nonidentity_uvo():
    model = tiny_model()
    engine = engine_for(model)
    obfuscated = engine.obfuscate_model(model).eval()
    p = engine.keys.keymat_transform.key
    for layer in obfuscated.model.layers:
        attention = layer.self_attn.attn
        inverses = [attention.query_input_inverse, attention.key_input_inverse, attention.value_input_inverse]
        for inverse in inverses:
            assert torch.allclose(p @ inverse, torch.eye(16), atol=1e-5)
        assert not torch.equal(inverses[0], inverses[1])
        metadata = attention.static_metadata
        assert not torch.equal(metadata.v_dense, torch.eye(metadata.v_dense.shape[0]))
        assert torch.allclose(metadata.v_dense @ metadata.v_dense_inverse, torch.eye(8), atol=1e-5)


def test_scalar_norm_restores_dimension_factor_and_epsilon_for_scaled_isometry():
    base = build_keymat_transform(4, 2, .0, 7)
    p = torch.cat([torch.eye(4) * 3, torch.zeros(4, 4)], dim=1)
    q = torch.cat([torch.eye(4) / 3, torch.zeros(4, 4)], dim=0)
    transform = replace(base, key=p, inverse=q)
    norm = SimpleNamespace(variance_epsilon=.1, weight=torch.ones(4))
    scalar = KeyMatFusedRMSNorm(norm, transform, kappa=3, mode="scalar_rms_approximation")
    x = torch.tensor([[.01, .02, -.03, .04], [1., 2., 3., 4.]])
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + .1)
    assert torch.allclose(scalar(x @ p), expected @ p, atol=1e-6)
    assert scalar.scalar_epsilon == pytest.approx((3 * (4 / 8)**.5)**2 * .1)


def test_metric_reference_matches_selected_inverse_without_claiming_scalar_equivalence():
    transform = build_keymat_transform(4, 2, .2, 9)
    norm = SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(4))
    reference = KeyMatFusedRMSNorm(norm, transform, kappa=999, mode="exact_metric_reference")
    x = torch.randn(5, 8)
    restored = x @ transform.inverse
    expected = x * torch.rsqrt(restored.square().mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(reference(x), expected, atol=2e-5, rtol=2e-5)


def test_compatible_inverse_samples_share_the_same_left_key():
    transform = build_keymat_transform(8, 4, .2, 4)
    first, second = sample_compatible_inverse(transform, 101), sample_compatible_inverse(transform, 102)
    assert not torch.equal(first, second)
    assert torch.allclose(transform.key @ first, torch.eye(8), atol=1e-5)
    assert torch.allclose(transform.key @ second, torch.eye(8), atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_engine_offline_fusion_restores_source_device_and_dtype(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime is unavailable; this is not a CUDA validation pass.")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("This GPU does not support bfloat16.")
    model = tiny_model().to(device=device, dtype=dtype)
    engine = engine_for(model)
    obfuscated = engine.obfuscate_model(model).eval()
    tokens = torch.tensor([[3, 4, 5, 6]], device=device)
    encoded = engine.keys.token_perm[tokens.cpu()].to(device)
    with torch.no_grad():
        expected = model(tokens, use_cache=False).logits
        actual = obfuscated(encoded, use_cache=False).logits
    restored = actual.index_select(-1, engine.keys.token_perm.to(device))
    tolerance = {torch.float32: 2e-5, torch.float16: .005, torch.bfloat16: .03}[dtype]
    assert actual.device.type == device
    assert actual.dtype == dtype
    assert obfuscated.model.embed_tokens.wrapper.weight.dtype == dtype
    assert obfuscated.model.layers[0].self_attn.attn.q_weight.dtype == dtype
    norm = obfuscated.model.layers[0].input_layernorm.inner
    assert norm.metric_matrix.dtype == torch.float32
    assert norm.metric_matrix.device.type == device
    assert torch.isfinite(restored).all()
    assert torch.allclose(restored, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_trace_keeps_restored_dimension_instead_of_being_overwritten(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware/runtime is unavailable.")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = tiny_model().to(device=device, dtype=dtype)
    engine = engine_for(model)
    recorder = TraceRecorder()
    obfuscated = engine.obfuscate_model(model, recorder=recorder).eval()
    encoded = engine.keys.token_perm[torch.tensor([[3, 4, 5]])].to(device)
    with torch.no_grad():
        obfuscated(encoded, use_cache=False)
    for layer in range(2):
        assert recorder.tensors[f"layer_{layer}_attn_out"].shape == (1, 3, 16)
        assert recorder.tensors[f"layer_{layer}_mlp_out"].shape == (1, 3, 16)
        assert torch.isfinite(recorder.tensors[f"layer_{layer}_attn_out"]).all()
