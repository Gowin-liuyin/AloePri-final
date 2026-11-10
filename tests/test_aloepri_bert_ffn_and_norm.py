from __future__ import annotations

import importlib
from types import SimpleNamespace

from torch import nn


def test_keymat_fused_bert_layernorm_warning_docstring_present() -> None:
    module = importlib.import_module("src.aloepri._implementation.bert.bert_layernorm_bridge")

    docstring = module.KeyMatFusedBertLayerNorm.__doc__

    assert docstring is not None
    assert "WARNING: This fused implementation computes mean/variance in obfuscated space." in docstring
    assert "LayerNorm is NOT equivariant to signed permutations." in docstring
    assert "Bridge version (KeyMatBertLayerNormBridge) is the ONLY correct path." in docstring
    assert "Use this fused class only for approximate benchmarks." in docstring


def test_layers_norm_exports_bridge_only_for_bert_layernorm() -> None:
    module = importlib.import_module("src.aloepri.layers.norm")

    assert hasattr(module, "KeyMatBertLayerNormBridge")
    assert hasattr(module, "build_keymat_bert_layernorm_bridge")
    assert hasattr(module, "wrap_bert_norm")
    assert not hasattr(module, "KeyMatFusedBertLayerNorm")
    assert not hasattr(module, "build_keymat_fused_bert_layernorm")
    assert "KeyMatFusedBertLayerNorm" not in module.__all__
    assert "build_keymat_fused_bert_layernorm" not in module.__all__


def test_resolve_activation_uses_hidden_act_from_config() -> None:
    module = importlib.import_module("src.aloepri._implementation.bert.bert_ffn_obfuscation")

    relu = module._resolve_activation(SimpleNamespace(config=SimpleNamespace(hidden_act="relu")))
    gelu_new = module._resolve_activation(SimpleNamespace(config=SimpleNamespace(hidden_act="gelu_new")))
    gelu = module._resolve_activation(SimpleNamespace(config=SimpleNamespace(hidden_act="gelu")))
    fallback = module._resolve_activation(SimpleNamespace())

    assert isinstance(relu, nn.ReLU)
    assert isinstance(gelu_new, nn.GELU)
    assert gelu_new.approximate == "tanh"
    assert isinstance(gelu, nn.GELU)
    assert isinstance(fallback, nn.GELU)
