from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from src.aloepri._implementation.export.tensor_io import safetensors_shapes


REQUIRED_STANDARD_KEYS = [
    "model.embed_tokens.weight",
    "lm_head.weight",
]


def _moe_serialized_shapes(expected,actual,model_type):
    """HF 5.19在内存中打包专家，save_pretrained会写回逐专家标准格式。

    只在某层没有打包张量、却存在标准逐专家名称时改写该层的预期；
    缺失、混合重复、额外专家与错误尺寸仍由完整集合比较拒绝。
    """
    if model_type not in ("qwen2_moe","mixtral"):return expected
    result=dict(expected)
    for key,shape in expected.items():
        if not key.endswith(".mlp.experts.gate_up_proj") or key in actual:continue
        base=key.removesuffix(".experts.gate_up_proj")
        saved_base=base if model_type=="qwen2_moe" else base.removesuffix(".mlp")+".block_sparse_moe"
        legacy_prefix=saved_base+".experts."
        if not any(name.startswith(legacy_prefix) for name in actual):continue
        count,twice_width,hidden=shape
        width=twice_width//2
        result.pop(key);result.pop(base+".experts.down_proj")
        for expert in range(count):
            for kind,dims in (("gate",(width,hidden)),("up",(width,hidden)),("down",(hidden,width))):
                name=kind+"_proj" if model_type=="qwen2_moe" else {"gate":"w1","up":"w3","down":"w2"}[kind]
                result[f"{saved_base}.experts.{expert}.{name}.weight"]=dims
        if model_type=="mixtral":
            result[saved_base+".gate.weight"]=result.pop(base+".gate.weight")
    return result


def build_stage_j_standard_weight_proof(server_dir: str | Path) -> dict[str, Any]:
    server_dir = Path(server_dir)
    result: dict[str, Any] = {
        "is_standard_weight_export": False,
        "validation_scope": "config_and_all_tensor_names_and_shapes_only",
        "actual_model_load_tested": False, "numerical_equivalence_tested": False, "privacy_tested": False,
        "layout": "invalid", "standard_key_count": 0, "buffered_key_count": 0,
        "missing_standard_keys": [], "unexpected_keys": [], "shape_mismatches": {},
        "sample_standard_keys": [], "sample_buffered_keys": [],
    }
    try:
        if not (server_dir / "config.json").is_file():
            raise FileNotFoundError("config.json is required.")
        shapes = safetensors_shapes(server_dir)
        buffered = [key for key in shapes if key.startswith(("buffer::", "param::"))]
        result.update(standard_key_count=len(shapes) - len(buffered), buffered_key_count=len(buffered),
                      sample_standard_keys=[key for key in shapes if key not in buffered][:20],
                      sample_buffered_keys=buffered[:20])
        config = AutoConfig.from_pretrained(server_dir, trust_remote_code=False, local_files_only=True)
        factory = AutoModel if config.model_type == "bert" else AutoModelForCausalLM
        # meta 只建立结构，不分配一个与原模型同样大的随机模型。
        with torch.device("meta"):
            model = factory.from_config(config, trust_remote_code=False)
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        expected=_moe_serialized_shapes(expected,shapes,config.model_type)
        missing = set(expected) - set(shapes)
        if getattr(config, "tie_word_embeddings", False) and "model.embed_tokens.weight" in shapes:
            missing.discard("lm_head.weight")
        unexpected = set(shapes) - set(expected)
        mismatch = {key: {"expected": expected[key], "actual": shapes[key]}
                    for key in expected.keys() & shapes.keys() if expected[key] != shapes[key]}
        passed = not missing and not unexpected and not mismatch and not buffered
        result.update(is_standard_weight_export=passed,
                      layout="standard_weight_visible" if passed else "invalid_standard_structure",
                      missing_standard_keys=sorted(missing), unexpected_keys=sorted(unexpected),
                      shape_mismatches=mismatch)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        result["error"] = str(exc)
    return result
