from __future__ import annotations

import random
import os
from typing import Any
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.defaults import DEFAULT_SYSTEM_PROMPT


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    normalized = dtype.lower()
    mapping: dict[str, torch.dtype | str] = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[normalized]


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cpu",
    dtype: str = "auto",
) -> tuple[Any, Any]:
    if os.name == "nt":
        os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    torch_dtype = resolve_torch_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = load_causal_model(model_name, device=device, dtype=dtype)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return tokenizer, model


def load_causal_model(model_name: str | Path, *, device="cpu", dtype="auto"):
    """Load local Windows Llama/Qwen tensors one shard at a time, bounding mapped storage."""
    root = Path(model_name)
    if os.name == "nt" and root.is_dir():
        config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        if config.model_type in {"llama", "qwen2"}:
            return load_local_model_streaming(root, config=config, device=device, dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=False, dtype=resolve_torch_dtype(dtype),
        **({"device_map": device} if device != "cpu" else {}))


def load_local_model_streaming(model_dir: str | Path, *, config=None, device="cpu", dtype="auto"):
    from safetensors import safe_open
    from src.aloepri._implementation.export.tensor_io import _checkpoint_files, safetensors_shapes
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    root = Path(model_dir)
    config = config or AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    shapes = safetensors_shapes(root)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    missing = set(expected) - set(shapes)
    if config.tie_word_embeddings and "model.embed_tokens.weight" in shapes:
        missing.discard("lm_head.weight")
    if missing or set(shapes) - set(expected) or any(shapes[name] != expected[name] for name in shapes if name in expected):
        raise ValueError("Local checkpoint tensor names/shapes do not match its configuration")
    requested_dtype = resolve_torch_dtype(dtype)
    configured_dtype = getattr(config, "dtype", None)
    if isinstance(configured_dtype, str):
        configured_dtype = resolve_torch_dtype(configured_dtype)
    target_dtype = configured_dtype if requested_dtype == "auto" else requested_dtype
    _, files, _ = _checkpoint_files(root)
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                parent_name, parameter_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                original = parent._parameters[parameter_name]
                value = handle.get_tensor(name)
                converted = value.to(device=device, dtype=target_dtype or value.dtype, copy=True)
                parent._parameters[parameter_name] = torch.nn.Parameter(converted, requires_grad=original.requires_grad)
                del value, converted, original
    model.tie_weights()
    for module in model.modules():
        if hasattr(module, "compute_default_rope_parameters") and hasattr(module, "inv_freq") and module.inv_freq.device.type == "meta":
            rope_type = module.config.rope_parameters.get("rope_type", "default")
            initialize = module.compute_default_rope_parameters if rope_type == "default" else ROPE_INIT_FUNCTIONS[rope_type]
            frequency, scaling = initialize(module.config, device=torch.device(device))
            module.inv_freq = frequency
            module.original_inv_freq = frequency.clone()
            module.attention_scaling = scaling
    remaining = [name for name, value in list(model.named_parameters()) + list(model.named_buffers()) if value.device.type == "meta"]
    if remaining:
        raise ValueError(f"Unmaterialized local model tensors: {remaining}")
    generation_file = root / "generation_config.json"
    model.generation_config = (GenerationConfig.from_pretrained(root, local_files_only=True)
                               if generation_file.is_file() else GenerationConfig.from_model_config(config))
    return model.eval()


def format_chat_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    text = format_chat_prompt(tokenizer, prompt, system_prompt=system_prompt)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return move_batch_to_device(encoded, device)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
