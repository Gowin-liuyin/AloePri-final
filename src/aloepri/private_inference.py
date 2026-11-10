"""本地模拟 token-ID 协议：客户端只置换输入编号、恢复输出编号。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from src.aloepri._implementation.export.checkpoint_io import load_stage_i_hf_bundle
from src.llama_local_dev import tokenize_llama_prompt
from src.model_loader import tokenize_prompt
from src.transforms import map_input_ids, unmap_output_ids


def load_release_profile(bundle_dir: str | Path, profile: str, *, dtype: str = "float32", device: str = "cpu") -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    catalog = json.loads((bundle_dir / "catalog.json").read_text(encoding="utf-8"))
    profile_map = {item["name"]: item for item in catalog["profiles"]}
    if profile not in profile_map:
        raise ValueError(f"Unknown release profile: {profile}")
    selected = profile_map[profile]
    bundle = load_stage_i_hf_bundle(bundle_dir / selected["server_dir"],
                                  client_secret_path=bundle_dir / selected["client_secret"], device=device, dtype=dtype)
    if bundle["perm_vocab"] is None or bundle["inv_perm_vocab"] is None:
        raise ValueError("Private inference requires the client secret on the client.")
    return bundle


@torch.inference_mode()
def generate_mapped_ids(model: Any, input_ids: torch.Tensor, *, attention_mask: torch.Tensor | None,
                        perm_vocab: torch.Tensor, inv_perm_vocab: torch.Tensor, max_new_tokens: int,
                        do_sample: bool = False, temperature: float = 0.65, top_k: int = 20,
                        top_p: float = 0.95) -> torch.Tensor:
    """服务器在混淆编号中完成生成；HF generate 复用 KV cache 并按 EOS 停止。"""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("Generation requires nonempty token IDs shaped [batch, sequence]; check the prompt and tokenizer files.")
    if max_new_tokens == 0:
        return input_ids[:, :0]
    if do_sample and (temperature <= 0 or not 0 < top_p <= 1 or top_k < 0):
        raise ValueError("Sampling requires temperature > 0, 0 < top_p <= 1 and top_k >= 0.")
    mapped = map_input_ids(input_ids, perm_vocab)
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "use_cache": True}
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is not None:
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        mapped_eos = [int(perm_vocab[int(token)]) for token in eos_ids]
        kwargs["eos_token_id"] = mapped_eos
        pad = getattr(model.generation_config, "pad_token_id", None)
        kwargs["pad_token_id"] = mapped_eos[0] if pad is None else int(perm_vocab[int(pad)])
    if do_sample:
        kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)
    # 检查模型原始分数，不能把后续 top-k 等操作故意屏蔽为 -Inf 的位置误判为模型故障。
    def check_raw_logits(_module, _inputs, output):
        if hasattr(output, "logits") and not bool(torch.isfinite(output.logits).all()):
            raise ValueError("Non-finite raw logits during private generation.")
    handle = model.register_forward_hook(check_raw_logits) if hasattr(model, "register_forward_hook") else None
    try:
        output = model.generate(input_ids=mapped, attention_mask=attention_mask, **kwargs)
    finally:
        if handle is not None:
            handle.remove()
    return unmap_output_ids(output[:, input_ids.shape[1]:], inv_perm_vocab)


@torch.inference_mode()
def generate_private_text(*, bundle_dir: str | Path, profile: str, prompt: str,
                          max_new_tokens: int, dtype: str = "float32", device: str = "cpu",
                          do_sample: bool = False, temperature: float = 0.65, top_k: int = 20,
                          top_p: float = 0.95) -> str:
    bundle = load_release_profile(bundle_dir, profile, dtype=dtype, device=device)
    tokenizer, model = bundle["tokenizer"], bundle["model"]
    if model.config.model_type == "qwen2" and getattr(tokenizer, "chat_template", None):
        encoded = tokenize_prompt(tokenizer, prompt, device=device)
    else:
        encoded = tokenize_llama_prompt(tokenizer, prompt, device=device)
    generated = generate_mapped_ids(model, encoded["input_ids"], attention_mask=encoded.get("attention_mask"),
                                   perm_vocab=bundle["perm_vocab"], inv_perm_vocab=bundle["inv_perm_vocab"],
                                   max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
                                   top_k=top_k, top_p=top_p)
    return tokenizer.decode(generated[0].cpu(), skip_special_tokens=True)
