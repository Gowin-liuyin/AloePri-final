"""Equivalence check: obfuscated Qwen3.5 text tower must reproduce the clean logits.

Run from the reference tree root (``Aloepri_Llama-3.2-1B-Instruct_obfuscation_code_20261103``).
A tiny random model is enough: the square-monomial transform is exact arithmetic, so a
shape-faithful toy exercises every code path (gated-delta-net layer, full-attention
layer, MLP, embed/head, final norm) without touching the 27B weights.
"""
from __future__ import annotations

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from src.aloepri._implementation.export.qwen3_5_shape_obfuscation import build_qwen3_5_square_model


class FakeTokenizer:
    """Minimal surface required by ``ordinary_token_ids``."""

    vocab_size = 64
    all_special_ids = [0, 1, 2]


def tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        dtype="float32",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: once flash-linear-attention is installed the "
                         "gated-delta-net path dispatches to a Triton kernel that rejects CPU tensors")
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(tiny_config()).eval().to(device=device, dtype=torch.float32)

    ids = torch.randint(3, 64, (2, 16), device=device)
    with torch.no_grad():
        base_logits = model(input_ids=ids).logits.float()

    model_obf, perm, inv_perm, transform, kinds = build_qwen3_5_square_model(
        model, FakeTokenizer(), seed=1234
    )
    model_obf = model_obf.to(device)
    perm, inv_perm = perm.to(device), inv_perm.to(device)
    print("layer kinds:", kinds)
    print("transform non-trivial perm:", not torch.equal(transform.perm, torch.arange(transform.dim)))
    print("transform non-trivial signs:", not bool(torch.all(transform.signs == 1).item()))

    permuted_ids = perm[ids]
    with torch.no_grad():
        obf_logits = model_obf(input_ids=permuted_ids).logits.float()

    candidates = {
        "identity": base_logits,
        "index_inv_perm": base_logits.index_select(-1, inv_perm),
        "index_perm": base_logits.index_select(-1, perm),
    }
    print("\nmax |obf - candidate|:")
    for name, ref in candidates.items():
        print(f"  {name:16s} {(obf_logits - ref).abs().max().item():.3e}")

    print("\nargmax agreement (identity layout):",
          float((obf_logits.argmax(-1) == base_logits.argmax(-1)).float().mean()))


if __name__ == "__main__":
    main()
