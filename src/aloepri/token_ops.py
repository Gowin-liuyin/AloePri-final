from __future__ import annotations

from typing import Any

import torch

from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids
from src.transforms import map_input_ids, restore_logits, unmap_output_ids


def build_vocab_keys(
    *,
    tokenizer: Any,
    model_vocab_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    movable_ids = ordinary_token_ids(tokenizer)
    perm_vocab = generate_vocab_permutation(
        vocab_size=model_vocab_size,
        seed=seed,
        movable_ids=movable_ids,
    )
    inv_perm_vocab = invert_permutation(perm_vocab)
    return perm_vocab, inv_perm_vocab


obfuscate_input_ids = map_input_ids


restore_output_logits = restore_logits


restore_output_ids = unmap_output_ids
