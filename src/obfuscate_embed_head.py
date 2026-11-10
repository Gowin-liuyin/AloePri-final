from __future__ import annotations

from copy import deepcopy

import torch


def _permute_weight_rows(
    weight: torch.Tensor,
    perm_vocab: torch.Tensor,
) -> torch.Tensor:
    perm_vocab = perm_vocab.to(weight.device)
    inverse = torch.empty_like(perm_vocab)
    inverse[perm_vocab] = torch.arange(perm_vocab.numel(), device=perm_vocab.device)
    # Gather bounds temporary storage without a full clone followed by a scatter.
    return weight.index_select(0, inverse)


permute_embedding_weight = _permute_weight_rows
permute_lm_head_weight = _permute_weight_rows
permute_output_bias = _permute_weight_rows


def build_vocab_permuted_model(
    model,
    perm_vocab: torch.Tensor,
    inv_perm_vocab: torch.Tensor | None = None,
    *,
    copy_model: bool = True,
):
    del inv_perm_vocab
    permuted_model = deepcopy(model) if copy_model else model
    input_embeddings = permuted_model.get_input_embeddings()
    output_embeddings = permuted_model.get_output_embeddings()

    tied = input_embeddings.weight is output_embeddings.weight
    with torch.no_grad():
        input_embeddings.weight.copy_(permute_embedding_weight(input_embeddings.weight.detach(), perm_vocab))
        if not tied:
            output_embeddings.weight.copy_(permute_lm_head_weight(output_embeddings.weight.detach(), perm_vocab))
        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.copy_(permute_output_bias(output_embeddings.bias.detach(), perm_vocab))

    return permuted_model
