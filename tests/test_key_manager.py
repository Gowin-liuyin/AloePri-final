import torch
from transformers import AutoTokenizer

from src.defaults import DEFAULT_MODEL_DIR
from src.key_manager import generate_vocab_permutation, invert_permutation, ordinary_token_ids, validate_permutation


def test_invert_permutation() -> None:
    perm = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
    inv = invert_permutation(perm)
    assert torch.equal(inv[perm], torch.arange(perm.numel()))
    assert torch.equal(perm[inv], torch.arange(perm.numel()))


def test_validate_permutation() -> None:
    assert validate_permutation(torch.tensor([1, 2, 0], dtype=torch.long))
    assert not validate_permutation(torch.tensor([1, 1, 0], dtype=torch.long))


def test_special_and_added_tokens_fixed() -> None:
    # 此规则不依赖模型训练结果；替身包含普通、特殊及新增token。
    class Tokenizer:
        vocab_size = 32
        all_special_ids = [0, 1, 2]

        def __len__(self):
            return 34

        def get_added_vocab(self):
            return {"extra_a": 32, "extra_b": 33}

    tokenizer = Tokenizer()
    model_vocab_size = len(tokenizer) + 16
    movable_ids = ordinary_token_ids(tokenizer)
    perm = generate_vocab_permutation(model_vocab_size, seed=1234, movable_ids=movable_ids)

    fixed_ids = set(tokenizer.get_added_vocab().values())
    fixed_ids.update(tokenizer.all_special_ids)
    fixed_ids.update(range(len(tokenizer), model_vocab_size))

    for token_id in fixed_ids:
        assert perm[token_id].item() == token_id
