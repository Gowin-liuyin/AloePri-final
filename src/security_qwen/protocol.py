"""攻击评测的共同约定：公开候选独立取样，已知明文配对必须明确启用。"""

from __future__ import annotations

import torch

from src.key_manager import ordinary_token_ids


def require_positive(**values: int) -> None:
    """在加载模型前检查数量参数，避免空训练集或 top-k 为零。"""
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")


def public_candidate_ids(tokenizer, *, size: int, seed: int) -> torch.Tensor:
    """只根据公开词表、数量和种子选候选，不接收测试答案或秘密置换。"""
    require_positive(candidate_pool_size=size)
    ids = ordinary_token_ids(tokenizer)
    if ids.numel() == 0:
        raise ValueError("Tokenizer has no ordinary token candidates")
    if size >= ids.numel():
        return ids
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
    return ids[torch.randperm(ids.numel(), generator=generator)[:size]]


def require_known_pairs(allow_simulated_known_pairs: bool) -> None:
    """本仓库用客户端密钥造训练配对；要求调用者明确接受这个额外条件。"""
    if not allow_simulated_known_pairs:
        raise ValueError(
            "This experiment uses the client secret to generate known plaintext/observation "
            "training pairs under the target key. Set allow_simulated_known_pairs=True "
            "(CLI: --allow-simulated-known-pairs) to run this simulation. "
            "It is not an attack with deployment weights alone."
        )


KNOWN_PAIR_PROTOCOL = {
    "training_pair_source": "client_secret_simulator",
    "attacker_assumption": "known_plaintext_observation_pairs_under_target_key",
    "deployment_only": False,
    "candidate_selection": "public_vocab_independent_of_test_answers",
    "paper_attack_reproduced": False,
    "risk_level_basis": "project_defined_thresholds_0.1_0.3_not_paper_privacy_guarantee",
}
