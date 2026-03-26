"""训练损失函数。"""

from __future__ import annotations

import torch
from torch import Tensor


def sequence_log_probability(logits: Tensor, scores: Tensor) -> Tensor:
    """计算评分向量在模型下的总对数概率。"""
    if scores.min().item() < 0:
        targets = scores.long()
    else:
        targets = scores.long() - 1

    log_probs = torch.log_softmax(logits, dim=-1)
    gathered = torch.gather(
        log_probs,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    return gathered.sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    ref_chosen_logp: Tensor,
    ref_rejected_logp: Tensor,
    beta: float,
) -> Tensor:
    """计算标准 DPO 损失。"""
    preference_gap = (
        (policy_chosen_logp - ref_chosen_logp)
        - (policy_rejected_logp - ref_rejected_logp)
    )
    return -torch.nn.functional.logsigmoid(beta * preference_gap).mean()
