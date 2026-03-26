"""评估指标。"""

from __future__ import annotations

import torch
from torch import Tensor


def score_accuracy(pred_scores: Tensor, target_scores: Tensor) -> float:
    """计算逐维准确率。"""
    correct = (pred_scores == target_scores).sum().item()
    total = target_scores.numel()
    return correct / max(total, 1)


def mean_absolute_error_per_dim(
    pred_scores: Tensor,
    target_scores: Tensor,
) -> Tensor:
    """计算每个评分维度的平均绝对误差。"""
    return torch.abs(pred_scores - target_scores).float().mean(dim=0)
