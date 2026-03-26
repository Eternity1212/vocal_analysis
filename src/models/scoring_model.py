"""评分模型骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn


@dataclass(slots=True)
class ModelSpec:
    """模型结构配置。"""

    input_shape: Sequence[int]
    score_dims: int
    num_classes: int
    hidden_dim: int = 256
    dropout: float = 0.1


class AudioScoringModel(nn.Module):
    """用于参考训练与 DPO 训练的基础评分模型。"""

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

        flattened_dim = 1
        for size in spec.input_shape:
            flattened_dim *= int(size)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, spec.hidden_dim),
            nn.ReLU(),
            nn.Dropout(spec.dropout),
            nn.Linear(
                spec.hidden_dim,
                spec.score_dims * spec.num_classes,
            ),
        )

    def forward(self, mfcc: Tensor) -> Tensor:
        logits = self.encoder(mfcc)
        batch_size = logits.shape[0]
        return logits.view(
            batch_size,
            self.spec.score_dims,
            self.spec.num_classes,
        )
