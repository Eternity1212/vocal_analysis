from __future__ import annotations

import torch

from src.training.losses import sequence_log_probability


def test_sequence_log_probability_output_shape() -> None:
    logits = torch.randn(2, 10, 5)
    scores = torch.randint(1, 6, (2, 10))
    logp = sequence_log_probability(logits, scores)
    assert logp.shape == (2,)
