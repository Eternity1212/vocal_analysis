"""参考模型与 DPO 模型训练器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.losses import dpo_loss, sequence_log_probability
from src.utils.io import ensure_directory


def _infer_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


@dataclass(slots=True)
class EpochResult:
    """单个 epoch 的聚合结果。"""

    loss: float
    metric: float


class ReferenceTrainer:
    """监督参考模型训练器。"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device_name: str = "auto",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = _infer_device(device_name)
        self.criterion = nn.CrossEntropyLoss()
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> EpochResult:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in dataloader:
            mfcc = batch["mfcc"].to(self.device)
            teacher_score = batch["teacher_score"].to(self.device) - 1

            logits = self.model(mfcc)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                teacher_score.reshape(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * mfcc.shape[0]
            total_correct += (preds == teacher_score).sum().item()
            total_count += teacher_score.numel()

        return EpochResult(
            loss=total_loss / max(len(dataloader.dataset), 1),
            metric=total_correct / max(total_count, 1),
        )

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> EpochResult:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in dataloader:
            mfcc = batch["mfcc"].to(self.device)
            teacher_score = batch["teacher_score"].to(self.device) - 1
            logits = self.model(mfcc)
            loss = self.criterion(
                logits.reshape(-1, logits.shape[-1]),
                teacher_score.reshape(-1),
            )

            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * mfcc.shape[0]
            total_correct += (preds == teacher_score).sum().item()
            total_count += teacher_score.numel()

        return EpochResult(
            loss=total_loss / max(len(dataloader.dataset), 1),
            metric=total_correct / max(total_count, 1),
        )


class DPOTrainer:
    """DPO 训练器。"""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        optimizer: Optimizer,
        beta: float,
        device_name: str = "auto",
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.beta = beta
        self.device = _infer_device(device_name)

        self.policy_model.to(self.device)
        self.reference_model.to(self.device)
        self.reference_model.eval()
        for parameter in self.reference_model.parameters():
            parameter.requires_grad = False

    def train_epoch(self, dataloader: DataLoader) -> EpochResult:
        self.policy_model.train()
        total_loss = 0.0
        total_margin = 0.0

        for batch in dataloader:
            mfcc = batch["mfcc"].to(self.device)
            teacher_score = batch["teacher_score"].to(self.device)
            model_score = batch["model_score"].to(self.device)

            policy_logits = self.policy_model(mfcc)
            with torch.no_grad():
                ref_logits = self.reference_model(mfcc)

            policy_chosen_logp = sequence_log_probability(policy_logits, teacher_score)
            policy_rejected_logp = sequence_log_probability(policy_logits, model_score)
            ref_chosen_logp = sequence_log_probability(ref_logits, teacher_score)
            ref_rejected_logp = sequence_log_probability(ref_logits, model_score)

            loss = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=self.beta,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            margin = (policy_chosen_logp - policy_rejected_logp).mean().item()
            total_loss += loss.item() * mfcc.shape[0]
            total_margin += margin * mfcc.shape[0]

        return EpochResult(
            loss=total_loss / max(len(dataloader.dataset), 1),
            metric=total_margin / max(len(dataloader.dataset), 1),
        )

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> EpochResult:
        self.policy_model.eval()
        total_loss = 0.0
        total_margin = 0.0

        for batch in dataloader:
            mfcc = batch["mfcc"].to(self.device)
            teacher_score = batch["teacher_score"].to(self.device)
            model_score = batch["model_score"].to(self.device)

            policy_logits = self.policy_model(mfcc)
            ref_logits = self.reference_model(mfcc)

            policy_chosen_logp = sequence_log_probability(policy_logits, teacher_score)
            policy_rejected_logp = sequence_log_probability(policy_logits, model_score)
            ref_chosen_logp = sequence_log_probability(ref_logits, teacher_score)
            ref_rejected_logp = sequence_log_probability(ref_logits, model_score)

            loss = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=self.beta,
            )

            margin = (policy_chosen_logp - policy_rejected_logp).mean().item()
            total_loss += loss.item() * mfcc.shape[0]
            total_margin += margin * mfcc.shape[0]

        return EpochResult(
            loss=total_loss / max(len(dataloader.dataset), 1),
            metric=total_margin / max(len(dataloader.dataset), 1),
        )


def save_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    checkpoint_name: str,
) -> Path:
    """保存模型参数。"""
    directory = ensure_directory(output_dir)
    checkpoint_path = directory / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path
