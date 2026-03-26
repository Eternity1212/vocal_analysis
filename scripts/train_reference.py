"""参考模型训练入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DPODataset
from src.models import AudioScoringModel, ModelSpec
from src.training import ReferenceTrainer, save_checkpoint
from src.utils import ensure_directory, load_yaml_config, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练参考模型")
    parser.add_argument(
        "--config",
        default="configs/train_reference.yaml",
        help="参考模型训练配置",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    set_global_seed(int(config["experiment"]["seed"]))

    train_dataset = DPODataset(config["data"]["train_jsonl"])
    val_dataset = DPODataset(config["data"]["val_jsonl"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
    )

    spec = ModelSpec(
        input_shape=config["model"]["input_shape"],
        score_dims=int(config["model"]["score_dims"]),
        num_classes=int(config["model"]["num_classes"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model = AudioScoringModel(spec)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    trainer = ReferenceTrainer(
        model=model,
        optimizer=optimizer,
        device_name=str(config["training"]["device"]),
    )

    output_dir = ensure_directory(config["experiment"]["output_dir"])
    best_metric = float("-inf")

    for epoch in range(int(config["training"]["epochs"])):
        train_result = trainer.train_epoch(train_loader)
        val_result = trainer.validate(val_loader)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_result.loss:.4f} "
            f"train_acc={train_result.metric:.4f} "
            f"val_loss={val_result.loss:.4f} "
            f"val_acc={val_result.metric:.4f}"
        )

        if val_result.metric > best_metric:
            best_metric = val_result.metric
            checkpoint_path = save_checkpoint(
                model=model,
                output_dir=output_dir,
                checkpoint_name=str(config["training"]["checkpoint_name"]),
            )
            print(f"已保存最佳参考模型: {checkpoint_path}")


if __name__ == "__main__":
    main()
