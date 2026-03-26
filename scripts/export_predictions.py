"""导出模型预测结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DPODataset
from src.models import AudioScoringModel, ModelSpec
from src.utils import ensure_directory, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出预测结果 JSONL")
    parser.add_argument("--config", default="configs/train_dpo.yaml", help="模型配置")
    parser.add_argument("--checkpoint", required=True, help="模型权重路径")
    parser.add_argument("--input-jsonl", required=True, help="输入数据集 JSONL")
    parser.add_argument("--output-jsonl", required=True, help="预测导出 JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    spec = ModelSpec(
        input_shape=config["model"]["input_shape"],
        score_dims=int(config["model"]["score_dims"]),
        num_classes=int(config["model"]["num_classes"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        dropout=float(config["model"]["dropout"]),
    )
    model = AudioScoringModel(spec)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    dataset = DPODataset(args.input_jsonl)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_path = Path(args.output_jsonl)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for batch in dataloader:
            logits = model(batch["mfcc"])
            preds = logits.argmax(dim=-1).squeeze(0).tolist()
            record = {
                "sample_id": batch["sample_id"][0],
                "voice_part": batch["voice_part"][0],
                "pred_score": [int(value) + 1 for value in preds],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"预测结果已导出到: {output_path}")


if __name__ == "__main__":
    main()
