"""合并老师评分与参考模型评分，生成 DPO JSONL。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dpo_dataset import DPORecord
from src.utils.io import ensure_directory, write_jsonl


def _load_manifest(path: str | Path, score_key: str) -> Dict[str, dict]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到清单文件: {manifest_path}")

    records: Dict[str, dict] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            content = line.strip()
            if not content:
                continue
            record = json.loads(content)
            sample_id = str(record["sample_id"])
            if score_key not in record:
                raise ValueError(
                    f"清单 {manifest_path} 缺少字段: {score_key}"
                )
            records[sample_id] = record
    return records


def build_records(
    teacher_manifest: Dict[str, dict],
    model_manifest: Dict[str, dict],
) -> List[DPORecord]:
    common_ids = sorted(set(teacher_manifest) & set(model_manifest))
    if not common_ids:
        raise ValueError("老师评分清单和模型评分清单没有交集样本")

    records: List[DPORecord] = []
    for sample_id in common_ids:
        teacher_record = teacher_manifest[sample_id]
        model_record = model_manifest[sample_id]
        records.append(
            DPORecord.from_dict(
                {
                    "sample_id": sample_id,
                    "audio_path": teacher_record["audio_path"],
                    "mfcc_path": teacher_record["mfcc_path"],
                    "voice_part": teacher_record["voice_part"],
                    "teacher_score": teacher_record["teacher_score"],
                    "model_score": model_record["model_score"],
                    "teacher_preferred_flag": 1,
                }
            )
        )
    return records


def split_records(
    records: List[DPORecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[DPORecord]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 DPO 数据集")
    parser.add_argument("--teacher-manifest", required=True, help="老师评分清单 JSONL")
    parser.add_argument("--model-manifest", required=True, help="参考模型评分清单 JSONL")
    parser.add_argument("--output-dir", required=True, help="DPO JSONL 输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=1314, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_manifest = _load_manifest(args.teacher_manifest, score_key="teacher_score")
    model_manifest = _load_manifest(args.model_manifest, score_key="model_score")
    records = build_records(teacher_manifest, model_manifest)
    splits = split_records(records, args.train_ratio, args.val_ratio, args.seed)

    output_dir = ensure_directory(args.output_dir)
    for split_name, split_records_list in splits.items():
        output_path = output_dir / f"{split_name}_dpo.jsonl"
        write_jsonl(output_path, [record.to_dict() for record in split_records_list])
        print(f"{split_name} 集已写出: {output_path}")


if __name__ == "__main__":
    main()
