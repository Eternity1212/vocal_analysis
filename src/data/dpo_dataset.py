"""DPO 数据集定义。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset

from src.feature.mfcc import load_mfcc_feature


REQUIRED_FIELDS = {
    "sample_id",
    "audio_path",
    "mfcc_path",
    "voice_part",
    "model_score",
    "teacher_score",
    "teacher_preferred_flag",
}


@dataclass(slots=True)
class DPORecord:
    """单条 DPO 样本记录。"""

    sample_id: str
    audio_path: str
    mfcc_path: str
    voice_part: str
    model_score: List[int]
    teacher_score: List[int]
    teacher_preferred_flag: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DPORecord":
        missing = REQUIRED_FIELDS - set(data)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"DPO 样本缺少字段: {missing_text}")

        return cls(
            sample_id=str(data["sample_id"]),
            audio_path=str(data["audio_path"]),
            mfcc_path=str(data["mfcc_path"]),
            voice_part=str(data["voice_part"]),
            model_score=[int(value) for value in data["model_score"]],
            teacher_score=[int(value) for value in data["teacher_score"]],
            teacher_preferred_flag=int(data["teacher_preferred_flag"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "audio_path": self.audio_path,
            "mfcc_path": self.mfcc_path,
            "voice_part": self.voice_part,
            "model_score": self.model_score,
            "teacher_score": self.teacher_score,
            "teacher_preferred_flag": self.teacher_preferred_flag,
        }


def load_jsonl_records(path: str | Path) -> List[DPORecord]:
    """读取 JSONL 并解析为记录对象列表。"""
    records: List[DPORecord] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"未找到 JSONL 文件: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content = line.strip()
            if not content:
                continue
            try:
                raw_record = json.loads(content)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 解析失败: {jsonl_path} 第 {line_number} 行"
                ) from error
            records.append(DPORecord.from_dict(raw_record))

    return records


class DPODataset(Dataset):
    """DPO 训练/验证数据集。"""

    def __init__(self, jsonl_path: str | Path):
        self.records = load_jsonl_records(jsonl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        return {
            "sample_id": record.sample_id,
            "voice_part": record.voice_part,
            "mfcc": load_mfcc_feature(record.mfcc_path),
            "model_score": torch.tensor(record.model_score, dtype=torch.long),
            "teacher_score": torch.tensor(record.teacher_score, dtype=torch.long),
            "teacher_preferred_flag": torch.tensor(
                record.teacher_preferred_flag,
                dtype=torch.long,
            ),
        }


def build_dpo_record(
    sample_id: str,
    audio_path: str,
    mfcc_path: str,
    voice_part: str,
    model_score: Iterable[int],
    teacher_score: Iterable[int],
    teacher_preferred_flag: int = 1,
) -> DPORecord:
    """构造一条标准化 DPO 样本。"""
    return DPORecord(
        sample_id=sample_id,
        audio_path=audio_path,
        mfcc_path=mfcc_path,
        voice_part=voice_part,
        model_score=[int(value) for value in model_score],
        teacher_score=[int(value) for value in teacher_score],
        teacher_preferred_flag=int(teacher_preferred_flag),
    )
