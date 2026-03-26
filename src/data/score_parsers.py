"""评分文件解析器。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


TECHNIQUE_ORDER = [
    "Vibrato",
    "Throat",
    "Position",
    "Open",
    "Clean",
    "Resonate",
    "Unify",
    "Falsetto",
    "Chest",
    "Nasal",
]


def normalize_sample_id(name: str) -> str:
    """统一样本 ID 命名。"""
    sample_id = Path(name).stem
    if sample_id.endswith("_MFCC"):
        sample_id = sample_id[: -len("_MFCC")]
    return sample_id


def parse_client_scores_excel(path: str | Path) -> List[int]:
    """解析 Client 风格的竖表评分 Excel。"""
    excel_path = Path(path)
    frame = pd.read_excel(excel_path, engine="openpyxl")
    columns = {str(col).strip(): col for col in frame.columns}
    name_column = None
    score_column = None

    for candidate in ("Class", "Skill Tag"):
        if candidate in columns:
            name_column = columns[candidate]
            break

    for candidate in ("Value", "Score"):
        if candidate in columns:
            score_column = columns[candidate]
            break

    if name_column is None or score_column is None:
        raise ValueError(
            f"评分文件缺少类别列或分数列: {excel_path}"
        )

    score_map: Dict[str, int] = {}
    for _, row in frame.iterrows():
        score_map[str(row[name_column]).strip()] = int(row[score_column])

    missing = [name for name in TECHNIQUE_ORDER if name not in score_map]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Client 评分文件缺少评分维度: {missing_text}")

    return [score_map[name] for name in TECHNIQUE_ORDER]


def parse_sft_label_excel(
    path: str | Path,
    score_dims: int = 10,
    score_column_index: int = 1,
) -> List[int]:
    """解析 sft 标签 Excel，默认读取第二列前 10 行。"""
    excel_path = Path(path)
    frame = pd.read_excel(excel_path, engine="openpyxl")
    if frame.shape[1] <= score_column_index:
        raise ValueError(f"标签文件列数不足: {excel_path}")

    values = frame.iloc[:score_dims, score_column_index].tolist()
    return [int(value) for value in values]


def parse_sft_prediction_excel(
    path: str | Path,
    technique_order: List[str] | None = None,
) -> Dict[str, List[int]]:
    """解析 sft 批量预测结果 Excel。"""
    order = technique_order or TECHNIQUE_ORDER
    excel_path = Path(path)
    frame = pd.read_excel(excel_path, engine="openpyxl")
    required_columns = ["Filename", *[f"Pred_{name}" for name in order]]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"sft 预测结果缺少列: {missing_text}")

    records: Dict[str, List[int]] = {}
    for _, row in frame.iterrows():
        sample_id = normalize_sample_id(str(row["Filename"]))
        records[sample_id] = [int(row[f"Pred_{name}"]) for name in order]
    return records


def find_first_mfcc_file(task_dir: str | Path) -> Path:
    """在任务目录中寻找第一个 MFCC 文件。"""
    task_path = Path(task_dir)
    for candidate in sorted(task_path.rglob("*_MFCC.xlsx")):
        return candidate
    raise FileNotFoundError(f"未在任务目录中找到 MFCC 文件: {task_path}")
