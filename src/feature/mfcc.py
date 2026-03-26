"""MFCC 特征读写辅助。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def load_mfcc_feature(path: str | Path) -> Tensor:
    """从 xlsx 或 npy 文件读取 MFCC 特征。"""
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"未找到 MFCC 文件: {feature_path}")

    suffix = feature_path.suffix.lower()
    if suffix == ".xlsx":
        values = pd.read_excel(
            feature_path,
            header=None,
            engine="openpyxl",
        ).values.astype(np.float32)
    elif suffix == ".npy":
        values = np.load(feature_path).astype(np.float32)
    else:
        raise ValueError(f"暂不支持的 MFCC 文件类型: {feature_path.suffix}")

    if values.ndim != 2:
        raise ValueError(
            f"MFCC 特征应为二维矩阵，当前形状为: {tuple(values.shape)}"
        )

    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)
