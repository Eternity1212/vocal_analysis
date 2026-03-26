"""配置加载工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容必须为字典: {config_path}")

    return data
