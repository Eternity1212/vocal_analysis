"""文件读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def ensure_directory(path: str | Path) -> Path:
    """确保目录存在。"""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> Path:
    """写出 JSONL 文件。"""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return output_path
