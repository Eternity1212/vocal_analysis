"""将参考模型输出标准化为 model_manifest.jsonl。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.score_parsers import (
    find_first_mfcc_file,
    normalize_sample_id,
    parse_client_scores_excel,
    parse_sft_prediction_excel,
)
from src.utils.io import ensure_directory, write_jsonl


def build_from_sft_prediction(
    prediction_excel: str | Path,
) -> List[Dict[str, object]]:
    prediction_map = parse_sft_prediction_excel(prediction_excel)
    return [
        {
            "sample_id": sample_id,
            "model_score": scores,
        }
        for sample_id, scores in sorted(prediction_map.items())
    ]


def build_from_client_task_dir(task_root: str | Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task_dir in sorted(Path(task_root).glob("task_*")):
        prediction_path = task_dir / "predictions.xlsx"
        if not prediction_path.exists():
            continue

        mfcc_path = find_first_mfcc_file(task_dir)
        sample_id = normalize_sample_id(mfcc_path.name)
        rows.append(
            {
                "sample_id": sample_id,
                "model_score": parse_client_scores_excel(prediction_path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成参考模型评分清单")
    parser.add_argument(
        "--source-type",
        choices=["sft_prediction_excel", "client_task_dir"],
        required=True,
        help="模型评分来源类型",
    )
    parser.add_argument("--input-path", required=True, help="输入路径")
    parser.add_argument(
        "--output-path",
        required=True,
        help="model_manifest.jsonl 输出路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_type == "sft_prediction_excel":
        rows = build_from_sft_prediction(args.input_path)
    else:
        rows = build_from_client_task_dir(args.input_path)

    output_path = Path(args.output_path)
    ensure_directory(output_path.parent)
    write_jsonl(output_path, rows)
    print(f"模型评分清单已生成: {output_path}")


if __name__ == "__main__":
    main()
