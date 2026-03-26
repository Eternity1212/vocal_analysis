"""将预测结果导出为前端可消费的 scores_data 结构。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.score_parsers import TECHNIQUE_ORDER
from src.utils.io import ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出前端 payload")
    parser.add_argument("--input-jsonl", required=True, help="预测结果 JSONL")
    parser.add_argument("--output-jsonl", required=True, help="前端 payload JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    ensure_directory(output_path.parent)

    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w",
        encoding="utf-8",
    ) as target:
        for line in source:
            content = line.strip()
            if not content:
                continue
            record = json.loads(content)
            pred_score = record["pred_score"]
            payload = {
                "sample_id": record["sample_id"],
                "voice_part": record.get("voice_part"),
                "scores_data": {
                    name: int(score)
                    for name, score in zip(TECHNIQUE_ORDER, pred_score)
                },
            }
            target.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"前端 payload 已导出: {output_path}")


if __name__ == "__main__":
    main()
