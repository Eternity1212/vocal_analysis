"""将老师评分文件标准化为 teacher_manifest.jsonl。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dpo_dataset import DPORecord
from src.data.score_parsers import (
    normalize_sample_id,
    parse_client_scores_excel,
    parse_sft_label_excel,
)
from src.utils.io import ensure_directory, write_jsonl


def _resolve_audio_path(audio_dir: str | Path, sample_id: str) -> str:
    return str(Path(audio_dir) / f"{sample_id}.wav")


def _resolve_mfcc_path(mfcc_dir: str | Path, sample_id: str) -> str:
    return str(Path(mfcc_dir) / f"{sample_id}_MFCC.xlsx")


def build_from_label_dir(
    label_dir: str | Path,
    audio_dir: str | Path,
    mfcc_dir: str | Path,
    voice_part: str,
    score_dims: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for label_path in sorted(Path(label_dir).glob("*.xlsx")):
        sample_id = normalize_sample_id(label_path.name)
        teacher_score = parse_sft_label_excel(label_path, score_dims=score_dims)
        records.append(
            DPORecord.from_dict(
                {
                    "sample_id": sample_id,
                    "audio_path": _resolve_audio_path(audio_dir, sample_id),
                    "mfcc_path": _resolve_mfcc_path(mfcc_dir, sample_id),
                    "voice_part": voice_part,
                    "model_score": [1] * score_dims,
                    "teacher_score": teacher_score,
                    "teacher_preferred_flag": 1,
                }
            ).to_dict()
        )
    return records


def build_from_client_dir(
    score_dir: str | Path,
    audio_dir: str | Path,
    mfcc_dir: str | Path,
    voice_part: str,
    file_pattern: str,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for score_path in sorted(Path(score_dir).glob(file_pattern)):
        sample_id = normalize_sample_id(score_path.name)
        teacher_score = parse_client_scores_excel(score_path)
        records.append(
            DPORecord.from_dict(
                {
                    "sample_id": sample_id,
                    "audio_path": _resolve_audio_path(audio_dir, sample_id),
                    "mfcc_path": _resolve_mfcc_path(mfcc_dir, sample_id),
                    "voice_part": voice_part,
                    "model_score": [1] * len(teacher_score),
                    "teacher_score": teacher_score,
                    "teacher_preferred_flag": 1,
                }
            ).to_dict()
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成老师评分清单")
    parser.add_argument(
        "--source-type",
        choices=["label_dir", "client_excel_dir"],
        required=True,
        help="老师评分来源类型",
    )
    parser.add_argument("--input-dir", required=True, help="输入目录")
    parser.add_argument("--audio-dir", required=True, help="音频目录")
    parser.add_argument("--mfcc-dir", required=True, help="MFCC 目录")
    parser.add_argument("--voice-part", required=True, help="声部")
    parser.add_argument(
        "--output-path",
        required=True,
        help="teacher_manifest.jsonl 输出路径",
    )
    parser.add_argument(
        "--score-dims",
        type=int,
        default=10,
        help="评分维度数量，仅 label_dir 模式使用",
    )
    parser.add_argument(
        "--file-pattern",
        default="*.xlsx",
        help="client_excel_dir 模式下的文件匹配模式",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_type == "label_dir":
        rows = build_from_label_dir(
            label_dir=args.input_dir,
            audio_dir=args.audio_dir,
            mfcc_dir=args.mfcc_dir,
            voice_part=args.voice_part,
            score_dims=args.score_dims,
        )
    else:
        rows = build_from_client_dir(
            score_dir=args.input_dir,
            audio_dir=args.audio_dir,
            mfcc_dir=args.mfcc_dir,
            voice_part=args.voice_part,
            file_pattern=args.file_pattern,
        )

    output_path = Path(args.output_path)
    ensure_directory(output_path.parent)
    write_jsonl(output_path, rows)
    print(f"老师评分清单已生成: {output_path}")


if __name__ == "__main__":
    main()
