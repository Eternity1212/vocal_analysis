"""生成可用于 DPO 全流程验证的 demo 数据。"""

from __future__ import annotations

import argparse
import math
import random
import sys
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.score_parsers import TECHNIQUE_ORDER
from src.utils.io import ensure_directory, write_jsonl


def _write_demo_audio(path: Path, duration_seconds: float, sample_rate: int) -> None:
    t = np.linspace(
        0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    frequency = random.choice([220.0, 246.94, 261.63, 293.66, 329.63])
    signal = 0.25 * np.sin(2 * math.pi * frequency * t)
    signal = (signal * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(signal.tobytes())


def _generate_demo_mfcc() -> np.ndarray:
    base = np.random.normal(loc=0.0, scale=1.0, size=(40, 128)).astype(np.float32)
    trend = np.linspace(-0.5, 0.5, 128, dtype=np.float32)
    return base + trend


def _generate_demo_scores(score_dims: int = 10) -> tuple[List[int], List[int]]:
    teacher_score = np.random.randint(2, 6, size=score_dims)
    noise = np.random.choice([-2, -1, 0, 1], size=score_dims, p=[0.1, 0.4, 0.3, 0.2])
    model_score = np.clip(teacher_score + noise, 1, 5)
    return teacher_score.tolist(), model_score.tolist()


def _build_manifest_row(
    sample_id: str,
    audio_path: Path,
    mfcc_path: Path,
    voice_part: str,
    score_key: str,
    score_value: List[int],
) -> Dict[str, object]:
    return {
        "sample_id": sample_id,
        "audio_path": str(audio_path),
        "mfcc_path": str(mfcc_path),
        "voice_part": voice_part,
        score_key: score_value,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 DPO demo 数据")
    parser.add_argument("--output-root", default="data", help="数据根目录")
    parser.add_argument("--num-samples", type=int, default=24, help="样本数量")
    parser.add_argument("--voice-part", default="sopran", help="声部")
    parser.add_argument("--sample-rate", type=int, default=16000, help="音频采样率")
    parser.add_argument("--seed", type=int, default=1314, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root = Path(args.output_root)
    audio_dir = ensure_directory(output_root / "raw" / "audio")
    mfcc_dir = ensure_directory(output_root / "processed" / "mfcc")
    manifest_dir = ensure_directory(output_root / "processed" / "manifests")

    teacher_rows: List[Dict[str, object]] = []
    model_rows: List[Dict[str, object]] = []

    for index in range(1, args.num_samples + 1):
        sample_id = f"demo_{index:04d}"
        audio_path = audio_dir / f"{sample_id}.wav"
        mfcc_path = mfcc_dir / f"{sample_id}_MFCC.xlsx"

        _write_demo_audio(audio_path, duration_seconds=1.2, sample_rate=args.sample_rate)
        pd.DataFrame(_generate_demo_mfcc()).to_excel(
            mfcc_path,
            index=False,
            header=False,
        )

        teacher_score, model_score = _generate_demo_scores(
            score_dims=len(TECHNIQUE_ORDER)
        )
        teacher_rows.append(
            _build_manifest_row(
                sample_id=sample_id,
                audio_path=audio_path,
                mfcc_path=mfcc_path,
                voice_part=args.voice_part,
                score_key="teacher_score",
                score_value=teacher_score,
            )
        )
        model_rows.append(
            _build_manifest_row(
                sample_id=sample_id,
                audio_path=audio_path,
                mfcc_path=mfcc_path,
                voice_part=args.voice_part,
                score_key="model_score",
                score_value=model_score,
            )
        )

    write_jsonl(manifest_dir / "teacher_manifest.jsonl", teacher_rows)
    write_jsonl(manifest_dir / "model_manifest.jsonl", model_rows)

    print(f"已生成 demo 样本数: {args.num_samples}")
    print(f"音频目录: {audio_dir}")
    print(f"MFCC 目录: {mfcc_dir}")
    print(f"老师评分清单: {manifest_dir / 'teacher_manifest.jsonl'}")
    print(f"模型评分清单: {manifest_dir / 'model_manifest.jsonl'}")


if __name__ == "__main__":
    main()
