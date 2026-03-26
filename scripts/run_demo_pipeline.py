"""一键跑通 DPO demo 流程。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_step(command: list[str], workdir: Path) -> None:
    print(f"\n>>> 执行: {' '.join(command)}")
    subprocess.run(command, cwd=workdir, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 DPO demo 全流程")
    parser.add_argument("--num-samples", type=int, default=24, help="demo 样本数")
    parser.add_argument("--voice-part", default="sopran", help="声部")
    parser.add_argument("--seed", type=int, default=1314, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exe = sys.executable

    _run_step(
        [
            python_exe,
            "scripts/generate_demo_dataset.py",
            "--output-root",
            "data",
            "--num-samples",
            str(args.num_samples),
            "--voice-part",
            args.voice_part,
            "--seed",
            str(args.seed),
        ],
        workdir=project_root,
    )

    _run_step(
        [
            python_exe,
            "scripts/build_dpo_dataset.py",
            "--teacher-manifest",
            "data/processed/manifests/teacher_manifest.jsonl",
            "--model-manifest",
            "data/processed/manifests/model_manifest.jsonl",
            "--output-dir",
            "data/processed/dpo",
        ],
        workdir=project_root,
    )

    _run_step(
        [
            python_exe,
            "scripts/train_reference.py",
            "--config",
            "configs/demo_train_reference.yaml",
        ],
        workdir=project_root,
    )

    _run_step(
        [
            python_exe,
            "scripts/train_dpo.py",
            "--config",
            "configs/demo_train_dpo.yaml",
        ],
        workdir=project_root,
    )

    _run_step(
        [
            python_exe,
            "scripts/export_predictions.py",
            "--config",
            "configs/demo_train_dpo.yaml",
            "--checkpoint",
            "outputs/dpo/policy_model_dpo.pt",
            "--input-jsonl",
            "data/processed/dpo/test_dpo.jsonl",
            "--output-jsonl",
            "outputs/frontend/predictions.jsonl",
        ],
        workdir=project_root,
    )

    _run_step(
        [
            python_exe,
            "scripts/export_frontend_payload.py",
            "--input-jsonl",
            "outputs/frontend/predictions.jsonl",
            "--output-jsonl",
            "outputs/frontend/frontend_payload.jsonl",
        ],
        workdir=project_root,
    )

    print("\nDemo 流程已跑完。")
    print("请重点检查以下文件：")
    print("- data/processed/manifests/teacher_manifest.jsonl")
    print("- data/processed/manifests/model_manifest.jsonl")
    print("- data/processed/dpo/train_dpo.jsonl")
    print("- models/reference/reference_model.pt")
    print("- outputs/dpo/policy_model_dpo.pt")
    print("- outputs/frontend/frontend_payload.jsonl")


if __name__ == "__main__":
    main()
