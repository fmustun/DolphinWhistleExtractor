#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.wmmsd.ultimate_suite import suite_species


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot WMMSD ultimate benchmark runner: prepare all suite species, "
            "run inference once, then evaluate multiple hard scenarios."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-root",
        default="",
        help="Local WMMSD root directory to scan recursively for audio files.",
    )
    source_group.add_argument(
        "--hf-dataset-id",
        default="",
        help="Optional Hugging Face dataset id such as confit/wmms-parquet.",
    )
    parser.add_argument("--hf-config", default="", help="Optional Hugging Face dataset config.")
    parser.add_argument(
        "--hf-splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to load when using --hf-dataset-id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Benchmark root output directory.",
    )
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "models" / "run3" / "model_vgg_best.pt"),
        help="Path to the whistle CNN checkpoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Positive-class threshold.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel worker threads.")
    parser.add_argument("--target-fs", type=int, default=96_000, help="Target sample rate.")
    parser.add_argument(
        "--clip-positive-min-detections",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Clip-level vote thresholds used during suite evaluation.",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--copy-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on staged files.")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    python = sys.executable
    output_dir = Path(args.output_dir).expanduser().resolve()
    selected_species = suite_species()

    prepare_cmd = [
        python,
        "external_benchmarks/wmmsd/prepare_wmmsd_manifest.py",
        "--output-dir",
        str(output_dir),
        "--copy-mode",
        args.copy_mode,
        "--species",
        *selected_species,
    ]
    if args.limit > 0:
        prepare_cmd.extend(["--limit", str(args.limit)])
    if args.source_root:
        prepare_cmd.extend(["--source-root", args.source_root])
    else:
        prepare_cmd.extend(["--hf-dataset-id", args.hf_dataset_id])
        if args.hf_config:
            prepare_cmd.extend(["--hf-config", args.hf_config])
        if args.hf_splits:
            prepare_cmd.extend(["--hf-splits", *args.hf_splits])

    inference_dir = output_dir / "inference_ultimate"
    inference_cmd = [
        python,
        "external_benchmarks/wmmsd/run_wmmsd_inference.py",
        "--prepared-dir",
        str(output_dir),
        "--output-dir",
        str(inference_dir),
        "--model-path",
        args.model_path,
        "--threshold",
        str(args.threshold),
        "--batch-size",
        str(args.batch_size),
        "--max-workers",
        str(args.max_workers),
        "--target-fs",
        str(args.target_fs),
    ]
    if args.limit > 0:
        inference_cmd.extend(["--limit", str(args.limit)])
    if args.cpu:
        inference_cmd.append("--cpu")

    eval_cmd = [
        python,
        "external_benchmarks/wmmsd/evaluate_wmmsd_ultimate_suite.py",
        "--prepared-dir",
        str(output_dir),
        "--inference-dir",
        str(inference_dir),
        "--clip-positive-min-detections",
        *[str(value) for value in args.clip_positive_min_detections],
    ]

    run_command(prepare_cmd)
    run_command(inference_cmd)
    run_command(eval_cmd)

    print(f"Prepared dir       : {output_dir}")
    print(f"Inference dir      : {inference_dir}")
    print(
        "Ultimate suite CSV : "
        f"{inference_dir / '_wmmsd_ultimate_suite' / 'scenario_summary.csv'}"
    )


if __name__ == "__main__":
    main()
