#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot Adriatic benchmark runner: prepare clip-level windows, run "
            "inference once, then evaluate clip metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", default="")
    parser.add_argument("--archive-path", default="")
    parser.add_argument("--full-recording-path", default="")
    parser.add_argument("--whistles-path", default="")
    parser.add_argument("--clicks-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "models" / "run3" / "model_vgg_best.pt"),
    )
    parser.add_argument("--window-seconds", type=float, default=0.4)
    parser.add_argument("--margin-seconds", type=float, default=0.4)
    parser.add_argument("--noise-to-whistle-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-whistle-quality", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--exclude-multiple-whistles", action="store_true")
    parser.add_argument("--ignore-clicks", action="store_true")
    parser.add_argument("--limit-positives", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--target-fs", type=int, default=96_000)
    parser.add_argument(
        "--clip-positive-min-detections",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    python = sys.executable
    output_dir = Path(args.output_dir).expanduser().resolve()

    prepare_cmd = [
        python,
        "external_benchmarks/adriatic/prepare_adriatic_clip_benchmark.py",
        "--output-dir",
        str(output_dir),
        "--window-seconds",
        str(args.window_seconds),
        "--margin-seconds",
        str(args.margin_seconds),
        "--noise-to-whistle-ratio",
        str(args.noise_to_whistle_ratio),
        "--seed",
        str(args.seed),
        "--min-whistle-quality",
        str(args.min_whistle_quality),
    ]
    if args.source_root:
        prepare_cmd.extend(["--source-root", args.source_root])
    if args.archive_path:
        prepare_cmd.extend(["--archive-path", args.archive_path])
    if args.full_recording_path:
        prepare_cmd.extend(["--full-recording-path", args.full_recording_path])
    if args.whistles_path:
        prepare_cmd.extend(["--whistles-path", args.whistles_path])
    if args.clicks_path:
        prepare_cmd.extend(["--clicks-path", args.clicks_path])
    if args.exclude_multiple_whistles:
        prepare_cmd.append("--exclude-multiple-whistles")
    if args.ignore_clicks:
        prepare_cmd.append("--ignore-clicks")
    if args.limit_positives > 0:
        prepare_cmd.extend(["--limit-positives", str(args.limit_positives)])

    inference_dir = output_dir / "inference"
    inference_cmd = [
        python,
        "external_benchmarks/adriatic/run_adriatic_inference.py",
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
    if args.cpu:
        inference_cmd.append("--cpu")

    eval_cmd = [
        python,
        "external_benchmarks/adriatic/evaluate_adriatic_clip_metrics.py",
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

    print(f"Prepared dir  : {output_dir}")
    print(f"Inference dir : {inference_dir}")
    print(f"Metrics dir   : {inference_dir / '_adriatic_clip_metrics'}")


if __name__ == "__main__":
    main()
