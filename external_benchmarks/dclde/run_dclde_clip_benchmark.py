#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot runner for the DCLDE clip-level whistle/noise benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "models" / "run3" / "model_vgg_best.pt"),
    )
    parser.add_argument("--subset", choices=("development", "evaluation", "both"), default="evaluation")
    parser.add_argument(
        "--annotation-version",
        choices=("2011", "2025_auto", "2025_only"),
        default="2011",
    )
    parser.add_argument("--species", nargs="+", default=[])
    parser.add_argument("--window-seconds", type=float, default=0.4)
    parser.add_argument("--margin-seconds", type=float, default=0.4)
    parser.add_argument("--noise-to-whistle-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-positives", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--target-fs", type=int, default=96_000)
    parser.add_argument("--clip-positive-min-detections", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-zero-annotation-files", action="store_true")
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Download the requested public NOAA DCLDE subset into --source-root before preparing clips.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing generated files in --output-dir before preparing and rerunning the benchmark.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def dataset_root_candidates(source_root: Path) -> list[Path]:
    return [
        source_root,
        source_root / "dclde_2011",
        source_root / "dclde" / "2011" / "dclde_2011",
    ]


def has_dataset_root(source_root: Path) -> bool:
    for candidate in dataset_root_candidates(source_root):
        if (candidate / "README.md").exists() and (
            (candidate / "development").is_dir() or (candidate / "evaluation").is_dir()
        ):
            return True
    return False


def maybe_download_source(args: argparse.Namespace, source_root: Path) -> None:
    source_exists = source_root.exists()
    source_ready = source_exists and has_dataset_root(source_root)
    if source_ready:
        return

    if not args.download_first:
        if not source_exists:
            raise FileNotFoundError(
                f"The requested DCLDE source root does not exist yet: {source_root}\n"
                "Run the NOAA downloader first, for example:\n"
                "python external_benchmarks/dclde/download_dclde_subset.py "
                f"--output-dir {source_root} --subset {args.subset} --annotated-only --include-docs"
            )
        raise FileNotFoundError(
            f"Could not find a usable DCLDE dataset under: {source_root}\n"
            "If this directory is meant to be populated automatically, rerun with `--download-first`."
        )

    download_cmd = [
        sys.executable,
        "external_benchmarks/dclde/download_dclde_subset.py",
        "--output-dir",
        str(source_root),
        "--subset",
        args.subset,
        "--annotated-only",
        "--include-docs",
    ]
    if args.annotation_version != "2011":
        download_cmd.append("--include-annotations2025")
    if args.species:
        download_cmd.extend(["--species", *args.species])

    run_command(download_cmd)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    maybe_download_source(args, source_root)

    prepare_cmd = [
        sys.executable,
        "external_benchmarks/dclde/prepare_dclde_clip_benchmark.py",
        "--source-root",
        str(source_root),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--subset",
        args.subset,
        "--annotation-version",
        args.annotation_version,
        "--window-seconds",
        str(args.window_seconds),
        "--margin-seconds",
        str(args.margin_seconds),
        "--noise-to-whistle-ratio",
        str(args.noise_to_whistle_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.species:
        prepare_cmd.extend(["--species", *args.species])
    if args.limit_positives > 0:
        prepare_cmd.extend(["--limit-positives", str(args.limit_positives)])
    if args.skip_zero_annotation_files:
        prepare_cmd.append("--skip-zero-annotation-files")
    if args.overwrite_output:
        prepare_cmd.append("--overwrite-output")

    output_dir = Path(args.output_dir).expanduser().resolve()
    inference_dir = output_dir / "inference"
    inference_cmd = [
        sys.executable,
        "external_benchmarks/dclde/run_dclde_inference.py",
        "--prepared-dir",
        str(output_dir),
        "--output-dir",
        str(inference_dir),
        "--model-path",
        str(Path(args.model_path).expanduser().resolve()),
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

    evaluate_cmd = [
        sys.executable,
        "external_benchmarks/dclde/evaluate_dclde_clip_metrics.py",
        "--prepared-dir",
        str(output_dir),
        "--inference-dir",
        str(inference_dir),
        "--clip-positive-min-detections",
        *[str(value) for value in args.clip_positive_min_detections],
    ]

    run_command(prepare_cmd)
    run_command(inference_cmd)
    run_command(evaluate_cmd)

    print(f"Prepared dir  : {output_dir}")
    print(f"Inference dir : {inference_dir}")
    print(f"Metrics dir   : {inference_dir / '_dclde_clip_metrics'}")


if __name__ == "__main__":
    main()
