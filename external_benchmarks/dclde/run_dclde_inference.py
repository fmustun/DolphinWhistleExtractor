#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREDICT_DIR = REPO_ROOT / "Predict_and_extract"
if str(PREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(PREDICT_DIR))

from external_benchmarks.dclde.common import read_csv_rows, write_json  # noqa: E402


def resample_audio_if_needed_scipy(
    audio: np.ndarray,
    fs: int,
    target_fs: int | None,
) -> tuple[int, np.ndarray]:
    audio = np.asarray(audio, dtype=np.float32)
    if target_fs is None or int(fs) == int(target_fs):
        return int(fs), audio

    source_fs = int(fs)
    destination_fs = int(target_fs)
    scale = gcd(source_fs, destination_fs)
    up = destination_fs // scale
    down = source_fs // scale
    resampled = resample_poly(audio, up, down)
    return destination_fs, np.asarray(resampled, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current VGG16 whistle detector on a prepared DCLDE flat "
            "staging directory produced by prepare_dclde_clip_benchmark.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "models" / "run3" / "model_vgg_best.pt"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--target-fs", type=int, default=96_000)
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-positive-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prepared_dir = Path(args.prepared_dir).expanduser().resolve()
    manifest_path = prepared_dir / "manifest_selected.csv"
    flat_recordings_dir = prepared_dir / "flat_recordings"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")
    if not flat_recordings_dir.is_dir():
        raise FileNotFoundError(f"Flat recordings directory not found: {flat_recordings_dir}")

    manifest_rows = read_csv_rows(manifest_path)
    staged_files = [row["staged_file_name"] for row in manifest_rows if row.get("staged_file_name")]
    if args.limit > 0:
        staged_files = staged_files[: args.limit]
    if not staged_files:
        raise SystemExit("No staged files available to process.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else prepared_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_payload = {
        "prepared_dir": str(prepared_dir),
        "manifest_path": str(manifest_path),
        "flat_recordings_dir": str(flat_recordings_dir),
        "output_dir": str(output_dir),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "threshold": args.threshold,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "target_fs": args.target_fs,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "cpu": args.cpu,
        "save_positive_images": args.save_positive_images,
        "requested_files": len(staged_files),
    }
    write_json(output_dir / "run_config.json", run_payload)

    import predict_and_extract_online as inference_module
    import shared.whistle_torch as whistle_torch

    whistle_torch.resample_audio_if_needed = resample_audio_if_needed_scipy
    inference_module.resample_audio_if_needed = resample_audio_if_needed_scipy
    process_predict_extract = inference_module.process_predict_extract

    process_predict_extract(
        recording_folder_path=str(flat_recordings_dir),
        saving_folder=str(output_dir),
        cut_low_freq=2.0,
        cut_high_freq=22.0,
        start_time=args.start_time,
        end_time=args.end_time,
        batch_size=args.batch_size,
        save_positives=args.save_positive_images,
        model_path=args.model_path,
        binary_threshold=args.threshold,
        max_workers=args.max_workers,
        specific_files=staged_files,
        target_fs=args.target_fs if args.target_fs > 0 else None,
        cpu_only=args.cpu,
    )

    print(f"Requested files : {len(staged_files)}")
    print(f"Prepared dir    : {prepared_dir}")
    print(f"Inference dir   : {output_dir}")
    print(
        "Next step       : python external_benchmarks/dclde/"
        "evaluate_dclde_clip_metrics.py "
        f"--prepared-dir {prepared_dir} --inference-dir {output_dir}"
    )


if __name__ == "__main__":
    main()

