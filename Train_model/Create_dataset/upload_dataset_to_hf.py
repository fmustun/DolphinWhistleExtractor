"""
Upload the dolphin whistle detection training dataset to Hugging Face Hub
using the `datasets` library (Parquet shards).

Instead of uploading 98k individual LFS files (which immediately hits the HF
rate limit of 1000 req/5 min), this script packs everything into a small number
of Parquet shards (~25 files) with embedded audio and spectrogram columns.

Dataset schema:
  audio        – Audio feature (WAV binary, decoded on demand)
  spectrogram  – Image feature (JPG binary)
  label        – ClassLabel  0=noise / 1=whistle
  file_name    – original filename, e.g. noise_038306.wav
  recording    – source recording name
  onset        – clip start time (s)
  offset       – clip end time   (s)

Usage:
  python upload_dataset_to_hf.py --repo <org/repo-name> [--private] [--token <HF_TOKEN>]

The HF token can also be supplied via the HF_TOKEN environment variable or a
prior `huggingface-cli login`.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Audio, ClassLabel, Dataset, Features, Image, Value
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/media/DOLPHIN1/Wh_detection_database_NEURIPS/all_years_100k_dataset")
WAV_DIR  = BASE_DIR / "wav_files"
SPEC_DIR = BASE_DIR / "spectrograms"
CSV_PATH = BASE_DIR / "dataset.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Build dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset() -> Dataset:
    print("Reading dataset.csv …")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows")

    # Resolve audio paths (use local WAV_DIR; ignore the absolute path in CSV)
    audio_paths = [str(WAV_DIR / Path(p).name) for p in df["file_path"]]

    # Spectrograms live in negatives/ (label=0) or positives/ (label=1)
    def spec_path(file_path: str, label: int) -> str:
        stem = Path(file_path).stem          # e.g. noise_038306
        sub  = "negatives" if label == 0 else "positives"
        return str(SPEC_DIR / sub / f"{stem}.jpg")

    spec_paths = [spec_path(p, l) for p, l in zip(df["file_path"], df["label"])]
    file_names = [Path(p).name for p in df["file_path"]]

    features = Features({
        "audio":       Audio(sampling_rate=None),   # preserves original sample rate
        "spectrogram": Image(),
        "label":       ClassLabel(names=["noise", "whistle"]),
        "file_name":   Value("string"),
        "recording":   Value("string"),
        "onset":       Value("float32"),
        "offset":      Value("float32"),
    })

    print("Building Dataset object …")
    ds = Dataset.from_dict(
        {
            "audio":       audio_paths,
            "spectrogram": spec_paths,
            "label":       df["label"].tolist(),
            "file_name":   file_names,
            "recording":   df["recording"].tolist(),
            "onset":       df["onset"].astype(float).tolist(),
            "offset":      df["offset"].astype(float).tolist(),
        },
        features=features,
    )
    return ds


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload dolphin whistle dataset to Hugging Face Hub (Parquet)."
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="ORG/REPO",
        help="HF dataset repo ID, e.g. dolphinteam/CNN_training_dataset",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the repo as private (default: public)",
    )
    parser.add_argument(
        "--token",
        default=None,
        metavar="HF_TOKEN",
        help="HF API token (overrides HF_TOKEN env var and cached login)",
    )
    parser.add_argument(
        "--shard-size",
        default="500MB",
        metavar="SIZE",
        help="Max size per Parquet shard, e.g. 500MB (default: 500MB)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    else:
        print("No --token provided; using cached HF credentials.")

    # ── Build ─────────────────────────────────────────────────────────────────
    ds = build_dataset()
    print(f"\nDataset ready: {ds}")

    # ── Push ──────────────────────────────────────────────────────────────────
    print(f"\nPushing to https://huggingface.co/datasets/{args.repo} …")
    print("(This encodes all audio + images into Parquet shards — may take a while.)\n")
    try:
        ds.push_to_hub(
            repo_id=args.repo,
            split="train",
            private=args.private,
            token=token,
            max_shard_size=args.shard_size,
        )
    except HfHubHTTPError as e:
        print(f"\nHTTP error during push: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nUpload finished.")
    print(f"Dataset page → https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
