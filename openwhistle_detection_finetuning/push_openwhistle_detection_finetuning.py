#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk
from huggingface_hub import HfApi

from openwhistle_detection_finetuning.common import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPO_ID,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push a locally rebuilt OpenWhistle 1.0 detection finetuning dataset "
            "to Hugging Face, plus README and build artifacts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local DatasetDict path produced by build_openwhistle_detection_finetuning.py.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Artifacts directory produced by build_openwhistle_detection_finetuning.py.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Falls back to local login state.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private if it does not already exist.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Shard size passed to push_to_hub().",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload OpenWhistle 1.0 detection finetuning dataset",
        help="Commit message used for both parquet upload and artifact uploads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    readme_path = artifacts_dir / "README.md"
    summary_path = artifacts_dir / "summary.json"
    manifest_path = artifacts_dir / "split_manifest.csv"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset dir: {dataset_dir}")
    for path in (readme_path, summary_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact: {path}")

    dataset_dict = load_from_disk(str(dataset_dir))
    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    dataset_dict.push_to_hub(
        repo_id=args.repo_id,
        token=args.token,
        max_shard_size=args.max_shard_size,
        embed_external_files=True,
        commit_message=args.commit_message,
        private=args.private,
    )

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
        commit_message=args.commit_message,
    )
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo="artifacts/summary.json",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
        commit_message=args.commit_message,
    )
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="artifacts/split_manifest.csv",
        repo_id=args.repo_id,
        repo_type="dataset",
        token=args.token,
        commit_message=args.commit_message,
    )

    print(f"Dataset pushed to {args.repo_id}")
    print(f"README uploaded from {readme_path}")
    print(f"Summary uploaded from {summary_path}")
    print(f"Split manifest uploaded from {manifest_path}")


if __name__ == "__main__":
    main()
