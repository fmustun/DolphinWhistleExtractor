#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwhistle_detection_finetuning.common import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPO_ID,
    DEFAULT_SEED,
    DEFAULT_SOURCE_CACHE_ROOT,
    DEFAULT_SPLIT_POLICY,
    DEFAULT_SOURCE_REPO_ID,
    DEFAULT_SOURCE_REVISION,
    DEFAULT_TEST_SIZE,
    build_dataset_dict,
    build_split_manifest,
    build_summary,
    ensure_clean_dir,
    load_source_dataframe,
    resolve_source_snapshot_dir,
    write_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild dolphinteam/OpenWhistle-1.0-Detection-Finetuning locally "
            "from the cached dolphinteam/DophinWhistle-Detection-Finetuning_OLD snapshot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-cache-root",
        default=str(DEFAULT_SOURCE_CACHE_ROOT),
        help="HF hub cache root for the legacy detection dataset.",
    )
    parser.add_argument(
        "--source-snapshot-dir",
        default=None,
        help="Explicit legacy snapshot directory. Overrides --source-cache-root.",
    )
    parser.add_argument(
        "--source-revision",
        default=DEFAULT_SOURCE_REVISION,
        help="Legacy revision name to resolve in the local HF cache.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save the rebuilt DatasetDict with save_to_disk().",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Where to write README, summary.json, and split_manifest.csv.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target repo id documented in the generated artifacts.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction assigned to the test split within each legacy source label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used for the deterministic split manifest.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("stratified", "grouped_rigorous"),
        default=DEFAULT_SPLIT_POLICY,
        help="How to split the legacy clips into train and test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit applied independently to train and test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output/artifact directories.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be strictly positive when provided.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = Path(args.output_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    source_snapshot_dir = resolve_source_snapshot_dir(
        source_cache_root=Path(args.source_cache_root),
        source_snapshot_dir=Path(args.source_snapshot_dir).resolve()
        if args.source_snapshot_dir
        else None,
        revision=args.source_revision,
    )

    ensure_clean_dir(output_dir, overwrite=args.overwrite)
    ensure_clean_dir(artifacts_dir, overwrite=args.overwrite)

    source_df = load_source_dataframe(source_snapshot_dir)
    manifest_df = build_split_manifest(
        source_df,
        test_size=args.test_size,
        seed=args.seed,
        split_policy=args.split_policy,
    )
    dataset_dict, split_frames = build_dataset_dict(
        manifest_df,
        limit_per_split=args.limit,
        selection_seed=args.seed,
    )
    dataset_dict.save_to_disk(str(output_dir))

    summary = build_summary(
        source_snapshot_dir=source_snapshot_dir,
        manifest_df=manifest_df,
        split_frames=split_frames,
        repo_id=args.repo_id,
        source_repo_id=DEFAULT_SOURCE_REPO_ID,
        test_size=args.test_size,
        seed=args.seed,
        limit_per_split=args.limit,
        split_policy=args.split_policy,
    )
    artifact_paths = write_artifacts(
        artifacts_dir=artifacts_dir,
        manifest_df=manifest_df,
        summary=summary,
    )

    print(f"Source snapshot : {source_snapshot_dir}")
    print(f"Dataset saved   : {output_dir}")
    print(f"Artifacts dir   : {artifacts_dir}")
    print(f"README          : {artifact_paths['readme_path']}")
    print(f"Summary         : {artifact_paths['summary_path']}")
    print(f"Split manifest  : {artifact_paths['manifest_path']}")
    print("Rows by split   :", summary["rows_by_split"])
    print("Binary counts   :", summary["binary_counts_by_split"])


if __name__ == "__main__":
    main()
