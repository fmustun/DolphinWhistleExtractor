#!/usr/bin/env python3
"""Rewrite f0_spectrogram with the legacy plotting style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_classification_f0_review import render_legacy_f0_plot
from repair_classification_f0_dataset import ensure_output_dir_is_empty, maybe_push_to_hub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite only the f0_spectrogram column of a saved DatasetDict using "
            "the legacy specgram + scatter renderer."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Input DatasetDict path saved with save_to_disk().",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output DatasetDict path where the rewritten dataset is saved.",
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory where a summary JSON is written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size passed to Dataset.map().",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Confidence threshold used by the legacy renderer.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="NFFT used by the legacy renderer.",
    )
    parser.add_argument(
        "--ymin-hz",
        type=float,
        default=0.0,
        help="Lower y-axis bound in Hz used by the legacy renderer.",
    )
    parser.add_argument(
        "--ymax-hz",
        type=float,
        default=22_000.0,
        help="Upper y-axis bound in Hz used by the legacy renderer.",
    )
    parser.add_argument(
        "--push-repo-id",
        default=None,
        help="Optional dataset repo id passed to push_to_hub().",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Config name used when pushing to the Hub.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token used for push_to_hub().",
    )
    parser.add_argument(
        "--commit-message",
        default="Rewrite classification F0 spectrograms with legacy renderer",
        help="Commit message used when pushing to the Hub.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Shard size passed to push_to_hub().",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be strictly positive.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1].")
    if args.nfft <= 0:
        raise ValueError("--nfft must be strictly positive.")
    if args.ymin_hz < 0.0:
        raise ValueError("--ymin-hz must be >= 0.")
    if args.ymax_hz <= args.ymin_hz:
        raise ValueError("--ymax-hz must be strictly greater than --ymin-hz.")
    if bool(args.push_repo_id) != bool(args.config_name):
        raise ValueError("--push-repo-id and --config-name must be provided together.")


def rewrite_split(
    *,
    split_name: str,
    split_dataset: Dataset,
    batch_size: int,
    confidence_threshold: float,
    nfft: int,
    ymin_hz: float,
    ymax_hz: float,
) -> Dataset:
    output_features = Features(dict(split_dataset.features))
    output_features["f0_spectrogram"] = HFImage()

    def rewrite_batch(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        rows = []
        for index in range(len(batch["audio"])):
            row = {column_name: batch[column_name][index] for column_name in batch.keys()}
            rows.append(
                render_legacy_f0_plot(
                    row,
                    confidence_threshold=confidence_threshold,
                    nfft=nfft,
                    ymin_hz=ymin_hz,
                    ymax_hz=ymax_hz,
                )
            )
        return {"f0_spectrogram": rows}

    return split_dataset.map(
        rewrite_batch,
        batched=True,
        batch_size=batch_size,
        features=output_features,
        load_from_cache_file=False,
        desc=f"Rewriting {split_name}",
    )


def write_artifacts(
    *,
    artifacts_dir: Path,
    input_dataset_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    dataset_dict: DatasetDict,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dataset_path": str(input_dataset_path.resolve()),
        "output_dataset_path": str(output_dir.resolve()),
        "legacy_renderer": {
            "confidence_threshold": float(args.confidence_threshold),
            "nfft": int(args.nfft),
            "ymin_hz": float(args.ymin_hz),
            "ymax_hz": float(args.ymax_hz),
        },
        "splits": {
            split_name: {
                "num_rows": int(dataset_dict[split_name].num_rows),
                "columns": list(dataset_dict[split_name].column_names),
            }
            for split_name in dataset_dict.keys()
        },
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)

    dataset_dict = load_from_disk(str(dataset_path))
    rewritten_splits = {}
    for split_name in dataset_dict.keys():
        rewritten_splits[split_name] = rewrite_split(
            split_name=split_name,
            split_dataset=dataset_dict[split_name],
            batch_size=args.batch_size,
            confidence_threshold=args.confidence_threshold,
            nfft=args.nfft,
            ymin_hz=args.ymin_hz,
            ymax_hz=args.ymax_hz,
        )

    rewritten_dataset = DatasetDict(rewritten_splits)
    ensure_output_dir_is_empty(output_dir)
    rewritten_dataset.save_to_disk(str(output_dir))
    write_artifacts(
        artifacts_dir=artifacts_dir,
        input_dataset_path=dataset_path,
        output_dir=output_dir,
        args=args,
        dataset_dict=rewritten_dataset,
    )

    maybe_push_to_hub(
        dataset_dict=rewritten_dataset,
        repo_id=args.push_repo_id,
        config_name=args.config_name,
        token=args.token,
        commit_message=args.commit_message,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
