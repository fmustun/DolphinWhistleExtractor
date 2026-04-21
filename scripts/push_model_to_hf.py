#!/usr/bin/env python3
"""Publish the final VGG whistle model to a Hugging Face model repo."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a trained whistle CNN checkpoint and its reports to Hugging Face.",
    )
    parser.add_argument(
        "--repo-id",
        default="dolphinteam/DolphinWhistle-CNN-VGG16",
        help="Target Hugging Face model repo.",
    )
    parser.add_argument(
        "--model-path",
        default="models/run_vgg_final/model_vgg_final_best.pt",
        help="Checkpoint to upload.",
    )
    parser.add_argument(
        "--summary-path",
        default="reports/run3/run_summary.json",
        help="Run summary JSON exported by Train_torch.py.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports/run3",
        help="Directory containing CSV reports for the run.",
    )
    parser.add_argument(
        "--figs-dir",
        default="figs/run3",
        help="Directory containing plots for the run.",
    )
    parser.add_argument(
        "--model-filename",
        default="model_vgg_final_best.pt",
        help="Filename to use in the uploaded repo.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist yet.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload VGG16 whistle detection model",
        help="Commit message used for the upload.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, uses local HF login state.",
    )
    return parser.parse_args()


def require_dependency() -> object:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "Missing dependency 'huggingface_hub'. Install it with "
            "`pip install huggingface_hub` before running this script."
        ) from exc
    return HfApi


def read_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_model_card(summary: dict, repo_id: str, model_filename: str) -> str:
    metrics = summary["metrics"]
    split_summary = summary["split_summary"]
    artifacts = summary["artifacts"]

    return f"""---
license: mit
library_name: pytorch
tags:
- bioacoustics
- audio-classification
- spectrogram
- dolphin
- whistle-detection
pipeline_tag: audio-classification
---

# {repo_id}

PyTorch VGG16-based classifier for local dolphin whistle vs noise detection on spectrograms.

## Checkpoint

- File: `{model_filename}`
- Best epoch: `{summary["best_epoch"]}`
- Best validation loss: `{summary["best_val_loss"]:.4f}`
- Input: `224x224` RGB spectrogram
- Task: binary classification `whistle` vs `noise`

## Dataset Protocol

- Source dataset: `{summary["dataset_source"]}`
- Training input source: `{summary["train_input_source"]}`
- Session-disjoint train/validation/test split
- Train: `{split_summary["train"]["rows"]}` windows across `{split_summary["train"]["sessions"]}` sessions
- Validation: `{split_summary["validation"]["rows"]}` windows across `{split_summary["validation"]["sessions"]}` sessions
- Test: `{split_summary["test"]["rows"]}` windows across `{split_summary["test"]["sessions"]}` sessions
- Test protocol: expanded manual classification evaluation built from the Hugging Face
  `all` config (`8354` whistle windows + matched noise)

## Metrics

### Validation

- Loss: `{metrics["validation"]["loss"]:.4f}`
- Accuracy: `{metrics["validation"]["accuracy"]:.4f}`
- F1: `{metrics["validation"]["f1"]:.4f}`
- Precision: `{metrics["validation"]["precision"]:.4f}`
- Recall: `{metrics["validation"]["recall"]:.4f}`

### Test

- Loss: `{metrics["test"]["loss"]:.4f}`
- Accuracy: `{metrics["test"]["accuracy"]:.4f}`
- F1: `{metrics["test"]["f1"]:.4f}`
- Precision: `{metrics["test"]["precision"]:.4f}`
- Recall: `{metrics["test"]["recall"]:.4f}`

## Included Artifacts

- `run_summary.json`
- `validation_confusion_matrix.csv`
- `test_confusion_matrix.csv`
- `validation_session_metrics.csv`
- `test_session_metrics.csv`
- training and ROC plots in `figures/`

## Notes

- Confusion matrix counts are listed in `run_summary.json`.
- This repo stores the raw checkpoint; loading/inference is expected to happen from the
  DolphinWhistleExtractor codebase.
- Training artifacts were exported from:
  - model: `{artifacts["model"]}`
  - reports: `{artifacts["reports_dir"]}`
  - figures: `{artifacts["figures_dir"]}`
"""


def stage_upload_dir(
    model_path: Path,
    summary_path: Path,
    reports_dir: Path,
    figs_dir: Path,
    model_filename: str,
    repo_id: str,
) -> Path:
    summary = read_summary(summary_path)
    tmpdir = Path(tempfile.mkdtemp(prefix="hf_whistle_model_"))

    shutil.copy2(model_path, tmpdir / model_filename)
    shutil.copy2(summary_path, tmpdir / "run_summary.json")

    for csv_name in (
        "validation_confusion_matrix.csv",
        "test_confusion_matrix.csv",
        "validation_session_metrics.csv",
        "test_session_metrics.csv",
    ):
        source = reports_dir / csv_name
        if source.exists():
            shutil.copy2(source, tmpdir / csv_name)

    figures_dir = tmpdir / "figures"
    figures_dir.mkdir(exist_ok=True)
    for fig_name in (
        "metrics_training.png",
        "roc_validation_test.png",
        "validation_confusion_matrix.png",
        "test_confusion_matrix.png",
    ):
        source = figs_dir / fig_name
        if source.exists():
            shutil.copy2(source, figures_dir / fig_name)

    readme = build_model_card(summary, repo_id=repo_id, model_filename=model_filename)
    (tmpdir / "README.md").write_text(readme, encoding="utf-8")
    return tmpdir


def main() -> None:
    args = parse_args()
    HfApi = require_dependency()

    model_path = Path(args.model_path)
    summary_path = Path(args.summary_path)
    reports_dir = Path(args.reports_dir)
    figs_dir = Path(args.figs_dir)

    for path in (model_path, summary_path, reports_dir, figs_dir):
        if not path.exists():
            raise SystemExit(f"Missing required path: {path}")

    upload_dir = stage_upload_dir(
        model_path=model_path,
        summary_path=summary_path,
        reports_dir=reports_dir,
        figs_dir=figs_dir,
        model_filename=args.model_filename,
        repo_id=args.repo_id,
    )

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(upload_dir),
        commit_message=args.commit_message,
    )
    print(f"Uploaded model package to https://huggingface.co/{args.repo_id}")
    print(f"Staged upload contents in: {upload_dir}")


if __name__ == "__main__":
    main()
