#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.wmmsd.common import read_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute binary clip-level metrics for a WMMSD inference run. "
            "A clip is predicted positive if its prediction CSV contains at least one detection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prepared-dir",
        required=True,
        help="Directory produced by prepare_wmmsd_manifest.py.",
    )
    parser.add_argument(
        "--inference-dir",
        required=True,
        help="Inference output directory produced by run_wmmsd_inference.py.",
    )
    parser.add_argument(
        "--positive-species",
        nargs="+",
        required=True,
        help="Species treated as positive class.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where evaluation CSV/JSON files will be written.",
    )
    return parser.parse_args()


def prediction_csv_path(inference_dir: Path, staged_file_name: str) -> Path:
    stem = Path(staged_file_name).stem
    return inference_dir / stem / f"{stem}.wav_predictions.csv"


def summarize_prediction_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "prediction_found": False,
            "detection_count": 0,
            "predicted_label": 0,
            "mean_confidence": "",
            "max_confidence": "",
        }

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {
            "prediction_found": True,
            "detection_count": 0,
            "predicted_label": 0,
            "mean_confidence": "",
            "max_confidence": "",
        }

    confidences = [float(row["confidence"]) for row in rows]
    return {
        "prediction_found": True,
        "detection_count": len(rows),
        "predicted_label": 1,
        "mean_confidence": float(np.mean(confidences)),
        "max_confidence": float(np.max(confidences)),
    }


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    accuracy = float(np.mean(labels == preds)) if labels.size else 0.0
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((recall + specificity) / 2.0)
    positive_prediction_rate = float(np.mean(preds == 1)) if preds.size else 0.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "positive_prediction_rate": positive_prediction_rate,
    }


def main() -> None:
    args = parse_args()

    prepared_dir = Path(args.prepared_dir).expanduser().resolve()
    inference_dir = Path(args.inference_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else inference_dir / "_wmmsd_binary_clip_metrics"
    )

    manifest_path = prepared_dir / "manifest_selected.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    positive_species = {str(value).strip() for value in args.positive_species}
    manifest_rows = read_csv_rows(manifest_path)

    per_clip_rows: list[dict[str, object]] = []
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    labels: list[int] = []
    preds: list[int] = []

    for row in manifest_rows:
        species = str(row["species"])
        true_label = 1 if species in positive_species else 0
        staged_file_name = row["staged_file_name"]
        prediction_path = prediction_csv_path(inference_dir, staged_file_name)
        pred_summary = summarize_prediction_csv(prediction_path)
        predicted_label = int(pred_summary["predicted_label"])

        clip_row = {
            "species": species,
            "source_rel_path": row["source_rel_path"],
            "staged_file_name": staged_file_name,
            "prediction_csv": str(prediction_path),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "detection_count": int(pred_summary["detection_count"]),
            "prediction_found": bool(pred_summary["prediction_found"]),
            "mean_confidence": pred_summary["mean_confidence"],
            "max_confidence": pred_summary["max_confidence"],
        }
        per_clip_rows.append(clip_row)
        grouped_rows[species].append(clip_row)
        labels.append(true_label)
        preds.append(predicted_label)

    labels_np = np.asarray(labels, dtype=np.int64)
    preds_np = np.asarray(preds, dtype=np.int64)
    overall_metrics = compute_binary_metrics(labels_np, preds_np)

    per_species_rows: list[dict[str, object]] = []
    for species in sorted(grouped_rows):
        rows = grouped_rows[species]
        species_labels = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
        species_preds = np.asarray([int(row["predicted_label"]) for row in rows], dtype=np.int64)
        metrics = compute_binary_metrics(species_labels, species_preds)
        per_species_rows.append(
            {
                "species": species,
                "clip_count": len(rows),
                **metrics,
                "mean_detections_per_clip": float(
                    np.mean([int(row["detection_count"]) for row in rows])
                ),
            }
        )

    confusion_rows = [
        {"actual": "negative", "predicted_negative": overall_metrics["tn"], "predicted_positive": overall_metrics["fp"]},
        {"actual": "positive", "predicted_negative": overall_metrics["fn"], "predicted_positive": overall_metrics["tp"]},
    ]

    summary_payload = {
        "prepared_dir": str(prepared_dir),
        "inference_dir": str(inference_dir),
        "positive_species": sorted(positive_species),
        **overall_metrics,
        "total_clips": int(labels_np.size),
        "positives": int(np.sum(labels_np == 1)),
        "negatives": int(np.sum(labels_np == 0)),
    }

    write_csv(
        output_dir / "per_clip.csv",
        per_clip_rows,
        [
            "species",
            "source_rel_path",
            "staged_file_name",
            "prediction_csv",
            "true_label",
            "predicted_label",
            "detection_count",
            "prediction_found",
            "mean_confidence",
            "max_confidence",
        ],
    )
    write_csv(
        output_dir / "per_species.csv",
        per_species_rows,
        [
            "species",
            "clip_count",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "balanced_accuracy",
            "positive_prediction_rate",
            "mean_detections_per_clip",
        ],
    )
    write_csv(
        output_dir / "confusion_matrix.csv",
        confusion_rows,
        ["actual", "predicted_negative", "predicted_positive"],
    )
    write_json(output_dir / "summary.json", summary_payload)

    print(f"Total clips         : {summary_payload['total_clips']}")
    print(f"Positives / negatives: {summary_payload['positives']} / {summary_payload['negatives']}")
    print(
        f"Confusion matrix    : tn={summary_payload['tn']} fp={summary_payload['fp']} "
        f"fn={summary_payload['fn']} tp={summary_payload['tp']}"
    )
    print(f"Accuracy            : {summary_payload['accuracy']:.4f}")
    print(f"Precision           : {summary_payload['precision']:.4f}")
    print(f"Recall              : {summary_payload['recall']:.4f}")
    print(f"F1                  : {summary_payload['f1']:.4f}")
    print(f"Specificity         : {summary_payload['specificity']:.4f}")
    print(f"Balanced accuracy   : {summary_payload['balanced_accuracy']:.4f}")
    print(f"Confusion CSV       : {output_dir / 'confusion_matrix.csv'}")
    print(f"Summary JSON        : {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
