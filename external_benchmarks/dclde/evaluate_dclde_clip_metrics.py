#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.dclde.common import read_csv_rows, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate clip-level whistle/noise metrics on the prepared DCLDE benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument(
        "--clip-positive-min-detections",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="A clip is predicted positive if detection_count is at least this value.",
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def prediction_csv_path(inference_dir: Path, staged_file_name_value: str) -> Path:
    stem = Path(staged_file_name_value).stem
    return inference_dir / stem / f"{stem}.wav_predictions.csv"


def summarize_prediction_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "prediction_found": False,
            "detection_count": 0,
            "mean_confidence": "",
            "max_confidence": "",
        }

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {
            "prediction_found": True,
            "detection_count": 0,
            "mean_confidence": "",
            "max_confidence": "",
        }

    confidences = [float(row["confidence"]) for row in rows]
    return {
        "prediction_found": True,
        "detection_count": len(rows),
        "mean_confidence": float(np.mean(confidences)),
        "max_confidence": float(np.max(confidences)),
    }


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    if labels.size == 0:
        return {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "positive_prediction_rate": 0.0,
        }

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    accuracy = float(np.mean(labels == preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_accuracy = float((recall + specificity) / 2.0)
    positive_prediction_rate = float(np.mean(preds == 1))
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
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else inference_dir / "_dclde_clip_metrics"

    manifest_path = prepared_dir / "manifest_selected.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    manifest_rows = read_csv_rows(manifest_path)
    clip_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        pred_summary = summarize_prediction_csv(prediction_csv_path(inference_dir, row["staged_file_name"]))
        clip_rows.append(
            {
                **row,
                "label": int(row["label"]),
                "prediction_found": bool(pred_summary["prediction_found"]),
                "detection_count": int(pred_summary["detection_count"]),
                "mean_confidence": pred_summary["mean_confidence"],
                "max_confidence": pred_summary["max_confidence"],
            }
        )

    clip_thresholds = sorted({max(1, int(value)) for value in args.clip_positive_min_detections})

    summary_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    per_clip_rows: list[dict[str, object]] = []

    labels = np.asarray([int(row["label"]) for row in clip_rows], dtype=np.int64)
    for threshold in clip_thresholds:
        preds = np.asarray([1 if int(row["detection_count"]) >= threshold else 0 for row in clip_rows], dtype=np.int64)
        metrics = compute_binary_metrics(labels, preds)
        summary_rows.append(
            {
                "clip_positive_min_detections": threshold,
                "total_clips": int(labels.size),
                "positives": int(np.sum(labels == 1)),
                "negatives": int(np.sum(labels == 0)),
                **metrics,
            }
        )
        confusion_rows.extend(
            [
                {
                    "clip_positive_min_detections": threshold,
                    "actual": "negative",
                    "predicted_negative": metrics["tn"],
                    "predicted_positive": metrics["fp"],
                },
                {
                    "clip_positive_min_detections": threshold,
                    "actual": "positive",
                    "predicted_negative": metrics["fn"],
                    "predicted_positive": metrics["tp"],
                },
            ]
        )
        for row, pred in zip(clip_rows, preds):
            per_clip_rows.append(
                {
                    "clip_positive_min_detections": threshold,
                    "true_label": int(row["label"]),
                    "predicted_label": int(pred),
                    "staged_file_name": row["staged_file_name"],
                    "dataset_split": row["dataset_split"],
                    "species": row["species"],
                    "recording": row["recording"],
                    "source_audio_relpath": row["source_audio_relpath"],
                    "annotation_version_used": row["annotation_version_used"],
                    "source_kind": row["source_kind"],
                    "detection_count": row["detection_count"],
                    "mean_confidence": row["mean_confidence"],
                    "max_confidence": row["max_confidence"],
                    "annotation_duration": row["annotation_duration"],
                    "annotation_node_count": row["annotation_node_count"],
                    "annotation_min_freq": row["annotation_min_freq"],
                    "annotation_max_freq": row["annotation_max_freq"],
                    "onset": row["onset"],
                    "offset": row["offset"],
                }
            )

    positive_df = pd.DataFrame([row for row in clip_rows if int(row["label"]) == 1])
    positive_group_rows: list[dict[str, object]] = []
    if not positive_df.empty:
        group_specs = [
            ("all_positives", positive_df),
            *((f"split_{name}", group.copy()) for name, group in positive_df.groupby("dataset_split", dropna=False)),
            *((f"species_{name}", group.copy()) for name, group in positive_df.groupby("species", dropna=False)),
            *(
                (f"annotation_version_{name}", group.copy())
                for name, group in positive_df.groupby("annotation_version_used", dropna=False)
            ),
        ]
        seen_names: set[str] = set()
        for group_name, group_df in group_specs:
            if group_name in seen_names:
                continue
            seen_names.add(group_name)
            row_payload: dict[str, object] = {
                "group_name": group_name,
                "clip_count": int(len(group_df)),
            }
            detection_counts = group_df["detection_count"].astype(int).to_numpy()
            for threshold in clip_thresholds:
                hits = int(np.sum(detection_counts >= threshold))
                row_payload[f"hits_at_{threshold}"] = hits
                row_payload[f"hit_rate_at_{threshold}"] = float(hits / len(group_df))
            positive_group_rows.append(row_payload)

    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "clip_positive_min_detections",
            "total_clips",
            "positives",
            "negatives",
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
        ],
    )
    write_csv(
        output_dir / "confusion_matrix.csv",
        confusion_rows,
        ["clip_positive_min_detections", "actual", "predicted_negative", "predicted_positive"],
    )
    write_csv(
        output_dir / "per_clip.csv",
        per_clip_rows,
        [
            "clip_positive_min_detections",
            "true_label",
            "predicted_label",
            "staged_file_name",
            "dataset_split",
            "species",
            "recording",
            "source_audio_relpath",
            "annotation_version_used",
            "source_kind",
            "detection_count",
            "mean_confidence",
            "max_confidence",
            "annotation_duration",
            "annotation_node_count",
            "annotation_min_freq",
            "annotation_max_freq",
            "onset",
            "offset",
        ],
    )
    write_csv(
        output_dir / "positive_group_summary.csv",
        positive_group_rows,
        ["group_name", "clip_count", *[f"hits_at_{threshold}" for threshold in clip_thresholds], *[f"hit_rate_at_{threshold}" for threshold in clip_thresholds]],
    )

    best_row = max(summary_rows, key=lambda row: (float(row["f1"]), float(row["balanced_accuracy"]))) if summary_rows else {}
    summary_payload = {
        "prepared_dir": str(prepared_dir),
        "inference_dir": str(inference_dir),
        "output_dir": str(output_dir),
        "clip_thresholds": clip_thresholds,
        "summary_rows": summary_rows,
        "best_threshold_metrics": best_row,
    }
    write_json(output_dir / "summary.json", summary_payload)

    print(f"Total clips         : {int(labels.size)}")
    print(f"Positives / negatives: {int(np.sum(labels == 1))} / {int(np.sum(labels == 0))}")
    if best_row:
        print(
            "Best threshold      : "
            f"{best_row['clip_positive_min_detections']} "
            f"(F1={float(best_row['f1']):.4f}, BA={float(best_row['balanced_accuracy']):.4f})"
        )
    print(f"Summary CSV         : {output_dir / 'summary.csv'}")
    print(f"Confusion CSV       : {output_dir / 'confusion_matrix.csv'}")
    print(f"Summary JSON        : {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

