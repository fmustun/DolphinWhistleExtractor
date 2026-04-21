#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.wmmsd.common import read_csv_rows, write_csv, write_json
from external_benchmarks.wmmsd.whistle_noise_suite import (
    OPTIONAL_AMBIGUOUS_POSITIVES,
    WHISTLE_NOISE_PROXY_SCENARIOS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a WMMSD inference run under whistle/noise proxy scenarios. "
            "This treats whistle-capable delphinids as positives instead of using "
            "species identity as the target."
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
        "--clip-positive-min-detections",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help=(
            "Clip-level vote thresholds. A clip is predicted positive only if its "
            "detection_count is at least this value."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where suite CSV/JSON outputs will be written.",
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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else inference_dir / "_wmmsd_whistle_noise_suite"
    )

    manifest_path = prepared_dir / "manifest_selected.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    manifest_rows = read_csv_rows(manifest_path)
    clip_rows: list[dict[str, object]] = []
    for row in manifest_rows:
        staged_file_name = row["staged_file_name"]
        pred_summary = summarize_prediction_csv(prediction_csv_path(inference_dir, staged_file_name))
        clip_rows.append(
            {
                "species": str(row["species"]),
                "source_rel_path": str(row["source_rel_path"]),
                "staged_file_name": staged_file_name,
                "prediction_found": bool(pred_summary["prediction_found"]),
                "detection_count": int(pred_summary["detection_count"]),
                "mean_confidence": pred_summary["mean_confidence"],
                "max_confidence": pred_summary["max_confidence"],
            }
        )

    clip_thresholds = sorted({max(1, int(value)) for value in args.clip_positive_min_detections})

    species_summary_rows: list[dict[str, object]] = []
    species_names = sorted({str(row["species"]) for row in clip_rows})
    for species in species_names:
        species_clips = [row for row in clip_rows if str(row["species"]) == species]
        summary_row: dict[str, object] = {
            "species": species,
            "clip_count": len(species_clips),
            "mean_detections_per_clip": float(
                np.mean([int(row["detection_count"]) for row in species_clips])
            ),
            "max_detection_count": int(
                max(int(row["detection_count"]) for row in species_clips)
            ),
        }
        for threshold in clip_thresholds:
            hits = sum(1 for row in species_clips if int(row["detection_count"]) >= threshold)
            summary_row[f"hits_at_{threshold}"] = hits
            summary_row[f"hit_rate_at_{threshold}"] = (
                float(hits / len(species_clips)) if species_clips else 0.0
            )
        species_summary_rows.append(summary_row)

    scenario_summary_rows: list[dict[str, object]] = []
    scenario_clip_rows: list[dict[str, object]] = []
    scenario_confusion_rows: list[dict[str, object]] = []

    for scenario in WHISTLE_NOISE_PROXY_SCENARIOS:
        scenario_name = str(scenario["name"])
        difficulty = str(scenario["difficulty"])
        description = str(scenario["description"])
        positive_species = {str(value) for value in scenario["positive_species"]}
        negative_species = {str(value) for value in scenario["negative_species"]}
        optional_included = tuple(
            str(value) for value in scenario.get("optional_ambiguous_included", ())
        )
        optional_excluded = tuple(
            str(value) for value in scenario.get("optional_ambiguous_excluded", ())
        )
        scenario_dataset = [
            row
            for row in clip_rows
            if str(row["species"]) in positive_species or str(row["species"]) in negative_species
        ]
        present_positive_species = sorted(
            {str(row["species"]) for row in scenario_dataset if str(row["species"]) in positive_species}
        )
        present_negative_species = sorted(
            {str(row["species"]) for row in scenario_dataset if str(row["species"]) in negative_species}
        )
        scenario_status = "ok" if present_positive_species and present_negative_species else "incomplete"

        for threshold in clip_thresholds:
            labels = np.asarray(
                [1 if str(row["species"]) in positive_species else 0 for row in scenario_dataset],
                dtype=np.int64,
            )
            preds = np.asarray(
                [1 if int(row["detection_count"]) >= threshold else 0 for row in scenario_dataset],
                dtype=np.int64,
            )
            metrics = compute_binary_metrics(labels, preds)
            positives = int(np.sum(labels == 1))
            negatives = int(np.sum(labels == 0))

            scenario_summary_rows.append(
                {
                    "scenario": scenario_name,
                    "difficulty": difficulty,
                    "status": scenario_status,
                    "description": description,
                    "clip_positive_min_detections": threshold,
                    "total_clips": int(labels.size),
                    "positives": positives,
                    "negatives": negatives,
                    "optional_ambiguous_included": ",".join(optional_included),
                    "optional_ambiguous_excluded": ",".join(optional_excluded),
                    "present_positive_species": ",".join(present_positive_species),
                    "present_negative_species": ",".join(present_negative_species),
                    **metrics,
                }
            )
            scenario_confusion_rows.extend(
                [
                    {
                        "scenario": scenario_name,
                        "difficulty": difficulty,
                        "status": scenario_status,
                        "clip_positive_min_detections": threshold,
                        "actual": "negative",
                        "predicted_negative": metrics["tn"],
                        "predicted_positive": metrics["fp"],
                    },
                    {
                        "scenario": scenario_name,
                        "difficulty": difficulty,
                        "status": scenario_status,
                        "clip_positive_min_detections": threshold,
                        "actual": "positive",
                        "predicted_negative": metrics["fn"],
                        "predicted_positive": metrics["tp"],
                    },
                ]
            )

            for row, pred in zip(scenario_dataset, preds):
                scenario_clip_rows.append(
                    {
                        "scenario": scenario_name,
                        "difficulty": difficulty,
                        "status": scenario_status,
                        "clip_positive_min_detections": threshold,
                        "species": row["species"],
                        "source_rel_path": row["source_rel_path"],
                        "staged_file_name": row["staged_file_name"],
                        "true_label": 1 if str(row["species"]) in positive_species else 0,
                        "predicted_label": int(pred),
                        "detection_count": row["detection_count"],
                        "mean_confidence": row["mean_confidence"],
                        "max_confidence": row["max_confidence"],
                    }
                )

    write_csv(
        output_dir / "species_detection_profile.csv",
        species_summary_rows,
        list(species_summary_rows[0].keys()) if species_summary_rows else ["species"],
    )
    write_csv(
        output_dir / "scenario_summary.csv",
        scenario_summary_rows,
        [
            "scenario",
            "difficulty",
            "status",
            "description",
            "clip_positive_min_detections",
            "total_clips",
            "positives",
            "negatives",
            "optional_ambiguous_included",
            "optional_ambiguous_excluded",
            "present_positive_species",
            "present_negative_species",
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
        output_dir / "scenario_confusion_matrix.csv",
        scenario_confusion_rows,
        [
            "scenario",
            "difficulty",
            "status",
            "clip_positive_min_detections",
            "actual",
            "predicted_negative",
            "predicted_positive",
        ],
    )
    write_csv(
        output_dir / "scenario_per_clip.csv",
        scenario_clip_rows,
        [
            "scenario",
            "difficulty",
            "status",
            "clip_positive_min_detections",
            "species",
            "source_rel_path",
            "staged_file_name",
            "true_label",
            "predicted_label",
            "detection_count",
            "mean_confidence",
            "max_confidence",
        ],
    )

    complete_rows = [row for row in scenario_summary_rows if str(row["status"]) == "ok"]
    best_by_f1 = (
        max(complete_rows, key=lambda row: (float(row["f1"]), float(row["balanced_accuracy"])))
        if complete_rows
        else {}
    )

    summary_payload = {
        "prepared_dir": str(prepared_dir),
        "inference_dir": str(inference_dir),
        "clip_positive_min_detections": clip_thresholds,
        "scenario_count": len(WHISTLE_NOISE_PROXY_SCENARIOS),
        "evaluated_species": species_names,
        "optional_ambiguous_species": list(OPTIONAL_AMBIGUOUS_POSITIVES),
        "best_overall_by_f1": best_by_f1,
    }
    write_json(output_dir / "suite_summary.json", summary_payload)

    print(f"Scenarios evaluated : {len(WHISTLE_NOISE_PROXY_SCENARIOS)}")
    print(f"Clip vote thresholds: {clip_thresholds}")
    print(f"Optional ambiguous species: {list(OPTIONAL_AMBIGUOUS_POSITIVES)}")
    if best_by_f1:
        print(
            "Best overall        : "
            f"{best_by_f1['scenario']} @ min_det={best_by_f1['clip_positive_min_detections']} "
            f"F1={float(best_by_f1['f1']):.4f} "
            f"BA={float(best_by_f1['balanced_accuracy']):.4f}"
        )
    print(f"Scenario summary CSV : {output_dir / 'scenario_summary.csv'}")
    print(f"Species profile CSV  : {output_dir / 'species_detection_profile.csv'}")
    print(f"Suite summary JSON   : {output_dir / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
