#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import wave
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.wmmsd.common import read_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate WMMSD inference as a positive-only benchmark: one clip counts "
            "as hit if the CNN emits at least one positive detection."
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
        "--output-dir",
        default="",
        help="Directory where evaluation CSV/JSON files will be written.",
    )
    return parser.parse_args()


def prediction_csv_path(inference_dir: Path, staged_file_name: str) -> Path:
    stem = Path(staged_file_name).stem
    return inference_dir / stem / f"{stem}.wav_predictions.csv"


def wav_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if frame_rate <= 0:
                return None
            return frame_count / frame_rate
    except Exception:
        return None


def summarize_prediction_csv(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "prediction_found": False,
            "detection_count": 0,
            "has_detection": False,
            "mean_confidence": "",
            "max_confidence": "",
            "first_detection_seconds": "",
            "last_detection_seconds": "",
        }

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {
            "prediction_found": True,
            "detection_count": 0,
            "has_detection": False,
            "mean_confidence": "",
            "max_confidence": "",
            "first_detection_seconds": "",
            "last_detection_seconds": "",
        }

    confidences = [float(row["confidence"]) for row in rows]
    starts = [float(row["initial_point"]) for row in rows]
    ends = [float(row["finish_point"]) for row in rows]
    return {
        "prediction_found": True,
        "detection_count": len(rows),
        "has_detection": True,
        "mean_confidence": statistics.fmean(confidences),
        "max_confidence": max(confidences),
        "first_detection_seconds": min(starts),
        "last_detection_seconds": max(ends),
    }


def main() -> None:
    args = parse_args()

    prepared_dir = Path(args.prepared_dir).expanduser().resolve()
    inference_dir = Path(args.inference_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else inference_dir / "_wmmsd_positive_only"
    )

    manifest_path = prepared_dir / "manifest_selected.csv"
    flat_recordings_dir = prepared_dir / "flat_recordings"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    manifest_rows = read_csv_rows(manifest_path)
    per_clip_rows: list[dict[str, object]] = []
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in manifest_rows:
        staged_file_name = row["staged_file_name"]
        staged_audio_path = flat_recordings_dir / staged_file_name
        prediction_path = prediction_csv_path(inference_dir, staged_file_name)
        prediction_summary = summarize_prediction_csv(prediction_path)
        clip_row = {
            "species": row["species"],
            "source_rel_path": row["source_rel_path"],
            "source_path": row["source_path"],
            "staged_file_name": staged_file_name,
            "staged_audio_path": str(staged_audio_path),
            "prediction_csv": str(prediction_path),
            "prediction_found": prediction_summary["prediction_found"],
            "detection_count": prediction_summary["detection_count"],
            "has_detection": prediction_summary["has_detection"],
            "mean_confidence": prediction_summary["mean_confidence"],
            "max_confidence": prediction_summary["max_confidence"],
            "first_detection_seconds": prediction_summary["first_detection_seconds"],
            "last_detection_seconds": prediction_summary["last_detection_seconds"],
            "clip_duration_seconds": wav_duration_seconds(staged_audio_path) or "",
        }
        per_clip_rows.append(clip_row)
        grouped_rows[str(row["species"])].append(clip_row)

    per_clip_rows.sort(key=lambda row: (str(row["species"]), str(row["staged_file_name"])))

    per_species_rows: list[dict[str, object]] = []
    total_clips = 0
    total_hits = 0
    total_detections = 0

    for species in sorted(grouped_rows):
        rows = grouped_rows[species]
        clip_count = len(rows)
        hit_count = sum(1 for row in rows if bool(row["has_detection"]))
        detections = sum(int(row["detection_count"]) for row in rows)
        hit_confidences = [
            float(row["max_confidence"])
            for row in rows
            if row["max_confidence"] != ""
        ]
        per_species_rows.append(
            {
                "species": species,
                "clip_count": clip_count,
                "clips_with_detection": hit_count,
                "clip_hit_rate": (hit_count / clip_count) if clip_count else "",
                "total_detections": detections,
                "mean_detections_per_clip": (detections / clip_count) if clip_count else "",
                "mean_max_confidence_on_hits": (
                    statistics.fmean(hit_confidences) if hit_confidences else ""
                ),
            }
        )
        total_clips += clip_count
        total_hits += hit_count
        total_detections += detections

    summary_payload = {
        "prepared_dir": str(prepared_dir),
        "inference_dir": str(inference_dir),
        "total_clips": total_clips,
        "clips_with_detection": total_hits,
        "overall_clip_hit_rate": (total_hits / total_clips) if total_clips else None,
        "total_detections": total_detections,
        "note": (
            "WMMSD positive-only summary: clip_hit_rate measures whether at least one "
            "whistle detection was emitted on a selected clip. This is not a strict "
            "whistle-recall metric because WMMSD clip labels are species-level here."
        ),
    }

    field_names_clip = [
        "species",
        "source_rel_path",
        "source_path",
        "staged_file_name",
        "staged_audio_path",
        "prediction_csv",
        "prediction_found",
        "detection_count",
        "has_detection",
        "mean_confidence",
        "max_confidence",
        "first_detection_seconds",
        "last_detection_seconds",
        "clip_duration_seconds",
    ]
    field_names_species = [
        "species",
        "clip_count",
        "clips_with_detection",
        "clip_hit_rate",
        "total_detections",
        "mean_detections_per_clip",
        "mean_max_confidence_on_hits",
    ]

    write_csv(output_dir / "per_clip.csv", per_clip_rows, field_names_clip)
    write_csv(output_dir / "per_species.csv", per_species_rows, field_names_species)
    write_json(output_dir / "summary.json", summary_payload)

    print(f"Total clips         : {total_clips}")
    print(f"Clips with detection: {total_hits}")
    if total_clips:
        print(f"Overall hit rate    : {total_hits / total_clips:.4f}")
    print(f"Total detections    : {total_detections}")
    print(f"Per-clip CSV        : {output_dir / 'per_clip.csv'}")
    print(f"Per-species CSV     : {output_dir / 'per_species.csv'}")
    print(f"Summary JSON        : {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
