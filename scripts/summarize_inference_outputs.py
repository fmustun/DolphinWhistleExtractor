#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Summarize whistle inference CSV outputs.'
    )
    parser.add_argument(
        '--inference-dir',
        required=True,
        help='Root directory containing per-recording prediction CSV files.',
    )
    parser.add_argument(
        '--recordings-dir',
        default='',
        help=(
            'Optional directory containing the source WAV/FLAC recordings. '
            'When provided, true durations and detections/hour are computed.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default='',
        help='Directory where summary CSV/JSON files will be written.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Number of most-detected files to export.',
    )
    return parser.parse_args()


def list_prediction_csvs(inference_dir: Path) -> list[Path]:
    return sorted(inference_dir.rglob('*.wav_predictions.csv'))


def safe_float(value: str) -> float:
    return float(value.strip())


def wav_duration_seconds(wav_path: Path) -> float | None:
    try:
        with wave.open(str(wav_path), 'rb') as handle:
            frame_rate = handle.getframerate()
            frame_count = handle.getnframes()
            if frame_rate <= 0:
                return None
            return frame_count / frame_rate
    except Exception:
        return None


def recording_path_for(file_name: str, recordings_dir: Path | None) -> Path | None:
    if recordings_dir is None:
        return None
    candidate = recordings_dir / file_name
    if candidate.exists():
        return candidate
    return None


def summarize_prediction_csv(csv_path: Path, recordings_dir: Path | None) -> dict[str, object]:
    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        file_name = csv_path.name.replace('.wav_predictions.csv', '.wav')
        confidences: list[float] = []
        starts: list[float] = []
        ends: list[float] = []
    else:
        file_name = str(rows[0]['file_name'])
        confidences = [safe_float(row['confidence']) for row in rows]
        starts = [safe_float(row['initial_point']) for row in rows]
        ends = [safe_float(row['finish_point']) for row in rows]

    recording_path = recording_path_for(file_name, recordings_dir)
    duration_seconds = wav_duration_seconds(recording_path) if recording_path else None
    observed_span_seconds = max(ends) if ends else 0.0
    effective_duration_seconds = duration_seconds if duration_seconds is not None else observed_span_seconds
    detections = len(rows)

    summary = {
        'file_name': file_name,
        'prediction_csv': str(csv_path),
        'recording_path': str(recording_path) if recording_path is not None else '',
        'detections': detections,
        'duration_seconds': effective_duration_seconds,
        'duration_from_audio_header': duration_seconds is not None,
        'first_detection_seconds': min(starts) if starts else '',
        'last_detection_seconds': max(ends) if ends else '',
        'mean_confidence': statistics.fmean(confidences) if confidences else '',
        'median_confidence': statistics.median(confidences) if confidences else '',
        'max_confidence': max(confidences) if confidences else '',
        'min_confidence': min(confidences) if confidences else '',
        'detections_per_hour': (
            detections / (effective_duration_seconds / 3600.0)
            if effective_duration_seconds and effective_duration_seconds > 0
            else ''
        ),
    }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    inference_dir = Path(args.inference_dir).expanduser().resolve()
    recordings_dir = Path(args.recordings_dir).expanduser().resolve() if args.recordings_dir else None
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else inference_dir / '_summary'
    )

    if not inference_dir.is_dir():
        raise FileNotFoundError(f'Inference directory not found: {inference_dir}')
    if recordings_dir is not None and not recordings_dir.is_dir():
        raise FileNotFoundError(f'Recordings directory not found: {recordings_dir}')

    prediction_csvs = list_prediction_csvs(inference_dir)
    summaries = [
        summarize_prediction_csv(csv_path, recordings_dir)
        for csv_path in prediction_csvs
    ]

    summaries_sorted = sorted(
        summaries,
        key=lambda row: int(row['detections']),
        reverse=True,
    )
    top_rows = summaries_sorted[: max(0, args.top_k)]

    duration_values = [
        float(row['duration_seconds'])
        for row in summaries
        if row['duration_seconds'] != ''
    ]
    confidence_values = [
        float(row['mean_confidence'])
        for row in summaries
        if row['mean_confidence'] != ''
    ]
    total_detections = sum(int(row['detections']) for row in summaries)
    total_duration_seconds = sum(duration_values)
    overall_detections_per_hour = (
        total_detections / (total_duration_seconds / 3600.0)
        if total_duration_seconds > 0
        else None
    )

    summary_payload = {
        'inference_dir': str(inference_dir),
        'recordings_dir': str(recordings_dir) if recordings_dir is not None else '',
        'prediction_csv_count': len(prediction_csvs),
        'total_detections': total_detections,
        'total_duration_seconds': total_duration_seconds,
        'total_duration_hours': total_duration_seconds / 3600.0 if total_duration_seconds else 0.0,
        'overall_detections_per_hour': overall_detections_per_hour,
        'files_with_audio_header_duration': sum(
            1 for row in summaries if bool(row['duration_from_audio_header'])
        ),
        'mean_file_level_confidence': statistics.fmean(confidence_values) if confidence_values else None,
        'median_file_level_confidence': statistics.median(confidence_values) if confidence_values else None,
        'top_k': args.top_k,
        'top_files': [
            {
                'file_name': row['file_name'],
                'detections': row['detections'],
                'detections_per_hour': row['detections_per_hour'],
            }
            for row in top_rows
        ],
    }

    field_names = [
        'file_name',
        'prediction_csv',
        'recording_path',
        'detections',
        'duration_seconds',
        'duration_from_audio_header',
        'first_detection_seconds',
        'last_detection_seconds',
        'mean_confidence',
        'median_confidence',
        'max_confidence',
        'min_confidence',
        'detections_per_hour',
    ]
    all_files_csv = output_dir / 'detections_per_file.csv'
    top_files_csv = output_dir / f'top_{args.top_k}_files.csv'
    summary_json = output_dir / 'summary.json'

    write_csv(all_files_csv, summaries_sorted, field_names)
    write_csv(top_files_csv, top_rows, field_names)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + '\n')

    print(f'Prediction CSV files: {len(prediction_csvs)}')
    print(f'Total detections    : {total_detections}')
    if total_duration_seconds > 0:
        print(f'Total duration (h)  : {total_duration_seconds / 3600.0:.3f}')
    if overall_detections_per_hour is not None:
        print(f'Detections/hour     : {overall_detections_per_hour:.3f}')
    print(f'Per-file CSV        : {all_files_csv}')
    print(f'Top files CSV       : {top_files_csv}')
    print(f'Summary JSON        : {summary_json}')


if __name__ == '__main__':
    main()
