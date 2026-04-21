#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn_dataset.create_within_session_noise_dataset import (  # noqa: E402
    build_forbidden_intervals,
    candidate_noise_windows,
    target_noise_count,
)
from external_benchmarks.dclde.common import (  # noqa: E402
    ensure_dir,
    safe_slug,
    staged_file_name,
    write_csv,
    write_json,
)
from external_benchmarks.dclde.silbidopy_reader import tonalReader  # noqa: E402


DEFAULT_WINDOW_SECONDS = 0.4
DEFAULT_MARGIN_SECONDS = 0.4
DEFAULT_NOISE_TO_WHISTLE_RATIO = 1.0
DEFAULT_SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a clip-level whistle/noise benchmark from the public DCLDE "
            "2011 corpus by centering windows on annotated whistle contours and "
            "mining negatives away from them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subset",
        choices=("development", "evaluation", "both"),
        default="evaluation",
        help="Which DCLDE split(s) to use.",
    )
    parser.add_argument(
        "--annotation-version",
        choices=("2011", "2025_auto", "2025_only"),
        default="2011",
        help="Which annotation tree to read when both are available.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=[],
        help="Optional exact species-folder filter.",
    )
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--margin-seconds", type=float, default=DEFAULT_MARGIN_SECONDS)
    parser.add_argument("--noise-to-whistle-ratio", type=float, default=DEFAULT_NOISE_TO_WHISTLE_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--limit-positives",
        type=int,
        default=0,
        help="Optional cap on the number of positive clips after filtering.",
    )
    parser.add_argument(
        "--skip-zero-annotation-files",
        action="store_true",
        help="Skip files whose selected annotation file contains zero contours.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing generated files in --output-dir before preparing the benchmark.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be strictly positive.")
    if args.margin_seconds < 0:
        raise ValueError("--margin-seconds must be non-negative.")
    if args.noise_to_whistle_ratio < 0:
        raise ValueError("--noise-to-whistle-ratio must be non-negative.")
    if args.limit_positives < 0:
        raise ValueError("--limit-positives must be non-negative.")


def locate_dataset_root(source_root: Path) -> Path:
    candidates = [
        source_root,
        source_root / "dclde_2011",
        source_root / "dclde" / "2011" / "dclde_2011",
    ]
    for candidate in candidates:
        if (candidate / "README.md").exists() and (
            (candidate / "development").is_dir() or (candidate / "evaluation").is_dir()
        ):
            return candidate

    matches = sorted(
        path
        for path in source_root.rglob("dclde_2011")
        if path.is_dir() and (path / "README.md").exists()
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Could not find the DCLDE dataset root under "
        f"{source_root}. Expected a folder containing README.md and "
        "`development/` or `evaluation/`."
    )


def centered_window(onset: float, offset: float, duration_seconds: float, window_seconds: float) -> tuple[float, float]:
    center = (float(onset) + float(offset)) / 2.0
    max_start = max(float(duration_seconds) - float(window_seconds), 0.0)
    start = min(max(center - float(window_seconds) / 2.0, 0.0), max_start)
    stop = start + float(window_seconds)
    return round(start, 6), round(stop, 6)


def extract_window_mono(audio_file: sf.SoundFile, onset: float, window_seconds: float) -> list[float]:
    frames = int(round(float(window_seconds) * float(audio_file.samplerate)))
    start_frame = int(round(float(onset) * float(audio_file.samplerate)))
    audio_file.seek(start_frame)
    clip = audio_file.read(frames=frames, dtype="float32", always_2d=False)
    if getattr(clip, "ndim", 1) > 1:
        clip = clip[:, 0]
    if len(clip) < frames:
        clip = list(clip) + [0.0] * (frames - len(clip))
    else:
        clip = list(clip[:frames])
    return clip


def selected_splits(subset: str) -> list[str]:
    if subset == "both":
        return ["development", "evaluation"]
    return [subset]


def selected_annotation_path(dataset_root: Path, wav_path: Path, annotation_version: str) -> tuple[Path | None, str]:
    relative = wav_path.relative_to(dataset_root)
    ann2011 = wav_path.with_suffix(".ann")
    ann2025 = (dataset_root / "annotations2025" / relative).with_suffix(".ann")

    if annotation_version == "2011":
        return (ann2011 if ann2011.exists() else None), "2011"
    if annotation_version == "2025_only":
        return (ann2025 if ann2025.exists() else None), "2025"
    if ann2025.exists():
        return ann2025, "2025"
    if ann2011.exists():
        return ann2011, "2011"
    return None, ""


def load_contours(annotation_path: Path) -> list[list[tuple[float, float]]]:
    if not annotation_path.exists():
        return []
    reader = tonalReader(str(annotation_path))
    return reader.getTimeFrequencyContours()


def output_dir_has_files(output_dir: Path) -> bool:
    return any(path.is_file() for path in output_dir.rglob("*"))


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in sorted(output_dir.iterdir()):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def contour_rows_from_file(
    wav_path: Path,
    annotation_path: Path,
    annotation_version_used: str,
    dataset_root: Path,
    window_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, float]], float, int]:
    audio_info = sf.info(str(wav_path))
    duration_seconds = float(audio_info.frames) / float(audio_info.samplerate)
    relative = wav_path.relative_to(dataset_root)
    parts = relative.parts
    dataset_split = parts[0] if len(parts) > 0 else ""
    species = parts[1] if len(parts) > 1 else ""
    recording = wav_path.name

    contours = load_contours(annotation_path)
    positive_rows: list[dict[str, object]] = []
    contour_intervals: list[dict[str, float]] = []

    for contour_index, contour in enumerate(contours, start=1):
        if not contour:
            continue
        times = [float(time) for time, _ in contour]
        freqs = [float(freq) for _, freq in contour]
        if not times:
            continue
        annotation_onset = min(times)
        annotation_offset = max(times)
        onset, offset = centered_window(
            onset=annotation_onset,
            offset=annotation_offset,
            duration_seconds=duration_seconds,
            window_seconds=window_seconds,
        )
        contour_intervals.append({"onset": annotation_onset, "offset": annotation_offset})
        positive_rows.append(
            {
                "label": 1,
                "clip_kind": f"{safe_slug(dataset_split, 16)}_{safe_slug(species, 24)}",
                "dataset_split": dataset_split,
                "species": species,
                "recording": recording,
                "source_audio_relpath": str(relative),
                "source_audio_path": str(wav_path),
                "source_annotation_path": str(annotation_path),
                "recording_duration_seconds": round(duration_seconds, 6),
                "sample_rate": int(audio_info.samplerate),
                "bit_depth": str(audio_info.subtype),
                "onset": onset,
                "offset": offset,
                "annotation_onset": round(annotation_onset, 6),
                "annotation_offset": round(annotation_offset, 6),
                "annotation_duration": round(annotation_offset - annotation_onset, 6),
                "annotation_center": round((annotation_onset + annotation_offset) / 2.0, 6),
                "annotation_node_count": len(contour),
                "annotation_min_freq": round(min(freqs), 3),
                "annotation_max_freq": round(max(freqs), 3),
                "annotation_version_used": annotation_version_used,
                "source_kind": "manual_positive_centered",
                "annotation_index": contour_index,
            }
        )

    return positive_rows, contour_intervals, duration_seconds, int(audio_info.samplerate)


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    source_root = Path(args.source_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if args.overwrite_output:
            clear_output_dir(output_dir)
        if output_dir_has_files(output_dir):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    flat_recordings_dir = ensure_dir(output_dir / "flat_recordings")

    dataset_root = locate_dataset_root(source_root)
    species_filter = set(args.species)

    wav_paths: list[Path] = []
    for split in selected_splits(args.subset):
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for wav_path in sorted(split_dir.rglob("*.wav")):
            relative = wav_path.relative_to(dataset_root)
            if len(relative.parts) < 2:
                continue
            species = relative.parts[1]
            if species_filter and species not in species_filter:
                continue
            wav_paths.append(wav_path)

    if not wav_paths:
        raise ValueError("No WAV files matched the requested DCLDE filters.")

    all_positive_rows: list[dict[str, object]] = []
    all_negative_candidates: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []

    for wav_path in wav_paths:
        annotation_path, annotation_version_used = selected_annotation_path(
            dataset_root=dataset_root,
            wav_path=wav_path,
            annotation_version=args.annotation_version,
        )
        if annotation_path is None:
            continue

        positive_rows, contour_intervals, duration_seconds, sample_rate = contour_rows_from_file(
            wav_path=wav_path,
            annotation_path=annotation_path,
            annotation_version_used=annotation_version_used,
            dataset_root=dataset_root,
            window_seconds=args.window_seconds,
        )
        if args.skip_zero_annotation_files and not positive_rows:
            continue

        relative = wav_path.relative_to(dataset_root)
        dataset_split = relative.parts[0] if len(relative.parts) > 0 else ""
        species = relative.parts[1] if len(relative.parts) > 1 else ""

        file_rows.append(
            {
                "dataset_split": dataset_split,
                "species": species,
                "recording": wav_path.name,
                "source_audio_relpath": str(relative),
                "source_audio_path": str(wav_path),
                "source_annotation_path": str(annotation_path),
                "annotation_version_used": annotation_version_used,
                "positive_contours": len(positive_rows),
                "duration_seconds": round(duration_seconds, 6),
                "sample_rate": sample_rate,
            }
        )
        all_positive_rows.extend(positive_rows)

        forbidden_rows = pd.DataFrame(contour_intervals, columns=["onset", "offset"])
        forbidden_intervals = build_forbidden_intervals(
            forbidden_rows,
            duration_seconds=duration_seconds,
            margin_seconds=args.margin_seconds,
        )
        candidate_windows = candidate_noise_windows(
            duration_seconds=duration_seconds,
            window_seconds=args.window_seconds,
            forbidden_intervals=forbidden_intervals,
        )
        for onset, offset in candidate_windows:
            all_negative_candidates.append(
                {
                    "label": 0,
                    "clip_kind": f"{safe_slug(dataset_split, 16)}_{safe_slug(species, 24)}",
                    "dataset_split": dataset_split,
                    "species": species,
                    "recording": wav_path.name,
                    "source_audio_relpath": str(relative),
                    "source_audio_path": str(wav_path),
                    "source_annotation_path": str(annotation_path),
                    "recording_duration_seconds": round(duration_seconds, 6),
                    "sample_rate": sample_rate,
                    "bit_depth": str(sf.info(str(wav_path)).subtype),
                    "onset": round(float(onset), 6),
                    "offset": round(float(offset), 6),
                    "annotation_onset": "",
                    "annotation_offset": "",
                    "annotation_duration": "",
                    "annotation_center": "",
                    "annotation_node_count": "",
                    "annotation_min_freq": "",
                    "annotation_max_freq": "",
                    "annotation_version_used": annotation_version_used,
                    "source_kind": "mined_noise",
                    "annotation_index": "",
                }
            )

    if not all_positive_rows:
        raise ValueError("No positive contours remain after filtering.")

    if args.limit_positives > 0 and len(all_positive_rows) > args.limit_positives:
        all_positive_rows = rng.sample(all_positive_rows, k=args.limit_positives)

    noise_target = target_noise_count(len(all_positive_rows), args.noise_to_whistle_ratio)
    sampled_negative_rows = (
        rng.sample(all_negative_candidates, k=min(noise_target, len(all_negative_candidates)))
        if noise_target > 0 and all_negative_candidates
        else []
    )

    rows_out = all_positive_rows + sampled_negative_rows
    rows_out = sorted(
        rows_out,
        key=lambda row: (
            str(row["dataset_split"]),
            str(row["species"]),
            str(row["recording"]),
            float(row["onset"]),
            -int(row["label"]),
        ),
    )

    audio_handles: dict[str, sf.SoundFile] = {}
    try:
        for index, row in enumerate(rows_out, start=1):
            source_audio_path = str(row["source_audio_path"])
            if source_audio_path not in audio_handles:
                audio_handles[source_audio_path] = sf.SoundFile(source_audio_path, "r")
            audio_file = audio_handles[source_audio_path]
            staged_name = staged_file_name(
                index=index,
                label=int(row["label"]),
                clip_kind=str(row["clip_kind"]),
                onset=float(row["onset"]),
                offset=float(row["offset"]),
            )
            destination_path = flat_recordings_dir / staged_name
            clip = extract_window_mono(
                audio_file=audio_file,
                onset=float(row["onset"]),
                window_seconds=args.window_seconds,
            )
            sf.write(str(destination_path), clip, audio_file.samplerate, subtype="PCM_16")
            row["staged_file_name"] = staged_name
            row["staged_file_path"] = str(destination_path)
    finally:
        for handle in audio_handles.values():
            handle.close()

    manifest_field_names = [
        "label",
        "clip_kind",
        "dataset_split",
        "species",
        "recording",
        "source_audio_relpath",
        "source_audio_path",
        "source_annotation_path",
        "staged_file_name",
        "staged_file_path",
        "recording_duration_seconds",
        "sample_rate",
        "bit_depth",
        "onset",
        "offset",
        "annotation_onset",
        "annotation_offset",
        "annotation_duration",
        "annotation_center",
        "annotation_node_count",
        "annotation_min_freq",
        "annotation_max_freq",
        "annotation_version_used",
        "source_kind",
        "annotation_index",
    ]
    write_csv(output_dir / "manifest_selected.csv", rows_out, manifest_field_names)
    write_csv(
        output_dir / "source_files.csv",
        file_rows,
        [
            "dataset_split",
            "species",
            "recording",
            "source_audio_relpath",
            "source_audio_path",
            "source_annotation_path",
            "annotation_version_used",
            "positive_contours",
            "duration_seconds",
            "sample_rate",
        ],
    )

    summary_payload = {
        "dataset_root": str(dataset_root),
        "input_descriptor": f"source_root:{source_root}",
        "subset": args.subset,
        "annotation_version": args.annotation_version,
        "species_filter": list(args.species),
        "window_seconds": args.window_seconds,
        "margin_seconds": args.margin_seconds,
        "noise_to_whistle_ratio": args.noise_to_whistle_ratio,
        "seed": args.seed,
        "wav_files_scanned": len(wav_paths),
        "source_files_with_annotations": len(file_rows),
        "positive_clips": len(all_positive_rows),
        "negative_candidates": len(all_negative_candidates),
        "negative_clips": len(sampled_negative_rows),
        "prepared_clips": len(rows_out),
        "flat_recordings_dir": str(flat_recordings_dir),
        "manifest_selected": str(output_dir / "manifest_selected.csv"),
        "source_files_csv": str(output_dir / "source_files.csv"),
        "negative_target_requested": noise_target,
    }
    write_json(output_dir / "prepare_summary.json", summary_payload)

    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# DCLDE Clip Benchmark",
                "",
                "Prepared with `prepare_dclde_clip_benchmark.py`.",
                "",
                "Generated files:",
                "- `flat_recordings/`",
                "- `manifest_selected.csv`",
                "- `source_files.csv`",
                "- `prepare_summary.json`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Dataset root       : {dataset_root}")
    print(f"WAV files scanned  : {len(wav_paths)}")
    print(f"Annotated sources  : {len(file_rows)}")
    print(f"Positive clips     : {len(all_positive_rows)}")
    print(f"Negative candidates: {len(all_negative_candidates)}")
    print(f"Negative clips     : {len(sampled_negative_rows)}")
    print(f"Prepared clips     : {len(rows_out)}")
    print(f"Flat recordings dir: {flat_recordings_dir}")
    print(f"Manifest selected  : {output_dir / 'manifest_selected.csv'}")


if __name__ == "__main__":
    main()
