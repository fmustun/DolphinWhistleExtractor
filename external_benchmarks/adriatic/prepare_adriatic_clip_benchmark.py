#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cnn_dataset.create_within_session_noise_dataset import (  # noqa: E402
    build_forbidden_intervals,
    candidate_noise_windows,
    merge_intervals,
    target_noise_count,
)
from external_benchmarks.adriatic.common import (  # noqa: E402
    ensure_dir,
    staged_file_name,
    write_csv,
    write_json,
)


DEFAULT_WINDOW_SECONDS = 0.4
DEFAULT_MARGIN_SECONDS = 0.4
DEFAULT_NOISE_TO_WHISTLE_RATIO = 1.0
DEFAULT_SEED = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a clip-level whistle/noise benchmark from the Adriatic trawling "
            "dataset using full_recording.wav, whistles.txt, and optionally clicks.txt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        default="",
        help="Dataset root directory. The script will search recursively for the expected files.",
    )
    parser.add_argument(
        "--archive-path",
        default="",
        help="Optional path to a .zip/.tar/.tar.gz archive containing the Adriatic files.",
    )
    parser.add_argument(
        "--full-recording-path",
        default="",
        help="Optional explicit path to full_recording.wav.",
    )
    parser.add_argument(
        "--whistles-path",
        default="",
        help="Optional explicit path to whistles.txt.",
    )
    parser.add_argument(
        "--clicks-path",
        default="",
        help="Optional explicit path to clicks.txt.",
    )
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--margin-seconds", type=float, default=DEFAULT_MARGIN_SECONDS)
    parser.add_argument("--noise-to-whistle-ratio", type=float, default=DEFAULT_NOISE_TO_WHISTLE_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-whistle-quality",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="Keep only whistles with quality >= this grade.",
    )
    parser.add_argument(
        "--exclude-multiple-whistles",
        action="store_true",
        help="Discard MW annotations and keep only single-whistle rows.",
    )
    parser.add_argument(
        "--ignore-clicks",
        action="store_true",
        help="Do not use clicks.txt intervals as forbidden regions for negative mining.",
    )
    parser.add_argument(
        "--limit-positives",
        type=int,
        default=0,
        help="Optional cap on the number of positive clips kept after filtering.",
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
    has_source_root = bool(str(args.source_root).strip())
    has_archive_path = bool(str(args.archive_path).strip())
    has_explicit_paths = bool(str(args.full_recording_path).strip()) or bool(str(args.whistles_path).strip())
    selected_modes = int(has_source_root) + int(has_archive_path) + int(has_explicit_paths)
    if selected_modes == 0:
        raise ValueError(
            "Provide one of: --source-root, --archive-path, or explicit "
            "--full-recording-path and --whistles-path."
        )
    if selected_modes > 1:
        raise ValueError(
            "Choose only one input mode: --source-root, --archive-path, or explicit file paths."
        )
    if has_explicit_paths and not (
        bool(str(args.full_recording_path).strip()) and bool(str(args.whistles_path).strip())
    ):
        raise ValueError(
            "When using explicit paths, both --full-recording-path and --whistles-path are required."
        )


def find_single_file(source_root: Path, pattern: str) -> Path:
    matches = sorted(source_root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under: {source_root}")
    if len(matches) > 1:
        preferred = [path for path in matches if path.name == pattern]
        if len(preferred) == 1:
            return preferred[0]
        raise FileExistsError(
            f"Expected a single {pattern} under {source_root}, found {len(matches)}. "
            f"Examples: {[str(path) for path in matches[:5]]}"
        )
    return matches[0]


def extract_single_file_from_archive(archive_path: Path, pattern: str, extract_root: Path) -> Path:
    archive_name = archive_path.name.lower()
    extracted_path = extract_root / pattern
    extracted_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as handle:
            members = [name for name in handle.namelist() if Path(name).name == pattern]
            if not members:
                raise FileNotFoundError(f"Could not find {pattern} inside archive: {archive_path}")
            if len(members) > 1:
                raise FileExistsError(
                    f"Expected a single {pattern} inside {archive_path}, found {len(members)}. "
                    f"Examples: {members[:5]}"
                )
            with handle.open(members[0], "r") as src, extracted_path.open("wb") as dst:
                dst.write(src.read())
            return extracted_path

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as handle:
            members = [member for member in handle.getmembers() if Path(member.name).name == pattern]
            if not members:
                raise FileNotFoundError(f"Could not find {pattern} inside archive: {archive_path}")
            if len(members) > 1:
                raise FileExistsError(
                    f"Expected a single {pattern} inside {archive_path}, found {len(members)}. "
                    f"Examples: {[member.name for member in members[:5]]}"
                )
            extracted_file = handle.extractfile(members[0])
            if extracted_file is None:
                raise FileNotFoundError(f"Could not extract {pattern} from archive: {archive_path}")
            with extracted_file, extracted_path.open("wb") as dst:
                dst.write(extracted_file.read())
            return extracted_path

    raise ValueError(
        f"Unsupported archive format for {archive_path}. Expected .zip, .tar, .tar.gz, or .tgz."
    )


def split_annotation_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    if "\t" in stripped:
        return [token.strip() for token in stripped.split("\t") if token.strip()]
    return stripped.split()


def parse_float_token(value: str) -> float:
    return float(str(value).strip().replace(",", "."))


def parse_whistles(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = split_annotation_line(raw_line)
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Invalid whistles.txt line {line_number}: {raw_line!r}")

        start = parse_float_token(parts[0])
        stop = parse_float_token(parts[1])
        if stop <= start:
            continue

        remainder = [token.strip() for token in parts[2:] if token.strip()]
        whistle_tag = ""
        whistle_quality: int | None = None

        if remainder:
            remainder_text = " ".join(remainder).upper()
            compact = "".join(remainder).upper()
            match = re.match(r"^(MW|W)([123])$", compact)
            if match:
                whistle_tag = match.group(1)
                whistle_quality = int(match.group(2))
            else:
                first = remainder[0].upper()
                if first in {"W", "MW"}:
                    whistle_tag = first
                    if len(remainder) > 1 and re.fullmatch(r"[123]", remainder[1]):
                        whistle_quality = int(remainder[1])
                else:
                    tag_match = re.search(r"(MW|W)", compact)
                    quality_match = re.search(r"([123])", compact)
                    if tag_match:
                        whistle_tag = tag_match.group(1)
                    if quality_match:
                        whistle_quality = int(quality_match.group(1))
            if not whistle_tag:
                tag_match = re.search(r"\b(MW|W)\b", remainder_text)
                if tag_match:
                    whistle_tag = tag_match.group(1)
            if whistle_quality is None:
                class_match = re.search(r"CLASS\s*:\s*([123])", remainder_text)
                if class_match:
                    whistle_quality = int(class_match.group(1))

        rows.append(
            {
                "annotation_onset": round(start, 6),
                "annotation_offset": round(stop, 6),
                "annotation_duration": round(stop - start, 6),
                "whistle_tag": whistle_tag or "W",
                "whistle_quality": whistle_quality,
                "annotation_line_number": line_number,
            }
        )

    if not rows:
        raise ValueError(f"No whistle rows parsed from: {path}")
    return pd.DataFrame(rows).sort_values(["annotation_onset", "annotation_offset"]).reset_index(drop=True)


def parse_clicks(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = split_annotation_line(raw_line)
        if not parts:
            continue
        if len(parts) < 2:
            raise ValueError(f"Invalid clicks.txt line {line_number}: {raw_line!r}")
        start = parse_float_token(parts[0])
        stop = parse_float_token(parts[1])
        if stop <= start:
            continue
        rows.append(
            {
                "click_onset": round(start, 6),
                "click_offset": round(stop, 6),
                "annotation_line_number": line_number,
            }
        )
    return pd.DataFrame(rows).sort_values(["click_onset", "click_offset"]).reset_index(drop=True)


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
        padding = [0.0] * (frames - len(clip))
        clip = list(clip) + padding
    else:
        clip = list(clip[:frames])
    return clip


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    archive_path = Path(args.archive_path).expanduser().resolve() if args.archive_path else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        existing_entries = [path for path in output_dir.iterdir()]
        if existing_entries:
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    flat_recordings_dir = ensure_dir(output_dir / "flat_recordings")

    if source_root is not None and any(
        token in str(source_root)
        for token in ("/chemin/vers/", "\\chemin\\vers\\", "path/to", "your/path")
    ):
        raise ValueError(
            "The value passed to --source-root still looks like a placeholder path. "
            "Replace it with the real Adriatic dataset folder, or pass "
            "--archive-path or explicit --full-recording-path and --whistles-path."
        )

    if args.full_recording_path:
        full_recording_path = Path(args.full_recording_path).expanduser().resolve()
        whistles_path = Path(args.whistles_path).expanduser().resolve()
        clicks_path = None if args.ignore_clicks or not args.clicks_path else Path(args.clicks_path).expanduser().resolve()
        input_descriptor = "explicit_paths"
    elif archive_path is not None:
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        with tempfile.TemporaryDirectory(prefix="adriatic_prepare_") as temp_dir_name:
            extract_root = Path(temp_dir_name)
            full_recording_path = extract_single_file_from_archive(archive_path, "full_recording.wav", extract_root)
            whistles_path = extract_single_file_from_archive(archive_path, "whistles.txt", extract_root)
            if args.ignore_clicks:
                clicks_path = None
            else:
                try:
                    clicks_path = extract_single_file_from_archive(archive_path, "clicks.txt", extract_root)
                except FileNotFoundError:
                    clicks_path = None
            input_descriptor = f"archive:{archive_path}"
            prepare_from_paths(
                args=args,
                full_recording_path=full_recording_path,
                whistles_path=whistles_path,
                clicks_path=clicks_path,
                source_root=source_root,
                archive_path=archive_path,
                input_descriptor=input_descriptor,
                output_dir=output_dir,
                flat_recordings_dir=flat_recordings_dir,
                rng=rng,
            )
            return
    else:
        assert source_root is not None
        full_recording_path = find_single_file(source_root, "full_recording.wav")
        whistles_path = find_single_file(source_root, "whistles.txt")
        if args.ignore_clicks:
            clicks_path = None
        else:
            try:
                clicks_path = find_single_file(source_root, "clicks.txt")
            except FileNotFoundError:
                clicks_path = None
        input_descriptor = f"source_root:{source_root}"

    prepare_from_paths(
        args=args,
        full_recording_path=full_recording_path,
        whistles_path=whistles_path,
        clicks_path=clicks_path,
        source_root=source_root,
        archive_path=archive_path,
        input_descriptor=input_descriptor,
        output_dir=output_dir,
        flat_recordings_dir=flat_recordings_dir,
        rng=rng,
    )


def prepare_from_paths(
    args: argparse.Namespace,
    full_recording_path: Path,
    whistles_path: Path,
    clicks_path: Path | None,
    source_root: Path | None,
    archive_path: Path | None,
    input_descriptor: str,
    output_dir: Path,
    flat_recordings_dir: Path,
    rng: random.Random,
) -> None:
    if not full_recording_path.exists():
        raise FileNotFoundError(f"full_recording.wav not found: {full_recording_path}")
    if not whistles_path.exists():
        raise FileNotFoundError(f"whistles.txt not found: {whistles_path}")
    if clicks_path is not None and not clicks_path.exists():
        raise FileNotFoundError(f"clicks.txt not found: {clicks_path}")

    whistles = parse_whistles(whistles_path)
    raw_whistle_count = int(len(whistles))
    if args.exclude_multiple_whistles:
        whistles = whistles.loc[whistles["whistle_tag"].astype(str) != "MW"].copy()
    whistles = whistles.loc[
        whistles["whistle_quality"].fillna(0).astype(int) >= int(args.min_whistle_quality)
    ].copy()
    if args.limit_positives > 0:
        whistles = whistles.iloc[: args.limit_positives].copy()
    if whistles.empty:
        raise ValueError("No whistles remain after filtering.")

    clicks = parse_clicks(clicks_path) if clicks_path is not None else pd.DataFrame(columns=["click_onset", "click_offset"])

    audio_info = sf.info(str(full_recording_path))
    duration_seconds = float(audio_info.frames) / float(audio_info.samplerate)

    positive_rows: list[dict[str, object]] = []
    for row in whistles.itertuples(index=False):
        onset, offset = centered_window(
            onset=float(row.annotation_onset),
            offset=float(row.annotation_offset),
            duration_seconds=duration_seconds,
            window_seconds=args.window_seconds,
        )
        positive_rows.append(
            {
                "label": 1,
                "clip_kind": "multiple_whistle" if str(row.whistle_tag) == "MW" else "single_whistle",
                "recording": full_recording_path.name,
                "session_id": "adriatic_trawl_2021",
                "source_audio_path": str(full_recording_path),
                "recording_duration_seconds": round(duration_seconds, 6),
                "sample_rate": int(audio_info.samplerate),
                "onset": onset,
                "offset": offset,
                "annotation_onset": float(row.annotation_onset),
                "annotation_offset": float(row.annotation_offset),
                "annotation_duration": float(row.annotation_duration),
                "whistle_tag": str(row.whistle_tag),
                "whistle_quality": int(row.whistle_quality) if pd.notna(row.whistle_quality) else "",
                "clicks_forbidden": bool(clicks_path is not None),
                "source_kind": "manual_positive_centered",
                "annotation_line_number": int(row.annotation_line_number),
            }
        )

    forbidden_rows = pd.DataFrame(
        {
            "onset": whistles["annotation_onset"].astype(float).tolist(),
            "offset": whistles["annotation_offset"].astype(float).tolist(),
        }
    )
    forbidden_intervals = build_forbidden_intervals(
        forbidden_rows,
        duration_seconds=duration_seconds,
        margin_seconds=args.margin_seconds,
    )
    if clicks_path is not None and not clicks.empty:
        click_forbidden_rows = pd.DataFrame(
            {
                "onset": clicks["click_onset"].astype(float).tolist(),
                "offset": clicks["click_offset"].astype(float).tolist(),
            }
        )
        forbidden_intervals = sorted(
            forbidden_intervals
            + build_forbidden_intervals(
                click_forbidden_rows,
                duration_seconds=duration_seconds,
                margin_seconds=args.margin_seconds,
            )
        )
    forbidden_intervals = merge_intervals(forbidden_intervals)

    candidate_windows = candidate_noise_windows(
        duration_seconds=duration_seconds,
        window_seconds=args.window_seconds,
        forbidden_intervals=forbidden_intervals,
    )
    target_noise_rows = target_noise_count(len(positive_rows), args.noise_to_whistle_ratio)
    sampled_noise_windows = (
        rng.sample(candidate_windows, k=min(target_noise_rows, len(candidate_windows)))
        if candidate_windows and target_noise_rows > 0
        else []
    )

    noise_rows: list[dict[str, object]] = []
    for onset, offset in sampled_noise_windows:
        noise_rows.append(
            {
                "label": 0,
                "clip_kind": "background_noise",
                "recording": full_recording_path.name,
                "session_id": "adriatic_trawl_2021",
                "source_audio_path": str(full_recording_path),
                "recording_duration_seconds": round(duration_seconds, 6),
                "sample_rate": int(audio_info.samplerate),
                "onset": round(float(onset), 6),
                "offset": round(float(offset), 6),
                "annotation_onset": "",
                "annotation_offset": "",
                "annotation_duration": "",
                "whistle_tag": "",
                "whistle_quality": "",
                "clicks_forbidden": bool(clicks_path is not None),
                "source_kind": "mined_noise",
                "annotation_line_number": "",
            }
        )

    rows_out = positive_rows + noise_rows
    rows_out = sorted(rows_out, key=lambda row: (float(row["onset"]), -int(row["label"])))

    with sf.SoundFile(str(full_recording_path), "r") as audio_file:
        for index, row in enumerate(rows_out, start=1):
            staged_name = staged_file_name(
                index=index,
                label=int(row["label"]),
                clip_kind=str(row["clip_kind"]),
                onset=float(row["onset"]),
                offset=float(row["offset"]),
            )
            destination_path = flat_recordings_dir / staged_name
            clip = extract_window_mono(audio_file, onset=float(row["onset"]), window_seconds=args.window_seconds)
            sf.write(str(destination_path), clip, audio_file.samplerate, subtype="PCM_16")
            row["staged_file_name"] = staged_name
            row["staged_file_path"] = str(destination_path)

    manifest_field_names = [
        "label",
        "clip_kind",
        "recording",
        "session_id",
        "source_audio_path",
        "staged_file_name",
        "staged_file_path",
        "recording_duration_seconds",
        "sample_rate",
        "onset",
        "offset",
        "annotation_onset",
        "annotation_offset",
        "annotation_duration",
        "whistle_tag",
        "whistle_quality",
        "clicks_forbidden",
        "source_kind",
        "annotation_line_number",
    ]
    write_csv(output_dir / "manifest_selected.csv", rows_out, manifest_field_names)

    quality_counts = (
        whistles["whistle_quality"].fillna(0).astype(int).value_counts().sort_index().to_dict()
    )
    tag_counts = whistles["whistle_tag"].astype(str).value_counts().sort_index().to_dict()
    summary_payload = {
        "input_descriptor": input_descriptor,
        "source_root": str(source_root),
        "archive_path": str(archive_path) if archive_path is not None else "",
        "full_recording_path": str(full_recording_path),
        "whistles_path": str(whistles_path),
        "clicks_path": str(clicks_path) if clicks_path is not None else "",
        "window_seconds": float(args.window_seconds),
        "margin_seconds": float(args.margin_seconds),
        "noise_to_whistle_ratio": float(args.noise_to_whistle_ratio),
        "seed": int(args.seed),
        "min_whistle_quality": int(args.min_whistle_quality),
        "exclude_multiple_whistles": bool(args.exclude_multiple_whistles),
        "ignore_clicks": bool(args.ignore_clicks),
        "raw_whistle_rows": raw_whistle_count,
        "selected_whistle_rows": len(positive_rows),
        "candidate_noise_windows": len(candidate_windows),
        "target_noise_rows": target_noise_rows,
        "sampled_noise_rows": len(noise_rows),
        "recording_duration_seconds": round(duration_seconds, 6),
        "sample_rate": int(audio_info.samplerate),
        "channels": int(audio_info.channels),
        "whistle_quality_counts": {str(key): int(value) for key, value in quality_counts.items()},
        "whistle_tag_counts": {str(key): int(value) for key, value in tag_counts.items()},
    }
    write_json(output_dir / "prepare_summary.json", summary_payload)

    dataset_card_lines = [
        "# Adriatic Clip Benchmark",
        "",
        "Clip-level whistle/noise benchmark built from the Adriatic trawling dataset.",
        "",
        "Rules used:",
        f"- positives are centered `{args.window_seconds:.3f}s` windows around whistle annotations",
        f"- whistle quality kept: `>= {args.min_whistle_quality}`",
        (
            "- multiple-whistle annotations are excluded"
            if args.exclude_multiple_whistles
            else "- multiple-whistle annotations are kept"
        ),
        (
            "- click intervals are excluded from negative mining"
            if clicks_path is not None
            else "- click intervals are ignored for negative mining"
        ),
        f"- negatives are `{args.window_seconds:.3f}s` windows mined with a safety margin of `{args.margin_seconds:.3f}s`",
        f"- target noise ratio: `{args.noise_to_whistle_ratio:.3f}`",
        "",
        f"Selected positives: {len(positive_rows)}",
        f"Sampled negatives: {len(noise_rows)}",
        f"Candidate negatives: {len(candidate_windows)}",
        "",
        "Files:",
        "- `manifest_selected.csv`",
        "- `prepare_summary.json`",
        "- `flat_recordings/`",
    ]
    (output_dir / "README.md").write_text("\n".join(dataset_card_lines) + "\n", encoding="utf-8")

    print(f"Input mode          : {input_descriptor}")
    print(f"Source root         : {source_root}")
    print(f"Archive path        : {archive_path if archive_path is not None else 'none'}")
    print(f"Full recording      : {full_recording_path}")
    print(f"Whistles file       : {whistles_path}")
    print(f"Clicks file         : {clicks_path if clicks_path is not None else 'ignored'}")
    print(f"Selected positives  : {len(positive_rows)}")
    print(f"Sampled negatives   : {len(noise_rows)}")
    print(f"Candidate negatives : {len(candidate_windows)}")
    print(f"Prepared dir        : {output_dir}")
    print(f"Flat recordings dir : {flat_recordings_dir}")


if __name__ == "__main__":
    main()
