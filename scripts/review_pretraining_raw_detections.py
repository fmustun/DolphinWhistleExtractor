#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.whistle_torch import (  # noqa: E402
    SpectrogramConfig,
    ensure_mono_float32,
    make_spectrogram_image,
    resample_audio_if_needed,
)


DEFAULT_DATASET_REPO = "dolphinteam/DolphinWhistle-Pretraining-2024-aug-dec-raw"
DEFAULT_PARQUET_URL = (
    "https://huggingface.co/datasets/"
    "dolphinteam/DolphinWhistle-Pretraining-2024-aug-dec-raw/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
DEFAULT_RECORDINGS_DIR = "/media/DOLPHIN1_robin/2024"
DEFAULT_OUTPUT_DIR = "reports/pretraining_2024_aug_dec_raw_review"
DEFAULT_FOCUS_MONTHS = ("2024-09", "2024-10")
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
FILE_NAME_PATTERN = re.compile(
    r"^Exp_(?P<day>\d{2})_(?P<month>[A-Za-z]{3})_(?P<year>\d{4})_"
    r"(?P<hour>\d{2})(?P<minute>\d{2})_channel_(?P<hydrophone>\d+)\.wav$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review raw whistle detections stored as metadata on Hugging Face, "
            "join them with local WAV recordings, and generate summary tables "
            "plus spectrogram contact sheets for manual verification."
        )
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help="Dataset repo id used in the JSON summary.",
    )
    parser.add_argument(
        "--parquet-url",
        default=DEFAULT_PARQUET_URL,
        help="Direct parquet URL for the raw detections dataset.",
    )
    parser.add_argument(
        "--recordings-dir",
        default=DEFAULT_RECORDINGS_DIR,
        help="Directory containing the source WAV recordings.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV/JSON/PNG outputs will be written.",
    )
    parser.add_argument(
        "--focus-months",
        nargs="+",
        default=list(DEFAULT_FOCUS_MONTHS),
        help="Months to prioritize for contact sheets, formatted as YYYY-MM.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Number of files to review per hydrophone within the focus months.",
    )
    parser.add_argument(
        "--spectrograms-per-file",
        type=int,
        default=4,
        help="Number of sampled detections rendered per reviewed file.",
    )
    parser.add_argument(
        "--min-review-duration-seconds",
        type=float,
        default=1800.0,
        help="Minimum file duration used when ranking files for review sheets.",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=96_000,
        help="Target sample rate used for review spectrograms.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square spectrogram size in pixels.",
    )
    parser.add_argument(
        "--refresh-parquet",
        action="store_true",
        help="Re-download the parquet file even if a cached copy already exists.",
    )
    return parser.parse_args()


def ensure_local_parquet(parquet_url: str, cache_path: Path, refresh: bool) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not cache_path.exists():
        print(f"Downloading parquet metadata to {cache_path}")
        urllib.request.urlretrieve(parquet_url, cache_path)
    return cache_path


def parse_file_name(file_name: str) -> dict[str, object]:
    match = FILE_NAME_PATTERN.match(file_name)
    if match is None:
        raise ValueError(f"Unsupported recording name: {file_name}")

    day = int(match.group("day"))
    month = MONTHS[match.group("month")]
    year = int(match.group("year"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    hydrophone = str(match.group("hydrophone"))

    timestamp = pd.Timestamp(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
    )
    return {
        "recording_datetime": timestamp,
        "date": timestamp.normalize(),
        "month": timestamp.strftime("%Y-%m"),
        "hydrophone_from_name": hydrophone,
    }


def load_detections(parquet_path: Path) -> pd.DataFrame:
    detections = pd.read_parquet(parquet_path)
    if detections.empty:
        raise ValueError(f"No detections found in {parquet_path}")

    unique_files = pd.DataFrame({"file_name": detections["file_name"].drop_duplicates()})
    parsed = unique_files["file_name"].map(parse_file_name).apply(pd.Series)
    unique_files = pd.concat([unique_files, parsed], axis=1)
    detections = detections.merge(unique_files, on="file_name", how="left")
    detections["hydrophone"] = detections["hydrophone"].astype(str)
    detections["hydrophone_name_matches_metadata"] = (
        detections["hydrophone"] == detections["hydrophone_from_name"]
    )
    detections["confidence"] = detections["confidence"].astype(np.float32)
    detections["duration"] = detections["duration"].astype(np.float32)
    detections["initial_point"] = detections["initial_point"].astype(np.float32)
    detections["finish_point"] = detections["finish_point"].astype(np.float32)
    return detections


def recording_path_for(file_name: str, recordings_dir: Path) -> Path:
    return recordings_dir / file_name


def probe_duration_seconds(audio_path: Path) -> float | None:
    if not audio_path.exists():
        return None
    try:
        return float(sf.info(str(audio_path)).duration)
    except Exception:
        return None


def build_file_stats(detections: pd.DataFrame, recordings_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = detections.groupby("file_name", sort=True)
    file_stats = grouped.agg(
        detections=("file_name", "size"),
        mean_confidence=("confidence", "mean"),
        median_confidence=("confidence", "median"),
        min_confidence=("confidence", "min"),
        max_confidence=("confidence", "max"),
        first_detection_seconds=("initial_point", "min"),
        last_detection_seconds=("finish_point", "max"),
        hydrophone=("hydrophone", "first"),
        hydrophone_name_matches_metadata=("hydrophone_name_matches_metadata", "all"),
        recording_datetime=("recording_datetime", "first"),
        date=("date", "first"),
        month=("month", "first"),
    ).reset_index()

    recording_paths = [recording_path_for(name, recordings_dir) for name in file_stats["file_name"]]
    file_stats["recording_path"] = [str(path) for path in recording_paths]
    file_stats["recording_exists"] = [path.exists() for path in recording_paths]
    file_stats["duration_seconds"] = [probe_duration_seconds(path) for path in recording_paths]
    file_stats["duration_from_audio_header"] = file_stats["duration_seconds"].notna()
    file_stats["duration_seconds"] = file_stats["duration_seconds"].fillna(
        file_stats["last_detection_seconds"].astype(float)
    )
    file_stats["duration_hours"] = file_stats["duration_seconds"] / 3600.0
    file_stats["detections_per_hour"] = np.where(
        file_stats["duration_hours"] > 0,
        file_stats["detections"] / file_stats["duration_hours"],
        np.nan,
    )

    missing = file_stats.loc[~file_stats["recording_exists"], [
        "file_name",
        "recording_path",
        "hydrophone",
        "month",
    ]].copy()
    return file_stats, missing


def summarize_by_period(
    detections: pd.DataFrame,
    file_stats: pd.DataFrame,
    period_column: str,
) -> pd.DataFrame:
    detection_summary = detections.groupby([period_column, "hydrophone"], sort=True).agg(
        total_detections=("file_name", "size"),
        mean_detection_confidence=("confidence", "mean"),
        median_detection_confidence=("confidence", "median"),
    ).reset_index()

    file_summary = file_stats.groupby([period_column, "hydrophone"], sort=True).agg(
        recording_files=("file_name", "size"),
        matched_recording_files=("recording_exists", "sum"),
        total_duration_seconds=("duration_seconds", "sum"),
        mean_file_confidence=("mean_confidence", "mean"),
        median_file_confidence=("mean_confidence", "median"),
    ).reset_index()

    summary = detection_summary.merge(
        file_summary,
        on=[period_column, "hydrophone"],
        how="left",
    )
    summary["total_duration_hours"] = summary["total_duration_seconds"] / 3600.0
    summary["detections_per_hour"] = np.where(
        summary["total_duration_hours"] > 0,
        summary["total_detections"] / summary["total_duration_hours"],
        np.nan,
    )
    return summary


def summarize_hydrophones(file_stats: pd.DataFrame, monthly_summary: pd.DataFrame) -> pd.DataFrame:
    first_seen = (
        file_stats.groupby("hydrophone", sort=True)["recording_datetime"]
        .min()
        .rename("first_recording_datetime")
        .reset_index()
    )
    summary = monthly_summary.groupby("hydrophone", sort=True).agg(
        months_present=("month", "nunique"),
        total_detections=("total_detections", "sum"),
        total_duration_seconds=("total_duration_seconds", "sum"),
    ).reset_index()
    summary["total_duration_hours"] = summary["total_duration_seconds"] / 3600.0
    summary["detections_per_hour"] = np.where(
        summary["total_duration_hours"] > 0,
        summary["total_detections"] / summary["total_duration_hours"],
        np.nan,
    )
    return summary.merge(first_seen, on="hydrophone", how="left")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_monthly_summaries(monthly_summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = monthly_summary.copy()
    plot_df["month"] = plot_df["month"].astype(str)
    hydrophones = sorted(plot_df["hydrophone"].astype(str).unique())

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    metric_specs = [
        ("total_detections", "Detections"),
        ("total_duration_hours", "Recording hours"),
        ("detections_per_hour", "Detections per hour"),
    ]

    for axis, (column, title) in zip(axes, metric_specs):
        for hydrophone in hydrophones:
            subset = plot_df.loc[plot_df["hydrophone"] == hydrophone].sort_values("month")
            axis.plot(
                subset["month"],
                subset[column],
                marker="o",
                linewidth=2,
                label=f"channel_{hydrophone}",
            )
        axis.set_title(title)
        axis.grid(True, axis="y", linestyle=":", alpha=0.4)
        axis.legend()

    axes[-1].set_xlabel("Month")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def read_audio_window(audio_path: Path, start_sec: float, end_sec: float) -> tuple[int, np.ndarray]:
    with sf.SoundFile(str(audio_path), "r") as handle:
        fs = int(handle.samplerate)
        start_frame = max(0, int(round(start_sec * fs)))
        stop_frame = max(start_frame + 1, int(round(end_sec * fs)))
        handle.seek(start_frame)
        audio = handle.read(stop_frame - start_frame, dtype="float32", always_2d=False)
    return fs, ensure_mono_float32(audio)


def render_detection_spectrogram(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    config: SpectrogramConfig,
) -> Image.Image:
    fs, audio = read_audio_window(audio_path, start_sec, end_sec)
    fs, audio = resample_audio_if_needed(audio, fs, config.target_fs)
    image_uint8 = make_spectrogram_image(audio, fs, config)
    return Image.fromarray(image_uint8)


def sample_detection_rows(rows: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    if rows.empty:
        return rows
    if len(rows) <= n_samples:
        return rows.sort_values("initial_point").reset_index(drop=True)
    ordered = rows.sort_values("initial_point").reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, num=n_samples, dtype=int)
    return ordered.iloc[indices].reset_index(drop=True)


def select_review_groups(
    file_stats: pd.DataFrame,
    focus_months: list[str],
    group_size: int,
    min_review_duration_seconds: float,
) -> dict[str, pd.DataFrame]:
    eligible = file_stats.loc[
        file_stats["month"].isin(focus_months)
        & (file_stats["duration_seconds"] >= float(min_review_duration_seconds))
    ].copy()
    eligible = eligible.sort_values(
        ["hydrophone", "detections_per_hour", "detections"],
        ascending=[True, False, False],
    )

    groups: dict[str, pd.DataFrame] = {}
    for hydrophone in sorted(eligible["hydrophone"].astype(str).unique()):
        group = eligible.loc[eligible["hydrophone"] == hydrophone].head(group_size).copy()
        if group.empty:
            continue
        groups[f"focus_channel_{hydrophone}"] = group
    return groups


def create_contact_sheet(
    group_name: str,
    file_rows: pd.DataFrame,
    detections: pd.DataFrame,
    output_dir: Path,
    spectrograms_per_file: int,
    config: SpectrogramConfig,
) -> tuple[Path, list[dict[str, object]]]:
    cell_size = config.image_size[0]
    margin = 18
    header_height = 80
    file_label_height = 58
    sample_label_height = 24
    row_height = file_label_height + cell_size + sample_label_height
    width = margin * (spectrograms_per_file + 1) + cell_size * spectrograms_per_file
    height = header_height + len(file_rows) * (row_height + margin) + margin

    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(28)
    label_font = get_font(15)
    small_font = get_font(13)

    draw.text((margin, 18), group_name.replace("_", " "), font=title_font, fill="black")
    draw.text(
        (margin, 48),
        f"{len(file_rows)} files, {spectrograms_per_file} detections sampled per file",
        font=small_font,
        fill="dimgray",
    )

    manifest_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(file_rows.to_dict(orient="records")):
        y_top = header_height + row_index * (row_height + margin)
        file_name = str(row["file_name"])
        audio_path = Path(str(row["recording_path"]))
        detections_for_file = sample_detection_rows(
            detections.loc[detections["file_name"] == file_name],
            spectrograms_per_file,
        )

        draw.text(
            (margin, y_top),
            (
                f"{row_index + 1}. {file_name}  |  detections={int(row['detections'])}  "
                f"|  per_hour={float(row['detections_per_hour']):.1f}"
            ),
            font=label_font,
            fill="black",
        )

        for sample_index, detection_row in enumerate(detections_for_file.to_dict(orient="records")):
            start_sec = float(detection_row["initial_point"])
            end_sec = float(detection_row["finish_point"])
            confidence = float(detection_row["confidence"])
            x_left = margin + sample_index * (cell_size + margin)
            y_image = y_top + file_label_height

            image = render_detection_spectrogram(
                audio_path=audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                config=config,
            )
            canvas.paste(image, (x_left, y_image))
            draw.text(
                (x_left, y_image + cell_size + 4),
                f"{start_sec:.1f}-{end_sec:.1f}s  conf={confidence:.3f}",
                font=small_font,
                fill="dimgray",
            )

            manifest_rows.append(
                {
                    "group": group_name,
                    "file_name": file_name,
                    "recording_path": str(audio_path),
                    "hydrophone": str(row["hydrophone"]),
                    "month": str(row["month"]),
                    "detections": int(row["detections"]),
                    "detections_per_hour": float(row["detections_per_hour"]),
                    "sample_index": sample_index,
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "confidence": confidence,
                }
            )

    output_path = output_dir / f"{group_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path, manifest_rows


def make_json_serializable(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_summary_payload(
    dataset_repo: str,
    parquet_url: str,
    detections: pd.DataFrame,
    file_stats: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    hydrophone_summary: pd.DataFrame,
    focus_months: list[str],
) -> dict[str, object]:
    matched_files = int(file_stats["recording_exists"].sum())
    total_duration_seconds = float(file_stats["duration_seconds"].sum())
    total_duration_hours = total_duration_seconds / 3600.0
    total_detections = int(len(detections))

    focus_rows = monthly_summary.loc[monthly_summary["month"].isin(focus_months)].copy()
    best_month_row = monthly_summary.sort_values("detections_per_hour", ascending=False).iloc[0]

    payload = {
        "dataset_repo": dataset_repo,
        "parquet_url": parquet_url,
        "total_detections": total_detections,
        "unique_recordings": int(file_stats["file_name"].nunique()),
        "matched_recording_files": matched_files,
        "missing_recording_files": int((~file_stats["recording_exists"]).sum()),
        "total_duration_seconds": total_duration_seconds,
        "total_duration_hours": total_duration_hours,
        "overall_detections_per_hour": (
            total_detections / total_duration_hours if total_duration_hours > 0 else None
        ),
        "focus_months": focus_months,
        "best_month_hydrophone_by_rate": {
            key: make_json_serializable(value)
            for key, value in best_month_row.to_dict().items()
        },
        "hydrophone_summary": [
            {
                key: make_json_serializable(value)
                for key, value in row.items()
            }
            for row in hydrophone_summary.to_dict(orient="records")
        ],
        "focus_month_summary": [
            {
                key: make_json_serializable(value)
                for key, value in row.items()
            }
            for row in focus_rows.to_dict(orient="records")
        ],
    }
    return payload


def main() -> None:
    args = parse_args()

    recordings_dir = Path(args.recordings_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    cache_dir = output_dir / "_cache"
    parquet_path = ensure_local_parquet(
        parquet_url=args.parquet_url,
        cache_path=cache_dir / "detections.parquet",
        refresh=bool(args.refresh_parquet),
    )

    if not recordings_dir.is_dir():
        raise FileNotFoundError(f"Recordings directory not found: {recordings_dir}")

    detections = load_detections(parquet_path)
    file_stats, missing_recordings = build_file_stats(detections, recordings_dir)
    monthly_summary = summarize_by_period(detections, file_stats, "month")
    daily_summary = summarize_by_period(detections, file_stats, "date")
    hydrophone_summary = summarize_hydrophones(file_stats, monthly_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe(file_stats.sort_values("detections", ascending=False), output_dir / "file_stats.csv")
    save_dataframe(monthly_summary, output_dir / "monthly_summary.csv")
    save_dataframe(daily_summary, output_dir / "daily_summary.csv")
    save_dataframe(hydrophone_summary, output_dir / "hydrophone_summary.csv")
    save_dataframe(missing_recordings, output_dir / "missing_recordings.csv")
    plot_monthly_summaries(monthly_summary, output_dir / "monthly_summary.png")

    config = SpectrogramConfig(
        image_size=(args.image_size, args.image_size),
        target_fs=args.target_fs,
    )
    review_groups = select_review_groups(
        file_stats=file_stats,
        focus_months=list(args.focus_months),
        group_size=int(args.group_size),
        min_review_duration_seconds=float(args.min_review_duration_seconds),
    )

    review_manifest_rows: list[dict[str, object]] = []
    review_manifest = {
        "focus_months": list(args.focus_months),
        "group_size": int(args.group_size),
        "spectrograms_per_file": int(args.spectrograms_per_file),
        "groups": {},
    }
    for group_name, rows in review_groups.items():
        image_path, manifest_rows = create_contact_sheet(
            group_name=group_name,
            file_rows=rows,
            detections=detections,
            output_dir=output_dir / "review_sheets",
            spectrograms_per_file=int(args.spectrograms_per_file),
            config=config,
        )
        review_manifest["groups"][group_name] = {
            "image_path": str(image_path),
            "files": [
                {
                    key: make_json_serializable(value)
                    for key, value in row.items()
                    if key
                    in {
                        "file_name",
                        "hydrophone",
                        "month",
                        "detections",
                        "duration_seconds",
                        "detections_per_hour",
                        "mean_confidence",
                    }
                }
                for row in rows.to_dict(orient="records")
            ],
        }
        review_manifest_rows.extend(manifest_rows)

    (output_dir / "review_sheets").mkdir(parents=True, exist_ok=True)
    (output_dir / "review_sheets" / "manifest.json").write_text(
        json.dumps(review_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(review_manifest_rows).to_csv(
        output_dir / "review_sheets" / "sample_manifest.csv",
        index=False,
    )

    summary_payload = build_summary_payload(
        dataset_repo=args.dataset_repo,
        parquet_url=args.parquet_url,
        detections=detections,
        file_stats=file_stats,
        monthly_summary=monthly_summary,
        hydrophone_summary=hydrophone_summary,
        focus_months=list(args.focus_months),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Detections loaded      : {len(detections)}")
    print(f"Unique recordings      : {file_stats['file_name'].nunique()}")
    print(f"Matched recordings     : {int(file_stats['recording_exists'].sum())}")
    print(f"Output directory       : {output_dir}")
    print(f"Monthly summary CSV    : {output_dir / 'monthly_summary.csv'}")
    print(f"Hydrophone summary CSV : {output_dir / 'hydrophone_summary.csv'}")
    print(f"Review manifest        : {output_dir / 'review_sheets' / 'manifest.json'}")


if __name__ == "__main__":
    main()
