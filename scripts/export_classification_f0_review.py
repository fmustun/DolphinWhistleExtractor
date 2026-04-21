#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import os
import re
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repair_classification_f0_dataset import decode_audio_payload


DEFAULT_DATASET_PATH = "cnn_dataset/datasets/classification_f0_rebuilt"
DEFAULT_OUTPUT_DIR = "reports/classification_f0_review"
DEFAULT_EXAMPLES_PER_SPLIT = 10
DEFAULT_SELECTION_MODE = "spread"
DEFAULT_RENDER_STYLE = "dataset"
DEFAULT_LEGACY_CONFIDENCE_THRESHOLD = 0.1
DEFAULT_LEGACY_NFFT = 1024
DEFAULT_LEGACY_YMIN_HZ = 0.0
DEFAULT_LEGACY_YMAX_HZ = 22_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a visual review of f0_spectrogram examples from a rebuilt "
            "classification dataset."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="DatasetDict path produced by rebuild_classification_f0_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG examples, CSV manifest, and index.html are written.",
    )
    parser.add_argument(
        "--examples-per-split",
        type=int,
        default=DEFAULT_EXAMPLES_PER_SPLIT,
        help="Number of examples exported for each split.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("spread", "first"),
        default=DEFAULT_SELECTION_MODE,
        help="Whether to sample evenly through the split or just take the first rows.",
    )
    parser.add_argument(
        "--render-style",
        choices=("dataset", "legacy"),
        default=DEFAULT_RENDER_STYLE,
        help=(
            "Use the stored dataset image or regenerate the historical-style F0 plot "
            "from audio + f0_* values."
        ),
    )
    parser.add_argument(
        "--legacy-confidence-threshold",
        type=float,
        default=DEFAULT_LEGACY_CONFIDENCE_THRESHOLD,
        help="Confidence threshold used when render-style=legacy.",
    )
    parser.add_argument(
        "--legacy-nfft",
        type=int,
        default=DEFAULT_LEGACY_NFFT,
        help="NFFT used when render-style=legacy.",
    )
    parser.add_argument(
        "--legacy-ymin-hz",
        type=float,
        default=DEFAULT_LEGACY_YMIN_HZ,
        help="Lower y-axis bound in Hz when render-style=legacy.",
    )
    parser.add_argument(
        "--legacy-ymax-hz",
        type=float,
        default=DEFAULT_LEGACY_YMAX_HZ,
        help="Upper y-axis bound in Hz when render-style=legacy.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.examples_per_split <= 0:
        raise ValueError("--examples-per-split must be strictly positive.")
    if not 0.0 <= args.legacy_confidence_threshold <= 1.0:
        raise ValueError("--legacy-confidence-threshold must be in [0, 1].")
    if args.legacy_nfft <= 0:
        raise ValueError("--legacy-nfft must be strictly positive.")
    if args.legacy_ymin_hz < 0.0:
        raise ValueError("--legacy-ymin-hz must be >= 0.")
    if args.legacy_ymax_hz <= args.legacy_ymin_hz:
        raise ValueError("--legacy-ymax-hz must be strictly greater than --legacy-ymin-hz.")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("_") or "sample"


def select_indices(num_rows: int, count: int, mode: str) -> list[int]:
    if num_rows <= count:
        return list(range(num_rows))
    if mode == "first":
        return list(range(count))
    indices = np.linspace(0, num_rows - 1, num=count, dtype=int)
    return [int(index) for index in indices]


def summarize_row(row: dict[str, object]) -> dict[str, object]:
    f0_hz = np.asarray(row["f0_hz"], dtype=np.float32)
    f0_conf = np.asarray(row["f0_conf"], dtype=np.float32)
    voiced = f0_conf > 0.05

    return {
        "num_points": int(len(f0_hz)),
        "voiced_ratio": float(np.mean(voiced)) if len(voiced) else 0.0,
        "f0_min_hz": float(np.min(f0_hz)) if len(f0_hz) else float("nan"),
        "f0_max_hz": float(np.max(f0_hz)) if len(f0_hz) else float("nan"),
        "conf_mean": float(np.mean(f0_conf)) if len(f0_conf) else float("nan"),
    }


def render_legacy_f0_plot(
    row: dict[str, object],
    *,
    confidence_threshold: float,
    nfft: int,
    ymin_hz: float,
    ymax_hz: float,
) -> Image.Image:
    audio, fs = decode_audio_payload(row["audio"])
    time = np.asarray(row["f0_time"], dtype=np.float32)
    f0 = np.asarray(row["f0_hz"], dtype=np.float32)
    confidence = np.asarray(row["f0_conf"], dtype=np.float32)

    if len(audio) == 0:
        return Image.new("RGB", (640, 480), color="white")

    mask = confidence > confidence_threshold
    duration = len(audio) / float(fs)
    fig_width = max(6.4, 6.4 * duration / 3.0)

    fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=150)
    ax.specgram(
        audio,
        Fs=fs,
        NFFT=nfft,
        noverlap=nfft - nfft // 8,
        cmap="Greys",
        vmin=-150,
    )
    if mask.any():
        scatter = ax.scatter(
            time[mask],
            f0[mask],
            c=confidence[mask],
            s=10,
            cmap="Reds",
            alpha=0.8,
            zorder=5,
        )
        fig.colorbar(scatter, ax=ax, label="Confidence")
    ax.set_xlim(0, duration)
    ax.set_ylim(float(ymin_hz), float(ymax_hz))
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    plt.close(fig)
    return Image.fromarray(image[:, :, :3], mode="RGB")


def export_split(
    *,
    split_name: str,
    split_dataset,
    output_dir: Path,
    examples_per_split: int,
    selection_mode: str,
    render_style: str,
    legacy_confidence_threshold: float,
    legacy_nfft: int,
    legacy_ymin_hz: float,
    legacy_ymax_hz: float,
) -> list[dict[str, object]]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset_index in select_indices(len(split_dataset), examples_per_split, selection_mode):
        row = split_dataset[int(dataset_index)]
        stats = summarize_row(row)

        if render_style == "legacy":
            image = render_legacy_f0_plot(
                row,
                confidence_threshold=legacy_confidence_threshold,
                nfft=legacy_nfft,
                ymin_hz=legacy_ymin_hz,
                ymax_hz=legacy_ymax_hz,
            )
        else:
            image = row["f0_spectrogram"]
            if not isinstance(image, Image.Image):
                image = Image.open(image)

        filename = (
            f"{split_name}_{int(dataset_index):05d}_"
            f"{slugify(str(row['name']))}_{float(row['onset']):.3f}_{float(row['offset']):.3f}.png"
        )
        image_path = split_dir / filename
        image.save(image_path)

        rows.append(
            {
                "split": split_name,
                "dataset_index": int(dataset_index),
                "image_path": str(image_path.relative_to(output_dir)),
                "name": str(row["name"]),
                "label": int(row["label"]),
                "onset": float(row["onset"]),
                "offset": float(row["offset"]),
                "duration": float(row["duration"]),
                "whistle_type": int(row["whistle_type"]),
                "whistle_name": str(row["whistle_name"]),
                "f0_ok": bool(row["f0_ok"]),
                "f0_bad_reason": str(row["f0_bad_reason"] or ""),
                "num_points": stats["num_points"],
                "voiced_ratio": stats["voiced_ratio"],
                "f0_min_hz": stats["f0_min_hz"],
                "f0_max_hz": stats["f0_max_hz"],
                "conf_mean": stats["conf_mean"],
            }
        )

    return rows


def write_manifest(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = output_dir / "sample_manifest.csv"
    fieldnames = [
        "split",
        "dataset_index",
        "image_path",
        "name",
        "label",
        "onset",
        "offset",
        "duration",
        "whistle_type",
        "whistle_name",
        "f0_ok",
        "f0_bad_reason",
        "num_points",
        "voiced_ratio",
        "f0_min_hz",
        "f0_max_hz",
        "conf_mean",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def build_card(row: dict[str, object]) -> str:
    image_rel = html.escape(str(row["image_path"]))
    title = html.escape(f"{row['split']} #{row['dataset_index']} - {row['name']}")
    whistle_name = html.escape(str(row["whistle_name"]))
    bad_reason = html.escape(str(row["f0_bad_reason"]))
    return f"""
    <article class="card">
      <img src="{image_rel}" alt="{title}">
      <div class="meta">
        <h2>{title}</h2>
        <p><strong>window</strong> {fmt_float(row['onset'])} - {fmt_float(row['offset'])} s</p>
        <p><strong>duration</strong> {fmt_float(row['duration'])} s</p>
        <p><strong>label</strong> {row['label']} | <strong>whistle_type</strong> {row['whistle_type']}</p>
        <p><strong>whistle_name</strong> {whistle_name}</p>
        <p><strong>f0_ok</strong> {row['f0_ok']} | <strong>reason</strong> {bad_reason or "-"}</p>
        <p><strong>points</strong> {row['num_points']} | <strong>voiced_ratio</strong> {fmt_float(row['voiced_ratio'])}</p>
        <p><strong>f0 range</strong> {fmt_float(row['f0_min_hz'], 1)} - {fmt_float(row['f0_max_hz'], 1)} Hz</p>
        <p><strong>mean conf</strong> {fmt_float(row['conf_mean'])}</p>
      </div>
    </article>
    """


def write_index_html(
    output_dir: Path,
    rows: list[dict[str, object]],
    dataset_path: Path,
    examples_per_split: int,
    selection_mode: str,
    render_style: str,
    legacy_ymin_hz: float,
    legacy_ymax_hz: float,
) -> Path:
    cards = "\n".join(build_card(row) for row in rows)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Classification F0 Review</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --paper: #fffaf0;
      --ink: #1d2a32;
      --muted: #5b6b73;
      --line: #d4c8b8;
      --accent: #d96846;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "DejaVu Sans", "Helvetica Neue", sans-serif;
      background: linear-gradient(180deg, #efe7d9 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px;
    }}
    header {{
      margin-bottom: 24px;
      padding: 20px 24px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 32px;
    }}
    .sub {{
      color: var(--muted);
      margin: 0;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 10px 24px rgba(35, 45, 54, 0.08);
    }}
    .card img {{
      display: block;
      width: 100%;
      image-rendering: pixelated;
      background: #000;
    }}
    .meta {{
      padding: 16px 18px 18px;
    }}
    .meta h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
      line-height: 1.35;
    }}
    .meta p {{
      margin: 6px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    strong {{
      color: var(--ink);
    }}
    a {{
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Classification F0 Review</h1>
      <p class="sub">
        Dataset: <code>{html.escape(str(dataset_path.resolve()))}</code><br>
        Selection: <code>{selection_mode}</code>, {examples_per_split} examples per split<br>
        Render style: <code>{html.escape(render_style)}</code>
        {f' | Y range: <code>{int(legacy_ymin_hz)}-{int(legacy_ymax_hz)} Hz</code>' if render_style == 'legacy' else ''}<br>
        Manifest: <a href="sample_manifest.csv">sample_manifest.csv</a>
      </p>
    </header>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    dataset_dict = load_from_disk(str(dataset_path))

    rows: list[dict[str, object]] = []
    for split_name in ("train", "test"):
        if split_name not in dataset_dict:
            continue
        rows.extend(
            export_split(
                split_name=split_name,
                split_dataset=dataset_dict[split_name],
                output_dir=output_dir,
                examples_per_split=args.examples_per_split,
                selection_mode=args.selection_mode,
                render_style=args.render_style,
                legacy_confidence_threshold=args.legacy_confidence_threshold,
                legacy_nfft=args.legacy_nfft,
                legacy_ymin_hz=args.legacy_ymin_hz,
                legacy_ymax_hz=args.legacy_ymax_hz,
            )
        )

    manifest_path = write_manifest(output_dir, rows)
    index_path = write_index_html(
        output_dir=output_dir,
        rows=rows,
        dataset_path=dataset_path,
        examples_per_split=args.examples_per_split,
        selection_mode=args.selection_mode,
        render_style=args.render_style,
        legacy_ymin_hz=args.legacy_ymin_hz,
        legacy_ymax_hz=args.legacy_ymax_hz,
    )

    print(f"Review HTML  : {index_path}")
    print(f"Review CSV   : {manifest_path}")
    print(f"Exported PNGs: {len(rows)}")


if __name__ == "__main__":
    main()
