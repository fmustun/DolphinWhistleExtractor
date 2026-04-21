from __future__ import annotations

import csv
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    if not slug:
        slug = "item"
    return slug[:max_length]


def staged_file_name(
    index: int,
    label: int,
    clip_kind: str,
    onset: float,
    offset: float,
    suffix: str = ".wav",
) -> str:
    onset_ms = int(round(float(onset) * 1000.0))
    offset_ms = int(round(float(offset) * 1000.0))
    label_slug = "whistle" if int(label) == 1 else "noise"
    kind_slug = safe_slug(clip_kind or "clip", max_length=40)
    return f"{index:05d}__{label_slug}__{kind_slug}__{onset_ms:09d}_{offset_ms:09d}{suffix}"
