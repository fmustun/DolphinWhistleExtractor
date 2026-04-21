#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.dclde.common import ensure_dir, gcs_download_url, write_json  # noqa: E402


BUCKET_API_ROOT = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
PREFIX = "dclde/2011/dclde_2011"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a selected subset of the public NOAA DCLDE 2011 corpus "
            "from Google Cloud Storage."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--subset",
        choices=("development", "evaluation", "both"),
        default="evaluation",
        help="Which audio split(s) to download.",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Keep only WAV files that have a matching 2011 .ann file.",
    )
    parser.add_argument(
        "--include-annotations2025",
        action="store_true",
        help="Also download revised 2025 annotation files when available.",
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Download README and overview markdown files.",
    )
    parser.add_argument(
        "--include-software",
        action="store_true",
        help="Download NOAA-provided software archives such as silbidopy.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=[],
        help="Optional exact species-folder filter (e.g. 'Tursiops truncatus-SoCal').",
    )
    parser.add_argument(
        "--limit-wavs",
        type=int,
        default=0,
        help="Optional cap on the number of WAV files downloaded after filtering.",
    )
    return parser.parse_args()


def list_objects(prefix: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    page_token = ""
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        url = BUCKET_API_ROOT + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken", "")
        if not page_token:
            break
    return items


def relative_species(object_name: str) -> str:
    parts = Path(object_name).parts
    if len(parts) < 5:
        return ""
    if parts[3] not in {"development", "evaluation"}:
        return ""
    return parts[4]


def subset_match(object_name: str, subset: str) -> bool:
    parts = Path(object_name).parts
    if len(parts) < 4:
        return False
    split = parts[3]
    if split not in {"development", "evaluation"}:
        return False
    return subset == "both" or split == subset


def select_object_names(
    objects: list[dict[str, object]],
    subset: str,
    annotated_only: bool,
    include_annotations2025: bool,
    include_docs: bool,
    include_software: bool,
    species_filter: set[str],
    limit_wavs: int,
) -> list[str]:
    object_names = sorted(str(item["name"]) for item in objects if item.get("name"))
    available = set(object_names)

    wav_names = []
    for name in object_names:
        if not name.endswith(".wav"):
            continue
        if not subset_match(name, subset):
            continue
        species = relative_species(name)
        if species_filter and species not in species_filter:
            continue
        if annotated_only and name[:-4] + ".ann" not in available:
            continue
        wav_names.append(name)

    if limit_wavs > 0:
        wav_names = wav_names[:limit_wavs]

    selected = set(wav_names)
    for wav_name in wav_names:
        ann_name = wav_name[:-4] + ".ann"
        if ann_name in available:
            selected.add(ann_name)
        if include_annotations2025:
            relative = Path(wav_name).relative_to(PREFIX)
            ann2025 = str((Path(PREFIX) / "annotations2025" / relative).with_suffix(".ann"))
            if ann2025 in available:
                selected.add(ann2025)

    if include_docs:
        for name in object_names:
            if Path(name).suffix.lower() == ".md":
                selected.add(name)

    if include_software:
        for name in object_names:
            if "/software/" in name and Path(name).suffix.lower() == ".zip":
                selected.add(name)

    return sorted(selected)


def download_object(object_name: str, output_dir: Path) -> Path:
    destination = output_dir / object_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    with urllib.request.urlopen(gcs_download_url(object_name), timeout=120) as response:
        destination.write_bytes(response.read())
    return destination


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    objects = list_objects(PREFIX)
    selected_names = select_object_names(
        objects=objects,
        subset=args.subset,
        annotated_only=bool(args.annotated_only),
        include_annotations2025=bool(args.include_annotations2025),
        include_docs=bool(args.include_docs),
        include_software=bool(args.include_software),
        species_filter=set(args.species),
        limit_wavs=int(args.limit_wavs),
    )

    downloaded_paths = [download_object(name, output_dir) for name in selected_names]
    payload = {
        "output_dir": str(output_dir),
        "requested_subset": args.subset,
        "annotated_only": bool(args.annotated_only),
        "include_annotations2025": bool(args.include_annotations2025),
        "include_docs": bool(args.include_docs),
        "include_software": bool(args.include_software),
        "species_filter": list(args.species),
        "limit_wavs": int(args.limit_wavs),
        "object_count": len(selected_names),
        "downloaded_count": len(downloaded_paths),
        "objects": selected_names,
    }
    write_json(output_dir / "download_summary.json", payload)

    print(f"Objects listed   : {len(objects)}")
    print(f"Objects selected : {len(selected_names)}")
    print(f"Output dir       : {output_dir}")
    print(f"Dataset root     : {output_dir / 'dclde' / '2011' / 'dclde_2011'}")


if __name__ == "__main__":
    main()

