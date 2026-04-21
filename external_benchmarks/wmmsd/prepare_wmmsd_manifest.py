#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external_benchmarks.wmmsd.common import (
    DEFAULT_TARGET_SPECIES,
    SUPPORTED_AUDIO_EXTENSIONS,
    canonicalize_species,
    ensure_dir,
    infer_species_from_path,
    materialize_audio_file,
    stage_file_name,
    write_audio_bytes,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a clean, flat WMMSD staging directory for whistle-CNN inference. "
            "The script supports either a local WMMSD folder tree or an unofficial "
            "Hugging Face parquet mirror."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source-root",
        default="",
        help="Local WMMSD root directory to scan recursively for audio files.",
    )
    source_group.add_argument(
        "--hf-dataset-id",
        default="",
        help="Optional Hugging Face dataset id such as confit/wmms-parquet.",
    )
    parser.add_argument(
        "--hf-config",
        default="",
        help="Optional Hugging Face dataset config.",
    )
    parser.add_argument(
        "--hf-splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to load when using --hf-dataset-id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifests and staged flat recordings will be written.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=list(DEFAULT_TARGET_SPECIES),
        help=(
            "Canonical species names to keep. For a first pass, "
            "Bottlenose_Dolphin and Common_Dolphin are a good starting point."
        ),
    )
    parser.add_argument(
        "--path-contains",
        nargs="*",
        default=[],
        help=(
            "Optional case-insensitive substrings that must all appear in the "
            "source relative path. Useful later if the archive exposes whistle-like "
            "categories in filenames or folders."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How selected audio files are materialized into the flat staging directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of selected files to stage.",
    )
    return parser.parse_args()


def path_filter_match(relative_path: str, required_substrings: list[str]) -> bool:
    if not required_substrings:
        return True
    haystack = relative_path.lower()
    return all(fragment.lower() in haystack for fragment in required_substrings)


def build_local_rows(
    source_root: Path,
    target_species: set[str],
    required_substrings: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_root).as_posix()
        inferred_species = infer_species_from_path(Path(relative_path))
        supported = path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        species_match = inferred_species in target_species if target_species else True
        path_match = path_filter_match(relative_path, required_substrings)
        rows.append(
            {
                "source_type": "local",
                "source_path": str(path.resolve()),
                "source_rel_path": relative_path,
                "species": inferred_species,
                "audio_extension": path.suffix.lower(),
                "is_supported_audio": supported,
                "species_match": species_match,
                "path_filter_match": path_match,
                "selected_by_filters": supported and species_match and path_match,
                "hf_dataset_id": "",
                "hf_split": "",
                "hf_row_index": "",
                "staged_file_name": "",
                "staged_path": "",
                "stage_status": "",
            }
        )
    return rows


def _extract_audio_path(audio_value: object) -> str:
    if isinstance(audio_value, dict):
        return str(audio_value.get("path") or "")
    path_value = getattr(audio_value, "path", "")
    return str(path_value or "")


def build_hf_rows(
    dataset_id: str,
    dataset_config: str,
    split_names: list[str],
    target_species: set[str],
    required_substrings: list[str],
) -> list[dict[str, object]]:
    try:
        from datasets import Audio, Dataset, load_dataset
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "The `datasets` package is required for --hf-dataset-id mode."
        ) from exc

    def resolve_cached_arrow_dir() -> Path | None:
        dataset_cache_slug = dataset_id.replace("/", "___")
        base_dir = Path.home() / ".cache" / "huggingface" / "datasets" / dataset_cache_slug
        if dataset_config:
            candidate = base_dir / dataset_config / "0.0.0"
            versions = sorted(path for path in candidate.iterdir() if path.is_dir()) if candidate.exists() else []
        else:
            versions = []
            if base_dir.exists():
                for config_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
                    candidate = config_dir / "0.0.0"
                    if candidate.exists():
                        versions.extend(sorted(path for path in candidate.iterdir() if path.is_dir()))
        return versions[-1] if versions else None

    def iter_local_cached_rows() -> list[tuple[str, int, dict[str, object]]]:
        cache_dir = resolve_cached_arrow_dir()
        if cache_dir is None:
            return []

        indexed_rows: list[tuple[str, int, dict[str, object]]] = []
        for split_name in split_names:
            split_files = sorted(cache_dir.glob(f"*{split_name}*.arrow"))
            if not split_files:
                continue
            split_row_index = 0
            for arrow_path in split_files:
                dataset = Dataset.from_file(str(arrow_path))
                dataset = dataset.cast_column("audio", Audio(decode=False))
                for row in dataset:
                    indexed_rows.append((split_name, split_row_index, row))
                    split_row_index += 1
        return indexed_rows

    rows: list[dict[str, object]] = []
    indexed_rows = iter_local_cached_rows()

    if not indexed_rows:
        writable_cache_dir = REPO_ROOT / "benchmark_data" / ".hf_cache_writable"
        for split_name in split_names:
            dataset = load_dataset(
                dataset_id,
                dataset_config or None,
                split=split_name,
                cache_dir=str(writable_cache_dir),
            )
            dataset = dataset.cast_column("audio", Audio(decode=False))
            indexed_rows.extend((split_name, row_index, row) for row_index, row in enumerate(dataset))

    for split_name, row_index, row in indexed_rows:
        species = canonicalize_species(str(row.get("species", "")))
        audio_value = row.get("audio") or {}
        audio_path = _extract_audio_path(audio_value)
        audio_bytes = audio_value.get("bytes") if isinstance(audio_value, dict) else None
        audio_suffix = Path(audio_path).suffix.lower() if audio_path else ""
        relative_path = (
            f"{split_name}/{Path(audio_path).name}"
            if audio_path
            else f"{split_name}/row_{row_index:06d}{audio_suffix or '.wav'}"
        )
        supported = audio_suffix in SUPPORTED_AUDIO_EXTENSIONS
        species_match = species in target_species if target_species else True
        path_match = path_filter_match(relative_path, required_substrings)
        rows.append(
            {
                "source_type": "huggingface",
                "source_path": audio_path,
                "source_rel_path": relative_path,
                "species": species,
                "audio_extension": audio_suffix,
                "is_supported_audio": supported,
                "species_match": species_match,
                "path_filter_match": path_match,
                "selected_by_filters": (bool(audio_path) or bool(audio_bytes)) and supported and species_match and path_match,
                "hf_dataset_id": dataset_id,
                "hf_split": split_name,
                "hf_row_index": row_index,
                "staged_file_name": "",
                "staged_path": "",
                "stage_status": "",
                "_audio_bytes": audio_bytes,
            }
        )
    return rows


def stage_selected_rows(
    rows: list[dict[str, object]],
    flat_recordings_dir: Path,
    copy_mode: str,
    limit: int,
) -> list[dict[str, object]]:
    selected_candidates = [row for row in rows if bool(row["selected_by_filters"])]
    if limit > 0:
        selected_candidates = selected_candidates[:limit]

    staged_rows: list[dict[str, object]] = []
    for selection_index, row in enumerate(selected_candidates, start=1):
        source_path = Path(str(row["source_path"])).expanduser()
        suffix = str(row["audio_extension"] or source_path.suffix.lower())
        staged_name = stage_file_name(
            selection_index,
            str(row["species"]),
            str(row["source_rel_path"]),
            suffix,
        )
        staged_path = flat_recordings_dir / staged_name
        row["staged_file_name"] = staged_name
        row["staged_path"] = str(staged_path)

        if source_path.exists():
            materialize_audio_file(source_path, staged_path, copy_mode)
            row["stage_status"] = "staged"
            staged_rows.append(row)
            continue

        audio_bytes = row.get("_audio_bytes")
        if isinstance(audio_bytes, bytes) and audio_bytes:
            write_audio_bytes(staged_path, audio_bytes)
            row["stage_status"] = "staged_from_bytes"
            staged_rows.append(row)
            continue

        row["stage_status"] = "missing_source"
    return staged_rows


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    flat_recordings_dir = ensure_dir(output_dir / "flat_recordings")
    target_species = {canonicalize_species(value) for value in args.species}

    if args.source_root:
        source_root = Path(args.source_root).expanduser().resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root not found: {source_root}")
        rows = build_local_rows(
            source_root=source_root,
            target_species=target_species,
            required_substrings=args.path_contains,
        )
        source_mode = "local"
        source_descriptor = str(source_root)
    else:
        rows = build_hf_rows(
            dataset_id=args.hf_dataset_id,
            dataset_config=args.hf_config,
            split_names=list(args.hf_splits),
            target_species=target_species,
            required_substrings=args.path_contains,
        )
        source_mode = "huggingface"
        source_descriptor = args.hf_dataset_id

    staged_rows = stage_selected_rows(
        rows=rows,
        flat_recordings_dir=flat_recordings_dir,
        copy_mode=args.copy_mode,
        limit=args.limit,
    )

    manifest_fields = [
        "source_type",
        "source_path",
        "source_rel_path",
        "species",
        "audio_extension",
        "is_supported_audio",
        "species_match",
        "path_filter_match",
        "selected_by_filters",
        "hf_dataset_id",
        "hf_split",
        "hf_row_index",
        "staged_file_name",
        "staged_path",
        "stage_status",
    ]
    staged_fields = manifest_fields
    unsupported_rows = [
        row
        for row in rows
        if bool(row["species_match"])
        and bool(row["path_filter_match"])
        and not bool(row["is_supported_audio"])
    ]

    write_csv(output_dir / "manifest_all.csv", rows, manifest_fields)
    write_csv(output_dir / "manifest_selected.csv", staged_rows, staged_fields)
    write_csv(output_dir / "unsupported_files.csv", unsupported_rows, manifest_fields)
    (output_dir / "specific_files.txt").write_text(
        "".join(f"{row['staged_file_name']}\n" for row in staged_rows),
        encoding="utf-8",
    )

    summary_payload = {
        "source_mode": source_mode,
        "source_descriptor": source_descriptor,
        "target_species": sorted(target_species),
        "path_contains": list(args.path_contains),
        "copy_mode": args.copy_mode,
        "limit": args.limit,
        "total_candidates": len(rows),
        "filter_matched_candidates": sum(1 for row in rows if bool(row["selected_by_filters"])),
        "staged_files": len(staged_rows),
        "unsupported_matching_files": len(unsupported_rows),
        "flat_recordings_dir": str(flat_recordings_dir),
    }
    write_json(output_dir / "prepare_summary.json", summary_payload)

    print(f"Source mode              : {source_mode}")
    print(f"Source descriptor        : {source_descriptor}")
    print(f"Total candidates         : {len(rows)}")
    print(
        "Filter-matched candidates: "
        f"{sum(1 for row in rows if bool(row['selected_by_filters']))}"
    )
    print(f"Staged files             : {len(staged_rows)}")
    print(f"Flat recordings dir      : {flat_recordings_dir}")
    print(f"Manifest all             : {output_dir / 'manifest_all.csv'}")
    print(f"Manifest selected        : {output_dir / 'manifest_selected.csv'}")
    print(f"Specific files list      : {output_dir / 'specific_files.txt'}")


if __name__ == "__main__":
    main()
