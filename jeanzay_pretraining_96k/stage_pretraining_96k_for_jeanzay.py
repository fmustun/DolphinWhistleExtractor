#!/usr/bin/env python3
"""
Build a compact Jean Zay staging tree for the 96 kHz pretraining rebuild.

This script runs locally on the workstation that has access to the original
segment directories and 96 kHz recordings. It:
1. resolves the dataset rows against the local segment index
2. collects the exact segment files and source recordings needed
3. stages them under a portable directory tree
4. writes a relative path_config.json that the Jean Zay wrapper can consume
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination staging root to create locally before transfer to Jean Zay.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root containing scripts/rebuild_pretraining_96k.py.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation"],
        default=["train", "validation"],
        help="Only collect the files required for these splits.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to populate the staging data tree.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write manifests and path config, do not stage files.",
    )
    parser.add_argument(
        "--archive-path",
        default="",
        help=(
            "Optional .tar archive to create from the collected files. When set, "
            "the script writes generated metadata under <output-root> and stores "
            "the portable data payload in the tar instead of copying files one by one."
        ),
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete the staging root before regenerating it.",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Optional local prep cache dir. Defaults to <output-root>/generated/prep_cache.",
    )
    parser.add_argument(
        "--rebuild-index-cache",
        action="store_true",
        help="Force rebuild of the local segment index cache.",
    )
    parser.add_argument(
        "--resample-type",
        default="soxr_hq",
        help="Resampler to use while resolving ambiguous collision rows.",
    )
    return parser.parse_args()


def load_base_module(repo_root: Path):
    script_path = repo_root / "scripts" / "rebuild_pretraining_96k.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Base rebuild script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("rebuild_pretraining_96k_base", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_clean_output(output_root: Path, reset_output: bool) -> None:
    if reset_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generated").mkdir(parents=True, exist_ok=True)
    (output_root / "data").mkdir(parents=True, exist_ok=True)


def stage_file(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    if link_mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    raise ValueError(f"Unsupported link mode: {link_mode}")


def stage_pairs(pairs: list[tuple[Path, Path]], link_mode: str, description: str) -> None:
    for src, dst in tqdm(pairs, desc=description):
        stage_file(src, dst, link_mode)


def archive_pairs(
    archive_path: Path,
    output_root: Path,
    generated_files: list[Path],
    segment_pairs: list[tuple[Path, Path]],
    recording_pairs: list[tuple[Path, Path]],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w") as tar:
        for generated_file in generated_files:
            tar.add(
                generated_file,
                arcname=str(generated_file.relative_to(output_root)),
                recursive=False,
            )
        for src, dst in tqdm(segment_pairs, desc="archive:segments"):
            tar.add(src, arcname=str(dst.relative_to(output_root)), recursive=False)
        for src, dst in tqdm(recording_pairs, desc="archive:recordings"):
            tar.add(src, arcname=str(dst.relative_to(output_root)), recursive=False)


def make_relative_path_config(base_module, generated_dir: Path) -> dict:
    return {
        "source_repo": base_module.SOURCE_REPO,
        "target_repo": base_module.TARGET_REPO,
        "segment_dirs": {
            year_label: f"../data/segments/{year_label}"
            for year_label in base_module.YEAR_SEGMENT_DIRS
        },
        "recording_dirs": {
            year_label: [f"../data/recordings/{year_label}"]
            for year_label in base_module.RECORDING_DIRS
        },
    }


def main() -> None:
    args = parse_args()
    if args.manifest_only and args.archive_path:
        raise SystemExit("--manifest-only cannot be combined with --archive-path")

    output_root = Path(args.output_root).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    archive_path = Path(args.archive_path).expanduser().resolve() if args.archive_path else None
    generated_dir = output_root / "generated"
    data_dir = output_root / "data"
    ensure_clean_output(output_root, args.reset_output)

    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else (generated_dir / "prep_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_module = load_base_module(repo_root)
    segment_index = base_module.build_segment_index(
        cache_dir,
        rebuild_cache=args.rebuild_index_cache,
    )
    recording_resolver = base_module.RecordingResolver(cache_dir)
    collision_resolver = base_module.CollisionResolver(
        cache_dir,
        resample_type=args.resample_type,
    )

    required_segments: dict[Path, Path] = {}
    required_recordings: dict[Path, Path] = {}
    split_summaries: dict[str, dict[str, int]] = {}

    for split_name in args.splits:
        split_summary = {
            "rows_seen": 0,
            "unique_matches": 0,
            "collision_rows": 0,
            "collision_resolved": 0,
            "collision_fallback": 0,
            "collision_decode_error": 0,
            "missing_index": 0,
            "missing_recording": 0,
        }
        ds = base_module.load_source_split(split_name)
        for row in tqdm(ds, desc=f"collect:{split_name}"):
            split_summary["rows_seen"] += 1

            audio_path = base_module.normalize_audio_name(row["audio"])
            hf_year = int(row["year"])
            hf_hydro = row["hydrophone"]
            year_label = base_module.YEAR_INT_TO_LABEL.get(hf_year)
            if year_label is None:
                split_summary["missing_index"] += 1
                continue

            key = (year_label, hf_hydro, audio_path)
            candidates = segment_index.get(key, [])
            if not candidates:
                split_summary["missing_index"] += 1
                continue

            for exp_name, seg_path_str in candidates:
                seg_path = Path(seg_path_str)
                required_segments.setdefault(
                    seg_path,
                    data_dir / "segments" / year_label / exp_name / seg_path.name,
                )

            exp_name = None
            if len(candidates) == 1:
                exp_name = candidates[0][0]
                split_summary["unique_matches"] += 1
            else:
                split_summary["collision_rows"] += 1
                collision_key = base_module.stable_key_hash(
                    (split_name, year_label, hf_hydro, audio_path)
                )
                decoded_audio = base_module.decode_source_audio(row["audio"])
                if decoded_audio is None:
                    split_summary["collision_decode_error"] += 1
                    split_summary["collision_fallback"] += 1
                    exp_name = base_module.choose_collision_fallback(candidates)[0]
                else:
                    hf_audio_array, hf_sampling_rate = decoded_audio
                    match = collision_resolver.resolve(
                        collision_key,
                        hf_audio_array,
                        hf_sampling_rate,
                        candidates,
                    )
                    if match is not None:
                        exp_name = match[0]
                        split_summary["collision_resolved"] += 1
                    else:
                        exp_name = base_module.choose_collision_fallback(candidates)[0]
                        split_summary["collision_fallback"] += 1

            rec_path = recording_resolver.find(exp_name, year_label)
            if rec_path is None:
                split_summary["missing_recording"] += 1
                continue

            required_recordings.setdefault(
                rec_path,
                data_dir / "recordings" / year_label / rec_path.name,
            )

        split_summaries[split_name] = split_summary

    recording_resolver.flush()
    collision_resolver.flush()

    segment_pairs = sorted(required_segments.items(), key=lambda item: str(item[0]))
    recording_pairs = sorted(required_recordings.items(), key=lambda item: str(item[0]))

    manifest = {
        "repo_root": str(repo_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "link_mode": args.link_mode,
        "manifest_only": args.manifest_only,
        "archive_path": str(archive_path) if archive_path is not None else "",
        "segment_file_count": len(segment_pairs),
        "recording_file_count": len(recording_pairs),
        "split_summaries": split_summaries,
        "segment_files": [
            {
                "source": str(src),
                "staged_relative_path": str(dst.relative_to(output_root)),
            }
            for src, dst in segment_pairs
        ],
        "recording_files": [
            {
                "source": str(src),
                "staged_relative_path": str(dst.relative_to(output_root)),
            }
            for src, dst in recording_pairs
        ],
    }
    manifest_path = generated_dir / "transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    path_config = make_relative_path_config(base_module, generated_dir)
    path_config_path = generated_dir / "path_config.json"
    path_config_path.write_text(json.dumps(path_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "path_config_path": str(path_config_path),
        "archive_path": str(archive_path) if archive_path is not None else "",
        "staging_mode": (
            "manifest_only"
            if args.manifest_only
            else ("archive" if archive_path is not None else "tree")
        ),
        "segment_file_count": len(segment_pairs),
        "recording_file_count": len(recording_pairs),
        "split_summaries": split_summaries,
    }
    summary_path = generated_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if archive_path is not None:
        archive_pairs(
            archive_path,
            output_root,
            [manifest_path, path_config_path, summary_path],
            segment_pairs,
            recording_pairs,
        )
    elif not args.manifest_only:
        stage_pairs(segment_pairs, args.link_mode, "stage:segments")
        stage_pairs(recording_pairs, args.link_mode, "stage:recordings")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
