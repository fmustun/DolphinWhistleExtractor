#!/usr/bin/env python3
"""
Verify the published 96 kHz pretraining dataset against the source dataset.

This verifier is designed to be practical on large audio datasets:
- it checks metadata row-by-row for the entire split
- it fully verifies ambiguous rows (collision rows) by comparing published
  audio against local candidate segments
- for source rows that can be decoded, it checks that the published row and
  the source row resolve to the same local candidate

The output is a JSON report with per-split summaries and detailed mismatch
records for rows that need manual review.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import zip_longest
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


DEFAULT_OUTPUT = Path("/tmp/pretraining_96k_verification_report.json")
FLOAT_TOL = 1e-4
MIN_CORR = 0.95


def load_base_module():
    script_path = Path(__file__).with_name("rebuild_pretraining_96k.py")
    spec = importlib.util.spec_from_file_location("rebuild_pretraining_96k", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
TARGET_AUDIO_DECODER = Audio()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo",
        default=BASE.SOURCE_REPO,
        help="Source HF dataset repo to compare against",
    )
    parser.add_argument(
        "--target-repo",
        default=BASE.TARGET_REPO,
        help="Published 96 kHz HF dataset repo to verify",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation"],
        default=["train", "validation"],
        help="Splits to verify",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/pretraining_96k_cache"),
        help="Cache dir used to load the local segment index",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to write the verification report JSON",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=100,
        help="Maximum number of detailed mismatch records to keep per split",
    )
    return parser.parse_args()


def load_stream(repo_id: str, split_name: str, *, raw_audio: bool):
    last_error = None
    auth_attempts = (
        {"streaming": True, "token": True},
        {"streaming": True},
    )
    for auth_kwargs in auth_attempts:
        try:
            ds = load_dataset(repo_id, split=split_name, **auth_kwargs)
            if raw_audio:
                ds = ds.cast_column("audio", BASE.SOURCE_AUDIO_FEATURE)
            return ds
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load split {split_name} from {repo_id}: {last_error}")


def normalize_audio(arr) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        # torchcodec returns [channels, samples], while soundfile usually gives
        # [samples, channels]. Collapse the likely channel axis in either case.
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32)


def decode_audio_field(audio_field) -> tuple[np.ndarray | None, int | None, str | None]:
    try:
        if hasattr(audio_field, "get_all_samples"):
            decoded = audio_field.get_all_samples()
            return normalize_audio(decoded.data), int(decoded.sample_rate), None
        if isinstance(audio_field, dict) and "array" in audio_field and "sampling_rate" in audio_field:
            return normalize_audio(audio_field["array"]), int(audio_field["sampling_rate"]), None
        decoded = TARGET_AUDIO_DECODER.decode_example(audio_field)
        return normalize_audio(decoded["array"]), int(decoded["sampling_rate"]), None
    except Exception as exc:
        return None, None, repr(exc)


def decode_source_audio_field(audio_field) -> tuple[np.ndarray | None, int | None, str | None]:
    decoded = BASE.decode_source_audio(audio_field)
    if decoded is None:
        return None, None, "source_decode_failed"
    audio, sampling_rate = decoded
    return audio, int(sampling_rate), None


def raw_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    if n < 100:
        return None
    aa = a[:n]
    bb = b[:n]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12)
    return float(np.abs(np.dot(aa, bb)) / denom)


def compare_float(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
    return abs(float(a) - float(b)) <= tol


def read_candidate_audio(seg_path: Path, target_sr: int) -> tuple[np.ndarray | None, int | None]:
    try:
        audio, sr = sf.read(str(seg_path), dtype="float32", always_2d=False)
    except Exception:
        return None, None

    audio = normalize_audio(audio)
    if len(audio) == 0 or not np.isfinite(audio).all():
        return None, sr
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return np.asarray(audio, dtype=np.float32), int(target_sr)


def read_candidate_audio_96k(
    recording_path: Path,
    start_time: float,
    end_time: float,
    target_sr: int,
) -> np.ndarray | None:
    try:
        audio = BASE.extract_segment_96k(recording_path, start_time, end_time)
    except Exception:
        return None

    audio = normalize_audio(audio)
    if len(audio) == 0 or not np.isfinite(audio).all():
        return None
    if BASE.TARGET_SR != target_sr:
        audio = librosa.resample(audio, orig_sr=BASE.TARGET_SR, target_sr=target_sr, res_type="soxr_hq")
    return np.asarray(audio, dtype=np.float32)


def resolve_audio_to_candidates(
    audio: np.ndarray,
    sampling_rate: int,
    candidates: list[tuple[str, str]],
) -> dict[str, object] | None:
    best_exp_name = None
    best_seg_path = None
    best_corr = -1.0
    details = []

    for exp_name, seg_path_str in candidates:
        seg_path = Path(seg_path_str)
        candidate_audio, _ = read_candidate_audio(seg_path, sampling_rate)
        frame_count = BASE.segment_frame_count(seg_path)
        corr = None
        if candidate_audio is not None:
            corr = raw_corr(audio, candidate_audio)
            if corr is not None and corr > best_corr:
                best_corr = corr
                best_exp_name = exp_name
                best_seg_path = str(seg_path)

        details.append({
            "exp_name": exp_name,
            "seg_path": str(seg_path),
            "frame_count": frame_count,
            "corr": corr,
        })

    if best_exp_name is None or best_corr < MIN_CORR:
        return None

    return {
        "exp_name": best_exp_name,
        "seg_path": best_seg_path,
        "best_corr": best_corr,
        "candidates": details,
    }


def resolve_target_audio_to_candidates(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    year_label: str,
    start_time: float,
    end_time: float,
    candidates: list[tuple[str, str]],
    recording_resolver,
) -> dict[str, object] | None:
    best_exp_name = None
    best_seg_path = None
    best_corr = -1.0
    details = []

    for exp_name, seg_path_str in candidates:
        recording_path = recording_resolver.find(exp_name, year_label)
        candidate_audio = None
        if recording_path is not None:
            candidate_audio = read_candidate_audio_96k(
                recording_path,
                start_time,
                end_time,
                sampling_rate,
            )

        corr = None
        if candidate_audio is not None:
            corr = raw_corr(audio, candidate_audio)
            if corr is not None and corr > best_corr:
                best_corr = corr
                best_exp_name = exp_name
                best_seg_path = seg_path_str

        details.append({
            "exp_name": exp_name,
            "seg_path": seg_path_str,
            "recording_path": None if recording_path is None else str(recording_path),
            "corr": corr,
        })

    if best_exp_name is None or best_corr < MIN_CORR:
        return None

    return {
        "exp_name": best_exp_name,
        "seg_path": best_seg_path,
        "best_corr": best_corr,
        "candidates": details,
    }


def metadata_mismatch(source_row, target_row) -> dict[str, dict[str, float | int | str]] | None:
    mismatches: dict[str, dict[str, float | int | str]] = {}
    fields = ("year", "hydrophone")
    for field in fields:
        if source_row[field] != target_row[field]:
            mismatches[field] = {"source": source_row[field], "target": target_row[field]}

    float_fields = ("start_time", "end_time", "duration")
    for field in float_fields:
        if not compare_float(source_row[field], target_row[field]):
            mismatches[field] = {"source": float(source_row[field]), "target": float(target_row[field])}

    return mismatches or None


def build_row_record(
    row_index: int,
    source_row,
    target_row,
    *,
    reason: str,
    extra: dict | None = None,
) -> dict:
    record = {
        "row_1based": row_index,
        "reason": reason,
        "source": {
            "year": int(source_row["year"]),
            "hydrophone": source_row["hydrophone"],
            "start_time": float(source_row["start_time"]),
            "end_time": float(source_row["end_time"]),
            "duration": float(source_row["duration"]),
            "audio_path": BASE.normalize_audio_name(source_row["audio"]),
        },
        "target": {
            "year": int(target_row["year"]),
            "hydrophone": target_row["hydrophone"],
            "start_time": float(target_row["start_time"]),
            "end_time": float(target_row["end_time"]),
            "duration": float(target_row["duration"]),
        },
    }
    if extra:
        record.update(extra)
    return record


def verify_split(
    split_name: str,
    *,
    source_repo: str,
    target_repo: str,
    segment_index: dict[tuple[str, str, str], list[tuple[str, str]]],
    recording_resolver,
    max_details: int,
) -> dict:
    source_ds = load_stream(source_repo, split_name, raw_audio=True)
    target_ds = load_stream(target_repo, split_name, raw_audio=False)

    summary = {
        "split": split_name,
        "rows_seen": 0,
        "source_rows_only": 0,
        "target_rows_only": 0,
        "metadata_mismatch_rows": 0,
        "missing_year_label_rows": 0,
        "missing_index_rows": 0,
        "unique_rows": 0,
        "collision_rows": 0,
        "collision_source_decode_error_rows": 0,
        "collision_target_decode_error_rows": 0,
        "collision_unresolved_target_rows": 0,
        "collision_expected_match_rows": 0,
        "collision_mismatch_rows": 0,
        "collision_fallback_rows": 0,
        "collision_fallback_match_rows": 0,
        "details": [],
    }

    iterator = zip_longest(source_ds, target_ds, fillvalue=None)
    progress = tqdm(iterator, desc=f"verify:{split_name}")
    for row_index, (source_row, target_row) in enumerate(progress, start=1):
        if source_row is None:
            summary["target_rows_only"] += 1
            continue
        if target_row is None:
            summary["source_rows_only"] += 1
            continue

        summary["rows_seen"] += 1
        mismatch = metadata_mismatch(source_row, target_row)
        if mismatch is not None:
            summary["metadata_mismatch_rows"] += 1
            if len(summary["details"]) < max_details:
                summary["details"].append(build_row_record(
                    row_index,
                    source_row,
                    target_row,
                    reason="metadata_mismatch",
                    extra={"mismatch": mismatch},
                ))

        audio_path = BASE.normalize_audio_name(source_row["audio"])
        year_label = BASE.YEAR_INT_TO_LABEL.get(int(source_row["year"]))
        if year_label is None:
            summary["missing_year_label_rows"] += 1
            continue

        key = (year_label, source_row["hydrophone"], audio_path)
        candidates = segment_index.get(key, [])
        if not candidates:
            summary["missing_index_rows"] += 1
            if len(summary["details"]) < max_details:
                summary["details"].append(build_row_record(
                    row_index,
                    source_row,
                    target_row,
                    reason="missing_index",
                ))
            continue

        if len(candidates) == 1:
            summary["unique_rows"] += 1
            continue

        summary["collision_rows"] += 1
        source_audio, source_sr, source_decode_error = decode_source_audio_field(source_row["audio"])
        target_audio, target_sr, target_decode_error = decode_audio_field(target_row["audio"])

        if source_decode_error is not None or source_audio is None:
            summary["collision_source_decode_error_rows"] += 1
        if target_decode_error is not None or target_audio is None:
            summary["collision_target_decode_error_rows"] += 1

        expected = None
        expectation_reason = None
        if source_audio is not None and source_sr is not None:
            expected = resolve_audio_to_candidates(source_audio, source_sr, candidates)
            expectation_reason = "source_resolution"

        if expected is None:
            fallback_exp_name, fallback_seg_path = BASE.choose_collision_fallback(candidates)
            expected = {
                "exp_name": fallback_exp_name,
                "seg_path": str(fallback_seg_path),
                "best_corr": None,
                "candidates": BASE.collision_candidate_details(candidates),
            }
            expectation_reason = "source_fallback"
            summary["collision_fallback_rows"] += 1

        actual = None
        if target_audio is not None and target_sr is not None:
            actual = resolve_target_audio_to_candidates(
                target_audio,
                target_sr,
                year_label=year_label,
                start_time=float(source_row["start_time"]),
                end_time=float(source_row["end_time"]),
                candidates=candidates,
                recording_resolver=recording_resolver,
            )

        if actual is None:
            summary["collision_unresolved_target_rows"] += 1
            if len(summary["details"]) < max_details:
                summary["details"].append(build_row_record(
                    row_index,
                    source_row,
                    target_row,
                    reason="target_collision_unresolved",
                    extra={
                        "expectation_reason": expectation_reason,
                        "expected": expected,
                        "source_decode_error": source_decode_error,
                        "target_decode_error": target_decode_error,
                    },
                ))
            continue

        if actual["exp_name"] == expected["exp_name"]:
            if expectation_reason == "source_fallback":
                summary["collision_fallback_match_rows"] += 1
            else:
                summary["collision_expected_match_rows"] += 1
            continue

        summary["collision_mismatch_rows"] += 1
        if len(summary["details"]) < max_details:
            summary["details"].append(build_row_record(
                row_index,
                source_row,
                target_row,
                reason="collision_candidate_mismatch",
                extra={
                    "expectation_reason": expectation_reason,
                    "expected": expected,
                    "actual": actual,
                    "source_decode_error": source_decode_error,
                    "target_decode_error": target_decode_error,
                },
            ))

    return summary


def main() -> None:
    args = parse_args()
    segment_index = BASE.build_segment_index(args.cache_dir, rebuild_cache=False)
    recording_resolver = BASE.RecordingResolver(args.cache_dir)

    report = {
        "source_repo": args.source_repo,
        "target_repo": args.target_repo,
        "splits": {},
    }

    for split_name in args.splits:
        report["splits"][split_name] = verify_split(
            split_name,
            source_repo=args.source_repo,
            target_repo=args.target_repo,
            segment_index=segment_index,
            recording_resolver=recording_resolver,
            max_details=args.max_details,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved verification report to {args.output_json}")


if __name__ == "__main__":
    main()
