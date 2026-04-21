#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from io import BytesIO
from math import gcd
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Image as HFImage,
    Sequence,
    Value,
    load_dataset,
)
from PIL import Image as PILImage
from scipy.signal import resample_poly
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.signal.windows import blackman


DEFAULT_CLASSIFICATION_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--dolphinteam--DophinWhistle-Classification-Finetuning"
)
DEFAULT_REFERENCE_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--dolphinteam--DolphinWhistle-Finetuning-withF0"
)
DEFAULT_CLASSIFICATION_REPO_ID = "dolphinteam/DophinWhistle-Classification-Finetuning"
DEFAULT_OUTPUT_DIR = Path("cnn_dataset/datasets/classification_f0_default_fixed")
DEFAULT_ARTIFACTS_DIR = Path("cnn_dataset/artifacts_classification_f0_default_fixed")
DEFAULT_CACHE_DIR = Path("/tmp/classification_f0_repair_cache")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CUT_LOW_FREQUENCY = 2.0
DEFAULT_CUT_HIGH_FREQUENCY = 22.0
DEFAULT_WLEN = 1024
DEFAULT_NFFT = 1024
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_SHARD_SIZE = "500MB"
SUPPORTED_SPLITS = ("train", "test")


@dataclass(frozen=True)
class ReferenceF0:
    f0_time: list[float]
    f0_hz: list[float]
    f0_conf: list[float]
    f0_ok: bool
    f0_bad_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair dolphinteam/DophinWhistle-Classification-Finetuning by "
            "replacing incorrect F0 tracks with the authoritative values from "
            "dolphinteam/DolphinWhistle-Finetuning-withF0 and regenerating a "
            "deterministic f0_spectrogram column."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classification-cache-root",
        default=str(DEFAULT_CLASSIFICATION_CACHE_ROOT),
        help="Local HF Hub cache root for the classification dataset repo.",
    )
    parser.add_argument(
        "--reference-cache-root",
        default=str(DEFAULT_REFERENCE_CACHE_ROOT),
        help="Local HF Hub cache root for the reference withF0 dataset repo.",
    )
    parser.add_argument(
        "--classification-revision",
        default="main",
        help="Classification revision to read from the local HF Hub cache.",
    )
    parser.add_argument(
        "--reference-revision",
        default="main",
        help="Reference withF0 revision to read from the local HF Hub cache.",
    )
    parser.add_argument(
        "--classification-config",
        default="default",
        help="Classification config/subdirectory to repair (default, all, unbalanced).",
    )
    parser.add_argument(
        "--reference-config",
        default="default",
        help="Reference withF0 config/subdirectory to use.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save the repaired DatasetDict with save_to_disk().",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Where to write JSON/README repair artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Writable cache dir used while loading local parquet shards.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output f0_spectrogram size in pixels (square).",
    )
    parser.add_argument(
        "--cut-low-frequency",
        type=float,
        default=DEFAULT_CUT_LOW_FREQUENCY,
        help="Lower frequency bound (kHz) for rendered spectrograms.",
    )
    parser.add_argument(
        "--cut-high-frequency",
        type=float,
        default=DEFAULT_CUT_HIGH_FREQUENCY,
        help="Upper frequency bound (kHz) for rendered spectrograms.",
    )
    parser.add_argument(
        "--wlen",
        type=int,
        default=DEFAULT_WLEN,
        help="Spectrogram window length.",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=DEFAULT_NFFT,
        help="FFT size used for the spectrogram.",
    )
    parser.add_argument(
        "--target-fs",
        type=int,
        default=None,
        help="Optional target sample rate for spectrogram rendering. Default keeps the original clip rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="datasets.map batch size for the repair pass.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional smoke-test limit applied independently to each split.",
    )
    parser.add_argument(
        "--allow-partial-reference",
        action="store_true",
        help=(
            "Allow rows without a withF0 match to keep their original f0_* values. "
            "By default the script aborts if any row is missing a reference match."
        ),
    )
    parser.add_argument(
        "--push-repo-id",
        default=None,
        help="Optional Hugging Face dataset repo id to push the repaired config to.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token used for push_to_hub().",
    )
    parser.add_argument(
        "--commit-message",
        default="Repair F0 tracks and regenerate f0_spectrogram",
        help="Commit message used when pushing to the Hub.",
    )
    parser.add_argument(
        "--max-shard-size",
        default=DEFAULT_MAX_SHARD_SIZE,
        help="Shard size passed to push_to_hub().",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size <= 0:
        raise ValueError("--image-size must be strictly positive.")
    if args.cut_low_frequency < 0:
        raise ValueError("--cut-low-frequency must be non-negative.")
    if args.cut_high_frequency <= args.cut_low_frequency:
        raise ValueError("--cut-high-frequency must be greater than --cut-low-frequency.")
    if args.wlen <= 0:
        raise ValueError("--wlen must be strictly positive.")
    if args.nfft < args.wlen:
        raise ValueError("--nfft must be >= --wlen.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be strictly positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be strictly positive when provided.")


def resolve_revision(cache_root: Path, revision: str) -> str:
    if revision != "main":
        return revision
    ref_path = cache_root / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing refs/main in cache root: {cache_root}")
    return ref_path.read_text(encoding="utf-8").strip()


def config_dir_name(config_name: str) -> str:
    return "data" if config_name == "default" else config_name


def snapshot_split_dir(cache_root: Path, revision: str, config_name: str) -> Path:
    resolved_revision = resolve_revision(cache_root, revision)
    split_dir = cache_root / "snapshots" / resolved_revision / config_dir_name(config_name)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing snapshot split directory for config={config_name!r}: {split_dir}"
        )
    return split_dir


def ensure_output_dir_is_empty(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return

    existing_entries = list(path.iterdir())
    if existing_entries:
        raise FileExistsError(f"Output directory already exists and is not empty: {path}")


def build_data_files(split_dir: Path, split_names: tuple[str, ...]) -> dict[str, list[str]]:
    data_files: dict[str, list[str]] = {}
    for split_name in split_names:
        files = sorted(split_dir.glob(f"{split_name}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet shards found for split={split_name!r} under {split_dir}"
            )
        data_files[split_name] = [str(path) for path in files]
    return data_files


def load_classification_dataset(
    *,
    split_dir: Path,
    cache_dir: Path,
    split_names: tuple[str, ...],
    limit: int | None,
) -> DatasetDict:
    data_files = build_data_files(split_dir, split_names)
    loaded = load_dataset("parquet", data_files=data_files, cache_dir=str(cache_dir))

    prepared_splits: dict[str, Dataset] = {}
    for split_name in split_names:
        split_dataset = loaded[split_name].cast_column("audio", Audio(decode=False))
        if "f0_spectrogram" in split_dataset.column_names:
            split_dataset = split_dataset.remove_columns(["f0_spectrogram"])
        if limit is not None:
            split_dataset = split_dataset.select(range(min(limit, split_dataset.num_rows)))
        prepared_splits[split_name] = split_dataset

    return DatasetDict(prepared_splits)


def row_key(name: str, onset: float, offset: float) -> tuple[str, float, float]:
    return (str(name), round(float(onset), 6), round(float(offset), 6))


def load_reference_lookup(
    *,
    split_dir: Path,
    split_names: tuple[str, ...],
) -> dict[str, dict[tuple[str, float, float], ReferenceF0]]:
    lookup_by_split: dict[str, dict[tuple[str, float, float], ReferenceF0]] = {}
    wanted_columns = [
        "name",
        "onset",
        "offset",
        "f0_time",
        "f0_hz",
        "f0_conf",
        "f0_ok",
        "f0_bad_reason",
    ]

    for split_name in split_names:
        split_lookup: dict[tuple[str, float, float], ReferenceF0] = {}
        files = sorted(split_dir.glob(f"{split_name}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No reference parquet shards found for split={split_name!r} under {split_dir}"
            )

        for parquet_path in files:
            rows = pq.read_table(parquet_path, columns=wanted_columns).to_pylist()
            for row in rows:
                key = row_key(row["name"], row["onset"], row["offset"])
                split_lookup[key] = ReferenceF0(
                    f0_time=[float(value) for value in row["f0_time"]],
                    f0_hz=[float(value) for value in row["f0_hz"]],
                    f0_conf=[float(value) for value in row["f0_conf"]],
                    f0_ok=bool(row["f0_ok"]),
                    f0_bad_reason=str(row["f0_bad_reason"] or ""),
                )

        lookup_by_split[split_name] = split_lookup

    return lookup_by_split


def split_coverage_summary(
    split_dataset: Dataset,
    split_lookup: dict[tuple[str, float, float], ReferenceF0],
) -> dict[str, object]:
    keys = [
        row_key(name, onset, offset)
        for name, onset, offset in zip(
            split_dataset["name"],
            split_dataset["onset"],
            split_dataset["offset"],
        )
    ]
    missing_keys = [key for key in keys if key not in split_lookup]
    return {
        "rows": int(split_dataset.num_rows),
        "matched_rows": int(len(keys) - len(missing_keys)),
        "missing_rows": int(len(missing_keys)),
        "missing_examples": [
            {
                "name": name,
                "onset": onset,
                "offset": offset,
            }
            for name, onset, offset in missing_keys[:5]
        ],
    }


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.asarray(audio, dtype=np.float32)


def decode_audio_payload(audio_payload: object) -> tuple[np.ndarray, int]:
    if not isinstance(audio_payload, dict):
        raise TypeError(f"Unsupported audio payload type: {type(audio_payload)!r}")

    audio_bytes = audio_payload.get("bytes")
    audio_path = audio_payload.get("path")
    if audio_bytes is not None:
        audio, fs = sf.read(BytesIO(audio_bytes), always_2d=False)
        return ensure_mono_float32(audio), int(fs)

    if audio_path:
        audio, fs = sf.read(str(audio_path), always_2d=False)
        return ensure_mono_float32(audio), int(fs)

    raise ValueError("Audio payload must contain either bytes or path.")


def resample_if_needed(audio: np.ndarray, fs: int, target_fs: int | None) -> tuple[np.ndarray, int]:
    if target_fs is None or int(target_fs) == int(fs):
        return ensure_mono_float32(audio), int(fs)

    factor = gcd(int(fs), int(target_fs))
    up = int(target_fs) // factor
    down = int(fs) // factor
    resampled = resample_poly(ensure_mono_float32(audio), up, down)
    return np.asarray(resampled, dtype=np.float32), int(target_fs)


def base_spectrogram_image(
    *,
    audio: np.ndarray,
    fs: int,
    image_size: int,
    cut_low_frequency: float,
    cut_high_frequency: float,
    wlen: int,
    nfft: int,
) -> np.ndarray:
    hop = round(0.5 * wlen)
    win = blackman(wlen, sym=False)
    f, _, sxx = scipy_spectrogram(
        ensure_mono_float32(audio),
        fs,
        nperseg=wlen,
        noverlap=wlen - hop,
        nfft=nfft,
        window=win,
        scaling="density",
        mode="psd",
    )

    sxx = 10.0 * np.log10(np.abs(sxx) + 1e-19)
    sxx = (sxx - np.min(sxx)) / (np.max(sxx) - np.min(sxx) + 1e-12) * 255.0

    low_hz = float(cut_low_frequency) * 1000.0
    high_hz = float(cut_high_frequency) * 1000.0
    low_idx = int(np.searchsorted(f, low_hz))
    high_idx = int(np.searchsorted(f, high_hz))
    high_idx = max(high_idx, low_idx + 1)

    cropped = np.flipud(sxx[low_idx:high_idx, :])
    gray = np.clip(cropped, 0, 255).astype(np.uint8)
    resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)


def overlay_f0_track(
    image_rgb: np.ndarray,
    *,
    duration_seconds: float,
    f0_time: list[float],
    f0_hz: list[float],
    cut_low_frequency: float,
    cut_high_frequency: float,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    duration = max(float(duration_seconds), 1e-9)
    low_hz = float(cut_low_frequency) * 1000.0
    high_hz = float(cut_high_frequency) * 1000.0

    points: list[tuple[int, int]] = []
    for time_seconds, hz in zip(f0_time, f0_hz):
        time_value = float(time_seconds)
        hz_value = float(hz)
        if not np.isfinite(time_value) or not np.isfinite(hz_value):
            continue
        if time_value < 0.0 or time_value > duration:
            continue
        if hz_value < low_hz or hz_value > high_hz:
            continue

        x = int(round((time_value / duration) * (width - 1)))
        y = int(round(((high_hz - hz_value) / (high_hz - low_hz)) * (height - 1)))
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        points.append((x, y))

    if len(points) >= 2:
        polyline = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            image_rgb,
            [polyline],
            isClosed=False,
            color=(255, 96, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    for x, y in points:
        cv2.circle(
            image_rgb,
            center=(x, y),
            radius=1,
            color=(255, 220, 0),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    return image_rgb


def render_f0_spectrogram(
    *,
    audio_payload: object,
    reference: ReferenceF0,
    image_size: int,
    cut_low_frequency: float,
    cut_high_frequency: float,
    wlen: int,
    nfft: int,
    target_fs: int | None,
) -> np.ndarray:
    audio, fs = decode_audio_payload(audio_payload)
    audio, fs = resample_if_needed(audio, fs, target_fs)
    image_rgb = base_spectrogram_image(
        audio=audio,
        fs=fs,
        image_size=image_size,
        cut_low_frequency=cut_low_frequency,
        cut_high_frequency=cut_high_frequency,
        wlen=wlen,
        nfft=nfft,
    )
    duration_seconds = float(len(audio)) / float(fs)
    rendered = overlay_f0_track(
        image_rgb,
        duration_seconds=duration_seconds,
        f0_time=reference.f0_time,
        f0_hz=reference.f0_hz,
        cut_low_frequency=cut_low_frequency,
        cut_high_frequency=cut_high_frequency,
    )
    rendered_uint8 = np.ascontiguousarray(rendered, dtype=np.uint8)
    return PILImage.fromarray(rendered_uint8, mode="RGB")


def repair_split_dataset(
    *,
    split_name: str,
    split_dataset: Dataset,
    split_lookup: dict[tuple[str, float, float], ReferenceF0],
    image_size: int,
    cut_low_frequency: float,
    cut_high_frequency: float,
    wlen: int,
    nfft: int,
    target_fs: int | None,
    batch_size: int,
    allow_partial_reference: bool,
) -> Dataset:
    output_features = Features(dict(split_dataset.features))
    output_features["f0_time"] = Sequence(Value("float32"))
    output_features["f0_hz"] = Sequence(Value("float32"))
    output_features["f0_conf"] = Sequence(Value("float32"))
    output_features["f0_ok"] = Value("bool")
    output_features["f0_bad_reason"] = Value("string")
    output_features["f0_spectrogram"] = HFImage()

    def repair_batch(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        repaired = {
            "f0_time": [],
            "f0_hz": [],
            "f0_conf": [],
            "f0_ok": [],
            "f0_bad_reason": [],
            "f0_spectrogram": [],
        }

        for index in range(len(batch["name"])):
            key = row_key(
                batch["name"][index],
                batch["onset"][index],
                batch["offset"][index],
            )
            reference = split_lookup.get(key)
            if reference is None:
                if not allow_partial_reference:
                    raise KeyError(
                        f"Missing reference F0 track for split={split_name!r} row={key!r}"
                    )
                reference = ReferenceF0(
                    f0_time=[float(value) for value in batch["f0_time"][index]],
                    f0_hz=[float(value) for value in batch["f0_hz"][index]],
                    f0_conf=[float(value) for value in batch["f0_conf"][index]],
                    f0_ok=bool(batch["f0_ok"][index]),
                    f0_bad_reason=str(batch["f0_bad_reason"][index] or ""),
                )

            repaired["f0_time"].append(reference.f0_time)
            repaired["f0_hz"].append(reference.f0_hz)
            repaired["f0_conf"].append(reference.f0_conf)
            repaired["f0_ok"].append(reference.f0_ok)
            repaired["f0_bad_reason"].append(reference.f0_bad_reason)
            repaired["f0_spectrogram"].append(
                render_f0_spectrogram(
                    audio_payload=batch["audio"][index],
                    reference=reference,
                    image_size=image_size,
                    cut_low_frequency=cut_low_frequency,
                    cut_high_frequency=cut_high_frequency,
                    wlen=wlen,
                    nfft=nfft,
                    target_fs=target_fs,
                )
            )

        return repaired

    return split_dataset.map(
        repair_batch,
        batched=True,
        batch_size=batch_size,
        features=output_features,
        desc=f"Repairing {split_name}",
    )


def write_artifacts(
    *,
    artifacts_dir: Path,
    args: argparse.Namespace,
    classification_revision: str,
    reference_revision: str,
    coverage_by_split: dict[str, dict[str, object]],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "classification_repo_id": DEFAULT_CLASSIFICATION_REPO_ID,
        "classification_config": args.classification_config,
        "classification_revision": classification_revision,
        "reference_repo_id": "dolphinteam/DolphinWhistle-Finetuning-withF0",
        "reference_config": args.reference_config,
        "reference_revision": reference_revision,
        "image_size": int(args.image_size),
        "cut_low_frequency_khz": float(args.cut_low_frequency),
        "cut_high_frequency_khz": float(args.cut_high_frequency),
        "wlen": int(args.wlen),
        "nfft": int(args.nfft),
        "target_fs": None if args.target_fs is None else int(args.target_fs),
        "coverage_by_split": coverage_by_split,
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Classification F0 Repair",
        "",
        f"Classification config: `{args.classification_config}`",
        f"Classification revision: `{classification_revision}`",
        f"Reference config: `{args.reference_config}`",
        f"Reference revision: `{reference_revision}`",
        f"Rendered image size: `{args.image_size}x{args.image_size}`",
        "",
        "Coverage summary:",
    ]
    for split_name in SUPPORTED_SPLITS:
        split_summary = coverage_by_split[split_name]
        lines.append(
            f"- `{split_name}`: rows={split_summary['rows']} "
            f"matched={split_summary['matched_rows']} missing={split_summary['missing_rows']}"
        )
    (artifacts_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_push_to_hub(
    *,
    dataset_dict: DatasetDict,
    repo_id: str | None,
    config_name: str,
    token: str | None,
    commit_message: str,
    max_shard_size: str,
) -> None:
    if not repo_id:
        return

    print(f"Pushing repaired dataset to {repo_id} (config={config_name})")
    dataset_dict.push_to_hub(
        repo_id,
        config_name=config_name,
        token=token,
        commit_message=commit_message,
        max_shard_size=max_shard_size,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    classification_cache_root = Path(args.classification_cache_root)
    reference_cache_root = Path(args.reference_cache_root)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    cache_dir = Path(args.cache_dir)

    classification_revision = resolve_revision(
        classification_cache_root,
        args.classification_revision,
    )
    reference_revision = resolve_revision(
        reference_cache_root,
        args.reference_revision,
    )

    classification_split_dir = snapshot_split_dir(
        classification_cache_root,
        args.classification_revision,
        args.classification_config,
    )
    reference_split_dir = snapshot_split_dir(
        reference_cache_root,
        args.reference_revision,
        args.reference_config,
    )

    print(f"Loading classification dataset from: {classification_split_dir}")
    classification_dataset = load_classification_dataset(
        split_dir=classification_split_dir,
        cache_dir=cache_dir,
        split_names=SUPPORTED_SPLITS,
        limit=args.limit,
    )

    print(f"Loading reference F0 lookup from: {reference_split_dir}")
    reference_lookup = load_reference_lookup(
        split_dir=reference_split_dir,
        split_names=SUPPORTED_SPLITS,
    )

    coverage_by_split = {
        split_name: split_coverage_summary(
            classification_dataset[split_name],
            reference_lookup[split_name],
        )
        for split_name in SUPPORTED_SPLITS
    }

    print("Reference coverage:")
    for split_name in SUPPORTED_SPLITS:
        summary = coverage_by_split[split_name]
        print(
            f"  {split_name:5s} rows={summary['rows']:4d} "
            f"matched={summary['matched_rows']:4d} missing={summary['missing_rows']:4d}"
        )
        if summary["missing_rows"] and not args.allow_partial_reference:
            missing_display = summary["missing_examples"]
            raise ValueError(
                f"Missing reference F0 rows for split={split_name!r}: {missing_display}"
            )

    repaired_splits: dict[str, Dataset] = {}
    for split_name in SUPPORTED_SPLITS:
        repaired_splits[split_name] = repair_split_dataset(
            split_name=split_name,
            split_dataset=classification_dataset[split_name],
            split_lookup=reference_lookup[split_name],
            image_size=args.image_size,
            cut_low_frequency=args.cut_low_frequency,
            cut_high_frequency=args.cut_high_frequency,
            wlen=args.wlen,
            nfft=args.nfft,
            target_fs=args.target_fs,
            batch_size=args.batch_size,
            allow_partial_reference=args.allow_partial_reference,
        )

    repaired_dataset = DatasetDict(repaired_splits)

    ensure_output_dir_is_empty(output_dir)
    print(f"Saving repaired dataset to disk: {output_dir}")
    repaired_dataset.save_to_disk(str(output_dir))
    print("Local save completed.")

    write_artifacts(
        artifacts_dir=artifacts_dir,
        args=args,
        classification_revision=classification_revision,
        reference_revision=reference_revision,
        coverage_by_split=coverage_by_split,
    )
    print(f"Artifacts written to: {artifacts_dir}")

    maybe_push_to_hub(
        dataset_dict=repaired_dataset,
        repo_id=args.push_repo_id,
        config_name=args.classification_config,
        token=args.token,
        commit_message=args.commit_message,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
