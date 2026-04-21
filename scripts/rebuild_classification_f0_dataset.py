#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchcrepe
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Image as HFImage,
    Sequence,
    Value,
    concatenate_datasets,
)

from repair_classification_f0_dataset import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CLASSIFICATION_CACHE_ROOT,
    DEFAULT_CLASSIFICATION_REPO_ID,
    DEFAULT_CUT_HIGH_FREQUENCY,
    DEFAULT_CUT_LOW_FREQUENCY,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_SHARD_SIZE,
    DEFAULT_NFFT,
    DEFAULT_WLEN,
    SUPPORTED_SPLITS,
    ReferenceF0,
    ensure_output_dir_is_empty,
    load_classification_dataset,
    maybe_push_to_hub,
    render_f0_spectrogram,
    resolve_revision,
    resample_if_needed,
    snapshot_split_dir,
)


DEFAULT_OUTPUT_DIR = Path("cnn_dataset/datasets/classification_f0_rebuilt")
DEFAULT_ARTIFACTS_DIR = Path("cnn_dataset/artifacts_classification_f0_rebuilt")
DEFAULT_CLASSIFICATION_DATASETS_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "datasets"
    / "dolphinteam___dophin_whistle-classification-finetuning"
)
DEFAULT_OLD_PIPELINE_ROOT = (
    Path.home()
    / "Documents"
    / "Old"
    / "Dolph2Vec"
    / "users"
    / "pablo"
    / "bioacoustic_F0_estimation"
)
DEFAULT_MODEL_PATH = (
    DEFAULT_OLD_PIPELINE_ROOT
    / "paper_experiments"
    / "crepe_weights"
    / "model_only-0_bottlenose_dolphins.pth"
)
DEFAULT_COMPRESS = 20.0
DEFAULT_STEP = 0.005
DEFAULT_DECODER = "weighted_argmax"
DEFAULT_CONFIDENCE_THRESHOLD = 0.05
DEFAULT_MIN_VOICED_RATIO = 0.05
DEFAULT_CLIP_BATCH_SIZE = 16
DEFAULT_GPU_FRAME_BATCH_SIZE = 512


@dataclass(frozen=True)
class PipelineResult:
    f0_time: list[float]
    f0_hz: list[float]
    f0_conf: list[float]
    f0_ok: bool
    f0_bad_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild F0 tracks for dolphinteam/DophinWhistle-Classification-Finetuning "
            "with the validated dolphin CREPE pipeline and regenerate f0_spectrogram."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--classification-cache-root",
        default=str(DEFAULT_CLASSIFICATION_CACHE_ROOT),
        help="Local HF Hub cache root for the classification dataset repo.",
    )
    parser.add_argument(
        "--classification-datasets-cache-root",
        default=str(DEFAULT_CLASSIFICATION_DATASETS_CACHE_ROOT),
        help="Local HF datasets cache root for configs not materialized in hub snapshots.",
    )
    parser.add_argument(
        "--classification-source",
        choices=("auto", "hub-snapshot", "datasets-cache"),
        default="auto",
        help="Where to load the classification config from.",
    )
    parser.add_argument(
        "--classification-revision",
        default="main",
        help="Classification revision to read from the local HF Hub cache.",
    )
    parser.add_argument(
        "--classification-config",
        default="default",
        help="Classification config/subdirectory to rebuild (default, all, unbalanced).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Where to save the rebuilt DatasetDict with save_to_disk().",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=str(DEFAULT_ARTIFACTS_DIR),
        help="Where to write JSON/README rebuild artifacts.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Writable cache dir used while loading local parquet shards.",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the dolphin-specific CREPE weights.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device for CREPE.",
    )
    parser.add_argument(
        "--decoder",
        choices=("argmax", "weighted_argmax", "viterbi"),
        default=DEFAULT_DECODER,
        help="Decoder used to postprocess CREPE predictions.",
    )
    parser.add_argument(
        "--compress",
        type=float,
        default=DEFAULT_COMPRESS,
        help="Compression factor used to shift dolphin whistles into CREPE's range.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=DEFAULT_STEP,
        help="Frame step in seconds.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold used for f0_ok classification.",
    )
    parser.add_argument(
        "--min-voiced-ratio",
        type=float,
        default=DEFAULT_MIN_VOICED_RATIO,
        help="Minimum voiced-frame ratio required to mark f0_ok=True.",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=DEFAULT_CLIP_BATCH_SIZE,
        help="Number of clips processed together before model inference.",
    )
    parser.add_argument(
        "--gpu-frame-batch-size",
        type=int,
        default=DEFAULT_GPU_FRAME_BATCH_SIZE,
        help="Maximum number of CREPE frames inferred at once.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Dataset row batch size used when materializing the rebuilt dataset.",
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
        help="Optional target sample rate for spectrogram rendering.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional smoke-test limit applied independently to each split.",
    )
    parser.add_argument(
        "--push-repo-id",
        default=None,
        help="Optional Hugging Face dataset repo id to push the rebuilt config to.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token used for push_to_hub().",
    )
    parser.add_argument(
        "--commit-message",
        default="Rebuild F0 tracks and regenerate f0_spectrogram",
        help="Commit message used when pushing to the Hub.",
    )
    parser.add_argument(
        "--max-shard-size",
        default=DEFAULT_MAX_SHARD_SIZE,
        help="Shard size passed to push_to_hub().",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.compress <= 0:
        raise ValueError("--compress must be strictly positive.")
    if args.step <= 0:
        raise ValueError("--step must be strictly positive.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1].")
    if not 0.0 <= args.min_voiced_ratio <= 1.0:
        raise ValueError("--min-voiced-ratio must be in [0, 1].")
    if args.clip_batch_size <= 0:
        raise ValueError("--clip-batch-size must be strictly positive.")
    if args.gpu_frame_batch_size <= 0:
        raise ValueError("--gpu-frame-batch-size must be strictly positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be strictly positive.")
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
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be strictly positive when provided.")


def resolve_device(device_name: str) -> str:
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def datasets_cache_config_dir(cache_root: Path, config_name: str) -> Path:
    config_root = cache_root / config_name / "0.0.0"
    if not config_root.exists():
        raise FileNotFoundError(
            f"Missing datasets cache directory for config={config_name!r}: {config_root}"
        )

    candidate_dirs = [path for path in config_root.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No materialized dataset-cache revisions found for config={config_name!r} under {config_root}"
        )
    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)


def load_classification_dataset_from_arrow_cache(
    *,
    config_dir: Path,
    split_names: tuple[str, ...],
    limit: int | None,
) -> DatasetDict:
    prepared_splits: dict[str, Dataset] = {}
    for split_name in split_names:
        split_files = sorted(
            config_dir.glob(f"dophin_whistle-classification-finetuning-{split_name}*.arrow")
        )
        if not split_files:
            raise FileNotFoundError(
                f"No arrow shards found for split={split_name!r} under {config_dir}"
            )

        split_datasets = [Dataset.from_file(str(path)) for path in split_files]
        split_dataset = (
            split_datasets[0]
            if len(split_datasets) == 1
            else concatenate_datasets(split_datasets)
        )
        split_dataset = split_dataset.cast_column("audio", Audio(decode=False))
        if "f0_spectrogram" in split_dataset.column_names:
            split_dataset = split_dataset.remove_columns(["f0_spectrogram"])
        if limit is not None:
            split_dataset = split_dataset.select(range(min(limit, split_dataset.num_rows)))
        prepared_splits[split_name] = split_dataset

    return DatasetDict(prepared_splits)


def load_classification_dataset_for_config(
    *,
    classification_source: str,
    classification_cache_root: Path,
    classification_datasets_cache_root: Path,
    classification_revision: str,
    classification_config: str,
    cache_dir: Path,
    split_names: tuple[str, ...],
    limit: int | None,
) -> tuple[DatasetDict, str]:
    if classification_source in ("auto", "hub-snapshot"):
        try:
            split_dir = snapshot_split_dir(
                classification_cache_root,
                classification_revision,
                classification_config,
            )
            dataset_dict = load_classification_dataset(
                split_dir=split_dir,
                cache_dir=cache_dir,
                split_names=split_names,
                limit=limit,
            )
            return dataset_dict, f"hub snapshot: {split_dir}"
        except FileNotFoundError:
            if classification_source == "hub-snapshot":
                raise

    cache_config_dir = datasets_cache_config_dir(
        classification_datasets_cache_root,
        classification_config,
    )
    dataset_dict = load_classification_dataset_from_arrow_cache(
        config_dir=cache_config_dir,
        split_names=split_names,
        limit=limit,
    )
    return dataset_dict, f"datasets cache: {cache_config_dir}"


def load_crepe_model(model_path: Path, device: str) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing CREPE weights: {model_path}")

    model = torchcrepe.Crepe("full").eval().to(device)
    try:
        state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    return model


def audio_payload_hash(audio_payload: object) -> str:
    if isinstance(audio_payload, dict):
        audio_bytes = audio_payload.get("bytes")
        audio_path = audio_payload.get("path")
        if audio_bytes is not None:
            return hashlib.md5(audio_bytes).hexdigest()
        if audio_path:
            return hashlib.md5(str(audio_path).encode("utf-8")).hexdigest()
    raise TypeError(f"Unsupported audio payload type for hashing: {type(audio_payload)!r}")


def classify_f0(
    *,
    f0_time: np.ndarray,
    f0_conf: np.ndarray,
    confidence_threshold: float,
    min_voiced_ratio: float,
) -> tuple[bool, str]:
    if len(f0_time) == 0:
        return False, "too_short"

    voiced_ratio = float(np.mean(f0_conf > confidence_threshold))
    if voiced_ratio < min_voiced_ratio:
        return False, "low_confidence"

    return True, ""


def batch_estimate_f0(
    *,
    audio_payloads: list[object],
    model: torch.nn.Module,
    device: str,
    decoder_name: str,
    compress: float,
    step: float,
    gpu_frame_batch_size: int,
    confidence_threshold: float,
    min_voiced_ratio: float,
) -> list[PipelineResult]:
    target_fs = int(torchcrepe.SAMPLE_RATE * compress)
    hop_length = int(step * compress * torchcrepe.SAMPLE_RATE)
    decoder = torchcrepe.decode.__dict__[decoder_name]

    frame_batches: list[torch.Tensor] = []
    clip_frame_counts: list[int] = []

    for audio_payload in audio_payloads:
        from repair_classification_f0_dataset import decode_audio_payload

        audio, fs = decode_audio_payload(audio_payload)
        audio_resampled, _ = resample_if_needed(audio, fs, target_fs)

        if len(audio_resampled) < hop_length:
            clip_frame_counts.append(0)
            continue

        generator = torchcrepe.core.preprocess(
            torch.tensor(audio_resampled).unsqueeze(0),
            torchcrepe.SAMPLE_RATE,
            hop_length=hop_length,
            batch_size=len(audio_resampled),
            device="cpu",
        )
        clip_frames = [frames for frames in generator]
        if not clip_frames:
            clip_frame_counts.append(0)
            continue

        clip_tensor = torch.cat(clip_frames, dim=0)
        clip_frame_counts.append(len(clip_tensor))
        frame_batches.append(clip_tensor)

    if not frame_batches:
        return [
            PipelineResult([], [], [], False, "too_short") for _ in audio_payloads
        ]

    all_frames = torch.cat(frame_batches, dim=0)
    pred_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(all_frames), gpu_frame_batch_size):
            frame_batch = all_frames[start : start + gpu_frame_batch_size].to(device)
            pred_batches.append(model(frame_batch).cpu())

    all_preds = torch.cat(pred_batches, dim=0)

    results: list[PipelineResult] = []
    frame_offset = 0
    for n_frames in clip_frame_counts:
        if n_frames == 0:
            results.append(PipelineResult([], [], [], False, "too_short"))
            continue

        clip_preds = all_preds[frame_offset : frame_offset + n_frames]
        frame_offset += n_frames

        preds_for_decode = clip_preds.T.unsqueeze(0)
        f0 = (torchcrepe.core.postprocess(preds_for_decode, decoder=decoder) * compress).squeeze()
        conf = clip_preds.max(dim=1).values.squeeze()

        f0_np = np.atleast_1d(f0.numpy()).astype(np.float32)
        conf_np = np.atleast_1d(conf.numpy()).astype(np.float32)
        time_np = (np.arange(len(f0_np), dtype=np.float32) * hop_length) / float(target_fs)

        length = min(len(time_np), len(f0_np), len(conf_np))
        time_np = time_np[:length]
        f0_np = f0_np[:length]
        conf_np = conf_np[:length]

        f0_ok, bad_reason = classify_f0(
            f0_time=time_np,
            f0_conf=conf_np,
            confidence_threshold=confidence_threshold,
            min_voiced_ratio=min_voiced_ratio,
        )
        results.append(
            PipelineResult(
                f0_time=time_np.astype(np.float32).tolist(),
                f0_hz=f0_np.astype(np.float32).tolist(),
                f0_conf=conf_np.astype(np.float32).tolist(),
                f0_ok=f0_ok,
                f0_bad_reason=bad_reason,
            )
        )

    return results


def rebuild_split_dataset(
    *,
    split_name: str,
    split_dataset: Dataset,
    model: torch.nn.Module,
    device: str,
    args: argparse.Namespace,
    map_cache_dir: Path,
) -> tuple[Dataset, dict[str, object]]:
    output_features = Features(dict(split_dataset.features))
    output_features["f0_time"] = Sequence(Value("float32"))
    output_features["f0_hz"] = Sequence(Value("float32"))
    output_features["f0_conf"] = Sequence(Value("float32"))
    output_features["f0_ok"] = Value("bool")
    output_features["f0_bad_reason"] = Value("string")
    output_features["f0_spectrogram"] = HFImage()

    cache: dict[str, PipelineResult] = {}

    def rebuild_batch(batch: dict[str, list[object]]) -> dict[str, list[object]]:
        audio_payloads = list(batch["audio"])
        batch_results: list[PipelineResult | None] = [None] * len(audio_payloads)
        missing_indices: list[int] = []

        for index, audio_payload in enumerate(audio_payloads):
            key = audio_payload_hash(audio_payload)
            cached = cache.get(key)
            if cached is not None:
                batch_results[index] = cached
            else:
                missing_indices.append(index)

        if missing_indices:
            for start in range(0, len(missing_indices), args.clip_batch_size):
                chunk_indices = missing_indices[start : start + args.clip_batch_size]
                computed = batch_estimate_f0(
                    audio_payloads=[audio_payloads[index] for index in chunk_indices],
                    model=model,
                    device=device,
                    decoder_name=args.decoder,
                    compress=args.compress,
                    step=args.step,
                    gpu_frame_batch_size=args.gpu_frame_batch_size,
                    confidence_threshold=args.confidence_threshold,
                    min_voiced_ratio=args.min_voiced_ratio,
                )
                for index, result in zip(chunk_indices, computed):
                    key = audio_payload_hash(audio_payloads[index])
                    cache[key] = result
                    batch_results[index] = result

        rebuilt = {
            "f0_time": [],
            "f0_hz": [],
            "f0_conf": [],
            "f0_ok": [],
            "f0_bad_reason": [],
            "f0_spectrogram": [],
        }

        for index, maybe_result in enumerate(batch_results):
            if maybe_result is None:
                raise RuntimeError("Missing F0 result for batch row.")

            result = maybe_result
            rebuilt["f0_time"].append(result.f0_time)
            rebuilt["f0_hz"].append(result.f0_hz)
            rebuilt["f0_conf"].append(result.f0_conf)
            rebuilt["f0_ok"].append(result.f0_ok)
            rebuilt["f0_bad_reason"].append(result.f0_bad_reason)
            rebuilt["f0_spectrogram"].append(
                render_f0_spectrogram(
                    audio_payload=batch["audio"][index],
                    reference=ReferenceF0(
                        f0_time=result.f0_time,
                        f0_hz=result.f0_hz,
                        f0_conf=result.f0_conf,
                        f0_ok=result.f0_ok,
                        f0_bad_reason=result.f0_bad_reason,
                    ),
                    image_size=args.image_size,
                    cut_low_frequency=args.cut_low_frequency,
                    cut_high_frequency=args.cut_high_frequency,
                    wlen=args.wlen,
                    nfft=args.nfft,
                    target_fs=args.target_fs,
                )
            )

        return rebuilt

    map_cache_dir.mkdir(parents=True, exist_ok=True)
    rebuilt_split = split_dataset.map(
        rebuild_batch,
        batched=True,
        batch_size=args.batch_size,
        features=output_features,
        load_from_cache_file=False,
        cache_file_name=str(map_cache_dir / f"{split_name}.arrow"),
        desc=f"Rebuilding {split_name}",
    )

    f0_ok_values = rebuilt_split["f0_ok"]
    reason_values = rebuilt_split["f0_bad_reason"]
    stats = {
        "rows": int(rebuilt_split.num_rows),
        "f0_ok_rows": int(sum(bool(value) for value in f0_ok_values)),
        "f0_bad_rows": int(sum(not bool(value) for value in f0_ok_values)),
        "bad_reason_counts": {},
    }
    for reason in reason_values:
        reason_key = str(reason or "")
        stats["bad_reason_counts"][reason_key] = stats["bad_reason_counts"].get(reason_key, 0) + 1
    return rebuilt_split, stats


def write_artifacts(
    *,
    artifacts_dir: Path,
    args: argparse.Namespace,
    classification_revision: str,
    split_stats: dict[str, dict[str, object]],
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "classification_repo_id": DEFAULT_CLASSIFICATION_REPO_ID,
        "classification_config": args.classification_config,
        "classification_revision": classification_revision,
        "pipeline": {
            "model_path": str(Path(args.model_path).resolve()),
            "device": resolve_device(args.device),
            "decoder": args.decoder,
            "compress": float(args.compress),
            "step_seconds": float(args.step),
            "confidence_threshold": float(args.confidence_threshold),
            "min_voiced_ratio": float(args.min_voiced_ratio),
        },
        "rendering": {
            "image_size": int(args.image_size),
            "cut_low_frequency_khz": float(args.cut_low_frequency),
            "cut_high_frequency_khz": float(args.cut_high_frequency),
            "wlen": int(args.wlen),
            "nfft": int(args.nfft),
            "target_fs": None if args.target_fs is None else int(args.target_fs),
        },
        "split_stats": split_stats,
    }
    (artifacts_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Classification F0 Rebuild",
        "",
        f"Classification config: `{args.classification_config}`",
        f"Classification revision: `{classification_revision}`",
        f"Model path: `{Path(args.model_path).resolve()}`",
        f"Pipeline: `compress={args.compress}` `step={args.step}` `decoder={args.decoder}` `confidence_threshold={args.confidence_threshold}`",
        f"Rendered image size: `{args.image_size}x{args.image_size}`",
        "",
        "Split summary:",
    ]
    for split_name in SUPPORTED_SPLITS:
        split_summary = split_stats[split_name]
        lines.append(
            f"- `{split_name}`: rows={split_summary['rows']} "
            f"f0_ok={split_summary['f0_ok_rows']} f0_bad={split_summary['f0_bad_rows']}"
        )
    (artifacts_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)

    classification_cache_root = Path(args.classification_cache_root)
    classification_datasets_cache_root = Path(args.classification_datasets_cache_root)
    output_dir = Path(args.output_dir)
    artifacts_dir = Path(args.artifacts_dir)
    cache_dir = Path(args.cache_dir)
    model_path = Path(args.model_path)
    map_cache_dir = cache_dir / "classification_rebuild_map_cache" / args.classification_config

    classification_revision = resolve_revision(
        classification_cache_root,
        args.classification_revision,
    )
    ensure_output_dir_is_empty(output_dir)
    ensure_output_dir_is_empty(artifacts_dir)

    dataset_dict, source_description = load_classification_dataset_for_config(
        classification_source=args.classification_source,
        classification_cache_root=classification_cache_root,
        classification_datasets_cache_root=classification_datasets_cache_root,
        classification_revision=args.classification_revision,
        classification_config=args.classification_config,
        cache_dir=cache_dir,
        split_names=SUPPORTED_SPLITS,
        limit=args.limit,
    )

    device = resolve_device(args.device)
    print(f"Loading classification dataset from {source_description}")
    print(
        "Using dolphin F0 pipeline with "
        f"compress={args.compress}, step={args.step}, decoder={args.decoder}, "
        f"confidence_threshold={args.confidence_threshold}"
    )
    model = load_crepe_model(model_path, device)

    rebuilt_splits: dict[str, Dataset] = {}
    split_stats: dict[str, dict[str, object]] = {}
    for split_name in SUPPORTED_SPLITS:
        rebuilt_split, stats = rebuild_split_dataset(
            split_name=split_name,
            split_dataset=dataset_dict[split_name],
            model=model,
            device=device,
            args=args,
            map_cache_dir=map_cache_dir,
        )
        rebuilt_splits[split_name] = rebuilt_split
        split_stats[split_name] = stats

    rebuilt_dataset = DatasetDict(rebuilt_splits)
    rebuilt_dataset.save_to_disk(str(output_dir))

    write_artifacts(
        artifacts_dir=artifacts_dir,
        args=args,
        classification_revision=classification_revision,
        split_stats=split_stats,
    )

    maybe_push_to_hub(
        dataset_dict=rebuilt_dataset,
        repo_id=args.push_repo_id,
        config_name=args.classification_config,
        token=args.token,
        commit_message=args.commit_message,
        max_shard_size=args.max_shard_size,
    )


if __name__ == "__main__":
    main()
