#!/usr/bin/env python3
"""Upload the local audio datasets used for whistle CNN training to Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class DatasetUploadSpec:
    dataset_dir: str
    repo_id: str
    artifact_dir: str | None = None
    title: str | None = None
    description: str | None = None


DEFAULT_SPECS: tuple[DatasetUploadSpec, ...] = (
    DatasetUploadSpec(
        dataset_dir="cnn_dataset/datasets/cnn_trainval_manual_test_audio_v3",
        repo_id="dolphinteam/DolphinWhistle-CNN-Dataset",
        artifact_dir="cnn_dataset/artifacts_cnn_trainval_manual_test_audio_v3",
        title="DolphinWhistle CNN Dataset",
        description=(
            "Final session-disjoint audio dataset used for whistle CNN training, "
            "with train/validation splits from the non-2019/2020 pool and a manual "
            "2019-2020 test split built from the full classification 'all' config."
        ),
    ),
)

DEFAULT_SPECTROGRAM_CACHE_ROOT = REPO_ROOT / "cnn_dataset" / "spectrogram_cache"
DEFAULT_FALLBACK_CACHE_DATASETS: tuple[str, ...] = (
    "cnn_dataset/datasets/cnn_trainval_manual_test_audio",
    "cnn_dataset/datasets/no_2019_2020_with_session_noise_split_audio",
)
DEFAULT_PUBLIC_CACHE_ROOT = REPO_ROOT / "cnn_dataset" / "prepared_hf_public_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload local Hugging Face datasets saved with load_from_disk to the Hub.",
    )
    parser.add_argument(
        "--only-final",
        action="store_true",
        help="Retained for compatibility. The script uploads only the final merged dataset.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create dataset repos as private if they do not already exist.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, uses the local HF login state.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Shard size passed to datasets.push_to_hub.",
    )
    parser.add_argument(
        "--embed-external-files",
        action="store_true",
        default=True,
        help="Embed external files (including audio) in uploaded parquet shards.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload local whistle CNN datasets",
        help="Commit message used for dataset uploads.",
    )
    parser.add_argument(
        "--prepared-cache-dir",
        default=None,
        help=(
            "Optional directory used to persist the public HF-ready dataset locally. "
            "If omitted, a repo-specific cache is created under cnn_dataset/prepared_hf_public_datasets."
        ),
    )
    parser.add_argument(
        "--force-rebuild-public-cache",
        action="store_true",
        help="Ignore any existing local prepared public dataset cache and rebuild it.",
    )
    return parser.parse_args()


def require_dependencies() -> tuple[object, object]:
    try:
        import datasets.arrow_writer as arrow_writer
        import datasets.io.parquet as parquet_io
        from datasets import load_from_disk
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "Missing dependencies. Install `datasets` and `huggingface_hub` before "
            "running this script."
        ) from exc

    # `datasets==3.6.0` on this machine can fail with a late import inside
    # `push_to_hub()` even though `get_writer_batch_size` exists. Preloading
    # the parquet module keeps the upload path stable.
    if not hasattr(arrow_writer, "get_writer_batch_size"):
        raise SystemExit(
            "Your local `datasets` install is missing `get_writer_batch_size`; "
            "please update the package before uploading."
        )
    _ = parquet_io.ParquetDatasetWriter
    return load_from_disk, HfApi


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_dataset(dataset_obj: object) -> tuple[list[str], dict[str, int], str]:
    split_names = list(dataset_obj.keys())
    rows_by_split = {split: int(dataset_obj[split].num_rows) for split in split_names}
    feature_names = list(dataset_obj[split_names[0]].features.keys()) if split_names else []
    feature_text = ", ".join(feature_names) if feature_names else "unknown"
    return split_names, rows_by_split, feature_text


def resolve_cache_namespace(dataset_dir: Path, cache_root: Path) -> Path | None:
    try:
        normalized_dataset_dir = dataset_dir.resolve().relative_to(REPO_ROOT.resolve())
    except Exception:
        normalized_dataset_dir = dataset_dir

    repo_slug = str(normalized_dataset_dir).replace("/", "__")
    base_dir = cache_root / repo_slug
    if not base_dir.exists():
        return None

    candidates = sorted(path for path in base_dir.iterdir() if path.is_dir())
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for candidate in candidates:
        metadata_path = candidate / "cache_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("repo_id") in {str(dataset_dir), str(normalized_dataset_dir)}:
            return candidate

    return candidates[0]


def load_cached_spectrogram(cache_namespaces: list[Path], split_name: str, index: int) -> np.ndarray:
    checked_paths: list[Path] = []
    for cache_namespace in cache_namespaces:
        cache_path = cache_namespace / split_name / f"{int(index):06d}.npy"
        checked_paths.append(cache_path)
        if not cache_path.exists():
            continue
        gray = np.load(cache_path, allow_pickle=False)
        if gray.ndim != 2:
            raise ValueError(f"Unexpected cached spectrogram shape at {cache_path}: {gray.shape}")
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    checked_display = "\n".join(f"  - {path}" for path in checked_paths)
    raise FileNotFoundError(
        f"Missing cached spectrogram for split={split_name!r} index={index}. Checked:\n"
        f"{checked_display}"
    )


def load_cached_spectrogram_batch(
    cache_namespaces: list[Path],
    split_name: str,
    indices: list[int],
) -> list[np.ndarray]:
    return [
        load_cached_spectrogram(cache_namespaces, split_name, int(index))
        for index in indices
    ]


def prepare_public_dataset(dataset_obj: object, dataset_dir: Path) -> object:
    from datasets import ClassLabel, DatasetDict, Features, Image as HFImage

    split_names = list(dataset_obj.keys())
    if not split_names:
        return dataset_obj

    reference_columns = list(dataset_obj[split_names[0]].column_names)
    wanted_columns = ["audio", "label", "file_name", "recording", "onset", "offset"]
    public_columns = [name for name in wanted_columns if name in reference_columns]
    cache_namespaces: list[Path] = []
    primary_namespace = resolve_cache_namespace(
        dataset_dir=dataset_dir,
        cache_root=DEFAULT_SPECTROGRAM_CACHE_ROOT,
    )
    if primary_namespace is not None:
        cache_namespaces.append(primary_namespace)
    for fallback_dataset in DEFAULT_FALLBACK_CACHE_DATASETS:
        fallback_namespace = resolve_cache_namespace(
            dataset_dir=REPO_ROOT / fallback_dataset,
            cache_root=DEFAULT_SPECTROGRAM_CACHE_ROOT,
        )
        if fallback_namespace is not None and fallback_namespace not in cache_namespaces:
            cache_namespaces.append(fallback_namespace)

    cleaned_splits = {}
    for split_name in split_names:
        split_dataset = dataset_obj[split_name].select_columns(public_columns)
        if cache_namespaces:
            split_features = Features(dict(split_dataset.features))
            split_features["spectrogram"] = HFImage()
            split_dataset = split_dataset.map(
                lambda _batch, indices, split_name=split_name: {
                    "spectrogram": load_cached_spectrogram_batch(
                        cache_namespaces,
                        split_name,
                        indices,
                    )
                },
                batched=True,
                with_indices=True,
                batch_size=256,
                desc=f"Loading cached spectrograms for {split_name}",
                features=split_features,
            )
            ordered_columns = [
                "audio",
                "spectrogram",
                "label",
                "file_name",
                "recording",
                "onset",
                "offset",
            ]
            split_dataset = split_dataset.select_columns(
                [name for name in ordered_columns if name in split_dataset.column_names]
            )
        if "label" in public_columns:
            split_dataset = split_dataset.cast_column(
                "label",
                ClassLabel(names=["noise", "whistle"]),
            )
        cleaned_splits[split_name] = split_dataset

    dataset_obj = DatasetDict(cleaned_splits)

    return dataset_obj


def build_readme(
    spec: DatasetUploadSpec,
    dataset_dir: Path,
    summary: dict | None,
    split_names: list[str],
    rows_by_split: dict[str, int],
    feature_text: str,
) -> str:
    lines = [
        "---",
        "license: mit",
        "task_categories:",
        "- audio-classification",
        "tags:",
        "- dolphin",
        "- bioacoustics",
        "- whistle-detection",
        "- audio",
        "- spectrogram",
        "---",
        "",
        f"# {spec.title or spec.repo_id}",
        "",
        spec.description or "Local audio dataset used for dolphin whistle CNN experiments.",
        "",
        "## Source",
        "",
        f"- Local dataset path: `{dataset_dir}`",
        f"- Splits: `{', '.join(split_names)}`",
        f"- Features: `{feature_text}`",
        "",
        "## Rows By Split",
        "",
    ]

    for split_name in split_names:
        lines.append(f"- `{split_name}`: `{rows_by_split[split_name]}` rows")

    if summary:
        lines.extend(
            [
                "",
                "## Local Build Summary",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
            ]
        )

    return "\n".join(lines) + "\n"


def default_public_cache_dir(spec: DatasetUploadSpec) -> Path:
    repo_slug = spec.repo_id.replace("/", "__")
    return DEFAULT_PUBLIC_CACHE_ROOT / repo_slug


def build_public_cache_metadata(
    *,
    spec: DatasetUploadSpec,
    dataset_dir: Path,
    artifact_dir: Path | None,
) -> dict:
    artifact_summary = read_json(artifact_dir / "summary.json") if artifact_dir else None
    return {
        "public_format_version": 1,
        "repo_id": spec.repo_id,
        "dataset_dir": str(dataset_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()) if artifact_dir else None,
        "artifact_summary": artifact_summary,
    }


def read_public_cache_metadata(cache_dir: Path) -> dict | None:
    return read_json(cache_dir / "_public_cache_metadata.json")


def write_public_cache_metadata(cache_dir: Path, payload: dict) -> None:
    metadata_path = cache_dir / "_public_cache_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_public_dataset_cache(
    *,
    load_from_disk: object,
    dataset_dir: Path,
    prepared_cache_dir: Path,
) -> object:
    dataset_obj = load_from_disk(str(dataset_dir))
    dataset_obj = prepare_public_dataset(dataset_obj, dataset_dir=dataset_dir)

    prepared_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{prepared_cache_dir.name}_build_",
        dir=str(prepared_cache_dir.parent),
    ) as tmpdir:
        staged_dir = Path(tmpdir) / "dataset"
        dataset_obj.save_to_disk(str(staged_dir))
        if prepared_cache_dir.exists():
            shutil.rmtree(prepared_cache_dir)
        staged_dir.replace(prepared_cache_dir)

    return dataset_obj


def load_or_prepare_public_dataset(
    *,
    load_from_disk: object,
    spec: DatasetUploadSpec,
    dataset_dir: Path,
    artifact_dir: Path | None,
    prepared_cache_dir: Path,
    force_rebuild: bool,
) -> tuple[object, dict | None]:
    expected_metadata = build_public_cache_metadata(
        spec=spec,
        dataset_dir=dataset_dir,
        artifact_dir=artifact_dir,
    )
    cached_metadata = read_public_cache_metadata(prepared_cache_dir)
    if (
        not force_rebuild
        and prepared_cache_dir.exists()
        and cached_metadata == expected_metadata
    ):
        print(f"Reusing prepared public dataset cache: {prepared_cache_dir}")
        return load_from_disk(str(prepared_cache_dir)), expected_metadata["artifact_summary"]

    if force_rebuild and prepared_cache_dir.exists():
        print(f"Rebuilding prepared public dataset cache: {prepared_cache_dir}")
    else:
        print(f"Building prepared public dataset cache: {prepared_cache_dir}")
    dataset_obj = build_public_dataset_cache(
        load_from_disk=load_from_disk,
        dataset_dir=dataset_dir,
        prepared_cache_dir=prepared_cache_dir,
    )
    write_public_cache_metadata(prepared_cache_dir, expected_metadata)
    return dataset_obj, expected_metadata["artifact_summary"]


def upload_dataset(
    *,
    load_from_disk: object,
    HfApi: object,
    spec: DatasetUploadSpec,
    private: bool,
    token: str | None,
    max_shard_size: str,
    embed_external_files: bool,
    commit_message: str,
    prepared_cache_dir: Path,
    force_rebuild_public_cache: bool,
) -> None:
    dataset_dir = Path(spec.dataset_dir)
    artifact_dir = Path(spec.artifact_dir) if spec.artifact_dir else None
    if not dataset_dir.exists():
        raise SystemExit(f"Missing dataset directory: {dataset_dir}")

    dataset_obj, summary = load_or_prepare_public_dataset(
        load_from_disk=load_from_disk,
        spec=spec,
        dataset_dir=dataset_dir,
        artifact_dir=artifact_dir,
        prepared_cache_dir=prepared_cache_dir,
        force_rebuild=force_rebuild_public_cache,
    )
    split_names, rows_by_split, feature_text = summarize_dataset(dataset_obj)

    api = HfApi(token=token)
    api.create_repo(
        repo_id=spec.repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    dataset_obj.push_to_hub(
        spec.repo_id,
        private=private,
        max_shard_size=max_shard_size,
        embed_external_files=embed_external_files,
        token=token,
        commit_message=commit_message,
    )

    readme_content = build_readme(
        spec=spec,
        dataset_dir=dataset_dir,
        summary=summary,
        split_names=split_names,
        rows_by_split=rows_by_split,
        feature_text=feature_text,
    )
    with tempfile.TemporaryDirectory(prefix="hf_dataset_card_") as tmpdir:
        readme_path = Path(tmpdir) / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        api.upload_file(
            repo_id=spec.repo_id,
            repo_type="dataset",
            path_in_repo="README.md",
            path_or_fileobj=str(readme_path),
            commit_message=f"{commit_message} (README)",
        )

    if artifact_dir:
        for extra_name in ("summary.json", "session_assignments.csv"):
            extra_path = artifact_dir / extra_name
            if extra_path.exists():
                api.upload_file(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    path_in_repo=f"artifacts/{extra_name}",
                    path_or_fileobj=str(extra_path),
                    commit_message=f"{commit_message} ({extra_name})",
                )

    print(f"Uploaded dataset to https://huggingface.co/datasets/{spec.repo_id}")


def main() -> None:
    args = parse_args()
    load_from_disk, HfApi = require_dependencies()

    specs = DEFAULT_SPECS
    for spec in specs:
        prepared_cache_dir = (
            Path(args.prepared_cache_dir)
            if args.prepared_cache_dir
            else default_public_cache_dir(spec)
        )
        upload_dataset(
            load_from_disk=load_from_disk,
            HfApi=HfApi,
            spec=spec,
            private=args.private,
            token=args.token,
            max_shard_size=args.max_shard_size,
            embed_external_files=args.embed_external_files,
            commit_message=args.commit_message,
            prepared_cache_dir=prepared_cache_dir,
            force_rebuild_public_cache=args.force_rebuild_public_cache,
        )


if __name__ == "__main__":
    main()
