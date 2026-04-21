from __future__ import annotations

import inspect
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Audio, ClassLabel, Dataset, DatasetDict, Features, Value

DEFAULT_SOURCE_CACHE_ROOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--dolphinteam--DophinWhistle-Detection-Finetuning-old"
)
DEFAULT_OUTPUT_DIR = Path(
    "openwhistle_detection_finetuning/output/OpenWhistle-1.0-Detection-Finetuning"
)
DEFAULT_ARTIFACTS_DIR = Path(
    "openwhistle_detection_finetuning/artifacts/OpenWhistle-1.0-Detection-Finetuning"
)
DEFAULT_REPO_ID = "dolphinteam/OpenWhistle-1.0-Detection-Finetuning"
DEFAULT_SOURCE_REPO_ID = "dolphinteam/DophinWhistle-Detection-Finetuning_OLD"
DEFAULT_SOURCE_REVISION = "main"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
DEFAULT_SPLIT_POLICY = "stratified"
LABEL_NAMES = ("noise", "whistle")
REQUIRED_SOURCE_COLUMNS = ("wav_path", "label", "name")


def make_audio_feature(*, sampling_rate=None, decode=None, mono=None) -> Audio:
    supported = inspect.signature(Audio.__init__).parameters
    kwargs = {}
    if sampling_rate is not None and "sampling_rate" in supported:
        kwargs["sampling_rate"] = sampling_rate
    if decode is not None and "decode" in supported:
        kwargs["decode"] = decode
    if mono is not None and "mono" in supported:
        kwargs["mono"] = mono
    return Audio(**kwargs)


OUTPUT_FEATURES = Features(
    {
        "audio": make_audio_feature(decode=False),
        "label": ClassLabel(names=list(LABEL_NAMES)),
        "name": Value("string"),
        "source_label": Value("string"),
    }
)


def ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_source_snapshot_dir(
    *,
    source_cache_root: Path,
    source_snapshot_dir: Path | None,
    revision: str,
) -> Path:
    if source_snapshot_dir is not None:
        if not source_snapshot_dir.is_dir():
            raise FileNotFoundError(f"Source snapshot dir not found: {source_snapshot_dir}")
        return source_snapshot_dir.resolve()

    cache_root = source_cache_root.resolve()
    revision_name = revision
    ref_path = cache_root / "refs" / revision
    if ref_path.exists():
        revision_name = ref_path.read_text(encoding="utf-8").strip()

    snapshot_dir = cache_root / "snapshots" / revision_name
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(
            f"Snapshot directory not found under {cache_root}. "
            f"Resolved revision={revision_name!r}."
        )
    return snapshot_dir


def load_source_dataframe(source_snapshot_dir: Path) -> pd.DataFrame:
    trials_path = source_snapshot_dir / "trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing trials.csv in {source_snapshot_dir}")

    source_df = pd.read_csv(trials_path)
    missing_columns = [name for name in REQUIRED_SOURCE_COLUMNS if name not in source_df.columns]
    if missing_columns:
        raise ValueError(
            f"trials.csv is missing required columns: {', '.join(missing_columns)}"
        )

    source_df = source_df.copy()
    source_df["wav_name"] = source_df["wav_path"].map(lambda value: Path(str(value)).name)
    name_mismatches = source_df.loc[source_df["wav_name"] != source_df["name"], ["wav_path", "name"]]
    if not name_mismatches.empty:
        example = name_mismatches.head(5).to_dict(orient="records")
        raise ValueError(f"Filename/name mismatches found in trials.csv: {example}")

    source_df["audio_path"] = source_df["name"].map(lambda name: source_snapshot_dir / str(name))
    missing_audio = source_df.loc[
        ~source_df["audio_path"].map(Path.exists), ["name", "audio_path"]
    ]
    if not missing_audio.empty:
        example = missing_audio.head(10).to_dict(orient="records")
        raise FileNotFoundError(f"Missing audio files referenced by trials.csv: {example}")

    if source_df["name"].duplicated().any():
        duplicates = (
            source_df.loc[source_df["name"].duplicated(keep=False), ["name", "label"]]
            .sort_values(["name", "label"])
            .head(20)
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate clip names found in trials.csv: {duplicates}")

    source_df["source_label"] = source_df["label"].astype(str)
    source_df["label_name"] = source_df["source_label"].map(
        lambda value: "noise" if value == "noise" else "whistle"
    )
    source_df["label_id"] = source_df["label_name"].map({"noise": 0, "whistle": 1}).astype(int)
    source_df = source_df.sort_values(["source_label", "name"]).reset_index(drop=True)
    return source_df


def _test_count(group_size: int, test_size: float) -> int:
    if group_size < 2:
        return 0
    proposed = int(round(group_size * test_size))
    proposed = max(1, proposed)
    proposed = min(group_size - 1, proposed)
    return proposed


def assign_source_group(source_label: str, name: str) -> str:
    if source_label != "noise":
        return f"identity::{source_label}"
    if name.startswith("new_noise_chunk-"):
        return "noise_family::new_noise_chunk"
    if name.startswith("noise_chunk-"):
        return "noise_family::noise_chunk"
    return "noise_family::other_noise"


def _build_group_summary(source_df: pd.DataFrame) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    grouped = source_df.groupby("source_group", sort=True)
    for source_group, group in grouped:
        label_name_counts = _counts_from_series(group["label_name"])
        source_label_counts = _counts_from_series(group["source_label"])
        summary.append(
            {
                "source_group": str(source_group),
                "rows": int(len(group)),
                "binary_counts": label_name_counts,
                "source_label_counts": source_label_counts,
                "contains_noise": int(label_name_counts.get("noise", 0) > 0),
                "contains_whistle": int(label_name_counts.get("whistle", 0) > 0),
            }
        )
    return summary


def _choose_grouped_test_groups(source_df: pd.DataFrame, *, test_size: float) -> set[str]:
    group_rows: list[dict[str, object]] = []
    total_rows = int(len(source_df))
    target_test_rows = float(total_rows) * float(test_size)
    total_noise = int((source_df["label_name"] == "noise").sum())
    total_whistle = int((source_df["label_name"] == "whistle").sum())

    for source_group, group in source_df.groupby("source_group", sort=True):
        noise_rows = int((group["label_name"] == "noise").sum())
        whistle_rows = int((group["label_name"] == "whistle").sum())
        group_rows.append(
            {
                "source_group": str(source_group),
                "rows": int(len(group)),
                "noise_rows": noise_rows,
                "whistle_rows": whistle_rows,
            }
        )

    if len(group_rows) < 2:
        raise ValueError("Need at least two source groups for a grouped split.")

    best_choice: tuple[float, float, float, tuple[str, ...]] | None = None
    names = [str(item["source_group"]) for item in group_rows]

    for subset_size in range(1, len(names)):
        for subset in combinations(names, subset_size):
            subset_set = set(subset)
            test_rows = sum(
                int(item["rows"]) for item in group_rows if item["source_group"] in subset_set
            )
            train_rows = total_rows - test_rows
            if test_rows == 0 or train_rows == 0:
                continue

            test_noise = sum(
                int(item["noise_rows"]) for item in group_rows if item["source_group"] in subset_set
            )
            test_whistle = sum(
                int(item["whistle_rows"]) for item in group_rows if item["source_group"] in subset_set
            )
            train_noise = total_noise - test_noise
            train_whistle = total_whistle - test_whistle

            if min(test_noise, test_whistle, train_noise, train_whistle) <= 0:
                continue

            test_ratio_error = abs(test_rows - target_test_rows) / max(target_test_rows, 1.0)
            test_noise_ratio = test_noise / max(test_rows, 1)
            total_noise_ratio = total_noise / max(total_rows, 1)
            label_ratio_error = abs(test_noise_ratio - total_noise_ratio)
            group_count_error = abs(len(subset_set) - (len(names) * test_size)) / max(len(names), 1)
            score = (test_ratio_error, label_ratio_error, group_count_error, tuple(sorted(subset)))

            if best_choice is None or score < best_choice:
                best_choice = score

    if best_choice is None:
        raise ValueError(
            "Could not find a grouped split that keeps both binary labels in train and test."
        )

    return set(best_choice[3])


def build_split_manifest(
    source_df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
    split_policy: str = DEFAULT_SPLIT_POLICY,
) -> pd.DataFrame:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")

    if split_policy not in {"stratified", "grouped_rigorous"}:
        raise ValueError(
            "--split-policy must be one of: stratified, grouped_rigorous."
        )

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    source_df = source_df.copy()
    source_df["source_group"] = [
        assign_source_group(source_label=str(row.source_label), name=str(row.name))
        for row in source_df.itertuples(index=False)
    ]

    if split_policy == "grouped_rigorous":
        test_groups = _choose_grouped_test_groups(source_df, test_size=test_size)
        for row in source_df.itertuples(index=True):
            split_name = "test" if row.source_group in test_groups else "train"
            records.append(
                {
                    "name": row.name,
                    "split": split_name,
                    "source_label": row.source_label,
                    "source_group": row.source_group,
                    "label_name": row.label_name,
                    "label_id": int(row.label_id),
                    "audio_path": str(Path(row.audio_path).resolve()),
                }
            )
        manifest = pd.DataFrame.from_records(records)
        manifest = manifest.sort_values(
            ["split", "source_group", "source_label", "name"]
        ).reset_index(drop=True)
        return manifest

    for source_label, group in source_df.groupby("source_label", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        test_count = _test_count(len(indices), test_size)
        test_indices = set(indices[:test_count].tolist())

        for row in group.itertuples(index=True):
            split_name = "test" if row.Index in test_indices else "train"
            records.append(
                {
                    "name": row.name,
                    "split": split_name,
                    "source_label": source_label,
                    "source_group": row.source_group,
                    "label_name": row.label_name,
                    "label_id": int(row.label_id),
                    "audio_path": str(Path(row.audio_path).resolve()),
                }
            )

    manifest = pd.DataFrame.from_records(records)
    manifest = manifest.sort_values(
        ["split", "source_group", "source_label", "name"]
    ).reset_index(drop=True)
    return manifest


def build_dataset_dict(
    manifest_df: pd.DataFrame,
    *,
    limit_per_split: int | None = None,
    selection_seed: int = DEFAULT_SEED,
) -> tuple[DatasetDict, dict[str, pd.DataFrame]]:
    split_frames: dict[str, pd.DataFrame] = {}
    dataset_dict = DatasetDict()

    for split_name in ("train", "test"):
        split_df = manifest_df.loc[manifest_df["split"] == split_name].copy()
        split_df = split_df.sort_values(
            ["source_group", "source_label", "name"]
        ).reset_index(drop=True)
        if limit_per_split is not None and len(split_df) > limit_per_split:
            split_df = split_df.sample(
                n=limit_per_split,
                random_state=selection_seed,
                replace=False,
            ).copy()
            split_df = split_df.sort_values(
                ["source_group", "source_label", "name"]
            ).reset_index(drop=True)
        split_frames[split_name] = split_df

        data = {
            "audio": [
                {"path": str(Path(path).resolve()), "bytes": None}
                for path in split_df["audio_path"].tolist()
            ],
            "label": [int(value) for value in split_df["label_id"].tolist()],
            "name": split_df["name"].astype(str).tolist(),
            "source_label": split_df["source_label"].astype(str).tolist(),
        }
        dataset_dict[split_name] = Dataset.from_dict(data, features=OUTPUT_FEATURES)

    return dataset_dict, split_frames


def _counts_from_series(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def build_summary(
    *,
    source_snapshot_dir: Path,
    manifest_df: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    repo_id: str,
    source_repo_id: str,
    test_size: float,
    seed: int,
    limit_per_split: int | None,
    split_policy: str,
) -> dict[str, object]:
    output_manifest_df = pd.concat(
        [frame.copy() for frame in split_frames.values()],
        ignore_index=True,
    )
    rows_by_split = {split: int(len(frame)) for split, frame in split_frames.items()}
    binary_counts_by_split = {
        split: _counts_from_series(frame["label_name"]) for split, frame in split_frames.items()
    }
    source_counts_by_split = {
        split: _counts_from_series(frame["source_label"]) for split, frame in split_frames.items()
    }

    summary = {
        "repo_id": repo_id,
        "source_repo_id": source_repo_id,
        "source_snapshot_dir": str(source_snapshot_dir),
        "split_policy": {
            "strategy": (
                "grouped_by_source_provenance"
                if split_policy == "grouped_rigorous"
                else "deterministic_stratified_by_source_label"
            ),
            "test_size": test_size,
            "seed": seed,
        },
        "limit_per_split": limit_per_split,
        "source_rows_total": int(len(manifest_df)),
        "rows_total": int(len(output_manifest_df)),
        "rows_by_split": rows_by_split,
        "binary_label_names": list(LABEL_NAMES),
        "binary_counts_total": _counts_from_series(output_manifest_df["label_name"]),
        "binary_counts_by_split": binary_counts_by_split,
        "source_label_counts_total": _counts_from_series(output_manifest_df["source_label"]),
        "source_label_counts_by_split": source_counts_by_split,
        "source_group_counts_by_split": {
            split: _counts_from_series(frame["source_group"]) for split, frame in split_frames.items()
        },
        "source_group_summary": _build_group_summary(source_df=source_df_from_manifest(output_manifest_df)),
        "features": [
            {"name": "audio", "dtype": "audio(decode=false)"},
            {"name": "label", "dtype": f"class_label({', '.join(LABEL_NAMES)})"},
            {"name": "name", "dtype": "string"},
            {"name": "source_label", "dtype": "string"},
        ],
    }
    return summary


def source_df_from_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    return manifest_df[
        ["name", "source_label", "source_group", "label_name"]
    ].drop_duplicates().reset_index(drop=True)


def render_readme(summary: dict[str, object]) -> str:
    rows_by_split = summary["rows_by_split"]
    binary_counts_total = summary["binary_counts_total"]
    binary_counts_by_split = summary["binary_counts_by_split"]
    source_counts_by_split = summary["source_label_counts_by_split"]
    source_group_counts_by_split = summary["source_group_counts_by_split"]
    split_policy = summary["split_policy"]
    source_repo_id = summary["source_repo_id"]

    def format_counts(counts: dict[str, int]) -> str:
        return ", ".join(f"{name}={value}" for name, value in counts.items())

    source_train = format_counts(source_counts_by_split["train"])
    source_test = format_counts(source_counts_by_split["test"])

    return f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# OpenWhistle-1.0-Detection-Finetuning

Binary whistle-vs-noise finetuning dataset rebuilt from `{source_repo_id}`.

## Source

- Source repo: `{source_repo_id}`
- Build rule: collapse every non-`noise` legacy label into `whistle`
- Split rule: `{split_policy["strategy"]}`
- Test size: `{split_policy["test_size"]}`
- Seed: `{split_policy["seed"]}`

## Features

- `audio`: audio clips stored with `decode=False`
- `label`: binary class label with values `noise` and `whistle`
- `name`: original clip filename from the legacy dataset
- `source_label`: original legacy label before the binary collapse

## Rows By Split

- `train`: `{rows_by_split["train"]}` rows
- `test`: `{rows_by_split["test"]}` rows

## Binary Label Counts

- `total`: `{format_counts(binary_counts_total)}`
- `train`: `{format_counts(binary_counts_by_split["train"])}`
- `test`: `{format_counts(binary_counts_by_split["test"])}`

## Source Label Counts

- `train`: `{source_train}`
- `test`: `{source_test}`

## Source Group Counts

- `train`: `{format_counts(source_group_counts_by_split["train"])}`
- `test`: `{format_counts(source_group_counts_by_split["test"])}`

## Example

```python
from datasets import Audio, load_dataset

dataset = load_dataset("dolphinteam/OpenWhistle-1.0-Detection-Finetuning")
decoded_train = dataset["train"].cast_column("audio", Audio())
sample = decoded_train[0]
```
"""


def write_artifacts(
    *,
    artifacts_dir: Path,
    manifest_df: pd.DataFrame,
    summary: dict[str, object],
) -> dict[str, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_path = artifacts_dir / "split_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    readme_path = artifacts_dir / "README.md"
    readme_path.write_text(render_readme(summary), encoding="utf-8")

    return {
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "readme_path": readme_path,
    }
