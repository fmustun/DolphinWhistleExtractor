#!/usr/bin/env python3
"""Push local classification F0 datasets to Hugging Face config by config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REPO_ID = "dolphinteam/DophinWhistle-Classification-Finetuning"
DEFAULT_DATASET_PATHS = {
    "default": {
        "exact": REPO_ROOT / "cnn_dataset" / "datasets" / "classification_f0_default_fixed_v2",
        "rebuilt": REPO_ROOT / "cnn_dataset" / "datasets" / "classification_f0_rebuilt_default",
    },
    "unbalanced": {
        "rebuilt": REPO_ROOT / "cnn_dataset" / "datasets" / "classification_f0_rebuilt_unbalanced",
    },
    "all": {
        "rebuilt": REPO_ROOT / "cnn_dataset" / "datasets" / "classification_f0_rebuilt_all",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Push local classification F0 datasets to the Hub without rebuilding "
            "them first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Destination Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--default-source",
        choices=("exact", "rebuilt"),
        default="exact",
        help=(
            "Source used for the 'default' config. 'exact' reuses the values copied "
            "from DolphinWhistle-Finetuning-withF0; 'rebuilt' uses the freshly "
            "re-estimated dataset."
        ),
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, uses local login state.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="500MB",
        help="Shard size passed to push_to_hub().",
    )
    parser.add_argument(
        "--commit-message",
        default="Update classification F0 datasets",
        help="Base commit message used for pushes.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=("default", "unbalanced", "all"),
        default=("default", "unbalanced", "all"),
        help="Configs to push.",
    )
    parser.add_argument(
        "--default-path",
        default=None,
        help="Optional explicit dataset path for the default config.",
    )
    parser.add_argument(
        "--unbalanced-path",
        default=None,
        help="Optional explicit dataset path for the unbalanced config.",
    )
    parser.add_argument(
        "--all-path",
        default=None,
        help="Optional explicit dataset path for the all config.",
    )
    return parser.parse_args()


def require_dependencies() -> object:
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - runtime guidance
        raise SystemExit(
            "Missing dependency 'datasets'. Install it before pushing the dataset."
        ) from exc
    return load_from_disk


def resolve_dataset_path(
    config_name: str,
    default_source: str,
    explicit_path: str | None,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
    elif config_name == "default":
        path = DEFAULT_DATASET_PATHS["default"][default_source]
    else:
        path = DEFAULT_DATASET_PATHS[config_name]["rebuilt"]

    if not path.exists():
        raise SystemExit(f"Missing local dataset directory for config={config_name!r}: {path}")
    return path


def summarize_rows(dataset_dict: object) -> str:
    parts = [
        f"{split_name}={int(dataset_dict[split_name].num_rows)}"
        for split_name in dataset_dict.keys()
    ]
    return ", ".join(parts)


def push_config(
    *,
    load_from_disk: object,
    repo_id: str,
    config_name: str,
    dataset_path: Path,
    token: str | None,
    max_shard_size: str,
    commit_message: str,
) -> None:
    dataset_dict = load_from_disk(str(dataset_path))
    row_summary = summarize_rows(dataset_dict)
    print(
        f"Pushing config={config_name} from {dataset_path} "
        f"with splits: {row_summary}"
    )
    dataset_dict.push_to_hub(
        repo_id,
        config_name=config_name,
        token=token,
        max_shard_size=max_shard_size,
        commit_message=f"{commit_message} ({config_name})",
    )


def main() -> None:
    args = parse_args()
    load_from_disk = require_dependencies()
    explicit_paths = {
        "default": args.default_path,
        "unbalanced": args.unbalanced_path,
        "all": args.all_path,
    }

    for config_name in args.configs:
        dataset_path = resolve_dataset_path(
            config_name,
            args.default_source,
            explicit_paths[config_name],
        )
        push_config(
            load_from_disk=load_from_disk,
            repo_id=args.repo_id,
            config_name=config_name,
            dataset_path=dataset_path,
            token=args.token,
            max_shard_size=args.max_shard_size,
            commit_message=args.commit_message,
        )


if __name__ == "__main__":
    main()
