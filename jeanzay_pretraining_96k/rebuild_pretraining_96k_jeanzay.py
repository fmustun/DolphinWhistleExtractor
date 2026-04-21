#!/usr/bin/env python3
"""
Jean Zay wrapper for scripts/rebuild_pretraining_96k.py.

It keeps the main rebuild logic in one place and only injects a JSON path
configuration so the rebuild can run on a staged Jean Zay data tree.

Typical usage on Jean Zay:
    python jeanzay_pretraining_96k/rebuild_pretraining_96k_jeanzay.py \
        --path-config /path/to/stage/generated/path_config.json \
        --push \
        --target-repo dolphinteam/DolphinWhistle-Pretraining-96k
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path-config",
        required=True,
        help="JSON file describing staged segment and recording directories.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root containing scripts/rebuild_pretraining_96k.py.",
    )
    parser.add_argument(
        "--print-resolved-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    return parser.parse_known_args()


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


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def normalize_config(path_config: Path) -> dict:
    config_dir = path_config.parent
    raw = json.loads(path_config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid path config: expected object in {path_config}")

    raw_segment_dirs = raw.get("segment_dirs")
    raw_recording_dirs = raw.get("recording_dirs")
    if not isinstance(raw_segment_dirs, dict) or not isinstance(raw_recording_dirs, dict):
        raise ValueError(
            "Path config must contain 'segment_dirs' and 'recording_dirs' objects."
        )

    segment_dirs: dict[str, Path] = {}
    for year_label, value in raw_segment_dirs.items():
        if not isinstance(value, str):
            raise ValueError(f"segment_dirs[{year_label!r}] must be a string path")
        segment_dirs[year_label] = resolve_path(value, config_dir)

    recording_dirs: dict[str, list[Path]] = {}
    for year_label, values in raw_recording_dirs.items():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(
                f"recording_dirs[{year_label!r}] must be a string or list of string paths"
            )
        recording_dirs[year_label] = [resolve_path(v, config_dir) for v in values]

    normalized = {
        "segment_dirs": segment_dirs,
        "recording_dirs": recording_dirs,
    }
    for key in ("source_repo", "target_repo"):
        if key in raw:
            normalized[key] = raw[key]
    return normalized


def apply_config(base_module, config: dict) -> None:
    base_module.YEAR_SEGMENT_DIRS = dict(config["segment_dirs"])
    base_module.RECORDING_DIRS = dict(config["recording_dirs"])
    if "source_repo" in config:
        base_module.SOURCE_REPO = str(config["source_repo"])
    if "target_repo" in config:
        base_module.TARGET_REPO = str(config["target_repo"])


def config_to_jsonable(config: dict) -> dict:
    jsonable = {
        "segment_dirs": {k: str(v) for k, v in config["segment_dirs"].items()},
        "recording_dirs": {k: [str(p) for p in v] for k, v in config["recording_dirs"].items()},
    }
    for key in ("source_repo", "target_repo"):
        if key in config:
            jsonable[key] = config[key]
    return jsonable


def main() -> None:
    args, passthrough = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    path_config = Path(args.path_config).expanduser().resolve()
    if not path_config.is_file():
        raise FileNotFoundError(f"Path config not found: {path_config}")

    base_module = load_base_module(repo_root)
    config = normalize_config(path_config)
    apply_config(base_module, config)

    if args.print_resolved_config:
        print(json.dumps(config_to_jsonable(config), indent=2, sort_keys=True))
        return

    sys.argv = [str(repo_root / "scripts" / "rebuild_pretraining_96k.py"), *passthrough]
    base_module.main()


if __name__ == "__main__":
    main()
