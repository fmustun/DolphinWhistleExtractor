
#!/usr/bin/env python3
"""
Rebuild the dolphinteam/DolphinWhistle-Pretraining dataset at 96 kHz.

Fixes compared with the original version:
- persistent disk caches for the segment index, original recording lookup, and
  collision resolution
- resumable processing with per-split checkpoints
- lazy decoding of HF audio rows: only collision rows download/decode source audio
- invalid temporary shards are ignored/cleaned on resume instead of crashing late
- completed splits can be reused directly after a late crash during save/push
- faster extraction/read path using soundfile instead of librosa when possible
- no premature deletion of temporary shard files before save/push completes

Usage:
    python scripts/rebuild_pretraining_96k.py \
        --save-local /home/pablo/DolphinWhistle-Pretraining-96k \
        --resume
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import pickle
import shutil
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration: segment directories (extracted at 44.1 kHz)
# ---------------------------------------------------------------------------
SEGMENT_BASE = Path("/media/DOLPHIN_robin/Analyses_alexis/Extracted_segments")
YEAR_SEGMENT_DIRS = {
    "2019_2020": SEGMENT_BASE / "2019_2020",
    "2021": SEGMENT_BASE / "2021",
    "2023": SEGMENT_BASE / "2023",
    "2024_analysed_(jan-apr)": SEGMENT_BASE / "2024_analysed_(jan-apr)",
}

YEAR_LABEL_TO_INT = {
    "2019_2020": {2019, 2020},
    "2021": {2021},
    "2023": {2023},
    "2024_analysed_(jan-apr)": {2024},
}
YEAR_INT_TO_LABEL = {}
for _label, _ints in YEAR_LABEL_TO_INT.items():
    for _y in _ints:
        YEAR_INT_TO_LABEL[_y] = _label

# ---------------------------------------------------------------------------
# Configuration: original 96 kHz recording directories
# ---------------------------------------------------------------------------
RECORDING_DIRS: dict[str, list[Path]] = {
    "2019_2020": [Path("/media/zfnews31/Dolphins/Sound")],
    "2021": [Path("/media/zfnews31/Dolphins/Sound/2021")],
    "2023": [Path("/media/DOLPHIN_robin/2023")],
    "2024_analysed_(jan-apr)": [Path("/media/DOLPHIN_robin/2024")],
}

TARGET_SR = 96_000
SOURCE_REPO = "dolphinteam/DolphinWhistle-Pretraining"
TARGET_REPO = "dolphinteam/DolphinWhistle-Pretraining-96k"
DEFAULT_RESAMPLE_TYPE = "soxr_hq"

# Conservative defaults: smaller flush/checkpoint cadence makes recovery
# cheaper and avoids large in-memory audio buffers on desktop machines.
BATCH_SIZE = 32
CHECKPOINT_EVERY = 32


def make_audio_feature(*, sampling_rate=None, decode=None, mono=None) -> Audio:
    """Build an Audio feature that works across datasets versions."""
    supported = inspect.signature(Audio.__init__).parameters
    kwargs = {}
    if sampling_rate is not None and "sampling_rate" in supported:
        kwargs["sampling_rate"] = sampling_rate
    if decode is not None and "decode" in supported:
        kwargs["decode"] = decode
    if mono is not None and "mono" in supported:
        kwargs["mono"] = mono
    return Audio(**kwargs)

OUTPUT_FEATURES = Features({
    "audio": make_audio_feature(sampling_rate=TARGET_SR, mono=True),
    "start_time": Value("float32"),
    "end_time": Value("float32"),
    "duration": Value("float32"),
    "year": Value("int32"),
    "hydrophone": Value("string"),
})
SOURCE_AUDIO_FEATURE = make_audio_feature(decode=False)
COLLISION_AUDIO_DECODER = make_audio_feature(decode=True, mono=True)
CHECKPOINT_DEFAULTS = {
    "processed_rows": 0,
    "rows_written": 0,
    "completed": False,
    "n_unique": 0,
    "n_collision_total": 0,
    "n_collision_resolved": 0,
    "n_collision_fallback": 0,
    "n_collision_decode_error": 0,
    "n_not_found_index": 0,
    "n_not_found_recording": 0,
    "n_extract_error": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", default=TARGET_REPO,
                        help="HF repo id for the new dataset")
    parser.add_argument("--push", action="store_true",
                        help="Push to HuggingFace after building")
    parser.add_argument("--private", action="store_true",
                        help="Create repo as private")
    parser.add_argument("--save-local", type=str, default=None,
                        help="Save dataset locally at this path")
    parser.add_argument("--max-shard-size", default="500MB")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation"],
                        default=["train", "validation"],
                        help="Only process these splits")
    parser.add_argument("--soft-mode", action="store_true",
                        help="Reduce CPU bursts and IO pressure to keep the machine responsive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only build the index and report stats, don't extract")
    parser.add_argument("--work-dir", type=str, default="/tmp/pretraining_96k_shards",
                        help="Temporary work directory")
    parser.add_argument("--cache-dir", type=str, default="/tmp/pretraining_96k_cache",
                        help="Persistent cache directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from per-split checkpoints if present")
    parser.add_argument("--reset-run-state", action="store_true",
                        help="Delete work-dir and cache-dir before starting, then rebuild from scratch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Rows to buffer before writing a shard")
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                        help="Rows between checkpoint flushes")
    parser.add_argument("--nice", type=int, default=0,
                        help="Positive niceness to apply to this process")
    parser.add_argument("--yield-every", type=int, default=0,
                        help="Sleep briefly every N processed rows to reduce desktop lag")
    parser.add_argument("--yield-seconds", type=float, default=0.0,
                        help="Sleep duration used with --yield-every")
    parser.add_argument("--resample-type", default=DEFAULT_RESAMPLE_TYPE,
                        help="librosa resample type, e.g. soxr_hq, soxr_mq, kaiser_fast")
    parser.add_argument("--rebuild-index-cache", action="store_true",
                        help="Force rebuild of cached segment index")
    parser.add_argument("--keep-temp-shards", action="store_true",
                        help="Do not delete temporary shard directories")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Small JSON / pickle helpers
# ---------------------------------------------------------------------------
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def stable_key_hash(parts: tuple[object, ...]) -> str:
    return hashlib.sha1("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def collision_cache_key(
    split_name: str,
    row_index: int,
    year_label: str,
    hydrophone: str,
    audio_path: str,
    start_time: float,
    end_time: float,
) -> str:
    return stable_key_hash((
        "collision-v2",
        split_name,
        f"row={row_index}",
        year_label,
        hydrophone,
        audio_path,
        f"{start_time:.6f}",
        f"{end_time:.6f}",
    ))


def normalize_audio_name(audio_info: dict) -> str:
    raw_path = audio_info.get("path")
    if not raw_path:
        return ""
    return Path(raw_path.split("::")[-1]).name


def decode_source_audio(audio_info: dict) -> tuple[np.ndarray, int] | None:
    try:
        if "array" in audio_info and "sampling_rate" in audio_info:
            array = audio_info["array"]
            sampling_rate = int(audio_info["sampling_rate"])
        else:
            decoded = COLLISION_AUDIO_DECODER.decode_example(audio_info)
            array = decoded["array"]
            sampling_rate = int(decoded["sampling_rate"])
    except Exception:
        return None

    return np.asarray(array, dtype=np.float32), sampling_rate


def load_source_split(split_name: str):
    ds = load_dataset(SOURCE_REPO, split=split_name, streaming=True)
    return ds.cast_column("audio", SOURCE_AUDIO_FEATURE)


SEGMENT_FRAME_COUNT_CACHE: dict[str, int | None] = {}


def segment_frame_count(seg_path: Path) -> int | None:
    cache_key = str(seg_path)
    if cache_key in SEGMENT_FRAME_COUNT_CACHE:
        return SEGMENT_FRAME_COUNT_CACHE[cache_key]

    try:
        with sf.SoundFile(str(seg_path), "r") as f:
            frame_count = len(f)
    except Exception:
        frame_count = None

    SEGMENT_FRAME_COUNT_CACHE[cache_key] = frame_count
    return frame_count


def choose_collision_fallback(candidates: list[tuple[str, str]]) -> tuple[str, Path]:
    """Prefer a non-empty candidate segment when collision resolution falls back."""
    if not candidates:
        raise ValueError("choose_collision_fallback requires at least one candidate")

    best_exp_name = candidates[0][0]
    best_seg_path = Path(candidates[0][1])
    best_frames = segment_frame_count(best_seg_path)
    best_score = best_frames if best_frames is not None else -1

    for exp_name, seg_path_str in candidates[1:]:
        seg_path = Path(seg_path_str)
        frame_count = segment_frame_count(seg_path)
        score = frame_count if frame_count is not None else -1
        if score > best_score:
            best_exp_name = exp_name
            best_seg_path = seg_path
            best_frames = frame_count
            best_score = score

    return best_exp_name, best_seg_path


def collision_candidate_details(
    candidates: list[tuple[str, str]],
    corr_by_seg_path: dict[str, float | None] | None = None,
) -> list[dict[str, str | int | float | None]]:
    details: list[dict[str, str | int | None]] = []
    for exp_name, seg_path_str in candidates:
        seg_path = Path(seg_path_str)
        item: dict[str, str | int | float | None] = {
            "exp_name": exp_name,
            "seg_path": str(seg_path),
            "frame_count": segment_frame_count(seg_path),
        }
        if corr_by_seg_path is not None:
            item["corr"] = corr_by_seg_path.get(str(seg_path))
        details.append(item)
    return details


def format_collision_candidates(
    details: list[dict[str, str | int | float | None]],
) -> str:
    parts: list[str] = []
    for item in details:
        frame_count = item.get("frame_count")
        frame_text = "?" if frame_count is None else str(frame_count)
        corr = item.get("corr")
        if corr is None:
            parts.append(f"{item['exp_name']}({frame_text} frames)")
        else:
            parts.append(f"{item['exp_name']}(corr={float(corr):.6f}, {frame_text} frames)")
    return ", ".join(parts)


def log_collision_fallback(
    cache_dir: Path,
    split_name: str,
    row_index: int | None,
    reason: str,
    audio_path: str,
    year_label: str,
    hydrophone: str,
    start_time: float,
    end_time: float,
    candidates: list[tuple[str, str]],
    chosen_exp_name: str,
) -> None:
    details = collision_candidate_details(candidates)
    summary = format_collision_candidates(details)
    print(
        "  Collision fallback:"
        f" row={row_index if row_index is not None else '?'}"
        f" reason={reason}"
        f" split={split_name}"
        f" year={year_label}"
        f" hydrophone={hydrophone}"
        f" audio={audio_path}"
        f" start={start_time:.3f}"
        f" end={end_time:.3f}"
        f" chosen={chosen_exp_name}"
        f" candidates=[{summary}]"
    )

    log_path = cache_dir / f"{split_name}_collision_fallbacks.jsonl"
    record = {
        "reason": reason,
        "split": split_name,
        "row_index": row_index,
        "year_label": year_label,
        "hydrophone": hydrophone,
        "audio_path": audio_path,
        "start_time": start_time,
        "end_time": end_time,
        "chosen_exp_name": chosen_exp_name,
        "candidates": details,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def log_collision_resolved(
    cache_dir: Path,
    split_name: str,
    row_index: int,
    audio_path: str,
    year_label: str,
    hydrophone: str,
    start_time: float,
    end_time: float,
    resolution: dict[str, object],
) -> None:
    details = list(resolution["candidates"])
    chosen_exp_name = str(resolution["exp_name"])
    source = str(resolution["source"])
    best_corr = resolution.get("best_corr")
    best_corr_text = "cached" if best_corr is None else f"{float(best_corr):.6f}"
    print(
        "  Collision resolved:"
        f" row={row_index}"
        f" source={source}"
        f" split={split_name}"
        f" year={year_label}"
        f" hydrophone={hydrophone}"
        f" audio={audio_path}"
        f" start={start_time:.3f}"
        f" end={end_time:.3f}"
        f" chosen={chosen_exp_name}"
        f" best_corr={best_corr_text}"
    )

    log_path = cache_dir / f"{split_name}_collision_resolved.jsonl"
    record = {
        "split": split_name,
        "row_index": row_index,
        "year_label": year_label,
        "hydrophone": hydrophone,
        "audio_path": audio_path,
        "start_time": start_time,
        "end_time": end_time,
        "source": source,
        "best_corr": best_corr,
        "chosen_exp_name": chosen_exp_name,
        "chosen_seg_path": str(resolution["seg_path"]),
        "threshold": resolution.get("threshold"),
        "candidates": details,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def maybe_sleep(processed_rows: int, yield_every: int, yield_seconds: float) -> None:
    if yield_every > 0 and yield_seconds > 0 and processed_rows % yield_every == 0:
        time.sleep(yield_seconds)


def apply_runtime_tuning(args: argparse.Namespace) -> None:
    if args.soft_mode:
        if args.batch_size == BATCH_SIZE:
            args.batch_size = 64
        if args.checkpoint_every == CHECKPOINT_EVERY:
            args.checkpoint_every = 64
        if args.yield_every == 0:
            args.yield_every = 32
        if args.yield_seconds == 0.0:
            args.yield_seconds = 0.02

    if args.nice > 0:
        try:
            applied_nice = os.nice(args.nice)
            print(f"Applied niceness: +{args.nice} (current nice={applied_nice})")
        except OSError as exc:
            print(f"WARNING: could not apply niceness +{args.nice}: {exc}")

    if args.soft_mode:
        print(
            "Soft mode enabled: "
            f"batch_size={args.batch_size}, "
            f"checkpoint_every={args.checkpoint_every}, "
            f"yield_every={args.yield_every}, "
            f"yield_seconds={args.yield_seconds:.3f}, "
            f"nice={args.nice}"
        )


def reset_run_state(work_dir: Path, cache_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Removed existing work dir: {work_dir}")
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"Removed existing cache dir: {cache_dir}")


# ---------------------------------------------------------------------------
# Step 1: Build / load cached segment index from year directories
# ---------------------------------------------------------------------------
def build_segment_index(cache_dir: Path, rebuild_cache: bool = False) -> dict[tuple[str, str, str], list[tuple[str, str]]]:
    """
    Returns:
        {(year_label, hydro, fname): [(exp_name, seg_path_str), ...]}
    """
    cache_path = cache_dir / "segment_index.pkl"
    if cache_path.exists() and not rebuild_cache:
        with cache_path.open("rb") as f:
            index = pickle.load(f)
        print(f"Loaded cached segment index from {cache_path}")
        return index

    index: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    total = 0
    for year_label, year_dir in YEAR_SEGMENT_DIRS.items():
        if not year_dir.is_dir():
            print(f"WARNING: missing segment dir {year_dir}")
            continue
        for exp_name in sorted(os.listdir(year_dir)):
            exp_path = year_dir / exp_name
            if not exp_path.is_dir():
                continue
            if "_channel_" in exp_name:
                hydro = "channel_" + exp_name.split("_channel_")[-1]
            else:
                hydro = "channel_0"
            for fname in os.listdir(exp_path):
                if fname.endswith(".wav"):
                    key = (year_label, hydro, fname)
                    index[key].append((exp_name, str(exp_path / fname)))
                    total += 1

    unique = sum(1 for v in index.values() if len(v) == 1)
    collisions = sum(1 for v in index.values() if len(v) > 1)
    print(f"Segment index: {total} segments, {len(index)} unique keys, "
          f"{unique} unambiguous, {collisions} collisions")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(dict(index), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved segment index cache to {cache_path}")
    return dict(index)


# ---------------------------------------------------------------------------
# Step 2: Find original recording path for an experiment
# ---------------------------------------------------------------------------
class RecordingResolver:
    def __init__(self, cache_dir: Path):
        self.cache_path = cache_dir / "recording_cache.json"
        self.cache: dict[str, str | None] = load_json(self.cache_path, {})

    def find(self, exp_name: str, year_label: str) -> Path | None:
        cache_key = f"{year_label}/{exp_name}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            return Path(cached) if cached else None

        found = None
        for rec_dir in RECORDING_DIRS.get(year_label, []):
            candidate = rec_dir / f"{exp_name}.wav"
            if candidate.exists():
                found = candidate
                break
            candidate = rec_dir / f"2022{exp_name}.wav"
            if candidate.exists():
                found = candidate
                break

        self.cache[cache_key] = str(found) if found else None
        return found

    def flush(self) -> None:
        save_json(self.cache_path, self.cache)


# ---------------------------------------------------------------------------
# Step 3: Resolve collisions by comparing audio
# ---------------------------------------------------------------------------
class CollisionResolver:
    def __init__(self, cache_dir: Path, resample_type: str = DEFAULT_RESAMPLE_TYPE):
        self.cache_path = cache_dir / "collision_cache.json"
        self.cache: dict[str, dict[str, str | float]] = load_json(self.cache_path, {})
        self.resample_type = resample_type
        self.match_threshold = 0.95

    def _candidate_audio(self, seg_path: Path, target_sr: int) -> np.ndarray | None:
        try:
            audio, sr = sf.read(str(seg_path), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=target_sr,
                    res_type=self.resample_type,
                )
            return audio.astype(np.float32, copy=False)
        except Exception:
            return None

    def resolve_with_details(
        self,
        cache_key: str,
        hf_audio_array: np.ndarray,
        hf_sr: int,
        candidates: list[tuple[str, str]],
    ) -> dict[str, object] | None:
        cached = self.cache.get(cache_key)
        if cached:
            corr_by_seg_path: dict[str, float | None] = {}
            seg_path_str = str(cached["seg_path"])
            corr_value = cached.get("corr")
            if corr_value is not None:
                corr_by_seg_path[seg_path_str] = float(corr_value)
            return {
                "exp_name": str(cached["exp_name"]),
                "seg_path": Path(seg_path_str),
                "source": "cache",
                "best_corr": float(corr_value) if corr_value is not None else None,
                "threshold": self.match_threshold,
                "candidates": collision_candidate_details(candidates, corr_by_seg_path or None),
            }

        best_match = None
        best_corr = -1.0
        corr_by_seg_path: dict[str, float | None] = {}
        for exp_name, seg_path_str in candidates:
            seg_path = Path(seg_path_str)
            corr_by_seg_path[str(seg_path)] = None
            local_audio = self._candidate_audio(seg_path, hf_sr)
            if local_audio is None:
                continue
            min_len = min(len(hf_audio_array), len(local_audio))
            if min_len < 100:
                continue
            a = hf_audio_array[:min_len]
            b = local_audio[:min_len]
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            corr = float(np.abs(np.dot(a, b)) / denom)
            corr_by_seg_path[str(seg_path)] = corr
            if corr > best_corr:
                best_corr = corr
                best_match = (exp_name, seg_path)

        if best_match is not None and best_corr > self.match_threshold:
            exp_name, seg_path = best_match
            self.cache[cache_key] = {
                "exp_name": exp_name,
                "seg_path": str(seg_path),
                "corr": best_corr,
            }
            return {
                "exp_name": exp_name,
                "seg_path": seg_path,
                "source": "computed",
                "best_corr": best_corr,
                "threshold": self.match_threshold,
                "candidates": collision_candidate_details(candidates, corr_by_seg_path),
            }
        return None

    def resolve(
        self,
        cache_key: str,
        hf_audio_array: np.ndarray,
        hf_sr: int,
        candidates: list[tuple[str, str]],
    ) -> tuple[str, Path] | None:
        details = self.resolve_with_details(cache_key, hf_audio_array, hf_sr, candidates)
        if details is None:
            return None
        return str(details["exp_name"]), Path(str(details["seg_path"]))

    def flush(self) -> None:
        save_json(self.cache_path, self.cache)


# ---------------------------------------------------------------------------
# Step 4: Extract segment from original recording at 96 kHz
# ---------------------------------------------------------------------------
def extract_segment_96k(
    recording_path: Path,
    start_time: float,
    end_time: float,
    resample_type: str = DEFAULT_RESAMPLE_TYPE,
) -> np.ndarray:
    """
    Fast path:
      - use soundfile seek/read to avoid librosa overhead
      - only resample if the source file is not already at TARGET_SR
    """
    duration = max(0.0, end_time - start_time)
    with sf.SoundFile(str(recording_path), "r") as f:
        sr = int(f.samplerate)
        start_frame = max(0, int(round(start_time * sr)))
        n_frames = max(0, int(round(duration * sr)))
        f.seek(start_frame)
        audio = f.read(n_frames, dtype="float32", always_2d=False)

    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=TARGET_SR,
            res_type=resample_type,
        )
    return np.asarray(audio, dtype=np.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def checkpoint_path(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / f"{split_name}_checkpoint.json"


def load_checkpoint(cache_dir: Path, split_name: str) -> dict:
    state = load_json(checkpoint_path(cache_dir, split_name), {})
    merged = dict(CHECKPOINT_DEFAULTS)
    if isinstance(state, dict):
        merged.update(state)
    merged["_has_rows_written"] = isinstance(state, dict) and "rows_written" in state
    return merged


def save_checkpoint(cache_dir: Path, split_name: str, state: dict) -> None:
    save_json(checkpoint_path(cache_dir, split_name), state)


# ---------------------------------------------------------------------------
# Shard writer: flush batches to disk as Arrow shards
# ---------------------------------------------------------------------------
class ShardWriter:
    """Accumulates rows in memory and flushes to disk as Arrow shards."""

    def __init__(
        self,
        output_dir: Path,
        split_name: str,
        batch_size: int = BATCH_SIZE,
        resume: bool = False,
    ):
        self.output_dir = output_dir / split_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_name = split_name
        self.batch_size = batch_size
        self.buffer: dict[str, list] = self._empty_buffer()
        self.total_rows = 0

        existing = sorted(self.output_dir.glob("shard_*"))

        if not resume and existing:
            for p in existing:
                shutil.rmtree(p, ignore_errors=True)
            existing = []

        self.shard_paths = self._valid_shard_paths(cleanup_invalid=resume)
        self.shard_idx = self._next_shard_idx(self.shard_paths)

        if resume and self.shard_paths:
            print(f"Resuming {split_name}: found {len(self.shard_paths)} valid shard(s) in {self.output_dir}")
            for shard_path in self.shard_paths:
                shard_ds = load_from_disk(str(shard_path))
                self.total_rows += len(shard_ds)

    @staticmethod
    def _empty_buffer() -> dict[str, list]:
        return {
            "audio": [],
            "start_time": [],
            "end_time": [],
            "duration": [],
            "year": [],
            "hydrophone": [],
        }

    def _valid_shard_paths(self, cleanup_invalid: bool) -> list[Path]:
        valid_shards: list[Path] = []
        for shard_path in sorted(self.output_dir.glob("shard_*")):
            try:
                shard_ds = load_from_disk(str(shard_path))
                len(shard_ds)
            except Exception as exc:
                print(f"WARNING: ignoring invalid shard {shard_path}: {exc}")
                if cleanup_invalid:
                    shutil.rmtree(shard_path, ignore_errors=True)
                continue
            valid_shards.append(shard_path)
        return valid_shards

    @staticmethod
    def _next_shard_idx(shard_paths: list[Path]) -> int:
        if not shard_paths:
            return 0
        shard_ids: list[int] = []
        for shard_path in shard_paths:
            try:
                shard_ids.append(int(shard_path.name.split("_")[-1]))
            except ValueError:
                continue
        if not shard_ids:
            return len(shard_paths)
        return max(shard_ids) + 1

    def add(self, audio: np.ndarray, start_time: float, end_time: float,
            duration: float, year: int, hydrophone: str) -> bool:
        self.buffer["audio"].append({"array": audio, "sampling_rate": TARGET_SR})
        self.buffer["start_time"].append(start_time)
        self.buffer["end_time"].append(end_time)
        self.buffer["duration"].append(duration)
        self.buffer["year"].append(year)
        self.buffer["hydrophone"].append(hydrophone)
        self.total_rows += 1

        if len(self.buffer["audio"]) >= self.batch_size:
            self.flush()
            return True
        return False

    def flush(self) -> None:
        if not self.buffer["audio"]:
            return
        ds = Dataset.from_dict(self.buffer, features=OUTPUT_FEATURES)
        shard_path = self.output_dir / f"shard_{self.shard_idx:05d}"
        ds.save_to_disk(str(shard_path))
        self.shard_paths.append(shard_path)
        self.shard_idx += 1
        self.buffer = self._empty_buffer()
        gc.collect()

    def finalize(self) -> tuple[Dataset, list[Path]]:
        """Flush remaining rows and concatenate all shards."""
        self.flush()

        shard_paths = self._valid_shard_paths(cleanup_invalid=False)
        if not shard_paths:
            return Dataset.from_dict(self._empty_buffer(), features=OUTPUT_FEATURES), []

        shards = [load_from_disk(str(p)) for p in shard_paths]
        ds = concatenate_datasets(shards)
        return ds, shard_paths

    @staticmethod
    def cleanup(shard_paths: list[Path]) -> None:
        for shard_path in shard_paths:
            shutil.rmtree(shard_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Dry-run: streaming, no audio download except for collisions
# ---------------------------------------------------------------------------
def dry_run_split(
    split_name: str,
    segment_index: dict[tuple[str, str, str], list[tuple[str, str]]],
    recording_resolver: RecordingResolver,
) -> None:
    print(f"\n{'='*60}")
    print(f"Dry-run: {split_name}")
    print(f"{'='*60}")

    ds = load_source_split(split_name)

    n_total = 0
    n_unique = 0
    n_collision = 0
    n_not_found_index = 0
    n_not_found_recording = 0
    missing_recordings: set[str] = set()

    for row in tqdm(ds, desc=f"{split_name}"):
        n_total += 1
        audio_path = normalize_audio_name(row["audio"])
        hf_year = int(row["year"])
        hf_hydro = row["hydrophone"]

        year_label = YEAR_INT_TO_LABEL.get(hf_year)
        if year_label is None:
            n_not_found_index += 1
            continue

        key = (year_label, hf_hydro, audio_path)
        candidates = segment_index.get(key, [])

        if len(candidates) == 0:
            n_not_found_index += 1
            continue
        elif len(candidates) == 1:
            exp_name = candidates[0][0]
            n_unique += 1
        else:
            exp_name = choose_collision_fallback(candidates)[0]
            n_collision += 1

        rec = recording_resolver.find(exp_name, year_label)
        if rec is None:
            n_not_found_recording += 1
            missing_recordings.add(f"{year_label}/{exp_name}")

    recording_resolver.flush()

    print(f"\nDry-run stats for {split_name}:")
    print(f"  Total rows:           {n_total}")
    print(f"  Unique matches:       {n_unique}")
    print(f"  Collisions (any):     {n_collision}")
    print(f"  Not found in index:   {n_not_found_index}")
    print(f"  Recording not found:  {n_not_found_recording}")
    if missing_recordings:
        print(f"  Missing recordings (first 20):")
        for m in sorted(missing_recordings)[:20]:
            print(f"    {m}")


# ---------------------------------------------------------------------------
# Full run: extract 96 kHz audio, shard to disk
# ---------------------------------------------------------------------------
def process_split(
    split_name: str,
    segment_index: dict[tuple[str, str, str], list[tuple[str, str]]],
    work_dir: Path,
    cache_dir: Path,
    recording_resolver: RecordingResolver,
    collision_resolver: CollisionResolver,
    batch_size: int = BATCH_SIZE,
    checkpoint_every: int = CHECKPOINT_EVERY,
    yield_every: int = 0,
    yield_seconds: float = 0.0,
    resample_type: str = DEFAULT_RESAMPLE_TYPE,
    resume: bool = False,
) -> tuple[Dataset, list[Path]]:
    print(f"\n{'='*60}")
    print(f"Processing split: {split_name}")
    print(f"{'='*60}")

    ckpt = load_checkpoint(cache_dir, split_name) if resume else dict(CHECKPOINT_DEFAULTS)
    writer = ShardWriter(work_dir, split_name, batch_size=batch_size, resume=resume)

    if resume and ckpt["_has_rows_written"] and writer.total_rows != int(ckpt["rows_written"]):
        print(
            f"WARNING: checkpoint/shard mismatch for {split_name} "
            f"(checkpoint rows_written={ckpt['rows_written']}, shard rows={writer.total_rows}). "
            "Rebuilding this split from scratch."
        )
        writer = ShardWriter(work_dir, split_name, batch_size=batch_size, resume=False)
        ckpt = dict(CHECKPOINT_DEFAULTS)

    if resume and ckpt["completed"] and writer.shard_paths:
        print(f"Split {split_name} already completed, reusing {len(writer.shard_paths)} shard(s).")
        return writer.finalize()

    # Use streaming to avoid loading the full dataset into RAM and keep the
    # audio column undecoded on the hot path.
    ds = load_source_split(split_name)

    if resume and ckpt["processed_rows"] > 0 and hasattr(ds, "skip"):
        print(f"Resuming {split_name} from source row {ckpt['processed_rows']}")
        ds = ds.skip(ckpt["processed_rows"])

    processed_rows = int(ckpt["processed_rows"])
    n_unique = int(ckpt["n_unique"])
    n_collision_total = int(ckpt["n_collision_total"])
    n_collision_resolved = int(ckpt["n_collision_resolved"])
    n_collision_fallback = int(ckpt["n_collision_fallback"])
    n_collision_decode_error = int(ckpt["n_collision_decode_error"])
    n_not_found_index = int(ckpt["n_not_found_index"])
    n_not_found_recording = int(ckpt["n_not_found_recording"])
    n_extract_error = int(ckpt["n_extract_error"])

    iterator = tqdm(ds, desc=f"{split_name}")
    for row in iterator:
        processed_rows += 1

        audio_path = normalize_audio_name(row["audio"])
        hf_year = int(row["year"])
        hf_hydro = row["hydrophone"]
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])

        year_label = YEAR_INT_TO_LABEL.get(hf_year)
        if year_label is None:
            n_not_found_index += 1
            continue

        key = (year_label, hf_hydro, audio_path)
        candidates = segment_index.get(key, [])

        exp_name = None
        if len(candidates) == 1:
            exp_name = candidates[0][0]
            n_unique += 1
        elif len(candidates) > 1:
            n_collision_total += 1
            collision_key = collision_cache_key(
                split_name=split_name,
                row_index=processed_rows,
                year_label=year_label,
                hydrophone=hf_hydro,
                audio_path=audio_path,
                start_time=start_time,
                end_time=end_time,
            )
            decoded_audio = decode_source_audio(row["audio"])
            if decoded_audio is None:
                n_collision_decode_error += 1
                exp_name = choose_collision_fallback(candidates)[0]
                log_collision_fallback(
                    cache_dir=cache_dir,
                    split_name=split_name,
                    row_index=processed_rows,
                    reason="decode_error",
                    audio_path=audio_path,
                    year_label=year_label,
                    hydrophone=hf_hydro,
                    start_time=start_time,
                    end_time=end_time,
                    candidates=candidates,
                    chosen_exp_name=exp_name,
                )
                n_collision_fallback += 1
            else:
                hf_audio_array, hf_sampling_rate = decoded_audio
                match_details = collision_resolver.resolve_with_details(
                    collision_key,
                    hf_audio_array,
                    hf_sampling_rate,
                    candidates,
                )
                if match_details is not None:
                    exp_name = str(match_details["exp_name"])
                    log_collision_resolved(
                        cache_dir=cache_dir,
                        split_name=split_name,
                        row_index=processed_rows,
                        audio_path=audio_path,
                        year_label=year_label,
                        hydrophone=hf_hydro,
                        start_time=start_time,
                        end_time=end_time,
                        resolution=match_details,
                    )
                    n_collision_resolved += 1
                else:
                    exp_name = choose_collision_fallback(candidates)[0]
                    log_collision_fallback(
                        cache_dir=cache_dir,
                        split_name=split_name,
                        row_index=processed_rows,
                        reason="no_audio_match",
                        audio_path=audio_path,
                        year_label=year_label,
                        hydrophone=hf_hydro,
                        start_time=start_time,
                        end_time=end_time,
                        candidates=candidates,
                        chosen_exp_name=exp_name,
                    )
                    n_collision_fallback += 1
        else:
            n_not_found_index += 1
            continue

        rec_path = recording_resolver.find(exp_name, year_label)
        if rec_path is None:
            n_not_found_recording += 1
            continue

        try:
            audio_96k = extract_segment_96k(
                rec_path,
                start_time,
                end_time,
                resample_type=resample_type,
            )
        except Exception as e:
            print(f"  Error extracting {audio_path} from {rec_path}: {e}")
            n_extract_error += 1
            continue

        shard_flushed = writer.add(
            audio=audio_96k,
            start_time=start_time,
            end_time=end_time,
            duration=float(row["duration"]),
            year=hf_year,
            hydrophone=hf_hydro,
        )
        if shard_flushed or processed_rows % checkpoint_every == 0:
            writer.flush()
            save_checkpoint(cache_dir, split_name, {
                "processed_rows": processed_rows,
                "rows_written": writer.total_rows,
                "completed": False,
                "n_unique": n_unique,
                "n_collision_total": n_collision_total,
                "n_collision_resolved": n_collision_resolved,
                "n_collision_fallback": n_collision_fallback,
                "n_collision_decode_error": n_collision_decode_error,
                "n_not_found_index": n_not_found_index,
                "n_not_found_recording": n_not_found_recording,
                "n_extract_error": n_extract_error,
            })
            recording_resolver.flush()
            collision_resolver.flush()
        maybe_sleep(processed_rows, yield_every, yield_seconds)

    writer.flush()
    save_checkpoint(cache_dir, split_name, {
        "processed_rows": processed_rows,
        "rows_written": writer.total_rows,
        "completed": True,
        "n_unique": n_unique,
        "n_collision_total": n_collision_total,
        "n_collision_resolved": n_collision_resolved,
        "n_collision_fallback": n_collision_fallback,
        "n_collision_decode_error": n_collision_decode_error,
        "n_not_found_index": n_not_found_index,
        "n_not_found_recording": n_not_found_recording,
        "n_extract_error": n_extract_error,
    })
    recording_resolver.flush()
    collision_resolver.flush()

    print(f"\nStats for {split_name}:")
    print(f"  Processed rows:       {processed_rows}")
    print(f"  Rows written:         {writer.total_rows}")
    print(f"  Unique matches:       {n_unique}")
    print(f"  Collision rows:       {n_collision_total}")
    print(f"  Collision resolved:   {n_collision_resolved}")
    print(f"  Collision fallback:   {n_collision_fallback}")
    print(f"  Collision decode err: {n_collision_decode_error}")
    print(f"  Not found in index:   {n_not_found_index}")
    print(f"  Recording not found:  {n_not_found_recording}")
    print(f"  Extraction errors:    {n_extract_error}")
    print(f"  Total rows produced:  {writer.total_rows}")

    return writer.finalize()


def main() -> None:
    args = parse_args()

    apply_runtime_tuning(args)

    if args.reset_run_state:
        args.resume = False

    if not args.save_local and not args.push and not args.keep_temp_shards:
        args.keep_temp_shards = True
        print("No --save-local/--push requested, keeping temporary shards by default.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.yield_every < 0:
        raise ValueError("--yield-every must be >= 0")
    if args.yield_seconds < 0:
        raise ValueError("--yield-seconds must be >= 0")

    cache_dir = Path(args.cache_dir)
    work_dir = Path(args.work_dir)

    if args.reset_run_state:
        reset_run_state(work_dir, cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    recording_resolver = RecordingResolver(cache_dir)
    collision_resolver = CollisionResolver(cache_dir, resample_type=args.resample_type)

    print("Building segment index...")
    segment_index = build_segment_index(cache_dir, rebuild_cache=args.rebuild_index_cache)

    if args.dry_run:
        for split_name in args.splits:
            dry_run_split(split_name, segment_index, recording_resolver)
        print("\nDry run complete.")
        return

    push_splits_immediately = args.push and not args.save_local
    split_datasets: dict[str, Dataset] = {}
    split_shards: dict[str, list[Path]] = {}
    for split_name in args.splits:
        split_ds, shard_paths = process_split(
            split_name,
            segment_index,
            work_dir,
            cache_dir,
            recording_resolver,
            collision_resolver,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            yield_every=args.yield_every,
            yield_seconds=args.yield_seconds,
            resample_type=args.resample_type,
            resume=args.resume,
        )
        split_shards[split_name] = shard_paths
        if push_splits_immediately:
            print(f"Pushing split {split_name} to {args.target_repo}...")
            split_ds.push_to_hub(
                args.target_repo,
                split=split_name,
                private=args.private,
                max_shard_size=args.max_shard_size,
            )
            print(f"Pushed split {split_name}.")
            if not args.keep_temp_shards:
                ShardWriter.cleanup(shard_paths)
                split_shards[split_name] = []
            del split_ds
            gc.collect()
            continue

        split_datasets[split_name] = split_ds

    if push_splits_immediately:
        if not args.keep_temp_shards:
            shutil.rmtree(work_dir, ignore_errors=True)
            print(f"Removed temporary shards under {work_dir}")
        print(f"Done! https://huggingface.co/datasets/{args.target_repo}")
        return

    ds_dict = DatasetDict(split_datasets)
    print(f"\nFinal dataset: {ds_dict}")

    if args.save_local:
        print(f"Saving locally to {args.save_local}...")
        ds_dict.save_to_disk(args.save_local)
        print("Local save done.")

    if args.push:
        print(f"Pushing to {args.target_repo}...")
        ds_dict.push_to_hub(
            args.target_repo,
            private=args.private,
            max_shard_size=args.max_shard_size,
        )
        print(f"Done! https://huggingface.co/datasets/{args.target_repo}")

    if not args.keep_temp_shards:
        for shard_paths in split_shards.values():
            ShardWriter.cleanup(shard_paths)
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Removed temporary shards under {work_dir}")


if __name__ == "__main__":
    main()
