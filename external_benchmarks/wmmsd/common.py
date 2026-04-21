from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac")
DEFAULT_TARGET_SPECIES = ("Bottlenose_Dolphin", "Common_Dolphin")

_SPECIES_ALIASES = {
    "Atlantic_Spotted_Dolphin": ("Atlantic_Spotted_Dolphin", "atlantic spotted dolphin"),
    "Bearded_Seal": ("Bearded_Seal", "bearded seal"),
    "Beluga,_White_Whale": ("Beluga,_White_Whale", "beluga", "white whale"),
    "Bottlenose_Dolphin": (
        "Bottlenose_Dolphin",
        "bottlenose dolphin",
        "tursiops truncatus",
    ),
    "Bowhead_Whale": ("Bowhead_Whale", "bowhead whale"),
    "Clymene_Dolphin": ("Clymene_Dolphin", "clymene dolphin"),
    "Common_Dolphin": ("Common_Dolphin", "common dolphin", "delphinus delphis"),
    "False_Killer_Whale": ("False_Killer_Whale", "false killer whale"),
    "Fin,_Finback_Whale": ("Fin,_Finback_Whale", "fin whale", "finback whale"),
    "Frasers_Dolphin": ("Frasers_Dolphin", "frasers dolphin", "fraser's dolphin"),
    "Grampus,_Rissos_Dolphin": (
        "Grampus,_Rissos_Dolphin",
        "grampus",
        "rissos dolphin",
        "risso's dolphin",
    ),
    "Harp_Seal": ("Harp_Seal", "harp seal"),
    "Humpback_Whale": ("Humpback_Whale", "humpback whale"),
    "Killer_Whale": ("Killer_Whale", "killer whale", "orca"),
    "Leopard_Seal": ("Leopard_Seal", "leopard seal"),
    "Long-Finned_Pilot_Whale": ("Long-Finned_Pilot_Whale", "long-finned pilot whale"),
    "Melon_Headed_Whale": ("Melon_Headed_Whale", "melon headed whale", "melon-headed whale"),
    "Minke_Whale": ("Minke_Whale", "minke whale"),
    "Narwhal": ("Narwhal", "narwhal"),
    "Northern_Right_Whale": ("Northern_Right_Whale", "northern right whale"),
    "Pantropical_Spotted_Dolphin": (
        "Pantropical_Spotted_Dolphin",
        "pantropical spotted dolphin",
    ),
    "Ross_Seal": ("Ross_Seal", "ross seal"),
    "Rough-Toothed_Dolphin": ("Rough-Toothed_Dolphin", "rough-toothed dolphin"),
    "Short-Finned_Pacific_Pilot_Whale": (
        "Short-Finned_Pacific_Pilot_Whale",
        "short-finned pacific pilot whale",
        "short-finned pilot whale",
    ),
    "Southern_Right_Whale": ("Southern_Right_Whale", "southern right whale"),
    "Sperm_Whale": ("Sperm_Whale", "sperm whale"),
    "Spinner_Dolphin": ("Spinner_Dolphin", "spinner dolphin"),
    "Striped_Dolphin": ("Striped_Dolphin", "striped dolphin"),
    "Walrus": ("Walrus", "walrus"),
    "Weddell_Seal": ("Weddell_Seal", "weddell seal"),
    "White-beaked_Dolphin": ("White-beaked_Dolphin", "white-beaked dolphin"),
    "White-sided_Dolphin": ("White-sided_Dolphin", "white-sided dolphin"),
}

NORMALIZED_ALIAS_TO_SPECIES = {
    re.sub(r"[^a-z0-9]+", "", alias.lower()): canonical
    for canonical, aliases in _SPECIES_ALIASES.items()
    for alias in aliases
}


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonicalize_species(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    normalized = normalize_token(value)
    canonical = NORMALIZED_ALIAS_TO_SPECIES.get(normalized)
    if canonical:
        return canonical
    return value.replace(" ", "_")


def infer_species_from_path(path: Path) -> str:
    candidates = [path.as_posix(), path.stem, *path.parts]
    for candidate in candidates:
        canonical = NORMALIZED_ALIAS_TO_SPECIES.get(normalize_token(candidate))
        if canonical:
            return canonical
    return ""


def safe_slug(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    if not slug:
        slug = "item"
    return slug[:max_length]


def stage_file_name(index: int, species: str, source_rel_path: str, suffix: str) -> str:
    digest = hashlib.sha1(source_rel_path.encode("utf-8")).hexdigest()[:10]
    stem = safe_slug(Path(source_rel_path).stem, max_length=48)
    species_slug = safe_slug(species or "Unknown", max_length=40)
    return f"{index:05d}__{species_slug}__{stem}__{digest}{suffix.lower()}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def materialize_audio_file(source_path: Path, destination_path: Path, mode: str) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()

    source_path = source_path.expanduser().resolve()
    if mode == "symlink":
        destination_path.symlink_to(source_path)
        return
    if mode == "hardlink":
        os.link(source_path, destination_path)
        return
    if mode == "copy":
        shutil.copy2(source_path, destination_path)
        return
    raise ValueError(f"Unsupported materialization mode: {mode}")


def write_audio_bytes(destination_path: Path, payload: bytes) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    destination_path.write_bytes(payload)


def write_csv(path: Path, rows: list[dict[str, object]], field_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
