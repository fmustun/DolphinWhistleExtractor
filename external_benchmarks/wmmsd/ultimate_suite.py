from __future__ import annotations

from external_benchmarks.wmmsd.common import _SPECIES_ALIASES


DEFAULT_POSITIVE_SPECIES: tuple[str, ...] = (
    "Bottlenose_Dolphin",
    "Common_Dolphin",
)

OTHER_DELPHINIDS: tuple[str, ...] = (
    "Atlantic_Spotted_Dolphin",
    "Clymene_Dolphin",
    "False_Killer_Whale",
    "Frasers_Dolphin",
    "Grampus,_Rissos_Dolphin",
    "Killer_Whale",
    "Long-Finned_Pilot_Whale",
    "Melon_Headed_Whale",
    "Pantropical_Spotted_Dolphin",
    "Rough-Toothed_Dolphin",
    "Short-Finned_Pacific_Pilot_Whale",
    "Spinner_Dolphin",
    "Striped_Dolphin",
    "White-beaked_Dolphin",
    "White-sided_Dolphin",
)

OTHER_ODONTOCETES_NON_DELPHINID: tuple[str, ...] = (
    "Beluga,_White_Whale",
    "Narwhal",
    "Sperm_Whale",
)

NON_ODONTOCETES: tuple[str, ...] = (
    "Bearded_Seal",
    "Bowhead_Whale",
    "Fin,_Finback_Whale",
    "Harp_Seal",
    "Humpback_Whale",
    "Leopard_Seal",
    "Minke_Whale",
    "Northern_Right_Whale",
    "Ross_Seal",
    "Southern_Right_Whale",
    "Walrus",
    "Weddell_Seal",
)

ALL_WMMSD_SPECIES: tuple[str, ...] = tuple(sorted(_SPECIES_ALIASES))
ALL_NON_TARGET_SPECIES: tuple[str, ...] = tuple(
    species
    for species in ALL_WMMSD_SPECIES
    if species not in set(DEFAULT_POSITIVE_SPECIES)
)

ULTIMATE_SCENARIOS: tuple[dict[str, object], ...] = (
    {
        "name": "easy_non_odontocetes",
        "difficulty": "easy",
        "description": (
            "Positives = bottlenose + common dolphin. Negatives = pinnipeds and baleen "
            "whales only. This is the easy out-of-family benchmark."
        ),
        "positive_species": DEFAULT_POSITIVE_SPECIES,
        "negative_species": NON_ODONTOCETES,
    },
    {
        "name": "hard_other_odontocetes",
        "difficulty": "hard",
        "description": (
            "Positives = bottlenose + common dolphin. Negatives = all other toothed "
            "whales and dolphins. This is the main hard cross-taxa benchmark."
        ),
        "positive_species": DEFAULT_POSITIVE_SPECIES,
        "negative_species": OTHER_DELPHINIDS + OTHER_ODONTOCETES_NON_DELPHINID,
    },
    {
        "name": "hard_other_delphinids",
        "difficulty": "hardest",
        "description": (
            "Positives = bottlenose + common dolphin. Negatives = other delphinids only. "
            "This is the nearest-neighbour benchmark and the hardest one to beat."
        ),
        "positive_species": DEFAULT_POSITIVE_SPECIES,
        "negative_species": OTHER_DELPHINIDS,
    },
    {
        "name": "ultimate_all_non_target",
        "difficulty": "ultimate",
        "description": (
            "Positives = bottlenose + common dolphin. Negatives = every other WMMSD "
            "species. This is the broad end-to-end benchmark."
        ),
        "positive_species": DEFAULT_POSITIVE_SPECIES,
        "negative_species": ALL_NON_TARGET_SPECIES,
    },
)


def suite_species() -> list[str]:
    selected = set(DEFAULT_POSITIVE_SPECIES)
    for scenario in ULTIMATE_SCENARIOS:
        selected.update(str(species) for species in scenario["negative_species"])
    return sorted(selected)
