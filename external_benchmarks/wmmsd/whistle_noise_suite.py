from __future__ import annotations

from itertools import combinations

from external_benchmarks.wmmsd.ultimate_suite import (
    DEFAULT_POSITIVE_SPECIES,
    NON_ODONTOCETES,
    OTHER_DELPHINIDS,
)


ALL_DELPHINIDS: tuple[str, ...] = DEFAULT_POSITIVE_SPECIES + OTHER_DELPHINIDS

OPTIONAL_AMBIGUOUS_POSITIVES: tuple[str, ...] = (
    "Beluga,_White_Whale",
    "Narwhal",
    "Sperm_Whale",
)

OPTIONAL_AMBIGUOUS_LABELS = {
    "Beluga,_White_Whale": "Beluga",
    "Narwhal": "Narwhal",
    "Sperm_Whale": "Sperm_Whale",
}

OPTIONAL_AMBIGUOUS_SLUGS = {
    "Beluga,_White_Whale": "beluga",
    "Narwhal": "narwhal",
    "Sperm_Whale": "sperm_whale",
}


def _combo_display(species_list: tuple[str, ...]) -> str:
    return ", ".join(OPTIONAL_AMBIGUOUS_LABELS[species] for species in species_list)


def _build_proxy_scenarios() -> tuple[dict[str, object], ...]:
    scenarios: list[dict[str, object]] = [
        {
            "name": "proxy_delphinids_vs_clear_noise",
            "difficulty": "proxy_core",
            "description": (
                "Proxy whistle/noise benchmark. Positives = all delphinids in WMMSD. "
                "Negatives = pinnipeds and baleen whales only. Beluga, Narwhal and "
                "Sperm_Whale are excluded because species labels alone do not tell us "
                "whether the clip contains a whistle."
            ),
            "positive_species": ALL_DELPHINIDS,
            "negative_species": NON_ODONTOCETES,
            "optional_ambiguous_included": (),
            "optional_ambiguous_excluded": OPTIONAL_AMBIGUOUS_POSITIVES,
        }
    ]

    for combo_size in range(1, len(OPTIONAL_AMBIGUOUS_POSITIVES) + 1):
        for combo in combinations(OPTIONAL_AMBIGUOUS_POSITIVES, combo_size):
            included = tuple(combo)
            excluded = tuple(
                species
                for species in OPTIONAL_AMBIGUOUS_POSITIVES
                if species not in included
            )
            scenario_slug = "_".join(
                OPTIONAL_AMBIGUOUS_SLUGS[species] for species in included
            )
            included_display = _combo_display(included)
            if excluded:
                description = (
                    "Proxy whistle/noise benchmark. Positives = all delphinids in WMMSD "
                    f"+ {included_display}. Negatives = pinnipeds and baleen whales only. "
                    f"{_combo_display(excluded)} remain excluded because species labels "
                    "alone do not tell us whether the clip contains a whistle."
                )
            else:
                description = (
                    "Broadest proxy whistle/noise benchmark. Positives = all delphinids "
                    f"in WMMSD + {included_display}. Negatives = pinnipeds and baleen "
                    "whales only."
                )
            scenarios.append(
                {
                    "name": f"proxy_delphinids_plus_{scenario_slug}_vs_clear_noise",
                    "difficulty": f"proxy_plus_{combo_size}",
                    "description": description,
                    "positive_species": ALL_DELPHINIDS + included,
                    "negative_species": NON_ODONTOCETES,
                    "optional_ambiguous_included": included,
                    "optional_ambiguous_excluded": excluded,
                }
            )

    return tuple(scenarios)


WHISTLE_NOISE_PROXY_SCENARIOS: tuple[dict[str, object], ...] = _build_proxy_scenarios()


def suite_species() -> list[str]:
    selected = set()
    for scenario in WHISTLE_NOISE_PROXY_SCENARIOS:
        selected.update(str(species) for species in scenario["positive_species"])
        selected.update(str(species) for species in scenario["negative_species"])
    return sorted(selected)
