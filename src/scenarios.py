"""Shared scenario definitions for controller comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    occupant_enabled: bool
    dynamic_comfort_enabled: bool
    peak_flattening_enabled: bool
    peak_weight: float = 0.0


SCENARIOS: dict[str, ScenarioConfig] = {
    "no_occupant": ScenarioConfig(
        name="no_occupant",
        occupant_enabled=False,
        dynamic_comfort_enabled=False,
        peak_flattening_enabled=False,
    ),
    "occupant_present": ScenarioConfig(
        name="occupant_present",
        occupant_enabled=True,
        dynamic_comfort_enabled=True,
        peak_flattening_enabled=False,
    ),
    "occupant_present_peak_flattening": ScenarioConfig(
        name="occupant_present_peak_flattening",
        occupant_enabled=True,
        dynamic_comfort_enabled=True,
        peak_flattening_enabled=True,
        peak_weight=0.1,
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    try:
        return SCENARIOS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Valid scenarios: {valid}") from exc

