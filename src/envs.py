"""CityLearn environment construction helpers.

Keep notebook-specific paths and building choices out of controller code by
passing them through this small factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citylearn.citylearn import CityLearnEnv


@dataclass(frozen=True)
class EnvConfig:
    schema_path: Path
    dataset_dir: Path
    buildings: list[str]
    start_time_step: int
    end_time_step: int
    random_seed: int = 49
    central_agent: bool = False
    active_actions: tuple[str, ...] = ("heating_device",)


def make_citylearn_env(config: EnvConfig, **overrides: Any) -> CityLearnEnv:
    """Create a CityLearn environment from a reproducible config object."""

    kwargs: dict[str, Any] = {
        "schema": str(config.schema_path),
        "root_directory": str(config.dataset_dir),
        "central_agent": config.central_agent,
        "buildings": config.buildings,
        "simulation_start_time_step": config.start_time_step,
        "simulation_end_time_step": config.end_time_step,
        "random_seed": config.random_seed,
        "active_actions": list(config.active_actions),
    }
    kwargs.update(overrides)
    return CityLearnEnv(**kwargs)

