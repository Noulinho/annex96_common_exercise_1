#!/usr/bin/env python
"""Fit the notebook's one-state RC model from a saved rollout dataset.

Example:
    python scripts/01_fit_rc_model.py \
        --dataset notebooks/january_pi_dataset.npz \
        --output results/models/rc/1state_rc_247942_january.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


OBSERVATION_NAMES = {
    "indoor": "indoor_dry_bulb_temperature",
    "outdoor": "outdoor_dry_bulb_temperature",
    "solar": "direct_solar_irradiance",
    "heating_demand": "heating_demand",
    "heating_power": "heating_electricity_consumption",
}

NOTEBOOK_MPC_OBSERVATIONS = [
    "month",
    "hour",
    "outdoor_dry_bulb_temperature",
    "direct_solar_irradiance",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "solar_generation",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point",
    "comfort_band",
    "indoor_dry_bulb_temperature_cooling_delta",
    "indoor_dry_bulb_temperature_heating_delta",
    "hvac_mode",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "cooling_electricity_consumption",
    "heating_electricity_consumption",
    "dhw_electricity_consumption",
    "electrical_storage_electricity_consumption",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the same one-state RC/COP model used in notebooks/my_mpc.ipynb. "
            "Observation indices are resolved automatically from the CityLearn environment."
        )
    )
    parser.add_argument("--dataset", default="notebooks/january_pi_dataset.npz")
    parser.add_argument("--output", default="results/models/rc/1state_rc_247942_january.json")
    parser.add_argument("--building-config", default="configs/building.yaml")
    parser.add_argument(
        "--observation-preset",
        default="auto",
        choices=["auto", "notebook_mpc_28", "env"],
        help=(
            "Observation-name source. 'auto' uses the 28-column my_mpc notebook "
            "layout when the dataset has 28 observations, otherwise falls back to env."
        ),
    )
    return parser


def _parse_simple_building_config(path: str | Path) -> dict:
    """Parse the small building YAML without adding a runtime YAML dependency."""

    path = Path(path)
    config: dict[str, object] = {}
    current_section = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        if not raw_line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1]
            config[current_section] = {}
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        parsed_value: object = value
        if value.isdigit():
            parsed_value = int(value)

        if raw_line.startswith(" ") and current_section is not None:
            section = config[current_section]
            assert isinstance(section, dict)
            section[key] = parsed_value
        else:
            current_section = None
            config[key] = parsed_value

    return config


def _env_config_from_file(path: str | Path) -> EnvConfig:
    from src.envs import EnvConfig

    config = _parse_simple_building_config(path)
    train_period = config["train_period"]
    assert isinstance(train_period, dict)

    return EnvConfig(
        schema_path=Path(str(config["schema_path"])),
        dataset_dir=Path(str(config["dataset_dir"])),
        buildings=[str(config["building"])],
        start_time_step=int(train_period["start_time_step"]),
        end_time_step=int(train_period["end_time_step"]),
        random_seed=int(config.get("random_seed", 49)),
    )


def _indices_from_names(names: list[str]) -> dict[str, int]:
    missing = [name for name in OBSERVATION_NAMES.values() if name not in names]
    if missing:
        available = "\n".join(f"  {i:>2}: {name}" for i, name in enumerate(names))
        raise RuntimeError(
            "Missing required observations for the notebook RC fit: "
            f"{missing}\nAvailable observations:\n{available}"
        )

    return {key: names.index(name) for key, name in OBSERVATION_NAMES.items()}


def _observation_names_from_env(building_config: str | Path) -> list[str]:
    from src.envs import make_citylearn_env

    env_config = _env_config_from_file(building_config)
    env = make_citylearn_env(env_config)
    env.reset()
    return list(env.unwrapped.buildings[0].active_observations)


def _observation_names_for_dataset(obs_dim: int, preset: str, building_config: str | Path) -> tuple[str, list[str]]:
    if preset == "notebook_mpc_28":
        return "notebook_mpc_28", NOTEBOOK_MPC_OBSERVATIONS

    if preset == "env":
        return "env", _observation_names_from_env(building_config)

    if obs_dim == len(NOTEBOOK_MPC_OBSERVATIONS):
        return "notebook_mpc_28", NOTEBOOK_MPC_OBSERVATIONS

    return "env", _observation_names_from_env(building_config)


def main() -> None:
    args = build_parser().parse_args()
    from src.models.fit_rc import fit_1r1c_from_arrays, save_rc_model, squeeze_citylearn_dataset

    x, u, xp = squeeze_citylearn_dataset(args.dataset)
    source, obs_names = _observation_names_for_dataset(
        x.shape[-1],
        args.observation_preset,
        args.building_config,
    )
    if len(obs_names) != x.shape[-1]:
        raise RuntimeError(
            f"Observation-name source '{source}' has {len(obs_names)} names, "
            f"but dataset has {x.shape[-1]} observation columns."
        )
    indices = _indices_from_names(obs_names)

    print(f"Observation source: {source}")
    print("Resolved observation indices")
    for key, idx in indices.items():
        print(f"  {key:>14}: {idx:>2} ({OBSERVATION_NAMES[key]})")

    model = fit_1r1c_from_arrays(
        x,
        u,
        xp,
        indoor_idx=indices["indoor"],
        outdoor_idx=indices["outdoor"],
        solar_idx=indices["solar"],
        heating_demand_idx=indices["heating_demand"],
        heating_power_idx=indices["heating_power"],
    )
    save_rc_model(model, args.output)

    print("RC parameters")
    for key, value in model.params.to_dict().items():
        print(f"  {key:>26}: {value:.8g}")
    print(f"Saved one-state RC model to {args.output}")


if __name__ == "__main__":
    main()
