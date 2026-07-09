#!/usr/bin/env python
"""Fit the notebook's one-state RC model from a saved rollout dataset.

Example:
    python scripts/01_fit_rc_model.py \
        --dataset notebooks/january_pi_dataset.npz \
        --output results/models/rc/1state_rc_247942_january.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import (
    DATASET_NAME,
    building_metadata,
    dataset_dir,
    format_building_path,
    resolve_building_entries,
    schema_path as dataset_schema_path,
)


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
    parser.add_argument(
        "--building",
        default="all",
        help="Building selector: all, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Single dataset path or path template. When fitting multiple buildings, use placeholders "
            "such as {building_id} or prefer --dataset-template."
        ),
    )
    parser.add_argument(
        "--dataset-template",
        default="results/datasets/january_pi/{building_id}/january_pi_dataset.npz",
        help="Per-building dataset path template used when --dataset is not supplied.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Single RC model output path or path template. When fitting multiple buildings, use placeholders "
            "such as {short_name} or prefer --output-template."
        ),
    )
    parser.add_argument(
        "--output-template",
        default="results/models/rc/1state_rc_{short_name}_january.json",
        help="Per-building RC model path template used when --output is not supplied.",
    )
    parser.add_argument(
        "--model-type",
        default="1r1c",
        choices=["1r1c", "2r2c"],
        help="RC model variant to fit.",
    )
    parser.add_argument(
        "--fit-method",
        default="ridge",
        choices=["ridge", "bounded_robust"],
        help="Fitting method. For 1R1C: 'ridge' (default) or 'bounded_robust'. For 2R2C: 'bounded_robust' adds physical bounds on b_sol/b_heat/b0.",
    )
    parser.add_argument(
        "--huber-scale",
        type=float,
        default=1.0,
        help="Soft-L1 residual scale for --fit-method bounded_robust.",
    )
    parser.add_argument(
        "--output-template-2r2c",
        default="results/models/rc/2r2c_{short_name}_january.json",
        help="Per-building 2R2C model path template (used when --model-type 2r2c).",
    )
    parser.add_argument("--summary", default="results/models/rc/rc_fit_summary.csv")
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


def _has_format_fields(path: str | Path) -> bool:
    text = str(path)
    return "{" in text and "}" in text


def _resolve_path_argument(
    *,
    explicit: str | Path | None,
    template: str | Path,
    building: dict,
    multiple_buildings: bool,
    label: str,
) -> Path:
    value = template if explicit is None else explicit
    if multiple_buildings and explicit is not None and not _has_format_fields(explicit):
        raise ValueError(
            f"--{label} points to one fixed path while --building selects multiple buildings. "
            f"Use --{label}-template or include a placeholder such as {{building_id}}."
        )
    return format_building_path(value, building)


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


def _env_config_for_building(building_name: str, seed: int = 49) -> EnvConfig:
    from src.envs import EnvConfig

    return EnvConfig(
        schema_path=dataset_schema_path(DATASET_NAME),
        dataset_dir=dataset_dir(DATASET_NAME),
        buildings=[building_name],
        start_time_step=0,
        end_time_step=743,
        random_seed=int(seed),
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


def _observation_names_from_env(
    building_config: str | Path,
    *,
    building_name: str | None = None,
    seed: int = 49,
) -> list[str]:
    from src.envs import make_citylearn_env

    env_config = (
        _env_config_from_file(building_config)
        if building_name is None
        else _env_config_for_building(building_name, seed=seed)
    )
    env = make_citylearn_env(env_config)
    env.reset()
    return list(env.unwrapped.buildings[0].active_observations)


def _observation_names_for_dataset(
    obs_dim: int,
    preset: str,
    building_config: str | Path,
    *,
    building_name: str | None = None,
    seed: int = 49,
) -> tuple[str, list[str]]:
    if preset == "notebook_mpc_28":
        return "notebook_mpc_28", NOTEBOOK_MPC_OBSERVATIONS

    if preset == "env":
        return "env", _observation_names_from_env(building_config, building_name=building_name, seed=seed)

    if obs_dim == len(NOTEBOOK_MPC_OBSERVATIONS):
        return "notebook_mpc_28", NOTEBOOK_MPC_OBSERVATIONS

    return "env", _observation_names_from_env(building_config, building_name=building_name, seed=seed)


def _fit_one(building: dict, args: argparse.Namespace, *, multiple_buildings: bool) -> dict:
    from src.models.fit_rc import (
        evaluate_1r1c_fit_from_arrays,
        evaluate_2r2c_fit_from_arrays,
        evaluate_2r2c_kalman_fit_from_arrays,
        fit_1r1c_bounded_robust_from_arrays,
        fit_1r1c_from_arrays,
        fit_2r2c_from_arrays,
        save_rc_model,
        squeeze_citylearn_dataset,
    )
    from src.models.rc_2r2c import save_2r2c_model

    model_type = getattr(args, "model_type", "1r1c")

    dataset_path = _resolve_path_argument(
        explicit=args.dataset,
        template=args.dataset_template,
        building=building,
        multiple_buildings=multiple_buildings,
        label="dataset",
    )
    if model_type == "2r2c":
        output_template = getattr(args, "output_template_2r2c", "results/models/rc/2r2c_{short_name}_january.json")
        output_path = _resolve_path_argument(
            explicit=args.output,
            template=output_template,
            building=building,
            multiple_buildings=multiple_buildings,
            label="output",
        )
    else:
        output_path = _resolve_path_argument(
            explicit=args.output,
            template=args.output_template,
            building=building,
            multiple_buildings=multiple_buildings,
            label="output",
        )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing January PI dataset for {building['display']}: {dataset_path}. "
            "Run scripts/00_collect_january_pi_data.py first."
        )

    seed = int(49 + 1000 * (int(building["index"]) - 1))
    x, u, xp, dones = squeeze_citylearn_dataset(dataset_path, return_dones=True)
    source, obs_names = _observation_names_for_dataset(
        x.shape[-1],
        args.observation_preset,
        args.building_config,
        building_name=str(building["name"]),
        seed=seed,
    )
    if len(obs_names) != x.shape[-1]:
        raise RuntimeError(
            f"Observation-name source '{source}' has {len(obs_names)} names, "
            f"but dataset has {x.shape[-1]} observation columns."
        )
    indices = _indices_from_names(obs_names)

    print(f"\n=== {building['display']}: {building['name']} ===")
    print(f"Dataset: {dataset_path}")
    print(f"Observation source: {source}")
    print("Resolved observation indices")
    for key, idx in indices.items():
        print(f"  {key:>14}: {idx:>2} ({OBSERVATION_NAMES[key]})")

    fit_kwargs = dict(
        indoor_idx=indices["indoor"],
        outdoor_idx=indices["outdoor"],
        solar_idx=indices["solar"],
        heating_demand_idx=indices["heating_demand"],
        heating_power_idx=indices["heating_power"],
    )

    if model_type == "2r2c":
        model = fit_2r2c_from_arrays(x, u, xp, **fit_kwargs, dones=dones,
                                      bounded=(args.fit_method == "bounded_robust"))
        save_2r2c_model(model, output_path)
        fit_metrics = evaluate_2r2c_fit_from_arrays(
            model,
            x,
            u,
            xp,
            indoor_idx=indices["indoor"],
            outdoor_idx=indices["outdoor"],
            solar_idx=indices["solar"],
            dones=dones,
        )
        fit_metrics.update(
            evaluate_2r2c_kalman_fit_from_arrays(
                model,
                x,
                u,
                xp,
                indoor_idx=indices["indoor"],
                outdoor_idx=indices["outdoor"],
                solar_idx=indices["solar"],
                dones=dones,
            )
        )
        p = model.params
        print("2R2C parameters")
        for key, value in p.to_dict().items():
            print(f"  {key:>26}: {value:.8g}")
        print(f"  tau_mass (h): {p.tau_mass_hours:.2f}")
        print(f"  tau_air_approx (h): {p.tau_air_approx_hours:.2f}")
        print("Fit metrics")
        for key, value in fit_metrics.items():
            print(f"  {key:>26}: {value:.6g}")
        print(f"Saved two-state RC model to {output_path}")
    else:
        if args.fit_method == "bounded_robust":
            model = fit_1r1c_bounded_robust_from_arrays(x, u, xp, **fit_kwargs, huber_scale=args.huber_scale)
        else:
            model = fit_1r1c_from_arrays(x, u, xp, **fit_kwargs)
        save_rc_model(model, output_path)
        fit_metrics = evaluate_1r1c_fit_from_arrays(
            model,
            x,
            u,
            xp,
            indoor_idx=indices["indoor"],
            outdoor_idx=indices["outdoor"],
            solar_idx=indices["solar"],
            dones=dones,
        )
        print("RC parameters")
        for key, value in model.params.to_dict().items():
            print(f"  {key:>26}: {value:.8g}")
        print("Fit metrics")
        for key, value in fit_metrics.items():
            print(f"  {key:>26}: {value:.6g}")
        print(f"Saved one-state RC model to {output_path}")

    return {
        **building_metadata(building),
        "dataset_path": str(dataset_path),
        "rc_model_path": str(output_path),
        "model_type": model_type,
        "fit_method": str(
            args.fit_method if model_type == "1r1c"
            else ("trajectory_minimize_bounded_robust" if args.fit_method == "bounded_robust"
                  else "trajectory_minimize_fixed")
        ),
        "observation_source": source,
        **model.params.to_dict(),
        **fit_metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    multiple_buildings = len(buildings) > 1
    rows = [_fit_one(building, args, multiple_buildings=multiple_buildings) for building in buildings]

    if rows:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved RC fit summary to {summary_path}")


if __name__ == "__main__":
    main()
