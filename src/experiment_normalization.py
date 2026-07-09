"""Normalization helpers for controller comparison experiments."""

from __future__ import annotations

import csv
import json
import numpy as np

from src.buildings import DATASET_NAME, dataset_dir, schema_path


def robust_scale(values, *, mode: str = "std", floor: float = 1e-6) -> float:
    """Return a positive scale for normalizing one physical quantity.

    The default is the usual standard-deviation scale used by z-scoring. Robust
    alternatives are useful when a profile has a few large outliers.
    """

    data = np.asarray(values, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float(max(floor, 1e-6))

    mode = str(mode).strip().lower()
    if mode in {"std", "zscore", "z_score"}:
        scale = float(np.nanstd(data))
    elif mode in {"iqr", "robust_std"}:
        q25, q75 = np.nanpercentile(data, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
    elif mode in {"range", "p95_p5"}:
        q05, q95 = np.nanpercentile(data, [5.0, 95.0])
        scale = float(q95 - q05)
    elif mode in {"minmax", "min_max", "data_range", "full_range"}:
        scale = float(np.nanmax(data) - np.nanmin(data))
    elif mode in {"maxabs", "max_abs"}:
        scale = float(np.nanmax(np.abs(data)))
    else:
        raise ValueError("mode must be one of: std, iqr, range, minmax, maxabs.")

    return float(max(scale, floor, 1e-6))


def _window(values, start: int, end: int) -> np.ndarray:
    data = np.asarray(values, dtype=float).reshape(-1)
    n = int(end) - int(start) + 1
    if data.size >= int(end) + 1:
        return data[int(start) : int(end) + 1]
    return data[:n]


def _full_dataset_temperature_series(
    *,
    building_name: str,
    dataset_name: str,
    columns: tuple[str, ...],
) -> np.ndarray:
    """Read full-year temperature columns directly from the CE1 CSV.

    CityLearn slices time series when a simulation window is used. For min-max
    scaling, use the full dataset CSV so the denominator is a real data range,
    not just a January/FEBruary standard deviation or a hand-tuned divisor.
    """

    schema = json.loads(schema_path(dataset_name).read_text())
    building_cfg = schema.get("buildings", {}).get(building_name)
    if building_cfg is None:
        available = ", ".join(schema.get("buildings", {}).keys())
        raise ValueError(f"Unknown building '{building_name}'. Available buildings: {available}")

    csv_path = dataset_dir(dataset_name) / building_cfg["energy_simulation"]
    values: list[float] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in columns:
                raw = row.get(column)
                if raw is None or raw == "":
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue

    return np.asarray(values, dtype=float)


def _observation_temperature_range(
    *,
    building_name: str,
    simulation_start: int,
    simulation_end: int,
    dataset_name: str,
) -> float:
    """Return CityLearn observation-space min-max range for indoor/setpoint T."""

    from citylearn.citylearn import CityLearnEnv

    env = CityLearnEnv(
        str(schema_path(dataset_name)),
        root_directory=str(dataset_dir(dataset_name)),
        central_agent=False,
        simulation_start_time_step=int(simulation_start),
        simulation_end_time_step=int(simulation_end),
    )

    names = [b.name for b in env.buildings]
    if building_name not in names:
        available = ", ".join(names)
        raise ValueError(f"Unknown building '{building_name}'. Available buildings: {available}")
    idx = names.index(building_name)
    obs_names = env.observation_names[idx]
    obs_space = env.observation_space[idx]
    columns = [
        "indoor_dry_bulb_temperature",
        "indoor_dry_bulb_temperature_heating_set_point",
    ]
    lows = []
    highs = []
    for column in columns:
        if column not in obs_names:
            continue
        j = obs_names.index(column)
        lows.append(float(obs_space.low[j]))
        highs.append(float(obs_space.high[j]))
    if not lows or not highs:
        raise ValueError("Could not find indoor/setpoint temperature bounds in observation space.")
    return float(max(highs) - min(lows))


def temperature_error_scale(
    *,
    building_name: str,
    simulation_start: int = 0,
    simulation_end: int = 743,
    dataset_name: str = DATASET_NAME,
    mode: str = "setpoint_std",
    floor: float = 0.25,
) -> float:
    """Scale the setpoint-tracking error from baseline data.

    This gives the comfort term a physical normalized-error interpretation:
    ``((T_in - T_sp) / scale_T)^2``. Modes prefixed by ``error_`` use the
    historical tracking error; modes prefixed by ``setpoint_`` or ``indoor_``
    use historical temperature-profile variability. Modes ending in
    ``_minmax`` use full-dataset max-minus-min ranges.
    """

    mode = str(mode).strip().lower()
    if mode in {"temperature_minmax", "dataset_minmax", "full_dataset_minmax"}:
        source = _full_dataset_temperature_series(
            building_name=building_name,
            dataset_name=dataset_name,
            columns=(
                "indoor_dry_bulb_temperature",
                "indoor_dry_bulb_temperature_heating_set_point",
            ),
        )
        return robust_scale(source, mode="minmax", floor=floor)
    if mode in {"observation_minmax", "observation_space_minmax", "obs_minmax"}:
        scale = _observation_temperature_range(
            building_name=building_name,
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            dataset_name=dataset_name,
        )
        return robust_scale([0.0, scale], mode="minmax", floor=floor)
    if mode in {"setpoint_minmax", "heating_setpoint_minmax"}:
        source = _full_dataset_temperature_series(
            building_name=building_name,
            dataset_name=dataset_name,
            columns=("indoor_dry_bulb_temperature_heating_set_point",),
        )
        return robust_scale(source, mode="minmax", floor=floor)
    if mode in {"indoor_minmax", "indoor_temperature_minmax"}:
        source = _full_dataset_temperature_series(
            building_name=building_name,
            dataset_name=dataset_name,
            columns=("indoor_dry_bulb_temperature",),
        )
        return robust_scale(source, mode="minmax", floor=floor)

    from citylearn.citylearn import CityLearnEnv

    env = CityLearnEnv(
        str(schema_path(dataset_name)),
        root_directory=str(dataset_dir(dataset_name)),
        central_agent=True,
        simulation_start_time_step=int(simulation_start),
        simulation_end_time_step=int(simulation_end),
    )
    env.reset()

    names = [b.name for b in env.buildings]
    if building_name not in names:
        available = ", ".join(names)
        raise ValueError(f"Unknown building '{building_name}'. Available buildings: {available}")

    building = env.buildings[names.index(building_name)]
    indoor = _window(
        building.energy_simulation.indoor_dry_bulb_temperature,
        simulation_start,
        simulation_end,
    )
    setpoint = _window(
        building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control,
        simulation_start,
        simulation_end,
    )
    n = min(len(indoor), len(setpoint))
    if mode in {"std", "zscore", "z_score"}:
        source = indoor[:n] - setpoint[:n]
        scale_mode = "std"
    elif mode in {"error_std", "error_zscore"}:
        source = indoor[:n] - setpoint[:n]
        scale_mode = "std"
    elif mode in {"error_iqr", "error_robust_std"}:
        source = indoor[:n] - setpoint[:n]
        scale_mode = "iqr"
    elif mode in {"error_range", "error_p95_p5"}:
        source = indoor[:n] - setpoint[:n]
        scale_mode = "range"
    elif mode in {"setpoint_std", "setpoint_zscore"}:
        source = setpoint[:n]
        scale_mode = "std"
    elif mode in {"setpoint_iqr", "setpoint_robust_std"}:
        source = setpoint[:n]
        scale_mode = "iqr"
    elif mode in {"setpoint_range", "setpoint_p95_p5"}:
        source = setpoint[:n]
        scale_mode = "range"
    elif mode in {"indoor_std", "indoor_zscore"}:
        source = indoor[:n]
        scale_mode = "std"
    elif mode in {"indoor_iqr", "indoor_robust_std"}:
        source = indoor[:n]
        scale_mode = "iqr"
    elif mode in {"indoor_range", "indoor_p95_p5"}:
        source = indoor[:n]
        scale_mode = "range"
    elif mode in {"error_minmax"}:
        source = indoor[:n] - setpoint[:n]
        scale_mode = "minmax"
    else:
        raise ValueError(
            "mode must be one of: error_std/error_iqr/error_range, "
            "setpoint_std/setpoint_iqr/setpoint_range, indoor_std/indoor_iqr/indoor_range, "
            "temperature_minmax/dataset_minmax/observation_minmax/setpoint_minmax/indoor_minmax."
        )

    return robust_scale(source, mode=scale_mode, floor=floor)
