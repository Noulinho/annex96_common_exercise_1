"""Helpers for CE1 district-load target experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.buildings import DATASET_NAME, dataset_dir, schema_path


@dataclass(frozen=True)
class DistrictTargetProfiles:
    """Profiles needed to score one controlled building against a district target.

    `background_without_controlled_heating` is the estimated full-district
    baseline electricity load with the controlled building's baseline space
    heating electricity removed. The controller then adds back its actual or
    predicted heating electricity.
    """

    target: list[float]
    background_without_controlled_heating: list[float]
    heating_action_power: list[float]
    target_scale: float
    target_scale_mode: str
    heating_nominal_power: float
    baseline_district_load: list[float]
    baseline_controlled_heating: list[float]

    def as_reward_kwargs(self, *, target_weight: float, target_mode: str = "tracking") -> dict[str, object]:
        return {
            "district_target_weight": float(target_weight),
            "district_target_profile": self.target,
            "district_background_profile": self.background_without_controlled_heating,
            "district_target_scale": float(self.target_scale),
            "district_target_mode": str(target_mode),
        }

    def district_stress_profile(self) -> list[float]:
        """Unitless CE1 stress signal for load shifting.

        Stress is zero when the baseline district load is below the CE1 target
        and one at the worst above-target baseline peak in the selected window.
        """

        baseline = np.asarray(self.baseline_district_load, dtype=float)
        target = np.asarray(self.target, dtype=float)
        n = min(len(baseline), len(target))
        raw = np.maximum(baseline[:n] - target[:n], 0.0)
        scale = max(float(np.nanmax(raw)) if len(raw) else 0.0, 1e-6)
        return (raw / scale).astype(float).tolist()

    def future_stress_profile(self, horizon: int = 6) -> list[float]:
        """Look-ahead stress signal for preheating before CE1 peak periods."""

        return future_stress_from_profile(self.district_stress_profile(), horizon=horizon)


def future_stress_from_profile(stress_profile, *, horizon: int = 6) -> list[float]:
    """Return max future stress over a finite look-ahead window."""

    stress = np.asarray(stress_profile, dtype=float)
    if stress.size == 0:
        return []

    horizon = max(int(horizon), 1)
    future = np.zeros_like(stress)
    for t in range(stress.size):
        if t + 1 >= stress.size:
            continue
        start = t + 1
        stop = min(t + horizon + 1, stress.size)
        future[t] = float(np.nanmax(stress[start:stop])) if stop > start else 0.0
    return np.clip(future, 0.0, 1.0).astype(float).tolist()


def _device_input_power(device, demand, outdoor_temperature, *, heating: bool) -> np.ndarray:
    try:
        values = device.get_input_power(demand, outdoor_temperature, heating=heating)
    except TypeError:
        values = device.get_input_power(demand)
    return np.asarray(values, dtype=float)


def _target_error_scale(
    *,
    target: np.ndarray,
    baseline_district: np.ndarray,
    background: np.ndarray | None = None,
    heating_action_power: np.ndarray | None = None,
    mode: str,
) -> float:
    mode = str(mode).strip().lower()
    if mode == "target_max":
        return max(float(np.nanmax(np.abs(target))) if len(target) else 0.0, 1.0)

    error = np.asarray(baseline_district, dtype=float) - np.asarray(target, dtype=float)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return 1.0

    if mode in {"controllable_error_minmax", "controllable_error_maxabs"}:
        if background is None or heating_action_power is None:
            raise ValueError(
                "controllable_error_minmax requires background and heating_action_power profiles."
            )
        target_arr = np.asarray(target, dtype=float)
        background_arr = np.asarray(background, dtype=float)
        action_arr = np.asarray(heating_action_power, dtype=float)
        n = min(len(target_arr), len(background_arr), len(action_arr))
        if n == 0:
            return 1.0
        # With u in [0, 1], the district target error must lie between these
        # two endpoints at each time step. Scaling by the largest endpoint
        # magnitude maps |error|/scale to [0, 1] for the controllable range.
        err_at_u0 = background_arr[:n] - target_arr[:n]
        err_at_u1 = background_arr[:n] + action_arr[:n] - target_arr[:n]
        endpoints = np.concatenate([err_at_u0, err_at_u1])
        endpoints = endpoints[np.isfinite(endpoints)]
        if endpoints.size == 0:
            return 1.0
        return max(float(np.nanmax(np.abs(endpoints))), 1.0)

    if mode in {"baseline_error_std", "error_std", "zscore"}:
        scale = float(np.nanstd(error))
    elif mode in {"baseline_error_iqr", "error_iqr", "robust_std"}:
        q25, q75 = np.nanpercentile(error, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
    elif mode in {"baseline_error_range", "error_range"}:
        q05, q95 = np.nanpercentile(error, [5.0, 95.0])
        scale = float(q95 - q05)
    elif mode in {"baseline_error_maxabs", "error_maxabs"}:
        scale = float(np.nanmax(np.abs(error)))
    else:
        raise ValueError(
            "target_scale_mode must be one of: target_max, baseline_error_std, "
            "baseline_error_iqr, baseline_error_range, baseline_error_maxabs, "
            "controllable_error_minmax."
        )

    return max(scale, 1.0)


def load_ce1_district_target_profiles(
    *,
    building_name: str,
    simulation_start: int = 744,
    simulation_end: int = 1415,
    dataset_name: str = DATASET_NAME,
    target_scale: float | None = None,
    target_scale_mode: str = "target_max",
    target_source: str = "ce1",
) -> DistrictTargetProfiles:
    """Return CE1 target/background arrays for one-building control.

    The CE1 target is district-level. For a single controlled building, the
    target-aware objective needs an estimate of the uncontrolled district load
    plus the controlled building's own heating electricity.
    """

    from citylearn.citylearn import CityLearnEnv

    data_dir = dataset_dir(dataset_name)
    env = CityLearnEnv(
        str(schema_path(dataset_name)),
        root_directory=str(data_dir),
        central_agent=True,
        simulation_start_time_step=int(simulation_start),
        simulation_end_time_step=int(simulation_end),
    )
    env.reset()

    names = [b.name for b in env.buildings]
    if building_name not in names:
        available = ", ".join(names)
        raise ValueError(f"Unknown building '{building_name}'. Available buildings: {available}")

    baseline_loads = [np.asarray(b._estimate_baseline_electricity_consumption(), dtype=float) for b in env.buildings]
    baseline_district = np.sum(np.stack(baseline_loads, axis=0), axis=0)

    controlled = env.buildings[names.index(building_name)]
    outdoor_temperature = np.asarray(controlled.weather.outdoor_dry_bulb_temperature, dtype=float)
    heating_demand = np.asarray(controlled.energy_simulation.heating_demand, dtype=float)
    baseline_heating = _device_input_power(
        controlled.heating_device,
        heating_demand,
        outdoor_temperature,
        heating=True,
    )
    hvac_mode = np.asarray(controlled.energy_simulation.hvac_mode, dtype=int)
    heating_available = np.isin(hvac_mode, [2, 3]).astype(float)
    heating_nominal_power = float(controlled.heating_device.nominal_power)
    heating_action_power = heating_nominal_power * heating_available

    target = np.asarray(controlled.district_load_target.district_load_target, dtype=float)
    n = min(len(target), len(baseline_district), len(baseline_heating), len(heating_action_power))
    source = str(target_source).strip().lower()
    if source == "mean_baseline":
        target = np.full(len(baseline_district), np.mean(baseline_district[:n]), dtype=float)
    elif source != "ce1":
        raise ValueError(f"target_source must be 'ce1' or 'mean_baseline', got {target_source!r}")
    target = target[:n]
    baseline_district = baseline_district[:n]
    baseline_heating = baseline_heating[:n]
    heating_action_power = heating_action_power[:n]
    background = baseline_district - baseline_heating

    if target_scale is None:
        scale_mode = str(target_scale_mode).strip().lower()
        scale = _target_error_scale(
            target=target,
            baseline_district=baseline_district,
            background=background,
            heating_action_power=heating_action_power,
            mode=scale_mode,
        )
    else:
        scale_mode = "manual"
        scale = max(float(target_scale), 1e-6)

    return DistrictTargetProfiles(
        target=target.astype(float).tolist(),
        background_without_controlled_heating=background.astype(float).tolist(),
        heating_action_power=heating_action_power.astype(float).tolist(),
        target_scale=scale,
        target_scale_mode=scale_mode,
        heating_nominal_power=heating_nominal_power,
        baseline_district_load=baseline_district.astype(float).tolist(),
        baseline_controlled_heating=baseline_heating.astype(float).tolist(),
    )
