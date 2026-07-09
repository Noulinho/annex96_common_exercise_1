"""Reward functions shared across controller comparisons."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Union

import numpy as np

from citylearn.reward_function import RewardFunction


def baseline_daily_mean_price_profile(
    baseline_load: list[float],
    *,
    simulation_start: int = 744,
    normalize: str = "mean",
    exp_beta: float = 2.0,
) -> list[float]:
    """Repeat the baseline mean-daily district load as an hourly price profile.

    Matches the function of the same name in scripts/03_run_mpc.py.
    """
    load = np.asarray(baseline_load, dtype=float).reshape(-1)
    if load.size == 0:
        return []
    hour_labels = (np.arange(load.size, dtype=int) + int(simulation_start)) % 24 + 1
    hourly = np.zeros(24, dtype=float)
    for hour in range(1, 25):
        values = load[hour_labels == hour]
        hourly[hour - 1] = float(np.nanmean(values)) if values.size else 0.0
    normalize = str(normalize).strip().lower()
    if normalize == "mean":
        scale = max(float(np.nanmean(hourly)), 1e-6)
        hourly = hourly / scale
    elif normalize == "exp_mean":
        scale = max(float(np.nanmean(hourly)), 1e-6)
        hourly = np.exp(float(exp_beta) * (hourly / scale - 1.0))
        hourly = hourly / max(float(np.nanmean(hourly)), 1e-6)
    elif normalize == "max":
        scale = max(float(np.nanmax(hourly)), 1e-6)
        hourly = hourly / scale
    return hourly[(hour_labels - 1).astype(int)].astype(float).tolist()


def load_simple_v4_price_profile(
    *,
    simulation_start: int = 744,
    baseline_path: str | Path | None = None,
    exp_beta: float = 2.0,
) -> list[float]:
    """Load the exp-normalised district baseline price profile used by simple_v4."""
    import pandas as pd

    if baseline_path is None:
        baseline_path = (
            Path(__file__).resolve().parents[1]
            / "results" / "raw" / "citylearn_baseline" / "no_occupant" / "baseline"
            / "district_rollout.csv"
        )
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        return []
    df = pd.read_csv(baseline_path)
    if "net_electricity_consumption" not in df.columns:
        return []
    return baseline_daily_mean_price_profile(
        df["net_electricity_consumption"].tolist(),
        simulation_start=simulation_start,
        normalize="exp_mean",
        exp_beta=exp_beta,
    )


def normalized_grid_stress_profile(
    values,
    *,
    mode: str = "normalized",
    percentile: float = 80.0,
    daily_top_k: int = 4,
):
    """Build a unitless grid-stress profile from a baseline demand curve."""

    load = np.asarray(values, dtype=float).reshape(-1)
    load = np.nan_to_num(load, nan=0.0, posinf=0.0, neginf=0.0)
    load = np.maximum(load, 0.0)
    if load.size == 0:
        return []

    mode = str(mode).strip().lower()
    if mode == "normalized":
        span = float(load.max() - load.min())
        if span <= 1e-12:
            return np.zeros_like(load, dtype=float).tolist()
        return ((load - load.min()) / span).astype(float).tolist()

    if mode == "percentile":
        threshold = float(np.percentile(load, percentile))
        return (load >= threshold).astype(float).tolist()

    if mode == "daily_top_k":
        profile = np.zeros_like(load, dtype=float)
        k = max(1, int(daily_top_k))
        for start in range(0, len(load), 24):
            stop = min(start + 24, len(load))
            day = load[start:stop]
            if day.size == 0:
                continue
            top = np.argsort(day)[-min(k, day.size) :]
            profile[start + top] = 1.0
        return profile.tolist()

    raise ValueError("mode must be 'normalized', 'percentile', or 'daily_top_k'.")


def load_grid_stress_profile(
    path,
    *,
    column: str = "net_electricity_consumption",
    mode: str = "normalized",
    percentile: float = 80.0,
    daily_top_k: int = 4,
):
    """Load a rollout CSV and convert its demand column to grid stress."""

    import pandas as pd

    data = pd.read_csv(path)
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in {path}. Available columns: {list(data.columns)}")
    return normalized_grid_stress_profile(
        data[column].to_numpy(dtype=float),
        mode=mode,
        percentile=percentile,
        daily_top_k=daily_top_k,
    )


def daily_window_shift_profile(
    *,
    simulation_start: int,
    simulation_end: int,
    preheat_hour_labels: tuple[int, ...] = (1, 2, 3, 13, 14, 15, 16, 17),
    reduce_hour_labels: tuple[int, ...] = (6, 7, 8, 9, 10, 20, 21, 22, 23),
    preheat_value: float = -1.0,
    reduce_value: float = 1.0,
) -> list[float]:
    """Signed daily profile: negative rewards heating, positive penalizes heating.

    Hour labels are 1..24 to match the district mean daily profile plots.
    """

    start = int(simulation_start)
    end = int(simulation_end)
    if end < start:
        return []

    preheat_hours = {int(h) for h in preheat_hour_labels}
    reduce_hours = {int(h) for h in reduce_hour_labels}
    profile = []
    for absolute_step in range(start, end + 1):
        hour_label = absolute_step % 24 + 1
        if hour_label in preheat_hours:
            profile.append(float(preheat_value))
        elif hour_label in reduce_hours:
            profile.append(float(reduce_value))
        else:
            profile.append(0.0)
    return profile


def time_bias_setpoint_profile(
    *,
    simulation_start: int,
    simulation_end: int,
    amplitude: float = 1.0,
    preheat_hour_labels: tuple[int, ...] = (1, 2, 3, 4, 5, 14, 15, 16, 17),
    reduce_hour_labels: tuple[int, ...] = (7, 8, 9, 10, 18, 19, 20, 21, 22),
) -> list[float]:
    """Daily setpoint bias profile using CE1 1..24 hour labels."""

    start = int(simulation_start)
    end = int(simulation_end)
    if end < start:
        return []

    amp = abs(float(amplitude))
    preheat_hours = {int(h) for h in preheat_hour_labels}
    reduce_hours = {int(h) for h in reduce_hour_labels}
    profile = []
    for absolute_step in range(start, end + 1):
        hour_label = absolute_step % 24 + 1
        if hour_label in preheat_hours:
            profile.append(amp)
        elif hour_label in reduce_hours:
            profile.append(-amp)
        else:
            profile.append(0.0)
    return profile


class BalancedComfortEnergyReward(RewardFunction):
    """Energy/peak-priority comfort reward for datasets without price/carbon.

    Stage cost per building:

        J_t = w_c D_t + w_e P_t + w_s lambda_t P^HVAC_t
              + w_p P_t^2 + w_r (P_t - P_{t-1})^2

    where P_t is normalized grid import and D_t is either squared setpoint
    tracking error or squared violation outside a soft thermal band around the
    active setpoint. D_t is divided by comfort_scale so the term can be given a
    z-score or min/max interpretation. The
    optional lambda_t profile is a unitless pseudo-price/grid-stress signal.
    For CE1, an optional district-load target term can compare
    background-district-load-plus-controlled-heating against the real
    `district_load_target` profile.
    """

    def __init__(
        self,
        env_metadata,
        comfort_weight: float = 0.5,
        energy_weight: float = 1.0,
        peak_weight: float = 3.0,
        ramp_weight: float = 1.0,
        shift_weight: float = 0.0,
        preheat_weight: float = 0.0,
        power_scale: float = 5.0,
        comfort_band: float | None = None,
        comfort_scale: float = 4.0,
        comfort_mode: str = "band",
        comfort_track_weight: float = 3.0,
        comfort_cold_weight: float = 60.0,
        comfort_hot_weight: float = 15.0,
        comfort_cold_tolerance: float = 0.5,
        comfort_hot_tolerance: float = 1.0,
        comfort_track_scale: float = 2.0,
        comfort_cold_scale: float = 1.5,
        comfort_hot_scale: float = 2.0,
        comfort_guard_exponent: float = 2.0,
        comfort_cold_k: float = 3.0,
        comfort_hot_k: float = 1.0,
        single_setpoint: bool = False,
        setpoint_bias_profile=None,
        time_bias_guard_band: float = 2.0,
        shift_profile=None,
        future_shift_profile=None,
        power_source: str = "net",
        action_power_coefficients=None,
        district_target_weight: float = 0.0,
        district_target_weight_profile=None,
        district_target_profile=None,
        district_background_profile=None,
        district_target_scale: float = 1.0,
        district_target_mode: str = "tracking",
        heat_budget_weight: float = 0.0,
        heat_budget_profile=None,
        heat_budget_window: int = 24,
        heat_budget_tolerance: float = 0.05,
        heat_budget_temperature_gate_margin: float | None = None,
        pi_prior_weight: float = 0.0,
        pi_prior_profile=None,
        pi_prior_temperature_gate_margin: float | None = None,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.comfort_weight = float(comfort_weight)
        self.energy_weight = float(energy_weight)
        self.peak_weight = float(peak_weight)
        self.ramp_weight = float(ramp_weight)
        self.shift_weight = float(shift_weight)
        self.preheat_weight = float(preheat_weight)
        self.power_scale = max(float(power_scale), 1e-6)
        self.comfort_band = None if comfort_band is None else float(comfort_band)
        self.comfort_scale = max(float(comfort_scale), 1e-6)
        self.comfort_mode = str(comfort_mode).strip().lower()
        if self.comfort_mode not in {"band", "setpoint", "asymmetric", "barrier", "time_bias_guard"}:
            raise ValueError("comfort_mode must be 'band', 'setpoint', 'asymmetric', 'barrier', or 'time_bias_guard'.")
        self.comfort_track_weight = float(comfort_track_weight)
        self.comfort_cold_weight = float(comfort_cold_weight)
        self.comfort_hot_weight = float(comfort_hot_weight)
        self.comfort_cold_tolerance = max(float(comfort_cold_tolerance), 0.0)
        self.comfort_hot_tolerance = max(float(comfort_hot_tolerance), 0.0)
        self.comfort_track_scale = max(float(comfort_track_scale), 1e-6)
        self.comfort_cold_scale = max(float(comfort_cold_scale), 1e-6)
        self.comfort_hot_scale = max(float(comfort_hot_scale), 1e-6)
        self.comfort_guard_exponent = max(float(comfort_guard_exponent), 1e-6)
        self.comfort_cold_k = max(float(comfort_cold_k), 1e-6)
        self.comfort_hot_k = max(float(comfort_hot_k), 1e-6)
        self.single_setpoint = bool(single_setpoint)
        self.setpoint_bias_profile = (
            None if setpoint_bias_profile is None else [float(v) for v in setpoint_bias_profile]
        )
        self.time_bias_guard_band = max(float(time_bias_guard_band), 0.0)
        self.shift_profile = None if shift_profile is None else [float(v) for v in shift_profile]
        self.future_shift_profile = (
            None if future_shift_profile is None else [float(v) for v in future_shift_profile]
        )
        self.power_source = str(power_source).strip().lower()
        if self.power_source not in {"net", "heating", "action_proxy"}:
            raise ValueError("power_source must be 'net', 'heating', or 'action_proxy'.")
        if action_power_coefficients is None:
            self.action_power_coefficients = None
        elif isinstance(action_power_coefficients, (int, float)):
            self.action_power_coefficients = [float(action_power_coefficients)]
        else:
            self.action_power_coefficients = [float(v) for v in action_power_coefficients]
        self.district_target_weight = float(district_target_weight)
        self.district_target_weight_profile = (
            None if district_target_weight_profile is None else [float(v) for v in district_target_weight_profile]
        )
        self.district_target_profile = None if district_target_profile is None else [float(v) for v in district_target_profile]
        self.district_background_profile = (
            None if district_background_profile is None else [float(v) for v in district_background_profile]
        )
        self.district_target_scale = max(float(district_target_scale), 1e-6)
        self.district_target_mode = str(district_target_mode).strip().lower()
        if self.district_target_mode not in {"tracking", "peak_only"}:
            raise ValueError("district_target_mode must be 'tracking' or 'peak_only'.")
        self.heat_budget_weight = float(heat_budget_weight)
        self.heat_budget_profile = None if heat_budget_profile is None else [float(v) for v in heat_budget_profile]
        self.heat_budget_window = max(int(heat_budget_window), 1)
        self.heat_budget_tolerance = max(float(heat_budget_tolerance), 1e-6)
        self.heat_budget_temperature_gate_margin = (
            None
            if heat_budget_temperature_gate_margin is None
            else max(float(heat_budget_temperature_gate_margin), 0.0)
        )
        self.pi_prior_weight = float(pi_prior_weight)
        self.pi_prior_profile = None if pi_prior_profile is None else [float(v) for v in pi_prior_profile]
        self.pi_prior_temperature_gate_margin = (
            None
            if pi_prior_temperature_gate_margin is None
            else max(float(pi_prior_temperature_gate_margin), 0.0)
        )
        self._previous_power: list[float | None] = []
        self._current_action_values: list[float] = []
        self._current_reference_setpoints: list[float | None] = []
        self._heat_budget_reference: list[list[float]] = []
        self._heat_budget_actual: list[list[float]] = []
        self._time_step = 0
        self.reset()

    def reset(self):
        if self.env_metadata is None:
            self._previous_power = []
            self._current_action_values = []
            self._current_reference_setpoints = []
            self._heat_budget_reference = []
            self._heat_budget_actual = []
        else:
            self._previous_power = [None for _ in self.env_metadata["buildings"]]
            self._current_action_values = [0.0 for _ in self.env_metadata["buildings"]]
            self._current_reference_setpoints = [None for _ in self.env_metadata["buildings"]]
            self._heat_budget_reference = [[] for _ in self.env_metadata["buildings"]]
            self._heat_budget_actual = [[] for _ in self.env_metadata["buildings"]]
        self._time_step = 0

    def set_current_reference_setpoints(self, setpoints) -> None:
        self._current_reference_setpoints = [
            None if value is None else float(value)
            for value in setpoints
        ]

    def set_current_actions(self, actions, buildings=None) -> None:
        values = []
        for action in actions:
            if isinstance(action, Mapping):
                value = action.get("heating_device", action.get("cooling_or_heating_device", 0.0))
            else:
                arr = np.asarray(action, dtype=float).reshape(-1)
                value = arr[0] if arr.size else 0.0
            values.append(max(float(value), 0.0))
        self._current_action_values = values

    def _setpoint(self, observation: Mapping[str, Union[int, float]], index: int | None = None) -> float:
        if (
            index is not None
            and index < len(self._current_reference_setpoints)
            and self._current_reference_setpoints[index] is not None
        ):
            return float(self._current_reference_setpoints[index])
        if "override_setpoint" in observation:
            return float(observation["override_setpoint"])
        if self.single_setpoint:
            return float(observation["indoor_dry_bulb_temperature_heating_set_point"])

        hvac_mode = int(observation.get("hvac_mode", 0))
        if hvac_mode == 1:
            return float(observation["indoor_dry_bulb_temperature_cooling_set_point"])
        if hvac_mode == 2:
            return float(observation["indoor_dry_bulb_temperature_heating_set_point"])

        heating = float(observation.get("heating_demand", 0.0))
        cooling = float(observation.get("cooling_demand", 0.0))
        if cooling > heating:
            return float(observation["indoor_dry_bulb_temperature_cooling_set_point"])
        return float(observation["indoor_dry_bulb_temperature_heating_set_point"])

    def _comfort_cost(self, observation: Mapping[str, Union[int, float]], index: int | None = None) -> float:
        indoor_temperature = float(observation["indoor_dry_bulb_temperature"])
        setpoint = self._setpoint(observation, index)
        if self.comfort_mode == "setpoint":
            error = indoor_temperature - setpoint
            return error * error
        if self.comfort_mode == "asymmetric":
            track = ((indoor_temperature - setpoint) / self.comfort_track_scale) ** 2
            cold = max(0.0, setpoint - self.comfort_cold_tolerance - indoor_temperature)
            hot = max(0.0, indoor_temperature - setpoint - self.comfort_hot_tolerance)
            cold = (cold / self.comfort_cold_scale) ** self.comfort_guard_exponent
            hot = (hot / self.comfort_hot_scale) ** self.comfort_guard_exponent
            return (
                self.comfort_track_weight * track
                + self.comfort_cold_weight * cold
                + self.comfort_hot_weight * hot
            )

        if self.comfort_mode == "barrier":
            # Exponential cold barrier + quadratic hot penalty.
            # Cold: (exp(k * max(T_sp - tol - T, 0)) - 1) — catastrophically expensive
            #       beyond ~1 °C, yet zero when comfortable.  Hot violations are cheap
            #       (quadratic) because over-heating is not the dominant failure mode
            #       in a heating-season dataset.
            cold_viol = max(0.0, setpoint - self.comfort_cold_tolerance - indoor_temperature)
            hot_viol = max(0.0, indoor_temperature - setpoint - self.comfort_hot_tolerance)
            cold_cost = math.expm1(self.comfort_cold_k * cold_viol)  # exp(k*v)-1
            hot_cost = (hot_viol / self.comfort_hot_scale) ** 2
            return (
                self.comfort_cold_weight * cold_cost
                + self.comfort_hot_weight * hot_cost
            )

        if self.comfort_mode == "time_bias_guard":
            bias = 0.0
            if self.setpoint_bias_profile:
                bias = self.setpoint_bias_profile[min(self._time_step, len(self.setpoint_bias_profile) - 1)]
            biased_setpoint = setpoint + bias
            track = ((indoor_temperature - biased_setpoint) / self.comfort_track_scale) ** 2
            cold = max(0.0, setpoint - self.time_bias_guard_band - indoor_temperature)
            hot = max(0.0, indoor_temperature - setpoint - self.time_bias_guard_band)
            cold = (cold / self.comfort_cold_scale) ** self.comfort_guard_exponent
            hot = (hot / self.comfort_hot_scale) ** self.comfort_guard_exponent
            return (
                self.comfort_track_weight * track
                + self.comfort_cold_weight * cold
                + self.comfort_hot_weight * hot
            )

        band = (
            float(observation.get("comfort_band", 2.0))
            if self.comfort_band is None
            else self.comfort_band
        )
        band = max(abs(band), 0.0)
        lower = setpoint - band
        upper = setpoint + band
        below = max(0.0, lower - indoor_temperature)
        above = max(0.0, indoor_temperature - upper)
        return below * below + above * above

    def _action_power_coefficient(self, index: int) -> float:
        if self.action_power_coefficients:
            return self.action_power_coefficients[min(index, len(self.action_power_coefficients) - 1)]
        if self.env_metadata is not None:
            buildings = self.env_metadata.get("buildings", [])
            if index < len(buildings):
                value = buildings[index].get("controlled_heating_nominal_power")
                if value is not None:
                    return float(value)
        return 1.0

    def _power(self, observation: Mapping[str, Union[int, float]], index: int = 0) -> float:
        if self.power_source == "action_proxy":
            action = self._current_action_values[min(index, len(self._current_action_values) - 1)] if self._current_action_values else 0.0
            return max(action * self._action_power_coefficient(index), 0.0)
        if self.power_source == "heating":
            return max(float(observation.get("heating_electricity_consumption", 0.0)), 0.0)
        return max(float(observation["net_electricity_consumption"]), 0.0)

    def calculate(self, observations):
        reward_list = []

        if len(self._previous_power) != len(observations):
            self._previous_power = [None for _ in observations]
        if len(self._heat_budget_reference) != len(observations):
            self._heat_budget_reference = [[] for _ in observations]
            self._heat_budget_actual = [[] for _ in observations]

        grid_stress = 0.0
        if self.shift_profile:
            grid_stress = self.shift_profile[min(self._time_step, len(self.shift_profile) - 1)]
        future_stress = 0.0
        if self.future_shift_profile:
            future_stress = self.future_shift_profile[min(self._time_step, len(self.future_shift_profile) - 1)]

        for i, observation in enumerate(observations):
            indoor_temperature = float(observation["indoor_dry_bulb_temperature"])
            setpoint = self._setpoint(observation, i)
            power = self._power(observation, i) / self.power_scale
            shift_power = power
            preheat_signal = max(0.0, 1.0 - grid_stress) * max(0.0, future_stress)
            previous_power = self._previous_power[i]
            ramp = 0.0 if previous_power is None else power - previous_power
            self._previous_power[i] = power

            comfort_cost = self._comfort_cost(observation, i) / self.comfort_scale
            heating_power = max(float(observation.get("heating_electricity_consumption", 0.0)), 0.0)
            heating_power_norm = heating_power / self.power_scale
            reference_heating_norm = 0.0
            if self.heat_budget_profile:
                budget_idx = min(self._time_step, len(self.heat_budget_profile) - 1)
                reference_heating_norm = max(float(self.heat_budget_profile[budget_idx]), 0.0) / self.power_scale
            elif self.pi_prior_profile:
                prior_idx = min(self._time_step, len(self.pi_prior_profile) - 1)
                reference_heating_norm = max(float(self.pi_prior_profile[prior_idx]), 0.0) / self.power_scale

            heat_budget_cost = 0.0
            if self.heat_budget_weight != 0.0 and self.heat_budget_profile:
                self._heat_budget_reference[i].append(reference_heating_norm)
                self._heat_budget_actual[i].append(heating_power_norm)
                if len(self._heat_budget_reference[i]) > self.heat_budget_window:
                    self._heat_budget_reference[i] = self._heat_budget_reference[i][-self.heat_budget_window :]
                    self._heat_budget_actual[i] = self._heat_budget_actual[i][-self.heat_budget_window :]
                reference_sum = float(np.sum(self._heat_budget_reference[i]))
                actual_sum = float(np.sum(self._heat_budget_actual[i]))
                if reference_sum > 1e-9:
                    gate_margin = self.heat_budget_temperature_gate_margin
                    gate_active = gate_margin is None or indoor_temperature <= setpoint - gate_margin
                    if gate_active:
                        allowed_deficit = self.heat_budget_tolerance * reference_sum
                        deficit = max(0.0, reference_sum - actual_sum - allowed_deficit)
                        heat_budget_cost = (deficit / max(allowed_deficit, 1e-6)) ** 2

            pi_prior_cost = 0.0
            if self.pi_prior_weight != 0.0 and self.pi_prior_profile:
                prior_idx = min(self._time_step, len(self.pi_prior_profile) - 1)
                reference = max(float(self.pi_prior_profile[prior_idx]), 0.0) / self.power_scale
                below_or_near_peak = max(0.0, min(1.0, grid_stress))
                low_stress = 1.0 - below_or_near_peak
                above_reference = max(0.0, heating_power_norm - reference)
                gate_margin = self.pi_prior_temperature_gate_margin
                hot_zone = gate_margin is not None and indoor_temperature > setpoint + gate_margin
                if hot_zone:
                    pi_prior_cost = above_reference * above_reference
                else:
                    pi_prior_cost = low_stress * (heating_power_norm - reference) ** 2
                    pi_prior_cost += below_or_near_peak * above_reference * above_reference

            district_target_cost = 0.0
            district_target_weight = self.district_target_weight
            if self.district_target_weight_profile:
                weight_idx = min(self._time_step, len(self.district_target_weight_profile) - 1)
                district_target_weight = self.district_target_weight_profile[weight_idx]
            if (
                district_target_weight != 0.0
                and self.district_target_profile
                and self.district_background_profile
            ):
                profile_idx = min(self._time_step, len(self.district_target_profile) - 1)
                background_idx = min(self._time_step, len(self.district_background_profile) - 1)
                target = self.district_target_profile[profile_idx]
                background = self.district_background_profile[background_idx]
                error = (background + heating_power - target) / self.district_target_scale
                if self.district_target_mode == "peak_only":
                    error = max(0.0, error)
                district_target_cost = error * error
            cost = (
                self.comfort_weight * comfort_cost
                + self.energy_weight * power
                + self.shift_weight * grid_stress * shift_power
                + self.peak_weight * power * power
                + self.ramp_weight * ramp * ramp
                + district_target_weight * district_target_cost
                + self.heat_budget_weight * heat_budget_cost
                + self.pi_prior_weight * pi_prior_cost
                - self.preheat_weight * preheat_signal * shift_power
            )
            reward_list.append(float(-cost))

        self._time_step += 1
        return [float(np.sum(reward_list))] if self.central_agent else reward_list


class LagrangianDistrictReward(BalancedComfortEnergyReward):
    """District reward with comfort as a Lagrangian constraint.

    Reward = -(district + energy + ramp + soft_hot_penalty) - λ * cold_violation_°C

    The cold comfort floor is a hard constraint enforced via an adaptive dual
    variable λ.  λ increases when mean cold violation exceeds the budget and
    decreases otherwise, so SAC is forced to learn load shifting organically
    rather than deleting heat to satisfy a linear comfort-district tradeoff.

    λ update (once per lagrange_update_period SAC-phase steps):
        λ ← clip(λ + lr * (mean_cold_viol_window - budget), λ_min, λ_max)

    Call ``step_sac()`` once per SAC-phase training step from the pipeline to
    trigger the update automatically.  ``pop_violations()`` and
    ``set_lagrange_multiplier()`` are available for external control.
    """

    def __init__(
        self,
        env_metadata,
        *,
        lagrange_multiplier: float = 1.0,
        lagrange_lr: float = 0.05,
        comfort_constraint_budget: float = 0.05,
        lagrange_update_period: int = 96,
        lagrange_min: float = 0.0,
        lagrange_max: float = 100.0,
        **kwargs,
    ):
        # Set Lagrangian attributes before super().__init__() since that calls reset()
        self.lagrange_multiplier = float(lagrange_multiplier)
        self.lagrange_lr = float(lagrange_lr)
        self.comfort_constraint_budget = float(comfort_constraint_budget)
        self.lagrange_update_period = int(lagrange_update_period)
        self.lagrange_min = float(lagrange_min)
        self.lagrange_max = float(lagrange_max)
        self._cold_violation_buffer: list[float] = []
        self._sac_step_count: int = 0
        self.lambda_trajectory: list[tuple] = []
        super().__init__(env_metadata, **kwargs)

    def reset(self):
        super().reset()
        self._cold_violation_buffer.clear()
        self._sac_step_count = 0

    def pop_violations(self) -> list[float]:
        out = self._cold_violation_buffer[:]
        self._cold_violation_buffer.clear()
        return out

    def set_lagrange_multiplier(self, val: float) -> None:
        self.lagrange_multiplier = float(np.clip(val, self.lagrange_min, self.lagrange_max))

    def step_sac(self) -> None:
        """Call once per SAC-phase training step; updates λ every lagrange_update_period steps."""
        self._sac_step_count += 1
        if self.lagrange_update_period > 0 and self._sac_step_count % self.lagrange_update_period == 0:
            violations = self.pop_violations()
            if violations:
                mean_viol = float(np.mean(violations))
                self.set_lagrange_multiplier(
                    self.lagrange_multiplier + self.lagrange_lr * (mean_viol - self.comfort_constraint_budget)
                )
                self.lambda_trajectory.append((self._sac_step_count, self.lagrange_multiplier, mean_viol))

    def calculate(self, observations):
        reward_list = []

        if len(self._previous_power) != len(observations):
            self._previous_power = [None for _ in observations]
        if len(self._heat_budget_reference) != len(observations):
            self._heat_budget_reference = [[] for _ in observations]
            self._heat_budget_actual = [[] for _ in observations]

        grid_stress = 0.0
        if self.shift_profile:
            grid_stress = self.shift_profile[min(self._time_step, len(self.shift_profile) - 1)]
        future_stress = 0.0
        if self.future_shift_profile:
            future_stress = self.future_shift_profile[min(self._time_step, len(self.future_shift_profile) - 1)]

        for i, observation in enumerate(observations):
            indoor_temperature = float(observation["indoor_dry_bulb_temperature"])
            setpoint = self._setpoint(observation)
            power = self._power(observation) / self.power_scale
            shift_power = power
            preheat_signal = max(0.0, 1.0 - grid_stress) * max(0.0, future_stress)
            previous_power = self._previous_power[i]
            ramp = 0.0 if previous_power is None else power - previous_power
            self._previous_power[i] = power

            # Constraint signal: raw °C below comfort floor — linear and bounded for stable λ updates
            cold_viol = max(0.0, setpoint - self.comfort_cold_tolerance - indoor_temperature)
            self._cold_violation_buffer.append(cold_viol)

            # Soft hot penalty stays in the objective (overheating is rare but still unwanted)
            hot_viol = max(0.0, indoor_temperature - setpoint - self.comfort_hot_tolerance)
            hot_cost = (hot_viol / self.comfort_hot_scale) ** 2

            heating_power = max(float(observation.get("heating_electricity_consumption", 0.0)), 0.0)
            heating_power_norm = heating_power / self.power_scale

            heat_budget_cost = 0.0
            if self.heat_budget_weight != 0.0 and self.heat_budget_profile:
                budget_idx = min(self._time_step, len(self.heat_budget_profile) - 1)
                ref_norm = max(float(self.heat_budget_profile[budget_idx]), 0.0) / self.power_scale
                self._heat_budget_reference[i].append(ref_norm)
                self._heat_budget_actual[i].append(heating_power_norm)
                if len(self._heat_budget_reference[i]) > self.heat_budget_window:
                    self._heat_budget_reference[i] = self._heat_budget_reference[i][-self.heat_budget_window:]
                    self._heat_budget_actual[i] = self._heat_budget_actual[i][-self.heat_budget_window:]
                reference_sum = float(np.sum(self._heat_budget_reference[i]))
                actual_sum = float(np.sum(self._heat_budget_actual[i]))
                if reference_sum > 1e-9:
                    gate_margin = self.heat_budget_temperature_gate_margin
                    gate_active = gate_margin is None or indoor_temperature <= setpoint - gate_margin
                    if gate_active:
                        allowed_deficit = self.heat_budget_tolerance * reference_sum
                        deficit = max(0.0, reference_sum - actual_sum - allowed_deficit)
                        heat_budget_cost = (deficit / max(allowed_deficit, 1e-6)) ** 2

            pi_prior_cost = 0.0
            if self.pi_prior_weight != 0.0 and self.pi_prior_profile:
                prior_idx = min(self._time_step, len(self.pi_prior_profile) - 1)
                reference = max(float(self.pi_prior_profile[prior_idx]), 0.0) / self.power_scale
                below_or_near_peak = max(0.0, min(1.0, grid_stress))
                above_reference = max(0.0, heating_power_norm - reference)
                gate_margin = self.pi_prior_temperature_gate_margin
                hot_zone = gate_margin is not None and indoor_temperature > setpoint + gate_margin
                if hot_zone:
                    pi_prior_cost = above_reference * above_reference
                else:
                    low_stress = 1.0 - below_or_near_peak
                    pi_prior_cost = low_stress * (heating_power_norm - reference) ** 2
                    pi_prior_cost += below_or_near_peak * above_reference * above_reference

            district_target_cost = 0.0
            district_target_weight = self.district_target_weight
            if self.district_target_weight_profile:
                weight_idx = min(self._time_step, len(self.district_target_weight_profile) - 1)
                district_target_weight = self.district_target_weight_profile[weight_idx]
            if (
                district_target_weight != 0.0
                and self.district_target_profile
                and self.district_background_profile
            ):
                profile_idx = min(self._time_step, len(self.district_target_profile) - 1)
                background_idx = min(self._time_step, len(self.district_background_profile) - 1)
                target = self.district_target_profile[profile_idx]
                background = self.district_background_profile[background_idx]
                error = (background + heating_power - target) / self.district_target_scale
                if self.district_target_mode == "peak_only":
                    error = max(0.0, error)
                district_target_cost = error * error

            objective_cost = (
                self.comfort_hot_weight * hot_cost
                + self.energy_weight * power
                + self.shift_weight * grid_stress * shift_power
                + self.peak_weight * power * power
                + self.ramp_weight * ramp * ramp
                + district_target_weight * district_target_cost
                + self.heat_budget_weight * heat_budget_cost
                + self.pi_prior_weight * pi_prior_cost
                - self.preheat_weight * preheat_signal * shift_power
            )
            reward_list.append(float(-objective_cost - self.lagrange_multiplier * cold_viol))

        self._time_step += 1
        return [float(np.sum(reward_list))] if self.central_agent else reward_list


class SimpleComfortEnergyReward(RewardFunction):
    """Mirrors the simple_v4 MPC stage cost as a reward signal.

    r = -(comfort_weight * (Tin - T_track)² / band²
          + energy_weight * GridStress * u_proxy
          + ramp_weight * (u_proxy - u_prev)²)

    where T_track = Tsp - tsp_lower_bias  (0 = track actual setpoint)
    and u_proxy = heating_electricity_consumption / power_scale.

    GridStress comes from shift_profile (exp-normalised district baseline daily mean).
    If shift_profile is None, GridStress = 1.0 (flat penalty).
    ramp_weight penalises large step changes in heating power to suppress bang-bang.
    """

    def __init__(
        self,
        env_metadata,
        *,
        comfort_weight: float = 0.5,
        energy_weight: float = 0.5,
        comfort_band: float = 2.0,
        tsp_lower_bias: float = 0.0,
        power_scale: float = 5.0,
        shift_profile: list[float] | None = None,
        cold_only: bool = False,
        ramp_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.comfort_weight = float(comfort_weight)
        self.energy_weight = float(energy_weight)
        self.comfort_band = float(comfort_band)
        self.tsp_lower_bias = float(tsp_lower_bias)
        self.power_scale = max(float(power_scale), 1e-6)
        self.shift_profile = list(shift_profile) if shift_profile is not None else None
        self.cold_only = bool(cold_only)
        self.ramp_weight = float(ramp_weight)
        self._time_step = 0
        self._previous_power: list[float | None] = []

    def reset(self):
        self._time_step = 0
        self._previous_power = []

    def calculate(self, observations):
        gs = 1.0
        if self.shift_profile:
            gs = float(self.shift_profile[min(self._time_step, len(self.shift_profile) - 1)])
        reward_list = []
        next_power: list[float] = []
        for i, obs in enumerate(observations):
            tin = float(obs.get("indoor_dry_bulb_temperature", 20.0))
            tsp = float(obs.get("indoor_dry_bulb_temperature_heating_set_point", 20.0))
            t_track = tsp - self.tsp_lower_bias
            u_proxy = max(float(obs.get("heating_electricity_consumption", 0.0)), 0.0) / self.power_scale
            error = t_track - tin if self.cold_only else tin - t_track
            if self.cold_only:
                error = max(0.0, error)
            comfort_cost = self.comfort_weight * error ** 2 / max(self.comfort_band ** 2, 1e-6)
            energy_cost = self.energy_weight * gs * u_proxy
            prev = self._previous_power[i] if i < len(self._previous_power) else None
            ramp = 0.0 if prev is None else u_proxy - prev
            ramp_cost = self.ramp_weight * ramp ** 2
            reward_list.append(-float(comfort_cost + energy_cost + ramp_cost))
            next_power.append(u_proxy)
        self._previous_power = next_power
        self._time_step += 1
        return [float(np.sum(reward_list))] if self.central_agent else reward_list
