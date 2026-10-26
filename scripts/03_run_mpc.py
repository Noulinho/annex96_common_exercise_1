#!/usr/bin/env python
"""Run the February MPC rollout from experiments/notebooks/my_mpc.ipynb.

This script intentionally mirrors the working notebook cell:

- fitted one-state RC/COP model loaded from JSON,
- same do-mpc dynamics and objective,
- same online offset/disturbance correction `w_est`,
- same February window 744-1415,
- same optional occupant wrapper with thermostat overrides.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import (
    DATASET_NAME,
    building_metadata,
    format_building_path,
    resolve_building_entries,
)
from src.district_target import future_stress_from_profile, load_ce1_district_target_profiles
from src.experiment_normalization import temperature_error_scale
from src.occupants.metrics import (
    kelvin_hours,
    occupant_probability_comfort_band,
    override_hours,
    setpoint_kelvin_hours,
)
from src.occupants.setpoint_schedule import (
    active_override_steps,
    apply_heating_override_schedule,
    reset_future_heating_setpoints,
)
from src.occupants.tdyn import TDynConfig, update_tdyn_reference
from src.rewards import BalancedComfortEnergyReward, daily_window_shift_profile, load_grid_stress_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="no_occupant",
        choices=["no_occupant", "occupant_present", "occupant_present_peak_flattening"],
    )
    parser.add_argument("--rc-model", default="results/models/rc/1state_rc_247942_january.json")
    parser.add_argument(
        "--rc-model-type",
        default="1r1c",
        choices=["1r1c", "2r2c"],
        help="RC model variant to use for the MPC dynamics.",
    )
    parser.add_argument(
        "--rc-model-2r2c",
        default="results/models/rc/2r2c_{short_name}_january.json",
        help="Path template for 2R2C model JSON (used when --rc-model-type 2r2c).",
    )
    parser.add_argument(
        "--two-state-observer",
        default="propagate",
        choices=["propagate", "kalman"],
        help="2R2C hidden mass observer. 'kalman' corrects latent Tmass from Tin residuals each step.",
    )
    parser.add_argument("--kalman-process-var", type=float, default=0.05)
    parser.add_argument("--kalman-measurement-var", type=float, default=0.25)
    parser.add_argument("--kalman-initial-var", type=float, default=4.0)
    parser.add_argument("--output-dir", default="results/raw/mpc")
    parser.add_argument("--summary-dir", default="results/summaries/mpc")
    parser.add_argument(
        "--building",
        default="all",
        help="Building selector: all, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument(
        "--occupancy-building",
        default="same",
        help="B-spline occupancy source. Use 'same' for the controlled building, or any building selector.",
    )
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional short smoke-test length.")
    parser.add_argument(
        "--simulation-start",
        type=int,
        default=FEB_START,
        help="First simulation time step for the rollout. Defaults to February start.",
    )
    parser.add_argument(
        "--simulation-end",
        type=int,
        default=FEB_END,
        help="Last simulation time step for the rollout. Defaults to February end.",
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument(
        "--objective",
        default="baseline",
        choices=["baseline", "balanced", "budget_shift", "simple"],
        help=(
            "No-occupant MPC objective. 'balanced' uses comfort/energy/peak/ramp terms. "
            "'budget_shift' shifts heat relative to the baseline MPC action with daily payback. "
            "'simple' uses only comfort (band violation^2 / band^2) and energy (u in [0,1]), "
            "both normalized to [0, 1] so equal 0.5/0.5 weights are meaningful."
        ),
    )
    parser.add_argument("--comfort-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--tsp-lower-bias", type=float, default=0.0,
                        help="simple objective: shift tracking target (Tsp - bias) to save energy. Default 0.0 = track Tsp exactly.")
    parser.add_argument("--peak-weight", type=float, default=3.0)
    parser.add_argument("--ramp-weight", type=float, default=1.0)
    parser.add_argument("--shift-weight", type=float, default=0.0)
    parser.add_argument("--budget-weight", type=float, default=0.0)
    parser.add_argument("--preheat-weight", type=float, default=0.0)
    parser.add_argument("--preheat-horizon", type=int, default=6)
    parser.add_argument("--comfort-band", type=float, default=2.0)
    parser.add_argument(
        "--comfort-mode",
        default="band",
        choices=["band", "setpoint", "asymmetric", "barrier", "time_bias_guard"],
        help="Balanced objective comfort term: soft band violation, exact setpoint tracking, asymmetric tradeoff, exponential cold barrier, or time-biased tracking with real-setpoint guard.",
    )
    parser.add_argument("--comfort-track-weight", type=float, default=3.0)
    parser.add_argument("--comfort-cold-weight", type=float, default=60.0)
    parser.add_argument("--comfort-hot-weight", type=float, default=15.0)
    parser.add_argument("--comfort-cold-tolerance", type=float, default=0.5)
    parser.add_argument("--comfort-hot-tolerance", type=float, default=1.0)
    parser.add_argument("--comfort-track-scale", type=float, default=2.0)
    parser.add_argument("--comfort-cold-scale", type=float, default=1.5)
    parser.add_argument("--comfort-hot-scale", type=float, default=2.0)
    parser.add_argument("--comfort-cold-k", type=float, default=3.0,
                        help="Exponential growth rate for cold barrier (barrier mode only).")
    parser.add_argument("--comfort-hot-k", type=float, default=1.0,
                        help="Exponential growth rate for hot barrier (barrier mode only).")
    parser.add_argument(
        "--time-bias-amplitude",
        type=float,
        default=1.0,
        help="Setpoint bias magnitude in degC for --comfort-mode time_bias_guard.",
    )
    parser.add_argument(
        "--time-bias-guard-band",
        type=float,
        default=2.0,
        help="Hard comfort guard around the true setpoint for --comfort-mode time_bias_guard.",
    )
    parser.add_argument(
        "--time-bias-guard-penalty",
        type=float,
        default=1e5,
        help="Soft-constraint penalty outside Tsp +/- guard band for --comfort-mode time_bias_guard.",
    )
    parser.add_argument(
        "--comfort-hot-tolerance-night",
        type=float,
        default=None,
        help=(
            "Hot tolerance for hours 0-5 (night preheating window). "
            "If set, MPC uses a wider band at night to allow thermal storage. "
            "Defaults to --comfort-hot-tolerance (no night widening)."
        ),
    )
    parser.add_argument("--comfort-hot-tolerance-night-end", type=int, default=6,
                        help="First hour of day where night tolerance reverts to day tolerance (default 6 = 6am).")
    parser.add_argument(
        "--single-setpoint",
        action="store_true",
        help="Track the heating setpoint as the single comfort reference.",
    )
    parser.add_argument(
        "--comfort-scale",
        type=float,
        default=None,
        help="Scale for squared comfort error. Defaults to comfort_band**2 for band mode or sigma_T**2 for setpoint mode.",
    )
    parser.add_argument(
        "--temperature-scale-mode",
        default="setpoint_std",
        choices=[
            "error_std",
            "error_iqr",
            "error_range",
            "error_minmax",
            "setpoint_std",
            "setpoint_iqr",
            "setpoint_range",
            "setpoint_minmax",
            "indoor_std",
            "indoor_iqr",
            "indoor_range",
            "indoor_minmax",
            "temperature_minmax",
            "dataset_minmax",
            "full_dataset_minmax",
            "observation_minmax",
            "observation_space_minmax",
            "obs_minmax",
        ],
        help="How to normalize setpoint error when --comfort-mode setpoint and --comfort-scale is omitted.",
    )
    parser.add_argument("--power-scale", type=float, default=5.0)
    parser.add_argument(
        "--power-source",
        default="net",
        choices=["net", "heating"],
        help="Power signal used by the balanced reward logger. Use heating with --raw-power-cost for flat heating-energy sweeps.",
    )
    parser.add_argument(
        "--ramp-scale",
        type=float,
        default=1.0,
        help="Normalization scale for the budget_shift heating ramp term in action units.",
    )
    parser.add_argument(
        "--daily-budget-scale",
        type=float,
        default=1.0,
        help="Normalization scale for the budget_shift running daily heat-budget debt in action-hours.",
    )
    parser.add_argument(
        "--baseline-action-label",
        default="baseline",
        help="No-occupant MPC label used as u_PI baseline for --objective budget_shift.",
    )
    parser.add_argument(
        "--grid-stress-rollout",
        default=None,
        help="Optional rollout CSV used to build a pseudo-price/grid-stress load-shifting signal.",
    )
    parser.add_argument("--grid-stress-column", default="net_electricity_consumption")
    parser.add_argument(
        "--grid-stress-mode",
        default="normalized",
        choices=["normalized", "percentile", "daily_top_k"],
    )
    parser.add_argument("--grid-stress-percentile", type=float, default=80.0)
    parser.add_argument("--grid-stress-daily-top-k", type=int, default=4)
    parser.add_argument(
        "--load-shape-profile",
        default="ce1_stress",
        choices=[
            "ce1_stress",
            "daily_windows",
            "baseline_daily_mean",
            "baseline_daily_mean_centered",
            "baseline_daily_mean_exp",
        ],
        help="Source for the shift term when no grid-stress rollout is supplied.",
    )
    parser.add_argument(
        "--load-shape-exp-beta",
        type=float,
        default=3.0,
        help="Beta for --load-shape-profile baseline_daily_mean_exp.",
    )
    parser.add_argument(
        "--district-target-weight",
        type=float,
        default=0.0,
        help="Weight on CE1 district-load target tracking. Uses the real district_target.csv.",
    )
    parser.add_argument(
        "--district-target-weight-profile",
        default="constant",
        choices=["constant", "morning_evening_boost"],
        help=(
            "Optional time-varying district target weight. "
            "'morning_evening_boost' uses separate weights for hour labels 6-10 and 20-23."
        ),
    )
    parser.add_argument(
        "--district-target-weight-default",
        type=float,
        default=None,
        help="Default district target weight for --district-target-weight-profile morning_evening_boost.",
    )
    parser.add_argument(
        "--district-target-weight-morning",
        type=float,
        default=None,
        help="District target weight for hour labels 6-10.",
    )
    parser.add_argument(
        "--district-target-weight-evening",
        type=float,
        default=None,
        help="District target weight for hour labels 20-23.",
    )
    parser.add_argument(
        "--district-target-scale",
        type=float,
        default=None,
        help="Normalization for district target error. Defaults to max absolute target value.",
    )
    parser.add_argument(
        "--district-target-scale-mode",
        default="target_max",
        choices=[
            "target_max",
            "baseline_error_std",
            "baseline_error_iqr",
            "baseline_error_range",
            "baseline_error_maxabs",
            "controllable_error_minmax",
            "controllable_error_maxabs",
        ],
        help="Scale CE1 target error by target magnitude or baseline-vs-target error statistics.",
    )
    parser.add_argument(
        "--district-target-source",
        default="ce1",
        choices=["ce1", "mean_baseline"],
        help="Target profile source: 'ce1' uses the official CE1 target; 'mean_baseline' uses a flat line at mean baseline district load.",
    )
    parser.add_argument(
        "--district-target-mode",
        default="tracking",
        choices=["tracking", "peak_only"],
        help="District target mode for reward and MPC objective.",
    )
    parser.add_argument(
        "--raw-power-cost",
        action="store_true",
        help="Use RC heat power / power_scale in the MPC cost instead of normalized action u.",
    )
    parser.add_argument(
        "--occupant",
        default="all",
        choices=["all", "occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"],
        help="Occupant to run for occupant_present. Default runs all four.",
    )
    parser.add_argument(
        "--occupant-mode",
        default="both",
        choices=["without_tdyn", "bspline_tdyn", "both"],
        help="Run baseline occupant MPC, B-spline gated adaptive T_dyn MPC, or both.",
    )
    return parser


CONTROL_BUILDING_NAME = "resstock-amy2018-2021-release-1-247942"
OCCUPANCY_BUILDING_NAME = "resstock-amy2018-2021-release-1-20199"
FEB_START = 744
FEB_END = 1415

FITTED_OCCUPANT_PARAMS = {
    "occ1_tolerant": {"T_pref": 21.861014, "b": 1.198849, "max_prob": 0.4},
    "occ2_sensitive": {"T_pref": 23.035896, "b": 3.451184, "max_prob": 0.4},
    "occ3_cold": {"T_pref": 19.957358, "b": 2.618482, "max_prob": 0.4},
    "occ4_hot": {"T_pref": 23.909844, "b": 2.570147, "max_prob": 0.4},
}


def set_experiment_seed(seed: int) -> int:
    import numpy as np

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def make_env(
    *,
    sim_start: int,
    sim_end: int,
    seed: int,
    reward_function=None,
    reward_function_kwargs=None,
    building_name: str = CONTROL_BUILDING_NAME,
):
    from citylearn.citylearn import CityLearnEnv

    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / "data" / "datasets" / DATASET_NAME
    schema_path = dataset_dir / "schema.json"

    kwargs = dict(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=True,
        buildings=[building_name],
        simulation_start_time_step=int(sim_start),
        simulation_end_time_step=int(sim_end),
        random_seed=int(seed),
        active_actions=["heating_device"],
    )
    if reward_function is not None:
        kwargs["reward_function"] = reward_function
    if reward_function_kwargs is not None:
        kwargs["reward_function_kwargs"] = reward_function_kwargs

    return CityLearnEnv(**kwargs)


class OccupantComfortReward:
    """Same notebook reward wrapper around CityLearn ComfortReward."""

    def __new__(cls, *args, **kwargs):
        from citylearn.reward_function import ComfortReward

        class _OccupantComfortReward(ComfortReward):
            def calculate(self, observations):
                reward_list = []
                for o in observations:
                    if "override_setpoint" in o:
                        o["indoor_dry_bulb_temperature_heating_set_point"] = o["override_setpoint"]
                    reward_list.append(super().calculate([o])[0])
                return [sum(reward_list)] if self.central_agent else reward_list

        return _OccupantComfortReward(*args, **kwargs)


class Occupant:
    def __init__(self, T_pref, b, max_prob=0.4, duration=3):
        self.T_pref = T_pref
        self.b = b
        self.max_prob = max_prob
        self.duration = duration
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def prob_increase(self, Tin):
        import numpy as np

        d = max(0, self.T_pref - Tin)
        return min(0.002 * np.exp(self.b * d), self.max_prob)

    def prob_decrease(self, Tin):
        import numpy as np

        d = max(0, Tin - self.T_pref)
        return min(0.002 * np.exp(self.b * d), self.max_prob)

    @staticmethod
    def override_delta(p_override):
        if p_override < 0.2:
            return 0.5
        if p_override < 0.4:
            return 1.0
        return 2.0

    def clear_override(self):
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def step(self, Tin, scheduled_Tsp, occupied=True):
        import numpy as np

        self.last_override_started = False
        if not occupied:
            self.clear_override()
            return scheduled_Tsp

        if self.override_timer > 0:
            self.override_timer -= 1
            return self.override_setpoint

        p_inc = self.prob_increase(Tin)
        p_dec = self.prob_decrease(Tin)
        r = np.random.rand()

        if r < p_inc:
            new_sp = scheduled_Tsp + self.override_delta(p_inc)
        elif r < p_inc + p_dec:
            new_sp = scheduled_Tsp - self.override_delta(p_dec)
        else:
            self.override_setpoint = None
            return scheduled_Tsp

        self.override_setpoint = new_sp
        self.override_timer = max(self.duration - 1, 0)
        self.last_override_started = True
        return new_sp


class OccupantWrapper:
    def __new__(cls, *args, **kwargs):
        import gymnasium as gym

        class _OccupantWrapper(gym.Wrapper):
            def __init__(self, env, occupant):
                super().__init__(env)
                self.occupant = occupant

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.occupant.override_timer = 0
                self.occupant.override_setpoint = None
                self.occupant.last_override_started = False
                self.override_count = 0
                self.last_feedback = 0
                b = self.env.unwrapped.buildings[0]
                reset_future_heating_setpoints(b, b.time_step)
                self.effective_setpoints = [
                    float(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[b.time_step])
                ]
                self.baseline_setpoints = [
                    float(
                        b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            b.time_step
                        ]
                    )
                ]
                return obs, info

            def step(self, action):
                b = self.env.unwrapped.buildings[0]
                current_ix = b.time_step
                Tin = float(b.energy_simulation.indoor_dry_bulb_temperature[current_ix])
                next_ix = min(
                    current_ix + 1,
                    len(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point) - 1,
                )
                occupied_next = float(b.energy_simulation.occupant_count[next_ix]) > 0.0
                scheduled_next_Tsp = float(
                    b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[next_ix]
                )

                if occupied_next:
                    new_Tsp = self.occupant.step(Tin, scheduled_next_Tsp, occupied=True)
                else:
                    new_Tsp = self.occupant.step(Tin, scheduled_next_Tsp, occupied=False)

                feedback = 0
                if self.occupant.last_override_started:
                    self.override_count += 1
                    if new_Tsp > scheduled_next_Tsp:
                        feedback = +1
                    elif new_Tsp < scheduled_next_Tsp:
                        feedback = -1

                self.last_feedback = feedback

                apply_heating_override_schedule(
                    b,
                    next_ix,
                    new_Tsp,
                    active_override_steps(self.occupant),
                )
                obs, reward, terminated, truncated, info = self.env.step(action)

                self.effective_setpoints.append(
                    float(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[b.time_step])
                )
                self.baseline_setpoints.append(
                    float(
                        b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            b.time_step
                        ]
                    )
                )
                return obs, reward, terminated, truncated, info

        return _OccupantWrapper(*args, **kwargs)


def observation_index_map(env) -> dict[str, int]:
    names = env.unwrapped.buildings[0].active_observations

    def idx(name):
        return names.index(name)

    return {
        "tin": idx("indoor_dry_bulb_temperature"),
        "tout": idx("outdoor_dry_bulb_temperature"),
        "solar": idx("direct_solar_irradiance"),
        "tout_p": [idx(f"outdoor_dry_bulb_temperature_predicted_{h}") for h in [1, 2, 3]],
        "solar_p": [idx(f"direct_solar_irradiance_predicted_{h}") for h in [1, 2, 3]],
        "tmin": idx("indoor_dry_bulb_temperature_heating_set_point"),
        "tmax": idx("indoor_dry_bulb_temperature_cooling_set_point"),
    }


def fitted_occupants() -> dict[str, Occupant]:
    return {name: Occupant(**params) for name, params in FITTED_OCCUPANT_PARAMS.items()}


def district_target_weight_profile(
    *,
    simulation_start: int,
    simulation_end: int,
    base_weight: float,
    profile: str = "constant",
    default_weight: float | None = None,
    morning_weight: float | None = None,
    evening_weight: float | None = None,
) -> list[float]:
    """Build an MPC horizon-compatible district target weight profile."""

    start = int(simulation_start)
    end = int(simulation_end)
    if end < start:
        return []

    profile = str(profile).strip().lower()
    base = float(base_weight)
    if profile == "constant":
        return [base for _ in range(end - start + 1)]
    if profile != "morning_evening_boost":
        raise ValueError("district target weight profile must be 'constant' or 'morning_evening_boost'.")

    default = base if default_weight is None else float(default_weight)
    morning = default if morning_weight is None else float(morning_weight)
    evening = morning if evening_weight is None else float(evening_weight)
    values = []
    for absolute_step in range(start, end + 1):
        hour_label = absolute_step % 24 + 1
        if 6 <= hour_label <= 10:
            values.append(morning)
        elif 20 <= hour_label <= 23:
            values.append(evening)
        else:
            values.append(default)
    return values


def baseline_daily_mean_price_profile(
    baseline_load: list[float],
    *,
    simulation_start: int = FEB_START,
    normalize: str = "mean",
    exp_beta: float = 3.0,
) -> list[float]:
    """Repeat the baseline mean-daily district load as an hourly price profile."""

    import numpy as np

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
    elif normalize == "mean_centered":
        scale = max(float(np.nanmean(hourly)), 1e-6)
        hourly = hourly / scale - 1.0
    elif normalize == "exp_mean":
        scale = max(float(np.nanmean(hourly)), 1e-6)
        hourly = np.exp(float(exp_beta) * (hourly / scale - 1.0))
        hourly = hourly / max(float(np.nanmean(hourly)), 1e-6)
    elif normalize == "max":
        scale = max(float(np.nanmax(hourly)), 1e-6)
        hourly = hourly / scale
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be one of: mean, mean_centered, exp_mean, max, none.")

    return hourly[(hour_labels - 1).astype(int)].astype(float).tolist()


def setpoint_bias_daily_profile(
    *,
    simulation_start: int,
    simulation_end: int,
    amplitude: float = 1.0,
    preheat_hour_labels: tuple[int, ...] = (1, 2, 3, 4, 5, 14, 15, 16, 17),
    reduce_hour_labels: tuple[int, ...] = (7, 8, 9, 10, 18, 19, 20, 21, 22),
) -> list[float]:
    """Daily setpoint bias profile using 1..24 hour labels.

    Positive bias preheats during low-load/pre-peak windows. Negative bias
    relaxes tracking during morning/evening high-load windows.
    """

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


def dataset_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / "data" / "datasets" / DATASET_NAME
    return dataset_dir / "schema.json", dataset_dir


def load_building_occupancy_series(building_name: str = OCCUPANCY_BUILDING_NAME):
    import pandas as pd

    schema_path, dataset_dir = dataset_paths()
    schema = json.loads(schema_path.read_text())
    csv_path = dataset_dir / schema["buildings"][building_name]["energy_simulation"]
    return pd.read_csv(csv_path)["occupant_count"].astype(int).to_numpy()


def fit_bspline_occupancy_model(*, seed: int, building_name: str = OCCUPANCY_BUILDING_NAME):
    from src.occupants.bspline_preferences import FixedBSplineIMCOccupancyPredictor

    occupancy_series = load_building_occupancy_series(building_name)
    occupancy_model = FixedBSplineIMCOccupancyPredictor(
        period=24,
        spline_degree=3,
        n_internal_knots=4,
        l2_phi=1e-3,
        min_samples_per_state=25,
        maxiter=1000,
        random_state=int(seed),
    )
    occupancy_model.fit(occupancy_series)
    return occupancy_model, occupancy_series


def forecast_occupancy_from_bsplines(occupancy_model, current_occupancy: int, time_step: int, horizon: int, alpha=0.5):
    import numpy as np

    occ_prob_h, _ = occupancy_model.predict_occupied_probability(
        current_state=int(current_occupancy),
        start_t=int(time_step),
        horizon=int(horizon),
    )
    occ_prob_h = np.clip(np.asarray(occ_prob_h, dtype=float), 0.0, 1.0)
    occ_gate_h = (occ_prob_h > float(alpha)).astype(float)
    return occ_prob_h, occ_gate_h


def get_citylearn_kpi(district_kpis, name: str):
    rows = district_kpis[district_kpis["cost_function"] == name]["value"]
    return float(rows.iloc[0]) if len(rows) > 0 else None


def rollout_history_dataframe(env, result):
    import numpy as np
    import pandas as pd

    b = env.unwrapped.buildings[0]
    n = len(result["Tin_hist"])
    history = pd.DataFrame(
        {
            "time_step": np.arange(n, dtype=int),
            "indoor_temperature": result["Tin_hist"],
            "tmin": result["Tmin_hist"],
            "tmax": result["Tmax_hist"],
            "action": result["u_hist"],
            "price": result["price_hist"],
        }
    )
    optional_arrays = {
        "reward": "reward_hist",
        "scheduled_setpoint": "scheduled_setpoint_hist",
        "setpoint_bias": "setpoint_bias_hist",
        "tracking_setpoint": "tracking_setpoint_hist",
    }
    for column, key in optional_arrays.items():
        if key in result:
            history[column] = np.asarray(result[key][:n], dtype=float)
    if hasattr(env, "effective_setpoints"):
        history["effective_setpoint"] = np.asarray(env.effective_setpoints[:n], dtype=float)
    if hasattr(env, "baseline_setpoints"):
        history["baseline_setpoint"] = np.asarray(env.baseline_setpoints[:n], dtype=float)
    if hasattr(b, "net_electricity_consumption"):
        history["net_electricity_consumption"] = np.asarray(b.net_electricity_consumption[:n], dtype=float)
    if hasattr(b, "heating_electricity_consumption"):
        history["heating_electricity_consumption"] = np.asarray(b.heating_electricity_consumption[:n], dtype=float)
    optional_columns = {
        "baseline_action": "baseline_action_hist",
        "shift_signal": "shift_signal_hist",
        "daily_budget_debt": "daily_budget_debt_hist",
        "district_target_weight": "district_target_weight_hist",
        "reference_setpoint": "reference_setpoints",
        "tdyn": "T_dyn_hist",
        "occupant_feedback": "feedback_hist",
        "occ_now": "occ_now_hist",
        "occ_hat_1step": "occ_hat_hist",
        "occ_bin_1step": "occ_bin_hist",
    }
    for column, key in optional_columns.items():
        if key in result:
            history[column] = np.asarray(result[key][:n], dtype=float)
    return history


def run_february_mpc(
    *,
    controller_name,
    rc_params,
    rc2_params=None,
    objective_mode="baseline",
    fixed_band=None,
    w_energy=1.0,
    w_comfort=1.0,
    lambda_u=1e-3,
    lambda_du=0.0,
    use_smooth_du=False,
    comfort_band=2.0,
    comfort_scale=4.0,
    comfort_mode="band",
    comfort_track_weight=3.0,
    comfort_cold_weight=60.0,
    comfort_hot_weight=15.0,
    comfort_cold_tolerance=0.5,
    comfort_hot_tolerance=1.0,
    comfort_hot_tolerance_night=None,
    comfort_hot_tolerance_night_end=6,
    comfort_track_scale=2.0,
    comfort_cold_scale=1.5,
    comfort_hot_scale=2.0,
    comfort_cold_k=3.0,
    comfort_hot_k=1.0,
    power_scale=5.0,
    setpoint_bias_profile=None,
    time_bias_guard_band=2.0,
    ramp_scale=1.0,
    normalize_cost_terms=False,
    shift_weight=0.0,
    budget_weight=0.0,
    daily_budget_scale=1.0,
    preheat_weight=0.0,
    simple_tsp_lower_bias: float = 0.0,
    grid_stress_profile=None,
    future_grid_stress_profile=None,
    baseline_action_profile=None,
    district_target_weight=0.0,
    district_target_profile=None,
    district_background_profile=None,
    district_action_power_profile=None,
    district_target_weight_profile=None,
    district_target_scale=1.0,
    district_target_mode="tracking",
    cold_penalty=1e5,
    hot_penalty=1e3,
    occupant=None,
    reward_function=None,
    reward_function_kwargs=None,
    two_state_observer: str = "propagate",
    kalman_process_var: float = 0.05,
    kalman_measurement_var: float = 0.25,
    kalman_initial_var: float = 4.0,
    seed=49,
    max_steps=None,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
    building_name: str = CONTROL_BUILDING_NAME,
):
    import casadi as ca
    import do_mpc
    import numpy as np

    set_experiment_seed(seed)

    use_2r2c = rc2_params is not None
    two_state_observer = str(two_state_observer).strip().lower()
    if two_state_observer not in {"propagate", "kalman"}:
        raise ValueError(f"Unknown two_state_observer: {two_state_observer}")
    if use_2r2c:
        cop_a = rc2_params["cop_a"]
        cop_b = rc2_params["cop_b"]
        power_per_unit_u = rc2_params["power_per_unit_action"]
        _a11 = rc2_params["a11"]
        _a12 = rc2_params["a12"]
        _a21 = rc2_params["a21"]
        _a22 = rc2_params["a22"]
        _b_tout = rc2_params["b_tout"]
        _b_sol = rc2_params["b_sol"]
        _b_heat = rc2_params["b_heat"]
        _b0 = rc2_params["b0"]
    else:
        A = rc_params["A"]
        Bd0 = rc_params["Bd0"]
        Bd1 = rc_params["Bd1"]
        Bd2 = rc_params["Bd2"]
        Bu = rc_params["Bu"]
        Tm = rc_params["mean_indoor_temperature"]
        cop_a = rc_params["cop_a"]
        cop_b = rc_params["cop_b"]
        power_per_unit_u = rc_params["power_per_unit_action"]
    objective_mode = str(objective_mode).strip().lower()
    use_budget_debt = objective_mode == "budget_shift"

    model = do_mpc.model.Model("discrete")
    Tin_m = model.set_variable("_x", "Tin")
    if use_2r2c:
        Tmass_m = model.set_variable("_x", "Tmass")
    else:
        Tmass_m = None
    if use_smooth_du:
        u_prev = model.set_variable("_x", "u_prev")
    else:
        u_prev = None
    if use_budget_debt:
        budget_debt_m = model.set_variable("_x", "budget_debt")
    else:
        budget_debt_m = None

    u_m = model.set_variable("_u", "u")
    Tout_m = model.set_variable("_tvp", "Tout")
    Sol_m = model.set_variable("_tvp", "Sol")
    Price_m = model.set_variable("_tvp", "Price")
    GridStress_m = model.set_variable("_tvp", "GridStress")
    FutureGridStress_m = model.set_variable("_tvp", "FutureGridStress")
    BaselineAction_m = model.set_variable("_tvp", "BaselineAction")
    BudgetPenalty_m = model.set_variable("_tvp", "BudgetPenalty")
    DistrictTarget_m = model.set_variable("_tvp", "DistrictTarget")
    DistrictBackground_m = model.set_variable("_tvp", "DistrictBackground")
    DistrictActionPower_m = model.set_variable("_tvp", "DistrictActionPower")
    DistrictTargetWeight_m = model.set_variable("_tvp", "DistrictTargetWeight")
    Tsp_m = model.set_variable("_tvp", "Tsp")
    Tmin_m = model.set_variable("_tvp", "Tmin")
    Tmax_m = model.set_variable("_tvp", "Tmax")
    HotTol_m = model.set_variable("_tvp", "HotTol")
    SetpointBias_m = model.set_variable("_tvp", "SetpointBias")
    w_est = model.set_variable("_tvp", "w_est")

    COP_expr = cop_a + cop_b * Tout_m
    COP_expr = ca.fmin(ca.fmax(COP_expr, 0.5), 6.0)
    if use_2r2c:
        Tin_next_m = (_a11 * Tin_m + _a12 * Tmass_m + _b_tout * Tout_m
                      + _b_sol * Sol_m + _b_heat * COP_expr * u_m + _b0 + w_est)
        Tmass_next_m = _a21 * Tin_m + _a22 * Tmass_m
    else:
        Tin_next_m = A * Tin_m + Bd0 * Tm + Bd1 * Tout_m + Bd2 * Sol_m + Bu * COP_expr * u_m + w_est

    model.set_expression("Tin_occ", Tin_next_m)
    model.set_rhs("Tin", Tin_next_m)
    if use_2r2c:
        model.set_rhs("Tmass", Tmass_next_m)
    if use_smooth_du:
        model.set_rhs("u_prev", u_m)
    if use_budget_debt:
        model.set_rhs("budget_debt", budget_debt_m + (u_m - BaselineAction_m))
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.supress_ipopt_output()
    setup_mpc = {
        "n_horizon": 24,
        "t_step": 1.0,
        "state_discretization": "discrete",
        "store_full_solution": True,
    }
    mpc.set_param(**setup_mpc)

    if objective_mode == "baseline":
        Tref = Tmin_m
        stage_cost = (Tin_m - Tref) ** 2 + lambda_u * u_m**2
        terminal_cost = (Tin_m - Tref) ** 2
    elif objective_mode == "balanced":
        comfort_mode = str(comfort_mode).strip().lower()
        if comfort_mode == "setpoint":
            comfort_error = Tin_m - Tsp_m
            comfort_cost = comfort_error**2 / comfort_scale
        elif comfort_mode == "asymmetric":
            track = comfort_track_weight * ((Tin_m - Tsp_m) / max(float(comfort_track_scale), 1e-6)) ** 2
            cold_violation = ca.fmax(Tsp_m - float(comfort_cold_tolerance) - Tin_m, 0.0)
            hot_violation = ca.fmax(Tin_m - Tsp_m - HotTol_m, 0.0)
            cold_cost = comfort_cold_weight * (cold_violation / max(float(comfort_cold_scale), 1e-6)) ** 2
            hot_cost = comfort_hot_weight * (hot_violation / max(float(comfort_hot_scale), 1e-6)) ** 2
            comfort_cost = track + cold_cost + hot_cost
        elif comfort_mode == "band":
            lower_comfort = Tsp_m - comfort_band
            upper_comfort = Tsp_m + comfort_band
            cold_violation = ca.fmax(lower_comfort - Tin_m, 0.0)
            hot_violation = ca.fmax(Tin_m - upper_comfort, 0.0)
            comfort_cost = (cold_violation**2 + hot_violation**2) / comfort_scale
        elif comfort_mode == "barrier":
            # Exponential cold barrier + quadratic hot penalty.
            # exp(k*v) - 1 grows catastrophically fast with cold violation v, making
            # it impossible for the district objective to "buy" underheating.
            cold_violation = ca.fmax(Tsp_m - float(comfort_cold_tolerance) - Tin_m, 0.0)
            hot_violation = ca.fmax(Tin_m - Tsp_m - float(comfort_hot_tolerance), 0.0)
            cold_cost = comfort_cold_weight * (ca.exp(float(comfort_cold_k) * cold_violation) - 1.0)
            hot_cost = comfort_hot_weight * (hot_violation / max(float(comfort_hot_scale), 1e-6)) ** 2
            comfort_cost = cold_cost + hot_cost
        elif comfort_mode == "time_bias_guard":
            biased_setpoint = Tsp_m + SetpointBias_m
            track = comfort_track_weight * ((Tin_m - biased_setpoint) / max(float(comfort_track_scale), 1e-6)) ** 2
            guard = max(float(time_bias_guard_band), 0.0)
            cold_guard = ca.fmax(Tsp_m - guard - Tin_m, 0.0)
            hot_guard = ca.fmax(Tin_m - Tsp_m - guard, 0.0)
            cold_cost = comfort_cold_weight * (cold_guard / max(float(comfort_cold_scale), 1e-6)) ** 2
            hot_cost = comfort_hot_weight * (hot_guard / max(float(comfort_hot_scale), 1e-6)) ** 2
            comfort_cost = track + cold_cost + hot_cost
        else:
            raise ValueError("comfort_mode must be 'band', 'setpoint', 'asymmetric', 'barrier', or 'time_bias_guard'.")
        heat_power = u_m if normalize_cost_terms else power_per_unit_u * u_m / power_scale
        energy_cost = heat_power
        shift_cost = GridStress_m * heat_power
        preheat_signal = ca.fmax(1.0 - GridStress_m, 0.0) * FutureGridStress_m
        preheat_credit = preheat_signal * heat_power
        peak_cost = heat_power**2
        ramp_cost = (u_m - u_prev) ** 2 if use_smooth_du else 0.0
        target_error = (DistrictBackground_m + DistrictActionPower_m * u_m - DistrictTarget_m) / max(
            float(district_target_scale),
            1e-6,
        )
        if str(district_target_mode).strip().lower() == "peak_only":
            target_error = ca.fmax(target_error, 0.0)
        district_target_cost = target_error**2
        stage_cost = (
            w_comfort * comfort_cost
            + w_energy * energy_cost
            + shift_weight * shift_cost
            - preheat_weight * preheat_credit
            + lambda_u * peak_cost
            + lambda_du * ramp_cost
            + DistrictTargetWeight_m * district_target_cost
        )
        terminal_cost = w_comfort * comfort_cost
    elif objective_mode == "budget_shift":
        cold_violation = ca.fmax(Tmin_m - Tin_m, 0.0)
        hot_violation = ca.fmax(Tin_m - Tmax_m, 0.0)
        comfort_cost = (cold_violation**2 + hot_violation**2) / max(float(comfort_scale), 1e-6)
        shift_cost = GridStress_m * (u_m - BaselineAction_m)
        budget_next = budget_debt_m + (u_m - BaselineAction_m)
        budget_cost = BudgetPenalty_m * (budget_next / max(float(daily_budget_scale), 1e-6)) ** 2
        if use_smooth_du:
            ramp_error = (u_m - u_prev) / max(float(ramp_scale), 1e-6)
            ramp_cost = ca.sqrt(ramp_error**2 + 1e-8)
        else:
            ramp_cost = 0.0
        stage_cost = (
            w_comfort * comfort_cost
            + shift_weight * shift_cost
            + budget_weight * budget_cost
            + lambda_du * ramp_cost
        )
        terminal_cost = w_comfort * comfort_cost
    elif objective_mode == "energy_comfort":
        energy_cost = Price_m * power_per_unit_u * u_m
        comfort_cost = (Tin_m - Tsp_m) ** 2
        smooth_cost = lambda_u * u_m**2
        if use_smooth_du:
            smooth_cost = smooth_cost + lambda_du * (u_m - u_prev) ** 2
        stage_cost = w_energy * energy_cost + w_comfort * comfort_cost + smooth_cost
        terminal_cost = w_comfort * comfort_cost
    elif objective_mode == "simple":
        # Two terms, both normalized to [0, 1] so equal weights (0.5/0.5) are meaningful.
        # Comfort: setpoint-tracking squared error divided by band^2.
        #          T_track = Tsp - simple_tsp_lower_bias shifts the target below the scheduled
        #          setpoint, reducing heating need in cold climates at the cost of slight
        #          cold discomfort. Default 0.0 = track Tsp exactly (v3 behaviour).
        # Energy:  GridStress(t) * u, where GridStress is the mean hourly district load
        #          (exp-normalised, peak > 1.0, off-peak ≈ 0.25), giving the MPC a strong
        #          incentive to pre-heat during cheap hours and coast during peaks.
        T_track_m = Tsp_m - float(simple_tsp_lower_bias)
        comfort_cost = (Tin_m - T_track_m) ** 2 / max(float(comfort_band) ** 2, 1e-6)
        energy_cost = GridStress_m * u_m
        stage_cost = w_comfort * comfort_cost + w_energy * energy_cost
        terminal_cost = w_comfort * comfort_cost
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    if objective_mode == "baseline" and use_smooth_du:
        stage_cost = stage_cost + lambda_du * (u_m - u_prev) ** 2

    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    mpc.bounds["lower", "_u", "u"] = 0.0
    mpc.bounds["upper", "_u", "u"] = 1.0
    if not (
        objective_mode == "budget_shift"
        or (
            objective_mode == "balanced"
            and str(comfort_mode).strip().lower() in {"setpoint", "asymmetric", "barrier"}
        )
    ):
        # For simple mode, shift the cold-penalty threshold down by simple_tsp_lower_bias.
        # Without this, the 1e5 penalty at Tin < Tmin completely overpowers the cost function,
        # making the tracking-reference shift in the comfort term have zero effect.
        cold_threshold_m = (
            (Tmin_m - float(simple_tsp_lower_bias)) if objective_mode == "simple" else Tmin_m
        )
        mpc.set_nl_cons("too_cold", cold_threshold_m - Tin_m, ub=0.0, soft_constraint=True, penalty_term_cons=cold_penalty)
        mpc.set_nl_cons("too_hot", Tin_m - Tmax_m, ub=0.0, soft_constraint=True, penalty_term_cons=hot_penalty)

    N = setup_mpc["n_horizon"]
    _hot_tol_night = comfort_hot_tolerance_night if comfort_hot_tolerance_night is not None else comfort_hot_tolerance
    _hot_tol_night_end = int(comfort_hot_tolerance_night_end)
    tvp_template = mpc.get_tvp_template()
    latest = {
        "Tout0": 0.0,
        "Sol0": 0.0,
        "Price0": 0.0,
        "Tsp0": 20.0,
        "Toutp": [0.0, 0.0, 0.0],
        "Solp": [0.0, 0.0, 0.0],
        "Pricep": [0.0, 0.0, 0.0],
        "Tspp": [20.0, 20.0, 20.0],
        "Tmin0": 18.0,
        "Tmax0": 24.0,
        "w_est": 0.0,
        "GridStress": [1.0 if objective_mode == "simple" else 0.0 for _ in range(N + 1)],
        "FutureGridStress": [0.0 for _ in range(N + 1)],
        "BaselineAction": [0.0 for _ in range(N + 1)],
        "BudgetPenalty": [0.0 for _ in range(N + 1)],
        "DistrictTarget": [0.0 for _ in range(N + 1)],
        "DistrictBackground": [0.0 for _ in range(N + 1)],
        "DistrictActionPower": [0.0 for _ in range(N + 1)],
        "DistrictTargetWeight": [float(district_target_weight) for _ in range(N + 1)],
        "HotTol": [float(comfort_hot_tolerance) for _ in range(N + 1)],
        "SetpointBias": [0.0 for _ in range(N + 1)],
    }

    def tvp_fun(_t_now):
        for k in range(N + 1):
            if k == 0:
                Tout_f, Sol_f, Price_f, Tsp_f = (
                    latest["Tout0"],
                    latest["Sol0"],
                    latest["Price0"],
                    latest["Tsp0"],
                )
            elif k <= 3:
                Tout_f, Sol_f, Price_f, Tsp_f = (
                    latest["Toutp"][k - 1],
                    latest["Solp"][k - 1],
                    latest["Pricep"][k - 1],
                    latest["Tspp"][k - 1],
                )
            else:
                Tout_f, Sol_f, Price_f, Tsp_f = (
                    latest["Toutp"][-1],
                    latest["Solp"][-1],
                    latest["Pricep"][-1],
                    latest["Tspp"][-1],
                )

            tvp_template["_tvp", k, "Tout"] = float(Tout_f)
            tvp_template["_tvp", k, "Sol"] = float(Sol_f)
            tvp_template["_tvp", k, "Price"] = float(Price_f)
            tvp_template["_tvp", k, "GridStress"] = float(latest["GridStress"][k])
            tvp_template["_tvp", k, "FutureGridStress"] = float(latest["FutureGridStress"][k])
            tvp_template["_tvp", k, "BaselineAction"] = float(latest["BaselineAction"][k])
            tvp_template["_tvp", k, "BudgetPenalty"] = float(latest["BudgetPenalty"][k])
            tvp_template["_tvp", k, "DistrictTarget"] = float(latest["DistrictTarget"][k])
            tvp_template["_tvp", k, "DistrictBackground"] = float(latest["DistrictBackground"][k])
            tvp_template["_tvp", k, "DistrictActionPower"] = float(latest["DistrictActionPower"][k])
            tvp_template["_tvp", k, "DistrictTargetWeight"] = float(latest["DistrictTargetWeight"][k])
            tvp_template["_tvp", k, "HotTol"] = float(latest["HotTol"][k])
            tvp_template["_tvp", k, "SetpointBias"] = float(latest["SetpointBias"][k])
            tvp_template["_tvp", k, "Tsp"] = float(Tsp_f)
            tvp_template["_tvp", k, "Tmin"] = float(latest["Tmin0"])
            tvp_template["_tvp", k, "Tmax"] = float(latest["Tmax0"])
            tvp_template["_tvp", k, "w_est"] = float(latest["w_est"])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    env = make_env(
        sim_start=sim_start,
        sim_end=sim_end,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
    )
    if occupant is not None:
        env = OccupantWrapper(env, occupant)

    obs, _ = env.reset(seed=seed)
    building = env.unwrapped.buildings[0]
    obs_idx = observation_index_map(env)
    mpc.reset_history()

    Tin_hist, Tmin_hist, Tmax_hist = [], [], []
    scheduled_setpoint_hist, setpoint_bias_hist, tracking_setpoint_hist = [], [], []
    u_hist, price_hist, reward_hist = [], [], []
    feedback_hist = []
    baseline_action_hist, shift_signal_hist, daily_budget_debt_hist = [], [], []
    district_target_weight_hist = []
    w_est_val = 0.0
    daily_budget_debt = 0.0
    gamma_w = 0.2
    prev_Tin = None
    prev_u = None
    prev_Tout = None
    prev_Sol = None
    Tmass_val = None  # 2R2C hidden thermal mass state (initialised at first step)
    Tmass_var = max(float(kalman_initial_var), 1e-12)
    last_u = 0.0
    if use_budget_debt and baseline_action_profile:
        last_u = float(baseline_action_profile[0])
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        Tin0 = float(o[obs_idx["tin"]])
        Tout0 = float(o[obs_idx["tout"]])
        Sol0 = float(o[obs_idx["solar"]])
        time_step = building.time_step
        Price0 = float(building.pricing.electricity_pricing[time_step])
        scheduled_setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
        Tsp0 = float(scheduled_setpoints[time_step])
        Toutp = [float(o[j]) for j in obs_idx["tout_p"]]
        Solp = [float(o[j]) for j in obs_idx["solar_p"]]
        Pricep = [
            float(building.pricing.electricity_pricing_predicted_1[time_step]),
            float(building.pricing.electricity_pricing_predicted_2[time_step]),
            float(building.pricing.electricity_pricing_predicted_3[time_step]),
        ]
        Tspp = [float(scheduled_setpoints[min(time_step + h, len(scheduled_setpoints) - 1)]) for h in [1, 2, 3]]

        if objective_mode == "budget_shift":
            Tmin0 = 20.0
            Tmax0 = 24.0
        elif objective_mode == "balanced" and str(comfort_mode).strip().lower() == "band":
            Tmin0 = Tsp0 - float(comfort_band)
            Tmax0 = Tsp0 + float(comfort_band)
        elif objective_mode == "balanced" and str(comfort_mode).strip().lower() == "setpoint":
            Tmin0 = Tsp0
            Tmax0 = Tsp0
        elif objective_mode == "balanced" and str(comfort_mode).strip().lower() == "time_bias_guard":
            guard = max(float(time_bias_guard_band), 0.0)
            Tmin0 = Tsp0 - guard
            Tmax0 = Tsp0 + guard
        elif objective_mode == "balanced" and str(comfort_mode).strip().lower() == "asymmetric":
            Tmin0 = Tsp0 - float(comfort_cold_tolerance)
            _cur_hot_tol = (
                float(_hot_tol_night) if (sim_start + step_count) % 24 < _hot_tol_night_end
                else float(comfort_hot_tolerance)
            )
            Tmax0 = Tsp0 + _cur_hot_tol
        elif fixed_band is None:
            Tmin0 = float(o[obs_idx["tmin"]])
            Tmax0 = float(o[obs_idx["tmax"]])
        else:
            Tmin0, Tmax0 = map(float, fixed_band)

        if use_2r2c and Tmass_val is None:
            Tmass_val = Tin0  # cold-start: mass in equilibrium with air

        if prev_Tin is not None:
            cop_prev = np.clip(cop_a + cop_b * prev_Tout, 0.5, 6.0)
            if use_2r2c:
                Tin_pred = (_a11 * prev_Tin + _a12 * Tmass_val + _b_tout * prev_Tout
                            + _b_sol * prev_Sol + _b_heat * cop_prev * prev_u + _b0 + w_est_val)
                if two_state_observer == "kalman":
                    innovation = Tin0 - Tin_pred
                    innovation_var = _a12 * _a12 * Tmass_var + max(float(kalman_measurement_var), 1e-12)
                    gain = 0.0 if innovation_var <= 1e-12 else Tmass_var * _a12 / innovation_var
                    Tmass_post = Tmass_val + gain * innovation
                    Tmass_var_post = max((1.0 - gain * _a12) * Tmass_var, 1e-12)
                    Tmass_val = _a21 * prev_Tin + _a22 * Tmass_post
                    Tmass_var = _a22 * _a22 * Tmass_var_post + max(float(kalman_process_var), 1e-12)
                else:
                    Tmass_val = _a21 * prev_Tin + _a22 * Tmass_val
            else:
                Tin_pred = A * prev_Tin + Bd0 * Tm + Bd1 * prev_Tout + Bd2 * prev_Sol + Bu * cop_prev * prev_u + w_est_val
            err = Tin0 - Tin_pred
            w_est_val = w_est_val + gamma_w * err

        if use_budget_debt and step_count % 24 == 0:
            daily_budget_debt = 0.0

        latest.update(
            {
                "Tout0": Tout0,
                "Sol0": Sol0,
                "Price0": Price0,
                "Tsp0": Tsp0,
                "Toutp": Toutp,
                "Solp": Solp,
                "Pricep": Pricep,
                "Tspp": Tspp,
                "Tmin0": Tmin0,
                "Tmax0": Tmax0,
                "w_est": w_est_val,
                "GridStress": [
                    (1.0 if objective_mode == "simple" else 0.0)
                    if not grid_stress_profile
                    else float(grid_stress_profile[min(step_count + h, len(grid_stress_profile) - 1)])
                    for h in range(N + 1)
                ],
                "FutureGridStress": [
                    0.0
                    if not future_grid_stress_profile
                    else float(future_grid_stress_profile[min(step_count + h, len(future_grid_stress_profile) - 1)])
                    for h in range(N + 1)
                ],
                "BaselineAction": [
                    0.0
                    if not baseline_action_profile
                    else float(baseline_action_profile[min(step_count + h, len(baseline_action_profile) - 1)])
                    for h in range(N + 1)
                ],
                "BudgetPenalty": [
                    1.0 if ((step_count + h) % 24 + 1) == 24 else 0.0
                    for h in range(N + 1)
                ],
                "DistrictTarget": [
                    0.0
                    if not district_target_profile
                    else float(district_target_profile[min(step_count + h, len(district_target_profile) - 1)])
                    for h in range(N + 1)
                ],
                "DistrictBackground": [
                    0.0
                    if not district_background_profile
                    else float(district_background_profile[min(step_count + h, len(district_background_profile) - 1)])
                    for h in range(N + 1)
                ],
                "DistrictActionPower": [
                    0.0
                    if not district_action_power_profile
                    else float(district_action_power_profile[min(step_count + h, len(district_action_power_profile) - 1)])
                    for h in range(N + 1)
                ],
                "DistrictTargetWeight": [
                    float(district_target_weight)
                    if not district_target_weight_profile
                    else float(district_target_weight_profile[min(step_count + h, len(district_target_weight_profile) - 1)])
                    for h in range(N + 1)
                ],
                "HotTol": [
                    float(_hot_tol_night) if (sim_start + step_count + h) % 24 < _hot_tol_night_end
                    else float(comfort_hot_tolerance)
                    for h in range(N + 1)
                ],
                "SetpointBias": [
                    0.0
                    if not setpoint_bias_profile
                    else float(setpoint_bias_profile[min(step_count + h, len(setpoint_bias_profile) - 1)])
                    for h in range(N + 1)
                ],
            }
        )

        x_values = [Tin0]
        if use_2r2c:
            x_values.append(float(Tmass_val) if Tmass_val is not None else Tin0)
        if use_smooth_du:
            x_values.append(last_u)
        if use_budget_debt:
            x_values.append(daily_budget_debt)
        x0 = np.asarray(x_values, dtype=float).reshape(-1, 1)
        u0 = float(np.asarray(mpc.make_step(x0)).reshape(-1)[0])
        obs, reward, terminated, truncated, _ = env.step([[u0]])
        baseline_action0 = float(latest["BaselineAction"][0])
        shift_signal0 = float(latest["GridStress"][0])
        if use_budget_debt:
            daily_budget_debt = daily_budget_debt + (u0 - baseline_action0)

        prev_Tin = Tin0
        prev_u = u0
        prev_Tout = Tout0
        prev_Sol = Sol0
        last_u = u0
        Tin_hist.append(Tin0)
        Tmin_hist.append(Tmin0)
        Tmax_hist.append(Tmax0)
        scheduled_setpoint_hist.append(Tsp0)
        setpoint_bias_hist.append(float(latest["SetpointBias"][0]))
        tracking_setpoint_hist.append(Tsp0 + float(latest["SetpointBias"][0]))
        u_hist.append(u0)
        price_hist.append(Price0)
        reward_hist.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))
        feedback_hist.append(float(getattr(env, "last_feedback", 0)))
        baseline_action_hist.append(baseline_action0)
        shift_signal_hist.append(shift_signal0)
        daily_budget_debt_hist.append(daily_budget_debt)
        district_target_weight_hist.append(float(latest["DistrictTargetWeight"][0]))

        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    if max_steps is None:
        kpis = env.unwrapped.evaluate().copy()
        district_kpis = kpis[kpis["level"] == "district"].copy()
    else:
        import pandas as pd

        kpis = pd.DataFrame(columns=["cost_function", "value", "level", "name"])
        district_kpis = kpis.copy()
    result = {
        "name": controller_name,
        "env": env,
        "kpis": kpis,
        "district_kpis": district_kpis,
        "Tin_hist": np.asarray(Tin_hist, dtype=float),
        "Tmin_hist": np.asarray(Tmin_hist, dtype=float),
        "Tmax_hist": np.asarray(Tmax_hist, dtype=float),
        "scheduled_setpoint_hist": np.asarray(scheduled_setpoint_hist, dtype=float),
        "setpoint_bias_hist": np.asarray(setpoint_bias_hist, dtype=float),
        "tracking_setpoint_hist": np.asarray(tracking_setpoint_hist, dtype=float),
        "u_hist": np.asarray(u_hist, dtype=float),
        "price_hist": np.asarray(price_hist, dtype=float),
        "reward_hist": np.asarray(reward_hist, dtype=float),
        "feedback_hist": np.asarray(feedback_hist, dtype=float),
        "baseline_action_hist": np.asarray(baseline_action_hist, dtype=float),
        "shift_signal_hist": np.asarray(shift_signal_hist, dtype=float),
        "daily_budget_debt_hist": np.asarray(daily_budget_debt_hist, dtype=float),
        "district_target_weight_hist": np.asarray(district_target_weight_hist, dtype=float),
        "override_count": int(getattr(env, "override_count", 0)),
    }
    if hasattr(env, "baseline_setpoints"):
        result["baseline_setpoints"] = np.asarray(env.baseline_setpoints, dtype=float)
    if hasattr(env, "effective_setpoints"):
        result["effective_setpoints"] = np.asarray(env.effective_setpoints, dtype=float)
    if occupant is not None:
        comfort_low, comfort_high = occupant_probability_comfort_band(occupant)
        result["target_temperature"] = float(occupant.T_pref)
        result["comfort_band_lower"] = float(comfort_low)
        result["comfort_band_upper"] = float(comfort_high)
    else:
        result["comfort_band_lower"] = 20.0
        result["comfort_band_upper"] = 24.0

    print(f"{controller_name} run finished")
    print(f"Overrides: {result['override_count']}")
    print(district_kpis[["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return result


def run_february_bspline_tdyn_mpc(
    *,
    controller_name,
    rc_params,
    occupant,
    fixed_band=None,
    alpha_occ=0.5,
    lambda_u=1e-3,
    lambda_du=0.0,
    use_smooth_du=False,
    cold_penalty=1e5,
    hot_penalty=1e3,
    reward_function=None,
    reward_function_kwargs=None,
    seed=49,
    max_steps=None,
    building_name: str = CONTROL_BUILDING_NAME,
    occupancy_building_name: str = OCCUPANCY_BUILDING_NAME,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
    comfort_track_weight: float = 0.1,
    comfort_cold_weight: float = 0.7,
    comfort_hot_weight: float = 0.2,
    comfort_track_scale: float = 2.0,
    comfort_cold_scale: float = 1.5,
    comfort_hot_scale: float = 2.0,
    time_bias_guard_band: float = 2.0,
    setpoint_bias_profile=None,
    T_dyn_init_mode="schedule",
    delta_up=0.5,
    delta_down=0.5,
    drift_to_pref=0.05,
    T_dyn_min=18.0,
    T_dyn_max=26.0,
):
    """Run B-spline diagnostic MPC with occupant overrides in the loop.

    For the B-spline Tdyn branch the controller reference is occupancy gated:
    schedule when predicted unoccupied, Tdyn when predicted occupied.
    """

    import casadi as ca
    import do_mpc
    import numpy as np

    set_experiment_seed(seed)

    A = rc_params["A"]
    Bd0 = rc_params["Bd0"]
    Bd1 = rc_params["Bd1"]
    Bd2 = rc_params["Bd2"]
    Bu = rc_params["Bu"]
    Tm = rc_params["mean_indoor_temperature"]
    cop_a = rc_params["cop_a"]
    cop_b = rc_params["cop_b"]

    occupancy_model, occupancy_series = fit_bspline_occupancy_model(
        seed=seed,
        building_name=occupancy_building_name,
    )
    occupant_preference = float(occupant.T_pref)

    model = do_mpc.model.Model("discrete")
    Tin_m = model.set_variable("_x", "Tin")
    if use_smooth_du:
        u_prev = model.set_variable("_x", "u_prev")
    else:
        u_prev = None

    u_m = model.set_variable("_u", "u")
    Tout_m = model.set_variable("_tvp", "Tout")
    Sol_m = model.set_variable("_tvp", "Sol")
    Tref_m = model.set_variable("_tvp", "Tdyn")
    Tmin_m = model.set_variable("_tvp", "Tmin")
    Tmax_m = model.set_variable("_tvp", "Tmax")
    SetpointBias_m = model.set_variable("_tvp", "SetpointBias")
    occ_bin_m = model.set_variable("_tvp", "occ_bin")
    occ_hat_m = model.set_variable("_tvp", "occ_hat")
    w_est = model.set_variable("_tvp", "w_est")

    COP_expr = cop_a + cop_b * Tout_m
    COP_expr = ca.fmin(ca.fmax(COP_expr, 0.5), 6.0)
    Tin_next_m = A * Tin_m + Bd0 * Tm + Bd1 * Tout_m + Bd2 * Sol_m + Bu * COP_expr * u_m + w_est

    model.set_expression("Tin_occ", Tin_next_m)
    model.set_rhs("Tin", Tin_next_m)
    if use_smooth_du:
        model.set_rhs("u_prev", u_m)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.supress_ipopt_output()
    setup_mpc = {
        "n_horizon": 24,
        "t_step": 1.0,
        "state_discretization": "discrete",
        "store_full_solution": True,
    }
    mpc.set_param(**setup_mpc)

    biased_reference_m = Tref_m + SetpointBias_m
    track = comfort_track_weight * ((Tin_m - biased_reference_m) / max(float(comfort_track_scale), 1e-6)) ** 2
    cold_guard = ca.fmax(Tref_m - float(time_bias_guard_band) - Tin_m, 0.0)
    hot_guard = ca.fmax(Tin_m - Tref_m - float(time_bias_guard_band), 0.0)
    cold_cost = comfort_cold_weight * (cold_guard / max(float(comfort_cold_scale), 1e-6)) ** 2
    hot_cost = comfort_hot_weight * (hot_guard / max(float(comfort_hot_scale), 1e-6)) ** 2
    comfort_cost = track + cold_cost + hot_cost
    stage_cost = comfort_cost + lambda_u * u_m**2
    terminal_cost = comfort_cost
    if use_smooth_du:
        stage_cost = stage_cost + lambda_du * (u_m - u_prev) ** 2

    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    mpc.bounds["lower", "_u", "u"] = 0.0
    mpc.bounds["upper", "_u", "u"] = 1.0
    mpc.set_nl_cons("too_cold", Tmin_m - Tin_m, ub=0.0, soft_constraint=True, penalty_term_cons=cold_penalty)
    mpc.set_nl_cons("too_hot", Tin_m - Tmax_m, ub=0.0, soft_constraint=True, penalty_term_cons=hot_penalty)

    N = setup_mpc["n_horizon"]
    tvp_template = mpc.get_tvp_template()
    latest = {
        "Tout0": 0.0,
        "Sol0": 0.0,
        "Tdyn": 21.0,
        "Tmin0": 18.0,
        "Tmax0": 26.0,
        "Tdyn_profile": np.full(N + 1, 21.0, dtype=float),
        "Tmin_profile": np.full(N + 1, 18.0, dtype=float),
        "Tmax_profile": np.full(N + 1, 26.0, dtype=float),
        "occ_hat": np.zeros(N + 1, dtype=float),
        "occ_bin": np.zeros(N + 1, dtype=float),
        "SetpointBias": np.zeros(N + 1, dtype=float),
        "Toutp": [0.0, 0.0, 0.0],
        "Solp": [0.0, 0.0, 0.0],
        "w_est": 0.0,
    }

    def tvp_fun(_t_now):
        for k in range(N + 1):
            if k == 0:
                Tout_f, Sol_f = latest["Tout0"], latest["Sol0"]
            elif k <= 3:
                Tout_f, Sol_f = latest["Toutp"][k - 1], latest["Solp"][k - 1]
            else:
                Tout_f, Sol_f = latest["Toutp"][-1], latest["Solp"][-1]

            tvp_template["_tvp", k, "Tout"] = float(Tout_f)
            tvp_template["_tvp", k, "Sol"] = float(Sol_f)
            tvp_template["_tvp", k, "Tdyn"] = float(latest["Tdyn_profile"][k])
            tvp_template["_tvp", k, "Tmin"] = float(latest["Tmin_profile"][k])
            tvp_template["_tvp", k, "Tmax"] = float(latest["Tmax_profile"][k])
            tvp_template["_tvp", k, "SetpointBias"] = float(latest["SetpointBias"][k])
            tvp_template["_tvp", k, "occ_hat"] = float(latest["occ_hat"][k])
            tvp_template["_tvp", k, "occ_bin"] = float(latest["occ_bin"][k])
            tvp_template["_tvp", k, "w_est"] = float(latest["w_est"])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    env = make_env(
        sim_start=sim_start,
        sim_end=sim_end,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
    )
    env = OccupantWrapper(env, occupant)

    obs, _ = env.reset(seed=seed)
    building = env.unwrapped.buildings[0]
    obs_idx = observation_index_map(env)
    mpc.reset_history()

    scheduled_setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
    effective_setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point
    scheduled0 = float(scheduled_setpoints[building.time_step])
    if T_dyn_init_mode == "schedule":
        T_dyn = scheduled0
    elif T_dyn_init_mode == "pref":
        T_dyn = occupant_preference
    else:
        raise ValueError("T_dyn_init_mode must be 'schedule' or 'pref'.")
    T_dyn = float(np.clip(T_dyn, T_dyn_min, T_dyn_max))

    Tin_hist, Tmin_hist, Tmax_hist = [], [], []
    u_hist, price_hist, reward_hist = [], [], []
    T_dyn_hist, reference_hist, feedback_hist = [], [], []
    occ_now_hist, occ_hat_hist, occ_bin_hist = [], [], []
    w_est_val = 0.0
    gamma_w = 0.2
    prev_Tin = None
    prev_u = None
    prev_Tout = None
    prev_Sol = None
    last_u = 0.0
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        Tin0 = float(o[obs_idx["tin"]])
        Tout0 = float(o[obs_idx["tout"]])
        Sol0 = float(o[obs_idx["solar"]])
        time_step = int(building.time_step)
        Price0 = float(building.pricing.electricity_pricing[time_step])
        Toutp = [float(o[j]) for j in obs_idx["tout_p"]]
        Solp = [float(o[j]) for j in obs_idx["solar_p"]]

        if prev_Tin is not None:
            cop_prev = np.clip(cop_a + cop_b * prev_Tout, 0.5, 6.0)
            Tin_pred = A * prev_Tin + Bd0 * Tm + Bd1 * prev_Tout + Bd2 * prev_Sol + Bu * cop_prev * prev_u + w_est_val
            err = Tin0 - Tin_pred
            w_est_val = w_est_val + gamma_w * err

        absolute_time_step = int(sim_start) + int(step_count)
        occ_now = int(occupancy_series[min(absolute_time_step, len(occupancy_series) - 1)])
        occ_hat_h, occ_bin_h = forecast_occupancy_from_bsplines(
            occupancy_model=occupancy_model,
            current_occupancy=occ_now,
            time_step=absolute_time_step,
            horizon=N,
            alpha=alpha_occ,
        )
        occ_hat_full = np.zeros(N + 1, dtype=float)
        occ_bin_full = np.zeros(N + 1, dtype=float)
        occ_hat_full[0] = float(occ_now)
        occ_bin_full[0] = float(occ_now > alpha_occ)
        occ_hat_full[1:] = occ_hat_h
        occ_bin_full[1:] = occ_bin_h
        bias_full = np.asarray(
            [
                0.0
                if not setpoint_bias_profile
                else float(setpoint_bias_profile[min(step_count + h, len(setpoint_bias_profile) - 1)])
                for h in range(N + 1)
            ],
            dtype=float,
        )
        schedule_full = np.asarray(
            [
                float(scheduled_setpoints[min(time_step + h, len(scheduled_setpoints) - 1)])
                for h in range(N + 1)
            ],
            dtype=float,
        )
        tdyn_full = np.full(N + 1, float(T_dyn), dtype=float)
        reference_full = (1.0 - occ_bin_full) * schedule_full + occ_bin_full * tdyn_full

        if fixed_band is None:
            Tmin_profile = reference_full - float(time_bias_guard_band)
            Tmax_profile = reference_full + float(time_bias_guard_band)
        else:
            fixed_min, fixed_max = map(float, fixed_band)
            Tmin_profile = np.full(N + 1, fixed_min, dtype=float)
            Tmax_profile = np.full(N + 1, fixed_max, dtype=float)
        Tmin0 = float(Tmin_profile[0])
        Tmax0 = float(Tmax_profile[0])

        latest.update(
            {
                "Tout0": Tout0,
                "Sol0": Sol0,
                "Tdyn": T_dyn,
                "Tdyn_profile": reference_full,
                "Tmin0": Tmin0,
                "Tmax0": Tmax0,
                "Tmin_profile": Tmin_profile,
                "Tmax_profile": Tmax_profile,
                "occ_hat": occ_hat_full,
                "occ_bin": occ_bin_full,
                "SetpointBias": bias_full,
                "Toutp": Toutp,
                "Solp": Solp,
                "w_est": w_est_val,
            }
        )

        x0 = np.array([[Tin0], [last_u]]) if use_smooth_du else np.array([[Tin0]])
        u0 = float(np.asarray(mpc.make_step(x0)).reshape(-1)[0])
        obs, reward, terminated, truncated, _ = env.step([[u0]])

        feedback = int(getattr(env, "last_feedback", 0))
        T_dyn = update_tdyn_reference(
            T_dyn,
            feedback,
            preferred_temperature=occupant_preference,
            config=TDynConfig(
                delta_up=float(delta_up),
                delta_down=float(delta_down),
                drift_to_pref=float(drift_to_pref),
                lower_bound=float(T_dyn_min),
                upper_bound=float(T_dyn_max),
                anchor_feedback_to_pref=True,
            ),
        )

        prev_Tin = Tin0
        prev_u = u0
        prev_Tout = Tout0
        prev_Sol = Sol0
        last_u = u0
        Tin_hist.append(Tin0)
        Tmin_hist.append(Tmin0)
        Tmax_hist.append(Tmax0)
        u_hist.append(u0)
        price_hist.append(Price0)
        reward_hist.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))
        reference_hist.append(float(reference_full[0]))
        T_dyn_hist.append(float(T_dyn))
        feedback_hist.append(float(feedback))
        occ_now_hist.append(float(occ_now))
        occ_hat_hist.append(float(occ_hat_h[0]) if len(occ_hat_h) else 0.0)
        occ_bin_hist.append(float(occ_bin_h[0]) if len(occ_bin_h) else 0.0)

        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    if max_steps is None:
        kpis = env.unwrapped.evaluate().copy()
        district_kpis = kpis[kpis["level"] == "district"].copy()
    else:
        import pandas as pd

        kpis = pd.DataFrame(columns=["cost_function", "value", "level", "name"])
        district_kpis = kpis.copy()

    result = {
        "name": controller_name,
        "target_temperature": occupant_preference,
        "comfort_band_lower": float(occupant_probability_comfort_band(occupant)[0]),
        "comfort_band_upper": float(occupant_probability_comfort_band(occupant)[1]),
        "T_dyn_init_mode": T_dyn_init_mode,
        "env": env,
        "kpis": kpis,
        "district_kpis": district_kpis,
        "Tin_hist": np.asarray(Tin_hist, dtype=float),
        "Tmin_hist": np.asarray(Tmin_hist, dtype=float),
        "Tmax_hist": np.asarray(Tmax_hist, dtype=float),
        "u_hist": np.asarray(u_hist, dtype=float),
        "price_hist": np.asarray(price_hist, dtype=float),
        "reward_hist": np.asarray(reward_hist, dtype=float),
        "reference_setpoints": np.asarray(reference_hist, dtype=float),
        "T_dyn_hist": np.asarray(T_dyn_hist, dtype=float),
        "feedback_hist": np.asarray(feedback_hist, dtype=float),
        "occ_now_hist": np.asarray(occ_now_hist, dtype=float),
        "occ_hat_hist": np.asarray(occ_hat_hist, dtype=float),
        "occ_bin_hist": np.asarray(occ_bin_hist, dtype=float),
        "predicted_occupied_fraction": float(np.mean(occ_bin_hist)) if len(occ_bin_hist) else 0.0,
        "override_count": int(getattr(env, "override_count", 0)),
    }
    if hasattr(env, "baseline_setpoints"):
        result["baseline_setpoints"] = np.asarray(env.baseline_setpoints, dtype=float)
    if hasattr(env, "effective_setpoints"):
        result["effective_setpoints"] = np.asarray(env.effective_setpoints, dtype=float)

    print(f"{controller_name} run finished")
    print(f"Occupant T_pref: {occupant_preference:.3f} C")
    if len(result["T_dyn_hist"]):
        print(f"T_dyn mean/final: {result['T_dyn_hist'].mean():.3f} / {result['T_dyn_hist'][-1]:.3f} C")
    print(f"Predicted occupied fraction: {result['predicted_occupied_fraction']:.3f}")
    print(f"Overrides: {result['override_count']}")
    print(district_kpis[["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return result


def summarize_run(label, result):
    district = result["district_kpis"]
    comfort_low = float(result.get("comfort_band_lower", 20.0))
    comfort_high = float(result.get("comfort_band_upper", 24.0))
    summary = {
        "case": label,
        "override_count": int(result.get("override_count", 0)),
        "override_hours": override_hours(
            result.get("baseline_setpoints", []),
            result.get("effective_setpoints", []),
        ),
        "reward_total": float(result["reward_hist"].sum()) if len(result["reward_hist"]) else 0.0,
        "kelvin_hours": kelvin_hours(result.get("Tin_hist", []), lower=comfort_low, upper=comfort_high),
        "setpoint_kelvin_hours": setpoint_kelvin_hours(
            result.get("Tin_hist", []),
            result.get("effective_setpoints", []),
        ),
        "electricity_consumption_total": get_citylearn_kpi(district, "electricity_consumption_total"),
        "discomfort_proportion": get_citylearn_kpi(district, "discomfort_proportion"),
        "discomfort_cold_proportion": get_citylearn_kpi(district, "discomfort_cold_proportion"),
        "discomfort_hot_proportion": get_citylearn_kpi(district, "discomfort_hot_proportion"),
        "daily_peak_average": get_citylearn_kpi(district, "daily_peak_average"),
    }
    if "target_temperature" in result:
        summary["target_temperature"] = float(result["target_temperature"])
    if "comfort_band_lower" in result:
        summary["comfort_band_lower"] = float(result["comfort_band_lower"])
    if "comfort_band_upper" in result:
        summary["comfort_band_upper"] = float(result["comfort_band_upper"])
    if "predicted_occupied_fraction" in result:
        summary["predicted_occupied_fraction"] = float(result["predicted_occupied_fraction"])
    if "T_dyn_hist" in result and len(result["T_dyn_hist"]):
        summary["tdyn_mean"] = float(result["T_dyn_hist"].mean())
        summary["tdyn_final"] = float(result["T_dyn_hist"][-1])
    return summary


def save_result(
    result,
    *,
    output_dir: Path,
    summary_dir: Path,
    scenario: str,
    label: str,
    metadata: dict | None = None,
):
    output_dir = output_dir / scenario / label
    summary_dir = summary_dir / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    history = rollout_history_dataframe(result["env"], result)
    history.to_csv(output_dir / "rollout.csv", index=False)
    result["kpis"].to_csv(output_dir / "kpis.csv", index=False)
    result["district_kpis"].to_csv(output_dir / "district_kpis.csv", index=False)
    metrics = summarize_run(label, result)
    if metadata:
        metrics.update(metadata)
    (summary_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def load_baseline_action_profile(output_root: Path, building: dict, label: str) -> list[float]:
    import pandas as pd

    path = output_root / building["result_id"] / "no_occupant" / label / "rollout.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baseline action profile for {building['display']}: {path}. "
            f"Run scripts/03_run_mpc.py --scenario no_occupant --building {building['result_id']} "
            f"--objective baseline --label {label} first."
        )
    data = pd.read_csv(path)
    if "action" not in data.columns:
        raise ValueError(f"Baseline rollout has no action column: {path}")
    return pd.to_numeric(data["action"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).tolist()


def resolve_rc_model_path(template: str | Path, building: dict) -> Path:
    template_path = format_building_path(template, building)
    default_template = Path("results/models/rc/1state_rc_247942_january.json")
    per_building_default = Path("results") / "models" / "rc" / f"1state_rc_{building['short_name']}_january.json"
    if Path(str(template)) == default_template:
        candidates = [per_building_default, template_path]
    else:
        candidates = [template_path, per_building_default]
    for candidate in dict.fromkeys(candidates):
        if Path(candidate).exists():
            return Path(candidate)
    return template_path


def resolve_occupancy_building_name(selector: str, building: dict) -> str:
    if str(selector).strip().lower() == "same":
        return str(building["name"])
    entries = resolve_building_entries(selector, DATASET_NAME)
    if len(entries) != 1:
        raise ValueError("--occupancy-building must resolve to one building, not 'all'.")
    return str(entries[0]["name"])


def main() -> None:
    args = build_parser().parse_args()
    import pandas as pd

    if args.scenario == "occupant_present_peak_flattening":
        raise SystemExit("Peak flattening is not implemented yet in the notebook MPC.")
    sim_start = int(args.simulation_start)
    sim_end = int(args.simulation_end)
    if sim_end < sim_start:
        raise ValueError("--simulation-end must be >= --simulation-start")

    output_root = Path(args.output_dir)
    summary_root = Path(args.summary_dir)
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    all_summary_rows = []

    for building in buildings:
        building_name = str(building["name"])
        occupancy_building_name = resolve_occupancy_building_name(args.occupancy_building, building)
        metadata = building_metadata(building, occupancy_building_name=occupancy_building_name)
        metadata["simulation_start_time_step"] = sim_start
        metadata["simulation_end_time_step"] = sim_end
        rc_model_path = resolve_rc_model_path(args.rc_model, building)
        if not rc_model_path.exists():
            raise FileNotFoundError(f"Missing RC model for {building['display']}: {rc_model_path}")
        rc_params = json.loads(rc_model_path.read_text())
        metadata["rc_model"] = str(rc_model_path)
        rc2_params = None
        if getattr(args, "rc_model_type", "1r1c") == "2r2c":
            rc2_path = Path(format_building_path(args.rc_model_2r2c, building))
            if not rc2_path.exists():
                raise FileNotFoundError(f"Missing 2R2C RC model for {building['display']}: {rc2_path}")
            rc2_params = json.loads(rc2_path.read_text())
            metadata["rc_model_2r2c"] = str(rc2_path)
            metadata["rc_model_type"] = "2r2c"
        grid_stress_profile = None
        if args.grid_stress_rollout is not None:
            grid_stress_path = format_building_path(args.grid_stress_rollout, building)
            grid_stress_profile = load_grid_stress_profile(
                grid_stress_path,
                column=args.grid_stress_column,
                mode=args.grid_stress_mode,
                percentile=args.grid_stress_percentile,
                daily_top_k=args.grid_stress_daily_top_k,
            )
            metadata["grid_stress_rollout"] = str(grid_stress_path)
            metadata["grid_stress_mode"] = args.grid_stress_mode
            metadata["grid_stress_profile_mean"] = float(sum(grid_stress_profile) / max(len(grid_stress_profile), 1))
        elif float(args.shift_weight) != 0.0 and args.load_shape_profile == "daily_windows":
            grid_stress_profile = daily_window_shift_profile(
                simulation_start=sim_start,
                simulation_end=sim_end,
            )
            metadata["grid_stress_mode"] = "daily_windows_signed"
            metadata["grid_stress_profile_mean"] = float(sum(grid_stress_profile) / max(len(grid_stress_profile), 1))
            metadata["grid_stress_profile_min"] = float(min(grid_stress_profile)) if grid_stress_profile else 0.0
            metadata["grid_stress_profile_max"] = float(max(grid_stress_profile)) if grid_stress_profile else 0.0
        district_profiles = None
        district_weight_profile = district_target_weight_profile(
            simulation_start=sim_start,
            simulation_end=sim_end,
            base_weight=args.district_target_weight,
            profile=args.district_target_weight_profile,
            default_weight=args.district_target_weight_default,
            morning_weight=args.district_target_weight_morning,
            evening_weight=args.district_target_weight_evening,
        )
        future_grid_stress_profile = None
        needs_district_profiles = (
            any(abs(float(weight)) > 0.0 for weight in district_weight_profile)
            or (
                (float(args.shift_weight) != 0.0 or float(args.preheat_weight) != 0.0)
                and grid_stress_profile is None
            )
        )
        if needs_district_profiles:
            district_profiles = load_ce1_district_target_profiles(
                building_name=building_name,
                simulation_start=sim_start,
                simulation_end=sim_end,
                dataset_name=DATASET_NAME,
                target_scale=args.district_target_scale,
                target_scale_mode=args.district_target_scale_mode,
                target_source=args.district_target_source,
            )
            metadata["district_target_weight"] = float(args.district_target_weight)
            metadata["district_target_weight_profile"] = str(args.district_target_weight_profile)
            metadata["district_target_weight_default"] = (
                float(args.district_target_weight)
                if args.district_target_weight_default is None
                else float(args.district_target_weight_default)
            )
            metadata["district_target_weight_morning"] = (
                None if args.district_target_weight_morning is None else float(args.district_target_weight_morning)
            )
            metadata["district_target_weight_evening"] = (
                None if args.district_target_weight_evening is None else float(args.district_target_weight_evening)
            )
            metadata["district_target_weight_profile_mean"] = float(
                sum(district_weight_profile) / max(len(district_weight_profile), 1)
            )
            metadata["district_target_weight_profile_max"] = float(
                max(district_weight_profile) if district_weight_profile else 0.0
            )
            metadata["district_target_mode"] = args.district_target_mode
            metadata["district_target_scale"] = float(district_profiles.target_scale)
            metadata["district_target_scale_mode"] = str(district_profiles.target_scale_mode)
            metadata["district_target_mean"] = float(sum(district_profiles.target) / max(len(district_profiles.target), 1))
            metadata["district_background_mean"] = float(
                sum(district_profiles.background_without_controlled_heating)
                / max(len(district_profiles.background_without_controlled_heating), 1)
            )
            metadata["controlled_heating_nominal_power"] = float(district_profiles.heating_nominal_power)
            if grid_stress_profile is None and (
                float(args.shift_weight) != 0.0 or float(args.preheat_weight) != 0.0
            ):
                if args.load_shape_profile in {
                    "baseline_daily_mean",
                    "baseline_daily_mean_centered",
                    "baseline_daily_mean_exp",
                }:
                    if args.load_shape_profile == "baseline_daily_mean_centered":
                        normalization = "mean_centered"
                    elif args.load_shape_profile == "baseline_daily_mean_exp":
                        normalization = "exp_mean"
                    else:
                        normalization = "mean"
                    grid_stress_profile = baseline_daily_mean_price_profile(
                        district_profiles.baseline_district_load,
                        simulation_start=sim_start,
                        normalize=normalization,
                        exp_beta=args.load_shape_exp_beta,
                    )
                    metadata["grid_stress_mode"] = str(args.load_shape_profile)
                    metadata["grid_stress_normalization"] = normalization
                    if normalization == "exp_mean":
                        metadata["grid_stress_exp_beta"] = float(args.load_shape_exp_beta)
                else:
                    grid_stress_profile = district_profiles.district_stress_profile()
                    metadata["grid_stress_mode"] = "ce1_baseline_above_target"
                metadata["grid_stress_profile_mean"] = float(
                    sum(grid_stress_profile) / max(len(grid_stress_profile), 1)
                )
                metadata["grid_stress_profile_min"] = float(min(grid_stress_profile)) if grid_stress_profile else 0.0
                metadata["grid_stress_profile_max"] = float(max(grid_stress_profile)) if grid_stress_profile else 0.0
            if float(args.preheat_weight) != 0.0:
                future_grid_stress_profile = district_profiles.future_stress_profile(args.preheat_horizon)
                metadata["preheat_weight"] = float(args.preheat_weight)
                metadata["preheat_horizon"] = int(args.preheat_horizon)
                metadata["future_grid_stress_profile_mean"] = float(
                    sum(future_grid_stress_profile) / max(len(future_grid_stress_profile), 1)
                )
        if (
            float(args.preheat_weight) != 0.0
            and future_grid_stress_profile is None
            and grid_stress_profile is not None
        ):
            future_grid_stress_profile = future_stress_from_profile(
                grid_stress_profile,
                horizon=args.preheat_horizon,
            )
            metadata["preheat_weight"] = float(args.preheat_weight)
            metadata["preheat_horizon"] = int(args.preheat_horizon)
            metadata["future_grid_stress_profile_mean"] = float(
                sum(future_grid_stress_profile) / max(len(future_grid_stress_profile), 1)
            )
        setpoint_bias_profile = None
        if args.objective == "balanced" and args.comfort_mode == "time_bias_guard":
            setpoint_bias_profile = setpoint_bias_daily_profile(
                simulation_start=sim_start,
                simulation_end=sim_end,
                amplitude=args.time_bias_amplitude,
            )
            metadata["setpoint_bias_profile"] = "daily_preheat_reduce"
            metadata["setpoint_bias_amplitude"] = float(args.time_bias_amplitude)
            metadata["setpoint_bias_preheat_hour_labels"] = [1, 2, 3, 4, 5, 14, 15, 16, 17]
            metadata["setpoint_bias_reduce_hour_labels"] = [7, 8, 9, 10, 18, 19, 20, 21, 22]
            metadata["time_bias_guard_band"] = float(args.time_bias_guard_band)
            metadata["time_bias_guard_penalty"] = float(args.time_bias_guard_penalty)

        output_dir = output_root / building["result_id"]
        summary_dir = summary_root / building["result_id"]
        summary_rows = []
        building_seed = int(args.seed + 1000 * (int(building["index"]) - 1))

        print(f"\n=== {building['display']}: {building_name} ===")
        print(f"RC model: {rc_model_path}")
        if args.scenario == "occupant_present":
            print(f"Occupancy model source: {occupancy_building_name}")

        if args.scenario == "no_occupant":
            if args.objective == "balanced":
                if args.comfort_mode in {"asymmetric", "time_bias_guard"}:
                    comfort_scale = 1.0 if args.comfort_scale is None else float(args.comfort_scale)
                    temperature_scale = float(args.comfort_track_scale)
                elif args.comfort_scale is None and args.comfort_mode == "setpoint":
                    temperature_scale = temperature_error_scale(
                        building_name=building_name,
                        simulation_start=0,
                        simulation_end=sim_end if "minmax" in args.temperature_scale_mode else sim_start - 1,
                        dataset_name=DATASET_NAME,
                        mode=args.temperature_scale_mode,
                    )
                    comfort_scale = temperature_scale**2
                else:
                    comfort_scale = (
                        float(args.comfort_band) ** 2
                        if args.comfort_scale is None
                        else float(args.comfort_scale)
                    )
                    temperature_scale = float(comfort_scale**0.5)
                metadata["comfort_mode"] = args.comfort_mode
                metadata["temperature_scale_mode"] = args.temperature_scale_mode
                metadata["temperature_error_scale"] = float(temperature_scale)
                metadata["comfort_scale"] = float(comfort_scale)
                metadata["comfort_track_weight"] = float(args.comfort_track_weight)
                metadata["comfort_cold_weight"] = float(args.comfort_cold_weight)
                metadata["comfort_hot_weight"] = float(args.comfort_hot_weight)
                metadata["comfort_cold_tolerance"] = float(args.comfort_cold_tolerance)
                metadata["comfort_hot_tolerance"] = float(args.comfort_hot_tolerance)
                if args.comfort_hot_tolerance_night is not None:
                    metadata["comfort_hot_tolerance_night"] = float(args.comfort_hot_tolerance_night)
                    metadata["comfort_hot_tolerance_night_end"] = int(args.comfort_hot_tolerance_night_end)
                metadata["comfort_track_scale"] = float(args.comfort_track_scale)
                metadata["comfort_cold_scale"] = float(args.comfort_cold_scale)
                metadata["comfort_hot_scale"] = float(args.comfort_hot_scale)
                metadata["comfort_cold_k"] = float(args.comfort_cold_k)
                metadata["comfort_hot_k"] = float(args.comfort_hot_k)
                metadata["single_setpoint"] = bool(args.single_setpoint)
                if args.comfort_mode == "time_bias_guard":
                    metadata["time_bias_amplitude"] = float(args.time_bias_amplitude)
                    metadata["time_bias_guard_band"] = float(args.time_bias_guard_band)
                    metadata["time_bias_guard_penalty"] = float(args.time_bias_guard_penalty)
                metadata["objective"] = "balanced"
                metadata["comfort_weight"] = float(args.comfort_weight)
                metadata["energy_weight"] = float(args.energy_weight)
                metadata["peak_weight"] = float(args.peak_weight)
                metadata["ramp_weight"] = float(args.ramp_weight)
                metadata["shift_weight"] = float(args.shift_weight)
                metadata["preheat_weight"] = float(args.preheat_weight)
                metadata["power_scale"] = float(args.power_scale)
                metadata["power_source"] = str(args.power_source)
                metadata["raw_power_cost"] = bool(args.raw_power_cost)
                reward_function_kwargs = dict(
                    comfort_weight=args.comfort_weight,
                    energy_weight=args.energy_weight,
                    peak_weight=args.peak_weight,
                    ramp_weight=args.ramp_weight,
                    shift_weight=args.shift_weight,
                    preheat_weight=args.preheat_weight,
                    comfort_band=args.comfort_band,
                    comfort_scale=comfort_scale,
                    comfort_mode=args.comfort_mode,
                    comfort_track_weight=args.comfort_track_weight,
                    comfort_cold_weight=args.comfort_cold_weight,
                    comfort_hot_weight=args.comfort_hot_weight,
                    comfort_cold_tolerance=args.comfort_cold_tolerance,
                    comfort_hot_tolerance=args.comfort_hot_tolerance,
                    comfort_track_scale=args.comfort_track_scale,
                    comfort_cold_scale=args.comfort_cold_scale,
                    comfort_hot_scale=args.comfort_hot_scale,
                    comfort_cold_k=args.comfort_cold_k,
                    comfort_hot_k=args.comfort_hot_k,
                    single_setpoint=args.single_setpoint,
                    power_scale=args.power_scale,
                    power_source=args.power_source,
                    setpoint_bias_profile=setpoint_bias_profile,
                    time_bias_guard_band=args.time_bias_guard_band,
                    shift_profile=grid_stress_profile,
                    future_shift_profile=future_grid_stress_profile,
                )
                if district_profiles is not None:
                    reward_function_kwargs.update(
                        district_profiles.as_reward_kwargs(
                            target_weight=args.district_target_weight,
                            target_mode=args.district_target_mode,
                        )
                    )
                    reward_function_kwargs["district_target_weight_profile"] = district_weight_profile
                objective_kwargs = dict(
                    controller_name=f"Balanced MPC | {building['display']}",
                    objective_mode="balanced",
                    w_energy=args.energy_weight,
                    w_comfort=args.comfort_weight,
                    lambda_u=args.peak_weight,
                    lambda_du=args.ramp_weight,
                    shift_weight=args.shift_weight,
                    preheat_weight=args.preheat_weight,
                    use_smooth_du=True,
                    comfort_band=args.comfort_band,
                    comfort_scale=comfort_scale,
                    comfort_mode=args.comfort_mode,
                    comfort_track_weight=args.comfort_track_weight,
                    comfort_cold_weight=args.comfort_cold_weight,
                    comfort_hot_weight=args.comfort_hot_weight,
                    comfort_cold_tolerance=args.comfort_cold_tolerance,
                    comfort_hot_tolerance=args.comfort_hot_tolerance,
                    comfort_hot_tolerance_night=args.comfort_hot_tolerance_night,
                    comfort_hot_tolerance_night_end=args.comfort_hot_tolerance_night_end,
                    comfort_track_scale=args.comfort_track_scale,
                    comfort_cold_scale=args.comfort_cold_scale,
                    comfort_hot_scale=args.comfort_hot_scale,
                    comfort_cold_k=args.comfort_cold_k,
                    comfort_hot_k=args.comfort_hot_k,
                    power_scale=args.power_scale,
                    setpoint_bias_profile=setpoint_bias_profile,
                    time_bias_guard_band=args.time_bias_guard_band,
                    normalize_cost_terms=not args.raw_power_cost,
                    grid_stress_profile=grid_stress_profile,
                    future_grid_stress_profile=future_grid_stress_profile,
                    district_target_weight=args.district_target_weight,
                    district_target_profile=None if district_profiles is None else district_profiles.target,
                    district_background_profile=(
                        None if district_profiles is None else district_profiles.background_without_controlled_heating
                    ),
                    district_action_power_profile=None if district_profiles is None else district_profiles.heating_action_power,
                    district_target_weight_profile=district_weight_profile,
                    district_target_scale=1.0 if district_profiles is None else district_profiles.target_scale,
                    district_target_mode=args.district_target_mode,
                    rc2_params=rc2_params,
                    reward_function=BalancedComfortEnergyReward,
                    reward_function_kwargs=reward_function_kwargs,
                )
                label = args.label
            elif args.objective == "budget_shift":
                comfort_scale = (
                    float(args.comfort_band) ** 2
                    if args.comfort_scale is None
                    else float(args.comfort_scale)
                )
                baseline_action_profile = load_baseline_action_profile(
                    output_root,
                    building,
                    args.baseline_action_label,
                )
                metadata["objective"] = "budget_shift"
                metadata["comfort_mode"] = "fixed_barrier_band"
                metadata["comfort_band_lower"] = 20.0
                metadata["comfort_band_upper"] = 24.0
                metadata["comfort_scale"] = float(comfort_scale)
                metadata["comfort_weight"] = float(args.comfort_weight)
                metadata["shift_weight"] = float(args.shift_weight)
                metadata["budget_weight"] = float(args.budget_weight)
                metadata["ramp_weight"] = float(args.ramp_weight)
                metadata["ramp_scale"] = float(args.ramp_scale)
                metadata["daily_budget_scale"] = float(args.daily_budget_scale)
                metadata["budget_settlement_hour_labels"] = [24]
                metadata["baseline_action_label"] = str(args.baseline_action_label)
                metadata["active_weight_sum"] = float(
                    args.comfort_weight + args.shift_weight + args.budget_weight + args.ramp_weight
                )
                objective_kwargs = dict(
                    controller_name=f"Budget-payback MPC | {building['display']}",
                    objective_mode="budget_shift",
                    w_comfort=args.comfort_weight,
                    lambda_du=args.ramp_weight,
                    shift_weight=args.shift_weight,
                    budget_weight=args.budget_weight,
                    use_smooth_du=True,
                    comfort_band=args.comfort_band,
                    comfort_scale=comfort_scale,
                    ramp_scale=args.ramp_scale,
                    daily_budget_scale=args.daily_budget_scale,
                    normalize_cost_terms=True,
                    grid_stress_profile=grid_stress_profile,
                    baseline_action_profile=baseline_action_profile,
                )
                label = args.label
            elif args.objective == "simple":
                # Build mean-daily district load as a price proxy (exp-normalised, beta=2).
                # Peak hours (morning ramp ~hour 7) cost ~1.5–2.0; off-peak (midday) ~0.25.
                # Using exp_mean amplifies peak/off-peak contrast (~5–7×) vs max-norm (~1.85×),
                # giving the MPC a strong incentive to pre-heat off-peak and coast at peaks.
                _bl_path = (
                    Path(__file__).resolve().parents[1]
                    / "results" / "raw" / "citylearn_baseline" / "no_occupant" / "baseline"
                    / "district_rollout.csv"
                )
                _simple_price_profile = None
                if _bl_path.exists():
                    import pandas as _pd
                    _bl = _pd.read_csv(_bl_path)
                    if "net_electricity_consumption" in _bl.columns:
                        _simple_price_profile = baseline_daily_mean_price_profile(
                            _bl["net_electricity_consumption"].tolist(),
                            simulation_start=sim_start,
                            normalize="exp_mean",
                            exp_beta=2.0,
                        )
                        metadata["simple_price_profile"] = "baseline_district_daily_mean_exp2_norm"
                        metadata["simple_price_profile_min"] = float(min(_simple_price_profile))
                        metadata["simple_price_profile_max"] = float(max(_simple_price_profile))
                else:
                    metadata["simple_price_profile"] = "flat_1.0_fallback"
                metadata["objective"] = "simple"
                metadata["comfort_band"] = float(args.comfort_band)
                metadata["comfort_scale"] = float(args.comfort_band) ** 2
                metadata["comfort_weight"] = float(args.comfort_weight)
                metadata["energy_weight"] = float(args.energy_weight)
                metadata["simple_tsp_lower_bias"] = float(args.tsp_lower_bias)
                objective_kwargs = dict(
                    controller_name=f"Simple comfort+energy MPC | {building['display']}",
                    objective_mode="simple",
                    w_comfort=args.comfort_weight,
                    w_energy=args.energy_weight,
                    comfort_band=args.comfort_band,
                    simple_tsp_lower_bias=args.tsp_lower_bias,
                    grid_stress_profile=_simple_price_profile,
                )
                label = args.label
            else:
                objective_kwargs = dict(
                    controller_name=f"Current MPC (baseline) | {building['display']}",
                    objective_mode="baseline",
                    lambda_u=1e-3,
                    lambda_du=0.0,
                    use_smooth_du=False,
                )
                label = args.label
            if rc2_params is not None:
                objective_kwargs.update(
                    two_state_observer=args.two_state_observer,
                    kalman_process_var=args.kalman_process_var,
                    kalman_measurement_var=args.kalman_measurement_var,
                    kalman_initial_var=args.kalman_initial_var,
                )
                metadata["two_state_observer"] = str(args.two_state_observer)
                if args.two_state_observer == "kalman":
                    metadata["kalman_process_var"] = float(args.kalman_process_var)
                    metadata["kalman_measurement_var"] = float(args.kalman_measurement_var)
                    metadata["kalman_initial_var"] = float(args.kalman_initial_var)

            run_cold_penalty = 1e5
            run_hot_penalty = 1e3
            if args.objective == "balanced" and args.comfort_mode == "time_bias_guard":
                run_cold_penalty = float(args.time_bias_guard_penalty)
                run_hot_penalty = float(args.time_bias_guard_penalty)

            result = run_february_mpc(
                rc_params=rc_params,
                fixed_band=None,
                cold_penalty=run_cold_penalty,
                hot_penalty=run_hot_penalty,
                seed=building_seed,
                max_steps=args.max_steps,
                sim_start=sim_start,
                sim_end=sim_end,
                building_name=building_name,
                **objective_kwargs,
            )
            summary_rows.append(
                save_result(
                    result,
                    output_dir=output_dir,
                    summary_dir=summary_dir,
                    scenario=args.scenario,
                    label=label,
                    metadata=metadata,
                )
            )

        elif args.scenario == "occupant_present":
            occupants = fitted_occupants()
            if args.occupant != "all":
                occupants = {args.occupant: occupants[args.occupant]}
            if args.objective != "balanced":
                raise ValueError("Final occupant experiment expects --objective balanced.")
            comfort_scale = 1.0 if args.comfort_mode in {"asymmetric", "time_bias_guard"} and args.comfort_scale is None else (
                float(args.comfort_band) ** 2
                if args.comfort_scale is None
                else float(args.comfort_scale)
            )
            metadata["objective"] = "balanced"
            metadata["comfort_mode"] = args.comfort_mode
            metadata["comfort_scale"] = float(comfort_scale)
            metadata["comfort_weight"] = float(args.comfort_weight)
            metadata["energy_weight"] = float(args.energy_weight)
            metadata["peak_weight"] = float(args.peak_weight)
            metadata["ramp_weight"] = float(args.ramp_weight)
            metadata["shift_weight"] = float(args.shift_weight)
            metadata["preheat_weight"] = float(args.preheat_weight)
            metadata["comfort_track_weight"] = float(args.comfort_track_weight)
            metadata["comfort_cold_weight"] = float(args.comfort_cold_weight)
            metadata["comfort_hot_weight"] = float(args.comfort_hot_weight)
            reward_function_kwargs = dict(
                comfort_weight=args.comfort_weight,
                energy_weight=args.energy_weight,
                peak_weight=args.peak_weight,
                ramp_weight=args.ramp_weight,
                shift_weight=args.shift_weight,
                preheat_weight=args.preheat_weight,
                comfort_band=args.comfort_band,
                comfort_scale=comfort_scale,
                comfort_mode=args.comfort_mode,
                comfort_track_weight=args.comfort_track_weight,
                comfort_cold_weight=args.comfort_cold_weight,
                comfort_hot_weight=args.comfort_hot_weight,
                comfort_cold_tolerance=args.comfort_cold_tolerance,
                comfort_hot_tolerance=args.comfort_hot_tolerance,
                comfort_track_scale=args.comfort_track_scale,
                comfort_cold_scale=args.comfort_cold_scale,
                comfort_hot_scale=args.comfort_hot_scale,
                comfort_cold_k=args.comfort_cold_k,
                comfort_hot_k=args.comfort_hot_k,
                single_setpoint=args.single_setpoint,
                power_scale=args.power_scale,
                power_source=args.power_source,
                setpoint_bias_profile=setpoint_bias_profile,
                time_bias_guard_band=args.time_bias_guard_band,
                shift_profile=grid_stress_profile,
                future_shift_profile=future_grid_stress_profile,
            )
            objective_kwargs = dict(
                objective_mode="balanced",
                w_energy=args.energy_weight,
                w_comfort=args.comfort_weight,
                lambda_u=args.peak_weight,
                lambda_du=args.ramp_weight,
                shift_weight=args.shift_weight,
                preheat_weight=args.preheat_weight,
                use_smooth_du=bool(float(args.ramp_weight) != 0.0),
                comfort_band=args.comfort_band,
                comfort_scale=comfort_scale,
                comfort_mode=args.comfort_mode,
                comfort_track_weight=args.comfort_track_weight,
                comfort_cold_weight=args.comfort_cold_weight,
                comfort_hot_weight=args.comfort_hot_weight,
                comfort_cold_tolerance=args.comfort_cold_tolerance,
                comfort_hot_tolerance=args.comfort_hot_tolerance,
                comfort_hot_tolerance_night=args.comfort_hot_tolerance_night,
                comfort_hot_tolerance_night_end=args.comfort_hot_tolerance_night_end,
                comfort_track_scale=args.comfort_track_scale,
                comfort_cold_scale=args.comfort_cold_scale,
                comfort_hot_scale=args.comfort_hot_scale,
                comfort_cold_k=args.comfort_cold_k,
                comfort_hot_k=args.comfort_hot_k,
                power_scale=args.power_scale,
                setpoint_bias_profile=setpoint_bias_profile,
                time_bias_guard_band=args.time_bias_guard_band,
                normalize_cost_terms=not args.raw_power_cost,
                grid_stress_profile=grid_stress_profile,
                future_grid_stress_profile=future_grid_stress_profile,
                rc2_params=rc2_params,
                reward_function=BalancedComfortEnergyReward,
                reward_function_kwargs=reward_function_kwargs,
                sim_start=sim_start,
                sim_end=sim_end,
            )

            for occ_idx, (name, occ) in enumerate(occupants.items(), start=1):
                case_seed = int(building_seed + occ_idx)
                if args.occupant_mode in ("without_tdyn", "both"):
                    label = f"{args.label}_{name}_without_tdyn" if args.occupant == "all" or args.occupant_mode == "both" else args.label
                    print(f"\n{label}")
                    result = run_february_mpc(
                        controller_name=f"Current MPC + {name} without T_dyn | {building['display']}",
                        rc_params=rc_params,
                        fixed_band=None,
                        cold_penalty=args.time_bias_guard_penalty,
                        hot_penalty=args.time_bias_guard_penalty,
                        occupant=deepcopy(occ),
                        seed=case_seed,
                        max_steps=args.max_steps,
                        building_name=building_name,
                        **objective_kwargs,
                    )
                    summary_rows.append(
                        save_result(
                            result,
                            output_dir=output_dir,
                            summary_dir=summary_dir,
                            scenario=args.scenario,
                            label=label,
                            metadata=metadata,
                        )
                    )

                if args.occupant_mode in ("bspline_tdyn", "both"):
                    label = f"{args.label}_{name}_bspline_tdyn" if args.occupant == "all" or args.occupant_mode == "both" else args.label
                    print(f"\n{label}")
                    result = run_february_bspline_tdyn_mpc(
                        controller_name=f"B-spline gated adaptive T_dyn MPC + {name} | {building['display']}",
                        rc_params=rc_params,
                        occupant=deepcopy(occ),
                        fixed_band=None,
                        alpha_occ=0.5,
                        lambda_u=args.peak_weight,
                        lambda_du=args.ramp_weight,
                        use_smooth_du=bool(float(args.ramp_weight) != 0.0),
                        cold_penalty=args.time_bias_guard_penalty,
                        hot_penalty=args.time_bias_guard_penalty,
                        reward_function=BalancedComfortEnergyReward,
                        reward_function_kwargs=reward_function_kwargs,
                        seed=case_seed,
                        max_steps=args.max_steps,
                        building_name=building_name,
                        occupancy_building_name=occupancy_building_name,
                        sim_start=sim_start,
                        sim_end=sim_end,
                        comfort_track_weight=args.comfort_track_weight,
                        comfort_cold_weight=args.comfort_cold_weight,
                        comfort_hot_weight=args.comfort_hot_weight,
                        comfort_track_scale=args.comfort_track_scale,
                        comfort_cold_scale=args.comfort_cold_scale,
                        comfort_hot_scale=args.comfort_hot_scale,
                        time_bias_guard_band=args.time_bias_guard_band,
                        setpoint_bias_profile=setpoint_bias_profile,
                        T_dyn_init_mode="schedule",
                        delta_up=0.5,
                        delta_down=0.5,
                        drift_to_pref=0.05,
                        T_dyn_min=18.0,
                        T_dyn_max=26.0,
                    )
                    summary_rows.append(
                        save_result(
                            result,
                            output_dir=output_dir,
                            summary_dir=summary_dir,
                            scenario=args.scenario,
                            label=label,
                            metadata=metadata,
                        )
                    )

        summary = pd.DataFrame(summary_rows)
        summary_path = summary_dir / args.scenario / "summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        all_summary_rows.extend(summary_rows)
        print("\nSummary")
        print(summary.to_string(index=False))
        print(f"Saved summary to {summary_path}")

    if len(buildings) > 1:
        combined = pd.DataFrame(all_summary_rows)
        combined_path = summary_root / f"{args.scenario}_summary_all_buildings.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined summary to {combined_path}")


if __name__ == "__main__":
    main()
