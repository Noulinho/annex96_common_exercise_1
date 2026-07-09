#!/usr/bin/env python
"""Run thesis-safe RBC/Safe-PI baselines with the final p2 reference logic.

This does not touch CityLearn's BaselineAgent.  It creates a separate simple
controller comparator:

- no occupant: PI tracks Tschedule + bias;
- without_tdyn: PI tracks the active/effective setpoint + bias;
- bspline_tdyn: PI tracks Tdyn + bias when predicted occupied, and
  Tschedule + bias when predicted unoccupied.

The unbiased reference is still passed to the reward function, which applies
the same time-bias profile internally as MPC/GNU-RL/SAC.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from citylearn.agents.rbc import PITemperatureController
from src.buildings import DATASET_NAME, dataset_building_entries
from src.occupants.metrics import (
    kelvin_hours,
    occupant_probability_comfort_band,
    override_hours,
    setpoint_kelvin_hours,
)
from src.occupants.wrappers import BSplineOccupancyObservationWrapper, OccupantEnvWrapper
from src.rewards import BalancedComfortEnergyReward, time_bias_setpoint_profile
from src.rlmpc_pipeline import (
    FITTED_OCCUPANT_PARAMS,
    Occupant,
    fit_bspline_occupancy_model,
    forecast_occupancy_from_bsplines,
)
from src.sac_merlin_pipeline import (
    _apply_safe_pi_exploration_action,
    _apply_tdyn_to_observations,
    _building_arrays,
    _set_reward_reference_setpoint,
    _tdyn_gate,
    _tdyn_initial_value,
    _update_tdyn_from_feedback,
    make_env,
)


PREFIX = "ce1_thesis_rbc_safepi"
MONTHS = {
    "jan": (0, 743),
    "feb": (744, 1415),
    "mar": (1416, 2159),
}
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]
MODES = ["no_occupant", "without_tdyn", "bspline_tdyn"]


def reward_kwargs(start: int, end: int) -> dict:
    bias = time_bias_setpoint_profile(
        simulation_start=start,
        simulation_end=end,
        amplitude=1.0,
    )
    return {
        "comfort_weight": 1.0,
        "energy_weight": 0.0,
        "peak_weight": 0.0,
        "ramp_weight": 0.0,
        "shift_weight": 0.0,
        "preheat_weight": 0.0,
        "comfort_scale": 1.0,
        "comfort_mode": "time_bias_guard",
        "comfort_track_weight": 0.1,
        "comfort_cold_weight": 0.7,
        "comfort_hot_weight": 0.2,
        "comfort_track_scale": 2.0,
        "comfort_cold_scale": 2.0,
        "comfort_hot_scale": 2.0,
        "comfort_guard_exponent": 2.0,
        "single_setpoint": True,
        "setpoint_bias_profile": bias,
        "time_bias_guard_band": 2.0,
        "power_source": "heating",
    }


def observation_setpoint(observations, env, value: float):
    return _apply_tdyn_to_observations(env, observations, float(value))


def output_case_label(month: str, mode: str, occupant_name: str | None) -> str:
    if mode == "no_occupant":
        return f"{PREFIX}_{month}_no_occupant"
    return f"{PREFIX}_{month}_{occupant_name}_{mode}"


def scenario_for_mode(mode: str) -> str:
    return "no_occupant" if mode == "no_occupant" else "occupant_present"


def comfort_band_for(occupant) -> tuple[float, float]:
    if occupant is None:
        return 20.0, 24.0
    return occupant_probability_comfort_band(occupant)


def summarize(label: str, result: dict, env, occupant) -> dict:
    low, high = comfort_band_for(occupant)
    kpis = env.unwrapped.evaluate().copy()
    district = kpis[kpis["level"].eq("district")].copy()

    def kpi(name: str) -> float:
        rows = district[district["cost_function"].eq(name)]
        if rows.empty:
            return float("nan")
        return float(rows["value"].iloc[0])

    return {
        "case": label,
        "override_count": int(getattr(env, "override_count", 0)),
        "override_hours": override_hours(
            result.get("baseline_setpoint", []),
            result.get("effective_setpoint", []),
        ),
        "reward_total": float(np.sum(result["reward"])),
        "kelvin_hours": kelvin_hours(result["indoor_temperature"], lower=low, upper=high),
        "setpoint_kelvin_hours": setpoint_kelvin_hours(
            result["indoor_temperature"],
            result.get("tracking_setpoint", result["effective_setpoint"]),
        ),
        "electricity_consumption_total": kpi("electricity_consumption_total"),
        "discomfort_proportion": kpi("discomfort_proportion"),
        "discomfort_cold_proportion": kpi("discomfort_cold_proportion"),
        "discomfort_hot_proportion": kpi("discomfort_hot_proportion"),
        "daily_peak_average": kpi("daily_peak_average"),
    }


def run_one(
    *,
    building: dict,
    month: str,
    mode: str,
    occupant_name: str | None,
    seed: int,
    output_root: Path,
    summary_root: Path,
    alpha_occ: float,
    safe_pi_cold_margin: float,
    safe_pi_hot_margin: float,
    safe_pi_cold_min_action: float,
    safe_pi_cold_kp: float,
    safe_pi_hot_max_action: float,
    safe_pi_ramp_limit: float | None,
    pi_kp: float,
    pi_ki: float,
) -> Path:
    start, end = MONTHS[month]
    kwargs = reward_kwargs(start, end)
    env = make_env(
        sim_start=start,
        sim_end=end,
        seed=seed,
        reward_function=BalancedComfortEnergyReward,
        reward_function_kwargs=kwargs,
        building_name=building["name"],
    )

    occupant = None
    occupancy_model = None
    occupancy_series = None
    if mode != "no_occupant":
        if occupant_name is None:
            raise ValueError("occupant_name is required for occupant modes.")
        occupant = Occupant(**FITTED_OCCUPANT_PARAMS[occupant_name])
        env = OccupantEnvWrapper(env, deepcopy(occupant))
        if mode == "bspline_tdyn":
            occupancy_model, occupancy_series = fit_bspline_occupancy_model(
                seed=seed,
                building_name=building["name"],
            )
            env = BSplineOccupancyObservationWrapper(
                env,
                occupancy_model=occupancy_model,
                occupancy_series=occupancy_series,
                forecast_fn=forecast_occupancy_from_bsplines,
                alpha=alpha_occ,
            )

    agent = PITemperatureController(
        env,
        kp=pi_kp,
        ki=pi_ki,
        temp_deadband=0.5,
        random_seed=seed,
    )
    observations, _ = env.reset(seed=seed)
    agent.reset()
    rng = np.random.default_rng(seed + 817)
    bias_profile = kwargs["setpoint_bias_profile"]
    T_dyn = None
    if mode == "bspline_tdyn":
        T_dyn = _tdyn_initial_value(
            env,
            occupant,
            init_mode="schedule",
            lower_bound=18.0,
            upper_bound=26.0,
        )

    terminated = truncated = False
    previous_action = None
    step = 0
    rewards: list[float] = []
    action_hist: list[float] = []
    raw_action_hist: list[float] = []
    action_source_hist: list[str] = []
    reference_hist: list[float] = []
    tracking_hist: list[float] = []
    bias_hist: list[float] = []
    tdyn_hist: list[float] = []
    occ_now_hist: list[float] = []
    occ_hat_hist: list[float] = []
    occ_bin_hist: list[float] = []
    feedback_hist: list[float] = []

    while not (terminated or truncated):
        building_obj = env.unwrapped.buildings[0]
        time_step = int(building_obj.time_step)
        scheduled = float(
            building_obj.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                time_step
            ]
        )
        if mode == "bspline_tdyn":
            absolute_step = int(start) + int(step)
            occ_now, occ_hat, occ_bin = _tdyn_gate(
                occupancy_model=occupancy_model,
                occupancy_series=occupancy_series,
                absolute_time_step=absolute_step,
                alpha_occ=alpha_occ,
            )
            true_reference = (1.0 - float(occ_bin)) * scheduled + float(occ_bin) * float(T_dyn)
        elif mode == "without_tdyn":
            true_reference = float(
                building_obj.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[time_step]
            )
            occ_now = occ_hat = occ_bin = np.nan
        else:
            true_reference = scheduled
            occ_now = occ_hat = occ_bin = np.nan

        bias = float(bias_profile[min(step, len(bias_profile) - 1)])
        tracking_reference = true_reference + bias
        observations_for_agent = observation_setpoint(observations, env, tracking_reference)
        _set_reward_reference_setpoint(env, true_reference)

        raw_actions = agent.predict(observations_for_agent, deterministic=True)
        action_info = _apply_safe_pi_exploration_action(
            action=float(np.asarray(raw_actions, dtype=float).reshape(-1)[0]),
            env=env,
            observations=observations_for_agent,
            rng=rng,
            cold_margin=safe_pi_cold_margin,
            hot_margin=safe_pi_hot_margin,
            cold_min_action=safe_pi_cold_min_action,
            cold_kp=safe_pi_cold_kp,
            hot_max_action=safe_pi_hot_max_action,
            random_prob=0.0,
            random_low=0.0,
            random_high=0.0,
            ramp_limit=safe_pi_ramp_limit,
            previous_action=previous_action,
        )
        actions = [[float(action_info["action"])]]
        observations, reward, terminated, truncated, _ = env.step(actions)

        rewards.append(float(np.sum(reward)))
        action_hist.append(float(action_info["action"]))
        raw_action_hist.append(float(action_info["raw_action"]))
        action_source_hist.append(str(action_info["action_source"]))
        reference_hist.append(float(true_reference))
        tracking_hist.append(float(tracking_reference))
        bias_hist.append(float(bias))
        feedback = float(getattr(env, "last_feedback", 0.0))
        feedback_hist.append(feedback)
        if mode == "bspline_tdyn":
            T_dyn = _update_tdyn_from_feedback(
                float(T_dyn),
                feedback,
                occupant,
                delta_up=0.5,
                delta_down=0.5,
                drift_to_pref=0.05,
                lower_bound=18.0,
                upper_bound=26.0,
            )
            tdyn_hist.append(float(T_dyn))
            occ_now_hist.append(float(occ_now))
            occ_hat_hist.append(float(occ_hat))
            occ_bin_hist.append(float(occ_bin))

        previous_action = float(action_info["action"])
        step += 1

    arrays = _building_arrays(env, n_steps=len(rewards))
    history = {
        "time_step": np.arange(len(rewards), dtype=int),
        "indoor_temperature": arrays["indoor_temperature"],
        "baseline_setpoint": arrays["baseline_setpoint"],
        "effective_setpoint": arrays["effective_setpoint"],
        "cooling_setpoint": arrays["cooling_setpoint"],
        "action": np.asarray(action_hist, dtype=float),
        "raw_action": np.asarray(raw_action_hist, dtype=float),
        "reward": np.asarray(rewards, dtype=float),
        "phase": np.full(len(rewards), "rbc_safepi", dtype=object),
        "action_source": np.asarray(action_source_hist, dtype=object),
        "reference_setpoint": np.asarray(reference_hist, dtype=float),
        "tracking_setpoint": np.asarray(tracking_hist, dtype=float),
        "setpoint_bias": np.asarray(bias_hist, dtype=float),
    }
    if "net_electricity_consumption" in arrays:
        history["net_electricity_consumption"] = arrays["net_electricity_consumption"]
    if "heating_electricity_consumption" in arrays:
        history["heating_electricity_consumption"] = arrays["heating_electricity_consumption"]
    if mode != "no_occupant":
        history["occupant_feedback"] = np.asarray(feedback_hist, dtype=float)
    if mode == "bspline_tdyn":
        history["tdyn"] = np.asarray(tdyn_hist, dtype=float)
        history["occ_now"] = np.asarray(occ_now_hist, dtype=float)
        history["occ_hat_1step"] = np.asarray(occ_hat_hist, dtype=float)
        history["occ_bin_1step"] = np.asarray(occ_bin_hist, dtype=float)

    label = output_case_label(month, mode, occupant_name)
    scenario = scenario_for_mode(mode)
    out_dir = output_root / building["result_id"] / scenario / label
    sum_dir = summary_root / building["result_id"] / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(history)
    frame.to_csv(out_dir / "rollout.csv", index=False)
    kpis = env.unwrapped.evaluate().copy()
    kpis.to_csv(out_dir / "kpis.csv", index=False)
    summary = summarize(label, history, env, occupant)
    summary_path = sum_dir / "summary.csv"
    summary_frame = pd.DataFrame([summary])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[old["case"].astype(str) != label]
        summary_frame = pd.concat([old, summary_frame], ignore_index=True, sort=False)
    summary_frame.to_csv(summary_path, index=False)
    manifest = {
        "case": label,
        "controller": "rbc_safe_pi",
        "month": month,
        "mode": mode,
        "occupant": occupant_name,
        "simulation_start": int(start),
        "simulation_end": int(end),
        "reference": "Tref+bias; bspline_tdyn gates Tref to Tschedule or Tdyn",
        "reward": "p2 normalized time_bias_guard D, no energy/gridstress",
        "safe_pi": {
            "pi_kp": pi_kp,
            "pi_ki": pi_ki,
            "cold_margin": safe_pi_cold_margin,
            "hot_margin": safe_pi_hot_margin,
            "cold_min_action": safe_pi_cold_min_action,
            "cold_kp": safe_pi_cold_kp,
            "hot_max_action": safe_pi_hot_max_action,
            "ramp_limit": safe_pi_ramp_limit,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir / "rollout.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", default=["jan", "feb", "mar"], choices=list(MONTHS))
    parser.add_argument("--modes", nargs="+", default=["no_occupant", "bspline_tdyn"], choices=MODES)
    parser.add_argument("--occupants", nargs="+", default=OCCUPANTS, choices=OCCUPANTS)
    parser.add_argument("--building", default="all")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--output-root", default="results/raw/rbc_safe_pi")
    parser.add_argument("--summary-root", default="results/summaries/rbc_safe_pi")
    parser.add_argument("--alpha-occ", type=float, default=0.5)
    parser.add_argument("--safe-pi-cold-margin", type=float, default=1.0)
    parser.add_argument("--safe-pi-hot-margin", type=float, default=1.0)
    parser.add_argument("--safe-pi-cold-min-action", type=float, default=0.25)
    parser.add_argument("--safe-pi-cold-kp", type=float, default=0.25)
    parser.add_argument("--safe-pi-hot-max-action", type=float, default=0.0)
    parser.add_argument("--safe-pi-ramp-limit", type=float, default=None)
    parser.add_argument("--pi-kp", type=float, default=0.2)
    parser.add_argument("--pi-ki", type=float, default=0.005)
    args = parser.parse_args()

    buildings = dataset_building_entries(DATASET_NAME)
    if args.building != "all":
        ids = {int(v) for v in str(args.building).replace(",", " ").split()}
        buildings = [b for b in buildings if int(b["index"]) in ids]
    output_root = Path(args.output_root)
    summary_root = Path(args.summary_root)

    total = 0
    for month in args.months:
        for mode in args.modes:
            occupant_names = [None] if mode == "no_occupant" else list(args.occupants)
            for occupant_name in occupant_names:
                for building in buildings:
                    path = run_one(
                        building=building,
                        month=month,
                        mode=mode,
                        occupant_name=occupant_name,
                        seed=args.seed,
                        output_root=output_root,
                        summary_root=summary_root,
                        alpha_occ=args.alpha_occ,
                        safe_pi_cold_margin=args.safe_pi_cold_margin,
                        safe_pi_hot_margin=args.safe_pi_hot_margin,
                        safe_pi_cold_min_action=args.safe_pi_cold_min_action,
                        safe_pi_cold_kp=args.safe_pi_cold_kp,
                        safe_pi_hot_max_action=args.safe_pi_hot_max_action,
                        safe_pi_ramp_limit=args.safe_pi_ramp_limit,
                        pi_kp=args.pi_kp,
                        pi_ki=args.pi_ki,
                    )
                    total += 1
                    print(f"[{total}] {path}")


if __name__ == "__main__":
    main()
