#!/usr/bin/env python3
"""Compare first-step 2R2C DiffMPC actions against explicit MPC rollouts."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.buildings import DATASET_NAME, resolve_building_entries
from src.rewards import load_simple_v4_price_profile, time_bias_setpoint_profile
from src.rlmpc_pipeline import (
    FEB_END,
    FEB_START,
    TwoStateDiffMPCPolicy,
    load_offline_2r2c_checkpoint,
    make_february_raw_env,
    observation_indices_from_names,
    set_experiment_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", default="all")
    parser.add_argument("--mpc-label", default="ce1_time_bias_guard_1c_2r2c_kalman")
    parser.add_argument("--checkpoint-label", default="ce1_time_bias_guard_1c_2r2c_kalman_h24")
    parser.add_argument("--output-label", default="ce1_time_bias_guard_1c_2r2c_kalman_rlmpc_parity_h24_guard")
    parser.add_argument("--simulation-start", type=int, default=FEB_START)
    parser.add_argument("--simulation-end", type=int, default=FEB_END)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--time-bias-amplitude", type=float, default=1.0)
    parser.add_argument("--time-bias-guard-band", type=float, default=2.0)
    parser.add_argument("--time-bias-guard-penalty", type=float, default=100000.0)
    parser.add_argument("--comfort-weight", type=float, default=0.5)
    parser.add_argument("--shift-weight", type=float, default=0.5)
    parser.add_argument("--power-scale", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def _setpoint_profile(env, sim_start: int, sim_end: int) -> list[float]:
    schedule = (
        env.unwrapped.buildings[0]
        .energy_simulation
        .indoor_dry_bulb_temperature_heating_set_point_without_control
    )
    n_steps = max(1, int(sim_end) - int(sim_start) + 1)
    return [float(schedule[min(int(sim_start) + i, len(schedule) - 1)]) for i in range(n_steps)]


def _matched_cost(rc2: dict, args) -> dict:
    return {
        "comfort_weight": float(args.comfort_weight),
        "energy_weight": 0.0,
        "peak_weight": 0.0,
        "ramp_weight": 0.0,
        "shift_weight": float(args.shift_weight),
        "preheat_weight": 0.0,
        "district_target_weight": 0.0,
        "district_target_scale": 1.0,
        "district_target_mode": "tracking",
        "comfort_band": 2.0,
        "comfort_scale": 1.0,
        "comfort_mode": "time_bias_guard",
        "comfort_track_weight": 3.0,
        "comfort_cold_weight": 60.0,
        "comfort_hot_weight": 15.0,
        "comfort_cold_tolerance": 0.5,
        "comfort_hot_tolerance": 1.0,
        "comfort_track_scale": 2.0,
        "comfort_cold_scale": 1.5,
        "comfort_hot_scale": 2.0,
        "time_bias_guard_band": float(args.time_bias_guard_band),
        "time_bias_guard_penalty": float(args.time_bias_guard_penalty),
        "power_scale": float(args.power_scale),
        "power_per_unit_action": float(rc2["power_per_unit_action"]),
        "normalize_terms": False,
    }


def main() -> None:
    args = build_parser().parse_args()
    set_experiment_seed(args.seed)
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    stress_profile = load_simple_v4_price_profile(
        simulation_start=int(args.simulation_start),
        exp_beta=2.0,
    )
    if not stress_profile:
        raise RuntimeError("Missing baseline_daily_mean_exp grid-stress profile.")
    bias_profile = time_bias_setpoint_profile(
        simulation_start=int(args.simulation_start),
        simulation_end=int(args.simulation_end),
        amplitude=float(args.time_bias_amplitude),
    )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for building in buildings:
        building_id = building["result_id"]
        seed = int(args.seed + 1000 * (int(building["index"]) - 1))
        mpc_rollout_path = (
            Path("results/raw/mpc")
            / building_id
            / "no_occupant"
            / args.mpc_label
            / "rollout.csv"
        )
        checkpoint_path = (
            Path("results/models/rlmpc/offline")
            / building_id
            / f"{args.checkpoint_label}.pt"
        )
        if not mpc_rollout_path.exists():
            raise FileNotFoundError(mpc_rollout_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        mpc_rollout = pd.read_csv(mpc_rollout_path)
        if args.max_steps is not None:
            mpc_rollout = mpc_rollout.iloc[: int(args.max_steps)].copy()

        mpc, checkpoint, torch_device = load_offline_2r2c_checkpoint(checkpoint_path, device=args.device)
        mpc.use_balanced_mpc_cost(**_matched_cost(checkpoint["rc2_params"], args))
        env = make_february_raw_env(
            seed=seed,
            building_name=str(building["name"]),
            sim_start=int(args.simulation_start),
            sim_end=int(args.simulation_end),
        )
        obs, _ = env.reset(seed=seed)
        indices = observation_indices_from_names(list(env.unwrapped.buildings[0].active_observations))
        policy = TwoStateDiffMPCPolicy(
            mpc,
            indices,
            device=torch_device,
            stress_profile=stress_profile,
            setpoint_profile=_setpoint_profile(env, args.simulation_start, args.simulation_end),
            setpoint_bias_profile=bias_profile,
            sigma=0.0,
            observer="kalman",
            kalman_process_var=0.05,
            kalman_measurement_var=0.25,
            kalman_initial_var=4.0,
        )

        qpth_warning_count = 0
        rows = []
        terminated = truncated = False
        for local_step, row in mpc_rollout.iterrows():
            if terminated or truncated:
                break
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                action = policy.predict(obs)
            text = buf.getvalue()
            qpth_warning_count += text.count("qpth warning")
            u_diffmpc = float(np.asarray(action).reshape(-1)[0])
            u_mpc = float(row["action"])
            obs_flat = np.asarray(obs, dtype=float).reshape(-1)
            tin_obs = float(obs_flat[indices["tin"]])
            rows.append(
                {
                    "building": building["display"],
                    "building_id": building_id,
                    "step": int(local_step),
                    "time_step": int(row.get("time_step", local_step)),
                    "u_mpc": u_mpc,
                    "u_diffmpc": u_diffmpc,
                    "action_error": u_diffmpc - u_mpc,
                    "abs_action_error": abs(u_diffmpc - u_mpc),
                    "tin_obs": tin_obs,
                    "tin_mpc_rollout": float(row["indoor_temperature"]),
                    "tin_replay_abs_error": abs(tin_obs - float(row["indoor_temperature"])),
                    "setpoint": float(row.get("scheduled_setpoint", np.nan)),
                    "shift_signal": float(row.get("shift_signal", np.nan)),
                }
            )
            obs, _, terminated, truncated, _ = env.step([[u_mpc]])

        df = pd.DataFrame(rows)
        all_rows.extend(rows)
        err = df["action_error"].to_numpy(dtype=float)
        abs_err = np.abs(err)
        summary_rows.append(
            {
                "building": building["display"],
                "building_id": building_id,
                "steps": int(len(df)),
                "action_mae": float(abs_err.mean()) if len(abs_err) else np.nan,
                "action_rmse": float(np.sqrt(np.mean(err * err))) if len(err) else np.nan,
                "action_bias": float(err.mean()) if len(err) else np.nan,
                "action_max_abs": float(abs_err.max()) if len(abs_err) else np.nan,
                "mpc_action_mean": float(df["u_mpc"].mean()) if len(df) else np.nan,
                "diffmpc_action_mean": float(df["u_diffmpc"].mean()) if len(df) else np.nan,
                "tin_replay_max_abs_error": float(df["tin_replay_abs_error"].max()) if len(df) else np.nan,
                "qpth_warning_count": int(qpth_warning_count),
                "checkpoint_horizon": int(checkpoint["model_config"]["horizon"]),
                "time_bias_guard_penalty": float(args.time_bias_guard_penalty),
            }
        )
        print(
            f"{building['display']}: action MAE {summary_rows[-1]['action_mae']:.4f}, "
            f"max {summary_rows[-1]['action_max_abs']:.4f}, qpth warnings {qpth_warning_count}"
        )

    out_prefix = Path("analysis") / args.output_label
    detail = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summary_rows)
    detail.to_csv(out_prefix.with_name(out_prefix.name + "_action_parity.csv"), index=False)
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_action_parity_summary.csv"), index=False)
    aggregate = {
        "controller": args.output_label,
        "buildings": int(len(summary)),
        "steps": int(summary["steps"].sum()),
        "action_mae_mean": float(summary["action_mae"].mean()),
        "action_rmse_mean": float(summary["action_rmse"].mean()),
        "action_max_abs_global": float(summary["action_max_abs"].max()),
        "qpth_warning_count_total": int(summary["qpth_warning_count"].sum()),
        "checkpoint_horizon_min": int(summary["checkpoint_horizon"].min()),
        "checkpoint_horizon_max": int(summary["checkpoint_horizon"].max()),
        "time_bias_guard_penalty": float(args.time_bias_guard_penalty),
    }
    out_prefix.with_name(out_prefix.name + "_action_parity_aggregate.json").write_text(
        json.dumps(aggregate, indent=2)
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
