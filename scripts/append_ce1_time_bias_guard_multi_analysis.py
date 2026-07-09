#!/usr/bin/env python
"""Append time-biased setpoint guard RLMPC/SAC comparison results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_analysis import aggregate_controller_rollouts
from src.rewards import load_simple_v4_price_profile, time_bias_setpoint_profile


LABEL = "ce1_time_bias_guard_1c_1r1c"
SAC_FULL_ACTION_LABEL = "ce1_time_bias_guard_1c_1r1c_a1"
SAC_MPC_JUMPSTART_LABEL = "ce1_time_bias_guard_1c_1r1c_mpcjs15_a1"
SAC_BC_ONLY_LABEL = "ce1_time_bias_guard_1c_1r1c_bc_mpc_only_es_a1"
SAC_BC_SAC10_LABEL = "ce1_time_bias_guard_1c_1r1c_bc_mpc_sac10_es_a1"
SAC_EXP_SHIELD_LABEL = "ce1_time_bias_guard_1c_1r1c_sac_exp_shield_auto_a1"
FIG_DIR = Path("figures") / f"{LABEL}_multi"
ANALYSIS_DIR = Path("analysis")
RESULTS = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)
FEB_START = 744
FEB_END = 1415

RUNS = [
    {
        "key": "mpc",
        "controller": "mpc",
        "label": LABEL,
        "display": "MPC time-bias guard",
        "comfort_slug": "mpc",
    },
    {
        "key": "rlmpc",
        "controller": "rlmpc",
        "label": LABEL,
        "display": "GNU-RL/RLMPC time-bias guard",
        "comfort_slug": "rlmpc",
    },
    {
        "key": "sac_clipped",
        "controller": "sac_merlin",
        "label": LABEL,
        "display": "SAC time-bias guard (actor 0-0.5)",
        "comfort_slug": "sac_merlin",
    },
    {
        "key": "sac_full_action",
        "controller": "sac_merlin",
        "label": SAC_FULL_ACTION_LABEL,
        "display": "SAC time-bias guard (actor 0-1)",
        "comfort_slug": "sac_merlin_a1",
    },
    {
        "key": "sac_mpc_jumpstart",
        "controller": "sac_merlin",
        "label": SAC_MPC_JUMPSTART_LABEL,
        "display": "SAC time-bias guard (MPC jump-start, 15 ep, actor 0-1)",
        "comfort_slug": "sac_merlin_mpcjs15_a1",
    },
    {
        "key": "sac_bc_only",
        "controller": "sac_merlin",
        "label": SAC_BC_ONLY_LABEL,
        "display": "SAC BC-only from MPC (PI+BC, actor 0-1)",
        "comfort_slug": "sac_merlin_bc_only_es_a1",
    },
    {
        "key": "sac_bc_sac10",
        "controller": "sac_merlin",
        "label": SAC_BC_SAC10_LABEL,
        "display": "SAC BC+reward from MPC (PI+BC+10 SAC, actor 0-1)",
        "comfort_slug": "sac_merlin_bc_sac10_es_a1",
    },
    {
        "key": "sac_exp_shield_auto",
        "controller": "sac_merlin",
        "label": SAC_EXP_SHIELD_LABEL,
        "display": "SAC MERLIN structured exploration + shield + auto-alpha (actor 0-1)",
        "comfort_slug": "sac_merlin_exp_shield_auto_a1",
    },
]

KPI_CASES = [
    ("mpc", "baseline", "Baseline"),
    ("mpc", "ce1_setpoint_energy_1r1c_c05_e05", "MPC setpoint+energy 1R1C"),
    ("mpc", "simple_v4", "simple_v4"),
    *[(run["controller"], run["label"], run["display"]) for run in RUNS],
]

WEIGHTS = {
    "comfort_weight": 0.5,
    "shift_weight": 0.5,
    "comfort_track_weight": 3.0,
    "comfort_track_scale": 2.0,
    "comfort_cold_weight": 60.0,
    "comfort_cold_scale": 1.5,
    "comfort_hot_weight": 15.0,
    "comfort_hot_scale": 2.0,
    "guard_band": 2.0,
    "power_scale": 5.0,
    "bias_amplitude": 1.0,
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _rollout_path(controller: str, building_id: str, label: str = LABEL) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label / "rollout.csv"


def _summary_path(controller: str) -> Path:
    return RESULTS / "summaries" / controller / "no_occupant_summary_all_buildings.csv"


def _building_summary_path(controller: str, building_id: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / "summary.csv"


def _case_summary(controller: str, case: str) -> pd.DataFrame:
    aggregate_path = _summary_path(controller)
    aggregate_subset = pd.DataFrame()
    if aggregate_path.exists():
        aggregate = _read(aggregate_path)
        aggregate_subset = aggregate[aggregate["case"] == case].copy()

    per_building = []
    for building in BUILDINGS:
        path = _building_summary_path(controller, building["result_id"])
        if not path.exists():
            continue
        summary = _read(path)
        subset = summary[summary["case"] == case].copy()
        if not subset.empty:
            per_building.append(subset)

    if per_building:
        data = pd.concat(per_building, ignore_index=True)
        if "building_id" in data.columns:
            data = data.drop_duplicates(subset=["building_id", "case"], keep="last")
        return data
    return aggregate_subset


def _load_rollouts(controller: str, label: str = LABEL) -> dict[str, pd.DataFrame]:
    return {
        building["result_id"]: _read(_rollout_path(controller, building["result_id"], label=label))
        for building in BUILDINGS
    }


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint-like column.")


def _time_step(data: pd.DataFrame) -> np.ndarray:
    if "time_step" in data.columns:
        return pd.to_numeric(data["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(data), dtype=int)


def _heating(data: pd.DataFrame) -> np.ndarray:
    if "heating_electricity_consumption" in data.columns:
        return np.maximum(pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0), 0.0).to_numpy(dtype=float)
    return np.maximum(pd.to_numeric(data["action"], errors="coerce").fillna(0.0), 0.0).to_numpy(dtype=float)


def _bias_and_stress(n: int) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(
        time_bias_setpoint_profile(
            simulation_start=FEB_START,
            simulation_end=FEB_START + n - 1,
            amplitude=WEIGHTS["bias_amplitude"],
        ),
        dtype=float,
    )
    stress = np.asarray(load_simple_v4_price_profile(simulation_start=FEB_START, exp_beta=2.0), dtype=float)
    if stress.size < n:
        stress = np.resize(stress, n)
    return bias[:n], stress[:n]


def _controller_kpis() -> pd.DataFrame:
    frames = []
    for controller, case, display in KPI_CASES:
        subset = _case_summary(controller, case)
        if subset.empty:
            continue
        subset["controller"] = display
        frames.append(subset)
    data = pd.concat(frames, ignore_index=True)
    data["discomfort_%"] = 100.0 * pd.to_numeric(data["discomfort_proportion"], errors="coerce")
    data["cold_%"] = 100.0 * pd.to_numeric(data["discomfort_cold_proportion"], errors="coerce")
    data["hot_%"] = 100.0 * pd.to_numeric(data["discomfort_hot_proportion"], errors="coerce")
    columns = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
    ]
    if "ramping_average" in data.columns:
        columns.append("ramping_average")
    out = data.groupby("controller", as_index=False)[columns].mean()
    order = {display: i for i, (_, _, display) in enumerate(KPI_CASES)}
    out["_order"] = out["controller"].map(order)
    return out.sort_values("_order").drop(columns="_order")


def _baseline_district() -> pd.DataFrame:
    data = _read(RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv")
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": data["net_electricity_consumption"].to_numpy(dtype=float),
            "target": data["district_target"].to_numpy(dtype=float),
            "controller": "CityLearn baseline",
        }
    )


def _district_frame(run: dict[str, str]) -> pd.DataFrame:
    frame, missing = aggregate_controller_rollouts(controller=run["controller"], case=run["label"], buildings=BUILDINGS)
    if missing:
        raise FileNotFoundError(f"Missing {run['display']} rollouts: {missing[:3]}")
    baseline = _baseline_district()
    n = min(len(frame), len(baseline))
    return pd.DataFrame(
        {
            "time_step": frame["time_step"].to_numpy(dtype=int)[:n],
            "load": frame["net_electricity_consumption"].to_numpy(dtype=float)[:n],
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "controller": run["display"],
        }
    )


def _district_kpis(district_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    baseline = district_frames[0]
    base_load = baseline["load"].to_numpy(dtype=float)
    base_total = max(float(np.sum(base_load)), 1e-9)
    base_peak = max(float(np.max(base_load)), 1e-9)
    base_ramp = max(float(np.sum(np.abs(np.diff(base_load)))), 1e-9)
    for frame in district_frames:
        load = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        n = min(len(load), len(target))
        load, target = load[:n], target[:n]
        hourly = pd.DataFrame({"load": load, "hour": np.arange(n) % 24}).groupby("hour")["load"].mean()
        rows.append(
            {
                "controller": frame["controller"].iloc[0],
                "total_kwh": float(np.sum(load)),
                "total_vs_citylearn": float(np.sum(load) / base_total),
                "peak_kwh": float(np.max(load)),
                "peak_vs_citylearn": float(np.max(load) / base_peak),
                "daily_profile_peak_kwh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))),
                "ramping_vs_citylearn": float(np.sum(np.abs(np.diff(load))) / base_ramp),
                "target_rmse_kwh": float(np.sqrt(np.mean((load - target) ** 2))),
                "above_target_kwh": float(np.sum(np.maximum(load - target, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _term_diagnostics(run: dict[str, str], rollouts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    hourly_rows = []
    display = run["display"]
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data)
        heat = _heating(data)
        time_step = _time_step(data)
        n = min(len(tin), len(tsp), len(heat), len(time_step))
        tin, tsp, heat, time_step = tin[:n], tsp[:n], heat[:n], time_step[:n]
        bias, stress = _bias_and_stress(n)
        tref = tsp + bias
        cold_guard = np.maximum(tsp - WEIGHTS["guard_band"] - tin, 0.0)
        hot_guard = np.maximum(tin - tsp - WEIGHTS["guard_band"], 0.0)
        track = (
            WEIGHTS["comfort_weight"]
            * WEIGHTS["comfort_track_weight"]
            * ((tin - tref) / WEIGHTS["comfort_track_scale"]) ** 2
        )
        cold = (
            WEIGHTS["comfort_weight"]
            * WEIGHTS["comfort_cold_weight"]
            * (cold_guard / WEIGHTS["comfort_cold_scale"]) ** 2
        )
        hot = (
            WEIGHTS["comfort_weight"]
            * WEIGHTS["comfort_hot_weight"]
            * (hot_guard / WEIGHTS["comfort_hot_scale"]) ** 2
        )
        shift = WEIGHTS["shift_weight"] * stress * heat / WEIGHTS["power_scale"]
        stage = track + cold + hot + shift
        guard_violation = (cold_guard > 0.0) | (hot_guard > 0.0)
        term_values = {
            "tracking_target": track,
            "guard_cold": cold,
            "guard_hot": hot,
            "time_weighted_heat": shift,
            "stage_cost": stage,
        }
        for term, values in term_values.items():
            rows.append(
                {
                    "controller": display,
                    "building": building["display"],
                    "term": term,
                    "mean": float(np.nanmean(values)),
                    "max": float(np.nanmax(values)),
                }
            )
        rows.append(
            {
                "controller": display,
                "building": building["display"],
                "term": "mae_to_true_tsp_c",
                "mean": float(np.nanmean(np.abs(tin - tsp))),
                "max": float(np.nanmax(np.abs(tin - tsp))),
            }
        )
        rows.append(
            {
                "controller": display,
                "building": building["display"],
                "term": "mae_to_biased_tref_c",
                "mean": float(np.nanmean(np.abs(tin - tref))),
                "max": float(np.nanmax(np.abs(tin - tref))),
            }
        )
        rows.append(
            {
                "controller": display,
                "building": building["display"],
                "term": "guard_violation_%",
                "mean": float(100.0 * np.nanmean(guard_violation)),
                "max": float(100.0 * np.nanmax(guard_violation)),
            }
        )
        hourly_rows.append(
            pd.DataFrame(
                {
                    "controller": display,
                    "hour": (time_step + FEB_START) % 24,
                    "tin_minus_tsp": tin - tsp,
                    "bias": bias,
                    "heat": heat,
                    "guard_violation": guard_violation.astype(float),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def _make_comfort_grid(run: dict[str, str], rollouts: dict[str, pd.DataFrame]) -> Path:
    display = run["display"]
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        data = rollouts[building["result_id"]]
        time_step = _time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data)
        n = min(len(time_step), len(tin), len(tsp))
        time_step, tin, tsp = time_step[:n], tin[:n], tsp[:n]
        bias, _ = _bias_and_stress(n)
        ax.plot(time_step, tin, lw=0.75, color="steelblue", label="T_in")
        ax.plot(time_step, tsp, lw=0.65, ls="--", color="black", label="Tsp")
        ax.plot(time_step, tsp + bias, lw=0.65, ls=":", color="tab:orange", label="Tsp+bias")
        ax.fill_between(time_step, tsp - WEIGHTS["guard_band"], tsp + WEIGHTS["guard_band"], alpha=0.10, color="green")
        ax.set_title(building["display"], fontsize=8)
        ax.set_ylim(14, 26)
        ax.grid(alpha=0.2)
    for ax in axes[len(BUILDINGS):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{display} - time-biased target with true-setpoint +/-2 C guard", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['comfort_slug']}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    target = frames[0][["time_step", "target"]]
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=name)
    ax.plot(target["time_step"], target["target"], color="black", lw=1.0, ls="--", label="CE1 district target")
    ax.set_title(f"Time-bias guard district load - {LABEL}")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries_multi.png"
    fig.savefig(load_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = target.assign(hour=target["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.4, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.1, ls="--", label="CE1 district target")
    ax.set_title("Mean hourly district load - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile_multi.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


def _make_term_plots(term_summary: pd.DataFrame, hourly: pd.DataFrame) -> tuple[Path, Path]:
    plot_terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]
    plot_data = term_summary[term_summary["term"].isin(plot_terms)].copy()
    pivot = plot_data.pivot(index="term", columns="controller", values="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.loc[plot_terms].plot(kind="bar", ax=ax)
    ax.set_title("Mean weighted stage-cost terms by controller")
    ax.set_ylabel("Mean weighted term value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    terms_path = FIG_DIR / "weighted_terms_by_controller.png"
    fig.savefig(terms_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    hourly_mean = hourly.groupby(["controller", "hour"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for name, group in hourly_mean.groupby("controller", sort=False):
        axes[0].plot(group["hour"], group["tin_minus_tsp"], marker="o", lw=1.2, label=name)
        axes[1].plot(group["hour"], group["heat"], marker="o", lw=1.2, label=name)
        axes[2].plot(group["hour"], 100.0 * group["guard_violation"], marker="o", lw=1.2, label=name)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("Tin - Tsp [C]")
    axes[1].set_ylabel("Heating electricity [kWh]")
    axes[2].set_ylabel("Guard violation [%]")
    axes[2].set_xlabel("Hour of day")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Hourly response to +1/-1 C time-bias schedule")
    fig.tight_layout()
    response_path = FIG_DIR / "hourly_bias_response_multi.png"
    fig.savefig(response_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return terms_path, response_path


def _case_dir(run: dict[str, str], building_id: str) -> Path:
    return RESULTS / "raw" / run["controller"] / building_id / "no_occupant" / run["label"]


def _load_bc_losses(runs: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for building in BUILDINGS:
            path = _case_dir(run, building["result_id"]) / "bc_training_losses.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = building["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_sac_losses(runs: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for building in BUILDINGS:
            path = _case_dir(run, building["result_id"]) / "sac_training_losses.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = building["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_online_rewards(runs: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for building in BUILDINGS:
            path = _case_dir(run, building["result_id"]) / "online_training_rollout.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = building["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _make_training_plots(
    bc_losses: pd.DataFrame,
    sac_losses: pd.DataFrame,
    online_rewards: pd.DataFrame,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths: list[Path] = []
    bc_summary = pd.DataFrame()
    sac_summary = pd.DataFrame()
    reward_summary = pd.DataFrame()
    exploration_summary = pd.DataFrame()

    if not bc_losses.empty:
        bc_summary = (
            bc_losses.groupby(["controller", "epoch"], as_index=False)
            .agg(
                bc_train_loss_mean=("bc_train_loss", "mean"),
                bc_train_loss_max=("bc_train_loss", "max"),
                bc_val_loss_mean=("bc_val_loss", "mean"),
                bc_val_loss_max=("bc_val_loss", "max"),
                bc_val_mae_mean=("bc_val_mae", "mean"),
                bc_val_abs_error_max=("bc_val_abs_error_max", "max"),
            )
            .sort_values(["controller", "epoch"])
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, group in bc_summary.groupby("controller", sort=False):
            ax.plot(group["epoch"], group["bc_train_loss_mean"], lw=1.4, label=f"{name} train mean")
            ax.plot(group["epoch"], group["bc_val_loss_mean"], lw=1.4, ls="--", label=f"{name} val mean")
            ax.plot(group["epoch"], group["bc_val_loss_max"], lw=0.9, ls=":", label=f"{name} val max")
        ax.set_title("Behavior cloning loss over epochs")
        ax.set_xlabel("BC epoch")
        ax.set_ylabel("MSE(action, MPC action)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "bc_training_losses_multi.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not sac_losses.empty:
        loss_columns = [
            "soft_q1_loss",
            "soft_q2_loss",
            "policy_loss",
            "alpha_loss",
            "alpha",
            "log_pi_mean",
            "q_new_actions_mean",
            "reward_batch_mean",
        ]
        available = [col for col in loss_columns if col in sac_losses.columns]
        sac_summary = (
            sac_losses.groupby(["controller", "episode"], as_index=False)[available]
            .agg(["mean", "max"])
        )
        sac_summary.columns = [
            "_".join(col).strip("_") if isinstance(col, tuple) else col
            for col in sac_summary.columns.to_flat_index()
        ]
        fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.3 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]
        for ax, col in zip(axes, available):
            for name, group in sac_summary.groupby("controller", sort=False):
                ax.plot(group["episode"], group[f"{col}_mean"], marker="o", lw=1.2, label=f"{name} mean")
                ax.plot(group["episode"], group[f"{col}_max"], marker="o", lw=0.9, ls="--", label=f"{name} max")
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Training episode")
        fig.suptitle("SAC critic/policy losses during reward training")
        fig.tight_layout()
        path = FIG_DIR / "sac_training_losses_multi.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not online_rewards.empty:
        reward_summary = (
            online_rewards.groupby(["controller", "episode", "phase"], as_index=False)
            .agg(reward_mean=("reward", "mean"), reward_sum=("reward", "sum"), reward_min=("reward", "min"))
            .sort_values(["controller", "episode", "phase"])
        )
        episode_reward = (
            online_rewards.groupby(["controller", "building", "episode"], as_index=False)["reward"].sum()
            .groupby(["controller", "episode"], as_index=False)
            .agg(reward_sum_mean=("reward", "mean"), reward_sum_max=("reward", "max"), reward_sum_min=("reward", "min"))
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, group in episode_reward.groupby("controller", sort=False):
            ax.plot(group["episode"], group["reward_sum_mean"], marker="o", lw=1.4, label=f"{name} mean")
            ax.fill_between(
                group["episode"],
                group["reward_sum_min"],
                group["reward_sum_max"],
                alpha=0.12,
            )
        ax.set_title("Online training reward by episode")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "online_training_rewards_multi.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        agg: dict[str, tuple[str, str]] = {
            "action_mean": ("action", "mean"),
            "action_max": ("action", "max"),
            "reward_mean": ("reward", "mean"),
            "reward_sum": ("reward", "sum"),
        }
        for column in [
            "policy_action",
            "policy_mean_action",
            "policy_log_std",
            "policy_std",
            "exploration_rate",
            "structured_exploration_applied",
            "comfort_shield_applied",
            "tin_before_action",
            "tsp_before_action",
        ]:
            if column in online_rewards.columns:
                agg[f"{column}_mean"] = (column, "mean")
                if column in {"structured_exploration_applied", "comfort_shield_applied"}:
                    agg[f"{column}_max"] = (column, "max")
        exploration_summary = (
            online_rewards.groupby(["controller", "episode"], as_index=False)
            .agg(**agg)
            .sort_values(["controller", "episode"])
        )
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        for name, group in exploration_summary.groupby("controller", sort=False):
            axes[0].plot(group["episode"], group["action_mean"], marker="o", lw=1.2, label=f"{name} actual")
            if "policy_action_mean" in group:
                axes[0].plot(group["episode"], group["policy_action_mean"], marker="o", lw=0.9, ls="--", label=f"{name} policy")
            if "structured_exploration_applied_mean" in group:
                axes[1].plot(group["episode"], group["structured_exploration_applied_mean"], marker="o", lw=1.2, label=name)
            if "comfort_shield_applied_mean" in group:
                axes[2].plot(group["episode"], group["comfort_shield_applied_mean"], marker="o", lw=1.2, label=name)
            if "policy_std_mean" in group:
                axes[3].plot(group["episode"], group["policy_std_mean"], marker="o", lw=1.2, label=name)
        axes[0].set_ylabel("action")
        axes[1].set_ylabel("structured exploration frac")
        axes[2].set_ylabel("shield frac")
        axes[3].set_ylabel("policy std")
        axes[3].set_xlabel("Training episode")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.suptitle("SAC exploration and safety diagnostics")
        fig.tight_layout()
        path = FIG_DIR / "sac_exploration_diagnostics_multi.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths, bc_summary, sac_summary, reward_summary, exploration_summary


def _markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    data = df.copy()
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(data.columns)
    rows = [headers] + data.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(row) for row in rows[1:]]])


def _append_once(path: Path, marker: str, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if marker in str(cell.get("source", "")):
            notebook.cells[index:index + len(cells)] = cells
            nbf.write(notebook, path)
            print(f"Updated existing {marker} section in {path}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {marker} to {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rollouts = {run["key"]: _load_rollouts(run["controller"], run["label"]) for run in RUNS}
    kpis = _controller_kpis()
    kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_controller_kpis.csv", index=False)

    district_frames = [_baseline_district(), *[_district_frame(run) for run in RUNS]]
    district_kpis = _district_kpis(district_frames)
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_district_kpis.csv", index=False)
    district_load_path, district_daily_path = _make_district_plots(district_frames)

    comfort_paths = {
        run["key"]: _make_comfort_grid(run, rollouts[run["key"]])
        for run in RUNS
    }

    term_frames = []
    hourly_frames = []
    for run in RUNS:
        terms, hourly = _term_diagnostics(run, rollouts[run["key"]])
        term_frames.append(terms)
        hourly_frames.append(hourly)
    terms_by_building = pd.concat(term_frames, ignore_index=True)
    hourly_response = pd.concat(hourly_frames, ignore_index=True)
    terms_by_building.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_term_diagnostics_by_building.csv", index=False)
    hourly_response.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_hourly_bias_response.csv", index=False)
    term_summary = (
        terms_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_term_diagnostics.csv", index=False)
    terms_path, response_path = _make_term_plots(term_summary, hourly_response)

    bc_losses = _load_bc_losses(RUNS)
    sac_losses = _load_sac_losses(RUNS)
    online_rewards = _load_online_rewards(RUNS)
    bc_losses.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_bc_training_losses_raw.csv", index=False)
    sac_losses.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_sac_training_losses_raw.csv", index=False)
    online_rewards.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_online_training_rewards_raw.csv", index=False)
    training_paths, bc_summary, sac_loss_summary, reward_summary, exploration_summary = _make_training_plots(
        bc_losses,
        sac_losses,
        online_rewards,
    )
    if not bc_summary.empty:
        bc_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_bc_training_loss_summary.csv", index=False)
    if not sac_loss_summary.empty:
        sac_loss_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_sac_training_loss_summary.csv", index=False)
    if not reward_summary.empty:
        reward_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_online_reward_summary.csv", index=False)
    if not exploration_summary.empty:
        exploration_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_exploration_summary.csv", index=False)

    qpth_warning_count = 0
    rl_log = Path("/tmp/rlmpc_time_bias_guard.log")
    if rl_log.exists():
        qpth_warning_count = rl_log.read_text(errors="ignore").count("qpth warning")

    overview_md = f"""## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation

Run labels: `{LABEL}` for MPC/GNU-RL/RLMPC/clipped SAC; `{SAC_FULL_ACTION_LABEL}` for full-action SAC; `{SAC_MPC_JUMPSTART_LABEL}` for SAC with a January MPC jump-start and 15 training episodes; `{SAC_BC_ONLY_LABEL}` for PI + MPC behavior cloning only; `{SAC_BC_SAC10_LABEL}` for PI + MPC behavior cloning + 10 SAC reward-training episodes; `{SAC_EXP_SHIELD_LABEL}` for PI jump-start + 10 SAC reward-training episodes with structured exploration, learned entropy alpha, and a cold comfort shield.

Cost/reward used for the new GNU-RL/RLMPC and SAC runs:

`J_k = 0.5 * [3*((Tin-(Tsp+bias))/2)^2 + 60*(cold_guard/1.5)^2 + 15*(hot_guard/2)^2] + 0.5*GridStress*P_heat/5`

`reward = -J_k`, `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and `cold_guard/hot_guard` are measured against the true setpoint with a `Tsp +/- 2 C` guard. Comfort and time-weighted heating weights sum to 1.0.

GNU-RL/RLMPC uses the same fixed matched inner MPC cost, so PPO cost-parameter updates are skipped. The qpth solver emitted `{qpth_warning_count}` inaccurate-solution warnings during the full RLMPC run; treat that controller as informative but numerically less clean than explicit MPC/SAC.

The full-action SAC row uses the same MERLIN-style pipeline and reward as clipped SAC, with only `action_scaling_coefficient=1.0` instead of `0.5`. The MPC jump-start SAC row replaces the first January PI exploration episode with actions generated by the same 1R1C time-bias MPC teacher, then trains SAC for the remaining episodes. The BC rows keep the January PI replay episode, train the SAC actor offline against January MPC action labels with validation early stopping (`patience=5`), then either deploy immediately or continue with 10 SAC reward-training episodes before February deployment. The exploration/shield SAC row removes MPC behavior cloning, learns the entropy coefficient (`target_entropy=-1`), replaces some SAC actions with structured random high-heat exploration during January reward training, and applies a PI-style heating floor when `Tin < Tsp - 2 C` during training and February deployment.

Mean building KPIs:

{_markdown_table(kpis)}

District aggregate KPIs:

{_markdown_table(district_kpis)}

Weighted term diagnostics:

{_markdown_table(term_summary)}
"""
    if not bc_summary.empty:
        overview_md += f"""

Behavior-cloning loss summary:

{_markdown_table(bc_summary.groupby("controller", as_index=False).tail(1))}
"""
    if not sac_loss_summary.empty:
        overview_md += f"""

SAC reward-training loss summary by final episode:

{_markdown_table(sac_loss_summary.groupby("controller", as_index=False).tail(1))}
"""
    if not reward_summary.empty:
        overview_md += f"""

Online reward summary:

{_markdown_table(reward_summary.tail(20))}
"""
    if not exploration_summary.empty:
        overview_md += f"""

SAC exploration/shield diagnostics by final episode:

{_markdown_table(exploration_summary.groupby("controller", as_index=False).tail(1))}
"""

    reward_landscape_paths = [
        Path("figures/reward_landscape/time_bias_guard_immediate_reward_surface.png"),
        Path("figures/reward_landscape/time_bias_guard_one_step_proxy_reward_surface_b8_1r1c.png"),
        Path("figures/reward_landscape/time_bias_guard_reward_vs_action_curves.png"),
    ]
    reward_landscape_paths = [path for path in reward_landscape_paths if path.exists()]

    image_cells = [
        nbf.v4.new_markdown_cell(f"![District load](../{district_load_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean hourly district profile](../{district_daily_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Weighted stage-cost terms](../{terms_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Hourly bias response](../{response_path.as_posix()})"),
        *[nbf.v4.new_markdown_cell(f"![Training diagnostic](../{path.as_posix()})") for path in training_paths],
        *[nbf.v4.new_markdown_cell(f"![Reward landscape](../{path.as_posix()})") for path in reward_landscape_paths],
    ]

    common_cells = [nbf.v4.new_markdown_cell(overview_md), *image_cells]
    _append_once(
        ANALYSIS_DIR / "02_mpc_results.ipynb",
        f"## CE1 Time-Biased Setpoint Guard MPC/SAC Comparison - `{LABEL}`",
        [
            nbf.v4.new_markdown_cell(overview_md.replace("## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation", f"## CE1 Time-Biased Setpoint Guard MPC/SAC Comparison - `{LABEL}`")),
            nbf.v4.new_markdown_cell(f"![MPC comfort all buildings](../{comfort_paths['mpc'].as_posix()})"),
            *image_cells,
        ],
    )
    _append_once(
        ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
        f"## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation",
        common_cells,
    )
    _append_once(
        ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
        f"## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation",
        common_cells,
    )
    _append_once(
        ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
        f"## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation",
        [
            *common_cells,
            nbf.v4.new_markdown_cell(f"![MPC comfort all buildings](../{comfort_paths['mpc'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![GNU-RL/RLMPC comfort all buildings](../{comfort_paths['rlmpc'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC clipped comfort all buildings](../{comfort_paths['sac_clipped'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC full-action comfort all buildings](../{comfort_paths['sac_full_action'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC MPC jump-start comfort all buildings](../{comfort_paths['sac_mpc_jumpstart'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC BC-only comfort all buildings](../{comfort_paths['sac_bc_only'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC BC+reward comfort all buildings](../{comfort_paths['sac_bc_sac10'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC exploration/shield comfort all buildings](../{comfort_paths['sac_exp_shield_auto'].as_posix()})"),
        ],
    )
    _append_once(
        ANALYSIS_DIR / "04_rlmpc_results.ipynb",
        f"## CE1 Time-Biased Setpoint Guard RLMPC - `{LABEL}`",
        [
            nbf.v4.new_markdown_cell(overview_md.replace("## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation", f"## CE1 Time-Biased Setpoint Guard RLMPC - `{LABEL}`")),
            nbf.v4.new_markdown_cell(f"![GNU-RL/RLMPC comfort all buildings](../{comfort_paths['rlmpc'].as_posix()})"),
            *image_cells,
        ],
    )
    _append_once(
        ANALYSIS_DIR / "08_sac_results.ipynb",
        f"## CE1 Time-Biased Setpoint Guard SAC Full-Action Ablation - `{SAC_FULL_ACTION_LABEL}`",
        [
            nbf.v4.new_markdown_cell(overview_md.replace("## CE1 Time-Biased Setpoint Guard - Full-Action SAC Ablation", f"## CE1 Time-Biased Setpoint Guard SAC Full-Action Ablation - `{SAC_FULL_ACTION_LABEL}`")),
            nbf.v4.new_markdown_cell(f"![SAC clipped comfort all buildings](../{comfort_paths['sac_clipped'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC full-action comfort all buildings](../{comfort_paths['sac_full_action'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC MPC jump-start comfort all buildings](../{comfort_paths['sac_mpc_jumpstart'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC BC-only comfort all buildings](../{comfort_paths['sac_bc_only'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC BC+reward comfort all buildings](../{comfort_paths['sac_bc_sac10'].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![SAC exploration/shield comfort all buildings](../{comfort_paths['sac_exp_shield_auto'].as_posix()})"),
            *image_cells,
        ],
    )


if __name__ == "__main__":
    main()
