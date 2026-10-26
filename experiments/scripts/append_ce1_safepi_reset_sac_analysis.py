#!/usr/bin/env python
"""Append safe-PI reset SAC results to SAC analysis notebooks."""

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


LABEL_CLIPPED = "ce1_time_bias_guard_1c_1r1c"
LABEL_A1 = "ce1_time_bias_guard_1c_1r1c_a1"
LABEL_EXP = "ce1_time_bias_guard_1c_1r1c_sac_exp_shield_auto_a1"
LABEL_RC = "ce1_rc_rollout_1r1c_sac_merlin_a1"
LABEL_SAFE = "ce1_time_bias_guard_1c_1r1c_sac_safepi_reset_auto_a1"
LABEL_PENALTY = "ce1_time_bias_guard_1c_1r1c_sac_safepi_penalty100_auto_a1"

FIG_DIR = Path("figures") / LABEL_PENALTY
ANALYSIS_DIR = Path("analysis")
RESULTS = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)
FEB_START = 744

RUNS = [
    {
        "key": "sac_clipped",
        "controller": "sac_merlin",
        "label": LABEL_CLIPPED,
        "display": "SAC time-bias guard (actor 0-0.5)",
        "slug": "sac_clipped",
    },
    {
        "key": "sac_a1",
        "controller": "sac_merlin",
        "label": LABEL_A1,
        "display": "SAC time-bias guard (actor 0-1)",
        "slug": "sac_a1",
    },
    {
        "key": "sac_exp",
        "controller": "sac_merlin",
        "label": LABEL_EXP,
        "display": "SAC structured exploration + shield + auto-alpha",
        "slug": "sac_exp_shield_auto",
    },
    {
        "key": "sac_rc",
        "controller": "sac_merlin",
        "label": LABEL_RC,
        "display": "SAC 1R1C RC-rollout obs (actor 0-1)",
        "slug": "sac_rc_rollout",
    },
    {
        "key": "sac_safe",
        "controller": "sac_merlin",
        "label": LABEL_SAFE,
        "display": "SAC safe-PI reset + auto-alpha (actor 0-1)",
        "slug": "sac_safepi_reset",
    },
    {
        "key": "sac_penalty100",
        "controller": "sac_merlin",
        "label": LABEL_PENALTY,
        "display": "SAC safe-PI reset + additive reset penalty -100",
        "slug": "sac_safepi_reset_penalty100",
    },
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


def _case_dir(run: dict, building_id: str) -> Path:
    return RESULTS / "raw" / run["controller"] / building_id / "no_occupant" / run["label"]


def _rollout_path(run: dict, building_id: str) -> Path:
    return _case_dir(run, building_id) / "rollout.csv"


def _summary_path(controller: str, building_id: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / "summary.csv"


def _case_summary(run: dict) -> pd.DataFrame:
    frames = []
    for building in BUILDINGS:
        path = _summary_path(run["controller"], building["result_id"])
        if not path.exists():
            continue
        data = _read(path)
        subset = data[data["case"] == run["label"]].copy()
        if not subset.empty:
            frames.append(subset.tail(1))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["controller"] = run["display"]
    return out


def _run_available(run: dict) -> bool:
    return all(_rollout_path(run, b["result_id"]).exists() for b in BUILDINGS)


def _available_runs() -> list[dict]:
    runs = []
    for run in RUNS:
        if _run_available(run):
            runs.append(run)
        else:
            print(f"Skipping incomplete run: {run['label']}")
    return runs


def _load_rollouts(run: dict) -> dict[str, pd.DataFrame]:
    return {b["result_id"]: _read(_rollout_path(run, b["result_id"])) for b in BUILDINGS}


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("No setpoint column found.")


def _time_step(data: pd.DataFrame) -> np.ndarray:
    if "time_step" in data.columns:
        return pd.to_numeric(data["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(data), dtype=int)


def _heating(data: pd.DataFrame) -> np.ndarray:
    if "heating_electricity_consumption" in data.columns:
        return np.maximum(
            pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0),
            0.0,
        ).to_numpy(dtype=float)
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


def _controller_kpis(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        summary = _case_summary(run)
        if not summary.empty:
            frames.append(summary)
    data = pd.concat(frames, ignore_index=True)
    data["discomfort_%"] = 100.0 * pd.to_numeric(data["discomfort_proportion"], errors="coerce")
    data["cold_%"] = 100.0 * pd.to_numeric(data["discomfort_cold_proportion"], errors="coerce")
    data["hot_%"] = 100.0 * pd.to_numeric(data["discomfort_hot_proportion"], errors="coerce")
    cols = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
    ]
    if "ramping_average" in data.columns:
        cols.append("ramping_average")
    out = data.groupby("controller", as_index=False)[cols].mean(numeric_only=True)
    order = {run["display"]: i for i, run in enumerate(runs)}
    out["_order"] = out["controller"].map(order)
    return out.sort_values("_order").drop(columns="_order")


def _baseline_district() -> pd.DataFrame:
    data = _read(RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv")
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(),
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0).to_numpy(),
            "controller": "CityLearn baseline",
        }
    )


def _district_frame(run: dict) -> pd.DataFrame:
    frame, missing = aggregate_controller_rollouts(
        controller=run["controller"],
        case=run["label"],
        buildings=BUILDINGS,
    )
    if missing:
        raise FileNotFoundError(f"Missing rollouts for {run['label']}: {missing[:3]}")
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


def _district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    base = frames[0]["load"].to_numpy(dtype=float)
    base_total = max(float(base.sum()), 1e-9)
    base_peak = max(float(base.max()), 1e-9)
    base_ramp = max(float(np.abs(np.diff(base)).sum()), 1e-9)
    for frame in frames:
        load = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        n = min(len(load), len(target))
        load, target = load[:n], target[:n]
        hourly = pd.DataFrame({"hour": np.arange(n) % 24, "load": load}).groupby("hour")["load"].mean()
        ramp = float(np.abs(np.diff(load)).sum()) if n > 1 else 0.0
        rows.append(
            {
                "controller": frame["controller"].iloc[0],
                "total_kwh": float(load.sum()),
                "total_vs_citylearn": float(load.sum()) / base_total,
                "peak_kwh": float(load.max()),
                "peak_vs_citylearn": float(load.max()) / base_peak,
                "daily_profile_peak_kwh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kwh": ramp,
                "ramping_vs_citylearn": ramp / base_ramp,
                "target_rmse_kwh": float(np.sqrt(np.mean((load - target) ** 2))),
                "above_target_kwh": float(np.maximum(load - target, 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _term_diagnostics(run: dict, rollouts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    hourly = []
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
        track = WEIGHTS["comfort_weight"] * WEIGHTS["comfort_track_weight"] * (
            (tin - tref) / WEIGHTS["comfort_track_scale"]
        ) ** 2
        cold = WEIGHTS["comfort_weight"] * WEIGHTS["comfort_cold_weight"] * (
            cold_guard / WEIGHTS["comfort_cold_scale"]
        ) ** 2
        hot = WEIGHTS["comfort_weight"] * WEIGHTS["comfort_hot_weight"] * (
            hot_guard / WEIGHTS["comfort_hot_scale"]
        ) ** 2
        shift = WEIGHTS["shift_weight"] * stress * heat / WEIGHTS["power_scale"]
        stage = track + cold + hot + shift
        guard = (cold_guard > 0.0) | (hot_guard > 0.0)
        for term, values in [
            ("tracking_target", track),
            ("guard_cold", cold),
            ("guard_hot", hot),
            ("time_weighted_heat", shift),
            ("stage_cost", stage),
            ("mae_to_true_tsp_c", np.abs(tin - tsp)),
            ("mae_to_biased_tref_c", np.abs(tin - tref)),
            ("guard_violation_%", 100.0 * guard.astype(float)),
        ]:
            rows.append(
                {
                    "controller": run["display"],
                    "building": building["display"],
                    "term": term,
                    "mean": float(np.nanmean(values)),
                    "max": float(np.nanmax(values)),
                }
            )
        hourly.append(
            pd.DataFrame(
                {
                    "controller": run["display"],
                    "hour": (time_step + FEB_START) % 24,
                    "tin_minus_tsp": tin - tsp,
                    "heat": heat,
                    "guard_violation": guard.astype(float),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(hourly, ignore_index=True)


def _online_rollouts(run: dict) -> pd.DataFrame:
    frames = []
    for building in BUILDINGS:
        path = _case_dir(run, building["result_id"]) / "online_training_rollout.csv"
        if not path.exists():
            continue
        data = _read(path)
        data["controller"] = run["display"]
        data["building"] = building["display"]
        data["building_id"] = building["result_id"]
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _sac_losses(runs: list[dict]) -> pd.DataFrame:
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


def _new_building_diagnostics(run: dict, rollouts: dict[str, pd.DataFrame], online: pd.DataFrame) -> pd.DataFrame:
    summary = _case_summary(run)
    rows = []
    for building in BUILDINGS:
        b_id = building["result_id"]
        rollout = rollouts[b_id]
        online_b = online[online["building_id"] == b_id]
        jan0 = online_b[online_b["episode"] == 0]
        summary_b = summary[summary["building_id"] == b_id]
        row = {
            "building": building["display"],
            "electricity_consumption_total": np.nan,
            "discomfort_%": np.nan,
            "cold_%": np.nan,
            "hot_%": np.nan,
            "feb_tin_min": float(pd.to_numeric(rollout["indoor_temperature"], errors="coerce").min()),
            "feb_tin_max": float(pd.to_numeric(rollout["indoor_temperature"], errors="coerce").max()),
            "jan_ep0_tin_min": float(pd.to_numeric(jan0["tin_before_action"], errors="coerce").min()) if not jan0.empty else np.nan,
            "jan_ep0_tin_max": float(pd.to_numeric(jan0["tin_before_action"], errors="coerce").max()) if not jan0.empty else np.nan,
            "jan_ep0_raw_next_tin_min": float(pd.to_numeric(jan0["temperature_reset_raw_tin"], errors="coerce").min()) if not jan0.empty else np.nan,
            "jan_ep0_raw_next_tin_max": float(pd.to_numeric(jan0["temperature_reset_raw_tin"], errors="coerce").max()) if not jan0.empty else np.nan,
            "reset_steps": int(pd.to_numeric(online_b["temperature_reset_applied"], errors="coerce").fillna(0).sum()) if not online_b.empty else 0,
            "replay_dropped_steps": int((~online_b["replay_transition_kept"].astype(bool)).sum()) if not online_b.empty else 0,
            "safe_pi_bootstrap_steps": int((online_b["phase"] == "safe_pi_bootstrap").sum()) if not online_b.empty else 0,
        }
        if not summary_b.empty:
            row["electricity_consumption_total"] = float(summary_b["electricity_consumption_total"].iloc[-1])
            row["discomfort_%"] = 100.0 * float(summary_b["discomfort_proportion"].iloc[-1])
            row["cold_%"] = 100.0 * float(summary_b["discomfort_cold_proportion"].iloc[-1])
            row["hot_%"] = 100.0 * float(summary_b["discomfort_hot_proportion"].iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def _save_comfort_grid(run: dict, rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        t = _time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data)
        n = min(len(t), len(tin), len(tsp))
        t, tin, tsp = t[:n], tin[:n], tsp[:n]
        bias, _ = _bias_and_stress(n)
        ax.plot(t, tin, lw=0.8, label="Indoor temperature")
        ax.plot(t, tsp, "--", lw=0.7, color="black", label="Setpoint")
        ax.plot(t, tsp + bias, lw=0.7, color="tab:orange", label="Effective setpoint")
        ax.fill_between(t, tsp - 2.0, tsp + 2.0, color="gray", alpha=0.16, label="+/-2 C guard")
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(BUILDINGS):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | February comfort", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "february_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_january_safe_pi_grid(run: dict, online: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = online[(online["building_id"] == building["result_id"]) & (online["episode"] == 0)].copy()
        t = pd.to_numeric(data["absolute_time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
        tin = pd.to_numeric(data["tin_before_action"], errors="coerce").to_numpy(dtype=float)
        tsp = pd.to_numeric(data["tsp_before_action"], errors="coerce").to_numpy(dtype=float)
        if len(t) == 0:
            ax.axis("off")
            continue
        ax.plot(t, tin, lw=0.9, label="Indoor temperature")
        ax.plot(t, tsp, "--", lw=0.7, color="black", label="Setpoint")
        ax.fill_between(t, tsp - 2.0, tsp + 2.0, color="gray", alpha=0.16, label="+/-2 C guard")
        ax.set_title(building["display"], fontsize=8)
        ax.set_ylim(10.0, 30.0)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(BUILDINGS):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle("January episode 0 safe-PI replay fill | state seen by SAC after safety reset", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "january_safe_pi_episode0_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    target = frames[0][["time_step", "target"]]
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=name)
    ax.plot(target["time_step"], target["target"], color="black", lw=1.0, ls="--", label="CE1 district target")
    ax.set_title("SAC-only district load - February")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = target.assign(hour=target["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.3, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Mean hourly district load - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


def _save_term_plots(term_summary: pd.DataFrame, hourly: pd.DataFrame) -> tuple[Path, Path]:
    terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]
    pivot = term_summary[term_summary["term"].isin(terms)].pivot(index="term", columns="controller", values="mean")
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.loc[[t for t in terms if t in pivot.index]].plot(kind="bar", ax=ax)
    ax.set_title("Mean weighted reward/cost terms")
    ax.set_ylabel("Mean weighted term")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    terms_path = FIG_DIR / "weighted_terms_by_controller.png"
    fig.savefig(terms_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    hourly_mean = hourly.groupby(["controller", "hour"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for name, group in hourly_mean.groupby("controller", sort=False):
        axes[0].plot(group["hour"], group["tin_minus_tsp"], marker="o", lw=1.1, label=name)
        axes[1].plot(group["hour"], group["heat"], marker="o", lw=1.1, label=name)
        axes[2].plot(group["hour"], 100.0 * group["guard_violation"], marker="o", lw=1.1, label=name)
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
    response_path = FIG_DIR / "hourly_bias_response.png"
    fig.savefig(response_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return terms_path, response_path


def _save_training_plots(losses: pd.DataFrame, online: pd.DataFrame) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    paths: list[Path] = []
    loss_summary = pd.DataFrame()
    reward_summary = pd.DataFrame()
    if not losses.empty:
        cols = [
            c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"]
            if c in losses.columns
        ]
        loss_summary = losses.groupby(["controller", "episode"], as_index=False)[cols].agg(["mean", "max"])
        loss_summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in loss_summary.columns.to_flat_index()
        ]
        fig, axes = plt.subplots(len(cols), 1, figsize=(13, 3.0 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            for name, group in loss_summary.groupby("controller", sort=False):
                ax.plot(group["episode"], group[f"{col}_mean"], lw=1.1, marker="o", label=f"{name} mean")
                ax.plot(group["episode"], group[f"{col}_max"], lw=0.8, ls="--", marker="o", label=f"{name} max")
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        axes[-1].set_xlabel("Training episode")
        fig.suptitle("SAC losses during reward training")
        fig.tight_layout()
        path = FIG_DIR / "sac_training_losses.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    if not online.empty:
        reward_summary = (
            online.groupby(["controller", "episode", "phase"], as_index=False)
            .agg(reward_mean=("reward", "mean"), reward_sum=("reward", "sum"), reward_min=("reward", "min"))
            .sort_values(["controller", "episode", "phase"])
        )
        per_building = online.groupby(["controller", "building", "episode"], as_index=False)["reward"].sum()
        ep = (
            per_building.groupby(["controller", "episode"], as_index=False)
            .agg(reward_sum_mean=("reward", "mean"), reward_sum_max=("reward", "max"), reward_sum_min=("reward", "min"))
        )
        fig, ax = plt.subplots(figsize=(13, 5))
        for name, group in ep.groupby("controller", sort=False):
            ax.plot(group["episode"], group["reward_sum_mean"], marker="o", lw=1.2, label=f"{name} mean")
            ax.fill_between(group["episode"], group["reward_sum_min"], group["reward_sum_max"], alpha=0.12)
        ax.set_title("Online training reward by episode")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode reward")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "online_training_rewards.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths, loss_summary, reward_summary


def _flag_values(data: pd.Series, index: pd.Index | None = None) -> pd.Series:
    if data is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(np.zeros(len(index), dtype=float), index=index)
    if data.dtype == object:
        return data.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    return pd.to_numeric(data, errors="coerce").fillna(0.0).astype(float)


def _save_reward_zoom_plots(
    online: pd.DataFrame,
    building_diag: pd.DataFrame,
    new_controller: str,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    paths: list[Path] = []
    if online.empty:
        return paths, pd.DataFrame(), pd.DataFrame()

    data = online.copy()
    phase = data["phase"].astype(str).str.lower() if "phase" in data.columns else pd.Series("", index=data.index)
    data = data[phase.str.contains("sac")].copy()
    if data.empty:
        return paths, pd.DataFrame(), pd.DataFrame()

    raw_col = "raw_reward" if "raw_reward" in data.columns else "reward"
    data["reward"] = pd.to_numeric(data["reward"], errors="coerce")
    data[raw_col] = pd.to_numeric(data[raw_col], errors="coerce")
    data[raw_col] = data[raw_col].fillna(data["reward"])
    data["action"] = pd.to_numeric(data.get("action", np.nan), errors="coerce")
    data["reset_penalty_flag"] = _flag_values(data.get("reset_penalty_applied"), data.index)
    data["temperature_reset_flag"] = _flag_values(data.get("temperature_reset_applied"), data.index)

    def q10(series: pd.Series) -> float:
        return float(series.quantile(0.10))

    def q90(series: pd.Series) -> float:
        return float(series.quantile(0.90))

    reward_summary = (
        data.groupby(["controller", "episode"], as_index=False)
        .agg(
            reward_mean=("reward", "mean"),
            raw_reward_mean=(raw_col, "mean"),
            reward_median=("reward", "median"),
            raw_reward_median=(raw_col, "median"),
            reward_p10=("reward", q10),
            reward_p90=("reward", q90),
            raw_reward_p10=(raw_col, q10),
            raw_reward_p90=(raw_col, q90),
            reset_penalty_frac=("reset_penalty_flag", "mean"),
            temperature_reset_frac=("temperature_reset_flag", "mean"),
            action_mean=("action", "mean"),
        )
        .sort_values(["controller", "episode"])
    )
    reward_summary.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_training_reward_mean_by_episode.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in reward_summary.groupby("controller", sort=False):
        ax.plot(
            group["episode"].to_numpy(dtype=float),
            group["raw_reward_mean"].to_numpy(dtype=float),
            marker="o",
            lw=1.2,
            label=name,
        )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Online SAC training reward by episode - per-step mean")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean raw reward per step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "online_training_rewards_mean_zoom.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    new_summary = reward_summary[reward_summary["controller"] == new_controller].copy()
    if not new_summary.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        x = new_summary["episode"].to_numpy(dtype=float)
        ax.plot(x, new_summary["raw_reward_mean"].to_numpy(dtype=float), marker="o", lw=1.4, label="Raw reward before reset penalty")
        ax.plot(x, new_summary["reward_mean"].to_numpy(dtype=float), marker="o", lw=1.4, label="Learning reward after reset penalty")
        ax.fill_between(
            x,
            new_summary["raw_reward_p10"].to_numpy(dtype=float),
            new_summary["raw_reward_p90"].to_numpy(dtype=float),
            alpha=0.12,
            label="Raw reward p10-p90",
        )
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("Penalty run reward zoom - raw reward vs replay reward")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean reward per step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "online_training_rewards_raw_vs_penalty_zoom.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    bad_mask = (
        (building_diag["discomfort_%"] >= 25.0)
        | (building_diag["hot_%"] >= 15.0)
        | (building_diag["feb_tin_max"] >= 30.0)
    )
    bad_buildings = building_diag.loc[bad_mask, "building"].tolist()
    new_data = data[(data["controller"] == new_controller) & (data["building"].isin(bad_buildings))].copy()
    catastrophe = pd.DataFrame()
    trends = pd.DataFrame()
    if not new_data.empty:
        catastrophe = (
            new_data.groupby(["building", "episode"], as_index=False)
            .agg(
                reward_mean=("reward", "mean"),
                raw_reward_mean=(raw_col, "mean"),
                reset_penalty_frac=("reset_penalty_flag", "mean"),
                temperature_reset_frac=("temperature_reset_flag", "mean"),
                action_mean=("action", "mean"),
            )
            .sort_values(["building", "episode"])
        )
        catastrophe.to_csv(
            ANALYSIS_DIR / f"{LABEL_PENALTY}_catastrophe_training_reward_by_episode.csv",
            index=False,
        )

        rows = []
        diag = building_diag.set_index("building")
        for building, group in catastrophe.groupby("building", sort=False):
            group = group.sort_values("episode")
            last3 = group.tail(3)
            slope = np.nan
            if len(last3) >= 2:
                slope = float(np.polyfit(last3["episode"].to_numpy(dtype=float), last3["raw_reward_mean"].to_numpy(dtype=float), 1)[0])
            row = {
                "building": building,
                "first_raw_reward_mean": float(group["raw_reward_mean"].iloc[0]),
                "last_raw_reward_mean": float(group["raw_reward_mean"].iloc[-1]),
                "delta_raw_reward_mean": float(group["raw_reward_mean"].iloc[-1] - group["raw_reward_mean"].iloc[0]),
                "last3_raw_reward_slope_per_episode": slope,
                "last_reset_penalty_%": 100.0 * float(group["reset_penalty_frac"].iloc[-1]),
                "last_action_mean": float(group["action_mean"].iloc[-1]),
            }
            if building in diag.index:
                row.update(
                    {
                        "feb_discomfort_%": float(diag.loc[building, "discomfort_%"]),
                        "feb_cold_%": float(diag.loc[building, "cold_%"]),
                        "feb_hot_%": float(diag.loc[building, "hot_%"]),
                        "feb_tin_max": float(diag.loc[building, "feb_tin_max"]),
                    }
                )
            rows.append(row)
        trends = pd.DataFrame(rows).sort_values("feb_discomfort_%", ascending=False)
        trends.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_catastrophe_reward_trends_penalty_only.csv", index=False)

        fig, ax = plt.subplots(figsize=(13, 6))
        for building, group in catastrophe.groupby("building", sort=False):
            ax.plot(group["episode"], group["raw_reward_mean"], marker="o", lw=1.1, label=building)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("Catastrophe buildings - raw SAC training reward")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean raw reward per step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = FIG_DIR / "catastrophe_building_raw_rewards.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(13, 6))
        for building, group in catastrophe.groupby("building", sort=False):
            ax.plot(group["episode"], 100.0 * group["reset_penalty_frac"], marker="o", lw=1.1, label=building)
        ax.set_title("Catastrophe buildings - reset penalty frequency during training")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Steps with reset penalty [%]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = FIG_DIR / "catastrophe_building_reset_penalty_fraction.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths, reward_summary, trends


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data._"
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    cols = list(show.columns)
    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _append_once(path: Path, marker: str, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if marker in str(cell.get("source", "")):
            notebook.cells[index:index + len(cells)] = cells
            nbf.write(notebook, path)
            print(f"Updated {marker} in {path}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {marker} to {path}")


def _image_cell(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.as_posix()})")


def _comfort_group_summary(building_diag: pd.DataFrame) -> pd.DataFrame:
    bad_mask = (
        (building_diag["discomfort_%"] >= 25.0)
        | (building_diag["hot_%"] >= 15.0)
        | (building_diag["feb_tin_max"] >= 30.0)
    )
    good_mask = (building_diag["discomfort_%"] < 10.0) & (building_diag["feb_tin_max"] <= 26.0)
    groups = [
        ("bad", building_diag[bad_mask]),
        ("mid", building_diag[~bad_mask & ~good_mask]),
        ("good", building_diag[good_mask]),
        ("all", building_diag),
    ]
    cols = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "feb_tin_min",
        "feb_tin_max",
        "reset_steps",
        "replay_dropped_steps",
    ]
    rows = []
    for name, group in groups:
        row = {"group": name, "n": len(group)}
        for col in cols:
            row[col] = float(group[col].mean()) if len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_reset_delta(building_diag: pd.DataFrame) -> pd.DataFrame:
    previous_path = ANALYSIS_DIR / f"{LABEL_SAFE}_building_diagnostics.csv"
    if not previous_path.exists():
        return pd.DataFrame()
    previous = _read(previous_path)
    keep = ["building", "discomfort_%", "cold_%", "hot_%", "feb_tin_max", "electricity_consumption_total"]
    merged = previous[keep].merge(building_diag[keep], on="building", suffixes=("_drop", "_penalty"))
    for col in ["discomfort_%", "cold_%", "hot_%", "feb_tin_max", "electricity_consumption_total"]:
        merged[f"delta_{col}"] = merged[f"{col}_penalty"] - merged[f"{col}_drop"]
    cols = [
        "building",
        "discomfort_%_drop",
        "discomfort_%_penalty",
        "delta_discomfort_%",
        "hot_%_drop",
        "hot_%_penalty",
        "delta_hot_%",
        "cold_%_drop",
        "cold_%_penalty",
        "delta_cold_%",
        "feb_tin_max_drop",
        "feb_tin_max_penalty",
        "delta_feb_tin_max",
        "electricity_consumption_total_drop",
        "electricity_consumption_total_penalty",
        "delta_electricity_consumption_total",
    ]
    return merged[cols].sort_values("delta_discomfort_%")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    runs = _available_runs()
    rollouts = {run["key"]: _load_rollouts(run) for run in runs}
    new_run = next(run for run in runs if run["label"] == LABEL_PENALTY)

    kpis = _controller_kpis(runs)
    kpis.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_controller_kpis.csv", index=False)

    district_frames = [_baseline_district()] + [_district_frame(run) for run in runs]
    district_kpis = _district_kpis(district_frames)
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_district_kpis.csv", index=False)
    district_load_path, district_daily_path = _save_district_plots(district_frames)

    term_frames = []
    hourly_frames = []
    for run in runs:
        terms, hourly = _term_diagnostics(run, rollouts[run["key"]])
        term_frames.append(terms)
        hourly_frames.append(hourly)
    terms_by_building = pd.concat(term_frames, ignore_index=True)
    hourly_response = pd.concat(hourly_frames, ignore_index=True)
    terms_by_building.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_term_diagnostics_by_building.csv", index=False)
    term_summary = (
        terms_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_term_diagnostics.csv", index=False)
    terms_path, hourly_path = _save_term_plots(term_summary, hourly_response)

    online_new = _online_rollouts(new_run)
    online_all = pd.concat([_online_rollouts(run) for run in runs], ignore_index=True)
    losses = _sac_losses(runs)
    training_paths, loss_summary, reward_summary = _save_training_plots(losses, online_all)
    if not loss_summary.empty:
        loss_summary.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_sac_training_loss_summary.csv", index=False)
    if not reward_summary.empty:
        reward_summary.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_online_reward_summary.csv", index=False)

    building_diag = _new_building_diagnostics(new_run, rollouts[new_run["key"]], online_new)
    building_diag.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_building_diagnostics.csv", index=False)
    group_summary = _comfort_group_summary(building_diag)
    group_summary.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_comfort_group_summary.csv", index=False)
    delta_diag = _safe_reset_delta(building_diag)
    if not delta_diag.empty:
        delta_diag.to_csv(ANALYSIS_DIR / f"{LABEL_PENALTY}_vs_safepi_reset_building_deltas.csv", index=False)
    reward_zoom_paths, reward_episode_summary, catastrophe_trends = _save_reward_zoom_plots(
        online_all,
        building_diag,
        new_run["display"],
    )

    feb_comfort_path = _save_comfort_grid(new_run, rollouts[new_run["key"]])
    jan_comfort_path = _save_january_safe_pi_grid(new_run, online_new)

    worst_hot = building_diag.sort_values("hot_%", ascending=False).head(8)
    reset_diag = building_diag[
        [
            "building",
            "jan_ep0_tin_min",
            "jan_ep0_tin_max",
            "jan_ep0_raw_next_tin_min",
            "jan_ep0_raw_next_tin_max",
            "reset_steps",
            "replay_dropped_steps",
            "safe_pi_bootstrap_steps",
        ]
    ]

    final_loss = pd.DataFrame()
    if not loss_summary.empty:
        final_loss = loss_summary.groupby("controller", as_index=False).tail(1)

    delta_best = delta_diag.head(10) if not delta_diag.empty else pd.DataFrame()
    delta_worst = delta_diag.tail(10).sort_values("delta_discomfort_%", ascending=False) if not delta_diag.empty else pd.DataFrame()

    overview = f"""## CE1 Safe-PI Reset Penalty MERLIN SAC - `{LABEL_PENALTY}`

Run label: `{LABEL_PENALTY}`. SAC-only ablation, no MPC teacher and no behavior cloning.

Pipeline: January episode 0 uses the safe PI replay-fill controller; the following January episodes use SAC reward training. During January training, runaway LSTM temperatures are still reset back near the setpoint. Safe-PI/bootstrap reset transitions are dropped as untrusted LSTM cleanup data, but **SAC-caused reset transitions are kept in replay with an additive `-100` reward penalty**. February deployment is not reset or shielded, so February KPIs are honest deployment results.

SAC settings: `action_scaling_coefficient=1.0`, automatic entropy tuning, robust replay normalization with observation/reward clipping, structured exploration, and cold comfort shield during training. Reward/cost is unchanged:

`J_k = 0.5 * [3*((Tin-(Tsp+bias))/2)^2 + 60*(cold_guard/1.5)^2 + 15*(hot_guard/2)^2] + 0.5*GridStress*P_heat/5`

`reward = -J_k`, `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and guards use true `Tsp +/- 2 C`.

Mean building KPIs:

{_markdown_table(kpis)}

District aggregate KPIs:

{_markdown_table(district_kpis)}

New run per-building diagnostics, worst hot discomfort first:

{_markdown_table(worst_hot)}

New run comfort groups (`bad = discomfort >= 25% or hot >= 15% or Tin max >= 30 C`; `good = discomfort < 10% and Tin max <= 26 C`):

{_markdown_table(group_summary)}

Best per-building deltas vs dropped-reset safe-PI run:

{_markdown_table(delta_best)}

Worst per-building deltas vs dropped-reset safe-PI run:

{_markdown_table(delta_worst)}

January safe-PI replay-fill diagnostics:

{_markdown_table(reset_diag)}

Weighted term diagnostics:

{_markdown_table(term_summary)}
"""
    if not final_loss.empty:
        overview += f"""
SAC reward-training loss summary by final episode:

{_markdown_table(final_loss)}
"""
    if not reward_summary.empty:
        overview += f"""
Online reward summary tail:

{_markdown_table(reward_summary.tail(30))}
"""
    if not reward_episode_summary.empty:
        overview += f"""
Readable per-step reward summary, final available episode by controller:

{_markdown_table(reward_episode_summary.groupby("controller", as_index=False).tail(1))}
"""
    if not catastrophe_trends.empty:
        overview += f"""
Catastrophe-building training reward trends for the new penalty run:

{_markdown_table(catastrophe_trends)}
"""

    cells = [
        nbf.v4.new_markdown_cell(overview),
        _image_cell(jan_comfort_path, "January safe-PI episode 0 comfort all buildings"),
        _image_cell(feb_comfort_path, "February comfort all buildings"),
        _image_cell(district_load_path, "District load"),
        _image_cell(district_daily_path, "Mean hourly district profile"),
        _image_cell(terms_path, "Weighted terms"),
        _image_cell(hourly_path, "Hourly response"),
        *[_image_cell(path, path.stem.replace("_", " ")) for path in training_paths],
        *[_image_cell(path, path.stem.replace("_", " ")) for path in reward_zoom_paths],
    ]

    marker = f"## CE1 Safe-PI Reset Penalty MERLIN SAC - `{LABEL_PENALTY}`"
    notebooks = [
        ANALYSIS_DIR / "08_sac_results.ipynb",
        ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
        ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
        ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
    ]
    for notebook in notebooks:
        if notebook.exists():
            _append_once(notebook, marker, cells)

    print("Done.")


if __name__ == "__main__":
    main()
