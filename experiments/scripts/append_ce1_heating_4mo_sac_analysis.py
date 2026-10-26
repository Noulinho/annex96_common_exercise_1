#!/usr/bin/env python
"""Append 4-month heating season SAC comparison to 08_sac_results.ipynb.

Experiments:
  MPC    : ce1_heating_4mo_2r2c_kalman  (April 2160-2879)
  RLMPC  : ce1_heating_4mo_2r2c_kalman  (April 2160-2879)
  SAC A  : ce1_heating_4mo_sac_no_bc_a1 (PI only, no BC)
  SAC B  : ce1_heating_4mo_sac_bc_a1    (PI + BC from MPC + SAC reward)
"""

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


MPC_LABEL = "ce1_heating_4mo_2r2c_kalman_dec"
SAC_NO_BC_LABEL = "ce1_heating_4mo_sac_no_bc_a1_dec"
SAC_BC_LABEL = "ce1_heating_4mo_sac_bc_a1_dec"

# December deployment period
DEPLOY_START = 8016
DEPLOY_END = 8759

FIG_DIR = Path("figures") / "ce1_heating_4mo_dec_multi"
ANALYSIS_DIR = Path("analysis")
RESULTS = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

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

RUNS = [
    {"key": "mpc",         "controller": "mpc",        "label": MPC_LABEL,      "display": "MPC 4-month 2R2C+Kalman (December)",                   "comfort_slug": "mpc"},
    {"key": "rlmpc",       "controller": "rlmpc",       "label": MPC_LABEL,      "display": "GNU-RL/RLMPC 4-month 2R2C+Kalman (December)",          "comfort_slug": "rlmpc"},
    {"key": "sac_no_bc",   "controller": "sac_merlin",  "label": SAC_NO_BC_LABEL,"display": "SAC 4-month PI-only (no BC, December)",                "comfort_slug": "sac_no_bc"},
    {"key": "sac_bc",      "controller": "sac_merlin",  "label": SAC_BC_LABEL,   "display": "SAC 4-month BC+reward from MPC (December)",            "comfort_slug": "sac_bc"},
]

KPI_CASES = [(r["controller"], r["label"], r["display"]) for r in RUNS]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _building_summary_path(controller: str, building_id: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / "summary.csv"


def _case_summary(controller: str, case: str) -> pd.DataFrame:
    per_building = []
    for b in BUILDINGS:
        path = _building_summary_path(controller, b["result_id"])
        if not path.exists():
            continue
        summary = _read(path)
        subset = summary[summary["case"] == case].copy()
        if not subset.empty:
            per_building.append(subset)
    if per_building:
        return pd.concat(per_building, ignore_index=True)
    agg_path = RESULTS / "summaries" / controller / "no_occupant_summary_all_buildings.csv"
    if agg_path.exists():
        agg = _read(agg_path)
        return agg[agg["case"] == case].copy()
    return pd.DataFrame()


def _rollout_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label / "rollout.csv"


def _load_rollouts(run: dict) -> dict[str, pd.DataFrame]:
    return {
        b["result_id"]: _read(_rollout_path(run["controller"], b["result_id"], run["label"]))
        for b in BUILDINGS
    }


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for col in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if col in data.columns:
            return pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
    raise ValueError("No setpoint column found.")


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
        time_bias_setpoint_profile(simulation_start=DEPLOY_START, simulation_end=DEPLOY_START + n - 1,
                                   amplitude=WEIGHTS["bias_amplitude"]), dtype=float)
    stress = np.asarray(load_simple_v4_price_profile(simulation_start=DEPLOY_START, exp_beta=2.0), dtype=float)
    if stress.size < n:
        stress = np.resize(stress, n)
    return bias[:n], stress[:n]


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def _controller_kpis() -> pd.DataFrame:
    frames = []
    for controller, case, display in KPI_CASES:
        subset = _case_summary(controller, case)
        if subset.empty:
            continue
        subset = subset.copy()
        subset["controller"] = display
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["discomfort_%"] = 100.0 * pd.to_numeric(data["discomfort_proportion"], errors="coerce")
    data["cold_%"] = 100.0 * pd.to_numeric(data["discomfort_cold_proportion"], errors="coerce")
    data["hot_%"] = 100.0 * pd.to_numeric(data["discomfort_hot_proportion"], errors="coerce")
    cols = ["electricity_consumption_total", "discomfort_%", "cold_%", "hot_%", "kelvin_hours", "daily_peak_average"]
    if "ramping_average" in data.columns:
        cols.append("ramping_average")
    out = data.groupby("controller", as_index=False)[cols].mean()
    order = {display: i for i, (_, _, display) in enumerate(KPI_CASES)}
    out["_order"] = out["controller"].map(order)
    return out.sort_values("_order").drop(columns="_order")


def _baseline_district() -> pd.DataFrame:
    # Use December-period baseline (generated by 09_run_citylearn_baseline.py with sim 8016-8759)
    path = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "december" / "district_rollout.csv"
    if not path.exists():
        # Fallback: full-year baseline, slice to April
        data = _read(RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv")
        sub = data  # use whatever is available
    else:
        sub = _read(path)
    n = len(sub)
    target_col = "district_target" if "district_target" in sub.columns else sub.columns[-1]
    return pd.DataFrame({
        "time_step": np.arange(n, dtype=int),
        "load": sub["net_electricity_consumption"].to_numpy(dtype=float),
        "target": sub[target_col].to_numpy(dtype=float) if target_col in sub.columns else np.zeros(n),
        "controller": "CityLearn baseline (April)",
    })


def _district_frame(run: dict) -> pd.DataFrame:
    frame, missing = aggregate_controller_rollouts(controller=run["controller"], case=run["label"], buildings=BUILDINGS)
    if missing:
        raise FileNotFoundError(f"Missing {run['display']} rollouts: {missing[:3]}")
    baseline = _baseline_district()
    n = min(len(frame), len(baseline))
    return pd.DataFrame({
        "time_step": frame["time_step"].to_numpy(dtype=int)[:n],
        "load": frame["net_electricity_consumption"].to_numpy(dtype=float)[:n],
        "target": baseline["target"].to_numpy(dtype=float)[:n],
        "controller": run["display"],
    })


def _district_kpis(district_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    base_load = district_frames[0]["load"].to_numpy(dtype=float)
    base_total = max(float(np.sum(base_load)), 1e-9)
    base_peak = max(float(np.max(base_load)), 1e-9)
    base_ramp = max(float(np.sum(np.abs(np.diff(base_load)))), 1e-9)
    for frame in district_frames:
        load = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        n = min(len(load), len(target))
        load, target = load[:n], target[:n]
        hourly = pd.DataFrame({"load": load, "hour": np.arange(n) % 24}).groupby("hour")["load"].mean()
        rows.append({
            "controller": frame["controller"].iloc[0],
            "total_kwh": float(np.sum(load)),
            "total_vs_citylearn": float(np.sum(load)) / base_total,
            "peak_kwh": float(np.max(load)),
            "peak_vs_citylearn": float(np.max(load)) / base_peak,
            "daily_profile_peak_kwh": float(hourly.max()),
            "daily_profile_peak_hour": int(hourly.idxmax()),
            "ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))),
            "ramping_vs_citylearn": float(np.sum(np.abs(np.diff(load)))) / base_ramp,
            "target_rmse_kwh": float(np.sqrt(np.mean((load - target) ** 2))),
            "above_target_kwh": float(np.sum(np.maximum(load - target, 0.0))),
        })
    return pd.DataFrame(rows)


def _term_diagnostics(run: dict, rollouts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    display = run["display"]
    rows, hourly_rows = [], []
    for b in BUILDINGS:
        data = rollouts[b["result_id"]]
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
        track = WEIGHTS["comfort_weight"] * WEIGHTS["comfort_track_weight"] * ((tin - tref) / WEIGHTS["comfort_track_scale"]) ** 2
        cold = WEIGHTS["comfort_weight"] * WEIGHTS["comfort_cold_weight"] * (cold_guard / WEIGHTS["comfort_cold_scale"]) ** 2
        hot = WEIGHTS["comfort_weight"] * WEIGHTS["comfort_hot_weight"] * (hot_guard / WEIGHTS["comfort_hot_scale"]) ** 2
        shift = WEIGHTS["shift_weight"] * stress * heat / WEIGHTS["power_scale"]
        stage = track + cold + hot + shift
        guard_violation = (cold_guard > 0.0) | (hot_guard > 0.0)
        for term, values in [("tracking_target", track), ("guard_cold", cold), ("guard_hot", hot),
                              ("time_weighted_heat", shift), ("stage_cost", stage)]:
            rows.append({"controller": display, "building": b["display"], "term": term,
                         "mean": float(np.nanmean(values)), "max": float(np.nanmax(values))})
        rows.append({"controller": display, "building": b["display"], "term": "mae_to_true_tsp_c",
                     "mean": float(np.nanmean(np.abs(tin - tsp))), "max": float(np.nanmax(np.abs(tin - tsp)))})
        rows.append({"controller": display, "building": b["display"], "term": "mae_to_biased_tref_c",
                     "mean": float(np.nanmean(np.abs(tin - tref))), "max": float(np.nanmax(np.abs(tin - tref)))})
        rows.append({"controller": display, "building": b["display"], "term": "guard_violation_%",
                     "mean": float(100.0 * np.nanmean(guard_violation)), "max": float(100.0 * np.nanmax(guard_violation))})
        hourly_rows.append(pd.DataFrame({
            "controller": display, "hour": time_step % 24,
            "tin_minus_tsp": tin - tsp, "bias": bias, "heat": heat,
            "guard_violation": guard_violation.astype(float),
        }))
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def _case_dir(run: dict, building_id: str) -> Path:
    return RESULTS / "raw" / run["controller"] / building_id / "no_occupant" / run["label"]


def _load_online_rewards(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for b in BUILDINGS:
            path = _case_dir(run, b["result_id"]) / "online_training_rollout.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = b["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_sac_losses(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for b in BUILDINGS:
            path = _case_dir(run, b["result_id"]) / "sac_training_losses.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = b["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_bc_losses(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for b in BUILDINGS:
            path = _case_dir(run, b["result_id"]) / "bc_training_losses.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = b["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Plots (reuse same style as previous cells)
# ---------------------------------------------------------------------------

def _make_comfort_grid(run: dict, rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, b in zip(axes, BUILDINGS):
        data = rollouts[b["result_id"]]
        ts = _time_step(data); tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data); n = min(len(ts), len(tin), len(tsp))
        ts, tin, tsp = ts[:n], tin[:n], tsp[:n]
        bias, _ = _bias_and_stress(n)
        ax.plot(ts, tin, lw=0.75, color="steelblue", label="T_in")
        ax.plot(ts, tsp, lw=0.65, ls="--", color="black", label="Tsp")
        ax.plot(ts, tsp + bias, lw=0.65, ls=":", color="tab:orange", label="Tsp+bias")
        ax.fill_between(ts, tsp - WEIGHTS["guard_band"], tsp + WEIGHTS["guard_band"], alpha=0.10, color="green")
        ax.set_title(b["display"], fontsize=8)
        ax.set_ylim(14, 28)
        ax.grid(alpha=0.2)
    for ax in axes[len(BUILDINGS):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} — April deployment (time-biased guard)", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / f"{run['comfort_slug']}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_district_plots(district_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(district_frames, ignore_index=True)
    target = district_frames[0][["time_step", "target"]]
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=name)
    ax.plot(target["time_step"], target["target"], color="black", lw=1.0, ls="--", label="CE1 district target")
    ax.set_title("4-month heating season — April deployment district load")
    ax.set_xlabel("April time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    p = FIG_DIR / "district_load_timeseries_multi.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = target.assign(hour=target["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.4, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.1, ls="--", label="CE1 district target")
    ax.set_title("Mean hourly district load — April"); ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean kWh")
    ax.set_xticks(range(24)); ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    dp = FIG_DIR / "district_mean_daily_profile_multi.png"
    fig.savefig(dp, dpi=140, bbox_inches="tight"); plt.close(fig)
    return p, dp


def _make_term_plots(term_summary: pd.DataFrame, hourly: pd.DataFrame) -> tuple[Path, Path]:
    plot_terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]
    plot_data = term_summary[term_summary["term"].isin(plot_terms)].copy()
    pivot = plot_data.pivot(index="term", columns="controller", values="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.loc[[t for t in plot_terms if t in pivot.index]].plot(kind="bar", ax=ax)
    ax.set_title("Mean weighted stage-cost terms by controller"); ax.set_ylabel("Mean weighted term"); ax.grid(axis="y", alpha=0.25); fig.tight_layout()
    tp = FIG_DIR / "weighted_terms_by_controller.png"
    fig.savefig(tp, dpi=140, bbox_inches="tight"); plt.close(fig)

    hourly_mean = hourly.groupby(["controller", "hour"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for name, group in hourly_mean.groupby("controller", sort=False):
        axes[0].plot(group["hour"], group["tin_minus_tsp"], marker="o", lw=1.2, label=name)
        axes[1].plot(group["hour"], group["heat"], marker="o", lw=1.2, label=name)
        axes[2].plot(group["hour"], 100.0 * group["guard_violation"], marker="o", lw=1.2, label=name)
    axes[0].axhline(0, color="black", lw=0.8); axes[0].set_ylabel("Tin - Tsp [C]")
    axes[1].set_ylabel("Heating electricity [kWh]"); axes[2].set_ylabel("Guard violation [%]"); axes[2].set_xlabel("Hour of day")
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("Hourly response — April deployment"); fig.tight_layout()
    rp = FIG_DIR / "hourly_bias_response_multi.png"
    fig.savefig(rp, dpi=140, bbox_inches="tight"); plt.close(fig)
    return tp, rp


def _make_training_plots(bc_losses, sac_losses, online_rewards):
    paths = []; sac_summary = pd.DataFrame(); reward_summary = pd.DataFrame(); bc_summary = pd.DataFrame()

    if not bc_losses.empty:
        bc_summary = bc_losses.groupby(["controller", "epoch"], as_index=False).agg(
            bc_train_loss_mean=("bc_train_loss", "mean"), bc_val_loss_mean=("bc_val_loss", "mean"),
            bc_val_mae_mean=("bc_val_mae", "mean"), bc_val_abs_error_max=("bc_val_abs_error_max", "max"))
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, group in bc_summary.groupby("controller", sort=False):
            ax.plot(group["epoch"], group["bc_train_loss_mean"], lw=1.4, label=f"{name} train")
            ax.plot(group["epoch"], group["bc_val_loss_mean"], lw=1.4, ls="--", label=f"{name} val")
        ax.set_title("BC loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
        p = FIG_DIR / "bc_training_losses_multi.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); paths.append(p)

    if not sac_losses.empty:
        available = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"] if c in sac_losses.columns]
        sac_summary = sac_losses.groupby(["controller", "episode"], as_index=False)[available].agg(["mean", "max"])
        sac_summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in sac_summary.columns.to_flat_index()]
        fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.3 * len(available)), sharex=True)
        if len(available) == 1: axes = [axes]
        for ax, col in zip(axes, available):
            for name, group in sac_summary.groupby("controller", sort=False):
                ax.plot(group["episode"], group[f"{col}_mean"], marker="o", lw=1.2, label=f"{name} mean")
                ax.plot(group["episode"], group[f"{col}_max"], marker="o", lw=0.9, ls="--", label=f"{name} max")
            ax.set_ylabel(col); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        axes[-1].set_xlabel("Episode"); fig.suptitle("SAC losses"); fig.tight_layout()
        p = FIG_DIR / "sac_training_losses_multi.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); paths.append(p)

    if not online_rewards.empty:
        reward_summary = (online_rewards.groupby(["controller", "episode", "phase"], as_index=False)
                          .agg(reward_mean=("reward", "mean"), reward_sum=("reward", "sum"), reward_min=("reward", "min"))
                          .sort_values(["controller", "episode", "phase"]))
        ep_reward = (online_rewards.groupby(["controller", "building", "episode"], as_index=False)["reward"].sum()
                     .groupby(["controller", "episode"], as_index=False)
                     .agg(reward_sum_mean=("reward", "mean"), reward_sum_max=("reward", "max"), reward_sum_min=("reward", "min")))
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, group in ep_reward.groupby("controller", sort=False):
            ax.plot(group["episode"], group["reward_sum_mean"], marker="o", lw=1.4, label=f"{name} mean")
            ax.fill_between(group["episode"], group["reward_sum_min"], group["reward_sum_max"], alpha=0.12)
        ax.set_title("Online training reward by episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Episode reward"); ax.grid(alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
        p = FIG_DIR / "online_training_rewards_multi.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths, bc_summary, sac_summary, reward_summary


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in (row[c] for c in cols)]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def _append_once(path: Path, marker: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if marker in str(cell.get("source", "")):
            nb.cells[i:i + len(cells)] = cells
            nbf.write(nb, path)
            print(f"Updated existing '{marker}' in {path}")
            return
    nb.cells.extend(cells)
    nbf.write(nb, path)
    print(f"Appended '{marker}' to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rollouts = {}
    for run in RUNS:
        try:
            rollouts[run["key"]] = _load_rollouts(run)
        except FileNotFoundError as e:
            print(f"WARNING: {run['display']} rollouts missing — {e}")
            rollouts[run["key"]] = {}

    kpis = _controller_kpis()
    kpis.to_csv(ANALYSIS_DIR / "ce1_heating_4mo_dec_controller_kpis.csv", index=False)

    district_frames = [_baseline_district()]
    for run in RUNS:
        if rollouts[run["key"]]:
            try:
                district_frames.append(_district_frame(run))
            except FileNotFoundError as e:
                print(f"WARNING: {e}")
    district_kpis = _district_kpis(district_frames)
    district_kpis.to_csv(ANALYSIS_DIR / "ce1_heating_4mo_dec_district_kpis.csv", index=False)
    district_load_path, district_daily_path = _make_district_plots(district_frames)

    comfort_paths = {}
    for run in RUNS:
        if rollouts[run["key"]]:
            comfort_paths[run["key"]] = _make_comfort_grid(run, rollouts[run["key"]])

    term_frames, hourly_frames = [], []
    for run in RUNS:
        if rollouts[run["key"]]:
            t, h = _term_diagnostics(run, rollouts[run["key"]])
            term_frames.append(t); hourly_frames.append(h)
    if term_frames:
        terms_by_building = pd.concat(term_frames, ignore_index=True)
        hourly_response = pd.concat(hourly_frames, ignore_index=True)
        terms_by_building.to_csv(ANALYSIS_DIR / "ce1_heating_4mo_dec_term_diagnostics_by_building.csv", index=False)
        term_summary = (terms_by_building.groupby(["controller", "term"], as_index=False)
                        .agg(mean=("mean", "mean"), max=("max", "max")).sort_values(["controller", "term"]))
        term_summary.to_csv(ANALYSIS_DIR / "ce1_heating_4mo_dec_term_diagnostics.csv", index=False)
        terms_path, response_path = _make_term_plots(term_summary, hourly_response)
    else:
        term_summary = pd.DataFrame()
        terms_path = response_path = None

    sac_runs = [r for r in RUNS if r["controller"] == "sac_merlin"]
    bc_losses = _load_bc_losses(sac_runs)
    sac_losses = _load_sac_losses(sac_runs)
    online_rewards = _load_online_rewards(sac_runs)
    training_paths, bc_summary, sac_loss_summary, reward_summary = _make_training_plots(bc_losses, sac_losses, online_rewards)

    MARKER = f"## CE1 4-Month Heating Season SAC - December Deployment - `{MPC_LABEL}`"
    overview_md = f"""{MARKER}

Training: **PI controller on Jan+Feb+Mar** (steps 0–2159) + **November** (steps 7296–8015) for RC model fitting. SAC reward-learning on Jan+Feb+Mar period.
Deployment: **December** (steps 8016–8759, 744 steps).

RC models fitted on Jan+Feb+Mar+Nov combined PI data (2878 transitions/building):
`2r2c_{{short_name}}_jan_feb_mar_nov.json` — 2R2C+Kalman observer at deployment.

Reward (same time-bias guard as previous runs):
`J_k = 0.5 * [3*((Tin-(Tsp+bias))/2)^2 + 60*(cold_guard/1.5)^2 + 15*(hot_guard/2)^2] + 0.5*GridStress*P_heat/5`

Controllers:
- **MPC** `{MPC_LABEL}`: 2R2C+Kalman MPC, April deployment
- **GNU-RL/RLMPC** `{MPC_LABEL}`: offline imitation from 4-month PI, PPO fine-tuning on April
- **SAC no-BC** `{SAC_NO_BC_LABEL}`: 1 PI episode + 10 SAC reward episodes, no BC
- **SAC BC+reward** `{SAC_BC_LABEL}`: 1 PI episode + BC from 2R2C+Kalman MPC teacher + 10 SAC reward episodes

Mean building KPIs:

{_markdown_table(kpis)}

District aggregate KPIs:

{_markdown_table(district_kpis)}

Weighted term diagnostics:

{_markdown_table(term_summary) if not term_summary.empty else '_No data._'}
"""
    if not bc_summary.empty:
        overview_md += f"\nBehavior-cloning loss summary:\n\n{_markdown_table(bc_summary.groupby('controller', as_index=False).tail(1))}\n"
    if not sac_loss_summary.empty:
        overview_md += f"\nSAC reward-training loss summary by final episode:\n\n{_markdown_table(sac_loss_summary.groupby('controller', as_index=False).tail(1))}\n"
    if not reward_summary.empty:
        overview_md += f"\nOnline reward summary:\n\n{_markdown_table(reward_summary.tail(40))}\n"

    cells = [nbf.v4.new_markdown_cell(overview_md)]
    for run in RUNS:
        if run["key"] in comfort_paths:
            cells.append(nbf.v4.new_markdown_cell(f"![{run['display']} comfort all buildings](../{comfort_paths[run['key']].as_posix()})"))
    cells.append(nbf.v4.new_markdown_cell(f"![District load](../{district_load_path.as_posix()})"))
    cells.append(nbf.v4.new_markdown_cell(f"![Mean hourly district profile](../{district_daily_path.as_posix()})"))
    if terms_path:
        cells.append(nbf.v4.new_markdown_cell(f"![Weighted stage-cost terms](../{terms_path.as_posix()})"))
        cells.append(nbf.v4.new_markdown_cell(f"![Hourly bias response](../{response_path.as_posix()})"))
    for p in training_paths:
        cells.append(nbf.v4.new_markdown_cell(f"![Training diagnostic](../{p.as_posix()})"))

    _append_once(ANALYSIS_DIR / "08_sac_results.ipynb", MARKER, cells)
    print("Done.")


if __name__ == "__main__":
    main()
