#!/usr/bin/env python
"""Append safe-PI + exact-MPC-cost SAC diagnostics to analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_analysis import aggregate_controller_rollouts
from src.district_target import load_ce1_district_target_profiles
from src.rewards import time_bias_setpoint_profile


MPC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman"
SAFE_RESET_LABEL = "ce1_time_bias_guard_1c_1r1c_sac_safepi_reset_auto_a1"
PENALTY_LABEL = "ce1_time_bias_guard_1c_1r1c_sac_safepi_penalty100_auto_a1"
OLD_EXACT_SAC_LABEL = "ce1_safe_pi_mpc_cost_a05"
SAC_LABEL = "ce1_mpc_cost_safepi_drop_es_a05"

FIG_DIR = Path("figures") / SAC_LABEL
ANALYSIS_DIR = Path("analysis")
RESULTS = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)
FEB_START = 744
FEB_END = 1415

RUNS = [
    {
        "key": "mpc_2r2c",
        "controller": "mpc",
        "label": MPC_LABEL,
        "display": "MPC time-bias guard 2R2C+Kalman",
        "slug": "mpc_2r2c_kalman",
    },
    {
        "key": "sac_safe_reset",
        "controller": "sac_merlin",
        "label": SAFE_RESET_LABEL,
        "display": "SAC safe-PI reset + auto-alpha (actor 0-1)",
        "slug": "sac_safepi_reset_auto_a1",
    },
    {
        "key": "sac_penalty100",
        "controller": "sac_merlin",
        "label": PENALTY_LABEL,
        "display": "SAC safe-PI reset + additive reset penalty -100",
        "slug": "sac_safepi_penalty100",
    },
    {
        "key": "sac_mpc_cost_a05",
        "controller": "sac_merlin",
        "label": OLD_EXACT_SAC_LABEL,
        "display": "SAC safe-PI + exact MPC cost (cap 0.5, 20 ep)",
        "slug": "sac_safe_pi_mpc_cost_a05",
    },
    {
        "key": "sac_mpc_cost_safepi_drop_es_a05",
        "controller": "sac_merlin",
        "label": SAC_LABEL,
        "display": "SAC exact MPC cost + safe-PI reset/drop early-stop (cap 0.5)",
        "slug": "sac_mpc_cost_safepi_drop_es_a05",
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
    "bias_amplitude": 1.0,
    "power_scale": 5.0,
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


def _metrics_json_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / f"{label}_metrics.json"


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
        rows = []
        for building in BUILDINGS:
            path = _metrics_json_path(run["controller"], building["result_id"], run["label"])
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            row["building"] = building["display"]
            row["building_id"] = building["result_id"]
            row["building_index"] = building["index"]
            row["building_name"] = building["name"]
            row["case"] = run["label"]
            rows.append(row)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        out["controller"] = run["display"]
        return out
    out = pd.concat(frames, ignore_index=True)
    out["controller"] = run["display"]
    return out


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("No setpoint column found.")


def _time_step(data: pd.DataFrame) -> np.ndarray:
    if "time_step" in data.columns:
        return pd.to_numeric(data["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(data), dtype=int)


def _load(data: pd.DataFrame) -> np.ndarray:
    if "net_electricity_consumption" in data.columns:
        return pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "electricity_consumption" in data.columns:
        return pd.to_numeric(data["electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _heating(data: pd.DataFrame) -> np.ndarray:
    if "heating_electricity_consumption" in data.columns:
        values = pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0)
    else:
        values = pd.to_numeric(data["action"], errors="coerce").fillna(0.0)
    return np.maximum(values.to_numpy(dtype=float), 0.0)


def _bool_float(series: pd.Series | None, index: pd.Index | None = None) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(np.zeros(len(index), dtype=float), index=index)
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _bias_profile(n: int, start: int = FEB_START) -> np.ndarray:
    return np.asarray(
        time_bias_setpoint_profile(
            simulation_start=start,
            simulation_end=start + n - 1,
            amplitude=WEIGHTS["bias_amplitude"],
        ),
        dtype=float,
    )[:n]


def _ce1_stress_profile(n: int) -> np.ndarray:
    profiles = load_ce1_district_target_profiles(
        building_name=str(BUILDINGS[0]["name"]),
        simulation_start=FEB_START,
        simulation_end=FEB_END,
    )
    stress = np.asarray(profiles.district_stress_profile(), dtype=float)
    if stress.size < n:
        stress = np.resize(stress, n)
    return stress[:n]


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
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
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
    stress_cache: dict[int, np.ndarray] = {}
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data)
        heat = _heating(data)
        time_step = _time_step(data)
        n = min(len(tin), len(tsp), len(heat), len(time_step))
        tin, tsp, heat, time_step = tin[:n], tsp[:n], heat[:n], time_step[:n]
        bias = _bias_profile(n)
        if n not in stress_cache:
            stress_cache[n] = _ce1_stress_profile(n)
        stress = stress_cache[n]
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
                    "building_id": building["result_id"],
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
        if run["controller"] != "sac_merlin":
            continue
        for building in BUILDINGS:
            path = _case_dir(run, building["result_id"]) / "sac_training_losses.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["controller"] = run["display"]
            data["building"] = building["display"]
            data["building_id"] = building["result_id"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _building_diagnostics(run: dict, rollouts: dict[str, pd.DataFrame], online: pd.DataFrame) -> pd.DataFrame:
    summary = _case_summary(run)
    rows = []
    for building in BUILDINGS:
        b_id = building["result_id"]
        rollout = rollouts[b_id]
        online_b = online[online["building_id"] == b_id] if not online.empty else pd.DataFrame()
        jan0 = online_b[online_b["episode"] == 0] if not online_b.empty else pd.DataFrame()
        summary_b = summary[summary["building_id"] == b_id] if not summary.empty else pd.DataFrame()
        tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
        row = {
            "building": building["display"],
            "building_id": b_id,
            "electricity_consumption_total": np.nan,
            "discomfort_%": np.nan,
            "cold_%": np.nan,
            "hot_%": np.nan,
            "kelvin_hours": np.nan,
            "daily_peak_average": np.nan,
            "feb_tin_min": float(tin.min()),
            "feb_tin_max": float(tin.max()),
            "jan_ep0_tin_min": float(pd.to_numeric(jan0["tin_before_action"], errors="coerce").min()) if "tin_before_action" in jan0 else np.nan,
            "jan_ep0_tin_max": float(pd.to_numeric(jan0["tin_before_action"], errors="coerce").max()) if "tin_before_action" in jan0 else np.nan,
            "safe_pi_applied_steps": float(pd.to_numeric(online_b.get("safe_pi_applied", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not online_b.empty else 0.0,
            "temperature_reset_steps": float(_bool_float(online_b.get("temperature_reset_applied"), online_b.index).sum()) if not online_b.empty else 0.0,
            "training_reward_last_ep": np.nan,
            "alpha_final": np.nan,
        }
        if not online_b.empty:
            sac = online_b[online_b["phase"].astype(str).str.contains("sac", case=False, na=False)].copy()
            if not sac.empty:
                last_ep = int(pd.to_numeric(sac["episode"], errors="coerce").max())
                row["training_reward_last_ep"] = float(
                    pd.to_numeric(sac.loc[sac["episode"] == last_ep, "reward"], errors="coerce").mean()
                )
        if not summary_b.empty:
            row["electricity_consumption_total"] = float(summary_b["electricity_consumption_total"].iloc[-1])
            row["discomfort_%"] = 100.0 * float(summary_b["discomfort_proportion"].iloc[-1])
            row["cold_%"] = 100.0 * float(summary_b["discomfort_cold_proportion"].iloc[-1])
            row["hot_%"] = 100.0 * float(summary_b["discomfort_hot_proportion"].iloc[-1])
            row["kelvin_hours"] = float(summary_b["kelvin_hours"].iloc[-1])
            row["daily_peak_average"] = float(summary_b["daily_peak_average"].iloc[-1])
            if "alpha_final" in summary_b.columns:
                row["alpha_final"] = float(pd.to_numeric(summary_b["alpha_final"], errors="coerce").iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows)


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
    cols = ["electricity_consumption_total", "discomfort_%", "cold_%", "hot_%", "feb_tin_min", "feb_tin_max", "training_reward_last_ep"]
    rows = []
    for name, group in groups:
        row = {"group": name, "n": len(group)}
        for col in cols:
            row[col] = float(group[col].mean()) if len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _early_stop_summary(run: dict) -> pd.DataFrame:
    summary = _case_summary(run)
    if summary.empty:
        return pd.DataFrame()
    cols = [
        "building",
        "training_episodes",
        "training_episodes_completed",
        "early_stop_triggered",
        "early_stop_episode",
        "early_stop_last_slope",
        "training_temperature_reset_steps",
        "training_temperature_reset_fraction",
        "safe_pi_random_prob",
        "alpha_final",
    ]
    present = [col for col in cols if col in summary.columns]
    out = summary[present].copy()
    if "building" in out.columns:
        out = out.sort_values("building")
    return out


def _make_comfort_grid(run: dict, rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        t = _time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data)
        n = min(len(t), len(tin), len(tsp))
        t, tin, tsp = t[:n], tin[:n], tsp[:n]
        bias = _bias_profile(n)
        ax.plot(t, tin, lw=0.8, label="Indoor temperature")
        ax.plot(t, tsp, "--", lw=0.7, color="black", label="Setpoint")
        ax.plot(t, tsp + bias, lw=0.7, color="tab:orange", label="Effective setpoint")
        ax.fill_between(t, tsp - WEIGHTS["guard_band"], tsp + WEIGHTS["guard_band"], color="gray", alpha=0.16, label="+/-2 C guard")
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(BUILDINGS):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | February comfort", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / f"{run['slug']}_february_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_january_safe_pi_grid(run: dict, online: pd.DataFrame) -> Path | None:
    if online.empty:
        return None
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = online[(online["building_id"] == building["result_id"]) & (online["episode"] == 0)].copy()
        if data.empty:
            ax.axis("off")
            continue
        t = pd.to_numeric(data["absolute_time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
        tin = pd.to_numeric(data["tin_before_action"], errors="coerce").to_numpy(dtype=float)
        tsp = pd.to_numeric(data["tsp_before_action"], errors="coerce").to_numpy(dtype=float)
        n = min(len(t), len(tin), len(tsp))
        ax.plot(t[:n], tin[:n], lw=0.9, label="Indoor temperature")
        ax.plot(t[:n], tsp[:n], "--", lw=0.7, color="black", label="Setpoint")
        ax.fill_between(t[:n], tsp[:n] - WEIGHTS["guard_band"], tsp[:n] + WEIGHTS["guard_band"], color="gray", alpha=0.16, label="+/-2 C guard")
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | January episode 0 safe-PI comfort", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "january_safe_pi_episode0_comfort_all25.png"
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
    ax.set_title("District load - February")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ts_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(ts_path, dpi=140, bbox_inches="tight")
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
    return ts_path, daily_path


def _make_term_plots(term_summary: pd.DataFrame, hourly: pd.DataFrame) -> tuple[Path, Path]:
    terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]
    pivot = term_summary[term_summary["term"].isin(terms)].pivot(index="term", columns="controller", values="mean")
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.loc[[t for t in terms if t in pivot.index]].plot(kind="bar", ax=ax)
    ax.set_title("Mean weighted cost terms")
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


def _make_training_plots(losses: pd.DataFrame, online: pd.DataFrame) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
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
                ax.plot(group["episode"], group[f"{col}_mean"], marker="o", lw=1.1, label=f"{name} mean")
                ax.plot(group["episode"], group[f"{col}_max"], marker="o", lw=0.8, ls="--", label=f"{name} max")
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        axes[-1].set_xlabel("Training episode")
        fig.suptitle("SAC loss diagnostics")
        fig.tight_layout()
        path = FIG_DIR / "sac_training_losses.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not online.empty:
        data = online.copy()
        data["reward"] = pd.to_numeric(data["reward"], errors="coerce")
        raw_col = "raw_reward" if "raw_reward" in data.columns else "reward"
        data[raw_col] = pd.to_numeric(data[raw_col], errors="coerce").fillna(data["reward"])
        data["reset_penalty_flag"] = _bool_float(data.get("reset_penalty_applied"), data.index)
        data["temperature_reset_flag"] = _bool_float(data.get("temperature_reset_applied"), data.index)
        reward_summary = (
            data.groupby(["controller", "episode", "phase"], as_index=False)
            .agg(
                reward_mean=("reward", "mean"),
                raw_reward_mean=(raw_col, "mean"),
                reward_sum=("reward", "sum"),
                reward_min=("reward", "min"),
                reset_penalty_frac=("reset_penalty_flag", "mean"),
                temperature_reset_frac=("temperature_reset_flag", "mean"),
            )
            .sort_values(["controller", "episode", "phase"])
        )
        sac = data[data["phase"].astype(str).str.contains("sac", case=False, na=False)].copy()
        ep = (
            sac.groupby(["controller", "building", "episode"], as_index=False)
            .agg(raw_reward_mean=(raw_col, "mean"), reward_mean=("reward", "mean"))
            .groupby(["controller", "episode"], as_index=False)
            .agg(
                raw_reward_mean=("raw_reward_mean", "mean"),
                raw_reward_p10=("raw_reward_mean", lambda s: float(s.quantile(0.10))),
                raw_reward_p90=("raw_reward_mean", lambda s: float(s.quantile(0.90))),
                reward_mean=("reward_mean", "mean"),
            )
        )
        fig, ax = plt.subplots(figsize=(13, 5))
        for name, group in ep.groupby("controller", sort=False):
            x = group["episode"].to_numpy(dtype=float)
            y = group["raw_reward_mean"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", lw=1.2, label=name)
            ax.fill_between(
                x,
                group["raw_reward_p10"].to_numpy(dtype=float),
                group["raw_reward_p90"].to_numpy(dtype=float),
                alpha=0.10,
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
    return paths, loss_summary, reward_summary


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
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    runs = _available_runs()
    rollouts = {run["key"]: _load_rollouts(run) for run in runs}
    new_run = next(run for run in runs if run["label"] == SAC_LABEL)

    kpis = _controller_kpis(runs)
    kpis.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_controller_kpis.csv", index=False)

    district_frames = [_baseline_district()] + [_district_frame(run) for run in runs]
    district_kpis = _district_kpis(district_frames)
    district_kpis.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_district_kpis.csv", index=False)
    district_ts_path, district_daily_path = _make_district_plots(district_frames)

    term_frames = []
    hourly_frames = []
    for run in runs:
        terms, hourly = _term_diagnostics(run, rollouts[run["key"]])
        term_frames.append(terms)
        hourly_frames.append(hourly)
    terms_by_building = pd.concat(term_frames, ignore_index=True)
    hourly_response = pd.concat(hourly_frames, ignore_index=True)
    terms_by_building.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_term_diagnostics_by_building.csv", index=False)
    term_summary = (
        terms_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    term_summary.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_term_diagnostics.csv", index=False)
    terms_path, response_path = _make_term_plots(term_summary, hourly_response)

    comfort_paths = {run["key"]: _make_comfort_grid(run, rollouts[run["key"]]) for run in runs}

    online_frames = [_online_rollouts(run) for run in runs if run["controller"] == "sac_merlin"]
    online = pd.concat([f for f in online_frames if not f.empty], ignore_index=True) if online_frames else pd.DataFrame()
    online.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_online_training_rewards_raw.csv", index=False)
    losses = _sac_losses(runs)
    losses.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_sac_training_losses_raw.csv", index=False)
    training_paths, loss_summary, reward_summary = _make_training_plots(losses, online)
    loss_summary.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_sac_loss_summary.csv", index=False)
    reward_summary.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_online_reward_summary.csv", index=False)

    new_online = _online_rollouts(new_run)
    building_diag = _building_diagnostics(new_run, rollouts[new_run["key"]], new_online)
    building_diag.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_building_diagnostics.csv", index=False)
    group_diag = _comfort_group_summary(building_diag)
    group_diag.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_comfort_group_diagnostics.csv", index=False)
    early_stop = _early_stop_summary(new_run)
    early_stop.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_early_stop_summary.csv", index=False)
    jan_path = _make_january_safe_pi_grid(new_run, new_online)

    final_losses = pd.DataFrame() if loss_summary.empty else loss_summary.groupby("controller", as_index=False).tail(1)
    final_rewards = pd.DataFrame() if reward_summary.empty else (
        reward_summary[reward_summary["phase"].astype(str).str.contains("sac", case=False, na=False)]
        .groupby("controller", as_index=False)
        .tail(1)
    )

    marker = f"## CE1 Safe-PI Exact MPC-Cost SAC - `{SAC_LABEL}`"
    overview = f"""## CE1 Safe-PI Exact MPC-Cost SAC - `{SAC_LABEL}`

Run completed for all 25 buildings. This is the MERLIN-style SAC run with the exact MPC stage cost, original action cap `0.5`, deterministic safe-PI January replay filling, training temperature reset/drop, reward-slope early stopping, and automatic entropy tuning.

Cost/reward:

`J_k = 0.5 * [3*((Tin-(Tsp+bias))/2)^2 + 60*(cold_guard/1.5)^2 + 15*(hot_guard/2)^2] + 0.5*GridStress*P_heat/5`

`reward = -J_k`, `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and `GridStress` is the CE1 baseline-above-target profile. The setpoint schedule was not changed.

Training details: the first January episode is safe PI, reset transitions are dropped from replay instead of receiving an additive penalty, and SAC training is allowed up to `35` January episodes but stops when recent mean raw reward per step has a flat slope.

Mean building KPIs:

{_markdown_table(kpis)}

District aggregate KPIs:

{_markdown_table(district_kpis)}

Weighted term diagnostics:

{_markdown_table(term_summary)}

New-run February comfort group diagnostics:

{_markdown_table(group_diag)}

New-run early-stop and reset diagnostics:

{_markdown_table(early_stop)}

Worst new-run buildings by February discomfort:

{_markdown_table(building_diag.sort_values("discomfort_%", ascending=False).head(10))}
"""
    if not final_losses.empty:
        overview += f"""
SAC reward-training loss summary by final episode:

{_markdown_table(final_losses)}
"""
    if not final_rewards.empty:
        overview += f"""
Online reward summary by final SAC episode:

{_markdown_table(final_rewards)}
"""

    cells = [nbf.v4.new_markdown_cell(overview)]
    if jan_path is not None:
        cells.append(_image_cell(jan_path, "January safe-PI episode 0 comfort"))
    cells.append(_image_cell(comfort_paths[new_run["key"]], "February comfort - new SAC run"))
    for run in runs:
        if run["key"] == new_run["key"]:
            continue
        cells.append(_image_cell(comfort_paths[run["key"]], f"February comfort - {run['display']}"))
    for path in [district_ts_path, district_daily_path, terms_path, response_path, *training_paths]:
        cells.append(_image_cell(path, path.stem.replace("_", " ")))

    for notebook in [
        ANALYSIS_DIR / "08_sac_results.ipynb",
        ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
        ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
        ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
    ]:
        if notebook.exists():
            _append_once(notebook, marker, cells)

    print("Done.")


if __name__ == "__main__":
    main()
