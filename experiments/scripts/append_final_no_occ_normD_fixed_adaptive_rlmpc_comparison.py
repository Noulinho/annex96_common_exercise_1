#!/usr/bin/env python3
"""Append Feb+March final no-occupant fixed/adaptive GNU-RL comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

import experiments.scripts.append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis as base
import experiments.scripts.append_final_no_occ_normD_controller_comparison as final
from src.occupants.metrics import kelvin_hours
from src.rewards import time_bias_setpoint_profile


PREFIX = "ce1_final_no_occ_normD010702_fixed_adaptive_rlmpc_comparison"
MARKER = f"## Final No-Occupant Normalized-D MPC/SAC/Fixed-Adaptive GNU-RL - `{PREFIX}`"

MPC_FEB = "ce1_final_no_occ_D010702_mpc_2r2c_kalman"
MPC_MAR = "ce1_final_no_occ_D010702_mpc_2r2c_kalman_mar"
SAC_FEB_BASE = "ce1_final_no_occ_D010702_sac_safepi9_a1"
SAC_FEB_BAD = "ce1_final_no_occ_D010702_sac_safepi9_bad10_a1"
SAC_MAR = "ce1_final_no_occ_D010702_sac_feb10_mar_a1"
RLMPC_FIXED_FEB = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_fixed"
RLMPC_FIXED_MAR = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_fixed_mar"
RLMPC_ADAPT_FEB = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_adaptive_alpha"
RLMPC_ADAPT_MAR = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_adaptive_alpha_mar"

FIG_DIR = Path("figures") / PREFIX
ANALYSIS_DIR = Path("analysis")
RESULTS = Path("results")

WEIGHTS = {
    "track": 0.1,
    "cold": 0.7,
    "hot": 0.2,
    "track_scale": 2.0,
    "cold_scale": 1.5,
    "hot_scale": 2.0,
    "guard": 2.0,
    "bias": 1.0,
}

NOTEBOOKS = [
    ANALYSIS_DIR / "02_mpc_results.ipynb",
    ANALYSIS_DIR / "04_rlmpc_results.ipynb",
    ANALYSIS_DIR / "08_sac_results.ipynb",
    ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
    ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
    ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
]

MONTHS = [
    {
        "period": "February",
        "key": "feb",
        "start": 744,
        "end": 1415,
        "baseline_dir": RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline",
        "runs": [
            {"key": "mpc", "display": "MPC", "controller": "mpc", "label": MPC_FEB},
            {"key": "rlmpc_fixed", "display": "GNU-RL fixed", "controller": "rlmpc", "label": RLMPC_FIXED_FEB},
            {"key": "rlmpc_adaptive", "display": "GNU-RL PPO-cost", "controller": "rlmpc", "label": RLMPC_ADAPT_FEB},
            {"key": "sac", "display": "SAC hybrid", "controller": "sac_merlin", "label": SAC_FEB_BASE, "hybrid": True},
        ],
    },
    {
        "period": "March",
        "key": "mar",
        "start": 1416,
        "end": 2159,
        "baseline_dir": RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "march_baseline",
        "runs": [
            {"key": "mpc", "display": "MPC", "controller": "mpc", "label": MPC_MAR},
            {"key": "rlmpc_fixed", "display": "GNU-RL fixed", "controller": "rlmpc", "label": RLMPC_FIXED_MAR},
            {"key": "rlmpc_adaptive", "display": "GNU-RL PPO-cost", "controller": "rlmpc", "label": RLMPC_ADAPT_MAR},
            {"key": "sac", "display": "SAC hybrid", "controller": "sac_merlin", "label": SAC_MAR},
        ],
    },
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def raw_dir(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label


def summary_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / f"{label}_metrics.json"


def read_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def kpi_values(path: Path) -> dict[str, float]:
    data = read_csv(path)
    district = data[data.get("level", "") == "district"].copy()
    if district.empty:
        district = data
    return {
        str(row["cost_function"]): float(row["value"])
        for _, row in district.iterrows()
        if pd.notna(row.get("value"))
    }


def setpoint(data: pd.DataFrame) -> np.ndarray:
    return base._setpoint(data)


def time_step(data: pd.DataFrame) -> np.ndarray:
    return base._time_step(data)


def load(data: pd.DataFrame) -> np.ndarray:
    return base._load(data)


def sac_feb_final_label(building_id: str) -> str:
    bad = raw_dir("sac_merlin", building_id, SAC_FEB_BAD) / "kpis.csv"
    return SAC_FEB_BAD if bad.exists() else SAC_FEB_BASE


def resolved_label(period: dict[str, object], run: dict[str, object], building_id: str) -> str:
    if period["key"] == "feb" and bool(run.get("hybrid")):
        return sac_feb_final_label(building_id)
    return str(run["label"])


def rollout_path(period: dict[str, object], run: dict[str, object], building_id: str) -> Path:
    label = resolved_label(period, run, building_id)
    return raw_dir(str(run["controller"]), building_id, label) / "rollout.csv"


def kpis_path(period: dict[str, object], run: dict[str, object], building_id: str) -> Path:
    label = resolved_label(period, run, building_id)
    return raw_dir(str(run["controller"]), building_id, label) / "kpis.csv"


def bias_profile(start: int, n: int) -> np.ndarray:
    return np.asarray(
        time_bias_setpoint_profile(
            simulation_start=start,
            simulation_end=start + n - 1,
            amplitude=WEIGHTS["bias"],
        ),
        dtype=float,
    )[:n]


def building_kpis() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period in MONTHS:
        for run in period["runs"]:
            for building in base.BUILDINGS:
                building_id = building["result_id"]
                data = read_csv(rollout_path(period, run, building_id))
                tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
                values = kpi_values(kpis_path(period, run, building_id))
                rows.append(
                    {
                        "period": period["period"],
                        "controller": run["display"],
                        "building": building["display"],
                        "building_id": building_id,
                        "source_label": resolved_label(period, run, building_id),
                        "electricity_consumption_total": values.get("electricity_consumption_total", np.nan),
                        "discomfort_%": 100.0 * values.get("discomfort_proportion", np.nan),
                        "cold_%": 100.0 * values.get("discomfort_cold_proportion", np.nan),
                        "hot_%": 100.0 * values.get("discomfort_hot_proportion", np.nan),
                        "kelvin_hours": kelvin_hours(tin, lower=20.0, upper=24.0),
                        "daily_peak_average": values.get("daily_peak_average", np.nan),
                        "ramping_average": values.get("ramping_average", np.nan),
                        "tin_min": float(np.nanmin(tin)),
                        "tin_max": float(np.nanmax(tin)),
                    }
                )
    return pd.DataFrame(rows)


def mean_building_kpis(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "ramping_average",
        "tin_min",
        "tin_max",
    ]
    return rows.groupby(["period", "controller"], as_index=False)[columns].mean(numeric_only=True)


def baseline_district(period: dict[str, object]) -> pd.DataFrame:
    data = read_csv(Path(period["baseline_dir"]) / "district_rollout.csv")
    target = data["district_target"] if "district_target" in data.columns else 0.0
    return pd.DataFrame(
        {
            "period": period["period"],
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "target": pd.to_numeric(target, errors="coerce").fillna(0.0),
            "controller": "CityLearn",
        }
    )


def district_frame(period: dict[str, object], run: dict[str, object]) -> pd.DataFrame:
    series: list[np.ndarray] = []
    steps: np.ndarray | None = None
    for building in base.BUILDINGS:
        data = read_csv(rollout_path(period, run, building["result_id"]))
        values = load(data)
        if steps is None:
            steps = time_step(data)
        n = min(len(steps), len(values))
        steps = steps[:n]
        series.append(values[:n])
    n = min(len(values) for values in series)
    baseline = baseline_district(period)
    return pd.DataFrame(
        {
            "period": period["period"],
            "time_step": np.asarray(steps[:n], dtype=int),
            "load": np.sum([values[:n] for values in series], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "controller": run["display"],
        }
    )


def district_frames() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for period in MONTHS:
        frames.append(baseline_district(period))
        frames.extend(district_frame(period, run) for run in period["runs"])
    return frames


def district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period_name, period_frames in pd.concat(frames, ignore_index=True).groupby("period", sort=False):
        baseline = period_frames[period_frames["controller"] == "CityLearn"]["load"].to_numpy(dtype=float)
        base_total = max(float(np.sum(baseline)), 1e-9)
        base_peak = max(float(np.max(baseline)), 1e-9)
        base_ramp = max(float(np.sum(np.abs(np.diff(baseline)))), 1e-9)
        for controller, frame in period_frames.groupby("controller", sort=False):
            load_values = frame["load"].to_numpy(dtype=float)
            target = frame["target"].to_numpy(dtype=float)
            n = min(len(load_values), len(target))
            load_values, target = load_values[:n], target[:n]
            hourly = pd.DataFrame({"hour": np.arange(n) % 24, "load": load_values}).groupby("hour")["load"].mean()
            ramp = float(np.sum(np.abs(np.diff(load_values)))) if n > 1 else 0.0
            rows.append(
                {
                    "period": period_name,
                    "controller": controller,
                    "total_kwh": float(np.sum(load_values)),
                    "total_vs_citylearn": float(np.sum(load_values) / base_total),
                    "peak_kwh": float(np.max(load_values)),
                    "peak_vs_citylearn": float(np.max(load_values) / base_peak),
                    "daily_profile_peak_kwh": float(hourly.max()),
                    "daily_profile_peak_hour": int(hourly.idxmax()),
                    "ramping_abs_kwh": ramp,
                    "ramping_vs_citylearn": float(ramp / base_ramp),
                    "target_rmse_kwh": float(np.sqrt(np.mean((load_values - target) ** 2))),
                    "above_target_kwh": float(np.sum(np.maximum(load_values - target, 0.0))),
                }
            )
    return pd.DataFrame(rows)


def district_hourly(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.assign(hour=combined["time_step"] % 24)
        .groupby(["period", "controller", "hour"], as_index=False)
        .agg(load=("load", "mean"), target=("target", "mean"))
    )


def term_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    hourly_rows: list[pd.DataFrame] = []
    for period in MONTHS:
        start = int(period["start"])
        for run in period["runs"]:
            for building in base.BUILDINGS:
                data = read_csv(rollout_path(period, run, building["result_id"]))
                tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
                tsp = setpoint(data)
                steps = time_step(data)
                n = min(len(tin), len(tsp), len(steps))
                tin, tsp, steps = tin[:n], tsp[:n], steps[:n]
                bias = bias_profile(start, n)
                tref = tsp + bias
                cold_guard = np.maximum(tsp - WEIGHTS["guard"] - tin, 0.0)
                hot_guard = np.maximum(tin - tsp - WEIGHTS["guard"], 0.0)
                track = WEIGHTS["track"] * ((tin - tref) / WEIGHTS["track_scale"]) ** 2
                cold = WEIGHTS["cold"] * (cold_guard / WEIGHTS["cold_scale"]) ** 2
                hot = WEIGHTS["hot"] * (hot_guard / WEIGHTS["hot_scale"]) ** 2
                stage = track + cold + hot
                guard = (cold_guard > 0.0) | (hot_guard > 0.0)
                for term, values in {
                    "tracking_target": track,
                    "guard_cold": cold,
                    "guard_hot": hot,
                    "stage_cost": stage,
                    "mae_to_true_tsp_c": np.abs(tin - tsp),
                    "mae_to_biased_tref_c": np.abs(tin - tref),
                }.items():
                    rows.append(
                        {
                            "period": period["period"],
                            "controller": run["display"],
                            "building": building["display"],
                            "building_id": building["result_id"],
                            "term": term,
                            "mean": float(np.nanmean(values)),
                            "max": float(np.nanmax(values)),
                        }
                    )
                rows.append(
                    {
                        "period": period["period"],
                        "controller": run["display"],
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "term": "guard_violation_%",
                        "mean": float(100.0 * np.nanmean(guard)),
                        "max": float(100.0 * np.nanmax(guard)),
                    }
                )
                hourly_rows.append(
                    pd.DataFrame(
                        {
                            "period": period["period"],
                            "controller": run["display"],
                            "building": building["display"],
                            "building_id": building["result_id"],
                            "hour": (steps + start) % 24,
                            "tin_minus_tsp": tin - tsp,
                            "guard_violation": guard.astype(float),
                            "stage_cost": stage,
                        }
                    )
                )
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def rlm_pc_alpha_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period in MONTHS:
        for run in period["runs"]:
            if str(run["controller"]) != "rlmpc":
                continue
            label = str(run["label"])
            for building in base.BUILDINGS:
                building_id = building["result_id"]
                metrics = read_metrics(summary_path("rlmpc", building_id, label))
                hist = safe_csv(raw_dir("rlmpc", building_id, label) / "ppo_history.csv")
                loss_col = next((c for c in ("loss", "ppo_loss", "policy_loss", "objective") if c in hist.columns), None)
                rows.append(
                    {
                        "period": period["period"],
                        "controller": run["display"],
                        "building": building["display"],
                        "building_id": building_id,
                        "adaptive": bool(
                            metrics.get("matched_mpc_adaptive_weights", metrics.get("balanced_adaptive_weights", False))
                        ),
                        "ppo_rows": int(len(hist)),
                        "ppo_loss_mean": float(pd.to_numeric(hist[loss_col], errors="coerce").mean())
                        if loss_col
                        else np.nan,
                        "alpha_track_before": metrics.get("alpha_track_before", np.nan),
                        "alpha_cold_before": metrics.get("alpha_cold_before", np.nan),
                        "alpha_hot_before": metrics.get("alpha_hot_before", np.nan),
                        "alpha_track_after": metrics.get("alpha_track_after", np.nan),
                        "alpha_cold_after": metrics.get("alpha_cold_after", np.nan),
                        "alpha_hot_after": metrics.get("alpha_hot_after", np.nan),
                        "rollout_reward": metrics.get("reward_total", np.nan),
                        "train_reward_total": metrics.get("train_reward_total", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def rlm_pc_alpha_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return (
        rows.groupby(["period", "controller"], as_index=False)
        .agg(
            adaptive=("adaptive", "max"),
            ppo_rows=("ppo_rows", "sum"),
            ppo_loss_mean=("ppo_loss_mean", "mean"),
            alpha_track_before=("alpha_track_before", "mean"),
            alpha_cold_before=("alpha_cold_before", "mean"),
            alpha_hot_before=("alpha_hot_before", "mean"),
            alpha_track_after=("alpha_track_after", "mean"),
            alpha_cold_after=("alpha_cold_after", "mean"),
            alpha_hot_after=("alpha_hot_after", "mean"),
            rollout_reward=("rollout_reward", "mean"),
            train_reward_total=("train_reward_total", "mean"),
        )
        .sort_values(["period", "controller"])
    )


def sac_training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reward_frames: list[pd.DataFrame] = []
    loss_frames: list[pd.DataFrame] = []
    cases = [
        ("February", "Jan safe-PI + 9 SAC", SAC_FEB_BASE, {b["result_id"] for b in base.BUILDINGS}),
        (
            "February",
            "Jan bad-building +10 SAC",
            SAC_FEB_BAD,
            {b["result_id"] for b in base.BUILDINGS if (raw_dir("sac_merlin", b["result_id"], SAC_FEB_BAD) / "rollout.csv").exists()},
        ),
        ("March", "Feb +10 SAC continuation", SAC_MAR, {b["result_id"] for b in base.BUILDINGS}),
    ]
    for period_name, phase_name, label, allowed in cases:
        for building in base.BUILDINGS:
            building_id = building["result_id"]
            if building_id not in allowed:
                continue
            case_dir = raw_dir("sac_merlin", building_id, label)
            reward_path = case_dir / "online_training_rollout.csv"
            if reward_path.exists():
                data = read_csv(reward_path)
                data["period"] = period_name
                data["training_case"] = phase_name
                data["building"] = building["display"]
                data["building_id"] = building_id
                data["source_label"] = label
                reward_frames.append(data)
            loss_path = case_dir / "sac_training_losses.csv"
            if loss_path.exists():
                losses = read_csv(loss_path)
                losses["period"] = period_name
                losses["training_case"] = phase_name
                losses["building"] = building["display"]
                losses["building_id"] = building_id
                losses["source_label"] = label
                loss_frames.append(losses)
    return (
        pd.concat(reward_frames, ignore_index=True) if reward_frames else pd.DataFrame(),
        pd.concat(loss_frames, ignore_index=True) if loss_frames else pd.DataFrame(),
    )


def sac_reward_summary(rewards: pd.DataFrame) -> pd.DataFrame:
    if rewards.empty:
        return pd.DataFrame()
    data = rewards[rewards["phase"].astype(str).str.contains("sac", case=False, na=False)].copy()
    if data.empty:
        return pd.DataFrame()
    raw_col = "raw_reward" if "raw_reward" in data.columns else "reward"
    data["raw_reward_plot"] = pd.to_numeric(data[raw_col], errors="coerce")
    data["reward_plot"] = pd.to_numeric(data["reward"], errors="coerce")
    if "temperature_reset_applied" in data.columns:
        reset = data["temperature_reset_applied"]
        if reset.dtype == object:
            data["reset_flag"] = reset.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        else:
            data["reset_flag"] = pd.to_numeric(reset, errors="coerce").fillna(0.0)
    else:
        data["reset_flag"] = 0.0
    return (
        data.groupby(["period", "training_case", "episode"], as_index=False)
        .agg(
            raw_reward_mean=("raw_reward_plot", "mean"),
            raw_reward_p10=("raw_reward_plot", lambda x: float(np.nanpercentile(x, 10))),
            raw_reward_p90=("raw_reward_plot", lambda x: float(np.nanpercentile(x, 90))),
            reward_mean=("reward_plot", "mean"),
            reset_fraction=("reset_flag", "mean"),
            action_mean=("action", "mean"),
        )
        .sort_values(["period", "training_case", "episode"])
    )


def sac_loss_summary(losses: pd.DataFrame) -> pd.DataFrame:
    if losses.empty:
        return pd.DataFrame()
    cols = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"] if c in losses.columns]
    if not cols:
        return pd.DataFrame()
    out = losses.groupby(["period", "training_case", "episode"], as_index=False)[cols].agg(["mean", "max"])
    out.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in out.columns.to_flat_index()]
    return out


def sac_feb_bad_delta() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for building in base.BUILDINGS:
        building_id = building["result_id"]
        bad_kpis = raw_dir("sac_merlin", building_id, SAC_FEB_BAD) / "kpis.csv"
        if not bad_kpis.exists():
            continue
        base_values = kpi_values(raw_dir("sac_merlin", building_id, SAC_FEB_BASE) / "kpis.csv")
        bad_values = kpi_values(bad_kpis)
        rows.append(
            {
                "building": building["display"],
                "building_id": building_id,
                "disc_%_ep9": 100.0 * base_values.get("discomfort_proportion", np.nan),
                "disc_%_final": 100.0 * bad_values.get("discomfort_proportion", np.nan),
                "delta_disc_%": 100.0
                * (bad_values.get("discomfort_proportion", np.nan) - base_values.get("discomfort_proportion", np.nan)),
                "cold_%_final": 100.0 * bad_values.get("discomfort_cold_proportion", np.nan),
                "hot_%_final": 100.0 * bad_values.get("discomfort_hot_proportion", np.nan),
                "energy_ep9": base_values.get("electricity_consumption_total", np.nan),
                "energy_final": bad_values.get("electricity_consumption_total", np.nan),
                "delta_energy": bad_values.get("electricity_consumption_total", np.nan)
                - base_values.get("electricity_consumption_total", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("delta_disc_%") if not out.empty else out


def compact_building_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        [
            "period",
            "controller",
            "building",
            "electricity_consumption_total",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "kelvin_hours",
            "daily_peak_average",
            "ramping_average",
            "tin_min",
            "tin_max",
            "source_label",
        ]
    ].rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
            "source_label": "label",
        }
    )


def mean_comfort_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        [
            "period",
            "controller",
            "electricity_consumption_total",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "kelvin_hours",
        ]
    ].rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
        }
    )


def mean_ops_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        [
            "period",
            "controller",
            "daily_peak_average",
            "ramping_average",
            "tin_min",
            "tin_max",
        ]
    ].rename(
        columns={
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
        }
    )


def district_shape_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        [
            "period",
            "controller",
            "total_vs_citylearn",
            "peak_vs_citylearn",
            "daily_profile_peak_kwh",
            "daily_profile_peak_hour",
        ]
    ].rename(
        columns={
            "total_vs_citylearn": "total_vs_CL",
            "peak_vs_citylearn": "peak_vs_CL",
            "daily_profile_peak_kwh": "daily_peak_kWh",
            "daily_profile_peak_hour": "daily_peak_hr",
        }
    )


def district_tracking_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        [
            "period",
            "controller",
            "ramping_vs_citylearn",
            "target_rmse_kwh",
            "above_target_kwh",
        ]
    ].rename(
        columns={
            "ramping_vs_citylearn": "ramp_vs_CL",
            "target_rmse_kwh": "target_RMSE",
            "above_target_kwh": "above_target",
        }
    )


def pivot_terms(summary: pd.DataFrame, value: str) -> pd.DataFrame:
    terms = ["tracking_target", "guard_cold", "guard_hot", "stage_cost", "guard_violation_%"]
    out = summary[summary["term"].isin(terms)].pivot_table(
        index=["period", "controller"], columns="term", values=value, aggfunc="mean"
    )
    out = out.reset_index()
    return out.rename(
        columns={
            "tracking_target": "track",
            "guard_cold": "cold_guard",
            "guard_hot": "hot_guard",
            "guard_violation_%": "guard_%",
            "stage_cost": "stage",
        }
    )


def markdown_table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    return base._markdown_table(df, digits=digits, max_rows=max_rows)


def image(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.as_posix()})")


def make_comfort_grid(period: dict[str, object], run: dict[str, object]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    start = int(period["start"])
    for ax, building in zip(axes, base.BUILDINGS):
        data = read_csv(rollout_path(period, run, building["result_id"]))
        t = time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = setpoint(data)
        n = min(len(t), len(tin), len(tsp))
        bias = bias_profile(start, n)
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.65, ls="--", color="black", label="Tsp")
        ax.plot(t[:n], tsp[:n] + bias[:n], lw=0.65, ls=":", color="tab:orange", label="Tsp+bias")
        ax.fill_between(t[:n], tsp[:n] - WEIGHTS["guard"], tsp[:n] + WEIGHTS["guard"], color="gray", alpha=0.13)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{period['period']} | {run['display']} | plot_comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{period['key']}_{run['key']}_comfort_all25.png"
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def make_safe_pi_grid() -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, base.BUILDINGS):
        data = read_csv(raw_dir("sac_merlin", building["result_id"], SAC_FEB_BASE) / "online_training_rollout.csv")
        subset = data[(data["episode"] == 0) & (data["phase"].astype(str) == "safe_pi_exploration")].copy()
        t = pd.to_numeric(subset["time_step"], errors="coerce").to_numpy(dtype=float)
        tin = pd.to_numeric(subset["tin_before_action"], errors="coerce").to_numpy(dtype=float)
        tsp = pd.to_numeric(subset["tsp_before_action"], errors="coerce").to_numpy(dtype=float)
        n = min(len(t), len(tin), len(tsp))
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.65, ls="--", color="black", label="Tsp")
        ax.fill_between(t[:n], tsp[:n] - 2.0, tsp[:n] + 2.0, color="gray", alpha=0.13)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("SAC safe-PI January bootstrap episode 0 comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "sac_safe_pi_january_episode0_comfort_all25.png"
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
    for ax, (period_name, group) in zip(axes, combined.groupby("period", sort=False)):
        for controller, data in group.groupby("controller", sort=False):
            ax.plot(data["time_step"], data["load"], lw=1.0, label=controller)
        baseline = group[group["controller"] == "CityLearn"]
        ax.plot(baseline["time_step"], baseline["target"], color="black", lw=0.9, ls="--", label="CE1 target")
        ax.set_title(f"{period_name} district load time series")
        ax.set_xlabel(f"{period_name} time step")
        ax.set_ylabel("District electricity [kWh]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries_feb_mar.png"
    fig.savefig(load_path, dpi=145, bbox_inches="tight")
    plt.close(fig)

    hourly = district_hourly(frames)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, (period_name, group) in zip(axes, hourly.groupby("period", sort=False)):
        for controller, data in group.groupby("controller", sort=False):
            ax.plot(data["hour"], data["load"], marker="o", lw=1.25, label=controller)
        target = group[group["controller"] == "CityLearn"]
        ax.plot(target["hour"], target["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
        ax.set_title(f"Mean hourly district load - {period_name}")
        ax.set_ylabel("Mean district electricity [kWh]")
        ax.set_xticks(range(24))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Hour of day")
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_hourly_profile_feb_mar.png"
    fig.savefig(daily_path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


def make_term_plots(term_summary: pd.DataFrame, hourly: pd.DataFrame) -> tuple[Path, Path]:
    terms = ["tracking_target", "guard_cold", "guard_hot", "stage_cost"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    for ax, (period_name, group) in zip(axes, term_summary[term_summary["term"].isin(terms)].groupby("period", sort=False)):
        pivot = group.pivot(index="term", columns="controller", values="mean").reindex(terms)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Mean weighted stage-cost terms - {period_name}")
        ax.set_ylabel("Mean weighted value")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    terms_path = FIG_DIR / "weighted_terms_feb_mar.png"
    fig.savefig(terms_path, dpi=145, bbox_inches="tight")
    plt.close(fig)

    hourly_mean = hourly.groupby(["period", "controller", "hour"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    for col, (period_name, group) in enumerate(hourly_mean.groupby("period", sort=False)):
        for controller, data in group.groupby("controller", sort=False):
            axes[0, col].plot(data["hour"], data["tin_minus_tsp"], marker="o", lw=1.1, label=controller)
            axes[1, col].plot(data["hour"], 100.0 * data["guard_violation"], marker="o", lw=1.1, label=controller)
            axes[2, col].plot(data["hour"], data["stage_cost"], marker="o", lw=1.1, label=controller)
        axes[0, col].set_title(f"{period_name}: Tin - Tsp")
        axes[1, col].set_title(f"{period_name}: guard violation")
        axes[2, col].set_title(f"{period_name}: stage cost")
        axes[2, col].set_xlabel("Hour")
        for row in range(3):
            axes[row, col].grid(alpha=0.25)
            axes[row, col].set_xticks(range(24))
    axes[0, 0].set_ylabel("C")
    axes[1, 0].set_ylabel("%")
    axes[2, 0].set_ylabel("cost")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    response_path = FIG_DIR / "hourly_bias_response_feb_mar.png"
    fig.savefig(response_path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return terms_path, response_path


def make_rlmpc_alpha_plot(summary: pd.DataFrame) -> Path:
    path = FIG_DIR / "rlmpc_fixed_vs_adaptive_alpha_weights_feb_mar.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    if summary.empty:
        axes[0].text(0.5, 0.5, "No GNU-RL alpha rows", ha="center", va="center")
        axes[0].axis("off")
        axes[1].axis("off")
    else:
        colors = ["#4c78a8", "#f58518", "#54a24b"]
        labels = ["track", "cold", "hot"]
        for ax, (period_name, group) in zip(axes, summary.groupby("period", sort=False)):
            x = np.arange(len(group))
            width = 0.35
            before = group[["alpha_track_before", "alpha_cold_before", "alpha_hot_before"]].to_numpy(dtype=float)
            after = group[["alpha_track_after", "alpha_cold_after", "alpha_hot_after"]].to_numpy(dtype=float)
            b_bottom = np.zeros(len(group))
            a_bottom = np.zeros(len(group))
            for idx, label in enumerate(labels):
                ax.bar(x - width / 2, before[:, idx], width, bottom=b_bottom, color=colors[idx], alpha=0.45)
                ax.bar(x + width / 2, after[:, idx], width, bottom=a_bottom, color=colors[idx], alpha=0.9, label=label)
                b_bottom += before[:, idx]
                a_bottom += after[:, idx]
            ax.set_xticks(x)
            ax.set_xticklabels(group["controller"], rotation=10, ha="right")
            ax.set_ylabel("Softmax comfort weight")
            ax.set_title(f"{period_name}: GNU-RL comfort weights before/after PPO")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sac_reward_plot(summary: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_online_training_reward_mean_readable_feb_mar.png"
    fig, ax = plt.subplots(figsize=(13, 5))
    if summary.empty:
        ax.text(0.5, 0.5, "No SAC reward rows", ha="center", va="center")
        ax.axis("off")
    else:
        for (period_name, case), data in summary.groupby(["period", "training_case"], sort=False):
            ax.plot(data["episode"], data["raw_reward_mean"], marker="o", lw=1.3, label=f"{period_name}: {case}")
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("SAC online reward by episode - mean per step")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean raw reward per step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sac_reset_plot(summary: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_online_training_reset_fraction_feb_mar.png"
    fig, ax = plt.subplots(figsize=(13, 4.5))
    if summary.empty:
        ax.text(0.5, 0.5, "No SAC reset rows", ha="center", va="center")
        ax.axis("off")
    else:
        for (period_name, case), data in summary.groupby(["period", "training_case"], sort=False):
            ax.plot(data["episode"], 100.0 * data["reset_fraction"], marker="o", lw=1.3, label=f"{period_name}: {case}")
        ax.set_title("SAC training reset rate")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Reset-applied steps [%]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sac_loss_plot(summary: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_training_losses_feb_mar.png"
    cols = [c for c in ["soft_q1_loss_mean", "soft_q2_loss_mean", "policy_loss_mean", "alpha_mean"] if c in summary.columns]
    if summary.empty or not cols:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No SAC loss rows", ha="center", va="center")
        ax.axis("off")
    else:
        fig, axes = plt.subplots(len(cols), 1, figsize=(13, 3.0 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            for (period_name, case), data in summary.groupby(["period", "training_case"], sort=False):
                ax.plot(data["episode"], data[col], marker="o", lw=1.0, label=f"{period_name}: {case}")
            ax.set_ylabel(col.replace("_mean", ""))
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Training episode")
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sac_bad_delta_plot(delta: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_feb_bad_building_plus9_vs_bad10_discomfort.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    if delta.empty:
        ax.text(0.5, 0.5, "No SAC bad-building continuation rows", ha="center", va="center")
        ax.axis("off")
    else:
        plot = delta.sort_values("disc_%_ep9", ascending=False)
        x = np.arange(len(plot))
        width = 0.38
        ax.bar(x - width / 2, plot["disc_%_ep9"], width, label="after 9 SAC episodes")
        ax.bar(x + width / 2, plot["disc_%_final"], width, label="after bad +10")
        ax.axhline(10.0, color="black", ls="--", lw=0.9, label="10% threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(plot["building"], rotation=45, ha="right")
        ax.set_ylabel("February discomfort [%]")
        ax.set_title("SAC bad-building continuation: February discomfort")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def save_outputs(
    bldg: pd.DataFrame,
    mean: pd.DataFrame,
    dist: pd.DataFrame,
    hourly: pd.DataFrame,
    term_building: pd.DataFrame,
    term_summary: pd.DataFrame,
    term_hourly: pd.DataFrame,
    alpha_rows: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    rewards: pd.DataFrame,
    losses: pd.DataFrame,
    reward_summary: pd.DataFrame,
    loss_summary: pd.DataFrame,
    bad_delta: pd.DataFrame,
) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    bldg.to_csv(ANALYSIS_DIR / f"{PREFIX}_building_kpis_all25.csv", index=False)
    compact_building_table(bldg).to_csv(ANALYSIS_DIR / f"{PREFIX}_building_kpis_all25_compact.csv", index=False)
    mean.to_csv(ANALYSIS_DIR / f"{PREFIX}_mean_building_kpis.csv", index=False)
    dist.to_csv(ANALYSIS_DIR / f"{PREFIX}_district_kpis.csv", index=False)
    hourly.to_csv(ANALYSIS_DIR / f"{PREFIX}_district_mean_hourly_profile.csv", index=False)
    term_building.to_csv(ANALYSIS_DIR / f"{PREFIX}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{PREFIX}_term_diagnostics.csv", index=False)
    term_hourly.to_csv(ANALYSIS_DIR / f"{PREFIX}_hourly_bias_response.csv", index=False)
    alpha_rows.to_csv(ANALYSIS_DIR / f"{PREFIX}_rlmpc_alpha_by_building.csv", index=False)
    alpha_summary.to_csv(ANALYSIS_DIR / f"{PREFIX}_rlmpc_alpha_summary.csv", index=False)
    rewards.to_csv(ANALYSIS_DIR / f"{PREFIX}_sac_online_training_rewards_raw.csv", index=False)
    losses.to_csv(ANALYSIS_DIR / f"{PREFIX}_sac_training_losses_raw.csv", index=False)
    reward_summary.to_csv(ANALYSIS_DIR / f"{PREFIX}_sac_online_reward_summary.csv", index=False)
    loss_summary.to_csv(ANALYSIS_DIR / f"{PREFIX}_sac_loss_summary.csv", index=False)
    bad_delta.to_csv(ANALYSIS_DIR / f"{PREFIX}_sac_bad_delta.csv", index=False)


def replace_section(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    start = None
    for idx, cell in enumerate(notebook.cells):
        if MARKER in str(cell.get("source", "")):
            start = idx
            break
    if start is None:
        notebook.cells.extend(cells)
        nbf.write(notebook, path)
        print(f"Appended {MARKER} to {path}")
        return
    end = start + 1
    while end < len(notebook.cells):
        source = str(notebook.cells[end].get("source", ""))
        if notebook.cells[end].get("cell_type") == "markdown" and source.startswith("## ") and MARKER not in source:
            break
        end += 1
    notebook.cells[start:end] = cells
    nbf.write(notebook, path)
    print(f"Updated {MARKER} in {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    bldg = building_kpis()
    mean = mean_building_kpis(bldg)
    frames = district_frames()
    dist = district_kpis(frames)
    hourly = district_hourly(frames)
    term_building, term_hourly = term_diagnostics()
    term_summary = (
        term_building.groupby(["period", "controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["period", "controller", "term"])
    )
    alpha_rows = rlm_pc_alpha_rows()
    alpha_summary = rlm_pc_alpha_summary(alpha_rows)
    rewards, losses = sac_training_frames()
    reward_summary = sac_reward_summary(rewards)
    loss_summary = sac_loss_summary(losses)
    bad_delta = sac_feb_bad_delta()

    save_outputs(
        bldg,
        mean,
        dist,
        hourly,
        term_building,
        term_summary,
        term_hourly,
        alpha_rows,
        alpha_summary,
        rewards,
        losses,
        reward_summary,
        loss_summary,
        bad_delta,
    )

    comfort_paths = {
        (period["key"], run["key"]): make_comfort_grid(period, run)
        for period in MONTHS
        for run in period["runs"]
    }
    safe_pi_path = make_safe_pi_grid()
    district_load_path, district_daily_path = make_district_plots(frames)
    terms_path, response_path = make_term_plots(term_summary, term_hourly)
    alpha_path = make_rlmpc_alpha_plot(alpha_summary)
    sac_reward_path = make_sac_reward_plot(reward_summary)
    sac_reset_path = make_sac_reset_plot(reward_summary)
    sac_loss_path = make_sac_loss_plot(loss_summary)
    bad_delta_path = make_sac_bad_delta_plot(bad_delta)

    term_mean = pivot_terms(term_summary, "mean")
    term_max = pivot_terms(term_summary, "max")
    all_buildings = compact_building_table(bldg)
    feb_buildings = all_buildings[all_buildings["period"] == "February"]
    mar_buildings = all_buildings[all_buildings["period"] == "March"]
    reward_display = reward_summary[
        ["period", "training_case", "episode", "raw_reward_mean", "reward_mean", "reset_fraction", "action_mean"]
    ].copy() if not reward_summary.empty else pd.DataFrame()
    if not reward_display.empty:
        reward_display["reset_%"] = 100.0 * reward_display.pop("reset_fraction")
    loss_display = loss_summary.tail(16).copy() if not loss_summary.empty else pd.DataFrame()
    still_bad = bad_delta[bad_delta["disc_%_final"] > 10.0].copy() if not bad_delta.empty else pd.DataFrame()

    overview = f"""{MARKER}

Final no-occupant comfort-only comparison for February and March.

Controller labels:

`{MPC_FEB}` / `{MPC_MAR}`: MPC with January-only 2R2C+Kalman RC fit, horizon 24, deployed in February and March.

`{RLMPC_FIXED_FEB}` / `{RLMPC_FIXED_MAR}`: fixed GNU-RL/RLMPC with the completed 1R1C imitation-learned model, horizon 6, and fixed comfort weights `(0.1, 0.7, 0.2)`.

`{RLMPC_ADAPT_FEB}` / `{RLMPC_ADAPT_MAR}`: same GNU-RL/RLMPC pipeline, but PPO updates only the inner-MPC softmax comfort weights `(alpha_track, alpha_cold, alpha_hot)` online during the deployment month.

`{SAC_FEB_BASE}` + `{SAC_FEB_BAD}`: SAC safe-PI bootstrap on January, 9 SAC reward episodes, then 10 extra episodes for bad-comfort buildings. `{SAC_MAR}` continues SAC training for 10 February episodes and deploys in March.

Cost/reward used here:

`D_k = 0.1*((Tin-(Tsp+bias))/2)^2 + 0.7*(max(Tsp-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tsp-2,0)/2)^2`

`J_k = D_k`, `reward = -D_k`.

There is no GridStress term, no electricity term, no peak term, and no ramp term. The weights sum to 1.0. `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and cold/hot guards use the true setpoint with a `Tsp +/- 2 C` band.

Mean building comfort KPIs:

{markdown_table(mean_comfort_table(mean), digits=3)}

Mean building operation KPIs:

{markdown_table(mean_ops_table(mean), digits=3)}

District aggregate KPIs - load shape:

{markdown_table(district_shape_table(dist), digits=3)}

District aggregate KPIs - tracking:

{markdown_table(district_tracking_table(dist), digits=3)}

Weighted term diagnostics - mean:

{markdown_table(term_mean, digits=4)}

Weighted term diagnostics - max:

{markdown_table(term_max, digits=3)}

GNU-RL fixed/adaptive PPO alpha summary:

{markdown_table(alpha_summary, digits=4)}

February all-building KPI table, 25 buildings x 4 controllers:

{markdown_table(feb_buildings, digits=3)}

March all-building KPI table, 25 buildings x 4 controllers:

{markdown_table(mar_buildings, digits=3)}

SAC February bad-building continuation deltas:

{markdown_table(bad_delta, digits=3)}

SAC buildings still above 10% February discomfort after the bad-building continuation:

{markdown_table(still_bad, digits=3)}
"""
    if not reward_display.empty:
        overview += f"""

SAC online reward summary - per-step means:

{markdown_table(reward_display, digits=3)}
"""
    if not loss_display.empty:
        overview += f"""

SAC loss summary, tail rows:

{markdown_table(loss_display, digits=3)}
"""
    overview += f"""

Saved compact tables:

- `analysis/{PREFIX}_building_kpis_all25.csv`
- `analysis/{PREFIX}_building_kpis_all25_compact.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_district_mean_hourly_profile.csv`
- `analysis/{PREFIX}_term_diagnostics.csv`
- `analysis/{PREFIX}_term_diagnostics_by_building.csv`
- `analysis/{PREFIX}_rlmpc_alpha_by_building.csv`
- `analysis/{PREFIX}_rlmpc_alpha_summary.csv`
- `analysis/{PREFIX}_sac_bad_delta.csv`
- `analysis/{PREFIX}_sac_online_reward_summary.csv`
- `analysis/{PREFIX}_sac_loss_summary.csv`
"""

    image_cells = []
    for period in MONTHS:
        for run in period["runs"]:
            title = f"{period['period']} {run['display']} plot_comfort all buildings"
            image_cells.append(image(comfort_paths[(period["key"], run["key"])], title))
    image_cells.extend(
        [
            image(safe_pi_path, "SAC safe-PI January episode 0 comfort"),
            image(district_load_path, "District load time series - February and March"),
            image(district_daily_path, "Mean hourly district load profile - February and March"),
            image(terms_path, "Weighted stage-cost terms - February and March"),
            image(response_path, "Hourly time-bias response - February and March"),
            image(alpha_path, "GNU-RL fixed/adaptive PPO alpha weights - February and March"),
            image(sac_reward_path, "SAC online reward per-step mean - readable scale"),
            image(sac_reset_path, "SAC training reset fraction"),
            image(sac_loss_path, "SAC losses"),
            image(bad_delta_path, "SAC bad-building continuation discomfort"),
        ]
    )
    cells = [nbf.v4.new_markdown_cell(overview), *image_cells]

    for notebook in NOTEBOOKS:
        if notebook.exists():
            replace_section(notebook, cells)

    print(f"Done: {PREFIX}")


if __name__ == "__main__":
    main()
