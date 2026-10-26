#!/usr/bin/env python3
"""Compare the original old RLMPC run against a fresh replicate label."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


OLD_LABEL = "ce1_time_bias_guard_1c_1r1c"
REP_LABEL = "ce1_time_bias_guard_1c_1r1c_rlmpc_replicate"
SCENARIO = "no_occupant"
BUILDINGS = list(range(1, 26))

ROOT = Path(".")
ANALYSIS_DIR = ROOT / "analysis"
FIG_DIR = ANALYSIS_DIR / f"{REP_LABEL}_check_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def raw_dir(building: int, label: str) -> Path:
    return ROOT / "results" / "raw" / "rlmpc" / f"building_{building:02d}" / SCENARIO / label


def metrics_path(building: int, label: str) -> Path:
    return (
        ROOT
        / "results"
        / "summaries"
        / "rlmpc"
        / f"building_{building:02d}"
        / SCENARIO
        / f"{label}_metrics.json"
    )


def max_abs_numeric_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    cols = [c for c in a.columns.intersection(b.columns) if pd.api.types.is_numeric_dtype(a[c])]
    if not cols:
        return 0.0
    diffs = []
    for col in cols:
        av = a[col].to_numpy(dtype=float)
        bv = b[col].to_numpy(dtype=float)
        if len(av) != len(bv):
            diffs.append(np.inf)
        else:
            diffs.append(np.nanmax(np.abs(av - bv)))
    return float(np.nanmax(diffs)) if diffs else 0.0


def compare_csv_file(building: int, filename: str) -> dict:
    old_path = raw_dir(building, OLD_LABEL) / filename
    rep_path = raw_dir(building, REP_LABEL) / filename
    row = {
        "building": building,
        "file": filename,
        "old_exists": old_path.exists(),
        "rep_exists": rep_path.exists(),
    }
    if not old_path.exists() or not rep_path.exists():
        return row

    try:
        old = pd.read_csv(old_path)
        old_empty = False
    except EmptyDataError:
        old = pd.DataFrame()
        old_empty = True
    try:
        rep = pd.read_csv(rep_path)
        rep_empty = False
    except EmptyDataError:
        rep = pd.DataFrame()
        rep_empty = True
    row.update(
        {
            "same_shape": old.shape == rep.shape,
            "same_columns": list(old.columns) == list(rep.columns),
            "exact_dataframe_equal": old.equals(rep),
            "old_empty": old_empty,
            "rep_empty": rep_empty,
            "n_rows_old": len(old),
            "n_rows_rep": len(rep),
            "max_abs_numeric_diff": max_abs_numeric_diff(old, rep),
        }
    )

    for col in [
        "action",
        "indoor_temperature",
        "effective_setpoint",
        "reward",
        "net_electricity_consumption",
        "heating_electricity_consumption",
    ]:
        if col in old.columns and col in rep.columns and len(old) == len(rep):
            row[f"{col}_max_abs_diff"] = float(
                np.nanmax(np.abs(old[col].to_numpy(float) - rep[col].to_numpy(float)))
            )
            row[f"{col}_mean_abs_diff"] = float(
                np.nanmean(np.abs(old[col].to_numpy(float) - rep[col].to_numpy(float)))
            )
    return row


def load_kpi_series(building: int, label: str) -> pd.Series:
    df = pd.read_csv(raw_dir(building, label) / "kpis.csv")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.groupby("cost_function")["value"].mean()


def compare_kpis() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        old = load_kpi_series(building, OLD_LABEL)
        rep = load_kpi_series(building, REP_LABEL)
        for key in sorted(set(old.index).intersection(rep.index)):
            ov = old.loc[key]
            rv = rep.loc[key]
            rows.append(
                {
                    "building": building,
                    "cost_function": key,
                    "old": ov,
                    "replicate": rv,
                    "diff": rv - ov,
                    "abs_diff": abs(rv - ov),
                }
            )
    return pd.DataFrame(rows)


def flatten_json(obj: object, prefix: str = "") -> dict:
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_json(value, child))
        return out
    return {prefix: obj}


def compare_metrics_json() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        old = flatten_json(json.loads(metrics_path(building, OLD_LABEL).read_text()))
        rep = flatten_json(json.loads(metrics_path(building, REP_LABEL).read_text()))
        for key in sorted(set(old).union(rep)):
            ov = old.get(key, None)
            rv = rep.get(key, None)
            numeric = isinstance(ov, (int, float)) and isinstance(rv, (int, float))
            if numeric:
                diff = float(rv) - float(ov)
                changed = not np.isclose(float(ov), float(rv), rtol=0.0, atol=1e-12, equal_nan=True)
            else:
                diff = np.nan
                changed = ov != rv
            rows.append(
                {
                    "building": building,
                    "key": key,
                    "old": ov,
                    "replicate": rv,
                    "numeric": numeric,
                    "diff": diff,
                    "changed": changed,
                }
            )
    return pd.DataFrame(rows)


def aggregate_rollouts(label: str) -> pd.DataFrame:
    frames = []
    for building in BUILDINGS:
        df = pd.read_csv(raw_dir(building, label) / "rollout.csv")
        df = df.copy()
        df["building"] = building
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    agg = (
        all_df.groupby("time_step", as_index=False)
        .agg(
            net_electricity_consumption=("net_electricity_consumption", "sum"),
            heating_electricity_consumption=("heating_electricity_consumption", "sum"),
            reward=("reward", "sum"),
        )
        .sort_values("time_step")
    )
    agg["hour"] = (agg["time_step"].astype(int) % 24).astype(int)
    return agg


def write_district_profile_comparison() -> pd.DataFrame:
    old = aggregate_rollouts(OLD_LABEL)
    rep = aggregate_rollouts(REP_LABEL)
    merged = old.merge(rep, on=["time_step", "hour"], suffixes=("_old", "_replicate"))
    for col in ["net_electricity_consumption", "heating_electricity_consumption", "reward"]:
        merged[f"{col}_diff"] = merged[f"{col}_replicate"] - merged[f"{col}_old"]

    hourly = (
        merged.groupby("hour", as_index=False)
        .agg(
            net_old=("net_electricity_consumption_old", "mean"),
            net_replicate=("net_electricity_consumption_replicate", "mean"),
            heat_old=("heating_electricity_consumption_old", "mean"),
            heat_replicate=("heating_electricity_consumption_replicate", "mean"),
            reward_old=("reward_old", "mean"),
            reward_replicate=("reward_replicate", "mean"),
        )
        .sort_values("hour")
    )
    hourly["net_diff"] = hourly["net_replicate"] - hourly["net_old"]
    hourly["heat_diff"] = hourly["heat_replicate"] - hourly["heat_old"]
    hourly["reward_diff"] = hourly["reward_replicate"] - hourly["reward_old"]

    plt.figure(figsize=(10, 4))
    plt.plot(hourly["hour"], hourly["net_old"], marker="o", label="old")
    plt.plot(hourly["hour"], hourly["net_replicate"], marker="o", linestyle="--", label="replicate")
    plt.title("Old RLMPC vs replicate - mean hourly district load")
    plt.xlabel("Hour")
    plt.ylabel("Mean district electricity [kWh]")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mean_hourly_district_load_old_vs_replicate.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 3.5))
    plt.plot(hourly["hour"], hourly["net_diff"], marker="o")
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Replicate minus old - mean hourly district load")
    plt.xlabel("Hour")
    plt.ylabel("Delta kWh")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mean_hourly_district_load_delta.png", dpi=160)
    plt.close()

    return hourly


def write_comfort_overlay() -> None:
    # Use the same four representative buildings that were used for quick sanity checks.
    buildings = [1, 8, 11, 22]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    for ax, building in zip(axes.flat, buildings):
        old = pd.read_csv(raw_dir(building, OLD_LABEL) / "rollout.csv")
        rep = pd.read_csv(raw_dir(building, REP_LABEL) / "rollout.csv")
        ax.plot(old["time_step"], old["indoor_temperature"], label="old Tin", linewidth=1.0)
        ax.plot(rep["time_step"], rep["indoor_temperature"], label="rep Tin", linewidth=0.9, linestyle="--")
        ax.plot(old["time_step"], old["effective_setpoint"], label="Tsp", linewidth=0.8, color="tab:orange")
        ax.fill_between(old["time_step"], old["tmin"], old["tmax"], color="0.85", alpha=0.45, label="comfort band")
        ax.set_title(f"Building {building}")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Old RLMPC vs replicate - comfort overlay")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comfort_overlay_representative_buildings.png", dpi=160)
    plt.close(fig)


def main() -> None:
    csv_rows = []
    for building in BUILDINGS:
        for filename in ["rollout.csv", "kpis.csv", "district_kpis.csv", "ppo_history.csv"]:
            csv_rows.append(compare_csv_file(building, filename))
    csv_diff = pd.DataFrame(csv_rows)
    csv_diff.to_csv(ANALYSIS_DIR / f"{REP_LABEL}_csv_diff_summary.csv", index=False)

    kpi_diff = compare_kpis()
    kpi_diff.to_csv(ANALYSIS_DIR / f"{REP_LABEL}_kpi_diff.csv", index=False)

    metric_diff = compare_metrics_json()
    metric_diff.to_csv(ANALYSIS_DIR / f"{REP_LABEL}_metrics_json_diff.csv", index=False)

    hourly = write_district_profile_comparison()
    hourly.to_csv(ANALYSIS_DIR / f"{REP_LABEL}_mean_hourly_district_profile_diff.csv", index=False)
    write_comfort_overlay()

    key_kpis = [
        "electricity_consumption_total",
        "discomfort_proportion",
        "discomfort_cold_proportion",
        "discomfort_hot_proportion",
        "daily_peak_average",
        "ramping_average",
    ]
    compact = (
        kpi_diff[kpi_diff["cost_function"].isin(key_kpis)]
        .groupby("cost_function")
        .agg(
            old_mean=("old", "mean"),
            replicate_mean=("replicate", "mean"),
            mean_diff=("diff", "mean"),
            max_abs_diff=("abs_diff", "max"),
        )
        .reset_index()
    )
    compact.to_csv(ANALYSIS_DIR / f"{REP_LABEL}_compact_kpi_diff.csv", index=False)

    rollout_rows = csv_diff[csv_diff["file"].eq("rollout.csv")]
    summary = {
        "old_label": OLD_LABEL,
        "replicate_label": REP_LABEL,
        "n_buildings": len(BUILDINGS),
        "all_rollouts_exact": bool(rollout_rows["exact_dataframe_equal"].all()),
        "max_rollout_numeric_diff": float(rollout_rows["max_abs_numeric_diff"].max()),
        "max_kpi_abs_diff": float(kpi_diff["abs_diff"].max()),
        "max_hourly_profile_abs_diff": float(hourly["net_diff"].abs().max()),
        "changed_numeric_metric_json_rows": int(
            metric_diff[metric_diff["numeric"] & metric_diff["changed"]].shape[0]
        ),
        "changed_non_numeric_metric_json_rows": int(
            metric_diff[(~metric_diff["numeric"]) & metric_diff["changed"]].shape[0]
        ),
    }
    pd.DataFrame([summary]).to_csv(ANALYSIS_DIR / f"{REP_LABEL}_replication_verdict.csv", index=False)

    print(json.dumps(summary, indent=2))
    print("\nCompact KPI diff:")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
