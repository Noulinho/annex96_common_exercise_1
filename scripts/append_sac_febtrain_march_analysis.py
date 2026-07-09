#!/usr/bin/env python3
"""Append SAC February-continuation / March-deployment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

import scripts.append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis as base
from src.occupants.metrics import kelvin_hours
from src.rewards import time_bias_setpoint_profile


PREFIX = "ce1_final_no_occ_D010702_sac_feb10_mar"
MARKER = f"## SAC February Reward Continuation -> March Deployment - `{PREFIX}`"

NEW_LABEL = "ce1_final_no_occ_D010702_sac_feb10_mar_a1"
PREV_BASE_LABEL = "ce1_final_no_occ_D010702_sac_safepi9_a1"
PREV_BAD_LABEL = "ce1_final_no_occ_D010702_sac_safepi9_bad10_a1"
MARCH_BASELINE_DIR = Path("results/raw/citylearn_baseline/no_occupant/march_baseline")

TRAIN_START = 744
TRAIN_END = 1415
EVAL_START = 1416
EVAL_END = 2159

FIG_DIR = Path("figures") / PREFIX
ANALYSIS_DIR = Path("analysis")
NOTEBOOKS = [
    ANALYSIS_DIR / "08_sac_results.ipynb",
    ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
]


def raw_dir(building_id: str, label: str) -> Path:
    return Path("results/raw/sac_merlin") / building_id / "no_occupant" / label


def previous_final_label(building_id: str) -> str:
    bad = raw_dir(building_id, PREV_BAD_LABEL) / "kpis.csv"
    return PREV_BAD_LABEL if bad.exists() else PREV_BASE_LABEL


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


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


def rollout(building_id: str, label: str = NEW_LABEL) -> pd.DataFrame:
    return read_csv(raw_dir(building_id, label) / "rollout.csv")


def setpoint(data: pd.DataFrame) -> np.ndarray:
    return base._setpoint(data)


def time_step(data: pd.DataFrame) -> np.ndarray:
    return base._time_step(data)


def load(data: pd.DataFrame) -> np.ndarray:
    return base._load(data)


def building_kpis(label_resolver=previous_final_label, *, period: str) -> pd.DataFrame:
    rows = []
    for building in base.BUILDINGS:
        building_id = building["result_id"]
        label = label_resolver(building_id) if callable(label_resolver) else str(label_resolver)
        data = rollout(building_id, label)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        values = kpi_values(raw_dir(building_id, label) / "kpis.csv")
        rows.append(
            {
                "period": period,
                "controller": "SAC",
                "label": label,
                "building": building["display"],
                "building_id": building_id,
                "electricity_consumption_total": values.get("electricity_consumption_total", np.nan),
                "discomfort_%": 100.0 * values.get("discomfort_proportion", np.nan),
                "cold_%": 100.0 * values.get("discomfort_cold_proportion", np.nan),
                "hot_%": 100.0 * values.get("discomfort_hot_proportion", np.nan),
                "kelvin_hours": kelvin_hours(tin, lower=20.0, upper=24.0),
                "daily_peak_average": values.get("daily_peak_average", np.nan),
                "ramping_average": values.get("ramping_average", np.nan),
                "tin_max": float(np.nanmax(tin)),
                "tin_min": float(np.nanmin(tin)),
            }
        )
    return pd.DataFrame(rows)


def mean_kpis(data: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "ramping_average",
        "tin_max",
        "tin_min",
    ]
    return data.groupby(["period", "controller"], as_index=False)[cols].mean(numeric_only=True)


def district_frame_sac() -> pd.DataFrame:
    series = []
    t = None
    for building in base.BUILDINGS:
        data = rollout(building["result_id"], NEW_LABEL)
        values = load(data)
        if t is None:
            t = time_step(data)
        n = min(len(t), len(values))
        t = t[:n]
        series.append(values[:n])
    n = min(len(values) for values in series)
    baseline = district_frame_baseline()
    return pd.DataFrame(
        {
            "time_step": np.asarray(t[:n], dtype=int),
            "load": np.sum([values[:n] for values in series], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "controller": "SAC Feb-train -> March",
        }
    )


def district_frame_baseline() -> pd.DataFrame:
    data = read_csv(MARCH_BASELINE_DIR / "district_rollout.csv")
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0),
            "controller": "CityLearn March baseline",
        }
    )


def district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0]["load"].to_numpy(dtype=float)
    base_total = max(float(np.sum(baseline)), 1e-9)
    base_peak = max(float(np.max(baseline)), 1e-9)
    base_ramp = max(float(np.sum(np.abs(np.diff(baseline)))), 1e-9)
    rows = []
    for frame in frames:
        load_values = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        n = min(len(load_values), len(target))
        load_values, target = load_values[:n], target[:n]
        hourly = pd.DataFrame({"hour": np.arange(n) % 24, "load": load_values}).groupby("hour")["load"].mean()
        ramp = float(np.sum(np.abs(np.diff(load_values)))) if n > 1 else 0.0
        rows.append(
            {
                "controller": frame["controller"].iloc[0],
                "total_kwh": float(np.sum(load_values)),
                "total_vs_citylearn_march": float(np.sum(load_values) / base_total),
                "peak_kwh": float(np.max(load_values)),
                "peak_vs_citylearn_march": float(np.max(load_values) / base_peak),
                "daily_profile_peak_kwh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kwh": ramp,
                "ramping_vs_citylearn_march": float(ramp / base_ramp),
                "target_rmse_kwh": float(np.sqrt(np.mean((load_values - target) ** 2))),
                "above_target_kwh": float(np.sum(np.maximum(load_values - target, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def term_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    hourly_rows = []
    for building in base.BUILDINGS:
        data = rollout(building["result_id"], NEW_LABEL)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = setpoint(data)
        steps = time_step(data)
        n = min(len(tin), len(tsp), len(steps))
        tin, tsp, steps = tin[:n], tsp[:n], steps[:n]
        bias = np.asarray(
            time_bias_setpoint_profile(
                simulation_start=EVAL_START,
                simulation_end=EVAL_START + n - 1,
                amplitude=1.0,
            ),
            dtype=float,
        )[:n]
        tref = tsp + bias
        cold_guard = np.maximum(tsp - 2.0 - tin, 0.0)
        hot_guard = np.maximum(tin - tsp - 2.0, 0.0)
        track = 0.1 * ((tin - tref) / 2.0) ** 2
        cold = 0.7 * (cold_guard / 1.5) ** 2
        hot = 0.2 * (hot_guard / 2.0) ** 2
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
                    "building": building["display"],
                    "building_id": building["result_id"],
                    "term": term,
                    "mean": float(np.nanmean(values)),
                    "max": float(np.nanmax(values)),
                }
            )
        rows.append(
            {
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
                    "building": building["display"],
                    "hour": (steps + EVAL_START) % 24,
                    "tin_minus_tsp": tin - tsp,
                    "guard_violation": guard.astype(float),
                    "stage_cost": stage,
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rewards = []
    losses = []
    for building in base.BUILDINGS:
        case_dir = raw_dir(building["result_id"], NEW_LABEL)
        reward_path = case_dir / "online_training_rollout.csv"
        if reward_path.exists():
            data = read_csv(reward_path)
            data["building"] = building["display"]
            data["building_id"] = building["result_id"]
            rewards.append(data)
        loss_path = case_dir / "sac_training_losses.csv"
        if loss_path.exists():
            data = read_csv(loss_path)
            data["building"] = building["display"]
            data["building_id"] = building["result_id"]
            losses.append(data)
    return (
        pd.concat(rewards, ignore_index=True) if rewards else pd.DataFrame(),
        pd.concat(losses, ignore_index=True) if losses else pd.DataFrame(),
    )


def reward_summary(rewards: pd.DataFrame) -> pd.DataFrame:
    if rewards.empty:
        return pd.DataFrame()
    data = rewards[rewards["phase"].astype(str).str.contains("sac", case=False, na=False)].copy()
    data["raw_reward_plot"] = pd.to_numeric(data.get("raw_reward", data["reward"]), errors="coerce")
    data["reward_plot"] = pd.to_numeric(data["reward"], errors="coerce")
    reset = data.get("temperature_reset_applied", False)
    if isinstance(reset, pd.Series):
        if reset.dtype == object:
            data["reset_flag"] = reset.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        else:
            data["reset_flag"] = pd.to_numeric(reset, errors="coerce").fillna(0.0)
    else:
        data["reset_flag"] = 0.0
    return (
        data.groupby("episode", as_index=False)
        .agg(
            raw_reward_mean=("raw_reward_plot", "mean"),
            raw_reward_min=("raw_reward_plot", "min"),
            raw_reward_max=("raw_reward_plot", "max"),
            reward_mean=("reward_plot", "mean"),
            reset_fraction=("reset_flag", "mean"),
            action_mean=("action", "mean"),
        )
        .sort_values("episode")
    )


def loss_summary(losses: pd.DataFrame) -> pd.DataFrame:
    if losses.empty:
        return pd.DataFrame()
    cols = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"] if c in losses.columns]
    out = losses.groupby("episode", as_index=False)[cols].agg(["mean", "max"])
    out.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in out.columns.to_flat_index()]
    return out


def reset_by_building(rewards: pd.DataFrame) -> pd.DataFrame:
    if rewards.empty:
        return pd.DataFrame()
    data = rewards.copy()
    reset = data.get("temperature_reset_applied", False)
    if isinstance(reset, pd.Series):
        if reset.dtype == object:
            data["reset_flag"] = reset.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        else:
            data["reset_flag"] = pd.to_numeric(reset, errors="coerce").fillna(0.0)
    else:
        data["reset_flag"] = 0.0
    return (
        data.groupby(["building", "building_id"], as_index=False)
        .agg(reset_fraction=("reset_flag", "mean"), action_mean=("action", "mean"), reward_mean=("reward", "mean"))
        .assign(reset_pct=lambda x: 100.0 * x["reset_fraction"])
        .sort_values("reset_pct", ascending=False)
    )


def comparison_table(prev: pd.DataFrame, new: pd.DataFrame, resets: pd.DataFrame) -> pd.DataFrame:
    prev_cols = prev[["building", "building_id", "discomfort_%", "cold_%", "hot_%", "electricity_consumption_total"]].rename(
        columns={
            "discomfort_%": "prev_feb_disc_%",
            "cold_%": "prev_feb_cold_%",
            "hot_%": "prev_feb_hot_%",
            "electricity_consumption_total": "prev_feb_energy",
        }
    )
    new_cols = new[
        [
            "building",
            "building_id",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "electricity_consumption_total",
            "tin_max",
            "tin_min",
        ]
    ].rename(
        columns={
            "discomfort_%": "march_disc_%",
            "cold_%": "march_cold_%",
            "hot_%": "march_hot_%",
            "electricity_consumption_total": "march_energy",
        }
    )
    out = prev_cols.merge(new_cols, on=["building", "building_id"], how="outer")
    if not resets.empty:
        out = out.merge(resets[["building_id", "reset_pct"]], on="building_id", how="left")
    out["disc_delta_march_minus_prev_feb"] = out["march_disc_%"] - out["prev_feb_disc_%"]
    return out.sort_values("march_disc_%", ascending=False)


def markdown_table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    return base._markdown_table(df, digits=digits, max_rows=max_rows)


def append_once(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if MARKER in str(cell.get("source", "")):
            notebook.cells[index:index + len(cells)] = cells
            nbf.write(notebook, path)
            print(f"Updated {MARKER} in {path}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {MARKER} to {path}")


def image(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.as_posix()})")


def make_comfort_grid() -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, base.BUILDINGS):
        data = rollout(building["result_id"], NEW_LABEL)
        t = time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = setpoint(data)
        n = min(len(t), len(tin), len(tsp))
        bias = np.asarray(
            time_bias_setpoint_profile(
                simulation_start=EVAL_START,
                simulation_end=EVAL_START + n - 1,
                amplitude=1.0,
            ),
            dtype=float,
        )[:n]
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.65, ls="--", color="black", label="Tsp")
        ax.plot(t[:n], tsp[:n] + bias[:n], lw=0.65, ls=":", color="tab:orange", label="Tsp+bias")
        ax.fill_between(t[:n], tsp[:n] - 2.0, tsp[:n] + 2.0, color="gray", alpha=0.13)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("SAC February reward continuation | March comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "sac_febtrain_march_comfort_all25.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=name)
    ax.plot(frames[0]["time_step"], frames[0]["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("SAC February-trained deployment - March district load")
    ax.set_xlabel("March time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    load_path = FIG_DIR / "sac_febtrain_march_district_load_timeseries.png"
    fig.savefig(load_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = frames[0].assign(hour=frames[0]["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.4, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Mean hourly district load - March")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    daily_path = FIG_DIR / "sac_febtrain_march_mean_hourly_profile.png"
    fig.savefig(daily_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


def make_reward_plots(summary: pd.DataFrame) -> tuple[Path, Path]:
    reward_path = FIG_DIR / "sac_feb_training_reward_mean.png"
    reset_path = FIG_DIR / "sac_feb_training_reset_fraction.png"
    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No reward rows", ha="center", va="center")
        ax.axis("off")
        fig.savefig(reward_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        fig.savefig(reset_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return reward_path, reset_path
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(summary["episode"], summary["raw_reward_mean"], marker="o", lw=1.5, label="Raw reward mean per step")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("SAC February continuation reward - per-step mean")
    ax.set_xlabel("Training episode label")
    ax.set_ylabel("Mean reward per step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reward_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(summary["episode"], 100.0 * summary["reset_fraction"], marker="o", lw=1.4)
    ax.set_title("SAC February continuation temperature-reset rate")
    ax.set_xlabel("Training episode label")
    ax.set_ylabel("Reset-applied steps [%]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(reset_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return reward_path, reset_path


def make_loss_plot(summary: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_feb_training_losses.png"
    if summary.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No loss rows", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    cols = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha"] if f"{c}_mean" in summary.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.7 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(summary["episode"], summary[f"{col}_mean"], marker="o", lw=1.2, label="mean")
        max_col = f"{col}_max"
        if max_col in summary.columns:
            ax.plot(summary["episode"], summary[max_col], marker="o", lw=0.9, ls="--", label="max")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Training episode label")
    fig.suptitle("SAC February continuation loss diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_delta_plot(compare: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_prev_feb_vs_febtrain_march_discomfort.png"
    plot = compare.sort_values("march_disc_%", ascending=False)
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(15, 5))
    width = 0.38
    ax.bar(x - width / 2, plot["prev_feb_disc_%"], width, label="Previous final SAC on February")
    ax.bar(x + width / 2, plot["march_disc_%"], width, label="Feb-trained SAC on March")
    ax.axhline(10.0, color="black", lw=0.9, ls="--", label="10% discomfort")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["building"], rotation=45, ha="right")
    ax.set_ylabel("Discomfort [%]")
    ax.set_title("SAC comfort tail before vs after February continuation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    prev_kpis = building_kpis(period="Previous final SAC, February")
    new_kpis = building_kpis(NEW_LABEL, period="Feb-trained SAC, March")
    all_kpis = pd.concat([prev_kpis, new_kpis], ignore_index=True)
    mean = mean_kpis(all_kpis)

    baseline = district_frame_baseline()
    sac = district_frame_sac()
    dist = district_kpis([baseline, sac])
    hourly_profile = (
        pd.concat([baseline, sac], ignore_index=True)
        .assign(hour=lambda x: x["time_step"] % 24)
        .groupby(["controller", "hour"], as_index=False)["load"]
        .mean()
    )

    terms_by_building, hourly_terms = term_diagnostics()
    term_summary = (
        terms_by_building.groupby("term", as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values("term")
    )

    rewards, losses = training_frames()
    rewards_summary = reward_summary(rewards)
    losses_summary = loss_summary(losses)
    resets = reset_by_building(rewards)
    compare = comparison_table(prev_kpis, new_kpis, resets)
    still_bad = compare[compare["march_disc_%"] > 10.0].copy()

    prefix = PREFIX
    all_kpis.to_csv(ANALYSIS_DIR / f"{prefix}_building_kpis_prev_feb_and_march.csv", index=False)
    new_kpis.to_csv(ANALYSIS_DIR / f"{prefix}_march_building_kpis_all25.csv", index=False)
    mean.to_csv(ANALYSIS_DIR / f"{prefix}_mean_building_kpis.csv", index=False)
    dist.to_csv(ANALYSIS_DIR / f"{prefix}_district_kpis.csv", index=False)
    hourly_profile.to_csv(ANALYSIS_DIR / f"{prefix}_district_mean_hourly_profile.csv", index=False)
    compare.to_csv(ANALYSIS_DIR / f"{prefix}_prev_feb_vs_march_building_delta.csv", index=False)
    terms_by_building.to_csv(ANALYSIS_DIR / f"{prefix}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{prefix}_term_diagnostics.csv", index=False)
    hourly_terms.to_csv(ANALYSIS_DIR / f"{prefix}_hourly_terms.csv", index=False)
    rewards.to_csv(ANALYSIS_DIR / f"{prefix}_online_training_rewards_raw.csv", index=False)
    losses.to_csv(ANALYSIS_DIR / f"{prefix}_sac_training_losses_raw.csv", index=False)
    rewards_summary.to_csv(ANALYSIS_DIR / f"{prefix}_online_reward_summary.csv", index=False)
    losses_summary.to_csv(ANALYSIS_DIR / f"{prefix}_loss_summary.csv", index=False)
    resets.to_csv(ANALYSIS_DIR / f"{prefix}_training_reset_by_building.csv", index=False)

    comfort_path = make_comfort_grid()
    load_path, daily_path = make_district_plots([baseline, sac])
    reward_path, reset_path = make_reward_plots(rewards_summary)
    loss_path = make_loss_plot(losses_summary)
    delta_path = make_delta_plot(compare)

    mean_display = mean.rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
        }
    )
    dist_display = dist.rename(
        columns={
            "total_vs_citylearn_march": "total_vs_CL_Mar",
            "peak_vs_citylearn_march": "peak_vs_CL_Mar",
            "ramping_vs_citylearn_march": "ramp_vs_CL_Mar",
            "daily_profile_peak_kwh": "daily_peak_kWh",
            "daily_profile_peak_hour": "daily_peak_hr",
            "target_rmse_kwh": "target_RMSE",
        }
    )
    new_display = new_kpis[
        [
            "building",
            "electricity_consumption_total",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "kelvin_hours",
            "daily_peak_average",
            "ramping_average",
            "tin_max",
            "tin_min",
        ]
    ].rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
        }
    )
    compare_display = compare[
        [
            "building",
            "prev_feb_disc_%",
            "march_disc_%",
            "disc_delta_march_minus_prev_feb",
            "march_cold_%",
            "march_hot_%",
            "reset_pct",
            "march_energy",
            "tin_max",
        ]
    ].rename(columns={"disc_delta_march_minus_prev_feb": "delta_disc", "reset_pct": "feb_reset_%"})
    still_bad_display = still_bad[
        ["building", "march_disc_%", "march_cold_%", "march_hot_%", "reset_pct", "march_energy", "tin_max"]
    ].rename(columns={"reset_pct": "feb_reset_%"})
    reward_display = rewards_summary[["episode", "raw_reward_mean", "reward_mean", "reset_fraction", "action_mean"]].copy()
    if not reward_display.empty:
        reward_display["reset_%"] = 100.0 * reward_display.pop("reset_fraction")
    loss_display = losses_summary.tail(10).copy()

    overview = f"""{MARKER}

SAC-only continuation test: start from the final hybrid SAC policies, train for 10 more SAC reward episodes on February, then deploy in March.

Resume sources:

- Buildings with the previous bad-building continuation resume from `{PREV_BAD_LABEL}`.
- The other buildings resume from `{PREV_BASE_LABEL}`.
- New continuation/deployment label: `{NEW_LABEL}`.

Training/evaluation windows:

- February training: time steps `{TRAIN_START}-{TRAIN_END}`.
- March deployment: time steps `{EVAL_START}-{EVAL_END}`.
- The run was launched outside the sandbox so SAC used CUDA.

Reward/cost:

`D_k = 0.1*((Tin-(Tsp+bias))/2)^2 + 0.7*(max(Tsp-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tsp-2,0)/2)^2`

`reward = -D_k`. No GridStress, no electricity term, no peak/ramp term.

Mean building KPIs:

{markdown_table(mean_display, digits=3)}

March district aggregate KPIs:

{markdown_table(dist_display, digits=3)}

All-building March KPIs:

{markdown_table(new_display, digits=3)}

Previous final February SAC vs February-trained March SAC:

{markdown_table(compare_display, digits=2)}

Buildings still above 10% March discomfort:

{markdown_table(still_bad_display, digits=2)}

March weighted term diagnostics:

{markdown_table(term_summary, digits=4)}
"""
    if not reward_display.empty:
        overview += f"""

February continuation reward summary:

{markdown_table(reward_display, digits=4)}
"""
    if not loss_display.empty:
        overview += f"""

SAC loss summary, final training episodes:

{markdown_table(loss_display, digits=4)}
"""
    overview += f"""

Saved compact tables:

- `analysis/{prefix}_march_building_kpis_all25.csv`
- `analysis/{prefix}_mean_building_kpis.csv`
- `analysis/{prefix}_district_kpis.csv`
- `analysis/{prefix}_prev_feb_vs_march_building_delta.csv`
- `analysis/{prefix}_online_reward_summary.csv`
- `analysis/{prefix}_loss_summary.csv`
- `analysis/{prefix}_training_reset_by_building.csv`
"""

    cells = [
        nbf.v4.new_markdown_cell(overview),
        image(comfort_path, "SAC February-trained March plot_comfort"),
        image(load_path, "SAC March district load"),
        image(daily_path, "SAC March mean hourly district load profile"),
        image(delta_path, "SAC previous February vs March discomfort"),
        image(reward_path, "SAC February training reward"),
        image(reset_path, "SAC February training reset fraction"),
        image(loss_path, "SAC February training losses"),
    ]
    for notebook in NOTEBOOKS:
        if notebook.exists():
            append_once(notebook, cells)
    print(f"Done: {PREFIX}")


if __name__ == "__main__":
    main()
