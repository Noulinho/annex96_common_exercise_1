#!/usr/bin/env python
"""Append SAC Jan+Feb training / March deployment analysis to 08_sac_results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries


PREFIX = "ce1_occ_D010702_sac_janfeb20_mar"
MARKER = f"## CE1 SAC Jan+Feb20 -> March - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOK = ANALYSIS / "08_sac_results.ipynb"
MARCH_BASELINE_DIR = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "march_baseline"

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]

RUNS = [
    {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "controller": "old March SAC",
        "display": "old SAC no occupant",
        "label": "ce1_final_no_occ_D010702_sac_feb10_mar_a1",
        "optional": True,
    },
    {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "controller": "replay-tail March SAC",
        "display": "replay-tail SAC no occupant",
        "label": "ce1_occ_D010702_sac_noreset_replaytail1024_feb10_mar_no_occupant",
        "optional": True,
    },
    {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "controller": "Jan+Feb20 SAC",
        "display": "Jan+Feb20 SAC no occupant",
        "label": f"{PREFIX}_no_occupant",
        "optional": False,
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "controller": "old March SAC",
        "display": "old SAC B-spline Tdyn",
        "label": "ce1_occ_D010702_sac_feb10_mar_bspline_tdyn",
        "optional": True,
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "controller": "replay-tail March SAC",
        "display": "replay-tail SAC B-spline Tdyn",
        "label": "ce1_occ_D010702_sac_noreset_replaytail1024_feb10_mar_bspline_tdyn",
        "optional": True,
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "controller": "Jan+Feb20 SAC",
        "display": "Jan+Feb20 SAC B-spline Tdyn",
        "label": f"{PREFIX}_bspline_tdyn",
        "optional": False,
    },
]

NEW_RUNS = [run for run in RUNS if run["controller"] == "Jan+Feb20 SAC"]


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def raw_dir(building: dict, run: dict) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building["result_id"] / run["scenario"] / run["label"]


def summary_path(building: dict, run: dict) -> Path:
    return RESULTS / "summaries" / "sac_merlin" / building["result_id"] / run["scenario"] / "summary.csv"


def run_exists(run: dict) -> bool:
    return all((raw_dir(building, run) / "rollout.csv").exists() for building in BUILDINGS)


def active_runs() -> list[dict]:
    runs = []
    missing_required = []
    for run in RUNS:
        exists = run_exists(run)
        if exists:
            runs.append(run)
        elif not run.get("optional", False):
            missing_required.append(run["label"])
    if missing_required:
        raise FileNotFoundError(f"Missing required SAC artifacts for labels: {missing_required}")
    return runs


def rollout(building: dict, run: dict) -> pd.DataFrame:
    return pd.read_csv(raw_dir(building, run) / "rollout.csv")


def training_rollout(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "online_training_rollout.csv"
    return pd.read_csv(path) if path.exists() else None


def losses(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "sac_training_losses.csv"
    return pd.read_csv(path) if path.exists() else None


def summary_row(building: dict, run: dict) -> pd.Series | None:
    path = summary_path(building, run)
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if "case" not in data.columns:
        return None
    rows = data[data["case"].astype(str) == run["label"]]
    if rows.empty:
        return None
    return rows.iloc[-1]


def num(row: pd.Series | None, key: str, default=np.nan) -> float:
    if row is None or key not in row or pd.isna(row[key]):
        return float(default)
    return float(row[key])


def kpi_values(building: dict, run: dict) -> dict[str, float]:
    path = raw_dir(building, run) / "kpis.csv"
    data = pd.read_csv(path)
    if "level" in data.columns:
        district = data[data["level"].astype(str) == "district"].copy()
        if not district.empty:
            data = district
    out: dict[str, float] = {}
    for _, row in data.iterrows():
        value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.notna(value):
            out[str(row["cost_function"])] = float(value)
    return out


def time_axis(frame: pd.DataFrame) -> np.ndarray:
    if "time_step" in frame.columns:
        return pd.to_numeric(frame["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(frame), dtype=int)


def load_series(frame: pd.DataFrame) -> np.ndarray:
    for column in ("net_electricity_consumption", "electricity_consumption", "load"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raise ValueError("Rollout has no load column.")


def active_setpoint(frame: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "tracking_setpoint", "baseline_setpoint", "scheduled_setpoint", "tmin"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def kelvin_hours_from_rollout(frame: pd.DataFrame) -> float:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    tsp = active_setpoint(frame)
    n = min(len(tin), len(tsp))
    if n == 0:
        return np.nan
    cold = np.maximum(tsp[:n] - 2.0 - tin[:n], 0.0)
    hot = np.maximum(tin[:n] - tsp[:n] - 2.0, 0.0)
    return float(np.nanmean(cold + hot))


def override_metrics(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    feedback = pd.to_numeric(frame.get("occupant_feedback", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0.0)
    values = feedback.to_numpy(dtype=float)
    active = np.abs(values) > 1e-12
    previous = np.r_[False, active[:-1]]
    starts = int(np.count_nonzero(active & ~previous))
    active_steps = int(np.count_nonzero(active))
    positive = int(np.count_nonzero(values > 1e-12))
    negative = int(np.count_nonzero(values < -1e-12))
    return starts, active_steps, positive, negative


def building_kpis(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for building in BUILDINGS:
            frame = rollout(building, run)
            kpis = kpi_values(building, run)
            summary = summary_row(building, run)
            tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            reward = pd.to_numeric(frame.get("reward", pd.Series(dtype=float)), errors="coerce")
            action = pd.to_numeric(frame.get("action", pd.Series(dtype=float)), errors="coerce")
            starts, active_steps, positive, negative = override_metrics(frame)
            rows.append(
                {
                    "mode": run["mode"],
                    "controller": run["controller"],
                    "label": run["label"],
                    "building": building["display"],
                    "building_id": building["result_id"],
                    "occupant": "none" if run["mode"] == "no_occupant" else occupant_for_building(building),
                    "energy": kpis.get("electricity_consumption_total", num(summary, "electricity_consumption_total")),
                    "disc_%": 100.0 * kpis.get("discomfort_proportion", num(summary, "discomfort_proportion")),
                    "cold_%": 100.0 * kpis.get("discomfort_cold_proportion", num(summary, "discomfort_cold_proportion")),
                    "hot_%": 100.0 * kpis.get("discomfort_hot_proportion", num(summary, "discomfort_hot_proportion")),
                    "KH": num(summary, "kelvin_hours", kelvin_hours_from_rollout(frame)),
                    "daily_peak": kpis.get("daily_peak_average", num(summary, "daily_peak_average")),
                    "ramping": kpis.get("ramping_average", np.nan),
                    "Tin_min": float(np.nanmin(tin)) if len(tin) else np.nan,
                    "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
                    "reward_sum": float(reward.fillna(0.0).sum()) if not reward.empty else num(summary, "reward_total"),
                    "reward_mean": float(reward.mean()) if not reward.empty else np.nan,
                    "action_mean": float(action.mean()) if not action.empty else np.nan,
                    "override_starts": starts if run["mode"] == "bspline_tdyn" else 0,
                    "override_active_steps": active_steps if run["mode"] == "bspline_tdyn" else 0,
                    "positive_feedback_steps": positive if run["mode"] == "bspline_tdyn" else 0,
                    "negative_feedback_steps": negative if run["mode"] == "bspline_tdyn" else 0,
                }
            )
    return pd.DataFrame(rows)


def reward_by_episode(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for building in BUILDINGS:
            data = training_rollout(building, run)
            if data is None or data.empty or "episode" not in data.columns:
                continue
            raw_col = "raw_reward" if "raw_reward" in data.columns else "reward"
            data = data.copy()
            data["raw_reward_plot"] = pd.to_numeric(data[raw_col], errors="coerce")
            data["reward_plot"] = pd.to_numeric(data.get("reward", data[raw_col]), errors="coerce")
            data["action_plot"] = pd.to_numeric(data.get("action", pd.Series(dtype=float)), errors="coerce")
            grouped = data.groupby("episode", as_index=False).agg(
                raw_reward_mean=("raw_reward_plot", "mean"),
                raw_reward_min=("raw_reward_plot", "min"),
                raw_reward_max=("raw_reward_plot", "max"),
                reward_mean=("reward_plot", "mean"),
                action_mean=("action_plot", "mean"),
            )
            grouped["mode"] = run["mode"]
            grouped["controller"] = run["controller"]
            grouped["building"] = building["display"]
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def losses_by_episode(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for building in BUILDINGS:
            data = losses(building, run)
            if data is None or data.empty or "episode" not in data.columns:
                continue
            keep = [c for c in ("soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha", "alpha_loss") if c in data.columns]
            if not keep:
                continue
            grouped = data.groupby("episode", as_index=False)[keep].mean(numeric_only=True)
            grouped["mode"] = run["mode"]
            grouped["controller"] = run["controller"]
            grouped["building"] = building["display"]
            frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def baseline_district() -> pd.DataFrame:
    data = pd.read_csv(MARCH_BASELINE_DIR / "district_rollout.csv")
    target = data["district_target"] if "district_target" in data.columns else 0.0
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "target": pd.to_numeric(target, errors="coerce").fillna(0.0),
            "mode": "baseline",
            "controller": "CityLearn March baseline",
        }
    )


def district_frame(run: dict) -> pd.DataFrame:
    series = []
    time = None
    for building in BUILDINGS:
        frame = rollout(building, run)
        load = load_series(frame)
        t = time_axis(frame)
        n = min(len(t), len(load))
        if time is None:
            time = t[:n]
        series.append(load[:n])
    n = min(len(s) for s in series)
    baseline = baseline_district()
    return pd.DataFrame(
        {
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([s[:n] for s in series], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "mode": run["mode"],
            "controller": run["controller"],
        }
    )


def district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0]["load"].to_numpy(dtype=float)
    base_total = max(float(np.sum(baseline)), 1e-9)
    base_peak = max(float(np.max(baseline)), 1e-9)
    base_ramp = max(float(np.sum(np.abs(np.diff(baseline)))), 1e-9)
    rows = []
    for frame in frames:
        load = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        n = min(len(load), len(target))
        load, target = load[:n], target[:n]
        hourly = pd.DataFrame({"hour": np.arange(n) % 24, "load": load}).groupby("hour")["load"].mean()
        ramp = float(np.sum(np.abs(np.diff(load)))) if n > 1 else 0.0
        rows.append(
            {
                "mode": frame["mode"].iloc[0],
                "controller": frame["controller"].iloc[0],
                "total_MWh": float(np.sum(load) / 1000.0),
                "total_vs_CL": float(np.sum(load) / base_total),
                "peak_kWh": float(np.max(load)),
                "peak_vs_CL": float(np.max(load) / base_peak),
                "daily_profile_peak_kWh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kWh": ramp,
                "ramping_vs_CL": float(ramp / base_ramp),
                "target_RMSE": float(np.sqrt(np.mean((load - target) ** 2))),
                "above_target_kWh": float(np.sum(np.maximum(load - target, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def make_comfort_grid(run: dict) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        frame = rollout(building, run)
        t = time_axis(frame)
        tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = active_setpoint(frame)
        n = min(len(t), len(tin), len(tsp))
        ax.plot(t[:n], tin[:n], lw=0.7, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.7, color="tab:orange", label="setpoint")
        if run["mode"] == "bspline_tdyn" and "tdyn" in frame.columns:
            tdyn = pd.to_numeric(frame["tdyn"], errors="coerce").to_numpy(dtype=float)
            m = min(n, len(tdyn))
            ax.plot(t[:m], tdyn[:m], lw=0.9, color="black", label="Tdyn")
        occ = "none" if run["mode"] == "no_occupant" else occupant_for_building(building)
        ax.set_title(f"{building['display']} | {occ}", fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | March deterministic comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['mode']}_janfeb20_sac_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    controlled = [f for f in frames if f["mode"].iloc[0] != "baseline"]
    baseline = frames[0]

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for ax, mode, title in zip(axes, ("no_occupant", "bspline_tdyn"), ("No occupant", "B-spline Tdyn")):
        ax.plot(baseline["time_step"], baseline["load"], lw=0.8, color="0.65", label="CityLearn baseline")
        for frame in controlled:
            if frame["mode"].iloc[0] == mode:
                ax.plot(frame["time_step"], frame["load"], lw=0.9, label=frame["controller"].iloc[0])
        ax.set_title(f"{title} - March district load")
        ax.set_ylabel("District electricity [kWh]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("March time step")
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries_march.png"
    fig.savefig(load_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    daily_frames = []
    for frame in frames:
        tmp = frame.copy()
        tmp["hour"] = np.arange(len(tmp)) % 24
        daily_frames.append(tmp.groupby(["mode", "controller", "hour"], as_index=False)["load"].mean())
    daily = pd.concat(daily_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, mode, title in zip(axes, ("no_occupant", "bspline_tdyn"), ("No occupant", "B-spline Tdyn")):
        base = daily[daily["mode"] == "baseline"]
        ax.plot(base["hour"], base["load"], marker="o", lw=1.0, color="0.65", label="CityLearn baseline")
        for name, group in daily[daily["mode"] == mode].groupby("controller", sort=False):
            ax.plot(group["hour"], group["load"], marker="o", lw=1.3, label=name)
        ax.set_title(f"{title} - mean hourly district load")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(24))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean district electricity [kWh]")
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_hourly_march.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


def make_reward_plot(rewards: pd.DataFrame) -> Path | None:
    if rewards.empty:
        return None
    data = rewards[rewards["controller"] == "Jan+Feb20 SAC"].copy()
    grouped = data.groupby(["mode", "episode"], as_index=False).agg(
        raw_reward_mean=("raw_reward_mean", "mean"),
        raw_reward_p10=("raw_reward_mean", lambda x: float(np.nanpercentile(x, 10))),
        raw_reward_p90=("raw_reward_mean", lambda x: float(np.nanpercentile(x, 90))),
        action_mean=("action_mean", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, mode, title in zip(axes, ("no_occupant", "bspline_tdyn"), ("No occupant", "B-spline Tdyn")):
        group = grouped[grouped["mode"] == mode]
        if group.empty:
            continue
        x = group["episode"].to_numpy(dtype=float)
        mean = group["raw_reward_mean"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", lw=1.4, label="mean")
        ax.fill_between(
            x,
            group["raw_reward_p10"].to_numpy(dtype=float),
            group["raw_reward_p90"].to_numpy(dtype=float),
            alpha=0.15,
            label="p10-p90 buildings",
        )
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_title(f"{title} Jan+Feb training reward")
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean raw reward per step")
    fig.tight_layout()
    path = FIG_DIR / "sac_janfeb_training_reward.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_loss_plot(loss_data: pd.DataFrame) -> Path | None:
    if loss_data.empty or "policy_loss" not in loss_data.columns:
        return None
    data = loss_data[loss_data["controller"] == "Jan+Feb20 SAC"].copy()
    grouped = data.groupby(["mode", "episode"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=False)
    panels = [
        ("no_occupant", "soft_q1_loss"),
        ("no_occupant", "policy_loss"),
        ("bspline_tdyn", "soft_q1_loss"),
        ("bspline_tdyn", "policy_loss"),
    ]
    for ax, (mode, metric) in zip(axes.ravel(), panels):
        group = grouped[grouped["mode"] == mode]
        if metric in group:
            ax.plot(group["episode"], group[metric], marker="o", lw=1.1)
        ax.set_title(f"{mode} - {metric}")
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "sac_janfeb_training_losses.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def mpc_tdyn_override_rows() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        occ = occupant_for_building(building)
        path = (
            RESULTS
            / "raw"
            / "mpc"
            / building["result_id"]
            / "occupant_present"
            / f"ce1_occ_D010702_mpc_mar_{occ}_bspline_tdyn"
            / "rollout.csv"
        )
        if not path.exists():
            continue
        starts, active_steps, positive, negative = override_metrics(pd.read_csv(path))
        rows.append(
            {
                "controller": "MPC March Tdyn",
                "building": building["display"],
                "occupant": occ,
                "override_starts": starts,
                "override_active_steps": active_steps,
                "positive_feedback_steps": positive,
                "negative_feedback_steps": negative,
            }
        )
    return pd.DataFrame(rows)


def make_override_plot(overrides: pd.DataFrame) -> Path | None:
    if overrides.empty:
        return None
    totals = (
        overrides.groupby("controller", as_index=False)[
            ["override_starts", "positive_feedback_steps", "negative_feedback_steps"]
        ]
        .sum()
        .sort_values("override_starts")
    )
    x = np.arange(len(totals))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, totals["positive_feedback_steps"], label="positive feedback")
    ax.bar(x, totals["negative_feedback_steps"], bottom=totals["positive_feedback_steps"], label="negative feedback")
    ax.set_xticks(x)
    ax.set_xticklabels(totals["controller"], rotation=20, ha="right")
    ax.set_ylabel("Override-active timesteps")
    ax.set_title("March B-spline Tdyn occupant overrides")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "tdyn_override_counts_march.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def table_display(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    html = out.to_html(index=False, escape=True, border=0)
    html = html.replace(
        "<table ",
        '<table style="border-collapse:collapse;font-size:11px;line-height:1.15;width:auto;" ',
        1,
    )
    html = html.replace(
        "<th>",
        '<th style="border:1px solid #666;padding:3px 6px;white-space:nowrap;font-weight:600;">',
    )
    html = html.replace(
        "<td>",
        '<td style="border:1px solid #666;padding:3px 6px;white-space:nowrap;">',
    )
    return (
        '<div style="max-width:100%;overflow-x:auto;margin:0.25rem 0 0.9rem 0;">'
        f"{html}</div>"
    )


def image_cell(path: Path | None, alt: str) -> nbf.NotebookNode | None:
    if path is None:
        return None
    rel = Path("..") / path
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{alt}]({rel.as_posix()})")


def append_once(cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [cell for cell in nb.cells if MARKER not in cell.get("source", "")]
    nb.cells.extend(cells)
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    runs = active_runs()
    new_runs = [run for run in runs if run["controller"] == "Jan+Feb20 SAC"]

    btab = building_kpis(runs)
    rewards = reward_by_episode(runs)
    loss_data = losses_by_episode(runs)
    frames = [baseline_district(), *[district_frame(run) for run in runs]]
    dist = district_kpis(frames)

    mean_cols = [
        "energy",
        "disc_%",
        "cold_%",
        "hot_%",
        "KH",
        "daily_peak",
        "ramping",
        "Tin_min",
        "Tin_max",
        "reward_mean",
        "action_mean",
        "override_starts",
        "override_active_steps",
    ]
    mean_table = btab.groupby(["mode", "controller"], as_index=False)[mean_cols].mean(numeric_only=True)
    worst_new = (
        btab[btab["controller"] == "Jan+Feb20 SAC"]
        .sort_values(["mode", "disc_%"], ascending=[True, False])
        .groupby("mode", as_index=False)
        .head(10)
        [["mode", "building", "occupant", "energy", "disc_%", "cold_%", "hot_%", "KH", "Tin_min", "Tin_max", "action_mean"]]
    )
    final_reward = (
        rewards[rewards["controller"] == "Jan+Feb20 SAC"]
        .sort_values("episode")
        .groupby(["mode", "controller"], as_index=False)
        .tail(1)
        [["mode", "controller", "episode", "raw_reward_mean", "reward_mean", "action_mean"]]
        if not rewards.empty
        else pd.DataFrame()
    )
    new_loss_final = (
        loss_data[loss_data["controller"] == "Jan+Feb20 SAC"]
        .sort_values("episode")
        .groupby(["mode", "controller"], as_index=False)
        .tail(1)
        if not loss_data.empty
        else pd.DataFrame()
    )

    override_rows = []
    override_rows.append(mpc_tdyn_override_rows())
    sac_tdyn = btab[btab["mode"] == "bspline_tdyn"][
        [
            "controller",
            "building",
            "occupant",
            "override_starts",
            "override_active_steps",
            "positive_feedback_steps",
            "negative_feedback_steps",
        ]
    ]
    override_rows.append(sac_tdyn)
    overrides = pd.concat([x for x in override_rows if not x.empty], ignore_index=True)
    override_summary = (
        overrides.groupby("controller", as_index=False)[
            ["override_starts", "override_active_steps", "positive_feedback_steps", "negative_feedback_steps"]
        ]
        .sum()
        .sort_values("override_active_steps")
    )

    btab.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all.csv", index=False)
    mean_table.to_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv", index=False)
    worst_new.to_csv(ANALYSIS / f"{PREFIX}_worst_new_building_rows.csv", index=False)
    rewards.to_csv(ANALYSIS / f"{PREFIX}_sac_reward_by_episode.csv", index=False)
    loss_data.to_csv(ANALYSIS / f"{PREFIX}_sac_losses_by_episode.csv", index=False)
    dist.to_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv", index=False)
    overrides.to_csv(ANALYSIS / f"{PREFIX}_tdyn_override_by_building.csv", index=False)
    override_summary.to_csv(ANALYSIS / f"{PREFIX}_tdyn_override_summary.csv", index=False)

    comfort_paths = [make_comfort_grid(run) for run in new_runs]
    district_load_path, district_daily_path = make_district_plots(frames)
    reward_path = make_reward_plot(rewards)
    loss_path = make_loss_plot(loss_data)
    override_path = make_override_plot(overrides)

    cells: list[nbf.NotebookNode] = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

SAC-only test for whether using more training weather/data fixes the March generalization problem.

Training/deployment protocol:

`ce1_occ_D010702_sac_janfeb20_mar_no_occupant` and `ce1_occ_D010702_sac_janfeb20_mar_bspline_tdyn`.

- Training window: January+February, absolute steps `0..1415`.
- Episode 0: safe-PI bootstrap.
- Episodes 1-19: SAC reward training.
- Deployment: deterministic actor rollout on March, absolute steps `1416..2159`.
- GPU: `--device cuda`.
- No training temperature reset and no reward change.

Reward/cost:

`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`, `reward = -D_k`.

For `bspline_tdyn`, `Tref` is occupancy-gated: the active setpoint follows `Tdyn` when occupied and the schedule when unoccupied.

Mean building KPIs:

{table_display(mean_table, digits=3)}

District KPIs:

{table_display(dist, digits=3)}

March Tdyn override summary:

{table_display(override_summary, digits=0)}

Final training reward rows:

{table_display(final_reward, digits=4)}

Final SAC loss rows:

{table_display(new_loss_final[[c for c in new_loss_final.columns if c in ['mode','controller','episode','soft_q1_loss','soft_q2_loss','policy_loss','alpha','alpha_loss']]], digits=4) if not new_loss_final.empty else 'No SAC loss rows found.'}

Worst Jan+Feb20 SAC March comfort rows:

{table_display(worst_new, digits=3)}

Saved full tables:

- `analysis/{PREFIX}_building_kpis_all.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_sac_reward_by_episode.csv`
- `analysis/{PREFIX}_sac_losses_by_episode.csv`
- `analysis/{PREFIX}_tdyn_override_by_building.csv`
- `analysis/{PREFIX}_tdyn_override_summary.csv`
"""
        )
    ]
    for path, alt in [
        (comfort_paths[0] if len(comfort_paths) > 0 else None, "Jan+Feb20 SAC no occupant March comfort"),
        (comfort_paths[1] if len(comfort_paths) > 1 else None, "Jan+Feb20 SAC B-spline Tdyn March comfort"),
        (district_load_path, "March district load"),
        (district_daily_path, "March mean hourly district load"),
        (reward_path, "Jan+Feb SAC training reward"),
        (loss_path, "Jan+Feb SAC training losses"),
        (override_path, "March Tdyn override counts"),
    ]:
        cell = image_cell(path, alt)
        if cell is not None:
            cells.append(cell)

    append_once(cells)
    print(f"Appended Jan+Feb SAC analysis to {NOTEBOOK}")
    print(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv")
    print(ANALYSIS / f"{PREFIX}_district_kpis.csv")
    print(ANALYSIS / f"{PREFIX}_tdyn_override_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
