#!/usr/bin/env python
"""Append Building 14 Jan-Nov SAC generalization diagnostic to 08_sac_results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, resolve_building_entries
from src.district_target import load_ce1_district_target_profiles


LABEL = "ce1_sac_b14_jannov10_dec_p2_clean_generalization"
MARKER = f"## CE1 SAC Building 14 Jan-Nov Generalization - `{LABEL}`"
BUILDING = resolve_building_entries("14", DATASET_NAME)[0]
SCENARIO = "no_occupant"
DEC_START = 8016
DEC_END = 8759

RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / LABEL
NOTEBOOK = ANALYSIS / "08_sac_results.ipynb"
RAW_DIR = RESULTS / "raw" / "sac_merlin" / BUILDING["result_id"] / SCENARIO / LABEL
SUMMARY = RESULTS / "summaries" / "sac_merlin" / BUILDING["result_id"] / SCENARIO / "summary.csv"


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False, floatfmt=".3f")


def setpoint_series(frame: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "tracking_setpoint", "baseline_setpoint", "scheduled_setpoint", "tmin"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    raise KeyError("No setpoint column found in rollout.")


def comfort_metrics(frame: pd.DataFrame) -> dict[str, float]:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    tsp = setpoint_series(frame)
    n = min(len(tin), len(tsp))
    tin = tin[:n]
    tsp = tsp[:n]
    diff = tin - tsp
    cold = tin < (tsp - 2.0)
    hot = tin > (tsp + 2.0)
    return {
        "reward_mean": float(pd.to_numeric(frame["reward"], errors="coerce").mean()),
        "action_mean": float(pd.to_numeric(frame["action"], errors="coerce").mean()),
        "mae_to_setpoint": float(np.nanmean(np.abs(diff))),
        "cold_%": float(np.nanmean(cold) * 100.0),
        "hot_%": float(np.nanmean(hot) * 100.0),
        "disc_%": float(np.nanmean(cold | hot) * 100.0),
        "KH_setpoint": float(np.nansum(np.abs(diff))),
        "Tin_min": float(np.nanmin(tin)),
        "Tin_max": float(np.nanmax(tin)),
    }


def kpi_lookup() -> dict[str, float]:
    if not SUMMARY.exists():
        return {}
    data = pd.read_csv(SUMMARY)
    label_column = "label" if "label" in data.columns else "case"
    row = data[data[label_column].astype(str).eq(LABEL)]
    if row.empty:
        return {}
    row = row.tail(1).iloc[0]
    out: dict[str, float] = {}
    for key in (
        "electricity_consumption_total",
        "daily_peak_average",
        "ramping_average",
        "training_temperature_reset_steps",
        "replay_dropped_steps",
        "online_safe_pi_steps",
        "online_sac_training_steps",
    ):
        if key in row:
            out[key] = float(pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0])
    return out


def make_comfort_plot(frame: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(frame))
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    tsp = setpoint_series(frame)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.fill_between(x, tsp - 2.0, tsp + 2.0, color="0.88", alpha=0.7, label="+/-2 C")
    ax.plot(x, tin, lw=1.0, label="Tin")
    ax.plot(x, tsp, lw=1.1, label="setpoint")
    ax.set_title("Building 14 SAC Jan-Nov trained -> December deployment comfort")
    ax.set_xlabel("December time step")
    ax.set_ylabel("Temperature [C]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "building14_december_comfort.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def make_daily_load_plot(frame: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    profiles = load_ce1_district_target_profiles(
        building_name=str(BUILDING["name"]),
        simulation_start=DEC_START,
        simulation_end=DEC_END,
    )
    controlled_heat = pd.to_numeric(frame["heating_electricity_consumption"], errors="coerce").to_numpy(dtype=float)
    baseline_district = np.asarray(profiles.baseline_district_load, dtype=float)
    baseline_heat = np.asarray(profiles.baseline_controlled_heating, dtype=float)
    target = np.asarray(profiles.target, dtype=float)
    n = min(len(controlled_heat), len(baseline_district), len(baseline_heat), len(target))
    time = np.arange(n)
    sac_district = baseline_district[:n] - baseline_heat[:n] + controlled_heat[:n]

    rows = []
    for name, load in (
        ("CityLearn baseline", baseline_district[:n]),
        ("SAC B14 Jan-Nov -> Dec", sac_district),
        ("CE1 target", target[:n]),
    ):
        tmp = pd.DataFrame({"controller": name, "hour": time % 24, "load": load})
        rows.append(tmp.groupby(["controller", "hour"], as_index=False)["load"].mean())
    daily = pd.concat(rows, ignore_index=True)
    out_csv = ANALYSIS / f"{LABEL}_daily_load_profile.csv"
    daily.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily.groupby("controller", sort=False):
        style = {"marker": "o", "lw": 1.6}
        if name == "CE1 target":
            style.update({"color": "black", "ls": "--", "lw": 1.2})
        ax.plot(group["hour"], group["load"], label=name, **style)
    ax.set_title("December mean hourly district load: Building 14 SAC replacement")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "building14_december_mean_hourly_district_load.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path, daily


def make_training_reward_plot() -> Path | None:
    path = RAW_DIR / "online_training_rollout.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if "episode" not in data.columns or "raw_reward" not in data.columns:
        return None
    reward_col = "raw_reward"
    grouped = (
        data.assign(
            episode=pd.to_numeric(data["episode"], errors="coerce"),
            reward=pd.to_numeric(data[reward_col], errors="coerce"),
        )
        .dropna(subset=["episode", "reward"])
        .groupby("episode", as_index=False)["reward"]
        .mean()
    )
    if grouped.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grouped["episode"], grouped["reward"], marker="o")
    ax.set_title("Building 14 Jan-Nov SAC training reward")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean raw reward per step")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = FIG_DIR / "building14_training_reward.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def load_shape_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    wide = daily.pivot(index="hour", columns="controller", values="load")
    base = wide["CityLearn baseline"].to_numpy(dtype=float)
    sac = wide["SAC B14 Jan-Nov -> Dec"].to_numpy(dtype=float)
    return pd.DataFrame(
        [
            {
                "metric": "daily_profile_peak_kwh",
                "CityLearn baseline": float(np.nanmax(base)),
                "SAC B14 Jan-Nov -> Dec": float(np.nanmax(sac)),
                "ratio": float(np.nanmax(sac) / max(np.nanmax(base), 1e-9)),
            },
            {
                "metric": "daily_profile_min_kwh",
                "CityLearn baseline": float(np.nanmin(base)),
                "SAC B14 Jan-Nov -> Dec": float(np.nanmin(sac)),
                "ratio": float(np.nanmin(sac) / max(np.nanmin(base), 1e-9)),
            },
            {
                "metric": "daily_profile_ramping_kwh",
                "CityLearn baseline": float(np.nansum(np.abs(np.diff(base)))),
                "SAC B14 Jan-Nov -> Dec": float(np.nansum(np.abs(np.diff(sac)))),
                "ratio": float(np.nansum(np.abs(np.diff(sac))) / max(np.nansum(np.abs(np.diff(base))), 1e-9)),
            },
        ]
    )


def image_md(path: Path, caption: str) -> str:
    notebook_relative = Path("..") / path
    return f"**{caption}:**\n\n![]({notebook_relative.as_posix()})"


def replace_or_append_notebook(markdown: str) -> None:
    source = NOTEBOOK.read_text(encoding="utf-8", errors="replace").rstrip("\x00")
    nb = nbf.reads(source, as_version=4)
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and str(cell.source).startswith(MARKER):
            nb.cells[idx] = nbf.v4.new_markdown_cell(markdown)
            break
    else:
        nb.cells.append(nbf.v4.new_markdown_cell(markdown))
    NOTEBOOK.write_text(nbf.writes(nb), encoding="utf-8")


def main() -> None:
    rollout_path = RAW_DIR / "rollout.csv"
    if not rollout_path.exists():
        raise FileNotFoundError(rollout_path)
    frame = pd.read_csv(rollout_path)
    comfort_path = make_comfort_plot(frame)
    daily_path, daily = make_daily_load_plot(frame)
    reward_path = make_training_reward_plot()

    metrics = comfort_metrics(frame)
    metrics.update(kpi_lookup())
    metric_df = pd.DataFrame([{k: round(v, 4) for k, v in metrics.items()}])
    metric_csv = ANALYSIS / f"{LABEL}_summary.csv"
    metric_df.to_csv(metric_csv, index=False)

    shape_df = load_shape_metrics(daily)
    shape_csv = ANALYSIS / f"{LABEL}_load_shape_summary.csv"
    shape_df.to_csv(shape_csv, index=False)

    recipe = pd.DataFrame(
        [
            {"field": "building", "value": "Building 14"},
            {"field": "training window", "value": "January-November, steps 0-8015"},
            {"field": "deployment window", "value": "December, steps 8016-8759"},
            {"field": "episodes", "value": "1 safe-PI + 9 SAC reward episodes"},
            {"field": "reward", "value": "p2 time-bias guard, weights 0.1/0.7/0.2"},
            {"field": "guard scales", "value": "cold=2.0, hot=2.0"},
            {"field": "auto entropy", "value": "on"},
            {"field": "robust normalization", "value": "off"},
            {"field": "obs/reward clipping", "value": "off/off"},
            {"field": "temperature reset", "value": "off"},
        ]
    )

    body = f"""{MARKER}

Single-building generalization diagnostic for a March-failing building. This is **p2**, not p3, and uses no replay-tail: the actor sees January-November during training and is deployed deterministically in December.

Configuration:

{md_table(recipe)}

December deployment KPIs:

{md_table(metric_df)}

December load-shape check:

{md_table(shape_df)}

{image_md(comfort_path, "Building 14 December plot_comfort-style trace")}

{image_md(daily_path, "December mean hourly district load with Building 14 SAC replacing baseline heating")}
"""
    if reward_path is not None:
        body += "\n\n" + image_md(reward_path, "Jan-Nov online training reward")
    body += f"""

Saved compact tables:

- `{metric_csv.as_posix()}`
- `{shape_csv.as_posix()}`
- `{(ANALYSIS / f"{LABEL}_daily_load_profile.csv").as_posix()}`
"""
    replace_or_append_notebook(body)


if __name__ == "__main__":
    main()
