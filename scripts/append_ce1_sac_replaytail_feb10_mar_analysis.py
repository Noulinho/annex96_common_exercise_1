#!/usr/bin/env python
"""Append SAC replay-tail February continuation / March comparison to 08_sac_results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries


PREFIX = os.environ.get("CE1_SAC_REPLAYTAIL_PREFIX", "ce1_occ_D010702_sac_noreset_replaytail1024_feb10_mar")
MARKER_TITLE = os.environ.get("CE1_SAC_REPLAYTAIL_MARKER_TITLE", "CE1 SAC Gentle Replay-Tail Feb10 -> March")
MARKER = f"## {MARKER_TITLE} - `{PREFIX}`"
REPLAY_DISPLAY = os.environ.get("CE1_SAC_REPLAYTAIL_DISPLAY", "replay-tail SAC")
REWARD_TEXT = os.environ.get(
    "CE1_SAC_REPLAYTAIL_REWARD_TEXT",
    "`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`, `reward = -D_k`.",
)
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOK = ANALYSIS / "08_sac_results.ipynb"
MARCH_BASELINE_DIR = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "march_baseline"
FEB_BASELINE_DIR = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline"
FEB_MPC_LABEL = "ce1_final_no_occ_D010702_mpc_2r2c_kalman"
FEB_RLMPC_LABEL = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_fixed"
FEB_SAC_JAN_LABEL = os.environ.get(
    "CE1_SAC_REPLAYTAIL_FEB_JAN_LABEL",
    "ce1_occ_D010702_sac_noreset_jan20_feb_no_occupant",
)
FEB_SAC_REPLAYTAIL_LABEL = os.environ.get(
    "CE1_SAC_REPLAYTAIL_FEB_EVAL_LABEL",
    "ce1_occ_D010702_sac_noreset_replaytail1024_feb10_febeval_no_occupant",
)

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]

RUNS = [
    {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "controller": "old March SAC",
        "display": "old SAC no occupant",
        "label": "ce1_final_no_occ_D010702_sac_feb10_mar_a1",
    },
    {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "controller": "replay-tail March SAC",
        "display": f"{REPLAY_DISPLAY} no occupant",
        "label": f"{PREFIX}_no_occupant",
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "controller": "old March SAC",
        "display": "old SAC B-spline Tdyn",
        "label": "ce1_occ_D010702_sac_feb10_mar_bspline_tdyn",
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "controller": "replay-tail March SAC",
        "display": f"{REPLAY_DISPLAY} B-spline Tdyn",
        "label": f"{PREFIX}_bspline_tdyn",
    },
]

TABLE_STYLE = """
<style>
.ce1-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 0.25rem 0 0.9rem 0;
}
.ce1-table {
  border-collapse: collapse;
  font-size: 11px;
  line-height: 1.15;
  width: auto;
}
.ce1-table th,
.ce1-table td {
  border: 1px solid #666;
  padding: 3px 6px;
  white-space: nowrap;
}
.ce1-table th {
  font-weight: 600;
}
</style>
""".strip()


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def raw_dir(building: dict, run: dict) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building["result_id"] / run["scenario"] / run["label"]


def summary_path(building: dict, run: dict) -> Path:
    return RESULTS / "summaries" / "sac_merlin" / building["result_id"] / run["scenario"] / "summary.csv"


def require_artifacts() -> None:
    missing = []
    for run in RUNS:
        for building in BUILDINGS:
            directory = raw_dir(building, run)
            for name in ("rollout.csv", "kpis.csv", "manifest.json"):
                if not (directory / name).exists():
                    missing.append(str(directory / name))
    if missing:
        sample = "\n".join(missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} required artifacts. First missing files:\n{sample}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def rollout(building: dict, run: dict) -> pd.DataFrame:
    return read_csv(raw_dir(building, run) / "rollout.csv")


def training_rollout(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "online_training_rollout.csv"
    return pd.read_csv(path) if path.exists() else None


def losses(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "sac_training_losses.csv"
    return pd.read_csv(path) if path.exists() else None


def manifest(building: dict, run: dict) -> dict:
    with (raw_dir(building, run) / "manifest.json").open() as handle:
        return json.load(handle)


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


def setpoint_kelvin_hours_from_rollout(frame: pd.DataFrame) -> float:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    tsp = active_setpoint(frame)
    n = min(len(tin), len(tsp))
    if n == 0:
        return np.nan
    valid = np.isfinite(tin[:n]) & np.isfinite(tsp[:n])
    if not np.any(valid):
        return np.nan
    return float(np.sum(np.abs(tin[:n][valid] - tsp[:n][valid])))


def building_kpis() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for building in BUILDINGS:
            frame = rollout(building, run)
            kpis = kpi_values(building, run)
            summary = summary_row(building, run)
            tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            reward = pd.to_numeric(frame.get("reward", pd.Series(dtype=float)), errors="coerce")
            action = pd.to_numeric(frame.get("action", pd.Series(dtype=float)), errors="coerce")
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
                    "KH_setpoint": num(summary, "setpoint_kelvin_hours", setpoint_kelvin_hours_from_rollout(frame)),
                    "daily_peak": kpis.get("daily_peak_average", num(summary, "daily_peak_average")),
                    "ramping": kpis.get("ramping_average", np.nan),
                    "Tin_min": float(np.nanmin(tin)) if len(tin) else np.nan,
                    "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
                    "reward_sum": float(reward.fillna(0.0).sum()) if not reward.empty else num(summary, "reward_total"),
                    "reward_mean": float(reward.mean()) if not reward.empty else np.nan,
                    "action_mean": float(action.mean()) if not action.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def per_building_delta_table(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = data[data["mode"] == mode].copy()
    old = rows[rows["controller"] == "old March SAC"].set_index("building")
    new = rows[rows["controller"] == "replay-tail March SAC"].set_index("building")
    common = [b for b in old.index if b in new.index]
    out = pd.DataFrame(
        {
            "building": common,
            "occupant": [new.loc[b, "occupant"] for b in common],
            "old_disc_%": [old.loc[b, "disc_%"] for b in common],
            "new_disc_%": [new.loc[b, "disc_%"] for b in common],
            "delta_disc_%": [new.loc[b, "disc_%"] - old.loc[b, "disc_%"] for b in common],
            "old_cold_%": [old.loc[b, "cold_%"] for b in common],
            "new_cold_%": [new.loc[b, "cold_%"] for b in common],
            "old_hot_%": [old.loc[b, "hot_%"] for b in common],
            "new_hot_%": [new.loc[b, "hot_%"] for b in common],
            "old_energy": [old.loc[b, "energy"] for b in common],
            "new_energy": [new.loc[b, "energy"] for b in common],
            "old_Tmax": [old.loc[b, "Tin_max"] for b in common],
            "new_Tmax": [new.loc[b, "Tin_max"] for b in common],
        }
    )
    return out.sort_values("delta_disc_%")


def reward_by_episode() -> pd.DataFrame:
    frames = []
    for run in RUNS:
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


def losses_by_episode() -> pd.DataFrame:
    frames = []
    for run in RUNS:
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


def replay_trim_summary() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        if run["controller"] != "replay-tail March SAC":
            continue
        for building in BUILDINGS:
            metrics = manifest(building, run).get("metrics", {})
            rows.append(
                {
                    "mode": run["mode"],
                    "building": building["display"],
                    "source_replay": metrics.get("resume_source_replay_size", np.nan),
                    "keep_last": metrics.get("resume_replay_keep_last", np.nan),
                    "after_trim": metrics.get("resume_replay_after_trim_size", np.nan),
                    "final_replay": metrics.get("online_replay_size", np.nan),
                    "feb_train_steps": metrics.get("online_train_steps", np.nan),
                    "reward_total": metrics.get("online_train_reward_total", np.nan),
                }
            )
    return pd.DataFrame(rows)


def baseline_district() -> pd.DataFrame:
    data = read_csv(MARCH_BASELINE_DIR / "district_rollout.csv")
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


def controller_rollout_dir(controller: str, building: dict, label: str) -> Path:
    scenario = "no_occupant"
    return RESULTS / "raw" / controller / building["result_id"] / scenario / label


def february_controller_district_frame(controller: str, label: str, display: str) -> pd.DataFrame:
    series = []
    time = None
    for building in BUILDINGS:
        frame = read_csv(controller_rollout_dir(controller, building, label) / "rollout.csv")
        load = load_series(frame)
        t = time_axis(frame)
        n = min(len(t), len(load))
        if time is None:
            time = t[:n]
        series.append(load[:n])
    n = min(len(s) for s in series)
    return pd.DataFrame(
        {
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([s[:n] for s in series], axis=0),
            "controller": display,
        }
    )


def february_baseline_district_frame() -> pd.DataFrame:
    data = read_csv(FEB_BASELINE_DIR / "district_rollout.csv")
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "controller": "CityLearn baseline",
        }
    )


def make_february_in_sample_replaytail_daily_plot() -> tuple[Path, Path]:
    frames = [
        february_baseline_district_frame(),
        february_controller_district_frame("mpc", FEB_MPC_LABEL, "MPC"),
        february_controller_district_frame("rlmpc", FEB_RLMPC_LABEL, "GNU-RL fixed"),
        february_controller_district_frame("sac_merlin", FEB_SAC_JAN_LABEL, "SAC Jan-trained -> Feb"),
        february_controller_district_frame(
            "sac_merlin",
            FEB_SAC_REPLAYTAIL_LABEL,
            f"{REPLAY_DISPLAY} Feb-trained -> Feb",
        ),
    ]
    daily_rows = []
    for frame in frames:
        tmp = frame.copy()
        tmp["hour"] = np.arange(len(tmp)) % 24
        daily = tmp.groupby("hour", as_index=False)["load"].mean()
        daily["controller"] = frame["controller"].iloc[0]
        daily_rows.append(daily)
    daily_profile = pd.concat(daily_rows, ignore_index=True)

    csv_path = ANALYSIS / f"{PREFIX}_feb_in_sample_replaytail_daily_profile.csv"
    daily_profile.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in daily_profile.groupby("controller", sort=False):
        lw = 1.8 if "replay-tail" in name else 1.25
        marker = "o" if name != "CityLearn baseline" else "s"
        ax.plot(group["hour"], group["load"], marker=marker, lw=lw, label=name)
    ax.set_title("February mean hourly district load - replay-tail policy evaluated in training month")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIG_DIR / "feb_mean_hourly_replaytail_in_sample_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path, csv_path


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
    fig.suptitle(f"{run['display']} | March comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['mode']}_{run['controller'].lower().replace(' ', '_')}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_february_in_sample_replaytail_comfort_grid() -> Path:
    run = {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "display": f"{REPLAY_DISPLAY} Feb-trained -> Feb",
        "label": FEB_SAC_REPLAYTAIL_LABEL,
    }
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        frame = rollout(building, run)
        t = time_axis(frame)
        tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = active_setpoint(frame)
        n = min(len(t), len(tin), len(tsp))
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.75, color="tab:orange", label="setpoint")
        ax.set_title(f"{building['display']} | none", fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{REPLAY_DISPLAY} Feb-trained -> Feb | February comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "feb_replaytail_in_sample_sac_comfort_all25.png"
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
            if frame["mode"].iloc[0] != mode:
                continue
            ax.plot(frame["time_step"], frame["load"], lw=0.9, label=frame["controller"].iloc[0])
        ax.set_title(f"{title} - March district load")
        ax.set_ylabel("District electricity [kWh]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("March time step")
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries_old_vs_replaytail.png"
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
    daily_path = FIG_DIR / "district_mean_hourly_old_vs_replaytail.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return load_path, daily_path


def make_reward_plot(rewards: pd.DataFrame) -> Path | None:
    if rewards.empty:
        return None
    grouped = rewards.groupby(["mode", "controller", "episode"], as_index=False).agg(
        raw_reward_mean=("raw_reward_mean", "mean"),
        raw_reward_p10=("raw_reward_mean", lambda x: float(np.nanpercentile(x, 10))),
        raw_reward_p90=("raw_reward_mean", lambda x: float(np.nanpercentile(x, 90))),
        action_mean=("action_mean", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, mode, title in zip(axes, ("no_occupant", "bspline_tdyn"), ("No occupant", "B-spline Tdyn")):
        for name, group in grouped[grouped["mode"] == mode].groupby("controller", sort=False):
            x = group["episode"].to_numpy(dtype=float)
            mean = group["raw_reward_mean"].to_numpy(dtype=float)
            ax.plot(x, mean, marker="o", lw=1.3, label=name)
            ax.fill_between(
                x,
                group["raw_reward_p10"].to_numpy(dtype=float),
                group["raw_reward_p90"].to_numpy(dtype=float),
                alpha=0.10,
            )
        ax.axhline(0.0, color="0.35", lw=0.8)
        ax.set_title(f"{title} February continuation reward")
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean raw reward per step")
    fig.tight_layout()
    path = FIG_DIR / "sac_feb_training_reward_old_vs_replaytail.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_loss_plot(loss_data: pd.DataFrame) -> Path | None:
    if loss_data.empty or "policy_loss" not in loss_data.columns:
        return None
    grouped = loss_data.groupby(["mode", "controller", "episode"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=False)
    panels = [("no_occupant", "soft_q1_loss"), ("no_occupant", "policy_loss"), ("bspline_tdyn", "soft_q1_loss"), ("bspline_tdyn", "policy_loss")]
    for ax, (mode, metric) in zip(axes.ravel(), panels):
        for name, group in grouped[grouped["mode"] == mode].groupby("controller", sort=False):
            if metric in group:
                ax.plot(group["episode"], group[metric], marker="o", lw=1.1, label=name)
        ax.set_title(f"{mode} - {metric}")
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "sac_feb_training_losses_old_vs_replaytail.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def table_display(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    html = out.to_html(index=False, escape=True, border=0, classes="ce1-table")
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
        '<div class="ce1-table-wrap" '
        'style="max-width:100%;overflow-x:auto;margin:0.25rem 0 0.9rem 0;">'
        f"{html}</div>"
    )


def image(path: Path, alt: str) -> nbf.NotebookNode:
    rel = Path("..") / path
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{alt}]({rel.as_posix()})")


def append_once(cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [cell for cell in nb.cells if MARKER not in cell.get("source", "")]
    nb.cells.extend(cells)
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    require_artifacts()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    btab = building_kpis()
    rewards = reward_by_episode()
    loss_data = losses_by_episode()
    replay = replay_trim_summary()
    frames = [baseline_district(), *[district_frame(run) for run in RUNS]]
    dist = district_kpis(frames)

    mean_cols = ["energy", "disc_%", "cold_%", "hot_%", "KH", "KH_setpoint", "daily_peak", "ramping", "Tin_min", "Tin_max", "reward_mean", "action_mean"]
    mean_table = btab.groupby(["mode", "controller"], as_index=False)[mean_cols].mean(numeric_only=True)
    no_delta = per_building_delta_table(btab, "no_occupant")
    tdyn_delta = per_building_delta_table(btab, "bspline_tdyn")
    replay_mean = replay.groupby("mode", as_index=False).mean(numeric_only=True)
    final_reward = (
        rewards.sort_values("episode")
        .groupby(["mode", "controller"], as_index=False)
        .tail(1)
        [["mode", "controller", "episode", "raw_reward_mean", "reward_mean", "action_mean"]]
        if not rewards.empty
        else pd.DataFrame()
    )

    btab.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all.csv", index=False)
    mean_table.to_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv", index=False)
    no_delta.to_csv(ANALYSIS / f"{PREFIX}_no_occupant_per_building_delta.csv", index=False)
    tdyn_delta.to_csv(ANALYSIS / f"{PREFIX}_bspline_tdyn_per_building_delta.csv", index=False)
    rewards.to_csv(ANALYSIS / f"{PREFIX}_sac_reward_by_episode.csv", index=False)
    loss_data.to_csv(ANALYSIS / f"{PREFIX}_sac_losses_by_episode.csv", index=False)
    replay.to_csv(ANALYSIS / f"{PREFIX}_replay_trim_summary.csv", index=False)
    dist.to_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv", index=False)

    comfort_paths = [make_comfort_grid(run) for run in RUNS]
    district_load_path, district_daily_path = make_district_plots(frames)
    feb_cheat_daily_path, feb_cheat_daily_csv = make_february_in_sample_replaytail_daily_plot()
    feb_cheat_comfort_path = make_february_in_sample_replaytail_comfort_grid()
    reward_path = make_reward_plot(rewards)
    loss_path = make_loss_plot(loss_data)

    cells: list[nbf.NotebookNode] = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

Gentle replay-buffer month-boundary test for SAC only.

The actors are the no-reset SAC agents trained for 20 January episodes. For February continuation, the agent keeps only the last `1024` January replay transitions, then trains for 10 February SAC reward episodes. With about `6710` February training transitions, uniform replay sampling is therefore mostly February data while retaining a small January anchor.

Reward used for this run:

{REWARD_TEXT}

Compared March deployments:

`old March SAC`: existing March SAC artifacts already in the notebook.

`replay-tail March SAC`: `{PREFIX}_no_occupant` and `{PREFIX}_bspline_tdyn`.

Mean building KPIs:

{TABLE_STYLE}

{table_display(mean_table, digits=3)}

District aggregate KPIs:

{table_display(dist, digits=3)}

Replay-buffer trim check:

{table_display(replay_mean, digits=1)}

Final February continuation reward:

{table_display(final_reward, digits=4)}

Per-building no-occupant March change, sorted by discomfort change:

{table_display(no_delta, digits=3)}

Per-building B-spline Tdyn March change, sorted by discomfort change:

{table_display(tdyn_delta, digits=3)}

Saved full tables:

- `analysis/{PREFIX}_building_kpis_all.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_no_occupant_per_building_delta.csv`
- `analysis/{PREFIX}_bspline_tdyn_per_building_delta.csv`
- `analysis/{PREFIX}_sac_reward_by_episode.csv`
- `analysis/{PREFIX}_sac_losses_by_episode.csv`
- `analysis/{PREFIX}_replay_trim_summary.csv`
- `{feb_cheat_daily_csv.as_posix()}`
- `{feb_cheat_comfort_path.as_posix()}`
"""
        )
    ]

    for path, run in zip(comfort_paths, RUNS):
        cells.append(image(path, f"{run['display']} March comfort"))
    cells.append(image(district_load_path, "SAC March district load old vs replay-tail"))
    cells.append(image(district_daily_path, "SAC March mean hourly district load old vs replay-tail"))
    cells.append(
        image(
            feb_cheat_comfort_path,
            "SAC replay-tail Feb-trained policy evaluated on February comfort",
        )
    )
    cells.append(
        image(
            feb_cheat_daily_path,
            "February mean hourly district load with replay-tail SAC evaluated in-sample",
        )
    )
    if reward_path is not None:
        cells.append(image(reward_path, "SAC February continuation reward old vs replay-tail"))
    if loss_path is not None:
        cells.append(image(loss_path, "SAC February continuation losses old vs replay-tail"))

    append_once(cells)
    print(f"Appended {NOTEBOOK}")
    print(f"Saved figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
