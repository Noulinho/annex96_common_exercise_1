#!/usr/bin/env python
"""Append SAC bad-building episode-20 continuation diagnostics to notebooks."""

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
from src.rewards import time_bias_setpoint_profile


BASE_LABEL = "ce1_time_bias_guard_1c_1r1c_sac_safepi_penalty100_auto_a1"
RESUME_LABEL = "ce1_time_bias_guard_1c_1r1c_sac_safepi_penalty100_resume20_bad_a1"
BASE_DISPLAY = "SAC safe-PI reset + additive reset penalty -100"
RESUME_DISPLAY = "SAC bad-building resume to episode 20"
HYBRID_DISPLAY = "SAC penalty100 + bad buildings resumed to ep20"

ANALYSIS_DIR = Path("analysis")
FIG_DIR = Path("figures") / RESUME_LABEL
RESULTS = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)
BAD_BUILDING_IDS = {
    "building_01",
    "building_02",
    "building_08",
    "building_11",
    "building_14",
    "building_15",
    "building_19",
    "building_20",
    "building_21",
    "building_22",
    "building_25",
}
FEB_START = 744


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _case_dir(building_id: str, label: str) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building_id / "no_occupant" / label


def _rollout(building_id: str, label: str) -> pd.DataFrame:
    return _read(_case_dir(building_id, label) / "rollout.csv")


def _online(building_id: str, label: str) -> pd.DataFrame:
    return _read(_case_dir(building_id, label) / "online_training_rollout.csv")


def _manifest_metrics(building_id: str, label: str) -> dict:
    path = _case_dir(building_id, label) / "manifest.json"
    return json.loads(path.read_text())["metrics"]


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


def _bool_float(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _building_kpis() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        if building_id not in BAD_BUILDING_IDS:
            continue
        base_metrics = _manifest_metrics(building_id, BASE_LABEL)
        resume_metrics = _manifest_metrics(building_id, RESUME_LABEL)
        base_rollout = _rollout(building_id, BASE_LABEL)
        resume_rollout = _rollout(building_id, RESUME_LABEL)
        for label, display, metrics, rollout in [
            (BASE_LABEL, BASE_DISPLAY, base_metrics, base_rollout),
            (RESUME_LABEL, RESUME_DISPLAY, resume_metrics, resume_rollout),
        ]:
            tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
            rows.append(
                {
                    "building": building["display"],
                    "building_id": building_id,
                    "controller": display,
                    "label": label,
                    "electricity_consumption_total": float(metrics["electricity_consumption_total"]),
                    "discomfort_%": 100.0 * float(metrics["discomfort_proportion"]),
                    "cold_%": 100.0 * float(metrics["discomfort_cold_proportion"]),
                    "hot_%": 100.0 * float(metrics["discomfort_hot_proportion"]),
                    "kelvin_hours": float(metrics["kelvin_hours"]),
                    "daily_peak_average": float(metrics["daily_peak_average"]),
                    "ramping_average": float(metrics.get("ramping_average", np.nan)),
                    "feb_tin_min": float(tin.min()),
                    "feb_tin_max": float(tin.max()),
                    "training_temperature_reset_steps": int(metrics.get("training_temperature_reset_steps", 0)),
                    "resume_loaded": bool(metrics.get("resume_loaded", False)),
                }
            )
    return pd.DataFrame(rows)


def _delta_table(kpis: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "building",
        "building_id",
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "ramping_average",
        "feb_tin_max",
    ]
    base = kpis[kpis["controller"] == BASE_DISPLAY][keep].copy()
    resume = kpis[kpis["controller"] == RESUME_DISPLAY][keep].copy()
    merged = base.merge(resume, on=["building", "building_id"], suffixes=("_ep10", "_ep20"))
    for column in keep[2:]:
        merged[f"delta_{column}"] = merged[f"{column}_ep20"] - merged[f"{column}_ep10"]
    return merged.sort_values("delta_discomfort_%")


def _mean_kpis(kpis: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "ramping_average",
        "feb_tin_max",
    ]
    return kpis.groupby("controller", as_index=False)[cols].mean(numeric_only=True)


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


def _district_from_rollouts(display: str, *, hybrid: bool) -> pd.DataFrame:
    loads = []
    time = None
    for building in BUILDINGS:
        building_id = building["result_id"]
        label = RESUME_LABEL if hybrid and building_id in BAD_BUILDING_IDS else BASE_LABEL
        rollout = _rollout(building_id, label)
        values = _load(rollout)
        if time is None:
            time = _time_step(rollout)
        n = min(len(values), len(time))
        loads.append(values[:n])
        time = time[:n]
    n = min(len(v) for v in loads)
    baseline = _baseline_district()
    return pd.DataFrame(
        {
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([v[:n] for v in loads], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "controller": display,
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


def _training_reward() -> pd.DataFrame:
    frames = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        if building_id not in BAD_BUILDING_IDS:
            continue
        for label, controller in [(BASE_LABEL, BASE_DISPLAY), (RESUME_LABEL, RESUME_DISPLAY)]:
            data = _online(building_id, label).copy()
            data["building"] = building["display"]
            data["building_id"] = building_id
            data["controller"] = controller
            frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    data = data[data["phase"].astype(str).str.contains("sac", case=False, na=False)].copy()
    raw = pd.to_numeric(data["raw_reward"] if "raw_reward" in data.columns else data["reward"], errors="coerce")
    reward = pd.to_numeric(data["reward"], errors="coerce")
    data["raw_reward_plot"] = raw.fillna(reward)
    data["reset_penalty_flag"] = _bool_float(data.get("reset_penalty_applied", pd.Series(False, index=data.index)))
    data["action"] = pd.to_numeric(data["action"], errors="coerce")
    out = (
        data.groupby(["controller", "building", "building_id", "episode"], as_index=False)
        .agg(
            raw_reward_mean=("raw_reward_plot", "mean"),
            reward_mean=("reward", "mean"),
            reset_penalty_frac=("reset_penalty_flag", "mean"),
            action_mean=("action", "mean"),
        )
        .sort_values(["building_id", "controller", "episode"])
    )
    return out


def _loss_summary() -> pd.DataFrame:
    frames = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        if building_id not in BAD_BUILDING_IDS:
            continue
        for label, controller in [(BASE_LABEL, BASE_DISPLAY), (RESUME_LABEL, RESUME_DISPLAY)]:
            path = _case_dir(building_id, label) / "sac_training_losses.csv"
            if not path.exists():
                continue
            data = _read(path)
            data["building"] = building["display"]
            data["building_id"] = building_id
            data["controller"] = controller
            frames.append(data)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    cols = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"] if c in data.columns]
    out = data.groupby(["controller", "episode"], as_index=False)[cols].agg(["mean", "max"])
    out.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in out.columns.to_flat_index()]
    return out.sort_values(["controller", "episode"])


def _save_comfort_grid() -> Path:
    rows = [b for b in BUILDINGS if b["result_id"] in BAD_BUILDING_IDS]
    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, building in zip(axes, rows):
        rollout = _rollout(building["result_id"], RESUME_LABEL)
        t = _time_step(rollout)
        tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(rollout)
        n = min(len(t), len(tin), len(tsp))
        bias = np.asarray(
            time_bias_setpoint_profile(
                simulation_start=FEB_START,
                simulation_end=FEB_START + n - 1,
                amplitude=1.0,
            ),
            dtype=float,
        )
        ax.plot(t[:n], tin[:n], lw=0.9, label="Indoor temperature")
        ax.plot(t[:n], tsp[:n], "--", lw=0.7, color="black", label="Setpoint")
        ax.plot(t[:n], tsp[:n] + bias[:n], lw=0.8, color="tab:orange", label="Effective setpoint")
        ax.fill_between(t[:n], tsp[:n] - 2.0, tsp[:n] + 2.0, color="gray", alpha=0.16)
        ax.set_title(building["display"], fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[len(rows):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Bad buildings | SAC resumed to episode 20 | February comfort", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / "bad_building_february_comfort_resume20.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_delta_plot(delta: pd.DataFrame) -> Path:
    plot = delta.sort_values("discomfort_%_ep10", ascending=False)
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(13, 5))
    width = 0.38
    ax.bar(x - width / 2, plot["discomfort_%_ep10"], width, label="Episode 10 policy")
    ax.bar(x + width / 2, plot["discomfort_%_ep20"], width, label="Resumed episode 20 policy")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["building"], rotation=45, ha="right")
    ax.set_ylabel("February discomfort [%]")
    ax.set_title("Bad-building February comfort: episode 10 vs resumed episode 20")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "bad_building_discomfort_ep10_vs_ep20.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_reward_plots(reward: pd.DataFrame) -> tuple[Path, Path]:
    mean = reward.groupby(["controller", "episode"], as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in mean.groupby("controller", sort=False):
        ax.plot(group["episode"], group["raw_reward_mean"], marker="o", lw=1.2, label=name)
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Bad buildings: online SAC raw reward continues after episode 10")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean raw reward per step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    reward_path = FIG_DIR / "bad_building_training_reward_ep1_to_ep20.png"
    fig.savefig(reward_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    resume = reward[reward["controller"] == RESUME_DISPLAY].copy()
    fig, ax = plt.subplots(figsize=(13, 5))
    for building, group in resume.groupby("building", sort=False):
        ax.plot(group["episode"], 100.0 * group["reset_penalty_frac"], marker="o", lw=1.1, label=building)
    ax.set_title("Bad buildings: reset penalty frequency during resumed episodes 11-20")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Steps with reset penalty [%]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    reset_path = FIG_DIR / "bad_building_resume_reset_penalty_fraction.png"
    fig.savefig(reset_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return reward_path, reset_path


def _save_loss_plot(losses: pd.DataFrame) -> Path | None:
    if losses.empty:
        return None
    cols = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"] if f"{c}_mean" in losses.columns]
    if not cols:
        return None
    fig, axes = plt.subplots(len(cols), 1, figsize=(13, 2.7 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        for name, group in losses.groupby("controller", sort=False):
            ax.plot(group["episode"], group[f"{col}_mean"], marker="o", lw=1.1, label=f"{name} mean")
            ax.plot(group["episode"], group[f"{col}_max"], marker="o", lw=0.8, ls="--", label=f"{name} max")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Training episode")
    fig.suptitle("Bad buildings: SAC losses before and after continuation")
    fig.tight_layout()
    path = FIG_DIR / "bad_building_sac_losses_ep1_to_ep20.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=name)
    ax.plot(frames[0]["time_step"], frames[0]["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Hybrid district load - bad buildings resumed to episode 20")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    load_path = FIG_DIR / "hybrid_district_load_timeseries.png"
    fig.savefig(load_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = frames[0].assign(hour=frames[0]["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.2, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Mean hourly district load - hybrid episode-20 bad-building policy")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    daily_path = FIG_DIR / "hybrid_district_mean_daily_profile.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


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


def _image_cell(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.as_posix()})")


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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    kpis = _building_kpis()
    delta = _delta_table(kpis)
    mean_kpis = _mean_kpis(kpis)
    reward = _training_reward()
    losses = _loss_summary()

    kpis.to_csv(ANALYSIS_DIR / f"{RESUME_LABEL}_bad_building_kpis.csv", index=False)
    delta.to_csv(ANALYSIS_DIR / f"{RESUME_LABEL}_bad_building_kpi_deltas.csv", index=False)
    mean_kpis.to_csv(ANALYSIS_DIR / f"{RESUME_LABEL}_bad_building_mean_kpis.csv", index=False)
    reward.to_csv(ANALYSIS_DIR / f"{RESUME_LABEL}_bad_building_training_reward.csv", index=False)
    losses.to_csv(ANALYSIS_DIR / f"{RESUME_LABEL}_bad_building_sac_losses.csv", index=False)

    district_frames = [
        _baseline_district(),
        _district_from_rollouts(BASE_DISPLAY, hybrid=False),
        _district_from_rollouts(HYBRID_DISPLAY, hybrid=True),
    ]
    district_kpis = _district_kpis(district_frames)
    district_kpis.to_csv(ANALYSIS_DIR / f"{RESUME_LABEL}_hybrid_district_kpis.csv", index=False)

    comfort_path = _save_comfort_grid()
    delta_path = _save_delta_plot(delta)
    reward_path, reset_path = _save_reward_plots(reward)
    loss_path = _save_loss_plot(losses)
    district_load_path, district_daily_path = _save_district_plots(district_frames)

    delta_cols = [
        "building",
        "discomfort_%_ep10",
        "discomfort_%_ep20",
        "delta_discomfort_%",
        "cold_%_ep20",
        "hot_%_ep20",
        "electricity_consumption_total_ep10",
        "electricity_consumption_total_ep20",
        "delta_electricity_consumption_total",
        "feb_tin_max_ep20",
    ]
    final_reward = reward.groupby(["controller", "episode"], as_index=False).mean(numeric_only=True)
    final_reward = final_reward.groupby("controller", as_index=False).tail(1)
    final_losses = pd.DataFrame() if losses.empty else losses.groupby("controller", as_index=False).tail(1)

    overview = f"""## CE1 SAC Bad-Building Continuation - `{RESUME_LABEL}`

Continuation experiment for the 11 catastrophe buildings from `{BASE_LABEL}`: Buildings 1, 2, 8, 11, 14, 15, 19, 20, 21, 22, and 25.

Each run loads the saved episode-10 penalty100 SAC agent state, replay buffer, normalization stats, actor/critic weights, target critics, and entropy coefficient, then trains 10 more January SAC episodes logged as episodes 11-20. The original saved artifact did not include optimizer moments, so optimizers restart fresh; this is close to a continuation, but not a perfect bit-for-bit resume.

February deployment is still unshielded. The hybrid district rows below use resumed episode-20 policies for the 11 bad buildings and the original penalty100 episode-10 policies for the other 14 buildings.

Bad-building mean KPIs:

{_markdown_table(mean_kpis)}

Per-building change, sorted by discomfort improvement:

{_markdown_table(delta[delta_cols])}

Hybrid full-district aggregate KPIs:

{_markdown_table(district_kpis)}

Final online reward by phase:

{_markdown_table(final_reward)}
"""
    if not final_losses.empty:
        overview += f"""
Final SAC loss summary:

{_markdown_table(final_losses)}
"""

    image_paths = [
        comfort_path,
        delta_path,
        reward_path,
        reset_path,
        district_load_path,
        district_daily_path,
    ]
    if loss_path is not None:
        image_paths.append(loss_path)
    cells = [
        nbf.v4.new_markdown_cell(overview),
        *[_image_cell(path, path.stem.replace("_", " ")) for path in image_paths],
    ]

    marker = f"## CE1 SAC Bad-Building Continuation - `{RESUME_LABEL}`"
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
