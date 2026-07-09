#!/usr/bin/env python
"""Append thesis-clean load-shape figures for the final CE1 experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_analysis import run_citylearn_baseline_district


PREFIX = "ce1_thesis_clean_load_figures"
OCC_PREFIX = "ce1_occ_p2s2_all_occ"
RBC_PREFIX = "ce1_thesis_rbc_safepi"
SAC_JAN_EVAL_CASE = f"{OCC_PREFIX}_sac_jan20_jan_no_occupant"
MARKER = f"## CE1 Thesis-Clean Load Figures - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOKS = [
    ANALYSIS / "02_mpc_results.ipynb",
    ANALYSIS / "04_rlmpc_results.ipynb",
    ANALYSIS / "08_sac_results.ipynb",
    ANALYSIS / "10_peak_load_profiles.ipynb",
    ANALYSIS / "11_no_occupant_single_building_comparison.ipynb",
    ANALYSIS / "12_simple_v4_comparison.ipynb",
]

BUILDINGS = dataset_building_entries(DATASET_NAME)
MONTHS = {
    "jan": {"display": "January", "start": 0, "end": 743, "baseline_case": "january"},
    "feb": {"display": "February", "start": 744, "end": 1415, "baseline_case": "baseline"},
    "mar": {"display": "March", "start": 1416, "end": 2159, "baseline_case": "march_baseline"},
}
OCCUPANTS = [
    ("occ1_tolerant", "occ1 tolerant"),
    ("occ2_sensitive", "occ2 sensitive"),
    ("occ3_cold", "occ3 cold"),
    ("occ4_hot", "occ4 hot"),
]
RBC_COMFORT_CASES = [
    ("jan", "no_occupant", None),
    ("feb", "no_occupant", None),
    ("mar", "no_occupant", None),
    ("feb", "without_tdyn", "occ1_tolerant"),
    ("feb", "without_tdyn", "occ2_sensitive"),
    ("feb", "without_tdyn", "occ3_cold"),
    ("feb", "without_tdyn", "occ4_hot"),
    ("feb", "bspline_tdyn", "occ1_tolerant"),
    ("feb", "bspline_tdyn", "occ2_sensitive"),
    ("feb", "bspline_tdyn", "occ3_cold"),
    ("feb", "bspline_tdyn", "occ4_hot"),
    ("mar", "without_tdyn", "occ1_tolerant"),
    ("mar", "without_tdyn", "occ2_sensitive"),
    ("mar", "without_tdyn", "occ3_cold"),
    ("mar", "without_tdyn", "occ4_hot"),
    ("mar", "bspline_tdyn", "occ1_tolerant"),
    ("mar", "bspline_tdyn", "occ2_sensitive"),
    ("mar", "bspline_tdyn", "occ3_cold"),
    ("mar", "bspline_tdyn", "occ4_hot"),
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_col(frame: pd.DataFrame) -> np.ndarray:
    for column in ("net_electricity_consumption", "load", "electricity_consumption"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raise KeyError("No electricity/load column found.")


def hourly_mean(load: np.ndarray, month_start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    load = np.asarray(load, dtype=float).reshape(-1)
    hours = (np.arange(len(load), dtype=int) + int(month_start)) % 24
    data = pd.DataFrame({"hour": hours, "load": load})
    grouped = data.groupby("hour", sort=True)["load"].mean()
    return grouped.index.to_numpy(dtype=int), grouped.to_numpy(dtype=float)


def gated_reference(data: pd.DataFrame) -> np.ndarray:
    if {"baseline_setpoint", "tdyn", "occ_bin_1step"}.issubset(data.columns):
        schedule = pd.to_numeric(data["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
        tdyn = pd.to_numeric(data["tdyn"], errors="coerce").to_numpy(dtype=float)
        gate = (
            pd.to_numeric(data["occ_bin_1step"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        n = min(len(schedule), len(tdyn), len(gate))
        reference = np.full(len(data), np.nan, dtype=float)
        reference[:n] = (1.0 - gate[:n]) * schedule[:n] + gate[:n] * tdyn[:n]
        if n < len(data) and "reference_setpoint" in data.columns:
            fallback = pd.to_numeric(data["reference_setpoint"], errors="coerce").to_numpy(dtype=float)
            reference[n:] = fallback[n:]
        return reference

    for column in ("reference_setpoint", "tracking_setpoint", "effective_setpoint", "baseline_setpoint"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    raise KeyError("No reference setpoint column found.")


def load_metrics(load: np.ndarray) -> dict[str, float]:
    load = np.asarray(load, dtype=float).reshape(-1)
    load = load[np.isfinite(load)]
    if len(load) == 0:
        return {
            "average_daily_peak": np.nan,
            "one_minus_load_factor": np.nan,
            "net_electricity": np.nan,
            "up_ramping": np.nan,
            "all_time_peak": np.nan,
            "grid_efficiency_cost": np.nan,
        }
    n_days = len(load) // 24
    daily = load[: n_days * 24].reshape(n_days, 24) if n_days else load.reshape(1, -1)
    peak = float(np.nanmax(load))
    daily_peak = np.nanmax(daily, axis=1)
    daily_mean = np.nanmean(daily, axis=1)
    one_minus_lf = np.nanmean(
        1.0
        - np.divide(
            daily_mean,
            daily_peak,
            out=np.full(len(daily_peak), np.nan, dtype=float),
            where=daily_peak > 0.0,
        )
    )
    up_ramping = float(np.nansum(np.maximum(np.diff(load), 0.0)))
    average_daily_peak = float(np.nanmean(daily_peak))
    return {
        "average_daily_peak": average_daily_peak,
        "one_minus_load_factor": float(one_minus_lf),
        "net_electricity": float(np.nansum(load)),
        "up_ramping": up_ramping,
        "all_time_peak": peak,
        "grid_efficiency_cost": float(np.nanmean([up_ramping, one_minus_lf, average_daily_peak, peak])),
    }


def cumulative_load_metrics(load: np.ndarray) -> pd.DataFrame:
    load = np.asarray(load, dtype=float).reshape(-1)
    load = np.nan_to_num(load, nan=0.0)
    n = len(load)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "step",
                "day",
                "average_daily_peak",
                "one_minus_load_factor",
                "net_electricity",
                "up_ramping",
                "all_time_peak",
            ]
        )

    steps = np.arange(n, dtype=int)
    cumulative_energy = np.cumsum(load)
    cumulative_max = np.maximum.accumulate(load)
    cumulative_mean = cumulative_energy / (steps + 1)
    one_minus_load_factor = 1.0 - np.divide(
        cumulative_mean,
        cumulative_max,
        out=np.full(n, np.nan, dtype=float),
        where=cumulative_max > 0.0,
    )
    up_ramping = np.zeros(n, dtype=float)
    if n > 1:
        up_ramping[1:] = np.cumsum(np.maximum(np.diff(load), 0.0))

    daily_peak_progress = np.empty(n, dtype=float)
    completed_daily_peaks: list[float] = []
    for day_start in range(0, n, 24):
        day_end = min(day_start + 24, n)
        running_day_peak = np.maximum.accumulate(load[day_start:day_end])
        previous_sum = float(np.sum(completed_daily_peaks))
        previous_count = len(completed_daily_peaks)
        for offset, peak_so_far in enumerate(running_day_peak):
            daily_peak_progress[day_start + offset] = (previous_sum + peak_so_far) / (previous_count + 1)
        completed_daily_peaks.append(float(np.max(load[day_start:day_end])))

    return pd.DataFrame(
        {
            "step": steps,
            "day": steps / 24.0,
            "average_daily_peak": daily_peak_progress,
            "one_minus_load_factor": one_minus_load_factor,
            "net_electricity": cumulative_energy,
            "up_ramping": up_ramping,
            "all_time_peak": cumulative_max,
        }
    )


def add_grid_efficiency_score(frame: pd.DataFrame) -> pd.DataFrame:
    """Add supplementary-style grid efficiency score normalized to baseline."""
    if frame.empty or "CityLearn baseline" not in set(frame["controller"]):
        return frame

    terms = ["up_ramping", "one_minus_load_factor", "average_daily_peak", "all_time_peak"]
    baseline = frame[frame["controller"].eq("CityLearn baseline")][["step", *terms]].rename(
        columns={term: f"{term}_base" for term in terms}
    )
    parts = []
    for _, group in frame.groupby("controller", sort=False):
        merged = group.merge(baseline, on="step", how="left")
        ratios = []
        for term in terms:
            base = pd.to_numeric(merged[f"{term}_base"], errors="coerce").to_numpy(dtype=float)
            value = pd.to_numeric(merged[term], errors="coerce").to_numpy(dtype=float)
            ratio = np.divide(
                value,
                base,
                out=np.full(len(merged), np.nan, dtype=float),
                where=np.abs(base) > 1.0e-12,
            )
            if group["controller"].iloc[0] == "CityLearn baseline":
                ratio = np.where(np.isfinite(ratio), ratio, 1.0)
            ratios.append(ratio)
        merged["grid_efficiency_cost"] = np.nanmean(np.vstack(ratios), axis=0)
        parts.append(merged.drop(columns=[f"{term}_base" for term in terms]))
    return pd.concat(parts, ignore_index=True)


def daily_metric_frame(load: np.ndarray, start_day: int = 0) -> pd.DataFrame:
    load = np.asarray(load, dtype=float).reshape(-1)
    load = np.nan_to_num(load, nan=0.0)
    n_days = len(load) // 24
    if n_days == 0:
        return pd.DataFrame(
            columns=["day", "daily_peak", "one_minus_load_factor", "net_electricity", "up_ramping", "all_time_peak"]
        )
    daily = load[: n_days * 24].reshape(n_days, 24)
    peaks = np.nanmax(daily, axis=1)
    means = np.nanmean(daily, axis=1)
    one_minus_lf = 1.0 - np.divide(means, peaks, out=np.full(n_days, np.nan, dtype=float), where=peaks > 0.0)
    return pd.DataFrame(
        {
            "day": np.arange(start_day, start_day + n_days, dtype=float),
            "daily_peak": peaks,
            "one_minus_load_factor": one_minus_lf,
            "net_electricity": np.nansum(daily, axis=1),
            "up_ramping": np.nansum(np.maximum(np.diff(daily, axis=1), 0.0), axis=1),
            "all_time_peak": peaks,
        }
    )


def controller_daily_metric_frames() -> dict[str, pd.DataFrame]:
    month_keys = ("jan", "feb", "mar")
    day_offsets: dict[str, int] = {}
    offset = 0
    for month in month_keys:
        baseline = no_occupant_load("baseline", month)
        n_days = 0 if baseline is None else len(baseline) // 24
        day_offsets[month] = offset
        offset += n_days

    frames: dict[str, pd.DataFrame] = {}
    for controller, label in {
        "baseline": "CityLearn baseline",
        "rbc_safe_pi": "Safe-PI baseline",
    }.items():
        parts = []
        for month in month_keys:
            load = no_occupant_load(controller, month)
            if load is not None:
                parts.append(daily_metric_frame(load, day_offsets[month]))
        if parts:
            frames[label] = pd.concat(parts, ignore_index=True)

    for controller, label in {"mpc": "MPC", "gnu_rl": "GNU-RL", "sac": "SAC"}.items():
        parts = []
        for month in ("feb", "mar"):
            load = no_occupant_load(controller, month)
            if load is not None:
                parts.append(daily_metric_frame(load, day_offsets[month]))
        if parts:
            frames[label] = pd.concat(parts, ignore_index=True)
    return frames


def baseline_district(month: str) -> pd.DataFrame:
    info = MONTHS[month]
    case = info["baseline_case"]
    output_dir = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / case
    path = output_dir / "district_rollout.csv"
    if not path.exists():
        run_citylearn_baseline_district(
            output_dir=output_dir,
            summary_dir=RESULTS / "summaries" / "citylearn_baseline" / "no_occupant" / case,
            simulation_start=info["start"],
            simulation_end=info["end"],
            seed=49,
        )
    return read_csv(path)


def rollout_path(source: str, building: dict, scenario: str, case: str) -> Path:
    return RESULTS / "raw" / source / building["result_id"] / scenario / case / "rollout.csv"


def aggregate(source: str, scenario: str, case: str) -> np.ndarray | None:
    series = []
    for building in BUILDINGS:
        path = rollout_path(source, building, scenario, case)
        if not path.exists():
            return None
        frame = read_csv(path)
        series.append(load_col(frame))
    n = min(len(s) for s in series)
    return np.sum(np.stack([s[:n] for s in series], axis=0), axis=0)


def markdown_table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    return display.to_markdown(index=False)


def no_occupant_case(controller: str, month: str) -> tuple[str, str, str]:
    if controller == "rbc_safe_pi":
        return "rbc_safe_pi", "no_occupant", f"{RBC_PREFIX}_{month}_no_occupant"
    if controller == "mpc":
        return "mpc", "no_occupant", f"{OCC_PREFIX}_mpc_{month}_no_occupant"
    if controller == "gnu_rl":
        return "rlmpc", "no_occupant", f"{OCC_PREFIX}_rlmpc_adaptive_{month}_no_occupant"
    if controller == "sac":
        if month == "jan":
            case = SAC_JAN_EVAL_CASE
        elif month == "feb":
            case = f"{OCC_PREFIX}_sac_jan20_feb_no_occupant"
        else:
            case = f"{OCC_PREFIX}_sac_feb10_mar_no_occupant"
        return "sac_merlin", "no_occupant", case
    raise ValueError(controller)


def bspline_case(controller: str, month: str, occupant: str) -> tuple[str, str, str]:
    if controller == "rbc_safe_pi":
        return "rbc_safe_pi", "occupant_present", f"{RBC_PREFIX}_{month}_{occupant}_bspline_tdyn"
    if controller == "mpc":
        return "mpc", "occupant_present", f"{OCC_PREFIX}_mpc_{month}_{occupant}_bspline_tdyn"
    if controller == "gnu_rl":
        return "rlmpc", "occupant_present", f"{OCC_PREFIX}_rlmpc_adaptive_{month}_{occupant}_bspline_tdyn"
    if controller == "sac":
        case = (
            f"{OCC_PREFIX}_sac_jan20_feb_{occupant}_bspline_tdyn"
            if month == "feb"
            else f"{OCC_PREFIX}_sac_feb10_mar_{occupant}_bspline_tdyn"
        )
        return "sac_merlin", "occupant_present", case
    raise ValueError(controller)


def plot_no_occupant_month(month: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    info = MONTHS[month]
    path = FIG_DIR / f"figure_A_no_occupant_{month}.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    base = baseline_district(month)
    h, y = hourly_mean(load_col(base), info["start"])
    ax.plot(h, y, color="0.25", marker="s", lw=2.0, label="CityLearn baseline")

    specs = [
        ("rbc_safe_pi", "Safe-PI baseline", "#9467bd"),
        ("mpc", "MPC", "#ff7f0e"),
        ("gnu_rl", "GNU-RL", "#2ca02c"),
        ("sac", "SAC", "#d62728"),
    ]
    missing = []
    for controller, label, color in specs:
        source, scenario, case = no_occupant_case(controller, month)
        load = aggregate(source, scenario, case)
        if load is None:
            missing.append(label)
            continue
        h, y = hourly_mean(load, info["start"])
        ax.plot(h, y, marker="o", lw=1.8, label=label, color=color)

    ax.set_title(f"Figure A: no-occupant mean hourly district load - {info['display']}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    if missing:
        ax.text(
            0.99,
            0.02,
            "missing: " + ", ".join(missing),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_no_occupant_feb_mar_combined() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "figure_A_no_occupant_feb_mar_combined.png"
    specs = [
        ("baseline", "CityLearn baseline", "0.25", "s"),
        ("rbc_safe_pi", "Safe-PI baseline", "#9467bd", "o"),
        ("mpc", "MPC", "#ff7f0e", "o"),
        ("gnu_rl", "GNU-RL", "#2ca02c", "o"),
        ("sac", "SAC", "#d62728", "o"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.2), sharex=True)
    for ax, month in zip(axes, ("feb", "mar")):
        info = MONTHS[month]
        missing = []
        for controller, label, color, marker in specs:
            load = no_occupant_load(controller, month)
            if load is None:
                missing.append(label)
                continue
            h, y = hourly_mean(load, info["start"])
            ax.plot(h, y, color=color, marker=marker, lw=1.85, label=label)
        ax.set_title(f"No-occupant mean hourly district load - {info['display']}")
        ax.set_ylabel("Mean district electricity [kWh]")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.25)
        if missing:
            ax.text(
                0.99,
                0.02,
                "missing: " + ", ".join(missing),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="0.35",
            )
    axes[-1].set_xlabel("Hour of day")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def no_occupant_load(controller: str, month: str) -> np.ndarray | None:
    if controller == "baseline":
        return load_col(baseline_district(month))
    source, scenario, case = no_occupant_case(controller, month)
    return aggregate(source, scenario, case)


def deployment_metric_table() -> pd.DataFrame:
    rows = []
    controller_labels = {
        "baseline": "CityLearn baseline",
        "rbc_safe_pi": "Safe-PI baseline",
        "mpc": "MPC",
        "gnu_rl": "GNU-RL",
        "sac": "SAC",
    }

    january_safe_pi = no_occupant_load("rbc_safe_pi", "jan")
    month_keys = ("jan", "feb", "mar")
    for controller, label in controller_labels.items():
        if controller in {"baseline", "rbc_safe_pi"}:
            monthly_loads = [no_occupant_load(controller, month) for month in month_keys]
            visible_from_step = 0
        else:
            monthly_loads = [
                january_safe_pi,
                no_occupant_load(controller, "feb"),
                no_occupant_load(controller, "mar"),
            ]
            visible_from_step = len(january_safe_pi) if january_safe_pi is not None else 0
        if any(load is None for load in monthly_loads):
            continue
        load = np.concatenate([np.asarray(part, dtype=float).reshape(-1) for part in monthly_loads])
        metrics = cumulative_load_metrics(load)
        metrics.insert(0, "controller", label)
        metrics.insert(1, "visible_from_step", visible_from_step)
        rows.append(metrics)

    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    frame = add_grid_efficiency_score(frame)
    out = ANALYSIS / f"{PREFIX}_no_occupant_deployment_kpi_timeseries.csv"
    frame.to_csv(out, index=False)
    return frame


def plot_deployment_metric_trajectory() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = deployment_metric_table()
    path = FIG_DIR / "no_occupant_deployment_kpi_trajectory.png"
    colors = {
        "CityLearn baseline": "0.2",
        "Safe-PI baseline": "#9467bd",
        "MPC": "#ff7f0e",
        "GNU-RL": "#2ca02c",
        "SAC": "#d62728",
    }
    linestyles = {
        "CityLearn baseline": "-",
        "Safe-PI baseline": "-",
        "MPC": "-",
        "GNU-RL": "--",
        "SAC": "-.",
    }
    markers = {
        "CityLearn baseline": None,
        "Safe-PI baseline": None,
        "MPC": "o",
        "GNU-RL": "s",
        "SAC": "^",
    }
    panels = [
        ("average_daily_peak", "Average\ndaily peak", "kWh"),
        ("all_time_peak", "Maximum\npeak demand", "kWh"),
        ("one_minus_load_factor", "One-minus\nload factor", "-"),
        ("net_electricity", "Net electric\ndemand", "kWh"),
        ("up_ramping", "Ramping", "kWh"),
        ("grid_efficiency_cost", "Grid efficiency\ncost", "score"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 9.4), sharex=True)
    axes = axes.ravel()
    for ax, (metric, title, ylabel) in zip(axes, panels):
        endpoints: list[tuple[str, float, float]] = []
        for controller, group in data.groupby("controller", sort=False):
            group = group.sort_values("step")
            visible_from_step = int(group["visible_from_step"].iloc[0])
            draw = group[group["step"] >= visible_from_step]
            xs = pd.to_numeric(draw["day"], errors="coerce").to_numpy(dtype=float)
            ys = pd.to_numeric(draw[metric], errors="coerce").to_numpy(dtype=float)
            if len(xs) == 0:
                continue
            ax.plot(
                xs,
                ys,
                lw=1.95 if controller in {"MPC", "GNU-RL", "SAC"} else 1.75,
                color=colors.get(controller, None),
                linestyle=linestyles.get(controller, "-"),
                marker=markers.get(controller),
                markevery=max(len(xs) // 5, 1),
                ms=3.2,
                mec=colors.get(controller, None),
                mfc="white" if controller in {"MPC", "GNU-RL", "SAC"} else colors.get(controller, None),
                label=controller,
                zorder=4 if controller in {"MPC", "GNU-RL", "SAC"} else 3,
            )
            endpoints.append((controller, float(xs[-1]), float(ys[-1])))
        ax.axvline(31.0, color="0.4", lw=0.8, ls="--", alpha=0.65)
        ax.axvline(59.0, color="0.4", lw=0.8, ls="--", alpha=0.65)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 90.0)
        ax.set_xticks([0.0, 31.0, 59.0, 90.0], ["Jan 1", "Feb 1", "Mar 1", "End Mar"])
        ax.grid(alpha=0.25)
        if metric in {"all_time_peak", "net_electricity"} and endpoints:
            finite = [(c, x, y) for c, x, y in endpoints if np.isfinite(y)]
            finite.sort(key=lambda item: item[2])
            offsets: dict[str, int] = {}
            for idx, (controller, _, _) in enumerate(finite):
                offsets[controller] = (idx - (len(finite) - 1) / 2.0) * 7
            for controller, x, y in finite:
                ax.annotate(
                    controller.replace("CityLearn ", "CL "),
                    xy=(x, y),
                    xytext=(5, offsets.get(controller, 0)),
                    textcoords="offset points",
                    color=colors.get(controller, "0.2"),
                    fontsize=7,
                    va="center",
                    ha="left",
                    clip_on=False,
                )
        if metric == "all_time_peak":
            advanced = data[data["controller"].isin(["MPC", "GNU-RL", "SAC"])]
            final = advanced.sort_values("step").groupby("controller", sort=False).tail(1)
            if len(final) >= 2 and final["all_time_peak"].nunique(dropna=True) == 1:
                value = final["all_time_peak"].iloc[0]
                ax.text(
                    0.03,
                    0.04,
                    f"MPC/GNU-RL/SAC overlap at {value:.1f} kWh",
                    transform=ax.transAxes,
                    fontsize=7,
                    color="0.35",
                    ha="left",
                    va="bottom",
                )
    for ax in axes[-2:]:
        ax.set_xlabel("Deployment time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("No-occupant deployment-type cumulative KPI trajectory", y=1.08, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_daily_normalized_deployment_metrics() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frames = controller_daily_metric_frames()
    path = FIG_DIR / "no_occupant_deployment_daily_kpi_normalized.png"
    csv_path = ANALYSIS / f"{PREFIX}_no_occupant_deployment_daily_kpi_normalized.csv"
    if "CityLearn baseline" not in frames:
        raise FileNotFoundError("CityLearn baseline daily metric frame missing.")

    baseline = frames["CityLearn baseline"].rename(
        columns={
            "daily_peak": "daily_peak_base",
            "one_minus_load_factor": "one_minus_load_factor_base",
            "net_electricity": "net_electricity_base",
            "up_ramping": "up_ramping_base",
            "all_time_peak": "all_time_peak_base",
        }
    )
    norm_terms = [
        ("daily_peak", "Average\ndaily peak"),
        ("one_minus_load_factor", "One-minus\nload factor"),
        ("net_electricity", "Net electric\ndemand"),
        ("up_ramping", "Ramping"),
        ("all_time_peak", "Maximum\npeak demand"),
    ]
    panels = [
        ("daily_peak", "Average\ndaily peak"),
        ("one_minus_load_factor", "One-minus\nload factor"),
        ("net_electricity", "Net electric\ndemand"),
        ("up_ramping", "Ramping"),
        ("grid_efficiency_cost", "Grid efficiency\ncost"),
    ]
    rows = []
    colors = {
        "Safe-PI baseline": "#9467bd",
        "MPC": "#ff7f0e",
        "GNU-RL": "#2ca02c",
        "SAC": "#d62728",
    }
    for controller, frame in frames.items():
        if controller == "CityLearn baseline":
            continue
        merged = frame.merge(baseline[["day", *[f"{m}_base" for m, _ in norm_terms]]], on="day", how="inner")
        ratio_columns = []
        for metric, _ in norm_terms:
            base_col = f"{metric}_base"
            merged[f"{metric}_norm"] = np.divide(
                pd.to_numeric(merged[metric], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(merged[base_col], errors="coerce").to_numpy(dtype=float),
                out=np.full(len(merged), np.nan, dtype=float),
                where=pd.to_numeric(merged[base_col], errors="coerce").to_numpy(dtype=float) != 0.0,
            )
            ratio_columns.append(f"{metric}_norm")
        merged["grid_efficiency_cost_norm"] = np.nanmean(
            merged[
                [
                    "up_ramping_norm",
                    "one_minus_load_factor_norm",
                    "daily_peak_norm",
                    "all_time_peak_norm",
                ]
            ].to_numpy(dtype=float),
            axis=1,
        )
        merged.insert(0, "controller", controller)
        rows.append(merged[["controller", "day", *ratio_columns, "grid_efficiency_cost_norm"]])
    normalized = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    normalized.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(5, 1, figsize=(9.5, 10.0), sharex=True)
    for ax, (metric, title) in zip(axes, panels):
        ax.axhline(1.0, color="0.1", lw=1.1, ls="--", label="CityLearn baseline")
        for controller in ("Safe-PI baseline", "MPC", "GNU-RL", "SAC"):
            data = normalized[normalized["controller"] == controller].sort_values("day")
            if data.empty:
                continue
            ax.plot(
                data["day"],
                data[f"{metric}_norm"],
                lw=1.6,
                color=colors[controller],
                label=controller,
            )
        ax.axvline(31.0, color="0.45", lw=0.8, ls="--", alpha=0.65)
        ax.axvline(59.0, color="0.45", lw=0.8, ls="--", alpha=0.65)
        ax.set_ylabel(title)
        ax.grid(alpha=0.23)
        ax.set_xlim(0.0, 90.0)
    axes[-1].set_xticks([0.0, 31.0, 59.0, 90.0], ["Jan 1", "Feb 1", "Mar 1", "End Mar"])
    axes[-1].set_xlabel("Deployment time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("No-occupant daily KPI trajectory normalized to CityLearn baseline", y=1.055, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_time_bias_windows() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "time_bias_windows_vs_baseline.png"
    base = baseline_district("feb")
    h, y = hourly_mean(load_col(base), MONTHS["feb"]["start"])
    hour_labels = np.asarray(h, dtype=int) + 1
    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    preheat_blocks = [(1, 5), (14, 17)]
    reduce_blocks = [(7, 10), (18, 22)]
    for start, end in preheat_blocks:
        ax.axvspan(start - 0.5, end + 0.5, color="#2ca02c", alpha=0.18, lw=0)
    for start, end in reduce_blocks:
        ax.axvspan(start - 0.5, end + 0.5, color="#d62728", alpha=0.16, lw=0)
    ax.plot(hour_labels, y, color="0.2", marker="s", ms=4.5, lw=1.8, label="CityLearn baseline")
    ax.set_title("Time-bias windows over baseline daily load", fontsize=11)
    ax.set_xlabel("Hour label (1-24)", fontsize=10)
    ax.set_ylabel("Mean district electricity [kWh]", fontsize=10)
    ax.set_xlim(0.5, 24.5)
    ax.set_xticks(range(1, 25))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(True, axis="x", alpha=0.10)
    ax.plot([], [], color="#2ca02c", lw=6, alpha=0.35, label="+1 C preheat")
    ax.plot([], [], color="#d62728", lw=6, alpha=0.35, label="-1 C reduce")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=260)
    plt.close(fig)
    return path


def plot_occupant_impact(controller: str, month: str) -> Path | None:
    display = {
        "rbc_safe_pi": "Safe-PI baseline",
        "mpc": "MPC",
        "gnu_rl": "GNU-RL",
        "sac": "SAC",
    }[controller]
    info = MONTHS[month]
    source, scenario, case = no_occupant_case(controller, month)
    no_occ = aggregate(source, scenario, case)
    if no_occ is None:
        return None

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"figure_B_{controller}_{month}_occupant_impact.png"
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    h, y = hourly_mean(no_occ, info["start"])
    ax.plot(h, y, color="0.2", marker="s", lw=2.1, label="no occupant")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    missing = []
    for (occupant, label), color in zip(OCCUPANTS, colors):
        source, scenario, case = bspline_case(controller, month, occupant)
        load = aggregate(source, scenario, case)
        if load is None:
            missing.append(label)
            continue
        h, y = hourly_mean(load, info["start"])
        ax.plot(h, y, marker="o", lw=1.7, label=label, color=color)
    ax.set_title(f"Figure B: occupant impact on {display} load - {info['display']}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    if missing:
        ax.text(
            0.99,
            0.02,
            "missing: " + ", ".join(missing),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def rbc_case(month: str, mode: str, occupant: str | None) -> tuple[str, str, str]:
    if mode == "no_occupant":
        return "rbc_safe_pi", "no_occupant", f"{RBC_PREFIX}_{month}_no_occupant"
    if occupant is None:
        raise ValueError("occupant is required for occupant-present Safe-PI/RBC cases.")
    return "rbc_safe_pi", "occupant_present", f"{RBC_PREFIX}_{month}_{occupant}_{mode}"


def rbc_rollout(building: dict, month: str, mode: str, occupant: str | None) -> pd.DataFrame | None:
    source, scenario, case = rbc_case(month, mode, occupant)
    path = rollout_path(source, building, scenario, case)
    if not path.exists():
        return None
    return read_csv(path)


def comfort_grid_path(month: str, mode: str, occupant: str | None) -> Path:
    suffix = "no_occupant" if occupant is None else f"{occupant}_{mode}"
    return FIG_DIR / f"safe_pi_comfort_{month}_{suffix}.png"


def set_local_temperature_ylim(ax, *arrays: np.ndarray, min_span: float = 5.0) -> None:
    finite = []
    for array in arrays:
        values = np.asarray(array, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if len(values):
            finite.append(values)
    if not finite:
        return
    values = np.concatenate(finite)
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, float(min_span))
    pad = max(0.4, 0.08 * span)
    ax.set_ylim(low - pad, high + pad)


def plot_safe_pi_comfort_grid(month: str, mode: str, occupant: str | None) -> Path | None:
    frames = [rbc_rollout(building, month, mode, occupant) for building in BUILDINGS]
    if any(frame is None for frame in frames):
        return None

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = comfort_grid_path(month, mode, occupant)
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building, data in zip(axes, BUILDINGS, frames):
        t = pd.to_numeric(data["time_step"], errors="coerce").to_numpy(dtype=float)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        ref = gated_reference(data)
        track = pd.to_numeric(data.get("tracking_setpoint", data["reference_setpoint"]), errors="coerce").to_numpy(dtype=float)
        n = min(len(t), len(tin), len(ref), len(track))
        band_low = ref[:n] - 2.0
        band_high = ref[:n] + 2.0
        ax.fill_between(t[:n], band_low, band_high, color="0.78", alpha=0.55, linewidth=0, label="comfort band")
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], track[:n], lw=0.75, color="tab:orange", label="setpoint")
        if mode == "bspline_tdyn" and "tdyn" in data.columns:
            tdyn = pd.to_numeric(data["tdyn"], errors="coerce").to_numpy(dtype=float)
            n2 = min(n, len(tdyn))
            ax.plot(t[:n2], tdyn[:n2], lw=0.75, color="black", label="Tdyn")
            set_local_temperature_ylim(ax, tin[:n], track[:n], band_low, band_high, tdyn[:n2])
        else:
            set_local_temperature_ylim(ax, tin[:n], track[:n], band_low, band_high)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    display_month = MONTHS[month]["display"]
    if mode == "no_occupant":
        title_mode = "no occupant"
    elif mode == "without_tdyn":
        title_mode = f"occupant overrides without Tdyn | {occupant}"
    else:
        title_mode = f"B-spline Tdyn | {occupant}"
    fig.suptitle(f"Safe-PI/RBC plot_comfort | {display_month} | {title_mode}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def select_safe_pi_action_day(month: str = "feb") -> tuple[dict, pd.DataFrame]:
    """Select a Safe-PI day that exposes saturation-driven hot/cold cycling."""
    best: dict | None = None
    best_data: pd.DataFrame | None = None
    for building in BUILDINGS:
        data = rbc_rollout(building, month, "no_occupant", None)
        if data is None or "action" not in data.columns:
            continue
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        ref = gated_reference(data)
        action = pd.to_numeric(data["action"], errors="coerce").to_numpy(dtype=float)
        n = min(len(tin), len(ref), len(action))
        n_days = n // 24
        if n_days == 0:
            continue
        for day in range(n_days):
            sl = slice(day * 24, (day + 1) * 24)
            tin_day = tin[sl]
            ref_day = ref[sl]
            action_day = action[sl]
            cold = ref_day - 2.0 > tin_day
            hot = tin_day > ref_day + 2.0
            cold_hours = int(np.count_nonzero(cold))
            hot_hours = int(np.count_nonzero(hot))
            action_switch = float(np.nansum(np.abs(np.diff(action_day))))
            saturated = float(np.nanmean((action_day <= 0.05) | (action_day >= 0.95)))
            edge_crossings = int(
                np.count_nonzero(
                    ((action_day[:-1] <= 0.05) & (action_day[1:] >= 0.95))
                    | ((action_day[:-1] >= 0.95) & (action_day[1:] <= 0.05))
                )
            )
            both = cold_hours > 0 and hot_hours > 0
            score = (
                8.0 * float(both)
                + action_switch
                + 0.25 * float(cold_hours + hot_hours)
                + 2.0 * saturated
                + float(edge_crossings)
            )
            if best is None or score > float(best["score"]):
                best = {
                    "month": MONTHS[month]["display"],
                    "building": building["display"],
                    "building_id": building["result_id"],
                    "day": day + 1,
                    "cold_hours": cold_hours,
                    "hot_hours": hot_hours,
                    "action_switch": action_switch,
                    "saturated_fraction": saturated,
                    "edge_crossings": edge_crossings,
                    "score": score,
                }
                best_data = data.iloc[sl].reset_index(drop=True).copy()
    if best is None or best_data is None:
        raise FileNotFoundError(f"No Safe-PI action rollouts found for {MONTHS[month]['display']}.")
    return best, best_data


def plot_safe_pi_no_occupant_action_day(month: str = "feb") -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    selection, data = select_safe_pi_action_day(month)
    path = FIG_DIR / f"safe_pi_no_occupant_action_day_{month}.png"
    pd.DataFrame([selection]).to_csv(
        ANALYSIS / f"{PREFIX}_safe_pi_action_day_selection_{month}.csv",
        index=False,
    )

    hours = np.arange(len(data), dtype=float)
    tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    ref = gated_reference(data)
    track = pd.to_numeric(
        data.get("tracking_setpoint", data.get("reference_setpoint", data["baseline_setpoint"])),
        errors="coerce",
    ).to_numpy(dtype=float)
    action = pd.to_numeric(data["action"], errors="coerce").to_numpy(dtype=float)
    source = (
        data["action_source"].astype(str).to_numpy()
        if "action_source" in data.columns
        else np.full(len(data), "", dtype=object)
    )
    n = min(len(hours), len(tin), len(ref), len(track), len(action), len(source))
    hours, tin, ref, track, action, source = (
        hours[:n],
        tin[:n],
        ref[:n],
        track[:n],
        action[:n],
        source[:n],
    )

    cold = ref - 2.0 > tin
    hot = tin > ref + 2.0
    cold_guard = np.char.find(source.astype(str), "cold") >= 0
    hot_guard = np.char.find(source.astype(str), "hot") >= 0

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.6, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_t, ax_u = axes
    ax_t.fill_between(hours, ref - 2.0, ref + 2.0, color="0.75", alpha=0.32, label="comfort band")
    ax_t.plot(hours, tin, color="#1f77b4", lw=2.0, marker="o", ms=3.0, label="$T_{in}$")
    ax_t.plot(hours, track, color="#ff7f0e", lw=1.7, ls="--", label="tracked setpoint")
    if np.any(cold):
        ax_t.scatter(hours[cold], tin[cold], color="#1f77b4", edgecolor="white", s=42, zorder=5, label="cold violation")
    if np.any(hot):
        ax_t.scatter(hours[hot], tin[hot], color="#d62728", edgecolor="white", s=42, zorder=5, label="hot violation")
    ax_t.set_ylabel("Indoor temp [C]")
    ax_t.grid(alpha=0.22)
    ax_t.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

    ax_u.step(hours, action, where="post", color="#9467bd", lw=2.2, label="heat-pump action $u$")
    if np.any(cold_guard):
        ax_u.scatter(hours[cold_guard], action[cold_guard], color="#1f77b4", marker="^", s=45, label="cold guard")
    if np.any(hot_guard):
        ax_u.scatter(hours[hot_guard], action[hot_guard], color="#d62728", marker="v", s=45, label="hot guard")
    ax_u.axhline(0.0, color="0.25", lw=0.8, alpha=0.5)
    ax_u.axhline(1.0, color="0.25", lw=0.8, alpha=0.5)
    ax_u.set_ylim(-0.06, 1.06)
    ax_u.set_xlim(0.0, max(23.0, float(n - 1)))
    ax_u.set_xticks(range(0, 24, 2))
    ax_u.set_ylabel("Action u")
    ax_u.set_xlabel("Hour of selected day")
    ax_u.grid(alpha=0.22)
    ax_u.legend(ncol=3, fontsize=8, frameon=False, loc="upper left")

    fig.suptitle(
        "Safe-PI/RBC no-occupant action trace: "
        f"{selection['building']}, {selection['month']} day {selection['day']}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def no_occupant_action_frame(controller: str, month: str) -> pd.DataFrame:
    rows = []
    if controller == "baseline":
        for hour in range(24):
            rows.append({"controller": "CityLearn baseline", "hour": hour, "action": 0.0})
        return pd.DataFrame(rows)

    for building in BUILDINGS:
        if controller == "rbc_safe_pi":
            data = rbc_rollout(building, month, "no_occupant", None)
        else:
            source, scenario, case = no_occupant_case(controller, month)
            path = rollout_path(source, building, scenario, case)
            data = read_csv(path) if path.exists() else None
        if data is None or "action" not in data.columns:
            continue
        action = pd.to_numeric(data["action"], errors="coerce").to_numpy(dtype=float)
        hours = np.arange(len(action), dtype=int) % 24
        for hour, value in zip(hours, action):
            if np.isfinite(value):
                rows.append(
                    {
                        "controller": {
                            "rbc_safe_pi": "Safe-PI/RBC",
                            "mpc": "MPC",
                            "gnu_rl": "GNU-RL adaptive",
                            "sac": "SAC",
                        }[controller],
                        "building": building["display"],
                        "hour": int(hour),
                        "action": float(value),
                    }
                )
    return pd.DataFrame(rows)


def plot_no_occupant_march_average_action() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "no_occupant_march_average_action_by_controller.png"
    frames = [
        no_occupant_action_frame(controller, "mar")
        for controller in ("baseline", "rbc_safe_pi", "mpc", "gnu_rl", "sac")
    ]
    raw = pd.concat(frames, ignore_index=True)
    summary = (
        raw.groupby(["controller", "hour"], as_index=False)
        .agg(action_mean=("action", "mean"), action_std=("action", "std"), samples=("action", "count"))
        .sort_values(["controller", "hour"])
    )
    summary.to_csv(ANALYSIS / f"{PREFIX}_no_occupant_march_average_action.csv", index=False)

    colors = {
        "CityLearn baseline": "0.25",
        "Safe-PI/RBC": "#9467bd",
        "MPC": "#ff7f0e",
        "GNU-RL adaptive": "#2ca02c",
        "SAC": "#d62728",
    }
    styles = {
        "CityLearn baseline": "--",
        "Safe-PI/RBC": "-",
        "MPC": "-",
        "GNU-RL adaptive": "-",
        "SAC": "-",
    }
    markers = {
        "CityLearn baseline": None,
        "Safe-PI/RBC": "o",
        "MPC": "s",
        "GNU-RL adaptive": "^",
        "SAC": "D",
    }
    order = ["CityLearn baseline", "Safe-PI/RBC", "MPC", "GNU-RL adaptive", "SAC"]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    for controller in order:
        data = summary[summary["controller"].eq(controller)]
        if data.empty:
            continue
        ax.plot(
            data["hour"],
            data["action_mean"],
            color=colors[controller],
            ls=styles[controller],
            marker=markers[controller],
            ms=4.0,
            lw=2.0,
            label=controller,
        )
    ax.set_title("No-occupant March average day: heat-pump action")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean action $u$")
    ax.set_xlim(0, 23)
    ax.set_ylim(-0.03, 0.55)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.24)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def safe_pi_kpi_rows() -> pd.DataFrame:
    rows = []
    for month in ("jan", "feb", "mar"):
        cases = [("no_occupant", None)]
        if month in {"feb", "mar"}:
            cases.extend(("without_tdyn", occ) for occ, _ in OCCUPANTS)
            cases.extend(("bspline_tdyn", occ) for occ, _ in OCCUPANTS)
        for mode, occupant in cases:
            for building in BUILDINGS:
                data = rbc_rollout(building, month, mode, occupant)
                if data is None:
                    continue
                tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
                ref = gated_reference(data)
                track = pd.to_numeric(data.get("tracking_setpoint", data["reference_setpoint"]), errors="coerce").to_numpy(dtype=float)
                load = load_col(data)
                n = min(len(tin), len(ref), len(track), len(load))
                tin = tin[:n]
                ref = ref[:n]
                track = track[:n]
                load = load[:n]
                cold = np.maximum(ref - 2.0 - tin, 0.0)
                hot = np.maximum(tin - ref - 2.0, 0.0)
                action = pd.to_numeric(data.get("action", pd.Series(np.nan, index=data.index)), errors="coerce").to_numpy(dtype=float)[:n]
                if mode == "no_occupant":
                    overrides = 0.0
                elif "occupant_feedback" in data.columns:
                    feedback = pd.to_numeric(data["occupant_feedback"], errors="coerce").fillna(0.0).to_numpy(dtype=float)[:n]
                    overrides = float(np.count_nonzero(np.abs(feedback) > 0.0))
                else:
                    overrides = np.nan
                rows.append(
                    {
                        "month": MONTHS[month]["display"],
                        "mode": mode,
                        "occupant": occupant or "none",
                        "building": building["display"],
                        "energy_kwh": float(np.sum(load)),
                        "disc_%": float(np.mean((cold > 0.0) | (hot > 0.0)) * 100.0),
                        "cold_%": float(np.mean(cold > 0.0) * 100.0),
                        "hot_%": float(np.mean(hot > 0.0) * 100.0),
                        "kelvin_hours": float(np.sum(cold + hot)),
                        "mae_to_tracked_setpoint": float(np.mean(np.abs(tin - track))),
                        "action_mean": float(np.nanmean(action)),
                        "Tin_min": float(np.nanmin(tin)),
                        "Tin_max": float(np.nanmax(tin)),
                        "overrides": overrides,
                    }
                )
    return pd.DataFrame(rows)


def save_safe_pi_kpis() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    rows = safe_pi_kpi_rows()
    full_path = ANALYSIS / f"{RBC_PREFIX}_kpis_by_building.csv"
    mean_path = ANALYSIS / f"{RBC_PREFIX}_kpis_mean.csv"
    rows.to_csv(full_path, index=False)
    mean = (
        rows.groupby(["month", "mode", "occupant"], as_index=False)
        .agg(
            buildings=("building", "count"),
            energy_kwh=("energy_kwh", "mean"),
            disc_pct=("disc_%", "mean"),
            cold_pct=("cold_%", "mean"),
            hot_pct=("hot_%", "mean"),
            kelvin_hours=("kelvin_hours", "mean"),
            mae_to_tracked_setpoint=("mae_to_tracked_setpoint", "mean"),
            action_mean=("action_mean", "mean"),
            Tin_min=("Tin_min", "min"),
            Tin_max=("Tin_max", "max"),
            overrides=("overrides", "sum"),
        )
        .sort_values(["month", "mode", "occupant"])
        .rename(columns={"disc_pct": "disc_%", "cold_pct": "cold_%", "hot_pct": "hot_%"})
    )
    mean.to_csv(mean_path, index=False)
    return rows, mean, full_path, mean_path


def compact_safe_pi_tables(mean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comfort = mean[
        [
            "month",
            "mode",
            "occupant",
            "energy_kwh",
            "disc_%",
            "cold_%",
            "hot_%",
            "kelvin_hours",
            "mae_to_tracked_setpoint",
        ]
    ].rename(
        columns={
            "month": "mon",
            "occupant": "occ",
            "energy_kwh": "kWh",
            "disc_%": "disc",
            "cold_%": "cold",
            "hot_%": "hot",
            "kelvin_hours": "KH",
            "mae_to_tracked_setpoint": "MAE",
        }
    )
    operation = mean[
        ["month", "mode", "occupant", "action_mean", "Tin_min", "Tin_max", "overrides"]
    ].rename(
        columns={
            "month": "mon",
            "occupant": "occ",
            "action_mean": "u_mean",
            "Tin_min": "Tmin",
            "Tin_max": "Tmax",
            "overrides": "ovr",
        }
    )
    return comfort, operation


def plot_sac_jan_eval_comfort_grid() -> Path | None:
    frames = []
    for building in BUILDINGS:
        path = rollout_path("sac_merlin", building, "no_occupant", SAC_JAN_EVAL_CASE)
        if not path.exists():
            return None
        frames.append(read_csv(path))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "sac_jan20_eval_january_comfort_no_occupant.png"
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building, data in zip(axes, BUILDINGS, frames):
        t = pd.to_numeric(data["time_step"], errors="coerce").to_numpy(dtype=float)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        if "effective_setpoint" in data.columns:
            tsp = pd.to_numeric(data["effective_setpoint"], errors="coerce").to_numpy(dtype=float)
        else:
            tsp = pd.to_numeric(data["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
        n = min(len(t), len(tin), len(tsp))
        band_low = tsp[:n] - 2.0
        band_high = tsp[:n] + 2.0
        ax.fill_between(t[:n], band_low, band_high, color="0.78", alpha=0.55, linewidth=0, label="comfort band")
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.75, color="tab:orange", label="setpoint")
        set_local_temperature_ylim(ax, tin[:n], tsp[:n], band_low, band_high)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("SAC Jan-trained actor deterministic deployment | January | no occupant", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def image_cell(path: Path, title: str):
    notebook_path = Path("..") / path
    return nbf.v4.new_markdown_cell(
        f"<!-- {MARKER} -->\n\n**{title}**\n\n![]({notebook_path.as_posix()})"
    )


def append_notebooks(cells):
    for notebook in NOTEBOOKS:
        nb = nbf.read(notebook, as_version=4)
        nb.cells = [c for c in nb.cells if MARKER not in c.get("source", "")]
        nb.cells.extend(cells)
        nbf.write(nb, notebook)


def main() -> None:
    fig_a = [plot_no_occupant_month(month) for month in ("jan", "feb", "mar")]
    fig_a_feb_mar = plot_no_occupant_feb_mar_combined()
    deployment_kpis = plot_deployment_metric_trajectory()
    daily_kpis = plot_daily_normalized_deployment_metrics()
    march_action = plot_no_occupant_march_average_action()
    bias = plot_time_bias_windows()
    fig_b = []
    for month in ("feb", "mar"):
        for controller in ("rbc_safe_pi", "mpc", "gnu_rl", "sac"):
            path = plot_occupant_impact(controller, month)
            if path is not None:
                fig_b.append((controller, month, path))
    safe_pi_comfort = []
    for month, mode, occupant in RBC_COMFORT_CASES:
        path = plot_safe_pi_comfort_grid(month, mode, occupant)
        if path is not None:
            safe_pi_comfort.append((month, mode, occupant, path))
    sac_jan_comfort = plot_sac_jan_eval_comfort_grid()
    _, safe_pi_mean, safe_pi_full_path, safe_pi_mean_path = save_safe_pi_kpis()
    safe_pi_comfort_table, safe_pi_operation_table = compact_safe_pi_tables(safe_pi_mean)

    cells = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

Thesis-clean load-shape figures.

Figure A keeps the no-occupant comparison readable: CityLearn baseline, the new
Safe-PI/RBC reference baseline, MPC, GNU-RL, and SAC. Missing curves are marked
on the plot instead of silently mixing old-cost artifacts.

The February/March Figure A panel stacks the two deployment months vertically so
the load-shifting behavior can be compared directly without the January
jump-start period.

The January SAC curve is now a deterministic deployment of the exact actor
trained on January (`1` Safe-PI/MERLIN replay episode + `19` SAC reward-training
episodes), evaluated back on January with no exploration and no additional
training.

The Safe-PI/RBC baseline is **not** CityLearn's `BaselineAgent`. It is a separate
simple controller that tracks:

`T_control_ref = Tref + bias`

with:

`Tref = Tschedule` for no occupant,
`Tref = effective occupant setpoint` for raw occupant overrides, and
`Tref = (1-occ_bin)*Tschedule + occ_bin*Tdyn` for B-spline Tdyn.

The time-bias windows are `+1 C` in hours 1-5 and 14-17, `-1 C` in hours 7-10
and 18-22, using CE1 1..24 hour labels.

Figure B keeps the occupant question separate: one controller per plot, comparing
`no_occupant` against `occ1` through `occ4` under the B-spline Tdyn branch.

The Safe-PI/RBC comfort grids below use the same compact `plot_comfort` style:
blue is `Tin`, orange is the tracked setpoint (`Tref+bias`), gray is the
`Tref +/- 2 C` guard band, and the B-spline Tdyn cases also show `Tdyn` as a
solid black line.

The cumulative deployment-type KPI trajectory is computed at every rollout
timestep from the district load traces. January contains only CityLearn baseline
and Safe-PI. MPC, GNU-RL, and SAC are not plotted in January; their cumulative
trajectories start at the February boundary after sharing the January Safe-PI
history.

The non-cumulative daily KPI trajectory computes each day's KPI from that day's
24-hour load profile and normalizes it against CityLearn baseline for the same
day. The dashed black line is therefore exactly the CityLearn baseline score of
`1.0`.
"""
        )
    ]
    cells.append(
        nbf.v4.new_markdown_cell(
            f"""<!-- {MARKER} -->

**Safe-PI/RBC KPI Summary**

Mean comfort/energy by month/mode/occupant:

{markdown_table(safe_pi_comfort_table, digits=3, max_rows=20)}

Mean operation/override by month/mode/occupant:

{markdown_table(safe_pi_operation_table, digits=3, max_rows=20)}

Saved full per-building table:

- `{safe_pi_full_path.as_posix()}`
- `{safe_pi_mean_path.as_posix()}`
"""
        )
    )
    cells.append(image_cell(bias, "Time-bias window justification"))
    cells.append(image_cell(deployment_kpis, "No-occupant cumulative deployment-type KPI trajectory"))
    cells.append(image_cell(daily_kpis, "No-occupant daily normalized deployment-type KPI trajectory"))
    cells.append(image_cell(march_action, "No-occupant March average action by controller"))
    cells.append(image_cell(fig_a_feb_mar, "Figure A - February/March no-occupant controller comparison"))
    for month, path in zip(("January", "February", "March"), fig_a):
        cells.append(image_cell(path, f"Figure A - {month} no-occupant controller comparison"))
    if sac_jan_comfort is not None:
        cells.append(image_cell(sac_jan_comfort, "SAC Jan-trained deterministic January plot_comfort"))
    for controller, month, path in fig_b:
        label = {
            "rbc_safe_pi": "Safe-PI baseline",
            "mpc": "MPC",
            "gnu_rl": "GNU-RL",
            "sac": "SAC",
        }[controller]
        cells.append(image_cell(path, f"Figure B - {MONTHS[month]['display']} {label} occupant impact"))
    for month, mode, occupant, path in safe_pi_comfort:
        title_month = MONTHS[month]["display"]
        title_mode = "no-occupant" if occupant is None else f"{occupant} B-spline Tdyn"
        cells.append(image_cell(path, f"Safe-PI/RBC plot_comfort - {title_month} {title_mode}"))

    append_notebooks(cells)
    print("Appended thesis-clean load figures:")
    extra_paths = [] if sac_jan_comfort is None else [sac_jan_comfort]
    for path in [bias, deployment_kpis, daily_kpis, march_action, fig_a_feb_mar, *fig_a, *extra_paths, *[p for _, _, p in fig_b], *[p for _, _, _, p in safe_pi_comfort]]:
        print(path)


if __name__ == "__main__":
    main()
