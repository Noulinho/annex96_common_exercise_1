#!/usr/bin/env python
"""Append MPC-only district-target tracking diagnostics to the MPC notebook."""

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
from src.district_target import load_ce1_district_target_profiles
from src.rewards import daily_window_shift_profile


LABEL = "ce1_mpc_district_target_tw_c035_d035m055e075_du15_u01"
PREVIOUS_LABEL = "ce1_daily_windows_w04_04_01_01"
OLD_DISTRICT_LABEL = "ce1_mpc_district_target_c04_d03_du25_u01"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

WEIGHTS = {
    "label": LABEL,
    "scenario": "no_occupant",
    "controller": "MPC only",
    "comfort_mode": "single heating setpoint tracking",
    "district_term": "time-weighted CE1 district target tracking",
    "comfort_weight": 0.35,
    "district_target_weight_default": 0.35,
    "district_target_weight_morning_6_10": 0.55,
    "district_target_weight_evening_20_23": 0.75,
    "delta_u_weight": 15.0,
    "u_weight": 0.1,
    "comfort_scale": 4.0,
    "district_target_scale_mode": "target_max",
    "district_target_weight_profile": "morning_evening_boost",
    "load_shape_profile": "district_target",
}

COMPARISON_RUNS = [
    (PREVIOUS_LABEL, "MPC daily-window w04/04/01/01"),
    (OLD_DISTRICT_LABEL, "MPC district-target c04/d03/du25/u01"),
    (LABEL, "MPC time-weighted target c035/d035-055-075/du15/u01"),
]

WINDOWS = {
    "midnight_preheat_1_3": [1, 2, 3],
    "morning_curtail_6_10": [6, 7, 8, 9, 10],
    "afternoon_preheat_13_17": [13, 14, 15, 16, 17],
    "evening_curtail_20_23": [20, 21, 22, 23],
}


def _rollout_path(label: str, building_id: str) -> Path:
    return RESULTS_DIR / "raw" / "mpc" / building_id / "no_occupant" / label / "rollout.csv"


def _baseline_rollout_path(building_id: str) -> Path:
    return RESULTS_DIR / "raw" / "mpc" / building_id / "no_occupant" / "baseline" / "rollout.csv"


def _metrics_path(label: str, building_id: str) -> Path:
    return RESULTS_DIR / "summaries" / "mpc" / building_id / "no_occupant" / f"{label}_metrics.json"


def _load_rollouts(label: str) -> dict[str, pd.DataFrame]:
    rollouts = {}
    missing = []
    for building in BUILDINGS:
        path = _rollout_path(label, building["result_id"])
        if not path.exists():
            missing.append(path)
            continue
        rollouts[building["result_id"]] = pd.read_csv(path)
    if missing:
        missing_text = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            missing_text += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing MPC rollouts for {label}:\n{missing_text}")
    return rollouts


def _load_metrics(label: str) -> pd.DataFrame:
    rows = []
    missing = []
    for building in BUILDINGS:
        path = _metrics_path(label, building["result_id"])
        if not path.exists():
            missing.append(path)
            continue
        row = json.loads(path.read_text())
        row.setdefault("building_id", building["result_id"])
        rows.append(row)
    if missing:
        missing_text = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            missing_text += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing MPC metrics for {label}:\n{missing_text}")
    return pd.DataFrame(rows)


def _baseline_district() -> pd.DataFrame:
    path = RESULTS_DIR / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    target = pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float),
            "target": target,
            "controller": "CityLearn baseline",
        }
    )


def _district_from_rollouts(display: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        series.append(
            pd.to_numeric(data["net_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    n = min(len(s) for s in series)
    load = np.sum(np.stack([s[:n] for s in series], axis=0), axis=0)
    return pd.DataFrame(
        {
            "time_step": np.arange(n, dtype=int),
            "load": load,
            "controller": display,
        }
    )


def _daily_profile(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    daily["hour_label"] = daily["time_step"] % 24 + 1
    return daily.groupby(["controller", "hour_label"], as_index=False)["load"].mean()


def _target_daily_profile(baseline: pd.DataFrame) -> pd.DataFrame:
    target = baseline[["time_step", "target"]].copy()
    target["hour_label"] = target["time_step"] % 24 + 1
    return target.groupby("hour_label", as_index=False)["target"].mean()


def _district_target_kpis(load: np.ndarray, target: np.ndarray) -> dict[str, float]:
    n = min(len(load), len(target))
    load = np.asarray(load[:n], dtype=float)
    target = np.asarray(target[:n], dtype=float)
    error = load - target
    above = np.maximum(error, 0.0)
    return {
        "district_target_rmse_kw": float(np.sqrt(np.mean(error**2))) if n else np.nan,
        "district_target_mae_kw": float(np.mean(np.abs(error))) if n else np.nan,
        "district_above_target_kwh": float(np.sum(above)) if n else np.nan,
        "district_peak_above_target_kw": float(np.max(above)) if n else np.nan,
    }


def _kpis(load: np.ndarray, target: np.ndarray | None = None) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    row = {
        "district_total_kwh": float(np.sum(load)),
        "district_mean_kw": float(np.mean(load)),
        "district_peak_kw": float(np.max(load)),
        "district_ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))) if len(load) > 1 else 0.0,
        "district_load_factor": float(np.mean(load) / max(np.max(load), 1e-9)),
    }
    if target is not None:
        row.update(_district_target_kpis(load, np.asarray(target, dtype=float)))
    return row


def _controller_summary(display: str, label: str, district: pd.DataFrame, target: np.ndarray) -> dict[str, float | str]:
    summary = _load_metrics(label)
    row = {
        "controller": display,
        "building_count": int(len(summary)),
        "mean_electricity_kpi": float(pd.to_numeric(summary["electricity_consumption_total"]).mean()),
        "mean_daily_peak_kpi": float(pd.to_numeric(summary["daily_peak_average"]).mean()),
        "mean_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).mean()),
        "max_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).max()),
        "mean_kelvin_hours": float(pd.to_numeric(summary["kelvin_hours"]).mean()),
        "max_kelvin_hours": float(pd.to_numeric(summary["kelvin_hours"]).max()),
    }
    return {**row, **_kpis(district["load"].to_numpy(dtype=float), target)}


def _window_effects(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline_daily = _daily_profile(baseline).set_index("hour_label")["load"]
    rows = []
    for frame in controller_frames:
        name = frame["controller"].iloc[0]
        daily = _daily_profile(frame).set_index("hour_label")["load"]
        for window_name, hour_labels in WINDOWS.items():
            delta = daily.loc[hour_labels].mean() - baseline_daily.loc[hour_labels].mean()
            rows.append(
                {
                    "controller": name,
                    "window": window_name,
                    "mean_delta_vs_baseline_kw": float(delta),
                }
            )
    return pd.DataFrame(rows)


def _scheduled_setpoint(building_id: str, n: int) -> np.ndarray:
    data = pd.read_csv(_baseline_rollout_path(building_id))
    column = "baseline_setpoint" if "baseline_setpoint" in data.columns else "tmin"
    values = pd.to_numeric(data[column], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    if len(values) < n:
        values = np.pad(values, (0, n - len(values)), mode="edge")
    return values[:n]


def _action_window_effects(rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        data = rollouts[building_id].copy()
        action = pd.to_numeric(data["action"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        baseline = pd.to_numeric(pd.read_csv(_baseline_rollout_path(building_id))["action"], errors="coerce").fillna(0.0)
        baseline = baseline.to_numpy(dtype=float)[: len(action)]
        hour_label = data["time_step"].to_numpy(dtype=int) % 24 + 1
        for window_name, hour_labels in WINDOWS.items():
            mask = np.isin(hour_label, hour_labels)
            rows.append(
                {
                    "building_id": building_id,
                    "window": window_name,
                    "mean_delta_u_vs_baseline_mpc": float(np.mean(action[mask] - baseline[mask])),
                }
            )
    return pd.DataFrame(rows)


def _term_diagnostics(rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        data = rollouts[building_id]
        n = len(data)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _scheduled_setpoint(building_id, n)
        action = pd.to_numeric(data["action"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        prev_action = np.r_[0.0, action[:-1]] if n else np.asarray([], dtype=float)
        if "district_target_weight" in data:
            district_weight = pd.to_numeric(data["district_target_weight"], errors="coerce").ffill().bfill()
            district_weight = district_weight.to_numpy(dtype=float)[:n]
        else:
            hour_label = data["time_step"].to_numpy(dtype=int) % 24 + 1
            district_weight = np.full(n, 0.35, dtype=float)
            district_weight[np.isin(hour_label, [6, 7, 8, 9, 10])] = 0.55
            district_weight[np.isin(hour_label, [20, 21, 22, 23])] = 0.75
        profiles = load_ce1_district_target_profiles(
            building_name=str(building["name"]),
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode="target_max",
        )
        target = np.asarray(profiles.target[:n], dtype=float)
        background = np.asarray(profiles.background_without_controlled_heating[:n], dtype=float)
        action_power = np.asarray(profiles.heating_action_power[:n], dtype=float)
        district_load_est = background + action_power * action[:n]
        district_error = (district_load_est - target) / max(float(profiles.target_scale), 1e-6)
        terms = {
            "comfort_setpoint_0p35": 0.35 * ((indoor[:n] - setpoint[:n]) ** 2 / 4.0),
            "district_target_time_weighted": district_weight[:n] * district_error**2,
            "delta_u_15": 15.0 * (action[:n] - prev_action[:n]) ** 2,
            "u_0p1": 0.1 * action[:n],
        }
        for term, values in terms.items():
            rows.append(
                {
                    "controller": "MPC time-weighted target c035/d035-055-075/du15/u01",
                    "building_id": building_id,
                    "term": term,
                    "mean_contribution": float(np.mean(values)) if len(values) else np.nan,
                    "sum_contribution": float(np.sum(values)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _make_comfort_grid(rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=False)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy(dtype=float)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _scheduled_setpoint(building["result_id"], len(data))
        ax.fill_between(time, 20.0, 24.0, color="0.86", alpha=0.75, label="20-24 band")
        ax.plot(time, indoor, lw=0.9, linestyle="-", label="Indoor temperature")
        ax.plot(time, setpoint, lw=0.8, linestyle="-", label="Scheduled setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("MPC no-occupant comfort, time-weighted district-target objective")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "mpc_time_weighted_district_target_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True, sort=False)

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, linestyle="-", label=name)
    ax.plot(baseline["time_step"], baseline["target"], color="0.15", lw=1.4, linestyle="-", label="District target")
    ax.set_title(f"No-occupant district load, time-weighted district-target objective ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily_mean = _daily_profile(combined)
    target_daily = _target_daily_profile(baseline)
    profile = pd.DataFrame(
        {
            "hour_label": np.arange(1, 25),
            "shape_signal": daily_window_shift_profile(simulation_start=0, simulation_end=23),
        }
    )
    preheat = profile[profile["shape_signal"] < 0.0]["hour_label"].to_numpy()
    curtail = profile[profile["shape_signal"] > 0.0]["hour_label"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    for hour in preheat:
        ax.axvspan(hour - 0.45, hour + 0.45, color="#d7ebff", alpha=0.35, linewidth=0)
    for hour in curtail:
        ax.axvspan(hour - 0.45, hour + 0.45, color="#ffe0d2", alpha=0.35, linewidth=0)
    for name, group in daily_mean.groupby("controller", sort=False):
        ax.plot(group["hour_label"], group["load"], marker="o", lw=1.7, linestyle="-", label=name)
    ax.plot(target_daily["hour_label"], target_daily["target"], marker="o", lw=1.7, linestyle="-", color="0.15", label="District target")
    ax.set_title(f"Mean daily district profile: time-weighted district-target objective ({LABEL})")
    ax.set_xlabel("Hour label")
    ax.set_ylabel("Mean district electricity [kW]")
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile.png"
    fig.savefig(daily_path, dpi=160)
    plt.close(fig)
    return load_path, daily_path


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
    return out.to_markdown(index=False)


def _append_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    nb.cells = [
        cell
        for cell in nb.cells
        if LABEL not in str(cell.get("source", ""))
        and "CE1 Time-Weighted District-Target MPC No-Occupant Run" not in str(cell.get("source", ""))
    ]
    nb.cells.extend(cells)
    nbf.write(nb, path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rollouts_by_label = {label: _load_rollouts(label) for label, _ in COMPARISON_RUNS}
    new_rollouts = rollouts_by_label[LABEL]
    baseline = _baseline_district()
    target = baseline["target"].to_numpy(dtype=float)
    controller_district = [
        _district_from_rollouts(display, rollouts_by_label[label]) for label, display in COMPARISON_RUNS
    ]

    comfort_path = _make_comfort_grid(new_rollouts)
    district_plot, daily_plot = _make_district_plots(baseline, controller_district)

    controller_summary = pd.DataFrame(
        [
            _controller_summary(display, label, frame, target)
            for (label, display), frame in zip(COMPARISON_RUNS, controller_district)
        ]
    )
    baseline_kpis = pd.DataFrame(
        [{"controller": "CityLearn baseline", **_kpis(baseline["load"].to_numpy(dtype=float), target)}]
    )
    district_kpis = pd.concat([baseline_kpis, controller_summary], ignore_index=True, sort=False)
    window_effects = _window_effects(baseline, controller_district)
    action_windows = (
        _action_window_effects(new_rollouts)
        .groupby("window", as_index=False)
        .agg(mean_delta_u_vs_baseline_mpc=("mean_delta_u_vs_baseline_mpc", "mean"))
    )
    term_diagnostics = _term_diagnostics(new_rollouts)
    term_summary = (
        term_diagnostics.groupby(["controller", "term"], as_index=False)
        .agg(mean_of_means=("mean_contribution", "mean"), sum_over_buildings=("sum_contribution", "sum"))
        .sort_values(["controller", "term"])
    )

    controller_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_kpis.csv", index=False)
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_district_kpis.csv", index=False)
    window_effects.to_csv(ANALYSIS_DIR / f"{LABEL}_window_effects.csv", index=False)
    action_windows.to_csv(ANALYSIS_DIR / f"{LABEL}_action_window_effects.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_term_diagnostics.csv", index=False)

    common_md = """
## CE1 Time-Weighted District-Target MPC No-Occupant Run

Appended run label: `__LABEL__`.

This is an MPC-only no-occupant run. It keeps the comfort term as single scheduled heating-setpoint tracking, but the district-target penalty is time weighted: stronger during hour labels `6-10`, strongest during `20-23`, and moderate elsewhere. The point is to attack the evening failure without globally flattening comfort.

Active terms:

```json
__WEIGHTS_JSON__
```

District and controller KPI summary:

__DISTRICT_TABLE__

Mean district-load change versus CityLearn baseline by target window:

__WINDOW_TABLE__

Mean MPC action change versus baseline MPC in the target windows:

__ACTION_WINDOW_TABLE__

Weighted term diagnostics for the new MPC run:

__TERM_TABLE__
"""
    common_md = (
        common_md.replace("__LABEL__", LABEL)
        .replace("__WEIGHTS_JSON__", json.dumps(WEIGHTS, indent=2))
        .replace("__DISTRICT_TABLE__", _markdown_table(district_kpis))
        .replace("__WINDOW_TABLE__", _markdown_table(window_effects))
        .replace("__ACTION_WINDOW_TABLE__", _markdown_table(action_windows))
        .replace("__TERM_TABLE__", _markdown_table(term_summary))
    )

    cells = [
        nbf.v4.new_markdown_cell(common_md),
        nbf.v4.new_markdown_cell(f"![MPC district-target comfort all buildings](../{comfort_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![District load](../{district_plot.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_plot.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "02_mpc_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote KPI CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
