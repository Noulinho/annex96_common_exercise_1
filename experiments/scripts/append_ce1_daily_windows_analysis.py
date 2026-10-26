#!/usr/bin/env python
"""Append no-occupant daily-window load-shaping results to analysis notebooks."""

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
from src.rewards import daily_window_shift_profile


LABEL = "ce1_daily_windows_w04_04_01_01"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

WEIGHTS = {
    "label": LABEL,
    "scenario": "no_occupant",
    "comfort_mode": "setpoint",
    "single_setpoint": True,
    "comfort_weight": 0.4,
    "comfort_scale": 4.0,
    "shift_weight": 0.4,
    "energy_weight": 0.1,
    "ramp_weight": 0.1,
    "peak_weight": 0.0,
    "preheat_weight": 0.0,
    "district_target_weight": 0.0,
    "load_shape_profile": "daily_windows",
    "preheat_hour_labels": [1, 2, 3, 13, 14, 15, 16, 17],
    "curtail_hour_labels": [6, 7, 8, 9, 10, 20, 21, 22, 23],
}

CONTROLLERS = {
    "mpc": "MPC",
    "rlmpc": "GNU-RL / RLMPC",
    "sac_merlin": "SAC",
}

WINDOWS = {
    "midnight_preheat_1_3": [1, 2, 3],
    "morning_curtail_6_10": [6, 7, 8, 9, 10],
    "afternoon_preheat_13_17": [13, 14, 15, 16, 17],
    "evening_curtail_20_23": [20, 21, 22, 23],
}


def _rollout_path(controller: str, building_id: str) -> Path:
    return RESULTS_DIR / "raw" / controller / building_id / "no_occupant" / LABEL / "rollout.csv"


def _summary_path(controller: str) -> Path:
    return RESULTS_DIR / "summaries" / controller / "no_occupant_summary_all_buildings.csv"


def _load_rollouts(controller: str) -> dict[str, pd.DataFrame]:
    rollouts = {}
    missing = []
    for building in BUILDINGS:
        path = _rollout_path(controller, building["result_id"])
        if not path.exists():
            missing.append(path)
            continue
        rollouts[building["result_id"]] = pd.read_csv(path)
    if missing:
        missing_text = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            missing_text += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing {controller} rollouts for {LABEL}:\n{missing_text}")
    return rollouts


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "baseline_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def _action_like(controller: str, data: pd.DataFrame) -> np.ndarray:
    if "controlled_action" in data.columns:
        values = data["controlled_action"]
    elif "action" in data.columns:
        values = data["action"]
    else:
        values = data["heating_electricity_consumption"]
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(values, 0.0, None)


def _district_from_rollouts(controller: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
            "controller": CONTROLLERS[controller],
        }
    )


def _baseline_district() -> pd.DataFrame:
    path = RESULTS_DIR / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float),
            "controller": "CityLearn baseline",
        }
    )


def _daily_profile(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    daily["hour_label"] = daily["time_step"] % 24 + 1
    return daily.groupby(["controller", "hour_label"], as_index=False)["load"].mean()


def _kpis(load: np.ndarray) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    return {
        "district_total_kwh": float(np.sum(load)),
        "district_mean_kw": float(np.mean(load)),
        "district_peak_kw": float(np.max(load)),
        "district_ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))) if len(load) > 1 else 0.0,
        "district_load_factor": float(np.mean(load) / max(np.max(load), 1e-9)),
    }


def _controller_summary(controller: str, district: pd.DataFrame) -> dict[str, float | str]:
    summary = pd.read_csv(_summary_path(controller))
    if "case" in summary.columns:
        selected = summary[summary["case"] == LABEL]
        if not selected.empty:
            summary = selected
    row = {
        "controller": CONTROLLERS[controller],
        "building_count": int(len(summary)),
        "mean_electricity_kpi": float(summary["electricity_consumption_total"].mean()),
        "mean_daily_peak_kpi": float(summary["daily_peak_average"].mean()),
        "mean_discomfort_proportion": float(summary["discomfort_proportion"].mean()),
        "max_discomfort_proportion": float(summary["discomfort_proportion"].max()),
        "mean_kelvin_hours": float(summary["kelvin_hours"].mean())
        if "kelvin_hours" in summary.columns
        else np.nan,
        "max_kelvin_hours": float(summary["kelvin_hours"].max()) if "kelvin_hours" in summary.columns else np.nan,
    }
    return {**row, **_kpis(district["load"].to_numpy(dtype=float))}


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


def _term_diagnostics(controller: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        n = len(data)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _setpoint(data)[:n]
        p = _action_like(controller, data)[:n]
        if p.size and np.nanmax(p) > 1.5:
            p = p / max(float(np.nanmax(p)), 1e-6)
        profile = np.asarray(daily_window_shift_profile(simulation_start=744, simulation_end=1415), dtype=float)[:n]
        ramp = np.diff(np.r_[p[0], p]) if len(p) else np.asarray([], dtype=float)
        comfort = WEIGHTS["comfort_weight"] * ((indoor[:n] - setpoint) ** 2 / WEIGHTS["comfort_scale"])
        energy = WEIGHTS["energy_weight"] * p
        shift = WEIGHTS["shift_weight"] * profile * p
        ramp_term = WEIGHTS["ramp_weight"] * ramp**2
        terms = {
            "comfort_0p4": comfort,
            "signed_daily_window_0p4": shift,
            "u_0p1": energy,
            "delta_u_0p1": ramp_term,
        }
        for term, values in terms.items():
            values = np.asarray(values, dtype=float)
            rows.append(
                {
                    "controller": CONTROLLERS[controller],
                    "building_id": building["result_id"],
                    "term": term,
                    "mean_contribution": float(np.mean(values)) if len(values) else np.nan,
                    "sum_contribution": float(np.sum(values)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _make_comfort_grid(controller: str, rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=False)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy(dtype=float)
        setpoint = _setpoint(data)
        ax.fill_between(time, 20.0, 24.0, color="0.86", alpha=0.75, label="20-24 comfort band")
        ax.plot(time, data["indoor_temperature"], lw=0.9, linestyle="-", label="Indoor temperature")
        ax.plot(time, setpoint, lw=0.9, linestyle="-", label="Single setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"{CONTROLLERS[controller]} no-occupant comfort, daily-window objective")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / f"{controller}_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True)

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, linestyle="-", label=name)
    ax.set_title(f"No-occupant district load, daily-window objective ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily_mean = _daily_profile(combined)
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
    ax.set_title(f"Mean daily district profile: daily-window objective ({LABEL})")
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
    nb.cells.extend(cells)
    nbf.write(nb, path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rollouts = {controller: _load_rollouts(controller) for controller in CONTROLLERS}
    comfort_paths = {controller: _make_comfort_grid(controller, data) for controller, data in rollouts.items()}

    baseline = _baseline_district()
    controller_district = [_district_from_rollouts(controller, data) for controller, data in rollouts.items()]
    district_plot, daily_plot = _make_district_plots(baseline, controller_district)

    controller_summary = pd.DataFrame(
        [_controller_summary(controller, frame) for controller, frame in zip(CONTROLLERS, controller_district)]
    )
    baseline_kpis = pd.DataFrame([{"controller": "CityLearn baseline", **_kpis(baseline["load"].to_numpy(dtype=float))}])
    district_kpis = pd.concat([baseline_kpis, controller_summary], ignore_index=True, sort=False)
    window_effects = _window_effects(baseline, controller_district)
    term_diagnostics = pd.concat(
        [_term_diagnostics(controller, data) for controller, data in rollouts.items()],
        ignore_index=True,
    )
    term_summary = (
        term_diagnostics.groupby(["controller", "term"], as_index=False)
        .agg(mean_of_means=("mean_contribution", "mean"), sum_over_buildings=("sum_contribution", "sum"))
        .sort_values(["controller", "term"])
    )

    controller_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_kpis.csv", index=False)
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_district_kpis.csv", index=False)
    window_effects.to_csv(ANALYSIS_DIR / f"{LABEL}_window_effects.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_term_diagnostics.csv", index=False)

    common_md = """
## CE1 Daily-Window No-Occupant Run

Appended run label: `__LABEL__`.

This run uses no occupants and no RBC schedule override. The only load-shaping signal is a signed daily profile in the cost/reward:

- negative during hour labels `1-3` and `13-17`, so heating is cheaper there;
- positive during hour labels `6-10` and `20-23`, so heating is more expensive there;
- zero elsewhere.

The four active weights add to one:

```json
__WEIGHTS_JSON__
```

MPC and GNU-RL/RLMPC use normalized `u` for the energy and signed daily-window terms. SAC uses the same reward structure with its saved controlled action in diagnostics; during training/evaluation the reward uses heating power normalized by the configured power scale.

District and controller KPI summary:

__DISTRICT_TABLE__

Mean district-load change versus CityLearn baseline by target window:

__WINDOW_TABLE__

Weighted term diagnostics:

__TERM_TABLE__
"""
    common_md = (
        common_md.replace("__LABEL__", LABEL)
        .replace("__WEIGHTS_JSON__", json.dumps(WEIGHTS, indent=2))
        .replace("__DISTRICT_TABLE__", _markdown_table(district_kpis))
        .replace("__WINDOW_TABLE__", _markdown_table(window_effects))
        .replace("__TERM_TABLE__", _markdown_table(term_summary))
    )

    notebook_map = {
        "mpc": "02_mpc_results.ipynb",
        "rlmpc": "04_rlmpc_results.ipynb",
        "sac_merlin": "08_sac_results.ipynb",
    }
    for controller, notebook in notebook_map.items():
        cells = [
            nbf.v4.new_markdown_cell(common_md),
            nbf.v4.new_markdown_cell(f"![{CONTROLLERS[controller]} comfort all buildings](../{comfort_paths[controller].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![District load](../{district_plot.as_posix()})"),
            nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_plot.as_posix()})"),
        ]
        _append_notebook(ANALYSIS_DIR / notebook, cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote KPI CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
