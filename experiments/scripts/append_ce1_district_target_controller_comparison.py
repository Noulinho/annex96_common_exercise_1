#!/usr/bin/env python
"""Append controller comparison for the CE1 district-target objective."""

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


LABEL = "ce1_mpc_district_target_c04_d03_du25_u01"
SECTION_TITLE = "CE1 District-Target Controller Comparison"
FIG_DIR = Path("figures") / LABEL / "controller_comparison"
ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

CONTROLLERS = {
    "mpc": "MPC",
    "rlmpc": "GNU-RL / RLMPC",
    "sac_merlin": "SAC",
}

WINDOWS = {
    "midnight_1_3": [1, 2, 3],
    "morning_peak_6_10": [6, 7, 8, 9, 10],
    "afternoon_valley_13_17": [13, 14, 15, 16, 17],
    "evening_peak_20_23": [20, 21, 22, 23],
}

WEIGHTS = {
    "label": LABEL,
    "scenario": "no_occupant",
    "comfort_mode": "single heating setpoint tracking",
    "comfort_weight": 0.4,
    "district_target_weight": 0.3,
    "delta_u_weight": 25.0,
    "u_weight": 0.1,
    "comfort_scale": 4.0,
    "district_target_scale_mode": "target_max",
    "sac_power_source": "heating",
    "sac_power_scale_mode": "nominal_heating",
}


def _rollout_path(controller: str, building_id: str) -> Path:
    return RESULTS_DIR / "raw" / controller / building_id / "no_occupant" / LABEL / "rollout.csv"


def _metrics_path(controller: str, building_id: str) -> Path:
    return RESULTS_DIR / "summaries" / controller / building_id / "no_occupant" / f"{LABEL}_metrics.json"


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
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "controller": "CityLearn baseline",
        }
    )


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
        lines = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            lines += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing {controller} rollouts for {LABEL}:\n{lines}")
    return rollouts


def _load_metrics(controller: str) -> pd.DataFrame:
    rows = []
    missing = []
    for building in BUILDINGS:
        path = _metrics_path(controller, building["result_id"])
        if not path.exists():
            missing.append(path)
            continue
        row = json.loads(path.read_text())
        row.setdefault("building_id", building["result_id"])
        rows.append(row)
    if missing:
        lines = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            lines += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing {controller} metrics for {LABEL}:\n{lines}")
    return pd.DataFrame(rows)


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


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "baseline_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def _action(data: pd.DataFrame) -> np.ndarray:
    for column in ("controlled_action", "action"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    heating = pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return heating / max(float(np.nanmax(heating)), 1e-6)


def _daily_profile(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    daily["hour_label"] = daily["time_step"].to_numpy(dtype=int) % 24 + 1
    return daily.groupby(["controller", "hour_label"], as_index=False)["load"].mean()


def _target_daily_profile(baseline: pd.DataFrame) -> pd.DataFrame:
    target = baseline[["time_step", "target"]].copy()
    target["hour_label"] = target["time_step"].to_numpy(dtype=int) % 24 + 1
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
        "district_load_factor": float(np.mean(load) / max(float(np.max(load)), 1e-9)),
    }
    if target is not None:
        row.update(_district_target_kpis(load, target))
    return row


def _controller_summary(controller: str, district: pd.DataFrame, target: np.ndarray) -> dict[str, float | str]:
    summary = _load_metrics(controller)
    row = {
        "controller": CONTROLLERS[controller],
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
            rows.append(
                {
                    "controller": name,
                    "window": window_name,
                    "mean_delta_vs_baseline_kw": float(daily.loc[hour_labels].mean() - baseline_daily.loc[hour_labels].mean()),
                }
            )
    return pd.DataFrame(rows)


def _action_window_effects(controller: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        action = _action(data)
        hour_label = data["time_step"].to_numpy(dtype=int) % 24 + 1
        for window_name, hour_labels in WINDOWS.items():
            mask = np.isin(hour_label, hour_labels)
            rows.append(
                {
                    "controller": CONTROLLERS[controller],
                    "building_id": building["result_id"],
                    "window": window_name,
                    "mean_action": float(np.mean(action[mask])),
                }
            )
    return pd.DataFrame(rows)


def _make_sac_comfort_grid(rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=False)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy(dtype=float)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _setpoint(data)
        ax.fill_between(time, 20.0, 24.0, color="0.86", alpha=0.75, label="20-24 band")
        ax.plot(time, indoor, lw=0.9, linestyle="-", label="Indoor temperature")
        ax.plot(time, setpoint, lw=0.8, linestyle="-", label="Scheduled setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("SAC no-occupant comfort, district-target objective")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "sac_district_target_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True, sort=False)

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, linestyle="-", label=name)
    ax.plot(baseline["time_step"], baseline["target"], color="0.15", lw=1.4, linestyle="-", label="District target")
    ax.set_title(f"No-occupant district load, district-target objective ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries_all_controllers.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily_mean = _daily_profile(combined)
    target_daily = _target_daily_profile(baseline)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily_mean.groupby("controller", sort=False):
        ax.plot(group["hour_label"], group["load"], marker="o", lw=1.7, linestyle="-", label=name)
    ax.plot(target_daily["hour_label"], target_daily["target"], marker="o", lw=1.7, linestyle="-", color="0.15", label="District target")
    ax.set_title(f"Mean daily district profile: district-target objective ({LABEL})")
    ax.set_xlabel("Hour label")
    ax.set_ylabel("Mean district electricity [kW]")
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile_all_controllers.png"
    fig.savefig(daily_path, dpi=160)
    plt.close(fig)
    return load_path, daily_path


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
    return out.to_markdown(index=False)


def _insert_after_mpc_section(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    nb.cells = [cell for cell in nb.cells if SECTION_TITLE not in str(cell.get("source", ""))]
    insert_at = len(nb.cells)
    for idx, cell in enumerate(nb.cells):
        source = str(cell.get("source", ""))
        if f"figures/{LABEL}/district_mean_daily_profile.png" in source:
            insert_at = idx + 1
            break
    nb.cells[insert_at:insert_at] = cells
    nbf.write(nb, path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rollouts = {controller: _load_rollouts(controller) for controller in CONTROLLERS}
    baseline = _baseline_district()
    target = baseline["target"].to_numpy(dtype=float)
    controller_district = [
        _district_from_rollouts(controller, rollouts[controller]) for controller in CONTROLLERS
    ]

    sac_comfort_path = _make_sac_comfort_grid(rollouts["sac_merlin"])
    district_path, daily_path = _make_district_plots(baseline, controller_district)

    baseline_kpis = pd.DataFrame(
        [{"controller": "CityLearn baseline", **_kpis(baseline["load"].to_numpy(dtype=float), target)}]
    )
    controller_kpis = pd.DataFrame(
        [
            _controller_summary(controller, frame, target)
            for controller, frame in zip(CONTROLLERS, controller_district)
        ]
    )
    district_kpis = pd.concat([baseline_kpis, controller_kpis], ignore_index=True, sort=False)
    window_effects = _window_effects(baseline, controller_district)
    action_windows = pd.concat(
        [_action_window_effects(controller, rollouts[controller]) for controller in CONTROLLERS],
        ignore_index=True,
    )
    action_summary = (
        action_windows.groupby(["controller", "window"], as_index=False)
        .agg(mean_action=("mean_action", "mean"))
        .sort_values(["controller", "window"])
    )

    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_comparison_district_kpis.csv", index=False)
    window_effects.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_comparison_window_effects.csv", index=False)
    action_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_comparison_action_windows.csv", index=False)

    markdown = f"""
## {SECTION_TITLE}

Same no-occupant district-target label: `{LABEL}`.

MPC, GNU-RL/RLMPC, and SAC are compared under the same active weights:

```json
{json.dumps(WEIGHTS, indent=2)}
```

SAC was trained with this reward present and uses heating power normalized by each building's nominal heating power for the energy/ramp terms.

District and controller KPI summary:

{_markdown_table(district_kpis)}

Mean district-load change versus CityLearn baseline by target window:

{_markdown_table(window_effects)}

Mean controller action by target window:

{_markdown_table(action_summary)}
"""
    cells = [
        nbf.v4.new_markdown_cell(markdown),
        nbf.v4.new_markdown_cell(f"![SAC district-target comfort all buildings](../{sac_comfort_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![District load controller comparison](../{district_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily controller comparison](../{daily_path.as_posix()})"),
    ]
    _insert_after_mpc_section(ANALYSIS_DIR / "02_mpc_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote comparison CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
