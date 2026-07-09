#!/usr/bin/env python
"""Append CE1 setpoint-normalized load-shifting results to analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_target import load_ce1_district_target_profiles


LABEL = "ce1_setpoint_norm_shift20"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
BUILDINGS = dataset_building_entries(DATASET_NAME)


def _rollout_path(controller: str, building_id: str) -> Path:
    return Path("results") / "raw" / controller / building_id / "no_occupant" / LABEL / "rollout.csv"


def _summary_path(controller: str) -> Path:
    return Path("results") / "summaries" / controller / "no_occupant_summary_all_buildings.csv"


def _load_controller_rollouts(controller: str) -> dict[str, pd.DataFrame]:
    rollouts = {}
    for building in BUILDINGS:
        path = _rollout_path(controller, building["result_id"])
        if not path.exists():
            raise FileNotFoundError(path)
        rollouts[building["result_id"]] = pd.read_csv(path)
    return rollouts


def _district_series(controller: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline_district = None
    target = None
    baseline_heating_sum = None
    controlled_heating_sum = None

    for building in BUILDINGS:
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode="baseline_error_std",
        )
        base = np.asarray(profiles.baseline_district_load, dtype=float)
        tgt = np.asarray(profiles.target, dtype=float)
        baseline_heat = np.asarray(profiles.baseline_controlled_heating, dtype=float)
        controlled_heat = rollouts[building["result_id"]]["heating_electricity_consumption"].to_numpy(dtype=float)
        n = min(len(base), len(tgt), len(baseline_heat), len(controlled_heat))
        if baseline_district is None:
            baseline_district = base[:n]
            target = tgt[:n]
            baseline_heating_sum = np.zeros(n, dtype=float)
            controlled_heating_sum = np.zeros(n, dtype=float)
        else:
            n = min(n, len(baseline_district))
            baseline_district = baseline_district[:n]
            target = target[:n]
            baseline_heating_sum = baseline_heating_sum[:n]
            controlled_heating_sum = controlled_heating_sum[:n]

        baseline_heating_sum += baseline_heat[:n]
        controlled_heating_sum += controlled_heat[:n]

    controlled_district = baseline_district - baseline_heating_sum + controlled_heating_sum
    return pd.DataFrame(
        {
            "time_step": np.arange(len(controlled_district), dtype=int),
            "load": controlled_district,
            "target": target,
            "controller": controller.upper() if controller == "mpc" else "GNU-RL / RLMPC",
        }
    )


def _baseline_district() -> pd.DataFrame:
    path = Path("results/raw/citylearn_baseline/no_occupant/baseline/district_rollout.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": data["net_electricity_consumption"].to_numpy(dtype=float),
            "target": data["district_target"].to_numpy(dtype=float),
            "controller": "CityLearn baseline",
        }
    )


def _district_kpis(load: np.ndarray, target: np.ndarray) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(load), len(target))
    load = load[:n]
    target = target[:n]
    above = np.maximum(load - target, 0.0)
    return {
        "total_kwh": float(np.sum(load)),
        "peak_kw": float(np.max(load)),
        "mean_kw": float(np.mean(load)),
        "target_rmse_kw": float(np.sqrt(np.mean((load - target) ** 2))),
        "above_target_kwh": float(np.sum(above)),
        "above_target_peak_kw": float(np.max(above)),
        "ramping_kwh": float(np.sum(np.abs(np.diff(load)))),
        "load_factor": float(np.mean(load) / max(np.max(load), 1e-9)),
    }


def _make_comfort_grid(controller: str, rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy()
        setpoint = data["tmin"].to_numpy(dtype=float)
        ax.plot(time, data["indoor_temperature"], lw=0.8, label="Indoor")
        ax.plot(time, setpoint, lw=0.8, ls="--", label="Setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{controller.upper() if controller == 'mpc' else 'GNU-RL / RLMPC'} comfort: exact setpoint tracking ({LABEL})")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / f"{controller}_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True)
    target = baseline[["time_step", "target"]].copy()

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, label=name)
    ax.plot(target["time_step"], target["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_title(f"CE1 district load: exact setpoint + normalized load shifting ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily = combined.copy()
    daily["hour"] = daily["time_step"] % 24
    daily_mean = daily.groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = target.assign(hour=target["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily_mean.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.5, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.2, ls="--", label="CE1 target")
    ax.set_title(f"CE1 mean daily district profile after control ({LABEL})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile.png"
    fig.savefig(daily_path, dpi=160)
    plt.close(fig)
    return load_path, daily_path


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    return out.to_markdown(index=False)


def _append_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    nb.cells.extend(cells)
    nbf.write(nb, path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mpc_rollouts = _load_controller_rollouts("mpc")
    rlmpc_rollouts = _load_controller_rollouts("rlmpc")

    mpc_comfort = _make_comfort_grid("mpc", mpc_rollouts)
    rlmpc_comfort = _make_comfort_grid("rlmpc", rlmpc_rollouts)

    baseline = _baseline_district()
    mpc_district = _district_series("mpc", mpc_rollouts)
    rlmpc_district = _district_series("rlmpc", rlmpc_rollouts)
    district_plot, daily_plot = _make_district_plots(baseline, [mpc_district, rlmpc_district])

    district_kpis = pd.DataFrame(
        [
            {"controller": "CityLearn baseline", **_district_kpis(baseline["load"], baseline["target"])},
            {"controller": "MPC", **_district_kpis(mpc_district["load"], mpc_district["target"])},
            {"controller": "GNU-RL / RLMPC", **_district_kpis(rlmpc_district["load"], rlmpc_district["target"])},
        ]
    )
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_district_kpis.csv", index=False)

    mpc_summary = pd.read_csv(_summary_path("mpc"))
    rlmpc_summary = pd.read_csv(_summary_path("rlmpc"))
    controller_summary = pd.DataFrame(
        [
            {
                "controller": "MPC",
                "mean_electricity_kpi": mpc_summary["electricity_consumption_total"].mean(),
                "mean_daily_peak_kpi": mpc_summary["daily_peak_average"].mean(),
                "mean_discomfort": mpc_summary["discomfort_proportion"].mean(),
                "max_discomfort": mpc_summary["discomfort_proportion"].max(),
            },
            {
                "controller": "GNU-RL / RLMPC",
                "mean_electricity_kpi": rlmpc_summary["electricity_consumption_total"].mean(),
                "mean_daily_peak_kpi": rlmpc_summary["daily_peak_average"].mean(),
                "mean_discomfort": rlmpc_summary["discomfort_proportion"].mean(),
                "max_discomfort": rlmpc_summary["discomfort_proportion"].max(),
            },
        ]
    )
    controller_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_kpis.csv", index=False)

    weights = {
        "label": LABEL,
        "comfort_mode": "setpoint",
        "temperature_scale_mode": "setpoint_std",
        "district_target_scale_mode": "baseline_error_std",
        "comfort_weight": 30.0,
        "energy_weight": 0.05,
        "peak_weight": 2.0,
        "ramp_weight": 0.1,
        "shift_weight": 20.0,
        "district_target_weight": 10.0,
    }

    common_md = """
## CE1 Exact-Setpoint Normalized Shift Run

Appended run label: `__LABEL__`.

Cost used by MPC and matched GNU-RL/RLMPC:

\\[
J_k =
30\\left(\\frac{T_k - T^{{sp}}_k}{\\sigma_T}\\right)^2
+ 0.05u_k
+ 2u_k^2
+ 0.1(u_k-u_{k-1})^2
+ 20\\lambda_k u_k
+ 10\\left(\\frac{L^{{base-bg}}_k + P^{{heat}}_k - L^{{CE1}}_k}{\\sigma_L}\\right)^2 .
\\]

Normalization:

- `Tsp` is the single dataset heating setpoint; `tmin = tmax = Tsp` in the saved rollouts.
- `sigma_T` is the building-specific January scheduled-setpoint standard deviation (`setpoint_std`), not a hand divisor.
- `u_k` is already min-max normalized by the actuator bounds `[0, 1]`.
- `lambda_k = max(L_baseline - L_CE1, 0) / max(max(L_baseline - L_CE1, 0))`.
- `sigma_L` is the February baseline district-load target-error standard deviation.

Weights:

```json
__WEIGHTS_JSON__
```

District KPI summary:

__DISTRICT_TABLE__

Controller KPI summary:

__CONTROLLER_TABLE__
"""
    common_md = (
        common_md.replace("__LABEL__", LABEL)
        .replace("__WEIGHTS_JSON__", json.dumps(weights, indent=2))
        .replace("__DISTRICT_TABLE__", _markdown_table(district_kpis))
        .replace("__CONTROLLER_TABLE__", _markdown_table(controller_summary))
    )

    combined_cells = [
        nbf.v4.new_markdown_cell(common_md),
        nbf.v4.new_markdown_cell(f"![District load](../{district_plot.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_plot.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![MPC comfort all buildings](../{mpc_comfort.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![GNU-RL RLMPC comfort all buildings](../{rlmpc_comfort.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb", combined_cells)

    mpc_cells = [
        nbf.v4.new_markdown_cell(common_md),
        nbf.v4.new_markdown_cell(f"![MPC comfort all buildings](../{mpc_comfort.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![District load](../{district_plot.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_plot.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "02_mpc_results.ipynb", mpc_cells)

    rlmpc_cells = [
        nbf.v4.new_markdown_cell(common_md + "\n\nNote: the qpth backend emitted repeated accuracy warnings during this aggressive matched-cost run; treat this as a numerical-diagnostic flag when interpreting GNU-RL/RLMPC."),
        nbf.v4.new_markdown_cell(f"![GNU-RL RLMPC comfort all buildings](../{rlmpc_comfort.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![District load](../{district_plot.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_plot.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "04_rlmpc_results.ipynb", rlmpc_cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote KPI CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
