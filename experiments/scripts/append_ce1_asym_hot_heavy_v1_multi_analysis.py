#!/usr/bin/env python
"""Append CE1 asym hot-heavy v1 results (all 3 controllers) to analysis notebooks."""

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


LABEL = "ce1_asym_hot_heavy_v1"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
BUILDINGS = dataset_building_entries(DATASET_NAME)

WEIGHTS = {
    "label": LABEL,
    "comfort_mode": "asymmetric",
    "comfort_weight": 1.0,
    "comfort_track_weight": 0.4,
    "comfort_track_scale": 2.0,
    "comfort_cold_weight": 0.1,
    "comfort_cold_tolerance": 0.5,
    "comfort_cold_scale": 1.5,
    "comfort_hot_weight": 2.0,
    "comfort_hot_tolerance": 0.5,
    "comfort_hot_scale": 1.5,
    "district_target_weight": 0.3,
    "district_target_mode": "full_quadratic",
    "district_target_scale_mode": "target_max",
    "ramp_weight": 25.0,
    "energy_weight": 0.1,
}

CONTROLLERS = {
    "mpc": ("MPC", "mpc"),
    "rlmpc": ("GNU-RL / RLMPC", "rlmpc"),
    "sac_merlin": ("SAC", "sac_merlin"),
}


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


def _setpoint_series(data: pd.DataFrame) -> np.ndarray:
    if "baseline_setpoint" in data.columns:
        return data["baseline_setpoint"].to_numpy(dtype=float)
    if "effective_setpoint" in data.columns:
        return data["effective_setpoint"].to_numpy(dtype=float)
    if "tmin" in data.columns:
        return data["tmin"].to_numpy(dtype=float) + WEIGHTS["comfort_cold_tolerance"]
    raise ValueError("Rollout has no setpoint-like column.")


def _action_series(data: pd.DataFrame) -> np.ndarray:
    if "controlled_action" in data.columns:
        return np.clip(data["controlled_action"].to_numpy(dtype=float), 0.0, 1.0)
    return np.clip(data["action"].to_numpy(dtype=float), 0.0, 1.0)


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
            target_scale_mode=WEIGHTS["district_target_scale_mode"],
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
    display, _ = CONTROLLERS[controller]
    return pd.DataFrame(
        {
            "time_step": np.arange(len(controlled_district), dtype=int),
            "load": controlled_district,
            "target": target,
            "controller": display,
        }
    )


def _district_kpis(load: np.ndarray, target: np.ndarray) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(load), len(target))
    load, target = load[:n], target[:n]
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
    display, _ = CONTROLLERS[controller]
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy()
        setpoint = _setpoint_series(data)
        lower = setpoint - WEIGHTS["comfort_cold_tolerance"]
        upper = setpoint + WEIGHTS["comfort_hot_tolerance"]
        ax.fill_between(time, lower, upper, color="0.85", alpha=0.7, label="soft comfort band")
        ax.plot(time, data["indoor_temperature"], lw=0.8, label="Indoor T")
        ax.plot(time, setpoint, lw=0.8, ls="--", label="Setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle(f"{display} — asymmetric hot-heavy comfort ({LABEL})")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [°C]")
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
    ax.plot(target["time_step"], target["target"], color="black", lw=1.0, ls="--", label="CE1 district target")
    ax.set_title(f"CE1 district load — MPC, RLMPC, SAC with asymmetric hot-heavy reward ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries_multi.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily = combined.copy()
    daily["hour"] = daily["time_step"] % 24
    daily_mean = daily.groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = target.assign(hour=target["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily_mean.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.5, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.2, ls="--", label="CE1 district target")
    ax.set_title(f"Mean daily district profile — MPC, RLMPC, SAC ({LABEL})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile_multi.png"
    fig.savefig(daily_path, dpi=160)
    plt.close(fig)
    return load_path, daily_path


def _term_diagnostics(controller: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    display, _ = CONTROLLERS[controller]
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        data = rollouts[building_id]
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode=WEIGHTS["district_target_scale_mode"],
        )
        n = len(data)
        setpoint = _setpoint_series(data)[:n]
        indoor = data["indoor_temperature"].to_numpy(dtype=float)[:n]
        error = indoor - setpoint
        cold = np.maximum(setpoint - WEIGHTS["comfort_cold_tolerance"] - indoor, 0.0)
        hot = np.maximum(indoor - setpoint - WEIGHTS["comfort_hot_tolerance"], 0.0)
        track_term = WEIGHTS["comfort_track_weight"] * (error / WEIGHTS["comfort_track_scale"]) ** 2
        cold_term = WEIGHTS["comfort_cold_weight"] * (cold / WEIGHTS["comfort_cold_scale"]) ** 2
        hot_term = WEIGHTS["comfort_hot_weight"] * (hot / WEIGHTS["comfort_hot_scale"]) ** 2

        action = _action_series(data)[:n]
        if controller == "sac_merlin":
            p = data["heating_electricity_consumption"].to_numpy(dtype=float)[:n] / max(
                profiles.heating_nominal_power, 1e-6
            )
        else:
            p = action
        p = np.clip(p, 0.0, None)
        ramp = np.diff(np.r_[p[0], p])

        background = np.asarray(profiles.background_without_controlled_heating, dtype=float)[:n]
        actual_heat = data["heating_electricity_consumption"].to_numpy(dtype=float)[:n]
        target = np.asarray(profiles.target, dtype=float)[:n]
        district_error = (background + actual_heat - target) / max(profiles.target_scale, 1e-6)
        district_term = WEIGHTS["district_target_weight"] * district_error ** 2

        terms = {
            "asym_track": track_term,
            "asym_cold": cold_term,
            "asym_hot": hot_term,
            "asym_comfort_total": track_term + cold_term + hot_term,
            "district_full_quadratic": district_term,
            "ramp_du25": WEIGHTS["ramp_weight"] * ramp ** 2,
            "u_energy": WEIGHTS["energy_weight"] * p ** 2,
        }
        for term, values in terms.items():
            rows.append(
                {
                    "controller": display,
                    "building_id": building_id,
                    "term": term,
                    "mean_contribution": float(np.mean(values)),
                    "max_contribution": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _summary_table(controller: str) -> dict[str, float | str]:
    display, _ = CONTROLLERS[controller]
    summary = pd.read_csv(_summary_path(controller))
    if "case" in summary.columns:
        selected = summary[summary["case"] == LABEL]
        if not selected.empty:
            summary = selected
    return {
        "controller": display,
        "mean_electricity_kpi": summary["electricity_consumption_total"].mean(),
        "mean_daily_peak_kpi": summary["daily_peak_average"].mean(),
        "mean_discomfort": summary["discomfort_proportion"].mean(),
        "max_discomfort": summary["discomfort_proportion"].max(),
        "mean_cold_discomfort": summary["discomfort_cold_proportion"].mean()
        if "discomfort_cold_proportion" in summary.columns else float("nan"),
        "mean_hot_discomfort": summary["discomfort_hot_proportion"].mean()
        if "discomfort_hot_proportion" in summary.columns else float("nan"),
    }


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

    rollouts = {controller: _load_controller_rollouts(controller) for controller in CONTROLLERS}
    comfort_paths = {
        controller: _make_comfort_grid(controller, controller_rollouts)
        for controller, controller_rollouts in rollouts.items()
    }

    baseline = _baseline_district()
    controller_district = [
        _district_series(controller, controller_rollouts)
        for controller, controller_rollouts in rollouts.items()
    ]
    load_path, daily_path = _make_district_plots(baseline, controller_district)

    district_kpis = pd.DataFrame(
        [{"controller": "CityLearn baseline", **_district_kpis(baseline["load"], baseline["target"])}]
        + [
            {"controller": frame["controller"].iloc[0], **_district_kpis(frame["load"], frame["target"])}
            for frame in controller_district
        ]
    )
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_district_kpis.csv", index=False)

    controller_summary = pd.DataFrame([_summary_table(controller) for controller in CONTROLLERS])
    controller_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_controller_kpis.csv", index=False)

    term_diagnostics = pd.concat(
        [_term_diagnostics(controller, r) for controller, r in rollouts.items()],
        ignore_index=True,
    )
    term_diagnostics.to_csv(ANALYSIS_DIR / f"{LABEL}_multi_term_diagnostics.csv", index=False)
    term_summary = (
        term_diagnostics.groupby(["controller", "term"], as_index=False)
        .agg(mean_of_means=("mean_contribution", "mean"), max_over_buildings=("max_contribution", "max"))
        .sort_values(["controller", "term"])
    )

    common_md = f"""
## CE1 Asymmetric Hot-Heavy v1 — MPC, GNU-RL/RLMPC, SAC Comparison

Appended run label: `{LABEL}`.

All three controllers share the same asymmetric comfort cost/reward with **hot violations 20× more expensive than cold**. This lets the optimizer coast through peak hours by allowing mild underheating, while preventing overheating.

Asymmetric comfort cost:

\\[
J^T_k =
w_{{track}}\\left(\\frac{{T_k-T^{{sp}}_k}}{{s_{{track}}}}\\right)^2
+w_{{cold}}\\left(\\frac{{\\max(0,T^{{sp}}_k-\\Delta_{{cold}}-T_k)}}{{s_{{cold}}}}\\right)^2
+w_{{hot}}\\left(\\frac{{\\max(0,T_k-T^{{sp}}_k-\\Delta_{{hot}})}}{{s_{{hot}}}}\\right)^2
\\]

Stage cost (MPC/RLMPC):
\\[
J_k = J^T_k + w_{{u}} u_k^2 + w_{{du}}(u_k-u_{{k-1}})^2 + w_L\\left(\\frac{{L^{{bg}}_k+P^{{ctrl}}_k-L^{{CE1}}_k}}{{S_L}}\\right)^2
\\]

SAC reward: \\(r_k = -J_k\\), using \\(p_k = P^{{heat}}_k / P^{{heat,max}}\\) as the normalized action proxy.

Weights:

```json
{json.dumps(WEIGHTS, indent=2)}
```

District KPI summary:

{_markdown_table(district_kpis)}

Controller KPI summary (per-building normalized KPIs):

{_markdown_table(controller_summary)}

Weighted normalized term diagnostics (mean over buildings):

{_markdown_table(term_summary)}
"""

    combined_cells = [
        nbf.v4.new_markdown_cell(common_md),
        nbf.v4.new_markdown_cell(f"![District load all controllers](../{load_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile all controllers](../{daily_path.as_posix()})"),
    ]
    for controller, path in comfort_paths.items():
        display, _ = CONTROLLERS[controller]
        combined_cells.append(
            nbf.v4.new_markdown_cell(f"![{display} comfort all buildings](../{path.as_posix()})")
        )
    _append_notebook(ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb", combined_cells)

    notebook_map = {
        "mpc": "02_mpc_results.ipynb",
        "rlmpc": "04_rlmpc_results.ipynb",
        "sac_merlin": "08_sac_results.ipynb",
    }
    for controller, notebook in notebook_map.items():
        display, _ = CONTROLLERS[controller]
        cells = [
            nbf.v4.new_markdown_cell(common_md),
            nbf.v4.new_markdown_cell(f"![{display} comfort all buildings](../{comfort_paths[controller].as_posix()})"),
            nbf.v4.new_markdown_cell(f"![District load all controllers](../{load_path.as_posix()})"),
            nbf.v4.new_markdown_cell(f"![Mean daily district profile all controllers](../{daily_path.as_posix()})"),
        ]
        _append_notebook(ANALYSIS_DIR / notebook, cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote CSVs to {ANALYSIS_DIR}")
    print("Appended to notebooks: 02, 04, 08, 11")


if __name__ == "__main__":
    main()
