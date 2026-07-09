#!/usr/bin/env python
"""Append CE1 asym hot-heavy + shift/preheat v3 (2R2C) MPC results to analysis/02_mpc_results.ipynb.

Compares v3_2r2c (two-state thermal model) against v3 (1R1C).
Same objective weights — only the RC model changes.
"""

from __future__ import annotations

import csv
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


LABEL = "ce1_asym_hot_heavy_shift_v3_2r2c"
LABEL_PREV = "ce1_asym_hot_heavy_shift_v3"
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
    "comfort_hot_weight": 1.5,
    "comfort_hot_tolerance": 0.5,
    "comfort_hot_scale": 1.5,
    "district_target_weight": 1.5,
    "district_target_mode": "peak_only",
    "district_target_scale_mode": "target_max",
    "shift_weight": 2.5,
    "preheat_weight": 2.5,
    "preheat_horizon": 6,
    "ramp_weight": 25.0,
    "energy_weight": 0.0,
}


def _rollout_path(label: str, building_id: str) -> Path:
    return Path("results") / "raw" / "mpc" / building_id / "no_occupant" / label / "rollout.csv"


def _load_rollouts(label: str) -> dict[str, pd.DataFrame]:
    rollouts = {}
    for building in BUILDINGS:
        path = _rollout_path(label, building["result_id"])
        if not path.exists():
            raise FileNotFoundError(path)
        rollouts[building["result_id"]] = pd.read_csv(path)
    return rollouts


def _setpoint_series(data: pd.DataFrame, cold_tol: float = 0.5) -> np.ndarray:
    if "baseline_setpoint" in data.columns:
        return data["baseline_setpoint"].to_numpy(dtype=float)
    if "effective_setpoint" in data.columns:
        return data["effective_setpoint"].to_numpy(dtype=float)
    if "tmin" in data.columns:
        return data["tmin"].to_numpy(dtype=float) + cold_tol
    raise ValueError("Rollout has no setpoint-like column.")


def _district_series(label: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
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
    controlled = baseline_district - baseline_heating_sum + controlled_heating_sum
    return pd.DataFrame({
        "time_step": np.arange(len(controlled), dtype=int),
        "load": controlled,
        "target": target,
        "baseline": baseline_district,
    })


def _district_kpis(label: str, load: np.ndarray, target: np.ndarray) -> dict:
    load = np.asarray(load, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(load), len(target))
    load, target = load[:n], target[:n]
    above = np.maximum(load - target, 0.0)
    return {
        "label": label,
        "total_kwh": float(np.sum(load)),
        "peak_kw": float(np.max(load)),
        "mean_kw": float(np.mean(load)),
        "target_rmse_kw": float(np.sqrt(np.mean((load - target) ** 2))),
        "above_target_kwh": float(np.sum(above)),
        "above_target_peak_kw": float(np.max(above)),
        "ramping_kwh": float(np.sum(np.abs(np.diff(load)))),
        "load_factor": float(np.mean(load) / max(np.max(load), 1e-9)),
    }


def _controller_kpis_from_rollouts(label: str) -> dict:
    """Compute mean per-building KPIs by reading each building's kpis.csv."""
    kpi_keys = ["electricity_consumption_total", "daily_peak_average",
                "discomfort_proportion", "discomfort_cold_proportion", "discomfort_hot_proportion"]
    accum = {k: [] for k in kpi_keys}
    for building in BUILDINGS:
        kpi_path = (Path("results") / "raw" / "mpc" / building["result_id"]
                    / "no_occupant" / label / "district_kpis.csv")
        if not kpi_path.exists():
            continue
        with open(kpi_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["cost_function"] in kpi_keys:
                    try:
                        accum[row["cost_function"]].append(float(row["value"]))
                    except (ValueError, TypeError):
                        pass
    return {
        "label": label,
        "mean_electricity_kpi": float(np.mean(accum["electricity_consumption_total"])) if accum["electricity_consumption_total"] else float("nan"),
        "mean_daily_peak_kpi": float(np.mean(accum["daily_peak_average"])) if accum["daily_peak_average"] else float("nan"),
        "mean_discomfort": float(np.mean(accum["discomfort_proportion"])) if accum["discomfort_proportion"] else float("nan"),
        "mean_cold_discomfort": float(np.mean(accum["discomfort_cold_proportion"])) if accum["discomfort_cold_proportion"] else float("nan"),
        "mean_hot_discomfort": float(np.mean(accum["discomfort_hot_proportion"])) if accum["discomfort_hot_proportion"] else float("nan"),
    }


def _term_diagnostics(label: str, weights: dict, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        data = rollouts[building_id]
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode=weights["district_target_scale_mode"],
        )
        n = len(data)
        setpoint = _setpoint_series(data, cold_tol=weights["comfort_cold_tolerance"])[:n]
        indoor = data["indoor_temperature"].to_numpy(dtype=float)[:n]
        error = indoor - setpoint
        cold = np.maximum(setpoint - weights["comfort_cold_tolerance"] - indoor, 0.0)
        hot = np.maximum(indoor - setpoint - weights["comfort_hot_tolerance"], 0.0)
        track_term = weights["comfort_track_weight"] * (error / weights["comfort_track_scale"]) ** 2
        cold_term = weights["comfort_cold_weight"] * (cold / weights["comfort_cold_scale"]) ** 2
        hot_term = weights["comfort_hot_weight"] * (hot / weights["comfort_hot_scale"]) ** 2

        if "controlled_action" in data.columns:
            p = np.clip(data["controlled_action"].to_numpy(dtype=float)[:n], 0.0, None)
        else:
            p = np.clip(data["action"].to_numpy(dtype=float)[:n], 0.0, None)
        ramp = np.diff(np.r_[p[0], p])

        stress = np.asarray(profiles.district_stress_profile(), dtype=float)[:n]
        future_stress = np.asarray(
            profiles.future_stress_profile(int(weights["preheat_horizon"])), dtype=float
        )[:n]
        background = np.asarray(profiles.background_without_controlled_heating, dtype=float)[:n]
        actual_heat = data["heating_electricity_consumption"].to_numpy(dtype=float)[:n]
        target_arr = np.asarray(profiles.target, dtype=float)[:n]
        district_error = (background + actual_heat - target_arr) / max(profiles.target_scale, 1e-6)
        if weights.get("district_target_mode") == "peak_only":
            district_error = np.maximum(district_error, 0.0)
        district_term = weights["district_target_weight"] * district_error ** 2

        terms = {
            "asym_track": track_term,
            "asym_cold": cold_term,
            "asym_hot": hot_term,
            "asym_comfort_total": track_term + cold_term + hot_term,
            "district_peak_only": district_term,
            "shift": weights["shift_weight"] * stress * p,
            "preheat_credit": weights["preheat_weight"] * np.maximum(0.0, 1.0 - stress) * future_stress * p,
            "ramp_du25": weights["ramp_weight"] * ramp ** 2,
        }
        for term, values in terms.items():
            rows.append({
                "label": label,
                "building_id": building_id,
                "term": term,
                "mean_contribution": float(np.mean(values)),
                "max_contribution": float(np.max(values)),
            })
    return pd.DataFrame(rows)


def _make_comfort_grid(label: str, rollouts: dict[str, pd.DataFrame], weights: dict) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy()
        setpoint = _setpoint_series(data, cold_tol=weights["comfort_cold_tolerance"])
        lower = setpoint - weights["comfort_cold_tolerance"]
        upper = setpoint + weights["comfort_hot_tolerance"]
        ax.fill_between(time, lower, upper, color="0.85", alpha=0.7, label="soft band")
        ax.plot(time, data["indoor_temperature"], lw=0.8, label="Indoor T")
        ax.plot(time, setpoint, lw=0.8, ls="--", label="Setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels_leg = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=3)
    fig.suptitle(f"MPC (2R2C) — asymmetric hot-heavy + shift/preheat ({label})")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [°C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / f"mpc_comfort_all25_{label}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_daily_profile_plot(series_prev: pd.DataFrame, series_cur: pd.DataFrame) -> Path:
    def hourly_mean(arr):
        arr = np.asarray(arr, dtype=float)
        hours = np.arange(len(arr)) % 24
        return np.array([arr[hours == h].mean() for h in range(24)])

    h = np.arange(24)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h, hourly_mean(series_prev["baseline"]), color="steelblue", marker="o", lw=1.5, label="CityLearn baseline")
    ax.plot(h, hourly_mean(series_prev["load"]), color="green", marker="o", lw=1.5, label=f"MPC {LABEL_PREV} (1R1C)")
    ax.plot(h, hourly_mean(series_cur["load"]), color="darkorange", marker="o", lw=1.5, label=f"MPC {LABEL} (2R2C)")
    ax.plot(h, hourly_mean(series_prev["target"]), color="black", marker="o", lw=1.2, label="CE1 district target")
    ax.set_title("Mean daily district profile — v3 1R1C vs v3 2R2C")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "mpc_mean_daily_profile_v3_vs_v3_2r2c.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_tau_mass_summary_plot() -> Path:
    """Bar plot of tau_mass_hours for each building's 2R2C fit."""
    import json as _json
    taus = []
    names = []
    for building in BUILDINGS:
        short_name = building.get("short_name", building["result_id"])
        model_path = Path(f"results/models/rc/2r2c_{short_name}_january.json")
        if not model_path.exists():
            continue
        params = _json.loads(model_path.read_text())
        a22 = params.get("a22", 0.0)
        tau = -1.0 / np.log(max(a22, 1e-9))
        taus.append(tau)
        names.append(short_name)

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["tab:orange" if t > 6 else "tab:blue" for t in taus]
    ax.bar(range(len(taus)), taus, color=colors)
    ax.axhline(6, color="red", ls="--", lw=1.2, label="τ = 6h threshold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("τ_mass [hours]")
    ax.set_title("2R2C thermal mass time constant per building (orange: τ > 6h)")
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "tau_mass_per_building.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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

    rollouts_cur = _load_rollouts(LABEL)
    rollouts_prev = _load_rollouts(LABEL_PREV)

    comfort_path = _make_comfort_grid(LABEL, rollouts_cur, WEIGHTS)
    tau_path = _make_tau_mass_summary_plot()

    series_prev = _district_series(LABEL_PREV, rollouts_prev)
    series_cur = _district_series(LABEL, rollouts_cur)
    daily_path = _make_daily_profile_plot(series_prev, series_cur)

    # Load existing v3 district KPIs, compute v3_2r2c fresh
    prev_district_kpis = pd.read_csv(ANALYSIS_DIR / f"{LABEL_PREV}_mpc_district_kpis.csv")
    prev_row = prev_district_kpis[prev_district_kpis["label"] == LABEL_PREV].iloc[0].to_dict()
    kpis = pd.DataFrame([
        prev_row,
        _district_kpis(LABEL, series_cur["load"], series_cur["target"]),
    ])
    kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_mpc_district_kpis.csv", index=False)

    # Per-building controller KPIs
    prev_ctrl = pd.read_csv(ANALYSIS_DIR / f"{LABEL_PREV}_mpc_controller_kpis.csv")
    prev_ctrl_row = prev_ctrl[prev_ctrl["label"] == LABEL_PREV].iloc[0].to_dict()
    cur_ctrl_row = _controller_kpis_from_rollouts(LABEL)
    ctrl_kpis = pd.DataFrame([prev_ctrl_row, cur_ctrl_row])
    ctrl_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_mpc_controller_kpis.csv", index=False)

    # Term diagnostics
    terms_prev = _term_diagnostics(LABEL_PREV, WEIGHTS, rollouts_prev)
    terms_cur = _term_diagnostics(LABEL, WEIGHTS, rollouts_cur)
    term_diagnostics = pd.concat([terms_prev, terms_cur], ignore_index=True)
    term_summary = (
        term_diagnostics.groupby(["label", "term"], as_index=False)
        .agg(mean_of_means=("mean_contribution", "mean"), max_over_buildings=("max_contribution", "max"))
        .sort_values(["label", "term"])
    )
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_mpc_term_diagnostics.csv", index=False)

    kpis_display = kpis[["label", "peak_kw", "target_rmse_kw", "above_target_kwh", "above_target_peak_kw", "ramping_kwh", "load_factor"]]

    md = f"""
## CE1 v3 2R2C — Two-state thermal model MPC

Appended run label: `{LABEL}`. Comparison against v3 1R1C (`{LABEL_PREV}`).

**Only change from v3:** RC model upgraded from 1R1C (single state) to **2R2C** (air + thermal mass states).
Same objective weights, same district mode (`peak_only`), same buildings.

**2R2C model summary:**
- Air state: Tin(k+1) = a11·Tin + a12·Tm + b_tout·Tout + b_sol·Sol + b_heat·COP·u + b0 + w_est
- Mass state: Tm(k+1) = a21·Tin + a22·Tm (hidden, propagated open-loop)
- τ_mass = −1/ln(a22) ranges from **1.4h to 200h** across buildings (median ≈ 3.7h)
- Buildings with τ_mass > 6h (potentially useful thermal storage): 112208, 20199, 223581, 319918, 376570, 425540, 4421, 485614, 88386

District KPI comparison (v3 1R1C vs v3 2R2C):

{_markdown_table(kpis_display)}

Per-building KPI comparison:

{_markdown_table(ctrl_kpis)}

Weighted term diagnostics (v3 1R1C vs v3 2R2C):

{_markdown_table(term_summary)}
"""

    cells = [
        nbf.v4.new_markdown_cell(md),
        nbf.v4.new_markdown_cell(f"![τ_mass per building](../{tau_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![MPC comfort all buildings v3_2r2c](../{comfort_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![MPC mean daily district profile v3 vs v3_2r2c](../{daily_path.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "02_mpc_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote CSVs to {ANALYSIS_DIR}")
    print("Appended to: analysis/02_mpc_results.ipynb")
    print("\n=== District KPI Summary ===")
    print(kpis[["label", "peak_kw", "target_rmse_kw", "above_target_kwh"]].to_string(index=False))
    print("\n=== Controller KPI Summary ===")
    print(ctrl_kpis.to_string(index=False))


if __name__ == "__main__":
    main()
