#!/usr/bin/env python
"""Append CE1 asym hot-heavy + shift/preheat v3 MPC-only results to experiments/analysis/02_mpc_results.ipynb.

v3 vs v2 changes:
  shift_weight       0.5 → 2.5  (5×)
  preheat_weight     0.5 → 2.5  (5×)
  district_target_weight 0.35 → 1.5
  district_target_mode   full_quadratic → peak_only
"""

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


LABEL = "ce1_asym_hot_heavy_shift_v3"
LABEL_PREV = "ce1_asym_hot_heavy_shift_v2"
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

WEIGHTS_PREV = {**WEIGHTS, "label": LABEL_PREV, "comfort_hot_weight": 1.5,
                "district_target_weight": 0.35, "district_target_mode": "full_quadratic",
                "shift_weight": 0.5, "preheat_weight": 0.5}


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
            "district_peak_only" if weights.get("district_target_mode") == "peak_only" else "district_full_quadratic": district_term,
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
    fig.suptitle(f"MPC — asymmetric hot-heavy + shift/preheat ({label})")
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
    ax.plot(h, hourly_mean(series_prev["load"]), color="green", marker="o", lw=1.5, label=f"MPC {LABEL_PREV}")
    ax.plot(h, hourly_mean(series_cur["load"]), color="darkorange", marker="o", lw=1.5, label=f"MPC {LABEL}")
    ax.plot(h, hourly_mean(series_prev["target"]), color="black", marker="o", lw=1.2, ls="--", label="CE1 district target")
    ax.set_title(f"Mean daily district profile — MPC v2 vs v3 (shift/preheat 5×, district peak_only)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "mpc_mean_daily_profile.png"
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

    series_prev = _district_series(LABEL_PREV, rollouts_prev)
    series_cur = _district_series(LABEL, rollouts_cur)
    daily_path = _make_daily_profile_plot(series_prev, series_cur)

    kpis = pd.DataFrame([
        {"label": LABEL_PREV, **_district_kpis(series_prev["load"], series_prev["target"])},
        {"label": LABEL, **_district_kpis(series_cur["load"], series_cur["target"])},
    ])
    kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_mpc_district_kpis.csv", index=False)

    summary_csv = Path("results/summaries/mpc/no_occupant_summary_all_buildings.csv")
    summary = pd.read_csv(summary_csv)
    kpi_cols = ["case", "electricity_consumption_total", "daily_peak_average",
                "discomfort_proportion", "discomfort_cold_proportion", "discomfort_hot_proportion"]
    ctrl_kpis = summary[summary["case"].isin([LABEL_PREV, LABEL])][kpi_cols].groupby("case").mean().reset_index()
    ctrl_kpis = ctrl_kpis.rename(columns={
        "case": "label",
        "electricity_consumption_total": "mean_electricity_kpi",
        "daily_peak_average": "mean_daily_peak_kpi",
        "discomfort_proportion": "mean_discomfort",
        "discomfort_cold_proportion": "mean_cold_discomfort",
        "discomfort_hot_proportion": "mean_hot_discomfort",
    })
    ctrl_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_mpc_controller_kpis.csv", index=False)

    terms_prev = _term_diagnostics(LABEL_PREV, WEIGHTS_PREV, rollouts_prev)
    terms_cur = _term_diagnostics(LABEL, WEIGHTS, rollouts_cur)
    term_diagnostics = pd.concat([terms_prev, terms_cur], ignore_index=True)
    term_summary = (
        term_diagnostics.groupby(["label", "term"], as_index=False)
        .agg(mean_of_means=("mean_contribution", "mean"), max_over_buildings=("max_contribution", "max"))
        .sort_values(["label", "term"])
    )
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_mpc_term_diagnostics.csv", index=False)

    md = f"""
## CE1 Asymmetric Hot-Heavy + Shift/Preheat v3 — MPC only

Appended run label: `{LABEL}`. Comparison against v2 (`{LABEL_PREV}`).

**v3 weight changes vs v2:**
- `shift_weight` 0.5 → 2.5 (5×) — signal now exceeds comfort term
- `preheat_weight` 0.5 → 2.5 (5×) — matched increase
- `district_target_weight` 0.35 → 1.5
- `district_target_mode` full\\_quadratic → **peak\\_only** — removes penalty on nighttime preheating load increase

```json
{json.dumps(WEIGHTS, indent=2)}
```

District KPI comparison (MPC only, v2 vs v3):

{_markdown_table(kpis)}

Per-building KPI comparison:

{_markdown_table(ctrl_kpis)}

Weighted term diagnostics (v2 vs v3):

{_markdown_table(term_summary)}
"""

    cells = [
        nbf.v4.new_markdown_cell(md),
        nbf.v4.new_markdown_cell(f"![MPC comfort all buildings v3](../{comfort_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![MPC mean daily district profile v2 vs v3](../{daily_path.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "02_mpc_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote CSVs to {ANALYSIS_DIR}")
    print("Appended to: experiments/analysis/02_mpc_results.ipynb")


if __name__ == "__main__":
    main()
