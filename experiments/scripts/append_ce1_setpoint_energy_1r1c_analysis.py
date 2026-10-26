#!/usr/bin/env python
"""Append corrected setpoint + flat heating-energy MPC analysis cells.

This is the single run requested after the sweep false start:

    label = ce1_setpoint_energy_1r1c_c05_e05
    model = 1R1C
    objective = 0.5 * (Tin - Tsp)^2 / 4 + 0.5 * P_heat / 5

No peak, ramp, shift, preheat, district target, or comfort-band dead zone.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK = Path("experiments/analysis/02_mpc_results.ipynb")
LABEL = "ce1_setpoint_energy_1r1c_c05_e05"
MARKER = f"## Setpoint + Flat Heating-Energy MPC - `{LABEL}`"


NOTEBOOK_CODE = r'''
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display

try:
    REPO
except NameError:
    REPO = Path.cwd().resolve()
    if not (REPO / "results").exists():
        REPO = REPO.parent
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from src.buildings import DATASET_NAME, dataset_building_entries
    RESULTS = REPO / "results"
    BUILDINGS = dataset_building_entries(DATASET_NAME)

from src.district_analysis import aggregate_controller_rollouts

CASE = "ce1_setpoint_energy_1r1c_c05_e05"
CONTROLLER = "mpc"
COMFORT_SCALE = 4.0
POWER_SCALE = 5.0
W_COMFORT = 0.5
W_ENERGY = 0.5
FIG_DIR = REPO / "figures" / CASE
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = REPO / "analysis"


def _rollout_path(label, building):
    return RESULTS / "raw" / CONTROLLER / building["result_id"] / "no_occupant" / label / "rollout.csv"


def _metric_path(label, building):
    return RESULTS / "summaries" / CONTROLLER / building["result_id"] / "no_occupant" / f"{label}_metrics.json"


def _setpoint_series(h):
    if "baseline_setpoint" in h.columns:
        return pd.to_numeric(h["baseline_setpoint"], errors="coerce").to_numpy(float)
    if "effective_setpoint" in h.columns:
        return pd.to_numeric(h["effective_setpoint"], errors="coerce").to_numpy(float)
    if "tmin" in h.columns and "tmax" in h.columns:
        lo = pd.to_numeric(h["tmin"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(h["tmax"], errors="coerce").to_numpy(float)
        if np.nanmean(np.abs(hi - lo)) < 0.25:
            return 0.5 * (lo + hi)
        return lo
    if "tmin" in h.columns:
        return pd.to_numeric(h["tmin"], errors="coerce").to_numpy(float)
    raise ValueError("Rollout has no setpoint-like column.")


def _load_rollouts(label):
    rollouts = {}
    missing = []
    for building in BUILDINGS:
        path = _rollout_path(label, building)
        if not path.exists():
            missing.append(path)
            continue
        rollouts[building["result_id"]] = pd.read_csv(path)
    return rollouts, missing


def _load_kpis(labels):
    rows = []
    for label in labels:
        for building in BUILDINGS:
            path = _metric_path(label, building)
            if not path.exists():
                continue
            metrics = json.loads(path.read_text())
            rows.append({
                "label": label,
                "building": building["display"],
                "electricity_kpi": metrics.get("electricity_consumption_total"),
                "discomfort_%": 100.0 * metrics.get("discomfort_proportion", np.nan),
                "cold_%": 100.0 * metrics.get("discomfort_cold_proportion", np.nan),
                "hot_%": 100.0 * metrics.get("discomfort_hot_proportion", np.nan),
                "kelvin_hours": metrics.get("kelvin_hours"),
                "daily_peak_kpi": metrics.get("daily_peak_average"),
                "ramping_kpi": metrics.get("ramping_average"),
            })
    return pd.DataFrame(rows)


def _term_diagnostics(rollouts):
    rows = []
    term_ts = []
    for building in BUILDINGS:
        h = rollouts.get(building["result_id"])
        if h is None:
            continue
        tin = pd.to_numeric(h["indoor_temperature"], errors="coerce").to_numpy(float)
        tsp = _setpoint_series(h)
        if "heating_electricity_consumption" in h.columns:
            heat = pd.to_numeric(h["heating_electricity_consumption"], errors="coerce").to_numpy(float)
        else:
            heat = pd.to_numeric(h["action"], errors="coerce").to_numpy(float)
        n = min(len(tin), len(tsp), len(heat))
        tin, tsp, heat = tin[:n], tsp[:n], np.maximum(heat[:n], 0.0)
        comfort = W_COMFORT * (tin - tsp) ** 2 / COMFORT_SCALE
        energy = W_ENERGY * heat / POWER_SCALE
        total = comfort + energy
        rows.append({
            "building": building["display"],
            "comfort_mean": float(np.nanmean(comfort)),
            "comfort_max": float(np.nanmax(comfort)),
            "energy_mean": float(np.nanmean(energy)),
            "energy_max": float(np.nanmax(energy)),
            "total_mean": float(np.nanmean(total)),
            "total_max": float(np.nanmax(total)),
            "mean_abs_setpoint_error": float(np.nanmean(np.abs(tin - tsp))),
            "max_abs_setpoint_error": float(np.nanmax(np.abs(tin - tsp))),
            "heating_kwh": float(np.nansum(heat)),
        })
        term_ts.append(pd.DataFrame({
            "time_step": np.arange(n, dtype=int),
            "comfort_track": comfort,
            "heating_energy": energy,
            "stage_cost": total,
        }))
    per_bld = pd.DataFrame(rows).set_index("building")
    summary = pd.DataFrame([
        {
            "term": "comfort_track [0.5*(Tin-Tsp)^2/4]",
            "mean_of_means": per_bld["comfort_mean"].mean(),
            "max_over_buildings": per_bld["comfort_mean"].max(),
            "mean_of_maxes": per_bld["comfort_max"].mean(),
            "max_of_maxes": per_bld["comfort_max"].max(),
        },
        {
            "term": "heating_energy [0.5*P_heat/5]",
            "mean_of_means": per_bld["energy_mean"].mean(),
            "max_over_buildings": per_bld["energy_mean"].max(),
            "mean_of_maxes": per_bld["energy_max"].mean(),
            "max_of_maxes": per_bld["energy_max"].max(),
        },
        {
            "term": "stage_cost [total]",
            "mean_of_means": per_bld["total_mean"].mean(),
            "max_over_buildings": per_bld["total_mean"].max(),
            "mean_of_maxes": per_bld["total_max"].mean(),
            "max_of_maxes": per_bld["total_max"].max(),
        },
    ]).set_index("term")
    if term_ts:
        ts = pd.concat(term_ts, ignore_index=True).groupby("time_step").mean()
    else:
        ts = pd.DataFrame()
    return per_bld, summary, ts


rollouts, missing_paths = _load_rollouts(CASE)
if missing_paths:
    print(f"Missing corrected rollouts: {len(missing_paths)}")
    for path in missing_paths[:5]:
        print("  ", path)

# --- 5x5 setpoint tracking grid: Tin vs actual setpoint only ---
fig, axes = plt.subplots(5, 5, figsize=(24, 17), sharex=True, sharey=True)
axes = axes.ravel()
for ax, building in zip(axes, BUILDINGS):
    h = rollouts.get(building["result_id"])
    if h is None:
        ax.set_title(f"{building['display']} MISSING", color="red", fontsize=8)
        ax.axis("off")
        continue
    x = h["time_step"].to_numpy() if "time_step" in h else np.arange(len(h))
    tin = pd.to_numeric(h["indoor_temperature"], errors="coerce").to_numpy(float)
    tsp = _setpoint_series(h)
    n = min(len(x), len(tin), len(tsp))
    ax.plot(x[:n], tin[:n], lw=0.75, color="steelblue", label="T_in")
    ax.plot(x[:n], tsp[:n], lw=0.75, ls="--", color="black", alpha=0.75, label="Tsp")
    ax.set_title(building["display"], fontsize=8)
    ax.set_ylim(15, 25)
    ax.grid(alpha=0.2)
for ax in axes[len(BUILDINGS):]:
    ax.axis("off")
axes[0].legend(fontsize=7)
fig.suptitle("MPC setpoint + flat heating energy (1R1C, 0.5/0.5) - February indoor temperature", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "mpc_setpoint_tracking_all25.png", dpi=120, bbox_inches="tight")
plt.show()

# --- KPI comparison ---
kpi_labels = ["baseline", "simple_v3", "simple_v4", CASE]
kpi_df = _load_kpis(kpi_labels)
kpi_df.to_csv(ANALYSIS_DIR / f"{CASE}_mpc_controller_kpis.csv", index=False)
mean_cmp = kpi_df.groupby("label")[
    ["electricity_kpi", "discomfort_%", "cold_%", "hot_%", "kelvin_hours", "daily_peak_kpi", "ramping_kpi"]
].mean().round(4)
print("Mean KPIs across buildings:")
display(mean_cmp)

# --- Weighted term behavior ---
per_bld_terms, term_summary, term_ts = _term_diagnostics(rollouts)
per_bld_terms.round(6).to_csv(ANALYSIS_DIR / f"{CASE}_mpc_term_diagnostics_by_building.csv")
term_summary.round(6).to_csv(ANALYSIS_DIR / f"{CASE}_mpc_term_diagnostics.csv")
print("Per-building weighted stage-cost terms:")
display(per_bld_terms.round(4))
print("Weighted term diagnostics:")
display(term_summary.round(4))
if not per_bld_terms.empty:
    total = per_bld_terms["total_mean"].mean()
    comfort_share = per_bld_terms["comfort_mean"].mean() / max(total, 1e-12)
    energy_share = per_bld_terms["energy_mean"].mean() / max(total, 1e-12)
    print(f"Comfort share: {100*comfort_share:.1f}% | Energy share: {100*energy_share:.1f}%")
    print(f"Mean abs setpoint error: {per_bld_terms['mean_abs_setpoint_error'].mean():.3f} C")

if not term_ts.empty:
    ax = term_ts[["comfort_track", "heating_energy", "stage_cost"]].plot(figsize=(14, 4), lw=1.0)
    ax.set_title("Weighted stage-cost terms over February, averaged across buildings")
    ax.set_xlabel("February time step")
    ax.set_ylabel("Weighted term value")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "weighted_terms_timeseries.png", dpi=120, bbox_inches="tight")
    plt.show()

# --- District load comparison ---
baseline_path = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv"
baseline_district = pd.read_csv(baseline_path)
baseline_load = baseline_district["net_electricity_consumption"].to_numpy(float)
target = baseline_district["district_target"].to_numpy(float) if "district_target" in baseline_district.columns else None

district_cases = [
    ("CityLearn baseline", None),
    ("simple_v4", "simple_v4"),
    ("setpoint+energy 1R1C", CASE),
]
district_df = pd.DataFrame({"time_step": np.arange(len(baseline_load)), "CityLearn baseline": baseline_load})
for display_name, label in district_cases[1:]:
    dist, missing = aggregate_controller_rollouts(
        controller=CONTROLLER, case=label, results_root=RESULTS, buildings=BUILDINGS
    )
    if missing:
        print(f"{display_name}: {len(missing)} missing rollouts")
        continue
    n = min(len(district_df), len(dist))
    district_df = district_df.iloc[:n].copy()
    district_df[display_name] = dist["net_electricity_consumption"].to_numpy(float)[:n]
if target is not None:
    district_df["CE1 district target"] = target[:len(district_df)]

district_rows = []
base_total = max(district_df["CityLearn baseline"].sum(), 1e-12)
base_peak = max(district_df["CityLearn baseline"].max(), 1e-12)
for col in [c for c in district_df.columns if c != "time_step"]:
    if col == "CE1 district target":
        continue
    load = district_df[col].to_numpy(float)
    district_rows.append({
        "case": col,
        "total_kwh": float(load.sum()),
        "total_vs_baseline": float(load.sum() / base_total),
        "peak_kw": float(load.max()),
        "peak_vs_baseline": float(load.max() / base_peak),
        "ramping_kwh": float(np.abs(np.diff(load)).sum()),
        "load_factor": float(load.mean() / max(load.max(), 1e-12)),
    })
district_kpis = pd.DataFrame(district_rows).set_index("case")
district_kpis.round(6).to_csv(ANALYSIS_DIR / f"{CASE}_mpc_district_kpis.csv")
print("District KPIs:")
display(district_kpis.round(4))

ax = district_df.plot(x="time_step", figsize=(15, 4), linewidth=0.9)
ax.set_title("District electricity load - baseline vs simple_v4 vs setpoint+energy 1R1C")
ax.set_xlabel("February time step")
ax.set_ylabel("District electricity [kWh]")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "district_load_timeseries.png", dpi=120, bbox_inches="tight")
plt.show()

hours = np.tile(np.arange(24), len(district_df) // 24 + 1)[:len(district_df)]
daily = district_df[[c for c in district_df.columns if c != "time_step"]].assign(_h=hours).groupby("_h").mean()
ax = daily.plot(figsize=(10, 4))
ax.set_title("Mean hourly district load - February")
ax.set_xlabel("Hour of day")
ax.set_ylabel("District electricity [kWh]")
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG_DIR / "district_mean_daily_profile.png", dpi=120, bbox_inches="tight")
plt.show()
'''


MARKDOWN_CELL = f'''{MARKER}

Corrected single run, after the failed sweep start.

Run label: `{LABEL}`.

Objective:

`J_k = 0.5 * (Tin - Tsp)^2 / 4.0 + 0.5 * P_heat / 5.0`

Notes:

- 1R1C MPC.
- Tracks the actual heating setpoint `Tsp`; there is no 20/24 comfort-band dead zone in the cost.
- Comfort and consumption weights sum to one: `0.5 + 0.5`.
- Both terms are normalized to comparable order: `2 C` tracking scale and `5 kWh/timestep` heating scale.
- Peak, ramp, shift, preheat, and district target weights are all zero.
- Energy term uses heating electricity, not a peak-load proxy.
'''


IMAGE_CELLS = [
    f"![Setpoint tracking all buildings](../figures/{LABEL}/mpc_setpoint_tracking_all25.png)",
    f"![Weighted term behavior](../figures/{LABEL}/weighted_terms_timeseries.png)",
    f"![District load comparison](../figures/{LABEL}/district_load_timeseries.png)",
    f"![Mean daily district profile](../figures/{LABEL}/district_mean_daily_profile.png)",
]


def main() -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    sources = ["".join(cell.get("source", "")) for cell in nb.cells]
    if any(MARKER in source for source in sources):
        print(f"{NOTEBOOK} already contains {LABEL} cells; not appending duplicates.")
        return

    nb.cells.append(nbf.v4.new_markdown_cell(MARKDOWN_CELL))
    nb.cells.append(nbf.v4.new_code_cell(NOTEBOOK_CODE))
    for image in IMAGE_CELLS:
        nb.cells.append(nbf.v4.new_markdown_cell(image))
    nbf.write(nb, NOTEBOOK)
    print(f"Appended {LABEL} analysis cells to {NOTEBOOK}")


if __name__ == "__main__":
    main()
