#!/usr/bin/env python
"""Append CE1 time-biased setpoint guard MPC analysis cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


NOTEBOOK_02 = Path("analysis/02_mpc_results.ipynb")
NOTEBOOK_10 = Path("analysis/10_peak_load_profiles.ipynb")
LABEL = "ce1_time_bias_guard_1c_1r1c"
MARKER_02 = f"## Time-Biased Setpoint Guard MPC - `{LABEL}`"
MARKER_10 = f"## MPC Time-Bias Guard District Profile - `{LABEL}`"


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

CASE = "ce1_time_bias_guard_1c_1r1c"
REFERENCE_CASE = "ce1_setpoint_energy_1r1c_c05_e05"
CONTROLLER = "mpc"
W_COMFORT = 0.5
W_SHIFT = 0.5
TRACK_WEIGHT = 3.0
TRACK_SCALE = 2.0
COLD_GUARD_WEIGHT = 60.0
COLD_GUARD_SCALE = 1.5
HOT_GUARD_WEIGHT = 15.0
HOT_GUARD_SCALE = 2.0
GUARD_BAND = 2.0
POWER_SCALE = 5.0
FIG_DIR = REPO / "figures" / CASE
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = REPO / "analysis"


def _rollout_path(label, building):
    return RESULTS / "raw" / CONTROLLER / building["result_id"] / "no_occupant" / label / "rollout.csv"


def _metric_path(label, building):
    return RESULTS / "summaries" / CONTROLLER / building["result_id"] / "no_occupant" / f"{label}_metrics.json"


def _load_rollouts(label):
    rollouts, missing = {}, []
    for building in BUILDINGS:
        path = _rollout_path(label, building)
        if path.exists():
            rollouts[building["result_id"]] = pd.read_csv(path)
        else:
            missing.append(path)
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


def _series(h, column, fallback=None):
    if column in h.columns:
        return pd.to_numeric(h[column], errors="coerce").to_numpy(float)
    if fallback is not None:
        return np.asarray(fallback, dtype=float)
    return np.full(len(h), np.nan)


rollouts, missing_paths = _load_rollouts(CASE)
if missing_paths:
    print(f"Missing {CASE} rollouts: {len(missing_paths)}")
    for path in missing_paths[:5]:
        print("  ", path)

# --- 5x5 comfort/target grid ---
fig, axes = plt.subplots(5, 5, figsize=(24, 17), sharex=True, sharey=True)
axes = axes.ravel()
for ax, building in zip(axes, BUILDINGS):
    h = rollouts.get(building["result_id"])
    if h is None:
        ax.set_title(f"{building['display']} MISSING", color="red", fontsize=8)
        ax.axis("off")
        continue
    x = h["time_step"].to_numpy() if "time_step" in h else np.arange(len(h))
    tin = _series(h, "indoor_temperature")
    tsp = _series(h, "scheduled_setpoint", fallback=_series(h, "tmin"))
    tref = _series(h, "tracking_setpoint", fallback=tsp)
    n = min(len(x), len(tin), len(tsp), len(tref))
    ax.plot(x[:n], tin[:n], lw=0.75, color="steelblue", label="T_in")
    ax.plot(x[:n], tsp[:n], lw=0.65, ls="--", color="black", alpha=0.75, label="Tsp")
    ax.plot(x[:n], tref[:n], lw=0.65, ls=":", color="tab:orange", label="Tref=Tsp+bias")
    ax.fill_between(x[:n], tsp[:n] - GUARD_BAND, tsp[:n] + GUARD_BAND, alpha=0.10, color="green", label="Tsp +/- 2 C guard")
    ax.set_title(building["display"], fontsize=8)
    ax.set_ylim(15, 25)
    ax.grid(alpha=0.2)
for ax in axes[len(BUILDINGS):]:
    ax.axis("off")
axes[0].legend(fontsize=7)
fig.suptitle("Time-biased setpoint guard MPC (1R1C, +/-1 C bias, Tsp +/-2 C guard)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG_DIR / "mpc_time_bias_guard_comfort_all25.png", dpi=120, bbox_inches="tight")
plt.show()

# --- KPI comparison ---
kpi_labels = ["baseline", REFERENCE_CASE, "simple_v4", CASE]
kpi_df = _load_kpis(kpi_labels)
kpi_df.to_csv(ANALYSIS_DIR / f"{CASE}_mpc_controller_kpis.csv", index=False)
mean_cmp = kpi_df.groupby("label")[
    ["electricity_kpi", "discomfort_%", "cold_%", "hot_%", "kelvin_hours", "daily_peak_kpi", "ramping_kpi"]
].mean().round(4)
print("Mean KPIs across buildings:")
display(mean_cmp)

# --- Weighted terms and bias response ---
rows, term_ts, hourly_rows = [], [], []
for building in BUILDINGS:
    h = rollouts.get(building["result_id"])
    if h is None:
        continue
    tin = _series(h, "indoor_temperature")
    tsp = _series(h, "scheduled_setpoint", fallback=_series(h, "tmin"))
    tref = _series(h, "tracking_setpoint", fallback=tsp)
    bias = _series(h, "setpoint_bias", fallback=np.zeros(len(h)))
    stress = _series(h, "shift_signal", fallback=np.ones(len(h)))
    heat = np.maximum(_series(h, "heating_electricity_consumption", fallback=_series(h, "action")), 0.0)
    x = h["time_step"].to_numpy() if "time_step" in h else np.arange(len(h))
    n = min(len(tin), len(tsp), len(tref), len(bias), len(stress), len(heat), len(x))
    tin, tsp, tref, bias, stress, heat, x = tin[:n], tsp[:n], tref[:n], bias[:n], stress[:n], heat[:n], x[:n]

    track = W_COMFORT * TRACK_WEIGHT * ((tin - tref) / TRACK_SCALE) ** 2
    cold_guard = np.maximum(tsp - GUARD_BAND - tin, 0.0)
    hot_guard = np.maximum(tin - tsp - GUARD_BAND, 0.0)
    cold_term = W_COMFORT * COLD_GUARD_WEIGHT * (cold_guard / COLD_GUARD_SCALE) ** 2
    hot_term = W_COMFORT * HOT_GUARD_WEIGHT * (hot_guard / HOT_GUARD_SCALE) ** 2
    shift_term = W_SHIFT * stress * heat / POWER_SCALE
    total = track + cold_term + hot_term + shift_term
    guard_violation = (cold_guard > 0.0) | (hot_guard > 0.0)

    rows.append({
        "building": building["display"],
        "track_mean": float(np.nanmean(track)),
        "track_max": float(np.nanmax(track)),
        "guard_cold_mean": float(np.nanmean(cold_term)),
        "guard_cold_max": float(np.nanmax(cold_term)),
        "guard_hot_mean": float(np.nanmean(hot_term)),
        "guard_hot_max": float(np.nanmax(hot_term)),
        "time_weighted_heat_mean": float(np.nanmean(shift_term)),
        "time_weighted_heat_max": float(np.nanmax(shift_term)),
        "total_mean": float(np.nanmean(total)),
        "total_max": float(np.nanmax(total)),
        "mae_to_tsp": float(np.nanmean(np.abs(tin - tsp))),
        "mae_to_biased_target": float(np.nanmean(np.abs(tin - tref))),
        "guard_violation_%": float(100.0 * np.nanmean(guard_violation)),
        "mean_temp_minus_tsp_preheat": float(np.nanmean(tin[bias > 0] - tsp[bias > 0])) if np.any(bias > 0) else np.nan,
        "mean_temp_minus_tsp_reduce": float(np.nanmean(tin[bias < 0] - tsp[bias < 0])) if np.any(bias < 0) else np.nan,
    })
    term_ts.append(pd.DataFrame({
        "time_step": x,
        "tracking_target": track,
        "guard_cold": cold_term,
        "guard_hot": hot_term,
        "time_weighted_heat": shift_term,
        "stage_cost": total,
    }))
    hours = (x.astype(int) + 744) % 24
    hourly_rows.append(pd.DataFrame({
        "hour": hours,
        "tin_minus_tsp": tin - tsp,
        "bias": bias,
        "heat": heat,
        "guard_violation": guard_violation.astype(float),
    }))

per_bld_terms = pd.DataFrame(rows).set_index("building")
per_bld_terms.round(6).to_csv(ANALYSIS_DIR / f"{CASE}_mpc_term_diagnostics_by_building.csv")
term_summary = pd.DataFrame([
    {"term": "tracking_target [0.5*3*((Tin-Tref)/2)^2]", "mean_of_means": per_bld_terms["track_mean"].mean(), "max_over_buildings": per_bld_terms["track_mean"].max(), "mean_of_maxes": per_bld_terms["track_max"].mean(), "max_of_maxes": per_bld_terms["track_max"].max()},
    {"term": "guard_cold [0.5*60*violation^2/1.5^2]", "mean_of_means": per_bld_terms["guard_cold_mean"].mean(), "max_over_buildings": per_bld_terms["guard_cold_mean"].max(), "mean_of_maxes": per_bld_terms["guard_cold_max"].mean(), "max_of_maxes": per_bld_terms["guard_cold_max"].max()},
    {"term": "guard_hot [0.5*15*violation^2/2^2]", "mean_of_means": per_bld_terms["guard_hot_mean"].mean(), "max_over_buildings": per_bld_terms["guard_hot_mean"].max(), "mean_of_maxes": per_bld_terms["guard_hot_max"].mean(), "max_of_maxes": per_bld_terms["guard_hot_max"].max()},
    {"term": "time_weighted_heat [0.5*GridStress*P_heat/5]", "mean_of_means": per_bld_terms["time_weighted_heat_mean"].mean(), "max_over_buildings": per_bld_terms["time_weighted_heat_mean"].max(), "mean_of_maxes": per_bld_terms["time_weighted_heat_max"].mean(), "max_of_maxes": per_bld_terms["time_weighted_heat_max"].max()},
    {"term": "stage_cost [total]", "mean_of_means": per_bld_terms["total_mean"].mean(), "max_over_buildings": per_bld_terms["total_mean"].max(), "mean_of_maxes": per_bld_terms["total_max"].mean(), "max_of_maxes": per_bld_terms["total_max"].max()},
]).set_index("term")
term_summary.round(6).to_csv(ANALYSIS_DIR / f"{CASE}_mpc_term_diagnostics.csv")
print("Per-building weighted stage-cost terms and guard response:")
display(per_bld_terms.round(4))
print("Weighted term diagnostics:")
display(term_summary.round(4))

if term_ts:
    term_mean = pd.concat(term_ts, ignore_index=True).groupby("time_step").mean()
    ax = term_mean[["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]].plot(figsize=(14, 4), lw=1.0)
    ax.set_title("Time-biased guard MPC - weighted stage-cost terms")
    ax.set_xlabel("February time step")
    ax.set_ylabel("Weighted term value")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "weighted_terms_timeseries.png", dpi=120, bbox_inches="tight")
    plt.show()

if hourly_rows:
    hourly = pd.concat(hourly_rows, ignore_index=True).groupby("hour").mean()
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hourly.index, hourly["tin_minus_tsp"], marker="o", label="mean Tin - Tsp", color="steelblue")
    ax1.plot(hourly.index, hourly["bias"], marker="s", label="target bias", color="tab:orange")
    ax1.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Temperature delta [C]")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, 100.0 * hourly["guard_violation"], color="tab:red", alpha=0.65, label="guard violation")
    ax2.set_ylabel("Guard violation [%]")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper right")
    ax1.set_title("Mean hourly bias response")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "hourly_bias_response.png", dpi=120, bbox_inches="tight")
    plt.show()

# --- District load comparison ---
baseline_path = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv"
baseline_district = pd.read_csv(baseline_path)
baseline_load = baseline_district["net_electricity_consumption"].to_numpy(float)
target = baseline_district["district_target"].to_numpy(float) if "district_target" in baseline_district.columns else None

district_df = pd.DataFrame({"time_step": np.arange(len(baseline_load)), "CityLearn baseline": baseline_load})
for display_name, label in [
    ("setpoint+energy 1R1C", REFERENCE_CASE),
    ("simple_v4", "simple_v4"),
    ("time-bias guard 1R1C", CASE),
]:
    dist, missing = aggregate_controller_rollouts(controller=CONTROLLER, case=label, results_root=RESULTS, buildings=BUILDINGS)
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
base_ramp = max(np.abs(np.diff(district_df["CityLearn baseline"].to_numpy(float))).sum(), 1e-12)
for col in [c for c in district_df.columns if c not in {"time_step", "CE1 district target"}]:
    load = district_df[col].to_numpy(float)
    district_rows.append({
        "case": col,
        "total_kwh": float(load.sum()),
        "total_vs_baseline": float(load.sum() / base_total),
        "peak_kw": float(load.max()),
        "peak_vs_baseline": float(load.max() / base_peak),
        "ramping_kwh": float(np.abs(np.diff(load)).sum()),
        "ramping_vs_baseline": float(np.abs(np.diff(load)).sum() / base_ramp),
        "load_factor": float(load.mean() / max(load.max(), 1e-12)),
    })
district_kpis = pd.DataFrame(district_rows).set_index("case")
district_kpis.round(6).to_csv(ANALYSIS_DIR / f"{CASE}_mpc_district_kpis.csv")
print("District KPIs:")
display(district_kpis.round(4))

ax = district_df.plot(x="time_step", figsize=(15, 4), linewidth=0.9)
ax.set_title("District electricity load - time-bias guard vs references")
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


MARKDOWN_02 = f'''{MARKER_02}

Run label: `{LABEL}`.

Structured time-based objective:

`Tref(t) = Tsp + bias(t)`

- `bias = +1 C` in preheat/low-load windows: hours `00-04` and `13-16`.
- `bias = -1 C` in high-load windows: hours `06-09` and `17-21`.
- True comfort guard is around the real setpoint: `Tsp +/- 2 C`.
- Guard penalty is `100000`.
- Stage objective uses biased tracking plus time-weighted heating: `0.5*comfort + 0.5*GridStress*P_heat/5`.
- 1R1C model, no CE1 district-target term.
'''


SNAPSHOT_02 = f'''### Rendered snapshot for `{LABEL}`

Mean KPIs across buildings:

| case | electricity KPI | discomfort % | cold % | hot % | daily peak KPI |
|---|---:|---:|---:|---:|---:|
| baseline | 1.0310 | 3.5029 | 2.3262 | 1.1767 | 1.0365 |
| setpoint+energy 1R1C | 1.0299 | 1.4004 | 1.0655 | 0.3349 | 1.0345 |
| simple_v4 | 0.9864 | 4.1979 | 3.8696 | 0.3283 | 1.0160 |
| time-bias guard 1R1C | 1.0237 | 2.2805 | 1.6344 | 0.6461 | 1.0207 |

District aggregate:

| case | total vs baseline | peak vs baseline | ramping vs baseline |
|---|---:|---:|---:|
| setpoint+energy 1R1C | 1.0326 | 0.9900 | 0.9281 |
| simple_v4 | 0.9778 | 0.9497 | 0.9079 |
| time-bias guard 1R1C | 1.0256 | 0.9273 | 0.9551 |

Bias response:

| diagnostic | value |
|---|---:|
| mean abs error to true `Tsp` | 0.6336 C |
| mean abs error to biased `Tref` | 0.6762 C |
| mean guard violation | 2.3905% |
| mean `Tin - Tsp` during preheat bias | +0.4960 C |
| mean `Tin - Tsp` during reduce bias | -0.4384 C |

Worst guard offenders: building 11 (`15.8%`), building 22 (`9.1%`), building 8 (`7.6%`).
'''


IMAGE_CELLS_02 = [
    f"![Time-bias guard comfort all buildings](../figures/{LABEL}/mpc_time_bias_guard_comfort_all25.png)",
    f"![Time-bias guard weighted terms](../figures/{LABEL}/weighted_terms_timeseries.png)",
    f"![Time-bias guard hourly response](../figures/{LABEL}/hourly_bias_response.png)",
    f"![Time-bias guard district load](../figures/{LABEL}/district_load_timeseries.png)",
    f"![Time-bias guard mean daily district profile](../figures/{LABEL}/district_mean_daily_profile.png)",
]


MARKDOWN_10 = f'''{MARKER_10}

MPC run label: `{LABEL}`.

This appends the district-load and mean-daily profile outputs from the time-biased setpoint guard run, compared against baseline, `simple_v4`, and the strict setpoint+energy run.
'''


IMAGE_CELLS_10 = [
    f"![Time-bias guard district load](../figures/{LABEL}/district_load_timeseries.png)",
    f"![Time-bias guard mean daily district profile](../figures/{LABEL}/district_mean_daily_profile.png)",
]


def _append_once(path: Path, marker: str, cells) -> None:
    nb = nbf.read(path, as_version=4)
    if any(marker in "".join(cell.get("source", "")) for cell in nb.cells):
        print(f"{path} already contains {LABEL}; not appending duplicates.")
        return
    nb.cells.extend(cells)
    nbf.write(nb, path)
    print(f"Appended {LABEL} cells to {path}")


def main() -> None:
    _append_once(
        NOTEBOOK_02,
        MARKER_02,
        [
            nbf.v4.new_markdown_cell(MARKDOWN_02),
            nbf.v4.new_code_cell(NOTEBOOK_CODE),
            nbf.v4.new_markdown_cell(SNAPSHOT_02),
            *[nbf.v4.new_markdown_cell(image) for image in IMAGE_CELLS_02],
        ],
    )
    if NOTEBOOK_10.exists():
        _append_once(
            NOTEBOOK_10,
            MARKER_10,
            [
                nbf.v4.new_markdown_cell(MARKDOWN_10),
                *[nbf.v4.new_markdown_cell(image) for image in IMAGE_CELLS_10],
            ],
        )


if __name__ == "__main__":
    main()
