#!/usr/bin/env python3
"""Compare 2R2C+Kalman MPC with and without the GridStress heat term."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIG_DIR = ROOT / "figures" / "mpc_gridstress_ablation"
RAW_MPC = ROOT / "results" / "raw" / "mpc"
SUMMARY_ROOT = ROOT / "results" / "summaries" / "mpc"
SCENARIO = "no_occupant"

PREFIX = "ce1_time_bias_guard_1c_2r2c_kalman_gridstress_ablation"
MARKER = f"## MPC GridStress Ablation - `{PREFIX}`"

GRID_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_mpc_rerun_verify"
NO_GRID_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_no_gridstress"

RUNS = [
    {
        "label": GRID_LABEL,
        "controller": "MPC with GridStress",
        "short": "GridStress",
        "objective": "0.5*D + 0.5*GridStress*P_heat/5",
    },
    {
        "label": NO_GRID_LABEL,
        "controller": "MPC comfort only",
        "short": "Comfort only",
        "objective": "D only (comfort_weight=1, shift_weight=0)",
    },
]

NOTEBOOKS = [
    ANALYSIS / "02_mpc_results.ipynb",
    ANALYSIS / "10_peak_load_profiles.ipynb",
]


def building_ids() -> list[str]:
    return [f"building_{i:02d}" for i in range(1, 26)]


def raw_dir(building_id: str, label: str) -> Path:
    return RAW_MPC / building_id / SCENARIO / label


def read_rollout(building_id: str, label: str) -> pd.DataFrame:
    path = raw_dir(building_id, label) / "rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_reference_building_kpis() -> pd.DataFrame:
    path = ANALYSIS / f"{GRID_LABEL}_building_kpis.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    data.insert(0, "controller", "MPC with GridStress")
    data.insert(1, "label", GRID_LABEL)
    return data


def read_no_grid_building_kpis() -> pd.DataFrame:
    rows = []
    for i, building_id in enumerate(building_ids(), start=1):
        path = SUMMARY_ROOT / building_id / SCENARIO / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        summary = pd.read_csv(path)
        row = summary[summary["case"].eq(NO_GRID_LABEL)]
        if row.empty:
            raise RuntimeError(f"Missing {NO_GRID_LABEL} in {path}")
        row = row.iloc[-1].to_dict()
        rollout = read_rollout(building_id, NO_GRID_LABEL)
        rows.append(
            {
                "controller": "MPC comfort only",
                "label": NO_GRID_LABEL,
                "building": f"Building {i}",
                "building_id": building_id,
                "electricity_consumption_total": row.get("electricity_consumption_total", np.nan),
                "discomfort_%": 100.0 * row.get("discomfort_proportion", np.nan),
                "cold_%": 100.0 * row.get("discomfort_cold_proportion", np.nan),
                "hot_%": 100.0 * row.get("discomfort_hot_proportion", np.nan),
                "kelvin_hours": row.get("kelvin_hours", np.nan),
                "daily_peak_average": row.get("daily_peak_average", np.nan),
                "tin_min": float(pd.to_numeric(rollout["indoor_temperature"], errors="coerce").min()),
                "tin_max": float(pd.to_numeric(rollout["indoor_temperature"], errors="coerce").max()),
            }
        )
    return pd.DataFrame(rows)


def read_building_kpis() -> pd.DataFrame:
    data = pd.concat([read_reference_building_kpis(), read_no_grid_building_kpis()], ignore_index=True)
    path = ANALYSIS / f"{PREFIX}_building_kpis.csv"
    data.to_csv(path, index=False)
    return data


def mean_building_kpis(building_kpis: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "tin_min",
        "tin_max",
    ]
    data = building_kpis.groupby("controller", as_index=False)[cols].mean(numeric_only=True)
    order = {run["controller"]: i for i, run in enumerate(RUNS)}
    data["_order"] = data["controller"].map(order)
    data = data.sort_values("_order").drop(columns="_order")
    data.to_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv", index=False)
    return data


def read_baseline_district() -> pd.DataFrame:
    path = ROOT / "results" / "raw" / "citylearn_baseline" / SCENARIO / "baseline" / "district_rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    return pd.DataFrame(
        {
            "controller": "CityLearn baseline",
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "heat": 0.0,
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0),
        }
    )


def district_frame(run: dict[str, str]) -> pd.DataFrame:
    loads = []
    heats = []
    time = None
    target = None
    for building_id in building_ids():
        data = read_rollout(building_id, run["label"])
        current_time = pd.to_numeric(data["time_step"], errors="coerce").to_numpy(dtype=int)
        load = pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        heat = pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if time is None:
            time = current_time
        n = min(len(time), len(current_time), len(load), len(heat))
        time = time[:n]
        loads.append(load[:n])
        heats.append(heat[:n])
    baseline = read_baseline_district()
    n = min(len(time), *(len(v) for v in loads), *(len(v) for v in heats), len(baseline))
    target = baseline["target"].to_numpy(dtype=float)[:n]
    return pd.DataFrame(
        {
            "controller": run["controller"],
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([v[:n] for v in loads], axis=0),
            "heat": np.sum([v[:n] for v in heats], axis=0),
            "target": target,
        }
    )


def district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline_total = float(frames[0]["load"].sum())
    baseline_peak = float(frames[0]["load"].max())
    baseline_ramping = float(np.abs(np.diff(frames[0]["load"].to_numpy(dtype=float))).sum())
    rows = []
    for frame in frames:
        load = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        hourly = frame.assign(hour=frame["time_step"] % 24).groupby("hour", as_index=False)["load"].mean()
        total = float(load.sum())
        peak = float(load.max())
        ramp = float(np.abs(np.diff(load)).sum())
        daily_idx = int(hourly["load"].idxmax())
        rows.append(
            {
                "controller": frame["controller"].iloc[0],
                "total_kwh": total,
                "total_vs_citylearn": total / baseline_total if baseline_total else np.nan,
                "peak_kwh": peak,
                "peak_vs_citylearn": peak / baseline_peak if baseline_peak else np.nan,
                "daily_profile_peak_kwh": float(hourly.loc[daily_idx, "load"]),
                "daily_profile_peak_hour": int(hourly.loc[daily_idx, "hour"]),
                "ramping_abs_kwh": ramp,
                "ramping_vs_citylearn": ramp / baseline_ramping if baseline_ramping else np.nan,
                "target_rmse_kwh": float(np.sqrt(np.mean((load - target) ** 2))),
                "above_target_kwh": float(np.maximum(load - target, 0.0).sum()),
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv", index=False)
    return data


def reference_gridstress() -> pd.DataFrame:
    ref = read_rollout("building_01", GRID_LABEL)
    return pd.DataFrame(
        {
            "time_step": pd.to_numeric(ref["time_step"], errors="coerce").to_numpy(dtype=int),
            "gridstress": pd.to_numeric(ref["shift_signal"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        }
    )


def role_diagnostics(frames: list[pd.DataFrame]) -> pd.DataFrame:
    stress = reference_gridstress()
    rows = []
    for frame in frames[1:]:
        data = frame.merge(stress, on="time_step", how="left")
        heat = data["heat"].to_numpy(dtype=float)
        load = data["load"].to_numpy(dtype=float)
        grid = data["gridstress"].to_numpy(dtype=float)
        high = grid > 1.0
        low = grid < 1.0
        rows.append(
            {
                "controller": frame["controller"].iloc[0],
                "mean_heat_high_stress_kwh": float(np.nanmean(heat[high])),
                "mean_heat_low_stress_kwh": float(np.nanmean(heat[low])),
                "high_over_low_heat_ratio": float(np.nanmean(heat[high]) / np.nanmean(heat[low])),
                "heat_share_high_stress_%": float(100.0 * heat[high].sum() / heat.sum()),
                "load_mean_high_stress_kwh": float(np.nanmean(load[high])),
                "load_mean_low_stress_kwh": float(np.nanmean(load[low])),
                "corr_gridstress_heat": float(np.corrcoef(grid, heat)[0, 1]),
                "corr_gridstress_load": float(np.corrcoef(grid, load)[0, 1]),
            }
        )
    data = pd.DataFrame(rows)
    data.to_csv(ANALYSIS / f"{PREFIX}_role_diagnostics.csv", index=False)
    return data


def action_difference_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, building_id in enumerate(building_ids(), start=1):
        with_grid = read_rollout(building_id, GRID_LABEL)
        no_grid = read_rollout(building_id, NO_GRID_LABEL)
        n = min(len(with_grid), len(no_grid))
        cols = [
            "action",
            "heating_electricity_consumption",
            "net_electricity_consumption",
            "indoor_temperature",
        ]
        diff = {
            col: (
                pd.to_numeric(no_grid[col].iloc[:n], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(with_grid[col].iloc[:n], errors="coerce").to_numpy(dtype=float)
            )
            for col in cols
        }
        rows.append(
            {
                "building": f"Building {i}",
                "building_id": building_id,
                "action_mae": float(np.nanmean(np.abs(diff["action"]))),
                "action_max_abs": float(np.nanmax(np.abs(diff["action"]))),
                "heat_mae_kwh": float(np.nanmean(np.abs(diff["heating_electricity_consumption"]))),
                "heat_max_abs_kwh": float(np.nanmax(np.abs(diff["heating_electricity_consumption"]))),
                "load_mae_kwh": float(np.nanmean(np.abs(diff["net_electricity_consumption"]))),
                "load_max_abs_kwh": float(np.nanmax(np.abs(diff["net_electricity_consumption"]))),
                "tin_mae_c": float(np.nanmean(np.abs(diff["indoor_temperature"]))),
                "tin_max_abs_c": float(np.nanmax(np.abs(diff["indoor_temperature"]))),
            }
        )
    per_building = pd.DataFrame(rows)
    aggregate = pd.DataFrame(
        [
            {
                "scope": "mean over buildings",
                "action_mae": per_building["action_mae"].mean(),
                "action_max_abs": per_building["action_max_abs"].max(),
                "heat_mae_kwh": per_building["heat_mae_kwh"].mean(),
                "heat_max_abs_kwh": per_building["heat_max_abs_kwh"].max(),
                "load_mae_kwh": per_building["load_mae_kwh"].mean(),
                "load_max_abs_kwh": per_building["load_max_abs_kwh"].max(),
                "tin_mae_c": per_building["tin_mae_c"].mean(),
                "tin_max_abs_c": per_building["tin_max_abs_c"].max(),
            }
        ]
    )
    per_building.to_csv(ANALYSIS / f"{PREFIX}_action_delta_by_building.csv", index=False)
    aggregate.to_csv(ANALYSIS / f"{PREFIX}_action_delta_summary.csv", index=False)
    return aggregate, per_building.sort_values("action_mae", ascending=False)


def weighted_terms() -> pd.DataFrame:
    stress = reference_gridstress()
    rows = []
    for run in RUNS:
        for building_id in building_ids():
            data = read_rollout(building_id, run["label"]).merge(stress, on="time_step", how="left")
            tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            tsp = pd.to_numeric(data["scheduled_setpoint"], errors="coerce").to_numpy(dtype=float)
            bias = pd.to_numeric(data["setpoint_bias"], errors="coerce").to_numpy(dtype=float)
            heat = pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            grid = pd.to_numeric(data["gridstress"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            tref = tsp + bias
            cold_guard = np.maximum(tsp - 2.0 - tin, 0.0)
            hot_guard = np.maximum(tin - tsp - 2.0, 0.0)
            track = 3.0 * ((tin - tref) / 2.0) ** 2
            cold = 60.0 * (cold_guard / 1.5) ** 2
            hot = 15.0 * (hot_guard / 2.0) ** 2
            d = track + cold + hot
            grid_heat = 0.5 * grid * heat / 5.0
            stage_with_grid = 0.5 * d + grid_heat
            stage_actual = stage_with_grid if run["label"] == GRID_LABEL else d
            guard = (cold_guard > 0.0) | (hot_guard > 0.0)
            terms = {
                "D_comfort": d,
                "tracking": track,
                "guard_cold": cold,
                "guard_hot": hot,
                "grid_heat_0p5": grid_heat,
                "stage_if_gridstress": stage_with_grid,
                "stage_actual_objective": stage_actual,
                "guard_violation_%": 100.0 * guard.astype(float),
            }
            for term, values in terms.items():
                rows.append(
                    {
                        "controller": run["controller"],
                        "building_id": building_id,
                        "term": term,
                        "mean": float(np.nanmean(values)),
                        "max": float(np.nanmax(values)),
                    }
                )
    data = pd.DataFrame(rows)
    data.to_csv(ANALYSIS / f"{PREFIX}_term_diagnostics_by_building.csv", index=False)
    summary = data.groupby(["controller", "term"], as_index=False).agg(mean=("mean", "mean"), max=("max", "max"))
    summary.to_csv(ANALYSIS / f"{PREFIX}_term_diagnostics.csv", index=False)
    return summary


def plot_comfort_grid(run: dict[str, str]) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{run['label']}_comfort_all25.png"
    building_kpis = read_building_kpis()
    fig, axes = plt.subplots(5, 5, figsize=(20, 13), sharex=True, sharey=False)
    axes = axes.ravel()
    for i, building_id in enumerate(building_ids(), start=1):
        ax = axes[i - 1]
        data = read_rollout(building_id, run["label"])
        t = np.arange(len(data))
        ax.fill_between(t, data["tmin"], data["tmax"], color="0.86", alpha=0.75, label="+/- comfort band")
        ax.plot(t, data["indoor_temperature"], color="#1f77b4", lw=0.8, label="Indoor temperature")
        ax.plot(t, data["scheduled_setpoint"], color="black", lw=0.7, ls="--", label="True setpoint")
        ax.plot(t, data["tracking_setpoint"], color="#ff7f0e", lw=0.7, label="Biased target")
        row = building_kpis[
            building_kpis["controller"].eq(run["controller"]) & building_kpis["building"].eq(f"Building {i}")
        ].iloc[0]
        ax.set_title(f"B{i}: {row['discomfort_%']:.1f}% discomfort", fontsize=8)
        ax.grid(True, alpha=0.25)
        if i == 1:
            ax.legend(fontsize=7, loc="upper left")
        if (i - 1) % 5 == 0:
            ax.set_ylabel("degC", fontsize=8)
        if i > 20:
            ax.set_xlabel("February step", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{run['controller']} | 2R2C+Kalman MPC | February comfort", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_district(frames: list[pd.DataFrame]) -> tuple[Path, Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(frames, ignore_index=True)
    daily = (
        combined.assign(hour=combined["time_step"] % 24)
        .groupby(["controller", "hour"], as_index=False)[["load", "heat", "target"]]
        .mean(numeric_only=True)
    )
    daily.to_csv(ANALYSIS / f"{PREFIX}_district_mean_hourly_profile.csv", index=False)

    timeseries_path = FIG_DIR / "district_load_timeseries.png"
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for controller, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=controller)
    ax.plot(frames[0]["time_step"], frames[0]["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("District load - February")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(timeseries_path, dpi=150)
    plt.close(fig)

    daily_path = FIG_DIR / "district_mean_hourly_profile.png"
    fig, ax = plt.subplots(figsize=(13, 5))
    for controller, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.4, label=controller)
    target = daily[daily["controller"].eq("CityLearn baseline")]
    ax.plot(target["hour"], target["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Mean hourly district load - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(daily_path, dpi=150)
    plt.close(fig)

    heat_path = FIG_DIR / "district_mean_hourly_heating_vs_gridstress.png"
    stress = reference_gridstress().assign(hour=lambda d: d["time_step"] % 24).groupby("hour", as_index=False)["gridstress"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for controller, group in daily[~daily["controller"].eq("CityLearn baseline")].groupby("controller", sort=False):
        ax.plot(group["hour"], group["heat"], marker="o", lw=1.4, label=f"{controller} heat")
    ax2 = ax.twinx()
    ax2.plot(stress["hour"], stress["gridstress"], color="black", ls="--", marker="x", lw=1.0, label="GridStress")
    ax.set_title("Mean hourly district heating vs GridStress")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean heating electricity [kWh]")
    ax2.set_ylabel("GridStress")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)
    fig.tight_layout()
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)
    return timeseries_path, daily_path, heat_path


def markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]) or pd.api.types.is_integer_dtype(data[col]):
            data[col] = data[col].map(lambda x: f"{float(x):.{digits}f}" if pd.notna(x) else "")
    return data.to_markdown(index=False)


def image_md(path: Path, title: str) -> str:
    rel = Path("..") / path.relative_to(ROOT)
    return f"![{title}]({rel.as_posix()})"


def append_notebooks(cells: list[nbf.NotebookNode]) -> None:
    for notebook in NOTEBOOKS:
        if not notebook.exists():
            continue
        nb = nbf.read(notebook, as_version=4)
        nb.cells = [
            cell
            for cell in nb.cells
            if not (cell.cell_type == "markdown" and isinstance(cell.source, str) and MARKER in cell.source)
        ]
        nb.cells.extend(cells)
        nbf.write(nb, notebook)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    building_kpis = read_building_kpis()
    mean_kpis = mean_building_kpis(building_kpis)
    frames = [read_baseline_district(), *[district_frame(run) for run in RUNS]]
    dist_kpis = district_kpis(frames)
    role = role_diagnostics(frames)
    action_delta, action_delta_buildings = action_difference_summary()
    terms = weighted_terms()

    comfort_paths = [plot_comfort_grid(run) for run in RUNS]
    timeseries_path, daily_path, heat_path = plot_district(frames)

    mean_display = mean_kpis.rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
            "daily_peak_average": "daily_peak",
        }
    )
    dist_display = dist_kpis.rename(
        columns={
            "total_vs_citylearn": "total_vs_CL",
            "peak_vs_citylearn": "peak_vs_CL",
            "daily_profile_peak_kwh": "daily_peak_kWh",
            "daily_profile_peak_hour": "daily_peak_hr",
            "ramping_vs_citylearn": "ramp_vs_CL",
            "target_rmse_kwh": "target_RMSE",
        }
    )
    worst = building_kpis.sort_values("discomfort_%", ascending=False)[
        ["controller", "building", "discomfort_%", "cold_%", "hot_%", "electricity_consumption_total", "tin_min", "tin_max"]
    ].head(10)
    term_pivot = terms.pivot(index="controller", columns="term", values="mean").reset_index()
    term_pivot.columns.name = None

    source = f"""{MARKER}

Sanity check for whether the GridStress heat term is doing anything.

Both controllers use the same February rollout, January-only `2R2C+Kalman` RC fits, true setpoint guard, and the same time-biased comfort target:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

Reference MPC:

`J_k = 0.5*D_k + 0.5*GridStress_k*P_heat,k/5`

Ablation MPC:

`J_k = D_k`

The ablation was run as `{NO_GRID_LABEL}` with `comfort_weight=1.0`, `shift_weight=0.0`, `energy_weight=0.0`, `raw_power_cost=True`, `power_source=heating`, and the same 2R2C+Kalman settings as `{GRID_LABEL}`.

Mean building KPIs:

{markdown_table(mean_display[["controller", "energy", "disc_%", "cold_%", "hot_%", "KH", "daily_peak", "tin_min", "tin_max"]], digits=4)}

District aggregate KPIs:

{markdown_table(dist_display[["controller", "total_kwh", "total_vs_CL", "peak_kwh", "peak_vs_CL", "daily_peak_kWh", "daily_peak_hr", "ramping_abs_kwh", "ramp_vs_CL", "target_RMSE", "above_target_kwh"]], digits=4)}

GridStress role diagnostics:

{markdown_table(role, digits=4)}

Action/load/temperature deltas, comfort-only minus GridStress MPC:

{markdown_table(action_delta, digits=6)}

Largest per-building action differences:

{markdown_table(action_delta_buildings.head(10), digits=6)}

Common diagnostic terms, mean over buildings:

{markdown_table(term_pivot, digits=4)}

Worst comfort rows:

{markdown_table(worst, digits=4)}

Saved tables:

- `analysis/{PREFIX}_building_kpis.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_role_diagnostics.csv`
- `analysis/{PREFIX}_action_delta_summary.csv`
- `analysis/{PREFIX}_action_delta_by_building.csv`
- `analysis/{PREFIX}_term_diagnostics.csv`
- `analysis/{PREFIX}_district_mean_hourly_profile.csv`

Takeaway:

The GridStress heat term has almost no behavioral effect in this exact MPC setup. Removing it leaves mean building comfort unchanged, total district energy almost unchanged, peak and daily-profile peak almost unchanged, and the action deltas are tiny. In this run, the visible load-shaping is coming mostly from the time-biased setpoint schedule plus the guard/comfort constraints, not from the `GridStress*P_heat` term.
"""
    image_source = "\n\n".join(
        [
            image_md(daily_path, "MPC GridStress ablation mean hourly district load"),
            image_md(heat_path, "MPC GridStress ablation hourly heating vs GridStress"),
            image_md(timeseries_path, "MPC GridStress ablation district load timeseries"),
            image_md(comfort_paths[0], "MPC with GridStress plot_comfort all buildings"),
            image_md(comfort_paths[1], "MPC comfort-only plot_comfort all buildings"),
        ]
    )
    cells = [nbf.v4.new_markdown_cell(source), nbf.v4.new_markdown_cell(image_source)]
    append_notebooks(cells)

    print("Mean building KPIs:")
    print(mean_display.to_string(index=False))
    print("\nDistrict KPIs:")
    print(dist_display.to_string(index=False))
    print("\nGridStress role diagnostics:")
    print(role.to_string(index=False))
    print("\nAction/load/temperature deltas:")
    print(action_delta.to_string(index=False))
    print(f"\nSaved figures in {FIG_DIR}")
    print("Appended notebooks:")
    for notebook in NOTEBOOKS:
        print(f"- {notebook}")


if __name__ == "__main__":
    main()
