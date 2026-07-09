#!/usr/bin/env python3
"""Append MPC normalized comfort-only D comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.occupants.metrics import kelvin_hours


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIG_DIR = ROOT / "figures" / "mpc_normalizedD_no_gridstress"
RAW_MPC = ROOT / "results" / "raw" / "mpc"
SUMMARY_ROOT = ROOT / "results" / "summaries" / "mpc"
SCENARIO = "no_occupant"

PREFIX = "ce1_final_no_occ_mpc_normD010702_check"
MARKER = f"## Final No-Occupant MPC Comfort-Only Normalized-D Check - `{PREFIX}`"

OLD_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_no_gridstress"
NEW_LABEL = "ce1_final_no_occ_D010702_mpc_2r2c_kalman"

RUNS = [
    {
        "label": OLD_LABEL,
        "controller": "MPC comfort only, old D weights",
        "short": "Old D 3/60/15",
        "track_weight": 3.0,
        "cold_weight": 60.0,
        "hot_weight": 15.0,
        "objective": "D_k with weights 3, 60, 15",
    },
    {
        "label": NEW_LABEL,
        "controller": "MPC comfort only, normalized D weights",
        "short": "New D 0.1/0.7/0.2",
        "track_weight": 0.1,
        "cold_weight": 0.7,
        "hot_weight": 0.2,
        "objective": "D_k with weights 0.1, 0.7, 0.2",
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


def read_summary_row(building_id: str, label: str) -> dict[str, object]:
    path = SUMMARY_ROOT / building_id / SCENARIO / "summary.csv"
    if not path.exists():
        return {}
    summary = pd.read_csv(path)
    row = summary[summary["case"].eq(label)]
    if row.empty:
        return {}
    return row.iloc[-1].to_dict()


def read_kpi_values(building_id: str, label: str) -> dict[str, float]:
    path = raw_dir(building_id, label) / "kpis.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    district = data[data.get("level", "") == "district"].copy()
    if district.empty:
        district = data
    return {
        str(row["cost_function"]): float(row["value"])
        for _, row in district.iterrows()
        if pd.notna(row.get("value"))
    }


def read_building_kpis() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for i, building_id in enumerate(building_ids(), start=1):
            summary = read_summary_row(building_id, run["label"])
            values = read_kpi_values(building_id, run["label"])
            rollout = read_rollout(building_id, run["label"])
            tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
            rows.append(
                {
                    "controller": run["controller"],
                    "label": run["label"],
                    "building": f"Building {i}",
                    "building_id": building_id,
                    "electricity_consumption_total": values.get(
                        "electricity_consumption_total",
                        summary.get("electricity_consumption_total", np.nan),
                    ),
                    "discomfort_%": 100.0
                    * values.get("discomfort_proportion", summary.get("discomfort_proportion", np.nan)),
                    "cold_%": 100.0
                    * values.get(
                        "discomfort_cold_proportion",
                        summary.get("discomfort_cold_proportion", np.nan),
                    ),
                    "hot_%": 100.0
                    * values.get(
                        "discomfort_hot_proportion",
                        summary.get("discomfort_hot_proportion", np.nan),
                    ),
                    "kelvin_hours": summary.get(
                        "kelvin_hours",
                        kelvin_hours(tin.to_numpy(dtype=float), lower=20.0, upper=24.0),
                    ),
                    "daily_peak_average": values.get("daily_peak_average", summary.get("daily_peak_average", np.nan)),
                    "ramping_average": values.get("ramping_average", summary.get("ramping_average", np.nan)),
                    "tin_min": float(tin.min()),
                    "tin_max": float(tin.max()),
                }
            )
    data = pd.DataFrame(rows)
    data.to_csv(ANALYSIS / f"{PREFIX}_building_kpis.csv", index=False)
    return data


def mean_building_kpis(building_kpis: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "ramping_average",
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


def district_frame(run: dict[str, object]) -> pd.DataFrame:
    loads = []
    heats = []
    time = None
    for building_id in building_ids():
        data = read_rollout(building_id, str(run["label"]))
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
    return pd.DataFrame(
        {
            "controller": run["controller"],
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([v[:n] for v in loads], axis=0),
            "heat": np.sum([v[:n] for v in heats], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
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
        daily_idx = int(hourly["load"].idxmax())
        total = float(load.sum())
        peak = float(load.max())
        ramp = float(np.abs(np.diff(load)).sum())
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


def weighted_terms() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for building_id in building_ids():
            data = read_rollout(building_id, str(run["label"]))
            tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            tsp = pd.to_numeric(data["scheduled_setpoint"], errors="coerce").to_numpy(dtype=float)
            bias = pd.to_numeric(data["setpoint_bias"], errors="coerce").to_numpy(dtype=float)
            tref = tsp + bias
            cold_guard = np.maximum(tsp - 2.0 - tin, 0.0)
            hot_guard = np.maximum(tin - tsp - 2.0, 0.0)
            track = float(run["track_weight"]) * ((tin - tref) / 2.0) ** 2
            cold = float(run["cold_weight"]) * (cold_guard / 1.5) ** 2
            hot = float(run["hot_weight"]) * (hot_guard / 2.0) ** 2
            stage = track + cold + hot
            guard = (cold_guard > 0.0) | (hot_guard > 0.0)
            terms = {
                "tracking_target": track,
                "guard_cold": cold,
                "guard_hot": hot,
                "stage_cost": stage,
                "guard_violation_%": 100.0 * guard.astype(float),
                "mae_to_true_tsp_c": np.abs(tin - tsp),
                "mae_to_biased_tref_c": np.abs(tin - tref),
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


def rollout_delta() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for i, building_id in enumerate(building_ids(), start=1):
        old = read_rollout(building_id, OLD_LABEL)
        new = read_rollout(building_id, NEW_LABEL)
        n = min(len(old), len(new))
        row = {"building": f"Building {i}", "building_id": building_id}
        for col, out in [
            ("action", "action"),
            ("heating_electricity_consumption", "heat_kwh"),
            ("net_electricity_consumption", "load_kwh"),
            ("indoor_temperature", "tin_c"),
        ]:
            diff = (
                pd.to_numeric(new[col].iloc[:n], errors="coerce").to_numpy(dtype=float)
                - pd.to_numeric(old[col].iloc[:n], errors="coerce").to_numpy(dtype=float)
            )
            row[f"{out}_mae"] = float(np.nanmean(np.abs(diff)))
            row[f"{out}_max_abs"] = float(np.nanmax(np.abs(diff)))
        rows.append(row)
    per_building = pd.DataFrame(rows)
    aggregate = pd.DataFrame(
        [
            {
                "scope": "mean over buildings",
                "action_mae": per_building["action_mae"].mean(),
                "action_max_abs": per_building["action_max_abs"].max(),
                "heat_kwh_mae": per_building["heat_kwh_mae"].mean(),
                "heat_kwh_max_abs": per_building["heat_kwh_max_abs"].max(),
                "load_kwh_mae": per_building["load_kwh_mae"].mean(),
                "load_kwh_max_abs": per_building["load_kwh_max_abs"].max(),
                "tin_c_mae": per_building["tin_c_mae"].mean(),
                "tin_c_max_abs": per_building["tin_c_max_abs"].max(),
            }
        ]
    )
    per_building.to_csv(ANALYSIS / f"{PREFIX}_rollout_delta_by_building.csv", index=False)
    aggregate.to_csv(ANALYSIS / f"{PREFIX}_rollout_delta_summary.csv", index=False)
    return aggregate, per_building.sort_values("action_mae", ascending=False)


def plot_comfort_grid(run: dict[str, object], building_kpis: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{run['label']}_comfort_all25.png"
    fig, axes = plt.subplots(5, 5, figsize=(20, 13), sharex=True, sharey=False)
    axes = axes.ravel()
    for i, building_id in enumerate(building_ids(), start=1):
        ax = axes[i - 1]
        data = read_rollout(building_id, str(run["label"]))
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
    fig.suptitle(f"{run['short']} | 2R2C+Kalman MPC | February comfort", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_district(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
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
    return timeseries_path, daily_path


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
    terms = weighted_terms()
    delta_summary, delta_buildings = rollout_delta()
    comfort_paths = [plot_comfort_grid(run, building_kpis) for run in RUNS]
    timeseries_path, daily_path = plot_district(frames)

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
    per_building_display = building_kpis[
        [
            "controller",
            "building",
            "electricity_consumption_total",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "kelvin_hours",
            "daily_peak_average",
            "tin_min",
            "tin_max",
        ]
    ].rename(columns={"electricity_consumption_total": "energy", "discomfort_%": "disc_%", "kelvin_hours": "KH"})
    worst = per_building_display.sort_values("disc_%", ascending=False).head(12)
    term_mean = terms.pivot(index="controller", columns="term", values="mean").reset_index()
    term_mean.columns.name = None
    term_max = terms.pivot(index="controller", columns="term", values="max").reset_index()
    term_max.columns.name = None

    source = f"""{MARKER}

MPC-only sanity check before rerunning SAC/RLMPC.

Both rows use:

- no occupant February deployment;
- January-only `2R2C+Kalman` RC fits;
- horizon 24;
- true setpoint tracking with the same `+1/-1 C` time-bias schedule;
- true-setpoint guard band `Tsp +/- 2 C`;
- no GridStress, no energy term, no peak/ramp/preheat/target term.

Previous no-GridStress MPC:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

New thesis-clean comfort-only MPC:

`D_k = 0.1*((Tin-(Tsp+bias))/2)^2 + 0.7*(max(Tsp-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tsp-2,0)/2)^2`

Run labels:

- `{OLD_LABEL}`: previous no-GridStress MPC with old `3/60/15` weights.
- `{NEW_LABEL}`: new normalized-D MPC with weights `0.1/0.7/0.2`.

Mean building KPIs:

{markdown_table(mean_display[["controller", "energy", "disc_%", "cold_%", "hot_%", "KH", "daily_peak", "ramping_average", "tin_min", "tin_max"]], digits=4)}

District aggregate KPIs:

{markdown_table(dist_display[["controller", "total_kwh", "total_vs_CL", "peak_kwh", "peak_vs_CL", "daily_peak_kWh", "daily_peak_hr", "ramping_abs_kwh", "ramp_vs_CL", "target_RMSE", "above_target_kwh"]], digits=4)}

All-building KPIs:

{markdown_table(per_building_display, digits=4)}

Worst comfort rows:

{markdown_table(worst, digits=4)}

Weighted term diagnostics - mean:

{markdown_table(term_mean, digits=4)}

Weighted term diagnostics - max:

{markdown_table(term_max, digits=4)}

Rollout deltas, new normalized-D minus previous no-GridStress:

{markdown_table(delta_summary, digits=6)}

Largest per-building rollout deltas:

{markdown_table(delta_buildings.head(10), digits=6)}

Saved compact tables:

- `analysis/{PREFIX}_building_kpis.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_term_diagnostics.csv`
- `analysis/{PREFIX}_term_diagnostics_by_building.csv`
- `analysis/{PREFIX}_rollout_delta_summary.csv`
- `analysis/{PREFIX}_rollout_delta_by_building.csv`
- `analysis/{PREFIX}_district_mean_hourly_profile.csv`

Takeaway:

The normalized `0.1/0.7/0.2` comfort-only MPC reproduces the previous no-GridStress MPC almost exactly in realized behavior. That means this is a cleaner thesis version of the same controller: no scientifically awkward `3/60/15` sum, no GridStress term, and no hidden energy/load objective.
"""
    image_source = "\n\n".join(
        [
            image_md(daily_path, "MPC normalized-D mean hourly district load"),
            image_md(timeseries_path, "MPC normalized-D district load timeseries"),
            image_md(comfort_paths[0], "Previous no-GridStress MPC plot_comfort all buildings"),
            image_md(comfort_paths[1], "New normalized-D MPC plot_comfort all buildings"),
        ]
    )
    append_notebooks([nbf.v4.new_markdown_cell(source), nbf.v4.new_markdown_cell(image_source)])

    print("Mean building KPIs:")
    print(mean_display.to_string(index=False))
    print("\nDistrict KPIs:")
    print(dist_display.to_string(index=False))
    print("\nRollout delta summary:")
    print(delta_summary.to_string(index=False))
    print(f"\nSaved figures in {FIG_DIR}")
    print("Appended notebooks:")
    for notebook in NOTEBOOKS:
        print(f"- {notebook}")


if __name__ == "__main__":
    main()
