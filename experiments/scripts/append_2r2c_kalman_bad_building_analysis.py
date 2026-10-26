#!/usr/bin/env python
"""Append fixed 2R2C+Kalman diagnostics for the bad-building March MPC runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, resolve_building_entries


BUILDINGS = resolve_building_entries("5,8,11,14,19", DATASET_NAME)
ANALYSIS = Path("analysis")
RESULTS = Path("results")
FIG_DIR = Path("figures") / "2r2c_kalman_bad_buildings"

LABEL_1R1C = "ce1_time_bias_guard_1c_1r1c_janfeb_march"
LABEL_2R2C_KALMAN = "ce1_time_bias_guard_1c_2r2c_kalman_janfeb_march"
SECTION_MARKER = "## Fixed 2R2C + Kalman Bad-Building Check"
END_MARKER = "<!-- END_2R2C_KALMAN_BAD_BUILDING_CHECK -->"

NOTEBOOKS = [
    Path('experiments/analysis/02_mpc_results.ipynb'),
    Path('experiments/analysis/04_rlmpc_results.ipynb'),
    Path('experiments/analysis/08_sac_results.ipynb'),
    Path('experiments/analysis/10_peak_load_profiles.ipynb'),
    Path('experiments/analysis/11_no_occupant_single_building_comparison.ipynb'),
    Path('experiments/analysis/12_simple_v4_comparison.ipynb'),
]


def _markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    data = df.copy()
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(data.columns)
    rows = [headers] + data.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(row) for row in rows[1:]]])


def _relative(path: Path) -> str:
    return "../" + path.as_posix()


def _load_summary_case(building_id: str, label: str) -> pd.Series:
    summary_path = RESULTS / "summaries" / "mpc" / building_id / "no_occupant" / "summary.csv"
    data = pd.read_csv(summary_path)
    rows = data[data["case"] == label]
    if rows.empty:
        raise FileNotFoundError(f"Missing summary case {label} in {summary_path}")
    return rows.iloc[-1]


def _load_ramping(building_id: str, label: str) -> float:
    path = RESULTS / "raw" / "mpc" / building_id / "no_occupant" / label / "kpis.csv"
    data = pd.read_csv(path)
    rows = data[data["cost_function"] == "ramping_average"]
    if rows.empty:
        return float("nan")
    return float(pd.to_numeric(rows["value"], errors="coerce").iloc[0])


def _rollout_path(building_id: str, label: str) -> Path:
    return RESULTS / "raw" / "mpc" / building_id / "no_occupant" / label / "rollout.csv"


def build_mpc_kpis() -> pd.DataFrame:
    rows = []
    old_kpis = pd.read_csv(ANALYSIS / "march_mpc_janfeb_rc_bad_building_kpis.csv")
    for building in BUILDINGS:
        old = old_kpis[old_kpis["building_id"] == building["result_id"]].iloc[0]
        rows.append(
            {
                "building": building["display"],
                "building_id": building["result_id"],
                "model": "1R1C Jan+Feb robust",
                "case": LABEL_1R1C,
                "electricity_consumption_total": float(old["electricity_consumption_total"]),
                "daily_peak_average": float(old["daily_peak_average"]),
                "ramping_average": float(old["ramping_average"]),
                "discomfort_%": float(old["discomfort_%"]),
                "cold_%": float(old["cold_%"]),
                "hot_%": float(old["hot_%"]),
                "kelvin_hours": float("nan"),
                "reward_total": float("nan"),
            }
        )

        summary = _load_summary_case(building["result_id"], LABEL_2R2C_KALMAN)
        rows.append(
            {
                "building": building["display"],
                "building_id": building["result_id"],
                "model": "2R2C fixed + Kalman",
                "case": LABEL_2R2C_KALMAN,
                "electricity_consumption_total": float(summary["electricity_consumption_total"]),
                "daily_peak_average": float(summary["daily_peak_average"]),
                "ramping_average": _load_ramping(building["result_id"], LABEL_2R2C_KALMAN),
                "discomfort_%": 100.0 * float(summary["discomfort_proportion"]),
                "cold_%": 100.0 * float(summary["discomfort_cold_proportion"]),
                "hot_%": 100.0 * float(summary["discomfort_hot_proportion"]),
                "kelvin_hours": float(summary["kelvin_hours"]),
                "reward_total": float(summary["reward_total"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "march_mpc_1r1c_vs_2r2c_kalman_bad_building_kpis.csv", index=False)
    return out


def build_kpi_delta(kpis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        sub = kpis[kpis["building"] == building["display"]]
        old = sub[sub["model"] == "1R1C Jan+Feb robust"].iloc[0]
        new = sub[sub["model"] == "2R2C fixed + Kalman"].iloc[0]
        rows.append(
            {
                "building": building["display"],
                "elec_delta": new["electricity_consumption_total"] - old["electricity_consumption_total"],
                "peak_delta": new["daily_peak_average"] - old["daily_peak_average"],
                "ramp_delta": new["ramping_average"] - old["ramping_average"],
                "discomfort_delta_pp": new["discomfort_%"] - old["discomfort_%"],
                "cold_delta_pp": new["cold_%"] - old["cold_%"],
                "hot_delta_pp": new["hot_%"] - old["hot_%"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "march_mpc_2r2c_kalman_bad_building_kpi_deltas.csv", index=False)
    return out


def load_forecast_comparison() -> pd.DataFrame:
    path = ANALYSIS / "1r1c_vs_2r2c_kalman_bad_buildings_comparison.csv"
    data = pd.read_csv(path)
    return data


def plot_forecast_rmse(forecast: pd.DataFrame) -> Path:
    holdout = forecast[forecast["dataset"] == "March holdout"].copy()
    pivot = holdout.pivot(index="building", columns="model", values="horizon_24h_rmse_c")
    pivot = pivot[["1R1C Jan+Feb robust", "2R2C fixed propagate", "2R2C fixed + Kalman"]]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("March holdout 24h forecast RMSE")
    ax.set_ylabel("RMSE [C]")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = FIG_DIR / "march_holdout_24h_forecast_rmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mpc_kpi_bars(kpis: pd.DataFrame) -> Path:
    view = kpis.pivot(index="building", columns="model", values="discomfort_%")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    view.plot(kind="bar", ax=ax, width=0.72, color=["tab:blue", "tab:green"])
    ax.set_title("March MPC discomfort on bad buildings")
    ax.set_ylabel("Discomfort [%]")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = FIG_DIR / "march_mpc_discomfort_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_energy_peak(kpis: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    for ax, metric, title in [
        (axes[0], "electricity_consumption_total", "Electricity consumption KPI"),
        (axes[1], "daily_peak_average", "Daily peak KPI"),
    ]:
        pivot = kpis.pivot(index="building", columns="model", values=metric)
        pivot.plot(kind="bar", ax=ax, width=0.72, color=["tab:blue", "tab:green"])
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend().remove()
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    path = FIG_DIR / "march_mpc_energy_peak_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mean_hourly_load() -> Path:
    rows = []
    for building in BUILDINGS:
        for label, model in [
            (LABEL_1R1C, "1R1C Jan+Feb robust"),
            (LABEL_2R2C_KALMAN, "2R2C fixed + Kalman"),
        ]:
            history = pd.read_csv(_rollout_path(building["result_id"], label))
            for _, row in history.iterrows():
                rows.append(
                    {
                        "model": model,
                        "building": building["display"],
                        "hour": int(row["time_step"]) % 24,
                        "net_electricity_consumption": float(row["net_electricity_consumption"]),
                        "heating_electricity_consumption": float(row["heating_electricity_consumption"]),
                    }
                )
    data = pd.DataFrame(rows)
    hourly = data.groupby(["model", "hour"], as_index=False)[["net_electricity_consumption", "heating_electricity_consumption"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for model, group in hourly.groupby("model"):
        group = group.sort_values("hour")
        ax.plot(group["hour"], group["net_electricity_consumption"], marker="o", label=model)
    ax.set_title("Mean hourly load on bad buildings - March MPC")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean summed electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = FIG_DIR / "march_bad_building_mean_hourly_load.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comfort(history: pd.DataFrame, *, title: str, path: Path) -> None:
    x = np.arange(len(history))
    tin = pd.to_numeric(history["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(history["scheduled_setpoint"], errors="coerce").to_numpy(dtype=float)
    effective = pd.to_numeric(history.get("tracking_setpoint", history["scheduled_setpoint"]), errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, tin, label="Indoor temperature", linewidth=2)
    ax.plot(x, baseline, "--", label="Baseline schedule", linewidth=2)
    ax.plot(x, effective, label="Effective setpoint", linewidth=2, color="tab:orange")
    ax.fill_between(x, 20.0, 24.0, color="gray", alpha=0.2, label="Comfort band (20.0-24.0 degC)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Temperature [degC]")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_comfort_plots() -> list[Path]:
    paths = []
    for building in BUILDINGS:
        history = pd.read_csv(_rollout_path(building["result_id"], LABEL_2R2C_KALMAN))
        path = FIG_DIR / f"plot_comfort_2r2c_kalman_march_{building['result_id']}.png"
        plot_comfort(
            history,
            title=f"{building['display']} | 2R2C+Kalman MPC | March comfort",
            path=path,
        )
        paths.append(path)
    return paths


def _section_cells(
    forecast: pd.DataFrame,
    kpis: pd.DataFrame,
    deltas: pd.DataFrame,
    figure_paths: list[Path],
    comfort_paths: list[Path],
) -> list[nbf.NotebookNode]:
    holdout = forecast[forecast["dataset"] == "March holdout"].copy()
    forecast_view = holdout[
        holdout["model"].isin(["1R1C Jan+Feb robust", "2R2C fixed propagate", "2R2C fixed + Kalman"])
    ][["building", "model", "one_step_rmse_c", "horizon_24h_rmse_c", "free_run_rmse_c"]]
    kpi_view = kpis[
        [
            "building",
            "model",
            "electricity_consumption_total",
            "daily_peak_average",
            "ramping_average",
            "discomfort_%",
            "cold_%",
            "hot_%",
        ]
    ]
    delta_view = deltas[
        [
            "building",
            "elec_delta",
            "peak_delta",
            "ramp_delta",
            "discomfort_delta_pp",
            "cold_delta_pp",
            "hot_delta_pp",
        ]
    ]

    overview = f"""{SECTION_MARKER}

This appends the fixed two-state RC check after adding a Kalman observer for the hidden thermal mass state. The RC parameters are fitted on Jan+Feb PI data for bad buildings `5,8,11,14,19`; March is the rollout holdout.

Files:

- Fixed 2R2C+Kalman models: `results/models/rc/2r2c_*_jan_feb_pi_kalman.json`
- Forecast comparison: `analysis/1r1c_vs_2r2c_kalman_bad_buildings_comparison.csv`
- March MPC KPI comparison: `analysis/march_mpc_1r1c_vs_2r2c_kalman_bad_building_kpis.csv`

March holdout forecast RMSE:

{_markdown_table(forecast_view)}

March MPC KPIs:

{_markdown_table(kpi_view)}

KPI deltas for `2R2C fixed + Kalman - 1R1C Jan+Feb robust`:

{_markdown_table(delta_view)}
"""

    cells: list[nbf.NotebookNode] = [nbf.v4.new_markdown_cell(overview)]
    for path in figure_paths:
        cells.append(nbf.v4.new_markdown_cell(f"![{path.stem}]({_relative(path)})"))
    cells.append(nbf.v4.new_markdown_cell("Clean `plot_comfort`-style March MPC comfort plots for the 2R2C+Kalman controller:"))
    for building, path in zip(BUILDINGS, comfort_paths):
        cells.append(nbf.v4.new_markdown_cell(f"![{building['display']} 2R2C Kalman comfort]({_relative(path)})"))
    cells.append(nbf.v4.new_markdown_cell(END_MARKER))
    return cells


def _append_section(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    start = None
    end = None
    for idx, cell in enumerate(notebook.cells):
        source = str(cell.get("source", ""))
        if SECTION_MARKER in source and start is None:
            start = idx
        if start is not None and END_MARKER in source:
            end = idx + 1
            break
    if start is not None and end is not None:
        notebook.cells[start:end] = cells
        print(f"Updated {SECTION_MARKER} in {path}")
    elif start is None:
        notebook.cells.extend(cells)
        print(f"Appended {SECTION_MARKER} to {path}")
    else:
        notebook.cells[start:start + 1] = cells
        print(f"Replaced partial {SECTION_MARKER} in {path}")
    nbf.write(notebook, path)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    forecast = load_forecast_comparison()
    kpis = build_mpc_kpis()
    deltas = build_kpi_delta(kpis)

    figure_paths = [
        plot_forecast_rmse(forecast),
        plot_mpc_kpi_bars(kpis),
        plot_energy_peak(kpis),
        plot_mean_hourly_load(),
    ]
    comfort_paths = make_comfort_plots()
    cells = _section_cells(forecast, kpis, deltas, figure_paths, comfort_paths)
    for notebook in NOTEBOOKS:
        _append_section(notebook, cells)


if __name__ == "__main__":
    main()
