#!/usr/bin/env python
"""Append Building 11 2R2C fit diagnostics and clean March MPC comfort plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, resolve_building_entries
from src.models.fit_rc import evaluate_1r1c_fit_from_arrays, squeeze_citylearn_dataset
from src.models.rc_1r1c import RC1R1CModel, RC1R1CParams
from src.models.rc_2r2c import RC2R2CModel, RC2R2CParams


BUILDINGS = resolve_building_entries("5,8,11,14,19", DATASET_NAME)
BUILDING_11 = resolve_building_entries("11", DATASET_NAME)[0]
MPC_LABEL = "ce1_time_bias_guard_1c_1r1c_janfeb_march"
SECTION_MARKER = "## Building 11 2R2C Check + Clean March MPC Comfort"
END_MARKER = "<!-- END_BUILDING11_2R2C_MARCH_MPC_COMFORT -->"
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / "building11_2r2c_march_mpc_comfort"
RESULTS = Path("results")

OBSERVATIONS = [
    "month",
    "hour",
    "outdoor_dry_bulb_temperature",
    "direct_solar_irradiance",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "solar_generation",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point",
    "comfort_band",
    "indoor_dry_bulb_temperature_cooling_delta",
    "indoor_dry_bulb_temperature_heating_delta",
    "hvac_mode",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "cooling_electricity_consumption",
    "heating_electricity_consumption",
    "dhw_electricity_consumption",
    "electrical_storage_electricity_consumption",
]
IDX = {name: OBSERVATIONS.index(name) for name in OBSERVATIONS}


def _load_1r1c(path: Path) -> RC1R1CModel:
    return RC1R1CModel(RC1R1CParams(**json.loads(path.read_text())))


def _load_2r2c(path: Path) -> RC2R2CModel:
    return RC2R2CModel(RC2R2CParams(**json.loads(path.read_text())))


def _load_dones(path: Path) -> np.ndarray | None:
    data = np.load(path)
    if "dones" not in data.files:
        return None
    return np.asarray(data["dones"]).reshape(-1)


def _segments(n: int, dones: np.ndarray | None) -> list[tuple[int, int]]:
    if dones is None:
        return [(0, n)]
    done_arr = np.asarray(dones).reshape(-1).astype(bool)
    ends = np.flatnonzero(done_arr) + 1
    starts = np.r_[0, ends[:-1]]
    out = [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= 2]
    return out or [(0, n)]


def evaluate_2r2c_fit(model: RC2R2CModel, x, u, xp, *, dones=None) -> dict[str, float]:
    tin = x[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
    target_next = xp[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
    tout = x[:, IDX["outdoor_dry_bulb_temperature"]].astype(float)
    solar = x[:, IDX["direct_solar_irradiance"]].astype(float)
    u = np.asarray(u, dtype=float)

    one_err_parts = []
    free_err_parts = []
    for start, end in _segments(len(tin), dones):
        tm = float(tin[start])
        one = []
        for k in range(start, end):
            tin_next, tm_next = model.predict_next(tin[k], tm, tout[k], solar[k], u[k])
            one.append(tin_next)
            tm = model.params.a21 * tin[k] + model.params.a22 * tm
        one = np.asarray(one, dtype=float)
        one_err_parts.append(one - target_next[start:end])

        free = model.rollout(tin[start], tout[start:end], solar[start:end], u[start:end])
        free_err_parts.append(free[1:] - tin[start + 1:end])

    one_err = np.concatenate(one_err_parts)
    free_err = np.concatenate(free_err_parts)

    def rmse(values):
        return float(np.sqrt(np.nanmean(values**2)))

    def mae(values):
        return float(np.nanmean(np.abs(values)))

    def bias(values):
        return float(np.nanmean(values))

    return {
        "one_step_rmse_c": rmse(one_err),
        "one_step_mae_c": mae(one_err),
        "one_step_bias_c": bias(one_err),
        "free_run_rmse_c": rmse(free_err),
        "free_run_mae_c": mae(free_err),
        "free_run_bias_c": bias(free_err),
    }


def build_building11_fit_table() -> pd.DataFrame:
    building = BUILDING_11
    dataset_specs = [
        (
            "Jan+Feb PI train",
            RESULTS / "datasets" / "jan_feb_pi" / building["result_id"] / "jan_feb_pi_dataset.npz",
        ),
        (
            "March PI holdout",
            RESULTS / "datasets" / "march_pi" / building["result_id"] / "march_pi_dataset.npz",
        ),
    ]
    model_specs = [
        (
            "Jan+Feb PI robust 1R1C",
            "1r1c",
            RESULTS / "models" / "rc" / f"1state_rc_{building['short_name']}_jan_feb_pi_robust.json",
        ),
        (
            "Former Jan 2R2C",
            "2r2c",
            RESULTS / "models" / "rc" / f"2r2c_{building['short_name']}_january.json",
        ),
        (
            "Jan+Feb PI 2R2C",
            "2r2c",
            RESULTS / "models" / "rc" / f"2r2c_{building['short_name']}_jan_feb_pi.json",
        ),
    ]
    rows = []
    for dataset_name, dataset_path in dataset_specs:
        x, u, xp = squeeze_citylearn_dataset(dataset_path)
        dones = _load_dones(dataset_path)
        for model_name, model_type, model_path in model_specs:
            if model_type == "1r1c":
                metrics = evaluate_1r1c_fit_from_arrays(
                    _load_1r1c(model_path),
                    x,
                    u,
                    xp,
                    indoor_idx=IDX["indoor_dry_bulb_temperature"],
                    outdoor_idx=IDX["outdoor_dry_bulb_temperature"],
                    solar_idx=IDX["direct_solar_irradiance"],
                    dones=dones,
                )
            else:
                metrics = evaluate_2r2c_fit(_load_2r2c(model_path), x, u, xp, dones=dones)
            rows.append(
                {
                    "building": building["display"],
                    "dataset": dataset_name,
                    "model": model_name,
                    "model_path": str(model_path),
                    **metrics,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "building11_2r2c_janfeb_march_rmse.csv", index=False)
    return out


def _kpi_from_file(path: Path) -> dict[str, float]:
    data = pd.read_csv(path)
    out = {}
    for metric in [
        "electricity_consumption_total",
        "daily_peak_average",
        "ramping_average",
        "discomfort_proportion",
        "discomfort_cold_proportion",
        "discomfort_hot_proportion",
    ]:
        subset = data[(data["cost_function"] == metric) & (data["level"] == "building")]
        if subset.empty:
            subset = data[data["cost_function"] == metric]
        out[metric] = float(pd.to_numeric(subset["value"], errors="coerce").iloc[0]) if not subset.empty else float("nan")
    return out


def build_march_mpc_kpis() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        case_dir = RESULTS / "raw" / "mpc" / building["result_id"] / "no_occupant" / MPC_LABEL
        kpis = _kpi_from_file(case_dir / "kpis.csv")
        rows.append(
            {
                "building": building["display"],
                "building_id": building["result_id"],
                "case": MPC_LABEL,
                "electricity_consumption_total": kpis["electricity_consumption_total"],
                "daily_peak_average": kpis["daily_peak_average"],
                "ramping_average": kpis["ramping_average"],
                "discomfort_%": 100.0 * kpis["discomfort_proportion"],
                "cold_%": 100.0 * kpis["discomfort_cold_proportion"],
                "hot_%": 100.0 * kpis["discomfort_hot_proportion"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "march_mpc_janfeb_rc_bad_building_kpis.csv", index=False)
    return out


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
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_comfort_plots() -> list[Path]:
    paths = []
    for building in BUILDINGS:
        rollout_path = RESULTS / "raw" / "mpc" / building["result_id"] / "no_occupant" / MPC_LABEL / "rollout.csv"
        history = pd.read_csv(rollout_path)
        path = FIG_DIR / f"plot_comfort_mpc_janfeb_rc_march_{building['result_id']}.png"
        plot_comfort(
            history,
            title=f"{building['display']} | MPC Jan+Feb RC | March comfort",
            path=path,
        )
        paths.append(path)
    return paths


def make_rmse_plot(fit: pd.DataFrame) -> Path:
    holdout = fit[fit["dataset"] == "March PI holdout"].copy()
    pivot = holdout.pivot(index="model", columns="dataset", values="free_run_rmse_c")
    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind="bar", ax=ax, legend=False)
    ax.set_title("Building 11 March holdout free-run RMSE")
    ax.set_ylabel("RMSE [C]")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "building11_march_2r2c_free_run_rmse.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
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


def _clean_bad_rc_rollout_images(notebook: nbf.NotebookNode) -> None:
    notebook.cells = [
        cell
        for cell in notebook.cells
        if "march_holdout_comfort_building_" not in str(cell.get("source", ""))
    ]


def _append_section(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    _clean_bad_rc_rollout_images(notebook)
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


def append_notebooks(fit: pd.DataFrame, kpis: pd.DataFrame, comfort_paths: list[Path], rmse_path: Path) -> None:
    fit_view = fit[
        [
            "dataset",
            "model",
            "one_step_rmse_c",
            "one_step_mae_c",
            "one_step_bias_c",
            "free_run_rmse_c",
            "free_run_mae_c",
            "free_run_bias_c",
        ]
    ]
    kpi_view = kpis[
        [
            "building",
            "electricity_consumption_total",
            "daily_peak_average",
            "ramping_average",
            "discomfort_%",
            "cold_%",
            "hot_%",
        ]
    ]
    overview = f"""{SECTION_MARKER}

Correction: the previous RC free-run comfort figures were too noisy and not the requested control diagnostic. I removed those image cells from the notebooks. This section uses clean `plot_comfort`-style plots from the actual March MPC rollouts only.

March MPC run:

`{MPC_LABEL}` using Jan+Feb PI robust 1R1C on bad buildings `5,8,11,14,19`, simulation window `1416-2159`.

Building 11 2R2C fit check:

{_markdown_table(fit_view)}

March MPC KPIs:

{_markdown_table(kpi_view)}
"""
    cells = [
        nbf.v4.new_markdown_cell(overview),
        nbf.v4.new_markdown_cell(f"![Building 11 2R2C RMSE]({_relative(rmse_path)})"),
        *[
            nbf.v4.new_markdown_cell(f"![{building['display']} clean MPC comfort]({_relative(path)})")
            for building, path in zip(BUILDINGS, comfort_paths)
        ],
        nbf.v4.new_markdown_cell(END_MARKER),
    ]
    for notebook in [
        Path('experiments/analysis/02_mpc_results.ipynb'),
        Path('experiments/analysis/04_rlmpc_results.ipynb'),
        Path('experiments/analysis/08_sac_results.ipynb'),
        Path('experiments/analysis/10_peak_load_profiles.ipynb'),
        Path('experiments/analysis/11_no_occupant_single_building_comparison.ipynb'),
        Path('experiments/analysis/12_simple_v4_comparison.ipynb'),
    ]:
        _append_section(notebook, cells)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fit = build_building11_fit_table()
    kpis = build_march_mpc_kpis()
    comfort_paths = make_comfort_plots()
    rmse_path = make_rmse_plot(fit)
    append_notebooks(fit, kpis, comfort_paths, rmse_path)


if __name__ == "__main__":
    main()
