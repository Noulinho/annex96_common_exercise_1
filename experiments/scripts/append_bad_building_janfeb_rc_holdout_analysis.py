#!/usr/bin/env python
"""Append Jan+Feb PI RC-fit and March holdout diagnostics for bad buildings."""

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


LABEL = "jan_feb_pi_robust_march_holdout"
BUILDING_SELECTOR = "5,8,11,14,19"
BUILDINGS = resolve_building_entries(BUILDING_SELECTOR, DATASET_NAME)
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / LABEL
MARCH_START = 1416

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

MODELS = [
    {
        "key": "former_january_rc",
        "display": "Former Jan PI RC",
        "path": lambda building: RESULTS / "models" / "rc" / f"1state_rc_{building['short_name']}_january.json",
        "color": "tab:gray",
    },
    {
        "key": "rich_january_id_robust_rc",
        "display": "Rich Jan ID robust RC",
        "path": lambda building: RESULTS / "models" / "rc" / f"1state_rc_{building['short_name']}_jan_id_robust.json",
        "color": "tab:orange",
    },
    {
        "key": "jan_feb_pi_robust_rc",
        "display": "Jan+Feb PI robust RC",
        "path": lambda building: RESULTS / "models" / "rc" / f"1state_rc_{building['short_name']}_jan_feb_pi_robust.json",
        "color": "tab:blue",
    },
]

DATASETS = [
    {
        "key": "january_pi_train",
        "display": "January PI train",
        "path": lambda building: RESULTS / "datasets" / "january_pi" / building["result_id"] / "january_pi_dataset.npz",
    },
    {
        "key": "rich_january_id",
        "display": "Rich January ID",
        "path": lambda building: RESULTS / "datasets" / "january_id" / building["result_id"] / "january_pi_prbs_dataset.npz",
    },
    {
        "key": "jan_feb_pi_train",
        "display": "Jan+Feb PI train",
        "path": lambda building: RESULTS / "datasets" / "jan_feb_pi" / building["result_id"] / "jan_feb_pi_dataset.npz",
    },
    {
        "key": "march_pi_holdout",
        "display": "March PI holdout",
        "path": lambda building: RESULTS / "datasets" / "march_pi" / building["result_id"] / "march_pi_dataset.npz",
    },
]


def _load_model(path: Path) -> RC1R1CModel:
    return RC1R1CModel(RC1R1CParams(**json.loads(path.read_text())))


def _load_dones(path: Path) -> np.ndarray | None:
    data = np.load(path)
    if "dones" not in data.files:
        return None
    return np.asarray(data["dones"]).reshape(-1)


def _dataset_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, u, xp = squeeze_citylearn_dataset(path)
    return x, u, xp


def _free_run_prediction(model: RC1R1CModel, x: np.ndarray, u: np.ndarray, dones: np.ndarray | None) -> np.ndarray:
    tin = x[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
    tout = x[:, IDX["outdoor_dry_bulb_temperature"]].astype(float)
    solar = x[:, IDX["direct_solar_irradiance"]].astype(float)
    pred = np.full_like(tin, np.nan, dtype=float)
    if dones is None:
        segments = [(0, len(tin))]
    else:
        done_arr = np.asarray(dones).reshape(-1).astype(bool)
        ends = np.flatnonzero(done_arr) + 1
        starts = np.r_[0, ends[:-1]]
        segments = [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]
        if not segments:
            segments = [(0, len(tin))]
    for start, end in segments:
        pred[start:end] = model.rollout(tin[start], tout[start:end], solar[start:end], u[start:end])
    return pred


def _one_step_prediction(model: RC1R1CModel, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return model.predict_next(
        x[:, IDX["indoor_dry_bulb_temperature"]].astype(float),
        x[:, IDX["outdoor_dry_bulb_temperature"]].astype(float),
        x[:, IDX["direct_solar_irradiance"]].astype(float),
        u,
    )


def build_fit_metrics() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        for model_spec in MODELS:
            model_path = model_spec["path"](building)
            model = _load_model(model_path)
            for dataset_spec in DATASETS:
                dataset_path = dataset_spec["path"](building)
                x, u, xp = _dataset_arrays(dataset_path)
                metrics = evaluate_1r1c_fit_from_arrays(
                    model,
                    x,
                    u,
                    xp,
                    indoor_idx=IDX["indoor_dry_bulb_temperature"],
                    outdoor_idx=IDX["outdoor_dry_bulb_temperature"],
                    solar_idx=IDX["direct_solar_irradiance"],
                    dones=_load_dones(dataset_path),
                )
                rows.append(
                    {
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "model": model_spec["display"],
                        "model_key": model_spec["key"],
                        "dataset": dataset_spec["display"],
                        "dataset_key": dataset_spec["key"],
                        "model_path": str(model_path),
                        "dataset_path": str(dataset_path),
                        **metrics,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / f"{LABEL}_fit_metrics.csv", index=False)
    return out


def build_march_kpis() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        path = RESULTS / "datasets" / "march_pi" / building["result_id"] / "march_pi_dataset.npz"
        x, u, _ = _dataset_arrays(path)
        tin = x[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
        tsp = x[:, IDX["indoor_dry_bulb_temperature_heating_set_point"]].astype(float)
        band = x[:, IDX["comfort_band"]].astype(float)
        net = x[:, IDX["net_electricity_consumption"]].astype(float)
        heat = x[:, IDX["heating_electricity_consumption"]].astype(float)
        action = np.asarray(u, dtype=float)
        cold = np.maximum(tsp - tin, 0.0)
        hot = np.maximum(tin - tsp, 0.0)
        guard = np.maximum(np.abs(tin - tsp) - band, 0.0)
        rows.append(
            {
                "building": building["display"],
                "building_id": building["result_id"],
                "steps": int(len(tin)),
                "pi_action_mean": float(np.nanmean(action)),
                "pi_action_max": float(np.nanmax(action)),
                "net_electricity_total_kwh": float(np.nansum(net)),
                "heating_electricity_total_kwh": float(np.nansum(heat)),
                "heating_electricity_mean_kwh": float(np.nanmean(heat)),
                "tin_mean_c": float(np.nanmean(tin)),
                "tin_min_c": float(np.nanmin(tin)),
                "tin_max_c": float(np.nanmax(tin)),
                "mae_to_tsp_c": float(np.nanmean(np.abs(tin - tsp))),
                "comfort_guard_violation_%": float(100.0 * np.nanmean(guard > 0.0)),
                "cold_to_tsp_%": float(100.0 * np.nanmean(cold > 0.0)),
                "hot_to_tsp_%": float(100.0 * np.nanmean(hot > 0.0)),
                "mean_guard_excess_c": float(np.nanmean(guard)),
                "max_guard_excess_c": float(np.nanmax(guard)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / f"{LABEL}_march_pi_kpis.csv", index=False)
    return out


def _plot_holdout_fit_bars(fit: pd.DataFrame) -> tuple[Path, Path]:
    holdout = fit[fit["dataset_key"] == "march_pi_holdout"].copy()
    paths = []
    for metric, slug, title in [
        ("one_step_rmse_c", "march_one_step_rmse.png", "March holdout one-step RMSE"),
        ("free_run_rmse_c", "march_free_run_rmse.png", "March holdout free-run RMSE"),
    ]:
        pivot = holdout.pivot(index="building", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("RMSE [C]")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / slug
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths[0], paths[1]


def _plot_training_vs_holdout(fit: pd.DataFrame) -> Path:
    data = fit[
        (fit["model_key"] == "jan_feb_pi_robust_rc")
        & (fit["dataset_key"].isin(["jan_feb_pi_train", "march_pi_holdout"]))
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, metric in zip(axes, ["one_step_rmse_c", "free_run_rmse_c"]):
        pivot = data.pivot(index="building", columns="dataset", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_ylabel("RMSE [C]")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Jan+Feb PI robust RC: training vs March holdout")
    fig.tight_layout()
    path = FIG_DIR / "jan_feb_training_vs_march_holdout.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_march_pi_kpis(kpis: pd.DataFrame) -> Path:
    metrics = [
        "mae_to_tsp_c",
        "comfort_guard_violation_%",
        "heating_electricity_total_kwh",
        "net_electricity_total_kwh",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        ax.bar(kpis["building"], kpis[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Observed March PI rollout KPIs")
    fig.tight_layout()
    path = FIG_DIR / "march_pi_kpis.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_comfort_and_models() -> list[Path]:
    paths = []
    for building in BUILDINGS:
        dataset_path = RESULTS / "datasets" / "march_pi" / building["result_id"] / "march_pi_dataset.npz"
        x, u, xp = _dataset_arrays(dataset_path)
        dones = _load_dones(dataset_path)
        steps = MARCH_START + np.arange(len(x))
        tin = x[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
        tsp = x[:, IDX["indoor_dry_bulb_temperature_heating_set_point"]].astype(float)
        band = x[:, IDX["comfort_band"]].astype(float)
        heat = x[:, IDX["heating_electricity_consumption"]].astype(float)

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(steps, tin, color="black", lw=1.1, label="Actual Tin")
        axes[0].plot(steps, tsp, color="tab:green", lw=0.9, ls="--", label="Tsp")
        axes[0].fill_between(steps, tsp - band, tsp + band, color="tab:green", alpha=0.10, label="+/- comfort band")

        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[2].plot(steps, heat, color="tab:red", lw=1.0, label="Observed PI heating electricity")

        for model_spec in MODELS:
            model = _load_model(model_spec["path"](building))
            free = _free_run_prediction(model, x, u, dones)
            one = _one_step_prediction(model, x, u)
            target_next = xp[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
            n = min(len(steps), len(free), len(one), len(target_next))
            axes[0].plot(steps[:n], free[:n], lw=0.9, color=model_spec["color"], label=f"{model_spec['display']} free-run")
            axes[1].plot(
                steps[:n],
                one[:n] - target_next[:n],
                lw=0.8,
                color=model_spec["color"],
                label=f"{model_spec['display']} one-step error",
            )

        axes[0].set_ylabel("Temperature [C]")
        axes[1].set_ylabel("One-step error [C]")
        axes[2].set_ylabel("Heating [kWh]")
        axes[2].set_xlabel("Simulation time step")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[0].set_ylim(max(5.0, float(np.nanmin(tin)) - 4.0), min(45.0, float(np.nanmax(tin)) + 4.0))
        fig.suptitle(f"{building['display']} March PI holdout: comfort and RC rollout")
        fig.tight_layout()
        path = FIG_DIR / f"march_holdout_comfort_{building['result_id']}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_hourly_residuals() -> Path:
    rows = []
    for building in BUILDINGS:
        dataset_path = RESULTS / "datasets" / "march_pi" / building["result_id"] / "march_pi_dataset.npz"
        x, u, xp = _dataset_arrays(dataset_path)
        target_next = xp[:, IDX["indoor_dry_bulb_temperature"]].astype(float)
        hours = ((MARCH_START + np.arange(len(x))) % 24).astype(int)
        for model_spec in MODELS:
            model = _load_model(model_spec["path"](building))
            err = _one_step_prediction(model, x, u) - target_next
            rows.append(
                pd.DataFrame(
                    {
                        "building": building["display"],
                        "model": model_spec["display"],
                        "hour": hours,
                        "abs_error": np.abs(err),
                        "error": err,
                    }
                )
            )
    data = pd.concat(rows, ignore_index=True)
    summary = data.groupby(["model", "hour"], as_index=False).agg(mae=("abs_error", "mean"), bias=("error", "mean"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for model_spec in MODELS:
        group = summary[summary["model"] == model_spec["display"]]
        axes[0].plot(group["hour"], group["mae"], marker="o", lw=1.2, color=model_spec["color"], label=model_spec["display"])
        axes[1].plot(group["hour"], group["bias"], marker="o", lw=1.2, color=model_spec["color"], label=model_spec["display"])
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("One-step MAE [C]")
    axes[1].set_ylabel("One-step bias [C]")
    axes[1].set_xlabel("Hour of day")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("March holdout one-step residuals by hour")
    fig.tight_layout()
    path = FIG_DIR / "march_hourly_residuals.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(ANALYSIS / f"{LABEL}_march_hourly_residuals_raw.csv", index=False)
    summary.to_csv(ANALYSIS / f"{LABEL}_march_hourly_residuals_summary.csv", index=False)
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


def _append_section(path: Path, marker: str, cells: list[nbf.NotebookNode]) -> None:
    end_marker = "<!-- END_JANFEB_RC_MARCH_HOLDOUT_SECTION -->"
    notebook = nbf.read(path, as_version=4)
    start = None
    end = None
    for idx, cell in enumerate(notebook.cells):
        source = str(cell.get("source", ""))
        if marker in source and start is None:
            start = idx
        if start is not None and end_marker in source:
            end = idx + 1
            break
    if start is not None and end is not None:
        notebook.cells[start:end] = cells
        print(f"Updated {marker} in {path}")
    elif start is None:
        notebook.cells.extend(cells)
        print(f"Appended {marker} to {path}")
    else:
        notebook.cells[start:start + 1] = cells
        print(f"Replaced marker cell for {marker} in {path}")
    nbf.write(notebook, path)


def append_notebooks(fit: pd.DataFrame, kpis: pd.DataFrame, figures: dict[str, Path | list[Path]]) -> None:
    marker = "## Jan+Feb PI 1R1C Fit - March Holdout"
    end_marker = "<!-- END_JANFEB_RC_MARCH_HOLDOUT_SECTION -->"
    holdout = fit[fit["dataset_key"] == "march_pi_holdout"][
        [
            "building",
            "model",
            "one_step_rmse_c",
            "one_step_mae_c",
            "one_step_bias_c",
            "free_run_rmse_c",
            "free_run_mae_c",
            "free_run_bias_c",
        ]
    ]
    train_vs_holdout = fit[
        (fit["model_key"] == "jan_feb_pi_robust_rc")
        & (fit["dataset_key"].isin(["jan_feb_pi_train", "march_pi_holdout"]))
    ][
        [
            "building",
            "dataset",
            "one_step_rmse_c",
            "one_step_bias_c",
            "free_run_rmse_c",
            "free_run_bias_c",
        ]
    ]
    kpi_view = kpis[
        [
            "building",
            "pi_action_mean",
            "net_electricity_total_kwh",
            "heating_electricity_total_kwh",
            "mae_to_tsp_c",
            "comfort_guard_violation_%",
            "tin_min_c",
            "tin_max_c",
        ]
    ]

    overview = f"""{marker}

Bad-fitted buildings: `{BUILDING_SELECTOR}`. PI controlled both fitting months, using Jan+Feb (`0-1415`) for fitting and March (`1416-2159`) as a held-out PI rollout.

The new model is `Jan+Feb PI robust RC`: bounded robust 1R1C fitted on PI-controlled Jan+Feb only. The comparison keeps the former January PI RC and the richer January-ID robust RC.

March holdout fit metrics:

{_markdown_table(holdout)}

Jan+Feb robust RC training-vs-holdout metrics:

{_markdown_table(train_vs_holdout)}

Observed March PI rollout KPIs:

{_markdown_table(kpi_view)}
"""
    common_cells = [
        nbf.v4.new_markdown_cell(overview),
        nbf.v4.new_markdown_cell(f"![March one-step RMSE]({_relative(figures['one_step'])})"),
        nbf.v4.new_markdown_cell(f"![March free-run RMSE]({_relative(figures['free_run'])})"),
        nbf.v4.new_markdown_cell(f"![Training vs holdout]({_relative(figures['train_holdout'])})"),
        nbf.v4.new_markdown_cell(f"![March PI KPIs]({_relative(figures['kpis'])})"),
        nbf.v4.new_markdown_cell(f"![Hourly residuals]({_relative(figures['hourly_residuals'])})"),
    ]
    comfort_cells = [
        nbf.v4.new_markdown_cell(f"![{building['display']} March comfort and RC rollout]({_relative(path)})")
        for building, path in zip(BUILDINGS, figures["comfort"])
    ]
    footer = nbf.v4.new_markdown_cell(end_marker)
    for notebook in [
        Path('experiments/analysis/02_mpc_results.ipynb'),
        Path('experiments/analysis/04_rlmpc_results.ipynb'),
        Path('experiments/analysis/08_sac_results.ipynb'),
        Path('experiments/analysis/10_peak_load_profiles.ipynb'),
        Path('experiments/analysis/11_no_occupant_single_building_comparison.ipynb'),
        Path('experiments/analysis/12_simple_v4_comparison.ipynb'),
    ]:
        cells = [*common_cells]
        if notebook.name in {"02_mpc_results.ipynb", "04_rlmpc_results.ipynb", "11_no_occupant_single_building_comparison.ipynb"}:
            cells.extend(comfort_cells)
        cells.append(footer)
        _append_section(notebook, marker, cells)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fit = build_fit_metrics()
    kpis = build_march_kpis()
    figures: dict[str, Path | list[Path]] = {}
    figures["one_step"], figures["free_run"] = _plot_holdout_fit_bars(fit)
    figures["train_holdout"] = _plot_training_vs_holdout(fit)
    figures["kpis"] = _plot_march_pi_kpis(kpis)
    figures["comfort"] = _plot_comfort_and_models()
    figures["hourly_residuals"] = _plot_hourly_residuals()
    append_notebooks(fit, kpis, figures)


if __name__ == "__main__":
    main()
