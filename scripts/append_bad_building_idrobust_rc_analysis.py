#!/usr/bin/env python
"""Append bad-building robust-ID RC comparison for MPC and GNU-RL/RLMPC."""

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
from src.rewards import load_simple_v4_price_profile, time_bias_setpoint_profile


BASE_LABEL = "ce1_time_bias_guard_1c_1r1c"
IDROBUST_LABEL = "ce1_time_bias_guard_1c_1r1c_idrobust"
BUILDING_SELECTOR = "5,8,11,14,19"
BUILDINGS = resolve_building_entries(BUILDING_SELECTOR, DATASET_NAME)
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / f"{IDROBUST_LABEL}_bad_buildings"
FEB_START = 744

NOTEBOOK_MPC_OBSERVATIONS = [
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

OBS_IDX = {
    "indoor": NOTEBOOK_MPC_OBSERVATIONS.index("indoor_dry_bulb_temperature"),
    "outdoor": NOTEBOOK_MPC_OBSERVATIONS.index("outdoor_dry_bulb_temperature"),
    "solar": NOTEBOOK_MPC_OBSERVATIONS.index("direct_solar_irradiance"),
}

WEIGHTS = {
    "comfort_weight": 0.5,
    "shift_weight": 0.5,
    "comfort_track_weight": 3.0,
    "comfort_track_scale": 2.0,
    "comfort_cold_weight": 60.0,
    "comfort_cold_scale": 1.5,
    "comfort_hot_weight": 15.0,
    "comfort_hot_scale": 2.0,
    "guard_band": 2.0,
    "power_scale": 5.0,
    "bias_amplitude": 1.0,
}

RUNS = [
    {
        "key": "mpc_old",
        "controller": "mpc",
        "label": BASE_LABEL,
        "display": "MPC former RC",
        "style": {"color": "tab:green", "linestyle": "-"},
    },
    {
        "key": "mpc_idrobust",
        "controller": "mpc",
        "label": IDROBUST_LABEL,
        "display": "MPC robust-ID RC",
        "style": {"color": "tab:blue", "linestyle": "-"},
    },
    {
        "key": "rlmpc_old",
        "controller": "rlmpc",
        "label": BASE_LABEL,
        "display": "GNU-RL/RLMPC former RC",
        "style": {"color": "tab:orange", "linestyle": "--"},
    },
    {
        "key": "rlmpc_idrobust",
        "controller": "rlmpc",
        "label": IDROBUST_LABEL,
        "display": "GNU-RL/RLMPC robust-ID RC",
        "style": {"color": "tab:red", "linestyle": "--"},
    },
]

KPI_NAMES = [
    "electricity_consumption_total",
    "discomfort_proportion",
    "discomfort_cold_proportion",
    "discomfort_hot_proportion",
    "daily_peak_average",
    "ramping_average",
    "zero_net_energy",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _rollout_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label / "rollout.csv"


def _kpis_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label / "kpis.csv"


def _rc_path(building: dict, *, robust: bool) -> Path:
    suffix = "jan_id_robust" if robust else "january"
    return RESULTS / "models" / "rc" / f"1state_rc_{building['short_name']}_{suffix}.json"


def _dataset_path(building: dict, *, rich_id: bool) -> Path:
    if rich_id:
        return RESULTS / "datasets" / "january_id" / building["result_id"] / "january_pi_prbs_dataset.npz"
    return RESULTS / "datasets" / "january_pi" / building["result_id"] / "january_pi_dataset.npz"


def _load_model(path: Path) -> RC1R1CModel:
    params = json.loads(path.read_text())
    return RC1R1CModel(RC1R1CParams(**params))


def _load_dones(path: Path) -> np.ndarray | None:
    data = np.load(path)
    if "dones" not in data.files:
        return None
    dones = np.asarray(data["dones"])
    return dones.reshape(-1)


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint-like column.")


def _time_step(data: pd.DataFrame) -> np.ndarray:
    if "time_step" in data.columns:
        return pd.to_numeric(data["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(data), dtype=int)


def _heating(data: pd.DataFrame) -> np.ndarray:
    if "heating_electricity_consumption" in data.columns:
        values = pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0)
    else:
        values = pd.to_numeric(data["action"], errors="coerce").fillna(0.0)
    return np.maximum(values.to_numpy(dtype=float), 0.0)


def _bias_and_stress(n: int) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(
        time_bias_setpoint_profile(
            simulation_start=FEB_START,
            simulation_end=FEB_START + n - 1,
            amplitude=WEIGHTS["bias_amplitude"],
        ),
        dtype=float,
    )
    stress = np.asarray(load_simple_v4_price_profile(simulation_start=FEB_START, exp_beta=2.0), dtype=float)
    if stress.size < n:
        stress = np.resize(stress, n)
    return bias[:n], stress[:n]


def _extract_raw_kpis(controller: str, building: dict, label: str) -> dict[str, float]:
    data = _read_csv(_kpis_path(controller, building["result_id"], label))
    row: dict[str, float] = {}
    for metric in KPI_NAMES:
        subset = data[data["cost_function"] == metric]
        if subset.empty:
            row[metric] = float("nan")
            continue
        building_subset = subset[subset["level"] == "building"]
        chosen = building_subset if not building_subset.empty else subset
        row[metric] = float(pd.to_numeric(chosen["value"], errors="coerce").iloc[0])
    return row


def _cost_terms(data: pd.DataFrame) -> dict[str, np.ndarray]:
    tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    tsp = _setpoint(data)
    heat = _heating(data)
    n = min(len(tin), len(tsp), len(heat))
    tin, tsp, heat = tin[:n], tsp[:n], heat[:n]
    bias, stress = _bias_and_stress(n)
    tref = tsp + bias
    cold_guard = np.maximum(tsp - WEIGHTS["guard_band"] - tin, 0.0)
    hot_guard = np.maximum(tin - tsp - WEIGHTS["guard_band"], 0.0)
    tracking = (
        WEIGHTS["comfort_weight"]
        * WEIGHTS["comfort_track_weight"]
        * ((tin - tref) / WEIGHTS["comfort_track_scale"]) ** 2
    )
    cold = (
        WEIGHTS["comfort_weight"]
        * WEIGHTS["comfort_cold_weight"]
        * (cold_guard / WEIGHTS["comfort_cold_scale"]) ** 2
    )
    hot = (
        WEIGHTS["comfort_weight"]
        * WEIGHTS["comfort_hot_weight"]
        * (hot_guard / WEIGHTS["comfort_hot_scale"]) ** 2
    )
    shift = WEIGHTS["shift_weight"] * stress * heat / WEIGHTS["power_scale"]
    stage = tracking + cold + hot + shift
    return {
        "tin": tin,
        "tsp": tsp,
        "bias": bias,
        "heat": heat,
        "tracking_target": tracking,
        "guard_cold": cold,
        "guard_hot": hot,
        "time_weighted_heat": shift,
        "stage_cost": stage,
        "tin_minus_tsp": tin - tsp,
        "tin_minus_biased_tref": tin - tref,
        "guard_violation": ((cold_guard > 0.0) | (hot_guard > 0.0)).astype(float),
    }


def build_fit_metrics() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        for model_key, robust in [("former_january_rc", False), ("robust_rich_id_rc", True)]:
            model_path = _rc_path(building, robust=robust)
            model = _load_model(model_path)
            for dataset_key, rich_id in [("january_pi", False), ("rich_january_id", True)]:
                dataset_path = _dataset_path(building, rich_id=rich_id)
                x, u, xp = squeeze_citylearn_dataset(dataset_path)
                metrics = evaluate_1r1c_fit_from_arrays(
                    model,
                    x,
                    u,
                    xp,
                    indoor_idx=OBS_IDX["indoor"],
                    outdoor_idx=OBS_IDX["outdoor"],
                    solar_idx=OBS_IDX["solar"],
                    dones=_load_dones(dataset_path),
                )
                rows.append(
                    {
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "model": model_key,
                        "dataset": dataset_key,
                        "rc_model_path": str(model_path),
                        **metrics,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_fit_metrics.csv", index=False)
    return out


def build_kpis_and_terms() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kpi_rows = []
    term_rows = []
    hourly_rows = []
    for run in RUNS:
        for building in BUILDINGS:
            rollout = _read_csv(_rollout_path(run["controller"], building["result_id"], run["label"]))
            terms = _cost_terms(rollout)
            raw = _extract_raw_kpis(run["controller"], building, run["label"])
            stage_cost_sum = float(np.nansum(terms["stage_cost"]))
            guard_violation_pct = float(100.0 * np.nanmean(terms["guard_violation"]))
            kpi_rows.append(
                {
                    "controller": run["display"],
                    "controller_key": run["key"],
                    "building": building["display"],
                    "building_id": building["result_id"],
                    "label": run["label"],
                    "electricity_consumption_total": raw["electricity_consumption_total"],
                    "daily_peak_average": raw["daily_peak_average"],
                    "ramping_average": raw["ramping_average"],
                    "zero_net_energy": raw["zero_net_energy"],
                    "discomfort_%": 100.0 * raw["discomfort_proportion"],
                    "cold_%": 100.0 * raw["discomfort_cold_proportion"],
                    "hot_%": 100.0 * raw["discomfort_hot_proportion"],
                    "guard_violation_%": guard_violation_pct,
                    "mae_to_true_tsp_c": float(np.nanmean(np.abs(terms["tin_minus_tsp"]))),
                    "mae_to_biased_tref_c": float(np.nanmean(np.abs(terms["tin_minus_biased_tref"]))),
                    "stage_cost_sum": stage_cost_sum,
                    "stage_cost_mean": float(np.nanmean(terms["stage_cost"])),
                    "heating_kwh_mean": float(np.nanmean(terms["heat"])),
                }
            )
            for term in [
                "tracking_target",
                "guard_cold",
                "guard_hot",
                "time_weighted_heat",
                "stage_cost",
                "tin_minus_tsp",
                "tin_minus_biased_tref",
                "guard_violation",
                "heat",
            ]:
                values = terms[term]
                term_rows.append(
                    {
                        "controller": run["display"],
                        "controller_key": run["key"],
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "term": term,
                        "mean": float(np.nanmean(values)),
                        "max": float(np.nanmax(values)),
                        "min": float(np.nanmin(values)),
                    }
                )
            time_step = _time_step(rollout)
            n = min(len(time_step), len(terms["tin_minus_tsp"]))
            hourly_rows.append(
                pd.DataFrame(
                    {
                        "controller": run["display"],
                        "controller_key": run["key"],
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "hour": (time_step[:n] + FEB_START) % 24,
                        "tin_minus_tsp": terms["tin_minus_tsp"][:n],
                        "heat": terms["heat"][:n],
                        "guard_violation": terms["guard_violation"][:n],
                        "stage_cost": terms["stage_cost"][:n],
                    }
                )
            )

    kpis = pd.DataFrame(kpi_rows)
    deltas = []
    for controller_prefix in ["mpc", "rlmpc"]:
        former_key = f"{controller_prefix}_old"
        robust_key = f"{controller_prefix}_idrobust"
        former = kpis[kpis["controller_key"] == former_key].set_index("building_id")
        robust = kpis[kpis["controller_key"] == robust_key].set_index("building_id")
        for building_id in sorted(set(former.index).intersection(robust.index)):
            for metric in [
                "electricity_consumption_total",
                "daily_peak_average",
                "discomfort_%",
                "cold_%",
                "hot_%",
                "guard_violation_%",
                "stage_cost_sum",
            ]:
                deltas.append(
                    {
                        "controller_family": controller_prefix.upper() if controller_prefix == "mpc" else "GNU-RL/RLMPC",
                        "building": robust.loc[building_id, "building"],
                        "building_id": building_id,
                        "metric": metric,
                        "former": float(former.loc[building_id, metric]),
                        "idrobust": float(robust.loc[building_id, metric]),
                        "delta_idrobust_minus_former": float(robust.loc[building_id, metric] - former.loc[building_id, metric]),
                    }
                )
    terms = pd.DataFrame(term_rows)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    kpis.to_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_kpis.csv", index=False)
    pd.DataFrame(deltas).to_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_kpi_deltas.csv", index=False)
    terms.to_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_term_diagnostics.csv", index=False)
    hourly.to_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_hourly_response.csv", index=False)
    return kpis, pd.DataFrame(deltas), terms


def _plot_fit_metrics(fit: pd.DataFrame) -> tuple[Path, Path]:
    paths = []
    for metric, slug, title in [
        ("one_step_rmse_c", "one_step_fit_rmse.png", "One-step 1R1C fit RMSE"),
        ("free_run_rmse_c", "free_run_fit_rmse.png", "Free-run 1R1C fit RMSE"),
    ]:
        data = fit[fit["dataset"].isin(["january_pi", "rich_january_id"])].copy()
        data["series"] = data["model"] + " on " + data["dataset"]
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot = data.pivot_table(index="building", columns="series", values=metric, aggfunc="mean")
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


def _plot_kpi_deltas(deltas: pd.DataFrame) -> Path:
    data = deltas[deltas["metric"].isin(["discomfort_%", "electricity_consumption_total", "daily_peak_average", "stage_cost_sum"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=False)
    axes = axes.ravel()
    for ax, metric in zip(axes, ["discomfort_%", "electricity_consumption_total", "daily_peak_average", "stage_cost_sum"]):
        metric_data = data[data["metric"] == metric]
        pivot = metric_data.pivot(index="building", columns="controller_family", values="delta_idrobust_minus_former")
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"ID-robust minus former: {metric}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "bad_building_kpi_deltas.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_selected_load_profiles() -> Path:
    rows = []
    for run in RUNS:
        for building in BUILDINGS:
            data = _read_csv(_rollout_path(run["controller"], building["result_id"], run["label"]))
            time_step = _time_step(data)
            load = pd.to_numeric(data["net_electricity_consumption"], errors="coerce").to_numpy(dtype=float)
            n = min(len(time_step), len(load))
            rows.append(
                pd.DataFrame(
                    {
                        "controller": run["display"],
                        "time_step": time_step[:n],
                        "hour": (time_step[:n] + FEB_START) % 24,
                        "load": load[:n],
                    }
                )
            )
    data = pd.concat(rows, ignore_index=True)
    aggregate = data.groupby(["controller", "time_step", "hour"], as_index=False)["load"].sum()
    hourly = aggregate.groupby(["controller", "hour"], as_index=False)["load"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    order = [run["display"] for run in RUNS]
    styles = {run["display"]: run["style"] for run in RUNS}
    for name in order:
        group = hourly[hourly["controller"] == name]
        ax.plot(
            group["hour"],
            group["load"],
            marker="o",
            lw=1.4,
            label=name,
            **styles[name],
        )
    ax.set_title("Mean hourly aggregate load for bad buildings - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Selected-building electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "bad_building_mean_hourly_load.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_hourly_response() -> Path:
    data = _read_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_hourly_response.csv")
    hourly = data.groupby(["controller", "hour"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for run in RUNS:
        group = hourly[hourly["controller"] == run["display"]]
        axes[0].plot(group["hour"], group["tin_minus_tsp"], marker="o", lw=1.3, label=run["display"], **run["style"])
        axes[1].plot(group["hour"], group["heat"], marker="o", lw=1.3, label=run["display"], **run["style"])
        axes[2].plot(group["hour"], 100.0 * group["guard_violation"], marker="o", lw=1.3, label=run["display"], **run["style"])
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("Tin - Tsp [C]")
    axes[1].set_ylabel("Heating electricity [kWh]")
    axes[2].set_ylabel("Guard violation [%]")
    axes[2].set_xlabel("Hour of day")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Hourly comfort/heating response on bad buildings")
    fig.tight_layout()
    path = FIG_DIR / "bad_building_hourly_response.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_terms(terms: pd.DataFrame) -> Path:
    plot_terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]
    summary = (
        terms[terms["term"].isin(plot_terms)]
        .groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
    )
    pivot = summary.pivot(index="term", columns="controller", values="mean").loc[plot_terms]
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mean weighted cost terms on bad buildings")
    ax.set_ylabel("Mean weighted term")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "bad_building_weighted_terms.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    summary.to_csv(ANALYSIS / f"{IDROBUST_LABEL}_bad_building_term_summary.csv", index=False)
    return path


def _plot_comfort_by_building() -> list[Path]:
    paths = []
    for building in BUILDINGS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, run in zip(axes, RUNS):
            data = _read_csv(_rollout_path(run["controller"], building["result_id"], run["label"]))
            time_step = _time_step(data)
            tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            tsp = _setpoint(data)
            n = min(len(time_step), len(tin), len(tsp))
            time_step, tin, tsp = time_step[:n], tin[:n], tsp[:n]
            bias, _ = _bias_and_stress(n)
            ax.plot(time_step, tin, lw=0.9, label="T_in", color="tab:blue")
            ax.plot(time_step, tsp, lw=0.8, ls="--", label="Tsp", color="black")
            ax.plot(time_step, tsp + bias, lw=0.8, ls=":", label="Tsp+bias", color="tab:orange")
            ax.fill_between(
                time_step,
                tsp - WEIGHTS["guard_band"],
                tsp + WEIGHTS["guard_band"],
                alpha=0.12,
                color="tab:green",
                label="+/-2 C guard",
            )
            ax.set_title(run["display"], fontsize=10)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=8)
        ymin = min(ax.get_ylim()[0] for ax in axes)
        ymax = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(max(10.0, ymin), min(32.0, ymax))
            ax.set_xlabel("February time step")
            ax.set_ylabel("Temperature [C]")
        fig.suptitle(f"{building['display']} comfort: former vs robust-ID RC")
        fig.tight_layout()
        path = FIG_DIR / f"comfort_{building['result_id']}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
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
    end_marker = "<!-- END_BAD_BUILDING_IDROBUST_RC_SECTION -->"
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


def append_notebooks(
    kpis: pd.DataFrame,
    deltas: pd.DataFrame,
    fit: pd.DataFrame,
    figures: dict[str, Path | list[Path]],
) -> None:
    marker = f"## Bad-Building Robust RC ID Check - `{IDROBUST_LABEL}`"
    end_marker = "<!-- END_BAD_BUILDING_IDROBUST_RC_SECTION -->"

    kpi_view = kpis[
        [
            "controller",
            "building",
            "electricity_consumption_total",
            "daily_peak_average",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "guard_violation_%",
            "stage_cost_sum",
        ]
    ]
    delta_view = deltas[deltas["metric"].isin(["discomfort_%", "electricity_consumption_total", "daily_peak_average", "stage_cost_sum"])]
    fit_view = fit[
        (fit["dataset"] == "rich_january_id")
        & (fit["model"].isin(["former_january_rc", "robust_rich_id_rc"]))
    ][["building", "model", "one_step_rmse_c", "free_run_rmse_c", "one_step_bias_c", "free_run_bias_c"]]

    overview = f"""{marker}

Selected bad buildings: `{BUILDING_SELECTOR}`. This section compares the former 1R1C January fit against the richer January-ID bounded robust 1R1C fit, then reruns only MPC and GNU-RL/RLMPC under the same time-bias guard cost.

Cost used in both reruns:

`J_k = 0.5 * [3*((Tin-(Tsp+bias))/2)^2 + 60*(cold_guard/1.5)^2 + 15*(hot_guard/2)^2] + 0.5*GridStress*P_heat/5`

`bias = +1 C` in preheat windows and `bias = -1 C` in reduction windows. The hard comfort guard diagnostics are measured against the true setpoint with `Tsp +/- 2 C`.

The robust-ID RC rerun improved some buildings, but it is not a clean universal fix. The clearest comfort gains are in the bad tail for explicit MPC on building 11 and for GNU-RL/RLMPC on buildings 14 and 19. Building 8 stays rough, and GNU-RL/RLMPC produced many qpth inaccurate-solution warnings during the robust-ID online run, so its numbers are useful but numerically less trustworthy than explicit MPC.

KPIs from raw result folders:

{_markdown_table(kpi_view, digits=4)}

ID-robust minus former deltas:

{_markdown_table(delta_view, digits=4)}

1R1C fit metrics on the richer January-ID dataset:

{_markdown_table(fit_view, digits=4)}
"""

    common_cells = [
        nbf.v4.new_markdown_cell(overview),
        nbf.v4.new_markdown_cell(f"![Fit one-step RMSE]({_relative(figures['fit_one_step'])})"),
        nbf.v4.new_markdown_cell(f"![Fit free-run RMSE]({_relative(figures['fit_free_run'])})"),
        nbf.v4.new_markdown_cell(f"![KPI deltas]({_relative(figures['kpi_deltas'])})"),
        nbf.v4.new_markdown_cell(f"![Bad-building mean hourly load]({_relative(figures['load'])})"),
        nbf.v4.new_markdown_cell(f"![Bad-building hourly response]({_relative(figures['hourly'])})"),
        nbf.v4.new_markdown_cell(f"![Bad-building weighted terms]({_relative(figures['terms'])})"),
    ]
    comfort_cells = [
        nbf.v4.new_markdown_cell(f"![{building['display']} comfort]({_relative(path)})")
        for building, path in zip(BUILDINGS, figures["comfort"])
    ]
    footer = nbf.v4.new_markdown_cell(end_marker)

    for notebook in [
        ANALYSIS / "02_mpc_results.ipynb",
        ANALYSIS / "04_rlmpc_results.ipynb",
        ANALYSIS / "08_sac_results.ipynb",
        ANALYSIS / "10_peak_load_profiles.ipynb",
        ANALYSIS / "11_no_occupant_single_building_comparison.ipynb",
        ANALYSIS / "12_simple_v4_comparison.ipynb",
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
    kpis, deltas, terms = build_kpis_and_terms()
    figures: dict[str, Path | list[Path]] = {}
    figures["fit_one_step"], figures["fit_free_run"] = _plot_fit_metrics(fit)
    figures["kpi_deltas"] = _plot_kpi_deltas(deltas)
    figures["load"] = _plot_selected_load_profiles()
    figures["hourly"] = _plot_hourly_response()
    figures["terms"] = _plot_terms(terms)
    figures["comfort"] = _plot_comfort_by_building()
    append_notebooks(kpis, deltas, fit, figures)


if __name__ == "__main__":
    main()
