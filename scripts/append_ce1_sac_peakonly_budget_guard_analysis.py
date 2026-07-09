#!/usr/bin/env python
"""Append SAC peak-only district + heat-budget guard results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_target import load_ce1_district_target_profiles


LABEL = "ce1_sac_peakonly_budget_guard_c045_l020_b020_pi010_du005"
PREVIOUS_SAC_LABEL = "ce1_mpc_district_target_c04_d03_du25_u01"
BASE_GUARD_LABEL = "ce1_sac_peakonly_budget_guard_c045_l020_b020_pi010_du005"
SECTION_TITLE = "CE1 SAC Peak-Only Budget-Guard Run"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

COMPARISON_RUNS = [
    ("mpc", PREVIOUS_SAC_LABEL, "MPC district-target c04/d03/du25/u01"),
    ("rlmpc", PREVIOUS_SAC_LABEL, "GNU-RL/RLMPC district-target c04/d03/du25/u01"),
    ("sac_merlin", PREVIOUS_SAC_LABEL, "SAC district-target c04/d03/du25/u01"),
    ("sac_merlin", LABEL, "SAC peak-only budget guard"),
]

WINDOWS = {
    "midnight_preheat_1_3": [1, 2, 3],
    "morning_peak_6_10": [6, 7, 8, 9, 10],
    "afternoon_valley_13_17": [13, 14, 15, 16, 17],
    "evening_peak_20_23": [20, 21, 22, 23],
}

WEIGHTS = {
    "label": LABEL,
    "scenario": "no_occupant",
    "controller": "SAC peak-only budget guard",
    "comfort_mode": "asymmetric cold barrier around the single heating setpoint",
    "comfort_weight": 0.45,
    "comfort_scale": 1.0,
    "comfort_track_weight": 0.0,
    "comfort_cold_weight": 12.0,
    "comfort_hot_weight": 2.0,
    "comfort_cold_tolerance": 0.2,
    "comfort_hot_tolerance": 1.0,
    "comfort_track_scale": 2.0,
    "comfort_cold_scale": 1.0,
    "comfort_hot_scale": 1.5,
    "district_target_weight": 0.20,
    "district_target_mode": "peak_only",
    "district_target_scale_mode": "target_max",
    "heat_budget_weight": 0.20,
    "heat_budget_window": 24,
    "heat_budget_tolerance": 0.08,
    "heat_budget_temperature_gate_margin": None,
    "pi_prior_weight": 0.10,
    "pi_prior_temperature_gate_margin": None,
    "ramp_weight": 0.05,
    "energy_weight": 0.0,
    "peak_weight": 0.0,
    "shift_weight": 0.0,
    "preheat_weight": 0.0,
    "power_source": "heating",
    "power_scale_mode": "nominal_heating",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=LABEL)
    parser.add_argument("--section-title", default=SECTION_TITLE)
    parser.add_argument("--controller-display", default="SAC peak-only budget guard")
    parser.add_argument("--include-base-guard", action="store_true")
    parser.add_argument("--comfort-track-weight", type=float, default=WEIGHTS["comfort_track_weight"])
    parser.add_argument("--comfort-cold-weight", type=float, default=WEIGHTS["comfort_cold_weight"])
    parser.add_argument("--comfort-hot-weight", type=float, default=WEIGHTS["comfort_hot_weight"])
    parser.add_argument("--comfort-cold-tolerance", type=float, default=WEIGHTS["comfort_cold_tolerance"])
    parser.add_argument("--comfort-hot-tolerance", type=float, default=WEIGHTS["comfort_hot_tolerance"])
    parser.add_argument("--comfort-track-scale", type=float, default=WEIGHTS["comfort_track_scale"])
    parser.add_argument("--comfort-cold-scale", type=float, default=WEIGHTS["comfort_cold_scale"])
    parser.add_argument("--comfort-hot-scale", type=float, default=WEIGHTS["comfort_hot_scale"])
    parser.add_argument("--district-target-weight", type=float, default=WEIGHTS["district_target_weight"])
    parser.add_argument("--heat-budget-weight", type=float, default=WEIGHTS["heat_budget_weight"])
    parser.add_argument(
        "--heat-budget-temperature-gate-margin",
        type=float,
        default=WEIGHTS["heat_budget_temperature_gate_margin"],
    )
    parser.add_argument("--pi-prior-weight", type=float, default=WEIGHTS["pi_prior_weight"])
    parser.add_argument(
        "--pi-prior-temperature-gate-margin",
        type=float,
        default=WEIGHTS["pi_prior_temperature_gate_margin"],
    )
    parser.add_argument("--ramp-weight", type=float, default=WEIGHTS["ramp_weight"])
    return parser.parse_args()


def _configure(args: argparse.Namespace) -> None:
    global LABEL, SECTION_TITLE, FIG_DIR, COMPARISON_RUNS, WEIGHTS
    LABEL = str(args.label)
    SECTION_TITLE = str(args.section_title)
    FIG_DIR = Path("figures") / LABEL
    WEIGHTS = {
        **WEIGHTS,
        "label": LABEL,
        "controller": str(args.controller_display),
        "comfort_track_weight": float(args.comfort_track_weight),
        "comfort_cold_weight": float(args.comfort_cold_weight),
        "comfort_hot_weight": float(args.comfort_hot_weight),
        "comfort_cold_tolerance": float(args.comfort_cold_tolerance),
        "comfort_hot_tolerance": float(args.comfort_hot_tolerance),
        "comfort_track_scale": float(args.comfort_track_scale),
        "comfort_cold_scale": float(args.comfort_cold_scale),
        "comfort_hot_scale": float(args.comfort_hot_scale),
        "district_target_weight": float(args.district_target_weight),
        "heat_budget_weight": float(args.heat_budget_weight),
        "heat_budget_temperature_gate_margin": (
            None
            if args.heat_budget_temperature_gate_margin is None
            else float(args.heat_budget_temperature_gate_margin)
        ),
        "pi_prior_weight": float(args.pi_prior_weight),
        "pi_prior_temperature_gate_margin": (
            None
            if args.pi_prior_temperature_gate_margin is None
            else float(args.pi_prior_temperature_gate_margin)
        ),
        "ramp_weight": float(args.ramp_weight),
    }
    COMPARISON_RUNS = [
        ("mpc", PREVIOUS_SAC_LABEL, "MPC district-target c04/d03/du25/u01"),
        ("rlmpc", PREVIOUS_SAC_LABEL, "GNU-RL/RLMPC district-target c04/d03/du25/u01"),
        ("sac_merlin", PREVIOUS_SAC_LABEL, "SAC district-target c04/d03/du25/u01"),
    ]
    if args.include_base_guard and LABEL != BASE_GUARD_LABEL:
        COMPARISON_RUNS.append(("sac_merlin", BASE_GUARD_LABEL, "SAC peak-only budget guard"))
    COMPARISON_RUNS.append(("sac_merlin", LABEL, str(args.controller_display)))


def _rollout_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS_DIR / "raw" / controller / building_id / "no_occupant" / label / "rollout.csv"


def _metrics_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS_DIR / "summaries" / controller / building_id / "no_occupant" / f"{label}_metrics.json"


def rebuild_sac_no_occupant_summary() -> Path:
    rows = []
    for building in BUILDINGS:
        summary_dir = RESULTS_DIR / "summaries" / "sac_merlin" / building["result_id"] / "no_occupant"
        for path in sorted(summary_dir.glob("*_metrics.json")):
            if "smoke" in path.name:
                continue
            row = json.loads(path.read_text())
            row.setdefault("building_id", building["result_id"])
            row.setdefault("building", building["display"])
            row.setdefault("building_index", building["index"])
            rows.append(row)
    out = RESULTS_DIR / "summaries" / "sac_merlin" / "no_occupant_summary_all_buildings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _baseline_district() -> pd.DataFrame:
    data = pd.read_csv(RESULTS_DIR / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv")
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0),
            "controller": "CityLearn baseline",
        }
    )


def _load_rollouts(controller: str, label: str) -> dict[str, pd.DataFrame]:
    rollouts = {}
    missing = []
    for building in BUILDINGS:
        path = _rollout_path(controller, building["result_id"], label)
        if not path.exists():
            missing.append(path)
            continue
        rollouts[building["result_id"]] = pd.read_csv(path)
    if missing:
        lines = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            lines += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(lines)
    return rollouts


def _load_metrics(controller: str, label: str) -> pd.DataFrame:
    rows = []
    missing = []
    for building in BUILDINGS:
        path = _metrics_path(controller, building["result_id"], label)
        if not path.exists():
            missing.append(path)
            continue
        row = json.loads(path.read_text())
        row.setdefault("building_id", building["result_id"])
        rows.append(row)
    if missing:
        lines = "\n".join(str(path) for path in missing[:8])
        if len(missing) > 8:
            lines += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(lines)
    return pd.DataFrame(rows)


def _district_from_rollouts(rollouts: dict[str, pd.DataFrame], label: str) -> pd.DataFrame:
    series = []
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        series.append(pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy())
    n = min(len(s) for s in series)
    return pd.DataFrame(
        {
            "time_step": np.arange(n, dtype=int),
            "load": np.sum(np.stack([s[:n] for s in series], axis=0), axis=0),
            "controller": label,
        }
    )


def _kpis(load: np.ndarray, target: np.ndarray | None = None) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    row = {
        "district_total_kwh": float(np.sum(load)),
        "district_mean_kw": float(np.mean(load)),
        "district_peak_kw": float(np.max(load)),
        "district_ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))) if len(load) > 1 else 0.0,
        "district_load_factor": float(np.mean(load) / max(float(np.max(load)), 1e-9)),
    }
    if target is not None:
        n = min(len(load), len(target))
        error = load[:n] - np.asarray(target[:n], dtype=float)
        above = np.maximum(error, 0.0)
        row.update(
            {
                "district_target_rmse_kw": float(np.sqrt(np.mean(error**2))),
                "district_target_mae_kw": float(np.mean(np.abs(error))),
                "district_above_target_kwh": float(np.sum(above)),
                "district_peak_above_target_kw": float(np.max(above)),
            }
        )
    return row


def _controller_summary(controller: str, label: str, display: str, district: pd.DataFrame, target: np.ndarray) -> dict[str, float | str]:
    summary = _load_metrics(controller, label)
    row = {
        "controller": display,
        "building_count": int(len(summary)),
        "mean_electricity_kpi": float(pd.to_numeric(summary["electricity_consumption_total"]).mean()),
        "mean_daily_peak_kpi": float(pd.to_numeric(summary["daily_peak_average"]).mean()),
        "mean_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).mean()),
        "max_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).max()),
    }
    if "kelvin_hours" in summary.columns:
        row["mean_kelvin_hours"] = float(pd.to_numeric(summary["kelvin_hours"]).mean())
        row["max_kelvin_hours"] = float(pd.to_numeric(summary["kelvin_hours"]).max())
    return {**row, **_kpis(district["load"].to_numpy(dtype=float), target)}


def _daily_profile(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["hour_label"] = data["time_step"].to_numpy(dtype=int) % 24 + 1
    return data.groupby(["controller", "hour_label"], as_index=False)["load"].mean()


def _window_effects(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline_daily = _daily_profile(baseline).set_index("hour_label")["load"]
    rows = []
    for frame in controller_frames:
        name = frame["controller"].iloc[0]
        daily = _daily_profile(frame).set_index("hour_label")["load"]
        for window_name, hour_labels in WINDOWS.items():
            rows.append(
                {
                    "controller": name,
                    "window": window_name,
                    "mean_delta_vs_baseline_kw": float(daily.loc[hour_labels].mean() - baseline_daily.loc[hour_labels].mean()),
                }
            )
    return pd.DataFrame(rows)


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "baseline_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def _make_comfort_grid(rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=False)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy(dtype=float)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _setpoint(data)
        ax.fill_between(time, 20.0, 24.0, color="0.86", alpha=0.75, label="20-24 band")
        ax.plot(time, indoor, lw=0.9, linestyle="-", label="Indoor temperature")
        ax.plot(time, setpoint, lw=0.8, linestyle="-", label="Scheduled setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("SAC peak-only budget guard: no-occupant comfort")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "sac_peakonly_budget_guard_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True, sort=False)
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, linestyle="-", label=name)
    ax.plot(baseline["time_step"], baseline["target"], color="0.15", lw=1.4, linestyle="-", label="District target")
    ax.set_title(f"No-occupant district load ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily = _daily_profile(combined)
    target_daily = baseline.assign(hour_label=baseline["time_step"].to_numpy(dtype=int) % 24 + 1).groupby("hour_label", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour_label"], group["load"], marker="o", lw=1.7, linestyle="-", label=name)
    ax.plot(target_daily["hour_label"], target_daily["target"], marker="o", lw=1.7, linestyle="-", color="0.15", label="District target")
    ax.set_title(f"Mean daily district profile ({LABEL})")
    ax.set_xlabel("Hour label")
    ax.set_ylabel("Mean district electricity [kW]")
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile.png"
    fig.savefig(daily_path, dpi=160)
    plt.close(fig)
    return load_path, daily_path


def _heating_budget_summary(sac_runs: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    baseline_total = 0.0
    totals = {name: 0.0 for name in sac_runs}
    for building in BUILDINGS:
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode="target_max",
        )
        baseline_total += float(np.sum(profiles.baseline_controlled_heating))
        for name, rollouts in sac_runs.items():
            totals[name] += float(
                pd.to_numeric(
                    rollouts[building["result_id"]]["heating_electricity_consumption"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )
    rows = [{"case": "CityLearn baseline heating", "heating_kwh": baseline_total, "ratio_vs_baseline": 1.0}]
    for name, total in totals.items():
        rows.append({"case": name, "heating_kwh": total, "ratio_vs_baseline": total / max(baseline_total, 1e-9)})
    return pd.DataFrame(rows)


def _term_diagnostics(rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        data = rollouts[building_id]
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode=WEIGHTS["district_target_scale_mode"],
        )
        n = len(data)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").ffill().bfill().to_numpy(dtype=float)[:n]
        setpoint = _setpoint(data)[:n]
        heating = (
            pd.to_numeric(data["heating_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)[:n]
        )
        power_scale = max(float(profiles.heating_nominal_power), 1e-6)
        heating_norm = np.clip(heating / power_scale, 0.0, None)
        ramp = np.diff(np.r_[heating_norm[0], heating_norm])

        error = indoor - setpoint
        cold = np.maximum(setpoint - WEIGHTS["comfort_cold_tolerance"] - indoor, 0.0)
        hot = np.maximum(indoor - setpoint - WEIGHTS["comfort_hot_tolerance"], 0.0)
        comfort_track = (
            WEIGHTS["comfort_weight"]
            * WEIGHTS["comfort_track_weight"]
            * (error / WEIGHTS["comfort_track_scale"]) ** 2
            / WEIGHTS["comfort_scale"]
        )
        comfort_cold = (
            WEIGHTS["comfort_weight"]
            * WEIGHTS["comfort_cold_weight"]
            * (cold / WEIGHTS["comfort_cold_scale"]) ** 2
            / WEIGHTS["comfort_scale"]
        )
        comfort_hot = (
            WEIGHTS["comfort_weight"]
            * WEIGHTS["comfort_hot_weight"]
            * (hot / WEIGHTS["comfort_hot_scale"]) ** 2
            / WEIGHTS["comfort_scale"]
        )
        comfort_total = comfort_track + comfort_cold + comfort_hot

        target = np.asarray(profiles.target, dtype=float)[:n]
        background = np.asarray(profiles.background_without_controlled_heating, dtype=float)[:n]
        district_error = np.maximum(background + heating - target, 0.0) / max(float(profiles.target_scale), 1e-6)
        district_target = WEIGHTS["district_target_weight"] * district_error**2

        baseline_heating = np.asarray(profiles.baseline_controlled_heating, dtype=float)[:n]
        baseline_norm = np.clip(baseline_heating / power_scale, 0.0, None)
        heat_budget = np.zeros(n, dtype=float)
        ref_history: list[float] = []
        actual_history: list[float] = []
        for i in range(n):
            ref_history.append(float(baseline_norm[i]))
            actual_history.append(float(heating_norm[i]))
            if len(ref_history) > int(WEIGHTS["heat_budget_window"]):
                ref_history = ref_history[-int(WEIGHTS["heat_budget_window"]) :]
                actual_history = actual_history[-int(WEIGHTS["heat_budget_window"]) :]
            reference_sum = float(np.sum(ref_history))
            actual_sum = float(np.sum(actual_history))
            if reference_sum > 1e-9:
                margin = WEIGHTS.get("heat_budget_temperature_gate_margin")
                gate_active = margin is None or indoor[i] <= setpoint[i] - float(margin)
                if gate_active:
                    allowed_deficit = float(WEIGHTS["heat_budget_tolerance"]) * reference_sum
                    deficit = max(0.0, reference_sum - actual_sum - allowed_deficit)
                    heat_budget[i] = WEIGHTS["heat_budget_weight"] * (
                        deficit / max(allowed_deficit, 1e-6)
                    ) ** 2

        stress = np.asarray(profiles.district_stress_profile(), dtype=float)[:n]
        below_or_near_peak = np.maximum(0.0, np.minimum(1.0, stress))
        low_stress = 1.0 - below_or_near_peak
        above_reference = np.maximum(heating_norm - baseline_norm, 0.0)
        margin = WEIGHTS.get("pi_prior_temperature_gate_margin")
        if margin is None:
            pi_prior_cost = low_stress * (heating_norm - baseline_norm) ** 2
            pi_prior_cost += below_or_near_peak * above_reference**2
        else:
            hot_zone = indoor > setpoint + float(margin)
            normal_cost = low_stress * (heating_norm - baseline_norm) ** 2
            normal_cost += below_or_near_peak * above_reference**2
            pi_prior_cost = np.where(hot_zone, above_reference**2, normal_cost)
        pi_prior = WEIGHTS["pi_prior_weight"] * pi_prior_cost
        ramp_term = WEIGHTS["ramp_weight"] * ramp**2
        total = comfort_total + district_target + heat_budget + pi_prior + ramp_term

        terms = {
            "comfort_track_weighted": comfort_track,
            "comfort_cold_weighted": comfort_cold,
            "comfort_hot_weighted": comfort_hot,
            "comfort_total_weighted": comfort_total,
            "district_peak_only_weighted": district_target,
            "heat_budget_weighted": heat_budget,
            "pi_prior_weighted": pi_prior,
            "ramp_weighted": ramp_term,
            "total_weighted_cost": total,
        }
        for term, values in terms.items():
            rows.append(
                {
                    "controller": WEIGHTS["controller"],
                    "building_id": building_id,
                    "term": term,
                    "mean_contribution": float(np.mean(values)),
                    "max_contribution": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else format(float(value), floatfmt))
    return out.to_markdown(index=False)


def _append_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    nb.cells = [cell for cell in nb.cells if SECTION_TITLE not in str(cell.get("source", ""))]
    nb.cells.extend(cells)
    nbf.write(nb, path)


def main() -> None:
    _configure(_parse_args())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined_summary = rebuild_sac_no_occupant_summary()

    baseline = _baseline_district()
    target = baseline["target"].to_numpy(dtype=float)
    rollout_map = {(controller, label): _load_rollouts(controller, label) for controller, label, _ in COMPARISON_RUNS}
    controller_frames = [
        _district_from_rollouts(rollout_map[(controller, label)], display)
        for controller, label, display in COMPARISON_RUNS
    ]

    baseline_kpis = pd.DataFrame(
        [{"controller": "CityLearn baseline", **_kpis(baseline["load"].to_numpy(dtype=float), target)}]
    )
    controller_kpis = pd.DataFrame(
        [
            _controller_summary(controller, label, display, frame, target)
            for (controller, label, display), frame in zip(COMPARISON_RUNS, controller_frames)
        ]
    )
    district_kpis = pd.concat([baseline_kpis, controller_kpis], ignore_index=True, sort=False)
    window_effects = _window_effects(baseline, controller_frames)
    heating_budget = _heating_budget_summary(
        {
            display: rollout_map[(controller, label)]
            for controller, label, display in COMPARISON_RUNS
            if controller == "sac_merlin"
        }
    )
    term_diagnostics = _term_diagnostics(rollout_map[("sac_merlin", LABEL)])
    term_summary = (
        term_diagnostics.groupby(["controller", "term"], as_index=False)
        .agg(
            mean_of_means=("mean_contribution", "mean"),
            max_over_buildings=("max_contribution", "max"),
            mean_of_maxes=("max_contribution", "mean"),
        )
        .sort_values(["controller", "term"])
    )

    comfort_path = _make_comfort_grid(rollout_map[("sac_merlin", LABEL)])
    district_path, daily_path = _make_district_plots(baseline, controller_frames)

    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_district_kpis.csv", index=False)
    window_effects.to_csv(ANALYSIS_DIR / f"{LABEL}_window_effects.csv", index=False)
    heating_budget.to_csv(ANALYSIS_DIR / f"{LABEL}_heating_budget.csv", index=False)
    term_diagnostics.to_csv(ANALYSIS_DIR / f"{LABEL}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_term_diagnostics.csv", index=False)

    markdown = f"""
## {SECTION_TITLE}

Appended run label: `{LABEL}`.

This is a SAC-only no-occupant run. No MPC/RLMPC controller was rerun. The new SAC reward tries to prevent the previous underheating failure by using:

- peak-only CE1 district target pressure, so being below the target is not rewarded;
- a rolling 24-hour heat-budget floor against the baseline/PI heating profile;
- a PI/baseline heating prior away from district stress;
- a cold-biased asymmetric comfort barrier;
- no direct energy-minimization term.

Active terms:

```json
{json.dumps(WEIGHTS, indent=2)}
```

District and controller KPI summary:

{_markdown_table(district_kpis)}

Mean district-load change versus CityLearn baseline by target window:

{_markdown_table(window_effects)}

Heating budget check:

{_markdown_table(heating_budget)}

Weighted reward-term diagnostics for the new SAC run:

{_markdown_table(term_summary)}

Rebuilt SAC all-building summary:

`{combined_summary}`
"""
    cells = [
        nbf.v4.new_markdown_cell(markdown),
        nbf.v4.new_markdown_cell(f"![SAC peak-only budget guard comfort all buildings](../{comfort_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![District load](../{district_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_path.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "08_sac_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote KPI CSVs to {ANALYSIS_DIR}")
    print(f"Rebuilt {combined_summary}")


if __name__ == "__main__":
    main()
