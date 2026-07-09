#!/usr/bin/env python
"""Append ce1_lagrangian_v1 SAC (Lagrangian-constrained district reward) results to 08_sac_results.ipynb."""

from __future__ import annotations

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


LABEL = "ce1_lagrangian_v1"
SECTION_TITLE = "CE1 SAC Lagrangian-Constrained District Run v1"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

# Baseline comparison: best MPC and previous SAC attempts
COMPARISON_RUNS = [
    ("mpc",      "ce1_mpc_district_target_c04_d03_du25_u01", "MPC best (c04/d03/du25/u01)"),
    ("rlmpc",    "ce1_mpc_district_target_c04_d03_du25_u01", "GNU-RL/RLMPC best"),
    ("sac_merlin", "ce1_sac_peakonly_budget_guard_c045_l020_b020_pi010_du005", "SAC peak-only budget guard"),
    ("sac_merlin", LABEL, "SAC Lagrangian v1"),
]

WINDOWS = {
    "midnight_preheat_1_3":  [1, 2, 3],
    "morning_peak_6_10":     [6, 7, 8, 9, 10],
    "afternoon_valley_13_17":[13, 14, 15, 16, 17],
    "evening_peak_20_23":    [20, 21, 22, 23],
}

WEIGHTS = {
    "label": LABEL,
    "scenario": "no_occupant",
    "controller": "SAC Lagrangian v1",
    "reward_type": "LagrangianDistrictReward",
    "lagrange_init": 1.0,
    "lagrange_lr": 0.05,
    "lagrange_update_period": 96,
    "comfort_budget_degC": 0.05,
    "lagrange_max": 100.0,
    "comfort_cold_tolerance": 0.5,
    "comfort_hot_tolerance": 1.0,
    "comfort_hot_weight": 5.0,
    "comfort_hot_scale": 2.0,
    "district_target_weight": 0.4,
    "district_target_mode": "tracking",
    "district_target_scale_mode": "target_max",
    "energy_weight": 0.05,
    "ramp_weight": 5.0,
    "peak_weight": 0.0,
    "power_source": "heating",
    "power_scale_mode": "nominal_heating",
    "training_episodes": 10,
}


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
        lines = "\n".join(str(p) for p in missing[:8])
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
        rows.append({**json.loads(path.read_text()), "building_id": building["result_id"]})
    if missing:
        lines = "\n".join(str(p) for p in missing[:8])
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
    row: dict[str, float] = {
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


def _controller_summary(controller: str, label: str, display: str, district: pd.DataFrame, target: np.ndarray) -> dict:
    summary = _load_metrics(controller, label)
    row: dict = {
        "controller": display,
        "building_count": int(len(summary)),
        "mean_electricity_kpi": float(pd.to_numeric(summary["electricity_consumption_total"]).mean()),
        "mean_daily_peak_kpi": float(pd.to_numeric(summary["daily_peak_average"]).mean()),
        "mean_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).mean()),
        "max_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).max()),
        "mean_cold_discomfort": float(pd.to_numeric(summary["discomfort_cold_proportion"]).mean()),
        "mean_hot_discomfort": float(pd.to_numeric(summary["discomfort_hot_proportion"]).mean()),
    }
    if "kelvin_hours" in summary.columns:
        row["mean_kelvin_hours"] = float(pd.to_numeric(summary["kelvin_hours"]).mean())
        row["max_kelvin_hours"] = float(pd.to_numeric(summary["kelvin_hours"]).max())
    # Lambda trajectory info for the Lagrangian run
    if controller == "sac_merlin" and label == LABEL:
        final_lambdas = []
        for building in BUILDINGS:
            mpath = _metrics_path(controller, building["result_id"], label)
            if mpath.exists():
                m = json.loads(mpath.read_text())
                if "lagrange_final_lambda" in m:
                    final_lambdas.append(float(m["lagrange_final_lambda"]))
        if final_lambdas:
            row["mean_final_lambda"] = float(np.mean(final_lambdas))
            row["min_final_lambda"] = float(np.min(final_lambdas))
            row["max_final_lambda"] = float(np.max(final_lambdas))
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
    for col in ("effective_setpoint", "baseline_setpoint", "tmin"):
        if col in data.columns:
            return pd.to_numeric(data[col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def _make_comfort_grid(rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=False)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy(dtype=float)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _setpoint(data)
        ax.fill_between(time, 20.0, 24.0, color="0.86", alpha=0.75, label="20-24 band")
        ax.plot(time, indoor, lw=0.9, label="Indoor temperature")
        ax.plot(time, setpoint, lw=0.8, label="Scheduled setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("SAC Lagrangian v1: no-occupant comfort — all 25 buildings")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "sac_merlin_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True, sort=False)

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, label=name)
    ax.plot(baseline["time_step"], baseline["target"], color="0.15", lw=1.4, linestyle="--", label="District target")
    ax.set_title(f"No-occupant district load — {LABEL}")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily = _daily_profile(combined)
    target_daily = (
        baseline.assign(hour_label=baseline["time_step"].to_numpy(dtype=int) % 24 + 1)
        .groupby("hour_label", as_index=False)["target"].mean()
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour_label"], group["load"], marker="o", lw=1.7, label=name)
    ax.plot(target_daily["hour_label"], target_daily["target"], marker="o", lw=1.7,
            color="0.15", linestyle="--", label="District target")
    ax.set_title(f"Mean daily district profile — {LABEL}")
    ax.set_xlabel("Hour label (1–24)")
    ax.set_ylabel("Mean district electricity [kW]")
    ax.set_xticks(range(1, 25))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_daily_profile.png"
    fig.savefig(daily_path, dpi=160)
    plt.close(fig)
    return load_path, daily_path


def _make_lambda_plot(rollouts: dict[str, pd.DataFrame]) -> Path | None:
    """Plot λ trajectory for each building (requires lagrange_trajectory in metrics JSON)."""
    trajectories = {}
    for building in BUILDINGS:
        mpath = _metrics_path("sac_merlin", building["result_id"], LABEL)
        if not mpath.exists():
            return None
        m = json.loads(mpath.read_text())
        traj = m.get("lagrange_trajectory")
        if not traj:
            return None
        steps, lambdas, _ = zip(*traj)
        trajectories[building["result_id"]] = (list(steps), list(lambdas))

    fig, ax = plt.subplots(figsize=(13, 5))
    for building_id, (steps, lambdas) in trajectories.items():
        ax.plot(steps, lambdas, lw=0.8, alpha=0.6)
    ax.set_title("Lagrange multiplier λ trajectory per building — SAC Lagrangian v1")
    ax.set_xlabel("SAC training step")
    ax.set_ylabel("λ")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = FIG_DIR / "lambda_trajectory.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _heating_budget_summary(rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    baseline_total = 0.0
    new_total = 0.0
    for building in BUILDINGS:
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode="target_max",
        )
        baseline_total += float(np.sum(profiles.baseline_controlled_heating))
        new_total += float(
            pd.to_numeric(rollouts[building["result_id"]]["heating_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .sum()
        )
    return pd.DataFrame([
        {"case": "CityLearn baseline heating", "heating_kwh": baseline_total, "ratio_vs_baseline": 1.0},
        {"case": LABEL, "heating_kwh": new_total, "ratio_vs_baseline": new_total / max(baseline_total, 1e-9)},
    ])


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else format(float(v), floatfmt))
    return out.to_markdown(index=False)


def _append_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if SECTION_TITLE not in str(c.get("source", ""))]
    nb.cells.extend(cells)
    nbf.write(nb, path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined_summary = rebuild_sac_no_occupant_summary()

    baseline = _baseline_district()
    target = baseline["target"].to_numpy(dtype=float)

    rollout_map = {(ctrl, lbl): _load_rollouts(ctrl, lbl) for ctrl, lbl, _ in COMPARISON_RUNS}
    controller_frames = [
        _district_from_rollouts(rollout_map[(ctrl, lbl)], display)
        for ctrl, lbl, display in COMPARISON_RUNS
    ]

    baseline_kpis = pd.DataFrame(
        [{"controller": "CityLearn baseline", **_kpis(baseline["load"].to_numpy(dtype=float), target)}]
    )
    controller_kpis = pd.DataFrame(
        [
            _controller_summary(ctrl, lbl, display, frame, target)
            for (ctrl, lbl, display), frame in zip(COMPARISON_RUNS, controller_frames)
        ]
    )
    district_kpis = pd.concat([baseline_kpis, controller_kpis], ignore_index=True, sort=False)
    window_effects = _window_effects(baseline, controller_frames)
    heating_budget = _heating_budget_summary(rollout_map[("sac_merlin", LABEL)])

    comfort_path = _make_comfort_grid(rollout_map[("sac_merlin", LABEL)])
    load_path, daily_path = _make_district_plots(baseline, controller_frames)
    lambda_path = _make_lambda_plot(rollout_map[("sac_merlin", LABEL)])

    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_district_kpis.csv", index=False)
    window_effects.to_csv(ANALYSIS_DIR / f"{LABEL}_window_effects.csv", index=False)
    heating_budget.to_csv(ANALYSIS_DIR / f"{LABEL}_heating_budget.csv", index=False)

    lambda_line = (
        f"\n\n![Lambda trajectory](../{lambda_path.as_posix()})" if lambda_path else ""
    )

    markdown = f"""
## {SECTION_TITLE}

Appended run label: `{LABEL}`.

**Key idea:** Comfort becomes a hard constraint enforced by an adaptive Lagrange multiplier λ; district target tracking is the sole objective. This eliminates the linear comfort-district tradeoff that SAC previously exploited by deleting heat uniformly.

Reward per step:

$$
r_t = -\\left(w_d \\left(\\frac{{\\hat{{P}}^{{district}}_t - P^{{target}}_t}}{{S_L}}\\right)^2
+ w_e \\frac{{P^{{heat}}_t}}{{P^{{max}}}}
+ w_r \\left(\\frac{{P^{{heat}}_t - P^{{heat}}_{{t-1}}}}{{P^{{max}}}}\\right)^2
+ w_{{hot}} \\left(\\frac{{\\max(0, T_t - T^{{sp}} - \\Delta_{{hot}})}}{{s_{{hot}}}}\\right)^2
\\right)
- \\lambda_t \\cdot \\max(0,\\, T^{{sp}} - \\Delta_{{cold}} - T_t)
$$

Dual update every {WEIGHTS['lagrange_update_period']} SAC-phase steps:

$$
\\lambda \\leftarrow \\text{{clip}}\\!\\left(\\lambda + \\eta_\\lambda \\left(\\overline{{v}}_{{cold}} - \\varepsilon\\right),\\; \\lambda_{{min}},\\; \\lambda_{{max}}\\right)
$$

where $\\overline{{v}}_{{cold}}$ is the mean raw cold violation in °C over the update window and $\\varepsilon={WEIGHTS['comfort_budget_degC']}$ °C is the acceptable budget.

Configuration:

```json
{json.dumps(WEIGHTS, indent=2)}
```

District and controller KPI summary:

{_markdown_table(district_kpis)}

Mean district-load change vs CityLearn baseline by target window:

{_markdown_table(window_effects)}

Heating budget check (sum over all buildings, February):

{_markdown_table(heating_budget)}

Rebuilt SAC all-building summary: `{combined_summary}`
"""

    cells: list[nbf.NotebookNode] = [nbf.v4.new_markdown_cell(markdown)]
    cells.append(nbf.v4.new_markdown_cell(f"![SAC Lagrangian v1 comfort all buildings](../{comfort_path.as_posix()})"))
    cells.append(nbf.v4.new_markdown_cell(f"![District load timeseries](../{load_path.as_posix()})"))
    cells.append(nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_path.as_posix()})"))
    if lambda_path:
        cells.append(nbf.v4.new_markdown_cell(f"![Lambda trajectory](../{lambda_path.as_posix()})"))

    _append_notebook(ANALYSIS_DIR / "08_sac_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote CSVs to {ANALYSIS_DIR}")
    print(f"Rebuilt {combined_summary}")
    print(f"Appended section '{SECTION_TITLE}' to 08_sac_results.ipynb")


if __name__ == "__main__":
    main()
