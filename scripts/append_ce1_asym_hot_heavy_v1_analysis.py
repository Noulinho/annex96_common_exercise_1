#!/usr/bin/env python
"""Append asymmetric hot-heavy MPC diagnostics to the MPC notebook."""

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
from src.rewards import daily_window_shift_profile


LABEL = "ce1_asym_hot_heavy_v1"
DAILY_WINDOW_LABEL = "ce1_daily_windows_w04_04_01_01"
BEST_DISTRICT_LABEL = "ce1_mpc_district_target_c04_d03_du25_u01"
FIG_DIR = Path("figures") / LABEL
ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
BUILDINGS = dataset_building_entries(DATASET_NAME)

WEIGHTS = {
    "label": LABEL,
    "scenario": "no_occupant",
    "controller": "MPC only",
    "comfort_mode": "asymmetric (hot-heavy: hot 20x more expensive than cold per degree)",
    "district_term": "CE1 district target full quadratic",
    "comfort_weight": 1.0,
    "comfort_track_weight": 0.4,
    "comfort_track_scale": 2.0,
    "comfort_cold_weight": 0.1,
    "comfort_cold_tolerance_deg": 0.5,
    "comfort_cold_scale": 1.5,
    "comfort_hot_weight": 2.0,
    "comfort_hot_tolerance_deg": 0.5,
    "comfort_hot_scale": 1.5,
    "district_target_weight": 0.3,
    "district_target_mode": "full_quadratic",
    "delta_u_weight": 25.0,
    "u_weight": 0.1,
    "cost_at_1deg_cold_past_tol": 0.011,
    "cost_at_1deg_hot_past_tol": 0.222,
    "hot_cold_ratio": 20.0,
    "rationale": "hot>>cold lets optimizer coast through peaks; track term maintains setpoint pull",
}

COMPARISON_RUNS = [
    (DAILY_WINDOW_LABEL, "MPC daily-window w04/04/01/01"),
    (BEST_DISTRICT_LABEL, "MPC district-target c04/d03/du25/u01"),
    (LABEL, "MPC asym hot-heavy v1"),
]

WINDOWS = {
    "midnight_preheat_1_3": [1, 2, 3],
    "morning_curtail_6_10": [6, 7, 8, 9, 10],
    "afternoon_preheat_13_17": [13, 14, 15, 16, 17],
    "evening_curtail_20_23": [20, 21, 22, 23],
}


def _rollout_path(label: str, building_id: str) -> Path:
    return RESULTS_DIR / "raw" / "mpc" / building_id / "no_occupant" / label / "rollout.csv"


def _baseline_rollout_path(building_id: str) -> Path:
    return RESULTS_DIR / "raw" / "mpc" / building_id / "no_occupant" / "baseline" / "rollout.csv"


def _metrics_path(label: str, building_id: str) -> Path:
    return RESULTS_DIR / "summaries" / "mpc" / building_id / "no_occupant" / f"{label}_metrics.json"


def _load_rollouts(label: str) -> dict[str, pd.DataFrame]:
    rollouts = {}
    missing = []
    for building in BUILDINGS:
        path = _rollout_path(label, building["result_id"])
        if not path.exists():
            missing.append(path)
            continue
        rollouts[building["result_id"]] = pd.read_csv(path)
    if missing:
        missing_text = "\n".join(str(p) for p in missing[:8])
        if len(missing) > 8:
            missing_text += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing rollouts for {label}:\n{missing_text}")
    return rollouts


def _load_metrics(label: str) -> pd.DataFrame:
    rows = []
    missing = []
    for building in BUILDINGS:
        path = _metrics_path(label, building["result_id"])
        if not path.exists():
            missing.append(path)
            continue
        row = json.loads(path.read_text())
        row.setdefault("building_id", building["result_id"])
        rows.append(row)
    if missing:
        missing_text = "\n".join(str(p) for p in missing[:8])
        if len(missing) > 8:
            missing_text += f"\n... and {len(missing) - 8} more"
        raise FileNotFoundError(f"Missing metrics for {label}:\n{missing_text}")
    return pd.DataFrame(rows)


def _baseline_district() -> pd.DataFrame:
    path = RESULTS_DIR / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    target = pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float),
            "target": target,
            "controller": "CityLearn baseline",
        }
    )


def _district_from_rollouts(display: str, rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    series = []
    for building in BUILDINGS:
        data = rollouts[building["result_id"]]
        series.append(
            pd.to_numeric(data["net_electricity_consumption"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
    n = min(len(s) for s in series)
    load = np.sum(np.stack([s[:n] for s in series], axis=0), axis=0)
    return pd.DataFrame({"time_step": np.arange(n, dtype=int), "load": load, "controller": display})


def _daily_profile(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    daily["hour_label"] = daily["time_step"] % 24 + 1
    return daily.groupby(["controller", "hour_label"], as_index=False)["load"].mean()


def _target_daily_profile(baseline: pd.DataFrame) -> pd.DataFrame:
    target = baseline[["time_step", "target"]].copy()
    target["hour_label"] = target["time_step"] % 24 + 1
    return target.groupby("hour_label", as_index=False)["target"].mean()


def _district_target_kpis(load: np.ndarray, target: np.ndarray) -> dict[str, float]:
    n = min(len(load), len(target))
    load, target = np.asarray(load[:n], dtype=float), np.asarray(target[:n], dtype=float)
    error = load - target
    above = np.maximum(error, 0.0)
    return {
        "district_target_rmse_kw": float(np.sqrt(np.mean(error**2))),
        "district_target_mae_kw": float(np.mean(np.abs(error))),
        "district_above_target_kwh": float(np.sum(above)),
        "district_peak_above_target_kw": float(np.max(above)),
    }


def _kpis(load: np.ndarray, target: np.ndarray | None = None) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    row = {
        "district_total_kwh": float(np.sum(load)),
        "district_mean_kw": float(np.mean(load)),
        "district_peak_kw": float(np.max(load)),
        "district_ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))) if len(load) > 1 else 0.0,
        "district_load_factor": float(np.mean(load) / max(np.max(load), 1e-9)),
    }
    if target is not None:
        row.update(_district_target_kpis(load, np.asarray(target, dtype=float)))
    return row


def _controller_summary(
    display: str, label: str, district: pd.DataFrame, target: np.ndarray
) -> dict[str, float | str]:
    summary = _load_metrics(label)
    row = {
        "controller": display,
        "building_count": int(len(summary)),
        "mean_electricity_kpi": float(pd.to_numeric(summary["electricity_consumption_total"]).mean()),
        "mean_daily_peak_kpi": float(pd.to_numeric(summary["daily_peak_average"]).mean()),
        "mean_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).mean()),
        "max_discomfort_proportion": float(pd.to_numeric(summary["discomfort_proportion"]).max()),
        "mean_discomfort_cold": float(pd.to_numeric(summary["discomfort_cold_proportion"]).mean()),
        "mean_discomfort_hot": float(pd.to_numeric(summary["discomfort_hot_proportion"]).mean()),
        "mean_kelvin_hours": float(pd.to_numeric(summary["kelvin_hours"]).mean()),
        "max_kelvin_hours": float(pd.to_numeric(summary["kelvin_hours"]).max()),
    }
    return {**row, **_kpis(district["load"].to_numpy(dtype=float), target)}


def _window_effects(baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline_daily = _daily_profile(baseline).set_index("hour_label")["load"]
    rows = []
    for frame in controller_frames:
        name = frame["controller"].iloc[0]
        daily = _daily_profile(frame).set_index("hour_label")["load"]
        for window_name, hour_labels in WINDOWS.items():
            delta = daily.loc[hour_labels].mean() - baseline_daily.loc[hour_labels].mean()
            rows.append({"controller": name, "window": window_name, "mean_delta_vs_baseline_kw": float(delta)})
    return pd.DataFrame(rows)


def _scheduled_setpoint(building_id: str, n: int) -> np.ndarray:
    data = pd.read_csv(_baseline_rollout_path(building_id))
    col = "baseline_setpoint" if "baseline_setpoint" in data.columns else "tmin"
    values = pd.to_numeric(data[col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    if len(values) < n:
        values = np.pad(values, (0, n - len(values)), mode="edge")
    return values[:n]


def _action_window_effects(rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        bid = building["result_id"]
        data = rollouts[bid].copy()
        action = pd.to_numeric(data["action"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        baseline = (
            pd.to_numeric(pd.read_csv(_baseline_rollout_path(bid))["action"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)[: len(action)]
        )
        hour_label = data["time_step"].to_numpy(dtype=int) % 24 + 1
        for window_name, hour_labels in WINDOWS.items():
            mask = np.isin(hour_label, hour_labels)
            rows.append(
                {
                    "building_id": bid,
                    "window": window_name,
                    "mean_delta_u_vs_baseline_mpc": float(np.mean(action[mask] - baseline[mask])),
                }
            )
    return pd.DataFrame(rows)


def _term_diagnostics(rollouts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tw = WEIGHTS["comfort_track_weight"]
    ts = WEIGHTS["comfort_track_scale"]
    cw = WEIGHTS["comfort_cold_weight"]
    ct = WEIGHTS["comfort_cold_tolerance_deg"]
    cs = WEIGHTS["comfort_cold_scale"]
    hw = WEIGHTS["comfort_hot_weight"]
    ht = WEIGHTS["comfort_hot_tolerance_deg"]
    hs = WEIGHTS["comfort_hot_scale"]
    dw = WEIGHTS["district_target_weight"]
    du_w = WEIGHTS["delta_u_weight"]
    u_w = WEIGHTS["u_weight"]

    rows = []
    for building in BUILDINGS:
        bid = building["result_id"]
        data = rollouts[bid]
        n = len(data)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _scheduled_setpoint(bid, n)
        action = pd.to_numeric(data["action"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        track_term = tw * ((indoor - setpoint) / ts) ** 2
        cold_viol = np.maximum(setpoint - ct - indoor, 0.0)
        hot_viol = np.maximum(indoor - setpoint - ht, 0.0)
        cold_term = cw * (cold_viol / cs) ** 2
        hot_term = hw * (hot_viol / hs) ** 2
        comfort_term = track_term + cold_term + hot_term

        delta_u = np.diff(action, prepend=action[0])
        ramp_term = du_w * delta_u**2
        energy_term = u_w * action

        dt_term = np.zeros(n)
        if "district_error" in data.columns:
            err = pd.to_numeric(data["district_error"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            dt_term = dw * err**2

        for term, values in {
            "asym_track_w04": track_term,
            "asym_cold_w01": cold_term,
            "asym_hot_w20": hot_term,
            "asym_comfort_total": comfort_term,
            "district_d03": dt_term,
            "ramp_du25": ramp_term,
            "u_0p1": energy_term,
        }.items():
            rows.append(
                {
                    "controller": "MPC asym hot-heavy v1",
                    "building_id": bid,
                    "term": term,
                    "mean_contribution": float(np.mean(values)),
                    "sum_contribution": float(np.sum(values)),
                }
            )
    return pd.DataFrame(rows)


def _make_comfort_grid(rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(20, 12), sharex=True, sharey=False)
    for ax, building in zip(axes.ravel(), BUILDINGS):
        data = rollouts[building["result_id"]]
        time = data["time_step"].to_numpy(dtype=float)
        indoor = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = _scheduled_setpoint(building["result_id"], len(data))
        ax.fill_between(time, 20.0, 24.0, color="0.86", alpha=0.75, label="20-24 band")
        ax.plot(time, indoor, lw=0.9, label="Indoor temperature")
        ax.plot(time, setpoint, lw=0.8, label="Scheduled setpoint")
        ax.set_title(building["result_id"], fontsize=8)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("MPC no-occupant comfort: asymmetric hot-heavy (hot 20× cold, district×0.3)")
    fig.supxlabel("February time step")
    fig.supylabel("Temperature [deg C]")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = FIG_DIR / "mpc_asym_hot_heavy_v1_comfort_all25.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _make_district_plots(
    baseline: pd.DataFrame, controller_frames: list[pd.DataFrame]
) -> tuple[Path, Path]:
    combined = pd.concat([baseline, *controller_frames], ignore_index=True, sort=False)

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.1, label=name)
    ax.plot(baseline["time_step"], baseline["target"], color="0.15", lw=1.4, label="District target")
    ax.set_title(f"No-occupant district load: asymmetric hot-heavy ({LABEL})")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kW]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=160)
    plt.close(fig)

    daily_mean = _daily_profile(combined)
    target_daily = _target_daily_profile(baseline)
    profile = pd.DataFrame(
        {
            "hour_label": np.arange(1, 25),
            "shape_signal": daily_window_shift_profile(simulation_start=0, simulation_end=23),
        }
    )
    preheat = profile[profile["shape_signal"] < 0.0]["hour_label"].to_numpy()
    curtail = profile[profile["shape_signal"] > 0.0]["hour_label"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    for hour in preheat:
        ax.axvspan(hour - 0.45, hour + 0.45, color="#d7ebff", alpha=0.35, linewidth=0)
    for hour in curtail:
        ax.axvspan(hour - 0.45, hour + 0.45, color="#ffe0d2", alpha=0.35, linewidth=0)
    for name, group in daily_mean.groupby("controller", sort=False):
        ax.plot(group["hour_label"], group["load"], marker="o", lw=1.7, label=name)
    ax.plot(
        target_daily["hour_label"], target_daily["target"],
        marker="o", lw=1.7, color="0.15", label="District target",
    )
    ax.set_title(f"Mean daily district profile: asymmetric hot-heavy ({LABEL})")
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


def _markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else format(float(v), floatfmt))
    return out.to_markdown(index=False)


def _append_notebook(path: Path, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(path, as_version=4)
    nb.cells = [
        cell for cell in nb.cells
        if LABEL not in str(cell.get("source", ""))
        and "CE1 Asymmetric Hot-Heavy MPC" not in str(cell.get("source", ""))
    ]
    nb.cells.extend(cells)
    nbf.write(nb, path)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rollouts_by_label = {label: _load_rollouts(label) for label, _ in COMPARISON_RUNS}
    new_rollouts = rollouts_by_label[LABEL]
    baseline = _baseline_district()
    target = baseline["target"].to_numpy(dtype=float)
    controller_district = [
        _district_from_rollouts(display, rollouts_by_label[label])
        for label, display in COMPARISON_RUNS
    ]

    comfort_path = _make_comfort_grid(new_rollouts)
    district_plot, daily_plot = _make_district_plots(baseline, controller_district)

    controller_summary = pd.DataFrame(
        [
            _controller_summary(display, label, frame, target)
            for (label, display), frame in zip(COMPARISON_RUNS, controller_district)
        ]
    )
    baseline_kpis = pd.DataFrame(
        [{"controller": "CityLearn baseline", **_kpis(baseline["load"].to_numpy(dtype=float), target)}]
    )
    district_kpis = pd.concat([baseline_kpis, controller_summary], ignore_index=True, sort=False)
    window_effects = _window_effects(baseline, controller_district)
    action_windows = (
        _action_window_effects(new_rollouts)
        .groupby("window", as_index=False)
        .agg(mean_delta_u_vs_baseline_mpc=("mean_delta_u_vs_baseline_mpc", "mean"))
    )
    term_diagnostics = _term_diagnostics(new_rollouts)
    term_summary = (
        term_diagnostics.groupby(["controller", "term"], as_index=False)
        .agg(mean_of_means=("mean_contribution", "mean"), sum_over_buildings=("sum_contribution", "sum"))
        .sort_values(["controller", "term"])
    )

    controller_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_controller_kpis.csv", index=False)
    district_kpis.to_csv(ANALYSIS_DIR / f"{LABEL}_district_kpis.csv", index=False)
    window_effects.to_csv(ANALYSIS_DIR / f"{LABEL}_window_effects.csv", index=False)
    action_windows.to_csv(ANALYSIS_DIR / f"{LABEL}_action_window_effects.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{LABEL}_term_diagnostics.csv", index=False)

    common_md = """
## CE1 Asymmetric Hot-Heavy MPC No-Occupant Run

Appended run label: `__LABEL__`.

**Hot violations penalized 20× more than cold** — the structural change needed to achieve genuine peak shaving. All prior barrier experiments failed because they protected the cold floor too rigidly; the optimizer had no room to let buildings coast through peak hours. The setpoint-mode best result (`c04/d03/du25/u01`) achieved 230 kW by allowing 3.4% cold violations as a load-shifting mechanism.

This formulation makes that mechanism explicit and one-sided:
- **Track term** (weight=0.4): symmetric quadratic around setpoint — same magnitude as the best result's setpoint mode
- **Cold violation** (weight=0.1): cheap extra penalty below T_sp−0.5°C — optimizer can undershoot during peaks
- **Hot violation** (weight=2.0): expensive penalty above T_sp+0.5°C — no wasted overheating, no district load for free

At 1°C past tolerance: cold costs 0.011, hot costs 0.222 — 20× asymmetry. District (weight=0.3, full quadratic) is now competitive with the comfort track term (~0.10). The optimizer will trade mild cold dips for district savings.

Active terms:

```json
__WEIGHTS_JSON__
```

District and controller KPI summary:

__DISTRICT_TABLE__

Mean district-load change versus CityLearn baseline by target window:

__WINDOW_TABLE__

Mean MPC action change versus baseline MPC in the target windows:

__ACTION_WINDOW_TABLE__

Weighted term diagnostics (post-hoc from rollout):

__TERM_TABLE__
"""
    common_md = (
        common_md.replace("__LABEL__", LABEL)
        .replace("__WEIGHTS_JSON__", json.dumps(WEIGHTS, indent=2))
        .replace("__DISTRICT_TABLE__", _markdown_table(district_kpis))
        .replace("__WINDOW_TABLE__", _markdown_table(window_effects))
        .replace("__ACTION_WINDOW_TABLE__", _markdown_table(action_windows))
        .replace("__TERM_TABLE__", _markdown_table(term_summary))
    )

    cells = [
        nbf.v4.new_markdown_cell(common_md),
        nbf.v4.new_markdown_cell(f"![Comfort all buildings](../{comfort_path.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![District load timeseries](../{district_plot.as_posix()})"),
        nbf.v4.new_markdown_cell(f"![Mean daily district profile](../{daily_plot.as_posix()})"),
    ]
    _append_notebook(ANALYSIS_DIR / "02_mpc_results.ipynb", cells)

    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote KPI CSVs to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
