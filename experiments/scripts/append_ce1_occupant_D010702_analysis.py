#!/usr/bin/env python
"""Append final occupant-in-loop D010702 controller comparison to notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.occupants.metrics import (
    kelvin_hours,
    occupant_probability_comfort_band,
    override_hours,
    setpoint_kelvin_hours,
)
from src.occupants.tdyn import TDynConfig, update_tdyn_reference
from src.rlmpc_pipeline import FITTED_OCCUPANT_PARAMS, Occupant


PREFIX = "ce1_occ_D010702"
MARKER = f"## CE1 Occupant-In-Loop Normalized-D Comparison - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOKS = [
    Path('experiments/analysis/02_mpc_results.ipynb'),
    Path('experiments/analysis/04_rlmpc_results.ipynb'),
    Path('experiments/analysis/08_sac_results.ipynb'),
    Path('experiments/analysis/10_peak_load_profiles.ipynb'),
    Path('experiments/analysis/11_no_occupant_single_building_comparison.ipynb'),
    Path('experiments/analysis/12_simple_v4_comparison.ipynb'),
]

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]
OCCUPANT_MODELS = {name: Occupant(**params) for name, params in FITTED_OCCUPANT_PARAMS.items()}
MODES = ["without_tdyn", "bspline_tdyn"]
DISPLAY_MODES = ["no_occupant", *MODES]
MONTHS = {
    "feb": {"display": "February", "start": 744, "end": 1415},
    "mar": {"display": "March", "start": 1416, "end": 2159},
}

CONTROLLERS = [
    {"key": "mpc", "display": "MPC", "source": "mpc"},
    {"key": "rlmpc_fixed", "display": "GNU-RL fixed model", "source": "rlmpc"},
    {"key": "rlmpc_dyn", "display": "GNU-RL adaptive model", "source": "rlmpc"},
    {"key": "sac", "display": "SAC", "source": "sac_merlin"},
]

NO_OCCUPANT_LABELS = {
    "feb": {
        "mpc": "ce1_final_no_occ_D010702_mpc_2r2c_kalman",
        "rlmpc_fixed": "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_fixed",
        "rlmpc_dyn": "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_adaptive_dyn",
        "sac": "ce1_final_no_occ_D010702_sac_safepi9_bad10_a1",
    },
    "mar": {
        "mpc": "ce1_final_no_occ_D010702_mpc_2r2c_kalman_mar",
        "rlmpc_fixed": "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_fixed_mar",
        "rlmpc_dyn": "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_adaptive_dyn_mar",
        "sac": "ce1_final_no_occ_D010702_sac_feb10_mar_a1",
    },
}


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def label(controller: str, month: str, mode: str, occupant: str) -> str:
    if mode == "no_occupant":
        return NO_OCCUPANT_LABELS[month][controller]
    if controller == "mpc":
        return f"{PREFIX}_mpc_{month}_{occupant}_{mode}"
    if controller == "rlmpc_fixed":
        return f"{PREFIX}_rlmpc_fixed_{month}_{occupant}_{mode}"
    if controller == "rlmpc_dyn":
        return f"{PREFIX}_rlmpc_dyn_{month}_{occupant}_{mode}"
    if controller == "sac":
        return f"{PREFIX}_sac_jan20_feb_{mode}" if month == "feb" else f"{PREFIX}_sac_feb10_mar_{mode}"
    raise ValueError(controller)


def case_scenario(mode: str) -> str:
    return "no_occupant" if mode == "no_occupant" else "occupant_present"


def mode_available(controller: str, month: str, mode: str) -> bool:
    if mode != "no_occupant":
        return True
    return controller in NO_OCCUPANT_LABELS.get(month, {})


def raw_dir(source: str, building_id: str, case: str, scenario: str = "occupant_present") -> Path:
    return RESULTS / "raw" / source / building_id / scenario / case


def available_case(controller: dict, month: str, mode: str, occupant: str | None, building: dict) -> str | None:
    """Resolve the actual case label, including hybrid no-occupant SAC labels."""

    scenario = case_scenario(mode)
    candidates = [label(controller["key"], month, mode, occupant)]
    if mode == "no_occupant" and controller["key"] == "sac" and month == "feb":
        candidates.append("ce1_final_no_occ_D010702_sac_safepi9_a1")

    for case in candidates:
        if (raw_dir(controller["source"], building["result_id"], case, scenario) / "rollout.csv").exists():
            return case
    return None


def summary_path(source: str, building_id: str, scenario: str = "occupant_present") -> Path:
    return RESULTS / "summaries" / source / building_id / scenario / "summary.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def rollout(source: str, building: dict, case: str, scenario: str = "occupant_present") -> pd.DataFrame:
    return read_csv(raw_dir(source, building["result_id"], case, scenario) / "rollout.csv")


def kpis(source: str, building: dict, case: str, scenario: str = "occupant_present") -> dict[str, float]:
    path = raw_dir(source, building["result_id"], case, scenario) / "kpis.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    if "level" in data.columns:
        district = data[data["level"].astype(str) == "district"].copy()
    else:
        district = pd.DataFrame()
    if district.empty:
        district = data
    return {
        str(row["cost_function"]): float(row["value"])
        for _, row in district.iterrows()
        if pd.notna(row.get("value"))
    }


def summary_row(source: str, building: dict, case: str, scenario: str = "occupant_present") -> pd.Series | None:
    path = summary_path(source, building["result_id"], scenario)
    if not path.exists():
        return None
    data = pd.read_csv(path)
    rows = data[data["case"].astype(str) == case]
    if rows.empty:
        return None
    return rows.iloc[-1]


def num(row: pd.Series | None, key: str, default=np.nan) -> float:
    if row is None:
        return float(default)
    if key not in row or pd.isna(row[key]):
        return float(default)
    return float(row[key])


def ref_setpoint(frame: pd.DataFrame) -> np.ndarray:
    for column in ("tdyn", "tracking_setpoint", "effective_setpoint", "baseline_setpoint", "scheduled_setpoint", "tmin"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def effective_setpoint(frame: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "baseline_setpoint", "scheduled_setpoint", "tmin"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def baseline_setpoint(frame: pd.DataFrame) -> np.ndarray | None:
    for column in ("baseline_setpoint", "scheduled_setpoint", "tmin"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return None


def tdyn_setpoint(frame: pd.DataFrame, occupant_name: str | None) -> np.ndarray | None:
    if occupant_name is None:
        return None
    if "tdyn" in frame.columns:
        return pd.to_numeric(frame["tdyn"], errors="coerce").to_numpy(dtype=float)
    if "occupant_feedback" not in frame.columns:
        return None

    base = baseline_setpoint(frame)
    if base is None or len(base) == 0:
        return None

    occupant = OCCUPANT_MODELS.get(occupant_name)
    preferred = None if occupant is None else float(occupant.T_pref)
    feedback = (
        pd.to_numeric(frame["occupant_feedback"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    n = min(len(base), len(feedback))
    if n == 0:
        return None

    config = TDynConfig()
    trace = np.zeros(n, dtype=float)
    trace[0] = float(np.clip(base[0], config.lower_bound, config.upper_bound))
    for idx in range(1, n):
        trace[idx] = update_tdyn_reference(
            trace[idx - 1],
            feedback[idx],
            preferred_temperature=preferred,
            config=config,
        )
    return trace


def rollout_summary(frame: pd.DataFrame, occupant_name: str | None, summary: pd.Series | None) -> dict[str, float]:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    if occupant_name is None:
        lower, upper = 20.0, 24.0
    else:
        occupant = OCCUPANT_MODELS[occupant_name]
        lower, upper = occupant_probability_comfort_band(occupant)
    reward_total = np.nan
    if "reward" in frame.columns:
        reward_total = float(pd.to_numeric(frame["reward"], errors="coerce").fillna(0.0).sum())

    feedback = None
    if "occupant_feedback" in frame.columns:
        feedback = pd.to_numeric(frame["occupant_feedback"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        override_count = float(np.count_nonzero(np.abs(feedback) > 0.0))
    else:
        override_count = num(summary, "override_count", 0.0 if occupant_name is None else np.nan) if summary is not None else (0.0 if occupant_name is None else np.nan)

    base = baseline_setpoint(frame)
    try:
        eff = effective_setpoint(frame)
    except ValueError:
        eff = None
    if base is not None and eff is not None:
        override_h = override_hours(base, eff)
        if np.isnan(override_count):
            active = np.abs(eff[: min(len(base), len(eff))] - base[: min(len(base), len(eff))]) > 1e-6
            starts = active & np.concatenate(([True], ~active[:-1]))
            override_count = float(np.count_nonzero(starts))
    elif feedback is not None:
        override_h = float(np.count_nonzero(np.abs(feedback) > 0.0))
    else:
        override_h = num(summary, "override_hours", 0.0 if occupant_name is None else np.nan) if summary is not None else (0.0 if occupant_name is None else np.nan)

    return {
        "kelvin_hours": kelvin_hours(tin, lower=lower, upper=upper),
        "setpoint_kelvin_hours": (
            setpoint_kelvin_hours(tin, eff)
            if eff is not None
            else (num(summary, "setpoint_kelvin_hours", np.nan) if summary is not None else np.nan)
        ),
        "override_count": override_count,
        "override_hours": override_h,
        "reward_total": reward_total if not np.isnan(reward_total) else (num(summary, "reward_total", np.nan) if summary is not None else np.nan),
        "tdyn_mean": float(pd.to_numeric(frame["tdyn"], errors="coerce").mean()) if "tdyn" in frame.columns else (num(summary, "tdyn_mean", np.nan) if summary is not None else np.nan),
        "tdyn_final": float(pd.to_numeric(frame["tdyn"], errors="coerce").iloc[-1]) if "tdyn" in frame.columns and len(frame) else (num(summary, "tdyn_final", np.nan) if summary is not None else np.nan),
    }


def load(frame: pd.DataFrame) -> np.ndarray:
    for column in ("net_electricity_consumption", "electricity_consumption", "heating_electricity_consumption"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raise ValueError("Rollout has no electricity column.")


def time(frame: pd.DataFrame) -> np.ndarray:
    if "time_step" in frame.columns:
        return pd.to_numeric(frame["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(frame), dtype=int)


def bias_for_time(t: np.ndarray) -> np.ndarray:
    hour_label = (np.asarray(t, dtype=int) % 24) + 1
    preheat = np.isin(hour_label, [1, 2, 3, 4, 5, 14, 15, 16, 17])
    reduce = np.isin(hour_label, [7, 8, 9, 10, 18, 19, 20, 21, 22])
    return np.where(preheat, 1.0, np.where(reduce, -1.0, 0.0))


def building_table(month: str) -> pd.DataFrame:
    rows = []
    for controller in CONTROLLERS:
        for mode in DISPLAY_MODES:
            if not mode_available(controller["key"], month, mode):
                continue
            for building in BUILDINGS:
                occ = None if mode == "no_occupant" else occupant_for_building(building)
                case = available_case(controller, month, mode, occ, building)
                if case is None:
                    continue
                scenario = case_scenario(mode)
                roll = rollout(controller["source"], building, case, scenario)
                summ = summary_row(controller["source"], building, case, scenario)
                derived = rollout_summary(roll, occ, summ)
                vals = kpis(controller["source"], building, case, scenario)
                tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
                rows.append(
                    {
                        "month": MONTHS[month]["display"],
                        "controller": controller["display"],
                        "mode": mode,
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "occupant": "none" if occ is None else occ,
                        "energy": vals.get("electricity_consumption_total", num(summ, "electricity_consumption_total")),
                        "disc_%": 100.0 * vals.get("discomfort_proportion", num(summ, "discomfort_proportion")),
                        "cold_%": 100.0 * vals.get("discomfort_cold_proportion", num(summ, "discomfort_cold_proportion")),
                        "hot_%": 100.0 * vals.get("discomfort_hot_proportion", num(summ, "discomfort_hot_proportion")),
                        "KH": derived["kelvin_hours"],
                        "KH_setpoint": derived["setpoint_kelvin_hours"],
                        "daily_peak": vals.get("daily_peak_average", num(summ, "daily_peak_average")),
                        "ramping": vals.get("ramping_average", np.nan),
                        "overrides": derived["override_count"],
                        "override_hours": derived["override_hours"],
                        "reward_total": derived["reward_total"],
                        "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
                        "tdyn_mean": derived["tdyn_mean"],
                        "tdyn_final": derived["tdyn_final"],
                    }
                )
    return pd.DataFrame(rows)


def mean_table(buildings: pd.DataFrame) -> pd.DataFrame:
    cols = ["energy", "disc_%", "cold_%", "hot_%", "KH", "KH_setpoint", "daily_peak", "ramping", "overrides", "override_hours", "Tin_max", "tdyn_mean", "tdyn_final"]
    return buildings.groupby(["month", "controller", "mode"], as_index=False)[cols].mean(numeric_only=True)


def override_delta_table(buildings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["month", "controller", "building", "building_id", "occupant"]
    for key, group in buildings.groupby(keys, sort=False):
        pivot = group.set_index("mode")
        if not {"without_tdyn", "bspline_tdyn"}.issubset(pivot.index):
            continue
        before = pivot.loc["without_tdyn"]
        after = pivot.loc["bspline_tdyn"]
        rows.append(
            {
                **dict(zip(keys, key)),
                "overrides_without": before["overrides"],
                "overrides_bspline": after["overrides"],
                "delta_overrides": after["overrides"] - before["overrides"],
                "override_hours_without": before["override_hours"],
                "override_hours_bspline": after["override_hours"],
                "delta_override_hours": after["override_hours"] - before["override_hours"],
                "disc_without_%": before["disc_%"],
                "disc_bspline_%": after["disc_%"],
                "delta_disc_%": after["disc_%"] - before["disc_%"],
                "energy_without": before["energy"],
                "energy_bspline": after["energy"],
                "delta_energy": after["energy"] - before["energy"],
            }
        )
    return pd.DataFrame(rows)


def district_frames(month: str) -> list[pd.DataFrame]:
    frames = []
    for controller in CONTROLLERS:
        for mode in DISPLAY_MODES:
            if not mode_available(controller["key"], month, mode):
                continue
            loads = []
            t0 = None
            for building in BUILDINGS:
                occ = None if mode == "no_occupant" else occupant_for_building(building)
                case = available_case(controller, month, mode, occ, building)
                if case is None:
                    continue
                roll = rollout(controller["source"], building, case, case_scenario(mode))
                values = load(roll)
                t = time(roll)
                if t0 is None:
                    t0 = t
                n = min(len(t0), len(values))
                t0 = t0[:n]
                loads.append(values[:n])
            if not loads:
                continue
            n = min(len(x) for x in loads)
            frames.append(
                pd.DataFrame(
                    {
                        "month": MONTHS[month]["display"],
                        "time_step": np.asarray(t0[:n], dtype=int),
                        "load": np.sum([x[:n] for x in loads], axis=0),
                        "controller": controller["display"],
                        "mode": mode,
                        "series": f"{controller['display']} | {mode}",
                    }
                )
            )
    return frames


def district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for frame in frames:
        values = frame["load"].to_numpy(dtype=float)
        hourly = frame.assign(hour=frame["time_step"] % 24).groupby("hour")["load"].mean()
        rows.append(
            {
                "month": frame["month"].iloc[0],
                "controller": frame["controller"].iloc[0],
                "mode": frame["mode"].iloc[0],
                "total_kwh": float(np.sum(values)),
                "peak_kwh": float(np.max(values)),
                "daily_profile_peak_kwh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kwh": float(np.sum(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
                "load_factor": float(np.mean(values) / max(np.max(values), 1e-9)),
            }
        )
    return pd.DataFrame(rows)


def rlmpc_dynamics_table() -> pd.DataFrame:
    rows = []
    for month in MONTHS:
        for mode in DISPLAY_MODES:
            if not mode_available("rlmpc_dyn", month, mode):
                continue
            for building in BUILDINGS:
                occ = None if mode == "no_occupant" else occupant_for_building(building)
                controller = next(item for item in CONTROLLERS if item["key"] == "rlmpc_dyn")
                case = available_case(controller, month, mode, occ, building)
                if case is None:
                    continue
                scenario = case_scenario(mode)
                path = raw_dir("rlmpc", building["result_id"], case, scenario) / "dynamics_history.csv"
                if not path.exists():
                    continue
                hist = pd.read_csv(path)
                if hist.empty:
                    continue
                initial = hist[hist["epoch"] == 0].head(1)
                if initial.empty:
                    initial = hist.head(1)
                final = hist[hist["epoch"] == -1].tail(1)
                if final.empty:
                    final = hist.tail(1)
                initial = initial.iloc[0]
                final = final.iloc[0]
                rows.append(
                    {
                        "month": MONTHS[month]["display"],
                        "mode": mode,
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "occupant": "none" if occ is None else occ,
                        "train_rmse_before": num(initial, "train_rmse_c"),
                        "train_rmse_after": num(final, "train_rmse_c"),
                        "train_rmse_delta": num(final, "train_rmse_c") - num(initial, "train_rmse_c"),
                        "val_rmse_before": num(initial, "val_rmse_c"),
                        "val_rmse_after": num(final, "val_rmse_c"),
                        "val_rmse_delta": num(final, "val_rmse_c") - num(initial, "val_rmse_c"),
                        "C_before": num(initial, "param_C"),
                        "C_after": num(final, "param_C"),
                        "Pnom_before": num(initial, "param_Pnom"),
                        "Pnom_after": num(final, "param_Pnom"),
                        "comfort_track_weight": num(final, "param_balanced_comfort_track_weight"),
                        "comfort_cold_weight": num(final, "param_balanced_comfort_cold_weight"),
                        "comfort_hot_weight": num(final, "param_balanced_comfort_hot_weight"),
                    }
                )
    return pd.DataFrame(rows)


def make_comfort_grid(month: str, controller: dict, mode: str) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        occ = None if mode == "no_occupant" else occupant_for_building(building)
        case = available_case(controller, month, mode, occ, building)
        if case is None:
            ax.set_axis_off()
            continue
        roll = rollout(controller["source"], building, case, case_scenario(mode))
        t = time(roll)
        tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        eff = effective_setpoint(roll)
        tdyn = tdyn_setpoint(roll, occ)
        n = min(len(t), len(tin), len(eff), len(tdyn) if tdyn is not None else len(t))
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], eff[:n], lw=0.75, color="tab:orange", label="setpoint")
        if tdyn is not None:
            ax.plot(t[:n], tdyn[:n], lw=0.75, color="black", label="Tdyn")
        ax.set_title(f"{building['display']} | {'none' if occ is None else occ}", fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{controller['display']} {mode} | {MONTHS[month]['display']} comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{month}_{controller['key']}_{mode}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_district_plots(month: str, frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("series", sort=False):
        ax.plot(group["time_step"], group["load"], lw=0.95, label=name)
    ax.set_title(f"Occupant-in-loop district load - {MONTHS[month]['display']}")
    ax.set_xlabel(f"{MONTHS[month]['display']} time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    ts_path = FIG_DIR / f"{month}_district_load_timeseries.png"
    fig.savefig(ts_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["series", "hour"], as_index=False)["load"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("series", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.2, label=name)
    ax.set_title(f"Mean hourly district load - {MONTHS[month]['display']}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    daily_path = FIG_DIR / f"{month}_district_mean_hourly_profile.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return ts_path, daily_path


def make_override_plot(month: str, overrides: pd.DataFrame) -> Path:
    data = overrides[overrides["month"] == MONTHS[month]["display"]].copy()
    agg = data.groupby("controller", as_index=False)[["overrides_without", "overrides_bspline"]].sum()
    x = np.arange(len(agg))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.18, agg["overrides_without"], width=0.36, label="without Tdyn")
    ax.bar(x + 0.18, agg["overrides_bspline"], width=0.36, label="B-spline Tdyn")
    ax.set_xticks(x, agg["controller"])
    ax.set_ylabel("Total override starts")
    ax.set_title(f"Occupant override starts - {MONTHS[month]['display']}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / f"{month}_override_counts.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def sac_training_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    reward_rows = []
    loss_rows = []
    cases = [
        ("February deploy", "without_tdyn", f"{PREFIX}_sac_jan20_feb_without_tdyn"),
        ("February deploy", "bspline_tdyn", f"{PREFIX}_sac_jan20_feb_bspline_tdyn"),
        ("March deploy", "without_tdyn", f"{PREFIX}_sac_feb10_mar_without_tdyn"),
        ("March deploy", "bspline_tdyn", f"{PREFIX}_sac_feb10_mar_bspline_tdyn"),
    ]
    for phase, mode, case in cases:
        for building in BUILDINGS:
            case_dir = raw_dir("sac_merlin", building["result_id"], case)
            train_path = case_dir / "online_training_rollout.csv"
            loss_path = case_dir / "sac_training_losses.csv"
            if train_path.exists():
                data = pd.read_csv(train_path)
                if not data.empty and {"episode", "raw_reward", "reward"}.issubset(data.columns):
                    grp = data.groupby("episode", as_index=False).agg(
                        raw_reward_mean=("raw_reward", "mean"),
                        reward_mean=("reward", "mean"),
                        action_mean=("action", "mean"),
                    )
                    grp["phase"] = phase
                    grp["mode"] = mode
                    grp["building"] = building["display"]
                    reward_rows.append(grp)
            if loss_path.exists():
                loss = pd.read_csv(loss_path)
                if not loss.empty:
                    grp = loss.groupby("episode", as_index=False).agg(
                        soft_q1_loss_mean=("soft_q1_loss", "mean"),
                        soft_q1_loss_max=("soft_q1_loss", "max"),
                        soft_q2_loss_mean=("soft_q2_loss", "mean"),
                        soft_q2_loss_max=("soft_q2_loss", "max"),
                        policy_loss_mean=("policy_loss", "mean"),
                        policy_loss_max=("policy_loss", "max"),
                        alpha_mean=("alpha", "mean"),
                        alpha_max=("alpha", "max"),
                    )
                    grp["phase"] = phase
                    grp["mode"] = mode
                    grp["building"] = building["display"]
                    loss_rows.append(grp)
    rewards = pd.concat(reward_rows, ignore_index=True) if reward_rows else pd.DataFrame()
    losses = pd.concat(loss_rows, ignore_index=True) if loss_rows else pd.DataFrame()
    if not rewards.empty:
        rewards = rewards.groupby(["phase", "mode", "episode"], as_index=False).mean(numeric_only=True)
    if not losses.empty:
        losses = losses.groupby(["phase", "mode", "episode"], as_index=False).mean(numeric_only=True)
    return rewards, losses


def make_sac_training_plots(rewards: pd.DataFrame, losses: pd.DataFrame) -> tuple[Path | None, Path | None]:
    reward_path = None
    if not rewards.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for (phase, mode), group in rewards.groupby(["phase", "mode"], sort=False):
            ax.plot(group["episode"], group["raw_reward_mean"], marker="o", lw=1.2, label=f"{phase} | {mode}")
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_title("SAC online reward by episode - per-step mean")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean raw reward per step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        reward_path = FIG_DIR / "sac_online_reward_by_episode.png"
        fig.savefig(reward_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    loss_path = None
    if not losses.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for (phase, mode), group in losses.groupby(["phase", "mode"], sort=False):
            label = f"{phase} | {mode}"
            axes[0].plot(group["episode"], group["soft_q1_loss_mean"], marker="o", lw=1.0, label=label)
            axes[1].plot(group["episode"], group["policy_loss_mean"], marker="o", lw=1.0, label=label)
        axes[0].set_title("Soft Q1 loss")
        axes[1].set_title("Policy loss")
        for ax in axes:
            ax.set_xlabel("Training episode")
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle("SAC loss means by episode")
        fig.tight_layout()
        loss_path = FIG_DIR / "sac_losses_by_episode.png"
        fig.savefig(loss_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return reward_path, loss_path


TABLE_STYLE = """
<style>
.ce1-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 0.25rem 0 0.9rem 0;
}
.ce1-table {
  border-collapse: collapse;
  font-size: 11px;
  line-height: 1.15;
  width: auto;
}
.ce1-table th,
.ce1-table td {
  border: 1px solid #666;
  padding: 3px 6px;
  white-space: nowrap;
}
.ce1-table th {
  font-weight: 600;
}
.ce1-table td {
  text-align: right;
}
.ce1-table td:first-child,
.ce1-table td:nth-child(2),
.ce1-table td:nth-child(3),
.ce1-table td:nth-child(4) {
  text-align: left;
}
</style>
""".strip()


CONTROLLER_SHORT = {
    "GNU-RL adaptive model": "GNU adapt",
    "GNU-RL fixed model": "GNU fixed",
}


def table_display(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for label_col in ("controller", "ctrl"):
        if label_col in out.columns:
            out[label_col] = out[label_col].replace(CONTROLLER_SHORT)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    html = out.to_html(index=False, escape=True, border=0, classes="ce1-table")
    html = html.replace(
        "<table ",
        '<table style="border-collapse:collapse;font-size:11px;line-height:1.15;width:auto;" ',
        1,
    )
    html = html.replace(
        "<th>",
        '<th style="border:1px solid #666;padding:3px 6px;white-space:nowrap;font-weight:600;">',
    )
    html = html.replace(
        "<td>",
        '<td style="border:1px solid #666;padding:3px 6px;white-space:nowrap;">',
    )
    return (
        '<div class="ce1-table-wrap" '
        'style="max-width:100%;overflow-x:auto;margin:0.25rem 0 0.9rem 0;">'
        f"{html}</div>"
    )


def compact_mean_table(means: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "month",
        "controller",
        "mode",
        "energy",
        "disc_%",
        "cold_%",
        "hot_%",
        "daily_peak",
        "overrides",
        "override_hours",
        "tdyn_final",
    ]
    available = [col for col in cols if col in means.columns]
    return means[available].rename(
        columns={
            "controller": "ctrl",
            "override_hours": "ovr_h",
            "overrides": "ovr",
            "daily_peak": "d_peak",
            "tdyn_final": "tdyn_f",
        }
    )


def compact_district_table(districts: pd.DataFrame) -> pd.DataFrame:
    out = districts.copy()
    out["total_MWh"] = out["total_kwh"] / 1000.0
    cols = [
        "month",
        "controller",
        "mode",
        "total_MWh",
        "peak_kwh",
        "daily_profile_peak_kwh",
        "daily_profile_peak_hour",
        "load_factor",
    ]
    return out[[col for col in cols if col in out.columns]].rename(
        columns={
            "controller": "ctrl",
            "daily_profile_peak_kwh": "day_peak",
            "daily_profile_peak_hour": "peak_hr",
            "load_factor": "lf",
        }
    )


def compact_override_table(overrides: pd.DataFrame) -> pd.DataFrame:
    out = (
        overrides.groupby(["month", "controller"], as_index=False)[
            ["overrides_without", "overrides_bspline", "delta_overrides", "delta_disc_%", "delta_energy"]
        ]
        .sum(numeric_only=True)
    )
    return out.rename(
        columns={
            "controller": "ctrl",
            "overrides_without": "ovr_no_tdyn",
            "overrides_bspline": "ovr_tdyn",
            "delta_overrides": "delta_ovr",
            "delta_disc_%": "d_disc_%",
            "delta_energy": "d_energy",
        }
    )


def compact_dynamics_summary(dynamics: pd.DataFrame) -> pd.DataFrame:
    if dynamics.empty:
        return pd.DataFrame()
    out = dynamics.groupby(["month", "mode"], as_index=False).agg(
        train_before=("train_rmse_before", "mean"),
        train_after=("train_rmse_after", "mean"),
        val_before=("val_rmse_before", "mean"),
        val_after=("val_rmse_after", "mean"),
        track_w=("comfort_track_weight", "mean"),
        cold_w=("comfort_cold_weight", "mean"),
        hot_w=("comfort_hot_weight", "mean"),
    )
    return out.rename(
        columns={
            "train_before": "tr_before",
            "train_after": "tr_after",
            "val_before": "val_before",
            "val_after": "val_after",
        }
    )


def worst_building_preview(all_buildings: pd.DataFrame, month_name: str, n: int = 12) -> pd.DataFrame:
    cols = [
        "controller",
        "mode",
        "building",
        "occupant",
        "energy",
        "disc_%",
        "cold_%",
        "hot_%",
        "overrides",
        "Tin_max",
    ]
    table = all_buildings[all_buildings["month"] == month_name].copy()
    table = table.sort_values("disc_%", ascending=False).head(n)
    return table[[col for col in cols if col in table.columns]].rename(
        columns={
            "controller": "ctrl",
            "building": "bldg",
            "occupant": "occ",
            "overrides": "ovr",
        }
    )


def image_cell(path: Path, title: str):
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{title}](../{path.as_posix()})")


def append_notebooks(cells) -> None:
    for path in NOTEBOOKS:
        nb = nbf.read(path, as_version=4)
        nb.cells = [cell for cell in nb.cells if MARKER not in str(cell.get("source", ""))]
        nb.cells.extend(cells)
        nbf.write(nb, path)
        print(f"Appended {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    building_tables = []
    mean_tables = []
    override_tables = []
    district_tables = []
    district_plot_paths = {}
    comfort_paths = {}
    override_paths = {}
    for month in MONTHS:
        btab = building_table(month)
        building_tables.append(btab)
        mean_tables.append(mean_table(btab))
        override_tables.append(override_delta_table(btab))
        frames = district_frames(month)
        district_tables.append(district_kpis(frames))
        district_plot_paths[month] = make_district_plots(month, frames)
        override_paths[month] = make_override_plot(month, override_tables[-1])
        for controller in CONTROLLERS:
            for mode in DISPLAY_MODES:
                if not mode_available(controller["key"], month, mode):
                    continue
                comfort_paths[(month, controller["key"], mode)] = make_comfort_grid(month, controller, mode)

    all_buildings = pd.concat(building_tables, ignore_index=True)
    means = pd.concat(mean_tables, ignore_index=True)
    overrides = pd.concat(override_tables, ignore_index=True)
    districts = pd.concat(district_tables, ignore_index=True)
    dynamics = rlmpc_dynamics_table()
    sac_rewards, sac_losses = sac_training_tables()
    reward_plot, loss_plot = make_sac_training_plots(sac_rewards, sac_losses)

    all_buildings.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all.csv", index=False)
    means.to_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv", index=False)
    overrides.to_csv(ANALYSIS / f"{PREFIX}_override_deltas.csv", index=False)
    districts.to_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv", index=False)
    if not dynamics.empty:
        dynamics.to_csv(ANALYSIS / f"{PREFIX}_rlmpc_dynamics_summary.csv", index=False)
    if not sac_rewards.empty:
        sac_rewards.to_csv(ANALYSIS / f"{PREFIX}_sac_reward_by_episode.csv", index=False)
    if not sac_losses.empty:
        sac_losses.to_csv(ANALYSIS / f"{PREFIX}_sac_loss_by_episode.csv", index=False)

    mean_display = compact_mean_table(means)
    district_display = compact_district_table(districts)
    override_summary = compact_override_table(overrides)
    dynamics_summary = compact_dynamics_summary(dynamics)

    cells = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}
{TABLE_STYLE}

Final occupant-in-loop run with cyclic occupant assignment:
B1/B5/... use `occ1_tolerant`, B2/B6/... use `occ2_sensitive`, B3/B7/... use `occ3_cold`, and B4/B8/... use `occ4_hot`.

Common cost/reward for all controllers:

`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`

`reward = -D_k`, no GridStress/energy term. `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and the true-reference guard is `Tref +/- 2 C`.

Three deployment references are shown where the artifacts exist: no occupant (`no_occupant`), raw occupant overrides (`without_tdyn`), and B-spline occupancy-predicted adaptive reference tracking (`bspline_tdyn`). GNU-RL has two fixed-cost variants: fixed model, and online RC-model adaptation from one-step `Tin(k+1)` prediction error. SAC uses one safe-PI January episode plus 19 January reward-training episodes for February deployment, then resumes for 10 February reward-training episodes for March deployment.

The B-spline Tdyn branch uses the Zhang-style dynamic reference update:

`Tdyn_{{k+1}} = clip(Tdyn_k + rho*(Tpref-Tdyn_k) + bounded_feedback_nudge_toward_Tpref)`

and the controller/reward reference is occupancy-gated:

`Tref_k = (1 - p_occ,k)*T_schedule,k + p_occ,k*Tdyn_k`.

Mean building KPIs:

{table_display(mean_display, digits=3)}

District aggregate KPIs:

{table_display(district_display, digits=3)}

Override summary:

{table_display(override_summary, digits=2)}

GNU-RL adaptive dynamics summary:

{table_display(dynamics_summary, digits=3) if not dynamics_summary.empty else "No adaptive-dynamics histories found."}

Saved compact tables:

- `analysis/{PREFIX}_building_kpis_all.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_override_deltas.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_rlmpc_dynamics_summary.csv`
- `analysis/{PREFIX}_sac_reward_by_episode.csv`
- `analysis/{PREFIX}_sac_loss_by_episode.csv`
"""
        )
    ]

    for month in MONTHS:
        cells.append(image_cell(district_plot_paths[month][0], f"{MONTHS[month]['display']} district load"))
        cells.append(image_cell(district_plot_paths[month][1], f"{MONTHS[month]['display']} mean hourly district load"))
        cells.append(image_cell(override_paths[month], f"{MONTHS[month]['display']} override counts"))
        for controller in CONTROLLERS:
            for mode in DISPLAY_MODES:
                path = comfort_paths.get((month, controller["key"], mode))
                if path is not None:
                    cells.append(image_cell(path, f"{MONTHS[month]['display']} {controller['display']} {mode} comfort"))

    if reward_plot is not None:
        cells.append(image_cell(reward_plot, "SAC reward by episode"))
    if loss_plot is not None:
        cells.append(image_cell(loss_plot, "SAC losses by episode"))

    for month_name in ("February", "March"):
        compact = worst_building_preview(all_buildings, month_name)
        cells.append(
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n### {month_name} Worst Per-Building Comfort Rows\n\n"
                f"Top 12 by discomfort shown here. Full per-building KPIs are saved in "
                f"`analysis/{PREFIX}_building_kpis_all.csv`.\n\n"
                f"{TABLE_STYLE}\n\n{table_display(compact, digits=3)}"
            )
        )

    append_notebooks(cells)


if __name__ == "__main__":
    main()
