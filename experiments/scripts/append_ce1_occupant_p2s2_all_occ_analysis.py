#!/usr/bin/env python
"""Append final p2 all-occupants controller comparison to notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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


PREFIX = "ce1_occ_p2s2_all_occ"
MARKER = f"## CE1 Occupant-In-Loop P2 All-Occupants Comparison - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
DATASET_DIR = Path("data") / "datasets" / DATASET_NAME
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
    {"key": "rlmpc_adaptive", "display": "GNU-RL adaptive model", "source": "rlmpc"},
    {"key": "sac", "display": "SAC", "source": "sac_merlin"},
]
BASELINE_CONTROLLER = {"key": "baseline", "display": "Baseline", "source": "mpc"}
RBC_CONTROLLER = {"key": "rbc_safepi", "display": "Safe-PI/RBC", "source": "rbc_safe_pi"}
CONTROLLERS_WITH_BASELINE = [BASELINE_CONTROLLER, *CONTROLLERS]
HEATMAP_CONTROLLERS = [
    RBC_CONTROLLER,
    {"key": "mpc", "display": "MPC", "source": "mpc"},
    {"key": "rlmpc_adaptive", "display": "GNU-RL", "source": "rlmpc"},
    {"key": "sac", "display": "SAC", "source": "sac_merlin"},
]
THESIS_CONTROLLERS = ["Safe-PI/RBC", "MPC", "GNU-RL adaptive model", "SAC"]
THESIS_CONTROLLER_LABELS = {
    "Safe-PI/RBC": "Safe-PI",
    "MPC": "MPC",
    "GNU-RL adaptive model": "GNU-RL\nadapt",
    "SAC": "SAC",
}

CITYLEARN_KPI_COLUMNS = [
    "electricity_consumption_total",
    "zero_net_energy",
    "carbon_emissions_total",
    "cost_total",
    "daily_peak_average",
    "all_time_peak_average",
    "ramping_average",
    "daily_one_minus_load_factor_average",
    "monthly_one_minus_load_factor_average",
    "discomfort_proportion",
    "discomfort_cold_proportion",
    "discomfort_hot_proportion",
    "discomfort_cold_delta_average",
    "discomfort_cold_delta_maximum",
    "discomfort_hot_delta_average",
    "discomfort_hot_delta_maximum",
]

_DATASET_CACHE: dict[str, pd.DataFrame] = {}
_WEATHER_CACHE: pd.DataFrame | None = None


def label(controller: str, month: str, mode: str, occupant: str) -> str:
    if mode == "no_occupant":
        if controller == "sac":
            return f"{PREFIX}_sac_jan20_feb_no_occupant" if month == "feb" else f"{PREFIX}_sac_feb10_mar_no_occupant"
        return f"{PREFIX}_{controller}_{month}_no_occupant"
    if controller == "mpc":
        return f"{PREFIX}_mpc_{month}_{occupant}_{mode}"
    if controller == "rlmpc_fixed":
        return f"{PREFIX}_rlmpc_fixed_{month}_{occupant}_{mode}"
    if controller == "rlmpc_adaptive":
        return f"{PREFIX}_rlmpc_adaptive_{month}_{occupant}_{mode}"
    if controller == "sac":
        return f"{PREFIX}_sac_jan20_feb_{occupant}_{mode}" if month == "feb" else f"{PREFIX}_sac_feb10_mar_{occupant}_{mode}"
    raise ValueError(controller)


def case_scenario(mode: str) -> str:
    return "no_occupant" if mode == "no_occupant" else "occupant_present"


def mode_available(controller: str, month: str, mode: str) -> bool:
    if controller == "baseline":
        return mode == "no_occupant" and baseline_case(month) is not None
    return True


def raw_dir(source: str, building_id: str, case: str, scenario: str = "occupant_present") -> Path:
    return RESULTS / "raw" / source / building_id / scenario / case


def baseline_case(month: str) -> str | None:
    # Existing per-building baseline artifacts are the 671-step February run.
    return "baseline" if month == "feb" else None


def available_case(controller: dict, month: str, mode: str, occupant: str | None, building: dict) -> str | None:
    """Resolve the actual case label for this controller/month/mode/occupant."""

    if controller["key"] == "baseline":
        if mode != "no_occupant" or occupant is not None:
            return None
        case = baseline_case(month)
        if case is None:
            return None
        scenario = case_scenario(mode)
        path = raw_dir(controller["source"], building["result_id"], case, scenario) / "rollout.csv"
        return case if path.exists() else None

    scenario = case_scenario(mode)
    candidates = [label(controller["key"], month, mode, occupant)]
    if (
        controller["key"] == "rlmpc_adaptive"
        and month == "mar"
        and mode != "no_occupant"
        and occupant is not None
    ):
        # March adaptive GNU-RL is launched one occupant mode at a time so it can
        # continue from the matching February checkpoint.  The lower-level
        # runner appends occupant/mode to the supplied label, so those artifacts
        # have a duplicated suffix.
        candidates.append(f"{candidates[0]}_{occupant}_{mode}")

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


def citylearn_kpi_values(values: dict[str, float], summary: pd.Series | None) -> dict[str, float]:
    return {
        name: values.get(name, num(summary, name, np.nan))
        for name in CITYLEARN_KPI_COLUMNS
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
    if {"baseline_setpoint", "tdyn", "occ_bin_1step"}.issubset(frame.columns):
        schedule = pd.to_numeric(frame["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
        tdyn = pd.to_numeric(frame["tdyn"], errors="coerce").to_numpy(dtype=float)
        gate = (
            pd.to_numeric(frame["occ_bin_1step"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        n = min(len(schedule), len(tdyn), len(gate))
        reference = np.full(len(frame), np.nan, dtype=float)
        reference[:n] = (1.0 - gate[:n]) * schedule[:n] + gate[:n] * tdyn[:n]
        if n < len(frame) and "reference_setpoint" in frame.columns:
            fallback = pd.to_numeric(frame["reference_setpoint"], errors="coerce").to_numpy(dtype=float)
            reference[n:] = fallback[n:]
        return reference

    for column in (
        "reference_setpoint",
        "tracking_setpoint",
        "effective_setpoint",
        "baseline_setpoint",
        "scheduled_setpoint",
        "tmin",
        "tdyn",
    ):
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
    try:
        ref = ref_setpoint(frame)
    except ValueError:
        ref = eff
    if occupant_name is None:
        override_h = 0.0
        override_count = 0.0
    elif base is not None and eff is not None:
        override_h = override_hours(base, eff)
        if np.isnan(override_count):
            active = np.abs(eff[: min(len(base), len(eff))] - base[: min(len(base), len(eff))]) > 1e-6
            starts = active & np.concatenate(([True], ~active[:-1]))
            override_count = float(np.count_nonzero(starts))
    elif feedback is not None:
        override_h = float(np.count_nonzero(np.abs(feedback) > 0.0))
    else:
        override_h = num(summary, "override_hours", 0.0 if occupant_name is None else np.nan) if summary is not None else (0.0 if occupant_name is None else np.nan)

    comfort_band_kh = kelvin_hours(tin, lower=lower, upper=upper)
    active_reference_kh = (
        setpoint_kelvin_hours(tin, ref)
        if ref is not None
        else (num(summary, "setpoint_kelvin_hours", np.nan) if summary is not None else np.nan)
    )

    return {
        "kelvin_hours": active_reference_kh,
        "kelvin_hours_to_comfort_band": comfort_band_kh,
        "setpoint_kelvin_hours": active_reference_kh,
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


def dataset_column(building: dict, column: str, steps: np.ndarray) -> np.ndarray:
    if column not in ("heating_demand",):
        return np.array([], dtype=float)
    path = DATASET_DIR / f"{building['name']}.csv"
    if not path.exists():
        return np.array([], dtype=float)
    key = str(path)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = pd.read_csv(path)
    data = _DATASET_CACHE[key]
    if column not in data.columns:
        return np.array([], dtype=float)
    indices = np.asarray(steps, dtype=int)
    valid = (indices >= 0) & (indices < len(data))
    if not np.any(valid):
        return np.array([], dtype=float)
    return pd.to_numeric(data.iloc[indices[valid]][column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def load_metrics(frame: pd.DataFrame, building: dict) -> dict[str, float]:
    values = load(frame)
    heating_electricity = (
        pd.to_numeric(frame["heating_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "heating_electricity_consumption" in frame.columns
        else np.array([], dtype=float)
    )
    heating_demand = (
        pd.to_numeric(frame["heating_demand"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "heating_demand" in frame.columns
        else dataset_column(building, "heating_demand", time(frame))
    )
    diff = np.diff(values) if len(values) > 1 else np.array([], dtype=float)
    return {
        "net_electricity_consumption_kwh": float(np.sum(values)),
        "electricity_import_kwh": float(np.sum(np.clip(values, 0.0, None))),
        "max_net_electricity_consumption_kwh": float(np.max(values)) if len(values) else np.nan,
        "ramping_up_kwh": float(np.sum(np.clip(diff, 0.0, None))) if len(diff) else 0.0,
        "ramping_abs_kwh": float(np.sum(np.abs(diff))) if len(diff) else 0.0,
        "heating_demand_kwh": float(np.sum(heating_demand)) if len(heating_demand) else np.nan,
        "heating_electricity_consumption_kwh": float(np.sum(heating_electricity)) if len(heating_electricity) else np.nan,
    }


def time(frame: pd.DataFrame) -> np.ndarray:
    if "time_step" in frame.columns:
        return pd.to_numeric(frame["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(frame), dtype=int)


def bias_for_time(t: np.ndarray) -> np.ndarray:
    hour_label = (np.asarray(t, dtype=int) % 24) + 1
    preheat = np.isin(hour_label, [1, 2, 3, 4, 5, 14, 15, 16, 17])
    reduce = np.isin(hour_label, [7, 8, 9, 10, 18, 19, 20, 21, 22])
    return np.where(preheat, 1.0, np.where(reduce, -1.0, 0.0))


def weather_frame() -> pd.DataFrame:
    global _WEATHER_CACHE
    if _WEATHER_CACHE is None:
        path = DATASET_DIR / "weather.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        _WEATHER_CACHE = pd.read_csv(path)
    return _WEATHER_CACHE


def atg_reference_temperature(abs_steps: np.ndarray) -> np.ndarray:
    """ISSO 74 ATG running outdoor reference temperature from the slide formula."""

    weather = weather_frame()
    tout = pd.to_numeric(weather["outdoor_dry_bulb_temperature"], errors="coerce").to_numpy(dtype=float)
    if len(tout) == 0:
        return np.full_like(abs_steps, np.nan, dtype=float)

    days = np.arange(int(np.ceil(len(tout) / 24.0)))
    daily_means = np.array(
        [
            np.nanmean(tout[day * 24 : min((day + 1) * 24, len(tout))])
            for day in days
        ],
        dtype=float,
    )
    if len(daily_means) == 0:
        return np.full_like(abs_steps, np.nan, dtype=float)

    teref = np.zeros_like(daily_means, dtype=float)
    for day in range(len(daily_means)):
        d0 = daily_means[day]
        d1 = daily_means[max(day - 1, 0)]
        d2 = daily_means[max(day - 2, 0)]
        d3 = daily_means[max(day - 3, 0)]
        teref[day] = (1.0 * d0 + 0.8 * d1 + 0.4 * d2 + 0.2 * d3) / 2.4

    abs_steps = np.asarray(abs_steps, dtype=int)
    day_index = np.clip(abs_steps // 24, 0, len(teref) - 1)
    return teref[day_index]


def atg_neutral_temperature(abs_steps: np.ndarray) -> np.ndarray:
    return 21.45 + 0.11 * atg_reference_temperature(abs_steps)


def classify_atg_share(a_share: float, b_share: float, c_share: float) -> str:
    if a_share >= 90.0:
        return "A"
    if b_share >= 80.0:
        return "B"
    if c_share >= 65.0:
        return "C"
    return "outside_C"


def atg_metrics_for_rollout(frame: pd.DataFrame, month: str) -> dict[str, float | str]:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    local_steps = time(frame)
    month_start = MONTHS[month]["start"]
    abs_steps = month_start + np.asarray(local_steps, dtype=int)
    n = min(len(tin), len(abs_steps))
    if n == 0:
        return {
            "atg_A_pct": np.nan,
            "atg_B_pct": np.nan,
            "atg_C_pct": np.nan,
            "atg_outside_C_pct": np.nan,
            "atg_mean_abs_delta_C": np.nan,
            "atg_p95_abs_delta_C": np.nan,
            "atg_class": "missing",
        }
    tin = tin[:n]
    neutral = atg_neutral_temperature(abs_steps[:n])
    valid = np.isfinite(tin) & np.isfinite(neutral)
    if not np.any(valid):
        return {
            "atg_A_pct": np.nan,
            "atg_B_pct": np.nan,
            "atg_C_pct": np.nan,
            "atg_outside_C_pct": np.nan,
            "atg_mean_abs_delta_C": np.nan,
            "atg_p95_abs_delta_C": np.nan,
            "atg_class": "missing",
        }
    delta = tin[valid] - neutral[valid]
    abs_delta = np.abs(delta)
    a_share = 100.0 * float(np.mean(abs_delta <= 1.25))
    b_share = 100.0 * float(np.mean(abs_delta <= 2.0))
    c_share = 100.0 * float(np.mean(abs_delta <= 2.5))
    return {
        "atg_A_pct": a_share,
        "atg_B_pct": b_share,
        "atg_C_pct": c_share,
        "atg_outside_C_pct": 100.0 - c_share,
        "atg_mean_abs_delta_C": float(np.mean(abs_delta)),
        "atg_p95_abs_delta_C": float(np.quantile(abs_delta, 0.95)),
        "atg_class": classify_atg_share(a_share, b_share, c_share),
    }


def atg_table(all_buildings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    data = all_buildings[all_buildings["controller"].isin(THESIS_CONTROLLERS)].copy()
    for _, row in data.iterrows():
        month_key = next(
            key for key, spec in MONTHS.items() if spec["display"] == row["month"]
        )
        roll = read_csv(
            raw_dir(row["source"], row["building_id"], row["case"], row["scenario"])
            / "rollout.csv"
        )
        rows.append(
            {
                "month": row["month"],
                "controller": row["controller"],
                "mode": row["mode"],
                "occupant": row["occupant"],
                "building": row["building"],
                "building_id": row["building_id"],
                **atg_metrics_for_rollout(roll, month_key),
            }
        )
    return pd.DataFrame(rows)


def atg_summary_table(atg: pd.DataFrame) -> pd.DataFrame:
    if atg.empty:
        return pd.DataFrame()
    classes = ["A", "B", "C", "outside_C", "missing"]
    rows = []
    for key, group in atg.groupby(["month", "controller", "mode", "occupant"], sort=False):
        counts = group["atg_class"].value_counts().to_dict()
        rows.append(
            {
                "month": key[0],
                "controller": key[1],
                "mode": key[2],
                "occupant": key[3],
                "buildings": int(len(group)),
                **{f"ATG_{klass}": int(counts.get(klass, 0)) for klass in classes},
                "A_time_mean_pct": float(pd.to_numeric(group["atg_A_pct"], errors="coerce").mean()),
                "B_time_mean_pct": float(pd.to_numeric(group["atg_B_pct"], errors="coerce").mean()),
                "C_time_mean_pct": float(pd.to_numeric(group["atg_C_pct"], errors="coerce").mean()),
                "outside_C_mean_pct": float(pd.to_numeric(group["atg_outside_C_pct"], errors="coerce").mean()),
                "mean_abs_delta_C": float(pd.to_numeric(group["atg_mean_abs_delta_C"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def building_table(month: str) -> pd.DataFrame:
    rows = []
    for controller in CONTROLLERS_WITH_BASELINE:
        for mode in DISPLAY_MODES:
            if not mode_available(controller["key"], month, mode):
                continue
            occupants = [None] if mode == "no_occupant" else OCCUPANTS
            for occ in occupants:
                for building in BUILDINGS:
                    case = available_case(controller, month, mode, occ, building)
                    if case is None:
                        continue
                    scenario = case_scenario(mode)
                    roll = rollout(controller["source"], building, case, scenario)
                    summ = summary_row(controller["source"], building, case, scenario)
                    derived = rollout_summary(roll, occ, summ)
                    vals = kpis(controller["source"], building, case, scenario)
                    tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
                    citylearn_kpis = citylearn_kpi_values(vals, summ)
                    raw_load = load_metrics(roll, building)
                    rows.append(
                        {
                            "month": MONTHS[month]["display"],
                            "controller": controller["display"],
                            "controller_key": controller["key"],
                            "source": controller["source"],
                            "case": case,
                            "scenario": scenario,
                            "mode": mode,
                            "building": building["display"],
                            "building_id": building["result_id"],
                            "occupant": "none" if occ is None else occ,
                            **citylearn_kpis,
                            "kelvin_hours": derived["kelvin_hours"],
                            "kelvin_hours_to_comfort_band": derived["kelvin_hours_to_comfort_band"],
                            "kelvin_hours_to_active_reference": derived["setpoint_kelvin_hours"],
                            **raw_load,
                            "overrides": derived["override_count"],
                            "override_hours": derived["override_hours"],
                            "reward_total": derived["reward_total"],
                            "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
                            "tdyn_mean": derived["tdyn_mean"],
                            "tdyn_final": derived["tdyn_final"],
                        }
                    )
    return pd.DataFrame(rows)


def rbc_case(month: str, mode: str, occupant_name: str | None) -> str | None:
    if mode == "no_occupant":
        return f"ce1_thesis_rbc_safepi_{month}_no_occupant"
    if occupant_name is None:
        return None
    return f"ce1_thesis_rbc_safepi_{month}_{occupant_name}_{mode}"


def rbc_building_table(month: str) -> pd.DataFrame:
    rows = []
    for mode in ("no_occupant", "without_tdyn", "bspline_tdyn"):
        occupants = [None] if mode == "no_occupant" else OCCUPANTS
        for occ in occupants:
            case = rbc_case(month, mode, occ)
            if case is None:
                continue
            scenario = case_scenario(mode)
            for building in BUILDINGS:
                root = raw_dir(RBC_CONTROLLER["source"], building["result_id"], case, scenario)
                if not (root / "rollout.csv").exists():
                    continue
                roll = rollout(RBC_CONTROLLER["source"], building, case, scenario)
                summ = summary_row(RBC_CONTROLLER["source"], building, case, scenario)
                derived = rollout_summary(roll, occ, summ)
                vals = kpis(RBC_CONTROLLER["source"], building, case, scenario)
                tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
                citylearn_kpis = citylearn_kpi_values(vals, summ)
                raw_load = load_metrics(roll, building)
                rows.append(
                    {
                        "month": MONTHS[month]["display"],
                        "controller": RBC_CONTROLLER["display"],
                        "controller_key": RBC_CONTROLLER["key"],
                        "source": RBC_CONTROLLER["source"],
                        "case": case,
                        "scenario": scenario,
                        "mode": mode,
                        "building": building["display"],
                        "building_id": building["result_id"],
                        "occupant": "none" if occ is None else occ,
                        **citylearn_kpis,
                        "kelvin_hours": derived["kelvin_hours"],
                        "kelvin_hours_to_comfort_band": derived["kelvin_hours_to_comfort_band"],
                        "kelvin_hours_to_active_reference": derived["setpoint_kelvin_hours"],
                        **raw_load,
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
    cols = [
        *CITYLEARN_KPI_COLUMNS,
        "kelvin_hours",
        "kelvin_hours_to_active_reference",
        "net_electricity_consumption_kwh",
        "electricity_import_kwh",
        "max_net_electricity_consumption_kwh",
        "ramping_up_kwh",
        "ramping_abs_kwh",
        "heating_demand_kwh",
        "heating_electricity_consumption_kwh",
        "overrides",
        "override_hours",
        "Tin_max",
        "tdyn_mean",
        "tdyn_final",
    ]
    return buildings.groupby(["month", "controller", "mode", "occupant"], as_index=False)[cols].mean(numeric_only=True)


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
                "discomfort_without": before["discomfort_proportion"],
                "discomfort_bspline": after["discomfort_proportion"],
                "delta_discomfort_proportion": after["discomfort_proportion"] - before["discomfort_proportion"],
                "electricity_consumption_without": before["electricity_consumption_total"],
                "electricity_consumption_bspline": after["electricity_consumption_total"],
                "delta_electricity_consumption_total": after["electricity_consumption_total"] - before["electricity_consumption_total"],
            }
        )
    return pd.DataFrame(rows)


def district_frames(month: str) -> list[pd.DataFrame]:
    frames = []
    for controller in CONTROLLERS_WITH_BASELINE:
        for mode in DISPLAY_MODES:
            if not mode_available(controller["key"], month, mode):
                continue
            occupants = [None] if mode == "no_occupant" else OCCUPANTS
            for occ in occupants:
                loads = []
                t0 = None
                for building in BUILDINGS:
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
                occ_label = "none" if occ is None else occ.replace("_", " ")
                frames.append(
                    pd.DataFrame(
                        {
                            "month": MONTHS[month]["display"],
                            "time_step": np.asarray(t0[:n], dtype=int),
                            "load": np.sum([x[:n] for x in loads], axis=0),
                            "controller": controller["display"],
                            "mode": mode,
                            "occupant": "none" if occ is None else occ,
                            "series": f"{controller['display']} | {mode} | {occ_label}",
                        }
                    )
                )
    return frames


def rbc_district_frames(month: str) -> list[pd.DataFrame]:
    frames = []
    for mode in ("no_occupant", "without_tdyn", "bspline_tdyn"):
        occupants = [None] if mode == "no_occupant" else OCCUPANTS
        for occ in occupants:
            case = rbc_case(month, mode, occ)
            if case is None:
                continue
            loads = []
            t0 = None
            for building in BUILDINGS:
                root = raw_dir(RBC_CONTROLLER["source"], building["result_id"], case, case_scenario(mode))
                if not (root / "rollout.csv").exists():
                    continue
                roll = rollout(RBC_CONTROLLER["source"], building, case, case_scenario(mode))
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
            occ_label = "none" if occ is None else occ.replace("_", " ")
            frames.append(
                pd.DataFrame(
                    {
                        "month": MONTHS[month]["display"],
                        "time_step": np.asarray(t0[:n], dtype=int),
                        "load": np.sum([x[:n] for x in loads], axis=0),
                        "controller": RBC_CONTROLLER["display"],
                        "mode": mode,
                        "occupant": "none" if occ is None else occ,
                        "series": f"{RBC_CONTROLLER['display']} | {mode} | {occ_label}",
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
                "occupant": frame["occupant"].iloc[0],
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
            if not mode_available("rlmpc_adaptive", month, mode):
                continue
            occupants = [None] if mode == "no_occupant" else OCCUPANTS
            for occ in occupants:
                for building in BUILDINGS:
                    controller = next(item for item in CONTROLLERS if item["key"] == "rlmpc_adaptive")
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
                            "update_type": str(final.get("update_type", "")),
                            "ppo_loss_mean": float(pd.to_numeric(hist.get("loss"), errors="coerce").mean())
                            if "loss" in hist.columns
                            else np.nan,
                            "ppo_loss_final": num(final, "loss"),
                            "train_rmse_before": num(initial, "train_rmse_c"),
                            "train_rmse_after": num(final, "train_rmse_c"),
                            "train_rmse_delta": num(final, "train_rmse_c") - num(initial, "train_rmse_c"),
                            "val_rmse_before": num(initial, "val_rmse_c"),
                            "val_rmse_after": num(final, "val_rmse_c"),
                            "val_rmse_delta": num(final, "val_rmse_c") - num(initial, "val_rmse_c"),
                            "C_before": num(initial, "param_C"),
                            "C_after": num(final, "param_C"),
                            "C_delta": num(final, "param_C") - num(initial, "param_C"),
                            "Rm_delta": num(final, "param_Rm") - num(initial, "param_Rm"),
                            "Rout_delta": num(final, "param_Rout") - num(initial, "param_Rout"),
                            "Aeff_delta": num(final, "param_Aeff") - num(initial, "param_Aeff"),
                            "Pnom_before": num(initial, "param_Pnom"),
                            "Pnom_after": num(final, "param_Pnom"),
                            "Pnom_delta": num(final, "param_Pnom") - num(initial, "param_Pnom"),
                            "comfort_track_weight": num(final, "param_balanced_comfort_track_weight"),
                            "comfort_cold_weight": num(final, "param_balanced_comfort_cold_weight"),
                            "comfort_hot_weight": num(final, "param_balanced_comfort_hot_weight"),
                        }
                    )
    return pd.DataFrame(rows)


def adaptive_gnurl_delta_table(all_buildings: pd.DataFrame) -> pd.DataFrame:
    """Compare fixed GNU-RL against adaptive GNU-RL on matched building cases."""

    fixed_name = "GNU-RL fixed model"
    adaptive_name = "GNU-RL adaptive model"
    metrics = [
        "discomfort_proportion",
        "discomfort_cold_proportion",
        "discomfort_hot_proportion",
        "electricity_consumption_total",
        "daily_peak_average",
        "ramping_average",
        "kelvin_hours_to_active_reference",
        "Tin_max",
    ]
    index_cols = ["month", "mode", "occupant", "building", "building_id"]
    subset = all_buildings[all_buildings["controller"].isin([fixed_name, adaptive_name])].copy()
    if subset.empty:
        return pd.DataFrame()

    value_cols = [col for col in metrics if col in subset.columns]
    wide = subset.pivot_table(index=index_cols, columns="controller", values=value_cols, aggfunc="first")
    if wide.empty:
        return pd.DataFrame()

    rows = wide.reset_index()
    out = rows[index_cols].copy()
    short_names = {
        "discomfort_proportion": "disc",
        "discomfort_cold_proportion": "cold",
        "discomfort_hot_proportion": "hot",
        "electricity_consumption_total": "energy",
        "daily_peak_average": "daily_peak",
        "ramping_average": "ramping",
        "kelvin_hours_to_active_reference": "kh",
        "Tin_max": "Tin_max",
    }
    for metric in value_cols:
        if (metric, fixed_name) not in rows.columns or (metric, adaptive_name) not in rows.columns:
            continue
        stem = short_names.get(metric, metric)
        out[f"{stem}_fixed"] = rows[(metric, fixed_name)]
        out[f"{stem}_adaptive"] = rows[(metric, adaptive_name)]
        out[f"{stem}_delta_adaptive_minus_fixed"] = rows[(metric, adaptive_name)] - rows[(metric, fixed_name)]

    if "disc_delta_adaptive_minus_fixed" in out.columns:
        out["disc_improve_pp_fixed_minus_adaptive"] = -out["disc_delta_adaptive_minus_fixed"]
    if "kh_delta_adaptive_minus_fixed" in out.columns:
        out["kh_improve_fixed_minus_adaptive"] = -out["kh_delta_adaptive_minus_fixed"]
    return out


def compact_adaptive_delta_summary(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty or "disc_improve_pp_fixed_minus_adaptive" not in delta.columns:
        return pd.DataFrame()
    rows = []
    for key, group in delta.groupby(["month", "mode"], sort=False):
        improve = 100.0 * pd.to_numeric(group["disc_improve_pp_fixed_minus_adaptive"], errors="coerce").dropna()
        if improve.empty:
            continue
        rows.append(
            {
                "month": key[0],
                "mode": key[1],
                "cases": int(len(improve)),
                "better": int((improve > 0.0).sum()),
                "worse": int((improve < 0.0).sum()),
                "mean_improve_pp": float(improve.mean()),
                "median_improve_pp": float(improve.median()),
                "p25": float(improve.quantile(0.25)),
                "p75": float(improve.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def make_adaptive_delta_plots(delta: pd.DataFrame) -> dict[str, Path]:
    if delta.empty or "disc_improve_pp_fixed_minus_adaptive" not in delta.columns:
        return {}
    bins = np.array([-np.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf])
    labels = ["<-2", "-2:-1", "-1:-0.5", "-0.5:0", "0:0.5", "0.5:1", "1:2", ">2"]
    groups = [
        ("February", "no_occupant", "No occupant\nFeb", "#4d4d4d"),
        ("March", "no_occupant", "No occupant\nMar", "#7f7f7f"),
        ("February", "bspline_tdyn", "B-spline Tdyn\nFeb", "#2ca02c"),
        ("March", "bspline_tdyn", "B-spline Tdyn\nMar", "#74c476"),
    ]

    fig, ax = plt.subplots(figsize=(15.6, 6.2))
    x = np.arange(len(labels), dtype=float)
    width = 0.18
    offset0 = -0.5 * width * (len(groups) - 1)
    for i, (month_name, mode, label, color) in enumerate(groups):
        values = 100.0 * pd.to_numeric(
            delta.loc[
                (delta["month"] == month_name) & (delta["mode"] == mode),
                "disc_improve_pp_fixed_minus_adaptive",
            ],
            errors="coerce",
        ).dropna()
        counts, _ = np.histogram(values, bins=bins)
        ax.bar(
            x + offset0 + i * width,
            counts,
            width=width,
            color=color,
            alpha=0.88,
            label=label,
        )
    ax.axvline(3.5, color="black", lw=1.0, alpha=0.55)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Discomfort improvement from PPO dynamics [pp], fixed - adaptive", fontsize=18)
    ax.set_ylabel("Matched building cases", fontsize=18)
    ax.set_title("GNU-RL adaptive PPO dynamics: discomfort change vs fixed model", fontsize=20)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(
        loc="upper left",
        ncol=4,
        fontsize=16,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.92,
    )
    fig.tight_layout()
    path = FIG_DIR / "gnurl_adaptive_ppo_discomfort_delta_hist_combined.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"February/March": path}


def comfort_series_for_case(
    controller: dict,
    month: str,
    mode: str,
    occupant_name: str | None,
    building: dict,
) -> dict[str, np.ndarray] | None:
    occ = None if mode == "no_occupant" else occupant_name
    case = available_case(controller, month, mode, occ, building)
    if case is None:
        return None
    roll = rollout(controller["source"], building, case, case_scenario(mode))
    tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    ref = ref_setpoint(roll)
    tdyn = tdyn_setpoint(roll, occ)
    n = min(len(tin), len(ref), len(tdyn) if tdyn is not None else len(tin))
    data = {
        "Tin": tin[:n],
        "ref": ref[:n],
    }
    if tdyn is not None:
        data["Tdyn"] = tdyn[:n]
    return data


def plot_comfort_grid_axis(
    ax,
    controller: dict,
    mode: str,
    occupant_name: str | None,
    building: dict,
    show_title: bool = True,
) -> None:
    offset = 0
    plotted = False
    boundary = None
    for month in ("feb", "mar"):
        data = comfort_series_for_case(controller, month, mode, occupant_name, building)
        if data is None:
            continue
        n = len(data["Tin"])
        x = np.arange(n, dtype=float) + offset
        ax.plot(x, data["Tin"], lw=0.6, color="steelblue", label="Tin" if not plotted else None)
        ax.plot(x, data["ref"], lw=0.6, color="tab:orange", label="setpoint" if not plotted else None)
        if "Tdyn" in data:
            ax.plot(x, data["Tdyn"], lw=0.6, color="black", label="Tdyn" if not plotted else None)
        plotted = True
        offset += n
        if month == "feb":
            boundary = offset
    if not plotted:
        ax.set_axis_off()
        return
    if boundary is not None and offset > boundary:
        ax.axvline(boundary - 0.5, color="0.25", lw=0.65, ls="--", alpha=0.55)
    if show_title:
        ax.set_title(building["display"], fontsize=6.5)
    ax.grid(alpha=0.18)


def make_comfort_grid_combined(controller: dict, mode: str, occupant_name: str | None) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(23, 15), sharex=False)
    axes = np.asarray(axes)
    for bidx, building in enumerate(BUILDINGS):
        row = bidx // 5
        col = bidx % 5
        ax = axes[row, col]
        plot_comfort_grid_axis(ax, controller, mode, occupant_name, building)
        if col == 0:
            ax.set_ylabel("Temperature [C]", fontsize=8)
        if row == 4:
            ax.set_xlabel("Feb + Mar time step", fontsize=7)
        else:
            ax.set_xlabel("")
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=7)
    occ_title = "none" if occupant_name is None else occupant_name
    fig.suptitle(
        f"{controller['display']} {mode} | {occ_title} | continuous February to March comfort",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    occ_slug = "none" if occupant_name is None else occupant_name
    path = FIG_DIR / f"feb_mar_{controller['key']}_{mode}_{occ_slug}_comfort_all25.png"
    fig.savefig(path, dpi=135, bbox_inches="tight")
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


def make_override_plot(month: str, buildings: pd.DataFrame) -> Path:
    data = buildings[
        (buildings["month"] == MONTHS[month]["display"])
        & (buildings["controller"].isin(THESIS_CONTROLLERS))
        & (buildings["occupant"].isin(OCCUPANTS))
        & (buildings["mode"].isin(["without_tdyn", "bspline_tdyn"]))
    ].copy()
    agg = (
        data.groupby(["controller", "mode", "occupant"], as_index=False)["overrides"]
        .sum()
        .pivot_table(
            index=["occupant", "controller"],
            columns="mode",
            values="overrides",
            aggfunc="first",
        )
        .reset_index()
    )

    controller_order = THESIS_CONTROLLERS
    mode_colors = {"without_tdyn": "#4c78a8", "bspline_tdyn": "#f58518"}
    value_cols = [col for col in ["without_tdyn", "bspline_tdyn"] if col in agg.columns]
    y_max = float(np.nanmax(agg[value_cols].to_numpy(dtype=float))) if value_cols else 1.0
    y_upper = max(1000.0, np.ceil(y_max / 1000.0) * 1000.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
    axes = axes.ravel()
    for ax, occupant_name in zip(axes, OCCUPANTS):
        sub = agg[agg["occupant"] == occupant_name].set_index("controller")
        x = np.arange(len(controller_order), dtype=float)
        without_values = np.array(
            [
                float(sub.loc[ctrl, "without_tdyn"])
                if ctrl in sub.index and "without_tdyn" in sub.columns and pd.notna(sub.loc[ctrl, "without_tdyn"])
                else np.nan
                for ctrl in controller_order
            ],
            dtype=float,
        )
        bspline_values = np.array(
            [
                float(sub.loc[ctrl, "bspline_tdyn"])
                if ctrl in sub.index and "bspline_tdyn" in sub.columns and pd.notna(sub.loc[ctrl, "bspline_tdyn"])
                else np.nan
                for ctrl in controller_order
            ],
            dtype=float,
        )
        ax.bar(
            x - 0.18,
            without_values,
            width=0.34,
            color=mode_colors["without_tdyn"],
            label="without Tdyn",
        )
        ax.bar(
            x + 0.18,
            bspline_values,
            width=0.34,
            color=mode_colors["bspline_tdyn"],
            label="B-spline Tdyn",
        )
        ax.set_title(HIST_LABELS.get(occupant_name, occupant_name), fontsize=17)
        ax.set_xticks(x, [THESIS_CONTROLLER_LABELS[c] for c in controller_order], rotation=18)
        ax.tick_params(axis="both", labelsize=13.5)
        ax.set_ylim(0, y_upper)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(500))
        ax.grid(axis="y", which="major", alpha=0.25)
        ax.grid(axis="y", which="minor", alpha=0.12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=14)
    fig.supylabel("Total override events", fontsize=15, x=0.015)
    fig.suptitle(f"Occupant override events by controller - {MONTHS[month]['display']}", fontsize=18)
    fig.tight_layout(rect=[0.035, 0.08, 1, 0.94])
    path = FIG_DIR / f"{month}_override_counts.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def heatmap_case(controller: dict, month: str, mode: str, occupant_name: str | None, building: dict) -> str | None:
    if controller["key"] == "rbc_safepi":
        case = rbc_case(month, mode, occupant_name)
        if case is None:
            return None
        root = raw_dir(controller["source"], building["result_id"], case, case_scenario(mode))
        return case if (root / "rollout.csv").exists() else None
    if mode == "no_occupant":
        occupant_name = None
    return available_case(controller, month, mode, occupant_name, building)


def tin_minus_actual_setpoint_values(controller: dict, month: str, mode: str, building: dict) -> np.ndarray:
    occupants = [None] if mode == "no_occupant" else OCCUPANTS
    values = []
    for occ in occupants:
        case = heatmap_case(controller, month, mode, occ, building)
        if case is None:
            continue
        roll = rollout(controller["source"], building, case, case_scenario(mode))
        if "indoor_temperature" not in roll.columns:
            continue
        tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        setpoint = effective_setpoint(roll)
        n = min(len(tin), len(setpoint))
        if n == 0:
            continue
        delta = tin[:n] - setpoint[:n]
        delta = delta[np.isfinite(delta)]
        if len(delta):
            values.append(delta)
    if not values:
        return np.asarray([], dtype=float)
    return np.concatenate(values)


def delta_frequency_matrix(
    controller: dict,
    month: str,
    mode: str,
    *,
    bins: np.ndarray,
    clip_limit: float,
) -> np.ndarray:
    matrix = np.zeros((len(bins) - 1, len(BUILDINGS)), dtype=float)
    for b_idx, building in enumerate(BUILDINGS):
        delta = tin_minus_actual_setpoint_values(controller, month, mode, building)
        if len(delta) == 0:
            matrix[:, b_idx] = np.nan
            continue
        clipped = np.clip(delta, -clip_limit, clip_limit)
        counts, _ = np.histogram(clipped, bins=bins)
        total = max(float(counts.sum()), 1.0)
        matrix[:, b_idx] = 100.0 * counts / total
    return matrix


def make_tin_setpoint_heatmaps(month: str = "mar") -> dict[str, Path]:
    paths: dict[str, Path] = {}
    clip_limit = 6.0
    bin_step = 0.25
    bins = np.arange(-clip_limit, clip_limit + bin_step, bin_step)
    centers = 0.5 * (bins[:-1] + bins[1:])
    y_ticks = np.arange(-clip_limit, clip_limit + 0.01, 1.0)
    building_labels = [b["display"].replace("Building ", "B") for b in BUILDINGS]

    for controller in HEATMAP_CONTROLLERS:
        no_occ = delta_frequency_matrix(controller, month, "no_occupant", bins=bins, clip_limit=clip_limit)
        tdyn = delta_frequency_matrix(controller, month, "bspline_tdyn", bins=bins, clip_limit=clip_limit)
        vmax = np.nanmax([np.nanmax(no_occ), np.nanmax(tdyn)])
        if not np.isfinite(vmax) or vmax <= 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.2), sharey=True)
        panels = [
            ("LoD 2: no occupant", no_occ),
            ("LoD 3: B-spline Tdyn occupants", tdyn),
        ]
        image = None
        for ax, (title, matrix) in zip(axes, panels):
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="Blues",
                vmin=0.0,
                vmax=vmax,
                extent=(-0.5, len(BUILDINGS) - 0.5, bins[0], bins[-1]),
            )
            ax.axhline(0.0, color="0.25", lw=0.9)
            ax.axhline(2.0, color="0.55", lw=0.6, ls="--")
            ax.axhline(-2.0, color="0.55", lw=0.6, ls="--")
            ax.set_title(title)
            ax.set_xlabel("Building ID")
            ax.set_xticks(np.arange(len(BUILDINGS)))
            ax.set_xticklabels(building_labels, rotation=90, fontsize=7)
            ax.set_yticks(y_ticks)
            ax.grid(color="white", lw=0.25, alpha=0.55)
        axes[0].set_ylabel(r"$T_{in} - T_{setpoint}$ [$^\circ$C]")
        if image is not None:
            cbar_ax = fig.add_axes([0.925, 0.22, 0.016, 0.56])
            cbar = fig.colorbar(image, cax=cbar_ax)
            cbar.set_label("Frequency share per building [%]")
        fig.suptitle(
            f"{controller['display']} | {MONTHS[month]['display']} | indoor temperature minus actual setpoint",
            fontsize=13,
        )
        fig.text(
            0.5,
            0.01,
            "Edge bins include values clipped beyond +/-6 C. LoD 3 aggregates occ1..occ4 for each building.",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.055, right=0.895, bottom=0.16, top=0.84, wspace=0.055)
        slug = controller["display"].lower().replace("/", "_").replace(" ", "_").replace("-", "_")
        path = FIG_DIR / f"{month}_{slug}_tin_minus_actual_setpoint_heatmap.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[controller["display"]] = path
    return paths


def sac_training_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    reward_rows = []
    loss_rows = []
    cases = [
        ("February deploy", "no_occupant", "none", f"{PREFIX}_sac_jan20_feb_no_occupant"),
        ("March deploy", "no_occupant", "none", f"{PREFIX}_sac_feb10_mar_no_occupant"),
    ]
    for occupant_name in OCCUPANTS:
        for mode in MODES:
            cases.extend(
                [
                    ("February deploy", mode, occupant_name, f"{PREFIX}_sac_jan20_feb_{occupant_name}_{mode}"),
                    ("March deploy", mode, occupant_name, f"{PREFIX}_sac_feb10_mar_{occupant_name}_{mode}"),
                ]
            )
    for phase, mode, occupant_name, case in cases:
        scenario = case_scenario(mode)
        for building in BUILDINGS:
            case_dir = raw_dir("sac_merlin", building["result_id"], case, scenario)
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
                    grp["occupant"] = occupant_name
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
                    grp["occupant"] = occupant_name
                    grp["building"] = building["display"]
                    loss_rows.append(grp)
    rewards = pd.concat(reward_rows, ignore_index=True) if reward_rows else pd.DataFrame()
    losses = pd.concat(loss_rows, ignore_index=True) if loss_rows else pd.DataFrame()
    if not rewards.empty:
        rewards = rewards.groupby(["phase", "mode", "occupant", "episode"], as_index=False).mean(numeric_only=True)
    if not losses.empty:
        losses = losses.groupby(["phase", "mode", "occupant", "episode"], as_index=False).mean(numeric_only=True)
    return rewards, losses


def make_sac_training_plots(rewards: pd.DataFrame, losses: pd.DataFrame) -> tuple[Path | None, Path | None]:
    reward_path = None
    if not rewards.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.25))
        reward_data = rewards[
            (rewards["mode"] == "no_occupant") & (rewards["occupant"] == "none")
        ].copy()
        for phase, group in reward_data.groupby("phase", sort=False):
            ax.plot(
                group["episode"],
                group["raw_reward_mean"],
                marker="o",
                lw=1.5,
                label=phase,
            )
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_title("SAC no-occupant reward by episode", fontsize=11)
        ax.set_xlabel("Training episode", fontsize=10)
        ax.set_ylabel("Mean raw reward per step", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, ncol=1, frameon=False)
        fig.tight_layout()
        reward_path = FIG_DIR / "sac_no_occupant_online_reward_by_episode.png"
        fig.savefig(reward_path, dpi=260, bbox_inches="tight")
        plt.close(fig)

    loss_path = None
    if not losses.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for (phase, mode, occupant_name), group in losses.groupby(["phase", "mode", "occupant"], sort=False):
            label = f"{phase} | {mode} | {occupant_name}"
            axes[0].plot(group["episode"], group["soft_q1_loss_mean"], marker="o", lw=1.0, label=label)
            axes[1].plot(group["episode"], group["policy_loss_mean"], marker="o", lw=1.0, label=label)
        axes[0].set_title("Soft Q1 loss")
        axes[1].set_title("Policy loss")
        for ax in axes:
            ax.set_xlabel("Training episode")
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=6, ncol=2)
        fig.suptitle("SAC loss means by episode")
        fig.tight_layout()
        loss_path = FIG_DIR / "sac_losses_by_episode.png"
        fig.savefig(loss_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    return reward_path, loss_path


HIST_METRIC_GROUPS = {
    "load": [
        ("electricity_consumption_total", "Imported-electricity score"),
        ("zero_net_energy", "Zero-net-energy score"),
        ("daily_peak_average", "Average daily peak score"),
        ("all_time_peak_average", "Maximum peak demand score"),
        ("ramping_average", "Up-ramping score"),
        ("daily_one_minus_load_factor_average", "Daily one-minus LF score"),
        ("monthly_one_minus_load_factor_average", "Monthly one-minus LF score"),
        ("net_electricity_consumption_kwh", "Building net electricity [kWh]"),
        ("heating_demand_kwh", "Input heating demand [kWh]"),
        ("heating_electricity_consumption_kwh", "HVAC heating electricity [kWh]"),
        ("ramping_abs_kwh", "Absolute ramping [kWh]"),
    ],
    "comfort": [
        ("discomfort_proportion", "Comfort cost [%]"),
        ("discomfort_cold_proportion", "Cold discomfort [%]"),
        ("discomfort_hot_proportion", "Hot discomfort [%]"),
        ("discomfort_cold_delta_average", "Mean cold violation [C]"),
        ("discomfort_hot_delta_average", "Mean hot violation [C]"),
        ("kelvin_hours", "Kelvin-hours, active reference"),
        ("kelvin_hours_to_comfort_band", "Kelvin-hours, comfort band"),
        ("kelvin_hours_to_active_reference", "Kelvin-hours, active reference"),
        ("Tin_max", "Max Tin [C]"),
    ],
    "overrides": [
        ("overrides", "Override events"),
        ("override_hours", "Override hours"),
    ],
}


HIST_COLORS = {
    "Baseline": "#4d4d4d",
    "MPC": "#1f77b4",
    "GNU-RL fixed model": "#ff7f0e",
    "GNU-RL adaptive model": "#2ca02c",
    "SAC": "#d62728",
    "Safe-PI/RBC": "#9467bd",
    "occ1_tolerant": "#1f77b4",
    "occ2_sensitive": "#ff7f0e",
    "occ3_cold": "#2ca02c",
    "occ4_hot": "#d62728",
}


HIST_LABELS = {
    "Baseline": "Baseline",
    "GNU-RL fixed model": "GNU fixed",
    "GNU-RL adaptive model": "GNU adapt",
    "Safe-PI/RBC": "Safe-PI",
    "occ1_tolerant": "occ1 tolerant",
    "occ2_sensitive": "occ2 sensitive",
    "occ3_cold": "occ3 cold",
    "occ4_hot": "occ4 hot",
}


def format_bin_edge(value: float) -> str:
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs_value >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def histogram_bins(groups: list[pd.Series], default_bins: int = 7) -> tuple[np.ndarray, list[str], float | None]:
    values = pd.concat([g.dropna() for g in groups], ignore_index=True) if groups else pd.Series(dtype=float)
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        edges = np.linspace(0.0, 1.0, default_bins + 1)
        labels = [f"{format_bin_edge(a)}-{format_bin_edge(b)}" for a, b in zip(edges[:-1], edges[1:])]
        return edges, labels, None
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-3)
        lower = max(0.0, vmin - pad) if vmin >= 0.0 else vmin - pad
        edges = np.linspace(lower, vmax + pad, 4)
        labels = [f"{format_bin_edge(a)}-{format_bin_edge(b)}" for a, b in zip(edges[:-1], edges[1:])]
        return edges, labels, None

    upper_source = values.to_numpy(dtype=float)
    q1, q3 = np.quantile(upper_source, [0.25, 0.75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr if iqr > 0 else vmax
    overflow_start = None
    if vmax > fence:
        inlier = upper_source[upper_source <= fence]
        if len(inlier) >= max(5, int(0.75 * len(upper_source))):
            overflow_start = float(inlier.max())

    pad = 0.04 * (vmax - vmin)
    lower = max(0.0, vmin - pad) if vmin >= 0.0 else vmin - pad
    upper = (overflow_start if overflow_start is not None else vmax) + pad
    locator = mticker.MaxNLocator(nbins=default_bins, steps=[1, 2, 2.5, 5, 10])
    bins = np.asarray(locator.tick_values(lower, upper), dtype=float)
    bins = bins[(bins >= lower) & (bins <= upper)]
    required_upper = overflow_start if overflow_start is not None else vmax
    if len(bins) < 3 or bins[0] > vmin or bins[-1] < required_upper:
        bins = np.asarray(locator.tick_values(lower, upper), dtype=float)
    bins = np.unique(bins)
    if overflow_start is not None:
        overflow_start = float(bins[-1])
        if overflow_start >= vmax:
            overflow_start = None
    labels = [f"{format_bin_edge(a)}-{format_bin_edge(b)}" for a, b in zip(bins[:-1], bins[1:])]
    if overflow_start is not None:
        labels.append(f">= {format_bin_edge(overflow_start)}")
    return bins, labels, overflow_start


def plot_hist_grid(
    *,
    data: pd.DataFrame,
    compare_col: str,
    compare_values: list[str],
    metric_group: str,
    title: str,
    path: Path,
) -> Path:
    metrics = HIST_METRIC_GROUPS[metric_group]
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.6, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (metric, metric_label) in zip(axes_flat, metrics):
        values_by_group = {
            value: pd.to_numeric(data.loc[data[compare_col] == value, metric], errors="coerce").dropna()
            for value in compare_values
            if metric in data.columns
        }
        ordered_values = [(value, values) for value, values in values_by_group.items() if not values.empty]
        plotted = False
        if ordered_values:
            n_groups = len(ordered_values)
            bins, bin_labels, overflow_start = histogram_bins([values for _, values in ordered_values], default_bins=7)
            x = np.arange(len(bin_labels), dtype=float)
            bar_width = 1.0 / max(n_groups, 1)
            for group_idx, (value, values) in enumerate(ordered_values):
                raw = values.to_numpy(dtype=float)
                regular_raw = raw[raw <= overflow_start] if overflow_start is not None else raw
                counts, _ = np.histogram(regular_raw, bins=bins)
                if overflow_start is not None:
                    counts = np.append(counts, np.count_nonzero(raw > overflow_start))
                positions = x - 0.5 + (group_idx + 0.5) * bar_width
                color = HIST_COLORS.get(value, "0.25")
                ax.bar(
                    positions,
                    counts,
                    width=bar_width,
                    color=color,
                    alpha=0.88,
                    edgecolor=color,
                    linewidth=0.0,
                    label=HIST_LABELS.get(value, value),
                )
                plotted = True
        if not plotted:
            ax.set_axis_off()
            continue
        ax.set_title(metric_label, fontsize=10)
        ax.set_ylabel("Buildings")
        ax.set_xlabel(metric_label)
        ax.set_xlim(-0.5, len(bin_labels) - 0.5)
        ax.set_xticks(np.arange(len(bin_labels)), bin_labels)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncols=min(2, n_groups), loc="upper right")
    for ax in axes_flat[len(metrics):]:
        ax.set_axis_off()
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.035, 1, 0.95), h_pad=1.6, w_pad=1.15)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_kpi_histograms(all_buildings: pd.DataFrame) -> dict[tuple[str, str, str, str], Path]:
    paths: dict[tuple[str, str, str, str], Path] = {}
    for month in MONTHS.values():
        month_name = month["display"]
        no_occ = all_buildings[
            (all_buildings["month"] == month_name)
            & (all_buildings["mode"] == "no_occupant")
            & (all_buildings["occupant"] == "none")
        ].copy()
        no_occ_controllers = [name for name in CONTROLLER_ORDER if name in set(no_occ["controller"])]
        for group_name in ("load", "comfort"):
            if no_occ.empty:
                continue
            path = FIG_DIR / f"{month_name.lower()}_no_occupant_{group_name}_hist_by_controller.png"
            paths[(month_name, "no_occupant", "controllers", group_name)] = plot_hist_grid(
                data=no_occ,
                compare_col="controller",
                compare_values=no_occ_controllers,
                metric_group=group_name,
                title=f"{month_name} no-occupant KPI distributions by controller",
                path=path,
            )
        for mode in MODES:
            data = all_buildings[
                (all_buildings["month"] == month_name)
                & (all_buildings["mode"] == mode)
                & (all_buildings["occupant"].isin(OCCUPANTS))
            ].copy()
            if data.empty:
                continue
            for controller_name in ["Safe-PI/RBC", "MPC", "GNU-RL fixed model", "GNU-RL adaptive model", "SAC"]:
                subset = data[data["controller"] == controller_name].copy()
                if subset.empty:
                    continue
                for group_name in ("load", "comfort", "overrides"):
                    path = FIG_DIR / (
                        f"{month_name.lower()}_{mode}_{controller_name.lower().replace('/', '_').replace(' ', '_')}"
                        f"_{group_name}_hist_by_occupant.png"
                    )
                    paths[(month_name, mode, controller_name, group_name)] = plot_hist_grid(
                        data=subset,
                        compare_col="occupant",
                        compare_values=OCCUPANTS,
                        metric_group=group_name,
                        title=f"{month_name} {mode} | {controller_name} | KPI distributions by occupant",
                        path=path,
                    )
    return paths


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


def table_display_large(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for label_col in ("controller", "ctrl"):
        if label_col in out.columns:
            out[label_col] = out[label_col].replace(CONTROLLER_SHORT)
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    html = out.to_html(index=False, escape=True, border=0, classes="ce1-table")
    return (
        '<div class="ce1-table-wrap" '
        'style="max-width:100%;max-height:680px;overflow:auto;margin:0.25rem 0 0.9rem 0;">'
        f"{html}</div>"
    )


KPI_COLUMNS_FULL = [
    *CITYLEARN_KPI_COLUMNS,
    "kelvin_hours",
    "kelvin_hours_to_comfort_band",
    "kelvin_hours_to_active_reference",
    "net_electricity_consumption_kwh",
    "electricity_import_kwh",
    "max_net_electricity_consumption_kwh",
    "ramping_up_kwh",
    "ramping_abs_kwh",
    "heating_demand_kwh",
    "heating_electricity_consumption_kwh",
    "overrides",
    "override_hours",
    "reward_total",
    "Tin_max",
    "tdyn_mean",
    "tdyn_final",
]


CONTROLLER_ORDER = [
    "Baseline",
    "Safe-PI/RBC",
    "MPC",
    "GNU-RL fixed model",
    "GNU-RL adaptive model",
    "SAC",
]


def full_building_kpi_table(all_buildings: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "month",
        "mode",
        "occupant",
        "building",
        "building_id",
        "controller",
        *KPI_COLUMNS_FULL,
        "source",
        "case",
    ]
    out = all_buildings[[col for col in cols if col in all_buildings.columns]].copy()
    controller_rank = {name: idx for idx, name in enumerate(CONTROLLER_ORDER)}
    out["_controller_rank"] = out["controller"].map(controller_rank).fillna(len(controller_rank))
    out = out.sort_values(["month", "mode", "occupant", "building_id", "_controller_rank"])
    return out.drop(columns=["_controller_rank"])


def wide_building_kpi_table(all_buildings: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["month", "mode", "occupant", "building", "building_id"]
    value_cols = [col for col in KPI_COLUMNS_FULL if col in all_buildings.columns]
    data = all_buildings[index_cols + ["controller", *value_cols]].copy()
    pivot = data.pivot_table(index=index_cols, columns="controller", values=value_cols, aggfunc="first")
    pieces = []
    for metric in value_cols:
        metric_frame = pivot[metric] if metric in pivot.columns.get_level_values(0) else pd.DataFrame(index=pivot.index)
        ordered = [controller for controller in CONTROLLER_ORDER if controller in metric_frame.columns]
        extra = [controller for controller in metric_frame.columns if controller not in ordered]
        metric_frame = metric_frame[ordered + extra]
        metric_frame = metric_frame.rename(columns={controller: f"{metric} | {HIST_LABELS.get(controller, controller)}" for controller in metric_frame.columns})
        pieces.append(metric_frame)
    out = pd.concat(pieces, axis=1).reset_index()
    return out.sort_values(index_cols)


def compact_mean_table(means: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "month",
        "controller",
        "mode",
        "occupant",
        "electricity_consumption_total",
        "zero_net_energy",
        "cost_total",
        "carbon_emissions_total",
        "daily_peak_average",
        "all_time_peak_average",
        "ramping_average",
        "daily_one_minus_load_factor_average",
        "monthly_one_minus_load_factor_average",
        "discomfort_proportion",
        "discomfort_cold_proportion",
        "discomfort_hot_proportion",
        "kelvin_hours",
        "kelvin_hours_to_comfort_band",
        "kelvin_hours_to_active_reference",
        "net_electricity_consumption_kwh",
        "heating_demand_kwh",
        "heating_electricity_consumption_kwh",
        "overrides",
        "override_hours",
        "Tin_max",
    ]
    available = [col for col in cols if col in means.columns]
    return means[available].rename(
        columns={
            "controller": "ctrl",
            "electricity_consumption_total": "electricity_consumption_total",
            "daily_peak_average": "daily_peak_average",
            "ramping_average": "ramping_average",
            "kelvin_hours": "kelvin_hours",
            "kelvin_hours_to_comfort_band": "KH_comfort_band",
            "kelvin_hours_to_active_reference": "KH_active_ref",
            "override_hours": "ovr_h",
            "overrides": "ovr",
        }
    )


def compact_district_table(districts: pd.DataFrame) -> pd.DataFrame:
    out = districts.copy()
    out["total_MWh"] = out["total_kwh"] / 1000.0
    cols = [
        "month",
        "controller",
        "mode",
        "occupant",
        "total_MWh",
        "peak_kwh",
        "daily_profile_peak_kwh",
        "daily_profile_peak_hour",
        "ramping_abs_kwh",
        "load_factor",
    ]
    return out[[col for col in cols if col in out.columns]].rename(
        columns={
            "controller": "ctrl",
            "peak_kwh": "peak",
            "daily_profile_peak_kwh": "day_peak",
            "daily_profile_peak_hour": "peak_hr",
            "ramping_abs_kwh": "ramp_kwh",
            "load_factor": "lf",
        }
    )


def compact_override_table(overrides: pd.DataFrame) -> pd.DataFrame:
    out = (
        overrides.groupby(["month", "controller", "occupant"], as_index=False)[
            [
                "overrides_without",
                "overrides_bspline",
                "delta_overrides",
                "delta_discomfort_proportion",
                "delta_electricity_consumption_total",
            ]
        ]
        .sum(numeric_only=True)
    )
    return out.rename(
        columns={
            "controller": "ctrl",
            "overrides_without": "ovr_no_tdyn",
            "overrides_bspline": "ovr_tdyn",
            "delta_overrides": "delta_ovr",
            "delta_discomfort_proportion": "delta_discomfort",
            "delta_electricity_consumption_total": "delta_electricity",
        }
    )


def compact_dynamics_summary(dynamics: pd.DataFrame) -> pd.DataFrame:
    if dynamics.empty:
        return pd.DataFrame()
    out = dynamics.groupby(["month", "mode", "occupant"], as_index=False).agg(
        ppo_loss=("ppo_loss_mean", "mean"),
        C_delta=("C_delta", "mean"),
        Rm_delta=("Rm_delta", "mean"),
        Rout_delta=("Rout_delta", "mean"),
        Aeff_delta=("Aeff_delta", "mean"),
        Pnom_delta=("Pnom_delta", "mean"),
        track_w=("comfort_track_weight", "mean"),
        cold_w=("comfort_cold_weight", "mean"),
        hot_w=("comfort_hot_weight", "mean"),
    )
    return out


def compact_atg_summary(atg_summary: pd.DataFrame) -> pd.DataFrame:
    if atg_summary.empty:
        return pd.DataFrame()
    cols = [
        "month",
        "controller",
        "mode",
        "occupant",
        "buildings",
        "ATG_A",
        "ATG_B",
        "ATG_C",
        "ATG_outside_C",
        "A_time_mean_pct",
        "B_time_mean_pct",
        "C_time_mean_pct",
        "outside_C_mean_pct",
        "mean_abs_delta_C",
    ]
    out = atg_summary[[col for col in cols if col in atg_summary.columns]].copy()
    return out.rename(
        columns={
            "controller": "ctrl",
            "buildings": "n",
            "ATG_outside_C": "ATG_out",
            "A_time_mean_pct": "A_time_%",
            "B_time_mean_pct": "B_time_%",
            "C_time_mean_pct": "C_time_%",
            "outside_C_mean_pct": "out_C_%",
            "mean_abs_delta_C": "mean_abs_dT",
        }
    )


def worst_building_preview(all_buildings: pd.DataFrame, month_name: str, n: int = 12) -> pd.DataFrame:
    cols = [
        "controller",
        "mode",
        "building",
        "occupant",
        "electricity_consumption_total",
        "discomfort_proportion",
        "discomfort_cold_proportion",
        "discomfort_hot_proportion",
        "overrides",
        "Tin_max",
    ]
    table = all_buildings[all_buildings["month"] == month_name].copy()
    table = table.sort_values("discomfort_proportion", ascending=False).head(n)
    return table[[col for col in cols if col in table.columns]].rename(
        columns={
            "controller": "ctrl",
            "building": "bldg",
            "occupant": "occ",
            "overrides": "ovr",
        }
    )


def representative_building_selection(all_buildings: pd.DataFrame) -> pd.DataFrame:
    data = all_buildings[
        all_buildings["mode"].eq("no_occupant")
        & all_buildings["occupant"].eq("none")
        & all_buildings["month"].isin(["February", "March"])
    ].dropna(subset=["discomfort_proportion"]).copy()
    if data.empty:
        return pd.DataFrame()
    summary = (
        data.groupby(["building_id", "building"], as_index=False)
        .agg(
            mean_discomfort_proportion=("discomfort_proportion", "mean"),
            median_discomfort_proportion=("discomfort_proportion", "median"),
            max_discomfort_proportion=("discomfort_proportion", "max"),
            mean_electricity_consumption_total=("electricity_consumption_total", "mean"),
            cases=("discomfort_proportion", "size"),
        )
        .sort_values(["mean_discomfort_proportion", "building_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    summary["discomfort_rank"] = np.arange(1, len(summary) + 1)
    median_value = float(summary["mean_discomfort_proportion"].median())

    selected = []
    for order, (label, reason, rows) in enumerate(
        (
            (
                "good",
                "low no-occupant discomfort and stable RC fit",
                summary[summary["building_id"].eq("building_12")],
            ),
            (
                "mid",
                "middle no-occupant comfort tail",
                summary[summary["building_id"].eq("building_15")],
            ),
            (
                "bad",
                "poor RC fit and poor no-occupant comfort for MPC and SAC",
                summary[summary["building_id"].eq("building_08")],
            ),
        ),
        start=1,
    ):
        rows = rows.copy()
        if rows.empty:
            rows = pd.DataFrame(
                {
                    "building_id": [f"building_{'12' if label == 'good' else '15' if label == 'mid' else '08'}"],
                    "building": [f"Building {'12' if label == 'good' else '15' if label == 'mid' else '8'}"],
                    "mean_discomfort_proportion": [np.nan],
                    "median_discomfort_proportion": [np.nan],
                    "max_discomfort_proportion": [np.nan],
                    "mean_electricity_consumption_total": [np.nan],
                    "cases": [0],
                    "discomfort_rank": [np.nan],
                }
            )
        rows["selection"] = label
        rows["selection_rank"] = order
        rows["selection_reason"] = reason
        rows["median_reference_discomfort_proportion"] = median_value
        selected.append(rows)
    return pd.concat(selected, ignore_index=True)


def compact_representative_building_selection(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty:
        return pd.DataFrame()
    out = selection[
        [
            "selection",
            "building",
            "discomfort_rank",
            "mean_discomfort_proportion",
            "median_discomfort_proportion",
            "max_discomfort_proportion",
            "mean_electricity_consumption_total",
            "cases",
            "selection_reason",
        ]
    ].copy()
    out["mean_disc_%"] = 100.0 * out.pop("mean_discomfort_proportion")
    out["median_disc_%"] = 100.0 * out.pop("median_discomfort_proportion")
    out["max_disc_%"] = 100.0 * out.pop("max_discomfort_proportion")
    out = out.rename(
        columns={
            "selection": "role",
            "building": "bldg",
            "discomfort_rank": "rank",
            "mean_electricity_consumption_total": "mean_energy",
            "selection_reason": "reason",
        }
    )
    return out[
        [
            "role",
            "bldg",
            "rank",
            "mean_disc_%",
            "median_disc_%",
            "max_disc_%",
            "mean_energy",
            "cases",
            "reason",
        ]
    ]


def selected_building_cases(all_buildings: pd.DataFrame, representative: pd.DataFrame) -> pd.DataFrame:
    if representative.empty:
        return pd.DataFrame()
    data = all_buildings.dropna(subset=["discomfort_proportion"]).copy()
    selected = representative[
        [
            "building_id",
            "selection",
            "selection_rank",
            "selection_reason",
            "discomfort_rank",
            "mean_discomfort_proportion",
            "median_discomfort_proportion",
            "max_discomfort_proportion",
        ]
    ].copy()
    out = data.merge(selected, on="building_id", how="inner")
    out = out.sort_values(
        ["month", "controller", "mode", "occupant", "selection_rank", "building_id"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)
    cols = [
        "month",
        "controller",
        "controller_key",
        "source",
        "case",
        "scenario",
        "mode",
        "occupant",
        "selection",
        "selection_rank",
        "selection_reason",
        "discomfort_rank",
        "mean_discomfort_proportion",
        "median_discomfort_proportion",
        "max_discomfort_proportion",
        "building",
        "building_id",
        *KPI_COLUMNS_FULL,
    ]
    return out[[col for col in cols if col in out.columns]]


def slug(text: str) -> str:
    chars = []
    for char in str(text).lower():
        chars.append(char if char.isalnum() else "_")
    return "_".join("".join(chars).split("_"))


def comfort_bounds_for_row(roll: pd.DataFrame, occupant_name: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(roll)
    if occupant_name != "none" and occupant_name in OCCUPANT_MODELS:
        low, high = occupant_probability_comfort_band(OCCUPANT_MODELS[occupant_name])
        return np.full(n, low, dtype=float), np.full(n, high, dtype=float)
    if "tmin" in roll.columns and "tmax" in roll.columns:
        low = pd.to_numeric(roll["tmin"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(roll["tmax"], errors="coerce").to_numpy(dtype=float)
        if len(low) == n and len(high) == n:
            return low, high
    return np.full(n, 20.0, dtype=float), np.full(n, 24.0, dtype=float)


def selected_rollout(row: pd.Series) -> pd.DataFrame:
    building = {"result_id": row["building_id"]}
    return rollout(str(row["source"]), building, str(row["case"]), str(row["scenario"]))


def plot_selected_comfort_axis(ax, rows: pd.DataFrame, mean_daily: bool = False) -> None:
    rows = rows.copy()
    rows["_month_order"] = rows["month"].map({"February": 0, "March": 1}).fillna(99)
    rows = rows.sort_values("_month_order")
    if rows.empty:
        ax.set_axis_off()
        return

    first_row = rows.iloc[0]
    offset = 0
    boundary = None
    plotted = False
    discomfort_parts = []
    for _, row in rows.iterrows():
        roll = selected_rollout(row)
        t = time(roll)
        month_name = str(row["month"])
        x_base = np.asarray(t, dtype=int) % 24 if mean_daily else np.arange(len(roll), dtype=int)
        x_base = x_base.astype(float)
        if mean_daily:
            x_base = x_base + (0.0 if month_name == "February" else 24.0)
        else:
            x_base = x_base + offset
        tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        ref = ref_setpoint(roll)
        tdyn = tdyn_setpoint(roll, None if row["occupant"] == "none" else str(row["occupant"]))
        low, high = comfort_bounds_for_row(roll, str(row["occupant"]))
        n = min(len(x_base), len(tin), len(ref), len(low), len(high), len(tdyn) if tdyn is not None else len(x_base))
        frame = pd.DataFrame(
            {
                "x": x_base[:n],
                "Tin": tin[:n],
                "ref": ref[:n],
                "low": low[:n],
                "high": high[:n],
            }
        )
        if tdyn is not None:
            frame["Tdyn"] = tdyn[:n]
        if mean_daily:
            frame = frame.groupby("x", as_index=False).mean(numeric_only=True)
        ax.fill_between(frame["x"], frame["low"], frame["high"], color="0.8", alpha=0.35, linewidth=0)
        ax.plot(frame["x"], frame["Tin"], lw=0.95, color="steelblue", label="Tin" if not plotted else None)
        ax.plot(frame["x"], frame["ref"], lw=0.85, color="tab:orange", label="setpoint" if not plotted else None)
        if "Tdyn" in frame.columns:
            ax.plot(frame["x"], frame["Tdyn"], lw=0.85, color="black", label="Tdyn" if not plotted else None)
        discomfort_pct = 100.0 * float(row.get("discomfort_proportion", np.nan))
        discomfort_parts.append(f"{month_name[:3]} {discomfort_pct:.1f}%")
        plotted = True
        if not mean_daily:
            offset += n
            if month_name == "February":
                boundary = offset

    if mean_daily:
        ax.axvline(23.5, color="0.25", lw=0.8, ls="--", alpha=0.55)
        ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 47])
        ax.set_xticklabels(["0", "6", "12", "18", "0", "6", "12", "18", "23"])
    elif boundary is not None and offset > boundary:
        ax.axvline(boundary - 0.5, color="0.25", lw=0.8, ls="--", alpha=0.55)

    role = str(first_row["selection"])
    rank = int(first_row.get("discomfort_rank", first_row["selection_rank"]))
    building_label = str(first_row["building"])
    if rows["building"].nunique() > 1:
        building_label = "/".join(rows["building"].astype(str).unique())
    ax.set_title(
        f"{role}: {building_label} | {'; '.join(discomfort_parts)} | global rank {rank}",
        fontsize=7,
    )
    ax.grid(alpha=0.2)


def plot_selected_single_month_axis(ax, row: pd.Series, mean_daily: bool = False) -> None:
    roll = selected_rollout(row)
    t = time(roll)
    tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    ref = ref_setpoint(roll)
    tdyn = tdyn_setpoint(roll, None if row["occupant"] == "none" else str(row["occupant"]))
    low, high = comfort_bounds_for_row(roll, str(row["occupant"]))
    n = min(len(t), len(tin), len(ref), len(low), len(high), len(tdyn) if tdyn is not None else len(t))
    frame = pd.DataFrame(
        {
            "x": (np.asarray(t[:n], dtype=int) % 24) if mean_daily else t[:n],
            "Tin": tin[:n],
            "ref": ref[:n],
            "low": low[:n],
            "high": high[:n],
        }
    )
    if tdyn is not None:
        frame["Tdyn"] = tdyn[:n]
    if mean_daily:
        frame = frame.groupby("x", as_index=False).mean(numeric_only=True)
    ax.fill_between(frame["x"], frame["low"], frame["high"], color="0.8", alpha=0.35, linewidth=0)
    ax.plot(frame["x"], frame["Tin"], lw=0.95, color="steelblue", label="Tin")
    ax.plot(frame["x"], frame["ref"], lw=0.85, color="tab:orange", label="setpoint")
    if "Tdyn" in frame.columns:
        ax.plot(frame["x"], frame["Tdyn"], lw=0.85, color="black", label="Tdyn")
    discomfort_pct = 100.0 * float(row.get("discomfort_proportion", np.nan))
    role = str(row["selection"])
    rank = int(row.get("discomfort_rank", row["selection_rank"]))
    ax.set_title(f"{role}: {row['building']} | case {discomfort_pct:.1f}% | global rank {rank}", fontsize=7)
    ax.grid(alpha=0.2)


def make_selected_comfort_plots(selected: pd.DataFrame) -> dict[tuple[str, str, str, str], Path]:
    paths: dict[tuple[str, str, str, str], Path] = {}
    if selected.empty:
        return paths
    group_cols = ["controller", "mode", "occupant"]
    for key, group in selected.groupby(group_cols, sort=False):
        controller_name, mode, occupant_name = key
        for kind, mean_daily in (("month", False), ("mean_daily", True)):
            fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), sharex=False)
            axes = np.asarray(axes)
            for c, selection_rank in enumerate(sorted(group["selection_rank"].dropna().unique())[:3]):
                ax = axes[c]
                rows = group[group["selection_rank"] == selection_rank]
                plot_selected_comfort_axis(ax, rows, mean_daily=mean_daily)
                if c == 0:
                    ax.set_ylabel("Indoor temperature [C]")
                ax.set_xlabel("Hour of day: Feb then Mar" if mean_daily else "Feb + Mar time step")
                ax.tick_params(labelsize=7)
            axes[0].legend(fontsize=7)
            fig.suptitle(
                f"{controller_name} | {mode} | {occupant_name} | continuous February to March representative {kind.replace('_', ' ')} comfort",
                fontsize=13,
            )
            fig.tight_layout()
            path = FIG_DIR / (
                f"selected_feb_mar_{slug(controller_name)}_{slug(mode)}_"
                f"{slug(occupant_name)}_{kind}_comfort.png"
            )
            fig.savefig(path, dpi=135, bbox_inches="tight")
            plt.close(fig)
            paths[(controller_name, mode, occupant_name, kind)] = path
    return paths


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
        rbc_tab = rbc_building_table(month)
        combined_btab = pd.concat([btab, rbc_tab], ignore_index=True) if not rbc_tab.empty else btab
        building_tables.append(combined_btab)
        mean_tables.append(mean_table(combined_btab))
        override_tables.append(override_delta_table(combined_btab))
        frames = district_frames(month)
        rbc_frames = rbc_district_frames(month)
        district_tables.append(district_kpis(frames + rbc_frames))
        district_plot_paths[month] = make_district_plots(month, frames)
        override_paths[month] = make_override_plot(month, combined_btab)

    for controller in CONTROLLERS_WITH_BASELINE:
        for mode in DISPLAY_MODES:
            if not any(mode_available(controller["key"], month, mode) for month in MONTHS):
                continue
            occupants = [None] if mode == "no_occupant" else OCCUPANTS
            for occupant_name in occupants:
                occ_key = "none" if occupant_name is None else occupant_name
                comfort_paths[(controller["key"], mode, occ_key)] = make_comfort_grid_combined(
                    controller,
                    mode,
                    occupant_name,
                )

    all_buildings = pd.concat(building_tables, ignore_index=True)
    means = pd.concat(mean_tables, ignore_index=True)
    overrides = pd.concat(override_tables, ignore_index=True)
    districts = pd.concat(district_tables, ignore_index=True)
    dynamics = rlmpc_dynamics_table()
    sac_rewards, sac_losses = sac_training_tables()
    reward_plot, loss_plot = make_sac_training_plots(sac_rewards, sac_losses)
    hist_paths = make_kpi_histograms(all_buildings)
    adaptive_deltas = adaptive_gnurl_delta_table(all_buildings)
    adaptive_delta_summary = compact_adaptive_delta_summary(adaptive_deltas)
    adaptive_delta_paths = make_adaptive_delta_plots(adaptive_deltas)
    tin_setpoint_heatmap_paths = make_tin_setpoint_heatmaps("mar")
    full_kpis = full_building_kpi_table(all_buildings)
    wide_kpis = wide_building_kpi_table(all_buildings)
    representative_buildings = representative_building_selection(all_buildings)
    representative_display = compact_representative_building_selection(representative_buildings)
    selected_cases = selected_building_cases(all_buildings, representative_buildings)
    selected_plot_paths = make_selected_comfort_plots(selected_cases)
    atg = atg_table(all_buildings)
    atg_summary = atg_summary_table(atg)

    all_buildings.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all_raw.csv", index=False)
    full_kpis.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all.csv", index=False)
    wide_kpis.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_wide_by_controller.csv", index=False)
    means.to_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv", index=False)
    overrides.to_csv(ANALYSIS / f"{PREFIX}_override_deltas.csv", index=False)
    districts.to_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv", index=False)
    atg.to_csv(ANALYSIS / f"{PREFIX}_isso74_atg_by_building.csv", index=False)
    atg_summary.to_csv(ANALYSIS / f"{PREFIX}_isso74_atg_summary.csv", index=False)
    if not representative_buildings.empty:
        representative_buildings.to_csv(ANALYSIS / f"{PREFIX}_representative_buildings.csv", index=False)
    if not adaptive_deltas.empty:
        adaptive_deltas.to_csv(ANALYSIS / f"{PREFIX}_gnurl_adaptive_ppo_deltas.csv", index=False)
        adaptive_delta_summary.to_csv(ANALYSIS / f"{PREFIX}_gnurl_adaptive_ppo_delta_summary.csv", index=False)
    if not selected_cases.empty:
        selected_cases.to_csv(ANALYSIS / f"{PREFIX}_selected_representative_building_cases.csv", index=False)
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
    atg_display = compact_atg_summary(atg_summary)

    cells = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}
{TABLE_STYLE}

Final occupant-in-loop run with **each occupant evaluated on all 25 buildings**. This replaces the earlier cyclic occupant assignment.

Cases:

- `no_occupant`: no occupant model, no replay-tail for SAC.
- `without_tdyn`: occupant overrides in the loop, controller tracks the active/effective setpoint.
- `bspline_tdyn`: occupant overrides plus B-spline/Tdyn reference tracking. SAC uses the gentle replay-tail for February-to-March continuation only in this Tdyn branch.

Common cost/reward for all controllers:

`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/2)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`

`reward = -D_k`, no GridStress/energy term. `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and the true-reference guard is `Tref +/- 2 C`.

MPC uses the existing January-fitted 2R2C+Kalman RC models. GNU-RL uses the existing January imitation checkpoints; the adaptive variant starts March from the February online-updated model and continues model-parameter adaptation. SAC uses one safe-PI January episode plus 19 January reward-training episodes for February deployment, then resumes for 10 February reward-training episodes for March deployment.

The B-spline branch uses the Zhang-style dynamic state update:

`Tdyn_{{k+1}} = clip(Tdyn_k + rho*(Tpref-Tdyn_k) + bounded_feedback_nudge_toward_Tpref)`

and the controller/reward reference is occupancy-gated:

`Tref_k = (1 - occ_bin,k)*T_schedule,k + occ_bin,k*Tdyn_k`.

So predicted unoccupied means the controller returns to the schedule; predicted occupied means it tracks Tdyn. In the comfort plots, the orange line is this gated controller reference and the black line is the underlying Tdyn state.

KPI naming:

- Building-level distributions use CityLearn scores where available: imported-electricity score, zero-net-energy score, average-daily-peak score, maximum-peak-demand score, up-ramping score, daily/monthly one-minus load-factor scores, and CityLearn discomfort proportions.
- District-level trajectories use the aggregate district net-electricity demand `P_k=sum_b P_el,b,k`; imported electricity is the positive/imported part `sum max(P_k,0)`.
- CityLearn `ramping_average` is cumulative **up-ramping** because `CostFunction.ramping(..., down_ramp=None)` defaults to `down_ramp=False`. The auxiliary `ramping_abs_kwh` column is the Zoltan-style absolute smoothness metric `sum |P_k-P_{{k-1}}|`.
- `daily_one_minus_load_factor_average` is the daily flatness score (`window=24`); `monthly_one_minus_load_factor_average` is reported only as an auxiliary building-level score.
- Added non-CityLearn auxiliary columns are raw building net electricity, input heating demand, HVAC heating electricity, Kelvin-hours, and override-event counts.
- In this dataset/artifact set, `cost_total` and `carbon_emissions_total` may be blank when CityLearn pricing or carbon-intensity inputs are unavailable/zero.

Mean building KPIs:

{table_display(mean_display, digits=3)}

District aggregate KPIs:

{table_display(district_display, digits=3)}

Override summary:

{table_display(override_summary, digits=2)}

ISSO 74 ATG comfort-class summary:

The ATG bands use `Tneutral = 21.45 + 0.11*Teref`, with `Teref = (Tout_today + 0.8*Tout_yesterday + 0.4*Tout_2_days_ago + 0.2*Tout_3_days_ago)/2.4`. A building is labelled A/B/C here by time-compliance: A if at least 90% of hours are inside `Tneutral +/- 1.25 C`, B if at least 80% are inside `+/- 2.0 C`, C if at least 65% are inside `+/- 2.5 C`; otherwise `outside_C`. The `%` columns report the mean share of hours inside each band across the 25 buildings.

{table_display(atg_display, digits=2)}

GNU-RL adaptive PPO dynamics summary:

{table_display(dynamics_summary, digits=3) if not dynamics_summary.empty else "No adaptive-dynamics histories found."}

GNU-RL adaptive PPO discomfort deltas versus fixed GNU-RL:

Positive `mean_improve_pp` means the PPO-adapted differentiable-MPC model reduced reported discomfort percentage points relative to the fixed GNU-RL model on matched building cases.

{table_display(adaptive_delta_summary, digits=3) if not adaptive_delta_summary.empty else "No fixed-vs-adaptive GNU-RL delta rows found."}

Saved compact tables:

- `analysis/{PREFIX}_building_kpis_all.csv`
- `analysis/{PREFIX}_building_kpis_all_raw.csv`
- `analysis/{PREFIX}_building_kpis_wide_by_controller.csv`
- `analysis/{PREFIX}_representative_buildings.csv`
- `analysis/{PREFIX}_selected_representative_building_cases.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_override_deltas.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_isso74_atg_by_building.csv`
- `analysis/{PREFIX}_isso74_atg_summary.csv`
- `analysis/{PREFIX}_rlmpc_dynamics_summary.csv`
- `analysis/{PREFIX}_gnurl_adaptive_ppo_deltas.csv`
- `analysis/{PREFIX}_gnurl_adaptive_ppo_delta_summary.csv`
- `analysis/{PREFIX}_sac_reward_by_episode.csv`
- `analysis/{PREFIX}_sac_loss_by_episode.csv`
"""
        )
    ]

    cells.append(
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### Full Per-Building KPI Comparison\n\n"
            "All available controller/case/building rows are shown below in long form. "
            "The wide controller-comparison version is saved as "
            f"`analysis/{PREFIX}_building_kpis_wide_by_controller.csv`.\n\n"
            f"{TABLE_STYLE}\n\n{table_display_large(full_kpis, digits=3)}"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### Wide Controller KPI Comparison\n\n"
            "Each row is one month/mode/occupant/building case; controller KPI values are side-by-side.\n\n"
            f"{TABLE_STYLE}\n\n{table_display_large(wide_kpis, digits=3)}"
        )
    )

    if not representative_display.empty:
        cells.append(
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n### Representative Buildings for Thesis Plots\n\n"
                "The selected comfort panels now use the same three buildings for every controller and case. "
                "The selection is anchored in the no-occupant diagnosis: Building 12 is the good-comfort stable-fit case, "
                "Building 15 is a middle-tail case, and Building 8 is the difficult case because it combines poor RC fit "
                "with poor no-occupant comfort for both MPC and SAC. Some poor-RC buildings survive in closed loop, so "
                "the bad example is chosen from the overlap between poor model fit and poor controller comfort. The "
                "occupant section reuses the same buildings so occupant effects are not mixed with changing examples.\n\n"
                f"{TABLE_STYLE}\n\n{table_display(representative_display, digits=3)}"
            )
        )

    if adaptive_delta_paths:
        cells.append(
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n### GNU-RL Adaptive PPO Diagnostics\n\n"
                "`fixed - adaptive` is plotted in reported discomfort percentage points. "
                "Positive bins mean PPO dynamics adaptation reduced discomfort; negative bins mean it increased discomfort."
            )
        )
        for month_name, path in adaptive_delta_paths.items():
            cells.append(image_cell(path, f"{month_name} GNU-RL adaptive PPO discomfort delta histogram"))

    for month in MONTHS:
        cells.append(image_cell(district_plot_paths[month][0], f"{MONTHS[month]['display']} district load"))
        cells.append(image_cell(district_plot_paths[month][1], f"{MONTHS[month]['display']} mean hourly district load"))
        cells.append(image_cell(override_paths[month], f"{MONTHS[month]['display']} override counts"))
        if month == "mar":
            cells.append(
                nbf.v4.new_markdown_cell(
                    f"""<!-- {MARKER} -->
### March Tin - actual setpoint frequency heatmaps

These heatmaps use the actual/effective thermostat setpoint, so occupant overrides are included in the B-spline Tdyn case. `LoD 2` is the no-occupant rollout. `LoD 3` aggregates the four occupant types (`occ1`..`occ4`) for each building under B-spline Tdyn tracking."""
                )
            )
            for controller_name in ["Safe-PI/RBC", "MPC", "GNU-RL", "SAC"]:
                heatmap = tin_setpoint_heatmap_paths.get(controller_name)
                if heatmap is not None:
                    cells.append(
                        image_cell(
                            heatmap,
                            f"March {controller_name} Tin minus actual setpoint heatmap",
                        )
                    )
        for group_name in ("load", "comfort"):
            hist = hist_paths.get((MONTHS[month]["display"], "no_occupant", "controllers", group_name))
            if hist is not None:
                cells.append(
                    image_cell(
                        hist,
                        f"{MONTHS[month]['display']} no-occupant {group_name} KPI distribution plots by controller",
                    )
                )
        for mode in MODES:
            for controller_name in ["Safe-PI/RBC", "MPC", "GNU-RL fixed model", "GNU-RL adaptive model", "SAC"]:
                for group_name in ("load", "comfort", "overrides"):
                    hist = hist_paths.get((MONTHS[month]["display"], mode, controller_name, group_name))
                    if hist is not None:
                        cells.append(
                            image_cell(
                                hist,
                                f"{MONTHS[month]['display']} {mode} {controller_name} {group_name} KPI distribution plots by occupant",
                            )
                        )

    if comfort_paths:
        cells.append(
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n### Combined February/March 25-Building Comfort Grids\n\n"
                "Each subplot is one continuous trajectory from the start of February to the end of March, with a dashed vertical line marking the month boundary."
            )
        )
        for controller in CONTROLLERS_WITH_BASELINE:
            for mode in DISPLAY_MODES:
                occupants = [None] if mode == "no_occupant" else OCCUPANTS
                for occupant_name in occupants:
                    occ_key = "none" if occupant_name is None else occupant_name
                    path = comfort_paths.get((controller["key"], mode, occ_key))
                    if path is not None:
                        cells.append(
                            image_cell(
                                path,
                                f"February/March {controller['display']} {mode} {occ_key} comfort",
                            )
                        )

    if reward_plot is not None:
        cells.append(image_cell(reward_plot, "SAC reward by episode"))
    if loss_plot is not None:
        cells.append(image_cell(loss_plot, "SAC losses by episode"))

    if selected_plot_paths:
        cells.append(
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n### Selected Building Comfort Plots\n\n"
                "Each grid uses the same good/mid/bad representative buildings chosen above. "
                "Each subplot runs continuously from February into March; the title reports the two month-specific discomfort values."
            )
        )
        for key, path in selected_plot_paths.items():
            controller_name, mode, occupant_name, kind = key
            cells.append(
                image_cell(
                    path,
                    f"February/March {controller_name} {mode} {occupant_name} selected {kind.replace('_', ' ')} comfort",
                )
            )

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
