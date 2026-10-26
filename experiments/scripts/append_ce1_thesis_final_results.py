#!/usr/bin/env python
r"""Generate thesis-ready CE1 result figures/tables from saved artifacts.

This script does not rerun controllers. It reads the final p2s2 occupant
artifacts, adds the missing March CityLearn baseline building KPI rows from the
CityLearn baseline KPI file, generates compact thesis figures/tables, appends a
notebook section, and patches the LaTeX result section with `\input{...}` and
`\includegraphics{...}` references.
"""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.occupants.metrics import occupant_probability_comfort_band, setpoint_kelvin_hours
from src.rlmpc_pipeline import FITTED_OCCUPANT_PARAMS, Occupant


PREFIX = "ce1_thesis_final_results"
MARKER = f"## CE1 Thesis Final Compact Results - `{PREFIX}`"
ROOT = Path(".")
ANALYSIS = ROOT / "analysis"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "figures" / PREFIX
LATEX_DIR = ROOT / "IEEE_Conference_Template__1_ (1)"
LATEX_FIG_DIR = LATEX_DIR / "figures" / "results"
LATEX_TABLE_DIR = LATEX_DIR / "tables"
NOTEBOOKS = [
    Path('experiments/analysis/02_mpc_results.ipynb'),
    Path('experiments/analysis/04_rlmpc_results.ipynb'),
    Path('experiments/analysis/08_sac_results.ipynb'),
    Path('experiments/analysis/10_peak_load_profiles.ipynb'),
    Path('experiments/analysis/11_no_occupant_single_building_comparison.ipynb'),
    Path('experiments/analysis/12_simple_v4_comparison.ipynb'),
]

BUILDINGS = dataset_building_entries(DATASET_NAME)
BUILDING_BY_NAME = {b["name"]: b for b in BUILDINGS}
BUILDING_BY_ID = {b["result_id"]: b for b in BUILDINGS}
DATASET_DIR = ROOT / "data" / "datasets" / DATASET_NAME

MONTHS = {
    "February": {"key": "feb", "start": 744, "end": 1415, "baseline_case": "baseline"},
    "March": {"key": "mar", "start": 1416, "end": 2159, "baseline_case": "march_baseline"},
}
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]
OCCUPANT_MODELS = {name: Occupant(**params) for name, params in FITTED_OCCUPANT_PARAMS.items()}
CONTROLLERS = ["Baseline", "Safe-PI/RBC", "MPC", "GNU-RL adaptive model", "SAC"]
OCC_CONTROLLERS = ["Safe-PI/RBC", "MPC", "GNU-RL adaptive model", "SAC"]
MEAN_DAILY_MONTH_GAP = 2.0
MEAN_DAILY_MONTH_SPAN = 24.0
REPRESENTATIVE_BUILDINGS = [
    (
        "good",
        1,
        "building_12",
        "low no-occupant discomfort and stable RC fit",
    ),
    (
        "mid",
        2,
        "building_15",
        "middle no-occupant comfort tail; retained as the median-style example",
    ),
    (
        "bad",
        3,
        "building_08",
        "poor RC fit and poor no-occupant comfort for MPC and SAC",
    ),
]

COLOR = {
    "Baseline": "#333333",
    "Safe-PI/RBC": "#9467bd",
    "MPC": "#ff7f0e",
    "GNU-RL adaptive model": "#2ca02c",
    "SAC": "#d62728",
    "occ1_tolerant": "#1f77b4",
    "occ2_sensitive": "#ff7f0e",
    "occ3_cold": "#2ca02c",
    "occ4_hot": "#d62728",
}
LABEL = {
    "Baseline": "CityLearn baseline",
    "Safe-PI/RBC": "Safe-PI",
    "GNU-RL adaptive model": "GNU-RL",
    "occ1_tolerant": "occ1 tolerant",
    "occ2_sensitive": "occ2 sensitive",
    "occ3_cold": "occ3 cold",
    "occ4_hot": "occ4 hot",
}

SCORE_COLUMNS = [
    "ramping_average",
    "daily_one_minus_load_factor_average",
    "daily_peak_average",
    "all_time_peak_average",
]
CANONICAL_BUILDING_KPIS = [
    ("electricity_consumption_total", "Elec. score"),
    ("daily_peak_average", "Daily peak"),
    ("all_time_peak_average", "Max peak"),
    ("ramping_average", "Ramping"),
    ("daily_one_minus_load_factor_average", "1-LF"),
    ("grid_efficiency_cost", "Grid"),
    ("discomfort_cold_proportion", "Cold [%]"),
    ("discomfort_hot_proportion", "Hot [%]"),
    ("comfort_cost", "Comfort [%]"),
    ("kelvin_hours", "KH"),
]
OCC_BUILDING_KPIS = CANONICAL_BUILDING_KPIS + [
    ("overrides", "Mean overrides/building"),
]
TABLE_VII_HISTOGRAM_KPIS = [
    ("electricity_consumption_total", "Elec. score"),
    ("daily_peak_average", "Daily peak"),
    ("all_time_peak_average", "Max peak"),
    ("ramping_average", "Ramping"),
    ("daily_one_minus_load_factor_average", "1-LF"),
    ("grid_efficiency_cost", "Grid"),
    ("discomfort_cold_pct", "Cold [%]"),
    ("discomfort_hot_pct", "Hot [%]"),
    ("comfort_cost_pct", "Comfort [%]"),
    ("kelvin_hours", "KH"),
]
BASELINE_TABLE_VII_METRICS = {
    "electricity_consumption_total",
    "discomfort_cold_pct",
    "discomfort_hot_pct",
    "comfort_cost_pct",
    "kelvin_hours",
}
HISTOGRAM_FIXED_BINS = {
    "electricity_consumption_total": [-np.inf, 0.8, 0.9, 0.95, 1.0, 1.05, 1.15, 1.3, np.inf],
    "daily_peak_average": [-np.inf, 0.9, 1.0, 1.05, 1.1, 1.2, 1.4, np.inf],
    "all_time_peak_average": [-np.inf, 0.9, 1.0, 1.05, 1.1, 1.2, 1.4, np.inf],
    "ramping_average": [-np.inf, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, np.inf],
    "daily_one_minus_load_factor_average": [-np.inf, 0.8, 0.9, 1.0, 1.05, 1.1, 1.25, 1.5, np.inf],
    "grid_efficiency_cost": [-np.inf, 0.9, 1.0, 1.05, 1.1, 1.25, 1.5, np.inf],
    "discomfort_cold_pct": [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, np.inf],
    "discomfort_hot_pct": [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, np.inf],
    "comfort_cost_pct": [0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, np.inf],
    "kelvin_hours": [0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, np.inf],
    "overrides": [0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, np.inf],
    "override_hours": [0.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, np.inf],
}


def latex_escape(value: object) -> str:
    text = str(value)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def month_slice(month: str) -> slice:
    spec = MONTHS[month]
    return slice(int(spec["start"]), int(spec["end"]))


def dataset_frame(building_id: str, month: str) -> pd.DataFrame:
    building = BUILDING_BY_ID[building_id]
    data = pd.read_csv(DATASET_DIR / f"{building['name']}.csv").iloc[month_slice(month)].reset_index(drop=True)
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "indoor_temperature": pd.to_numeric(data["indoor_dry_bulb_temperature"], errors="coerce"),
            "baseline_setpoint": pd.to_numeric(
                data["indoor_dry_bulb_temperature_heating_set_point"], errors="coerce"
            ),
            "tmin": pd.to_numeric(data["indoor_dry_bulb_temperature_heating_set_point"], errors="coerce"),
            "tmax": pd.to_numeric(data["indoor_dry_bulb_temperature_cooling_set_point"], errors="coerce"),
        }
    )


def ref_setpoint(frame: pd.DataFrame) -> np.ndarray:
    if {"baseline_setpoint", "tdyn", "occ_bin_1step"}.issubset(frame.columns):
        schedule = pd.to_numeric(frame["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
        tdyn = pd.to_numeric(frame["tdyn"], errors="coerce").to_numpy(dtype=float)
        gate = pd.to_numeric(frame["occ_bin_1step"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        n = min(len(schedule), len(tdyn), len(gate))
        ref = np.full(len(frame), np.nan, dtype=float)
        ref[:n] = (1.0 - gate[:n]) * schedule[:n] + gate[:n] * tdyn[:n]
        return ref
    for column in (
        "reference_setpoint",
        "tracking_setpoint",
        "effective_setpoint",
        "baseline_setpoint",
        "scheduled_setpoint",
        "tmin",
    ):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("No setpoint-like column found.")


def active_reference_kelvin_hours(frame: pd.DataFrame) -> float:
    """Total K h absolute indoor-temperature deviation from the active reference."""
    if "indoor_temperature" in frame.columns:
        tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    elif "indoor_dry_bulb_temperature" in frame.columns:
        tin = pd.to_numeric(frame["indoor_dry_bulb_temperature"], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError("No indoor-temperature column found.")
    return setpoint_kelvin_hours(tin, ref_setpoint(frame), timestep_hours=1.0)


def roll_path(row: pd.Series) -> Path:
    scenario = row.get("scenario", np.nan)
    if pd.isna(scenario) or str(scenario).lower() == "nan":
        scenario = "no_occupant" if row.get("mode") == "no_occupant" else "occupant_present"
    return RESULTS / "raw" / str(row["source"]) / str(row["building_id"]) / str(scenario) / str(row["case"]) / "rollout.csv"


def selected_rollout(row: pd.Series) -> pd.DataFrame:
    if row.get("source") == "baseline_dataset":
        return dataset_frame(str(row["building_id"]), str(row["month"]))
    path = roll_path(row)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def comfort_bounds(frame: pd.DataFrame, occupant: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    if occupant in OCCUPANT_MODELS:
        low, high = occupant_probability_comfort_band(OCCUPANT_MODELS[occupant])
        return np.full(n, low), np.full(n, high)
    if {"tmin", "tmax"}.issubset(frame.columns):
        low = pd.to_numeric(frame["tmin"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(frame["tmax"], errors="coerce").to_numpy(dtype=float)
        return low, high
    return np.full(n, 20.0), np.full(n, 24.0)


def sanitize_reference_band(
    ref: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    fallback_half_width: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(ref), len(low), len(high))
    ref = np.asarray(ref[:n], dtype=float)
    low = np.asarray(low[:n], dtype=float)
    high = np.asarray(high[:n], dtype=float)

    fallback_low = ref - float(fallback_half_width)
    fallback_high = ref + float(fallback_half_width)
    bad = (~np.isfinite(low)) | (~np.isfinite(high)) | (high <= low)
    if bad.all():
        return ref, fallback_low, fallback_high
    low = low.copy()
    high = high.copy()
    low[bad] = fallback_low[bad]
    high[bad] = fallback_high[bad]
    return ref, low, high


def tdyn_values(frame: pd.DataFrame) -> np.ndarray | None:
    if "tdyn" not in frame.columns:
        return None
    return pd.to_numeric(frame["tdyn"], errors="coerce").to_numpy(dtype=float)


def plot_reference_and_bounds(row: pd.Series, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the reference line and comfort band that should be shown in thesis plots."""
    if str(row.get("mode")) == "no_occupant" and str(row.get("occupant")) == "none":
        canonical = dataset_frame(str(row["building_id"]), str(row["month"]))
        ref = pd.to_numeric(canonical["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(canonical["tmin"], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(canonical["tmax"], errors="coerce").to_numpy(dtype=float)
        return sanitize_reference_band(ref, low, high)
    ref = ref_setpoint(frame)
    if str(row.get("mode")) == "bspline_tdyn":
        band_center = tdyn_values(frame)
        if band_center is None or not np.isfinite(band_center).any():
            band_center = ref
        low = band_center - 2.0
        high = band_center + 2.0
    else:
        low, high = comfort_bounds(frame, str(row["occupant"]))
    return sanitize_reference_band(ref, low, high)


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    buildings = pd.read_csv(ANALYSIS / "ce1_occ_p2s2_all_occ_building_kpis_all.csv")
    districts = pd.read_csv(ANALYSIS / "ce1_occ_p2s2_all_occ_district_kpis.csv")
    buildings = augment_baseline_building_rows(buildings)
    buildings = apply_active_reference_kh_definition(buildings)
    districts = augment_baseline_district_rows(districts)
    buildings = add_canonical_kpis(buildings)
    return buildings, districts


def apply_active_reference_kh_definition(buildings: pd.DataFrame) -> pd.DataFrame:
    """Make the thesis KH column the total K h distance to active Tref."""
    out = buildings.copy()
    if "kelvin_hours_to_comfort_band" not in out.columns:
        out["kelvin_hours_to_comfort_band"] = pd.to_numeric(out.get("kelvin_hours"), errors="coerce")
    if "kelvin_hours_to_active_reference" not in out.columns:
        out["kelvin_hours_to_active_reference"] = np.nan

    active = pd.to_numeric(out["kelvin_hours_to_active_reference"], errors="coerce")
    missing = ~np.isfinite(active.to_numpy(dtype=float))
    if missing.any():
        values = active.copy()
        for idx, row in out.loc[missing].iterrows():
            try:
                values.loc[idx] = active_reference_kelvin_hours(selected_rollout(row))
            except Exception:
                values.loc[idx] = np.nan
        out["kelvin_hours_to_active_reference"] = values

    out["kelvin_hours"] = pd.to_numeric(out["kelvin_hours_to_active_reference"], errors="coerce")
    return out


def add_canonical_kpis(buildings: pd.DataFrame) -> pd.DataFrame:
    out = buildings.copy()
    for column in [
        "discomfort_cold_proportion",
        "discomfort_hot_proportion",
        "ramping_average",
        "daily_one_minus_load_factor_average",
        "daily_peak_average",
        "all_time_peak_average",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if {"discomfort_cold_proportion", "discomfort_hot_proportion"}.issubset(out.columns):
        out["comfort_cost"] = out["discomfort_cold_proportion"] + out["discomfort_hot_proportion"]
    elif "discomfort_proportion" in out.columns:
        out["comfort_cost"] = pd.to_numeric(out["discomfort_proportion"], errors="coerce")
    else:
        out["comfort_cost"] = np.nan

    if set(SCORE_COLUMNS).issubset(out.columns):
        out["grid_efficiency_cost"] = out[SCORE_COLUMNS].mean(axis=1)
    else:
        out["grid_efficiency_cost"] = np.nan

    return out


def augment_baseline_building_rows(buildings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, spec in MONTHS.items():
        kpi_path = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / spec["baseline_case"] / "kpis.csv"
        if not kpi_path.exists():
            continue
        data = pd.read_csv(kpi_path)
        bdata = data[data["level"].astype(str).eq("building")]
        if bdata.empty:
            continue
        pivot = bdata.pivot_table(index="name", columns="cost_function", values="value", aggfunc="first")
        for name, values in pivot.iterrows():
            if name not in BUILDING_BY_NAME:
                continue
            building = BUILDING_BY_NAME[name]
            frame = dataset_frame(building["result_id"], month)
            tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            base = pd.to_numeric(frame["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
            ref_kh = active_reference_kelvin_hours(frame)
            row = {col: np.nan for col in buildings.columns}
            row.update(
                {
                    "month": month,
                    "mode": "no_occupant",
                    "occupant": "none",
                    "building": building["display"],
                    "building_id": building["result_id"],
                    "controller": "Baseline",
                    "source": "baseline_dataset",
                    "case": spec["baseline_case"],
                    "scenario": "no_occupant",
                    "kelvin_hours": ref_kh,
                    "kelvin_hours_to_active_reference": ref_kh,
                    "overrides": 0.0,
                    "override_hours": 0.0,
                    "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
                }
            )
            for metric in SCORE_COLUMNS:
                if metric in row:
                    row[metric] = 1.0
            for metric, value in values.dropna().items():
                if metric in row:
                    row[metric] = float(value)
            rows.append(row)
    if rows:
        buildings = buildings[buildings["controller"].ne("Baseline")].copy()
        buildings = pd.concat([buildings, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return buildings


def augment_baseline_district_rows(districts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, spec in MONTHS.items():
        path = RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / spec["baseline_case"] / "district_rollout.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        load = pd.to_numeric(frame["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        t = pd.to_numeric(frame["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
        hourly = pd.DataFrame({"hour": t % 24, "load": load}).groupby("hour")["load"].mean()
        rows.append(
            {
                "month": month,
                "controller": "Baseline",
                "mode": "no_occupant",
                "occupant": "none",
                "total_kwh": float(np.sum(load)),
                "peak_kwh": float(np.max(load)),
                "daily_profile_peak_kwh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kwh": float(np.sum(np.abs(np.diff(load)))) if len(load) > 1 else 0.0,
                "load_factor": float(np.mean(load) / max(np.max(load), 1e-9)),
            }
        )
    if rows:
        districts = districts[districts["controller"].ne("Baseline")].copy()
        districts = pd.concat([districts, pd.DataFrame(rows)], ignore_index=True, sort=False)
    return districts


def mean_no_occupant_table(buildings: pd.DataFrame, month: str) -> pd.DataFrame:
    data = buildings[
        buildings["month"].eq(month)
        & buildings["mode"].eq("no_occupant")
        & buildings["occupant"].eq("none")
        & buildings["controller"].isin(CONTROLLERS)
    ].copy()
    rows = (
        data.groupby("controller", as_index=False)[[m for m, _ in CANONICAL_BUILDING_KPIS]]
        .mean(numeric_only=True)
        .set_index("controller")
        .reindex(CONTROLLERS)
        .reset_index()
    )
    rows["discomfort_cold_proportion"] *= 100.0
    rows["discomfort_hot_proportion"] *= 100.0
    rows["comfort_cost"] *= 100.0
    rows["controller"] = rows["controller"].map(
        lambda value: "GNU-RL adaptive" if value == "GNU-RL adaptive model" else value
    )
    return rows.rename(columns={metric: label for metric, label in CANONICAL_BUILDING_KPIS})


def selected_no_occupant_table(buildings: pd.DataFrame) -> pd.DataFrame:
    rep = representative_rows(buildings).sort_values("selection_rank")
    building_labels = {
        row["building_id"]: f"{row['building'].replace('Building ', 'B')} {row['selection']}"
        for _, row in rep.iterrows()
    }
    rows = []
    for month in ("February", "March"):
        month_short = "Feb." if month == "February" else "Mar."
        for _, building in rep.iterrows():
            for controller in CONTROLLERS:
                match = buildings[
                    buildings["month"].eq(month)
                    & buildings["mode"].eq("no_occupant")
                    & buildings["occupant"].eq("none")
                    & buildings["building_id"].eq(building["building_id"])
                    & buildings["controller"].eq(controller)
                ]
                if match.empty:
                    continue
                source = match.iloc[0]
                record = {
                    "Month": month_short,
                    "Building": building_labels[building["building_id"]],
                    "Controller": "GNU-RL adaptive" if controller == "GNU-RL adaptive model" else controller,
                }
                for metric, label in CANONICAL_BUILDING_KPIS:
                    value = pd.to_numeric(pd.Series([source.get(metric)]), errors="coerce").iloc[0]
                    if metric in {"discomfort_cold_proportion", "discomfort_hot_proportion", "comfort_cost"}:
                        value *= 100.0
                    record[label] = value
                rows.append(record)
    return pd.DataFrame(rows)


def mean_occupant_controller_table(buildings: pd.DataFrame, month: str) -> pd.DataFrame:
    data = buildings[
        buildings["month"].eq(month)
        & buildings["mode"].eq("bspline_tdyn")
        & buildings["occupant"].isin(OCCUPANTS)
        & buildings["controller"].isin(OCC_CONTROLLERS)
    ].copy()
    rows = (
        data.groupby(["occupant", "controller"], as_index=False)[[m for m, _ in OCC_BUILDING_KPIS]]
        .mean(numeric_only=True)
        .sort_values(["occupant", "controller"])
    )
    for col in ["discomfort_cold_proportion", "discomfort_hot_proportion", "comfort_cost"]:
        rows[col] *= 100.0
    rows["occupant"] = rows["occupant"].map(lambda x: LABEL.get(x, x))
    rows["controller"] = rows["controller"].map(lambda x: LABEL.get(x, x))
    return rows.rename(columns={metric: label for metric, label in OCC_BUILDING_KPIS})


def format_table_with_winners(
    df: pd.DataFrame,
    path: Path,
    *,
    group_col: str | None = None,
    group_cols: list[str] | None = None,
    lower_is_better: bool = True,
    digits: int = 2,
    winner_exclude: dict[str, set[str]] | None = None,
    hidden_cols: set[str] | None = None,
    separator_cols: list[str] | None = None,
    separator_after_cols: list[str] | None = None,
) -> None:
    table = df.copy()
    hidden_cols = hidden_cols or set()
    visible_cols = [c for c in table.columns if c not in hidden_cols]
    numeric_cols = [c for c in visible_cols if pd.api.types.is_numeric_dtype(table[c])]
    grouping = group_cols or ([group_col] if group_col is not None else None)
    eligible = pd.Series(True, index=table.index)
    for col, excluded_values in (winner_exclude or {}).items():
        if col in table.columns:
            eligible &= ~table[col].astype(str).isin(excluded_values)
    for col in numeric_cols:
        values = pd.to_numeric(table[col], errors="coerce")
        if grouping is None:
            group_values = values[eligible]
            best = group_values.min() if lower_is_better else group_values.max()
            mask = eligible & np.isclose(values, best, rtol=1e-9, atol=1e-9)
        else:
            mask = pd.Series(False, index=table.index)
            for _, idx in table.groupby(grouping, dropna=False).groups.items():
                group_eligible = eligible.loc[idx]
                group_values = values.loc[idx][group_eligible]
                if group_values.dropna().empty:
                    continue
                best = group_values.min() if lower_is_better else group_values.max()
                mask.loc[idx] = group_eligible & np.isclose(values.loc[idx], best, rtol=1e-9, atol=1e-9)
        formatted = []
        for value, is_best in zip(values, mask):
            if pd.isna(value):
                text = "--"
            else:
                text = f"{value:.{digits}f}"
            if is_best and text != "--":
                text = rf"\textbf{{{text}}}"
            formatted.append(text)
        table[col] = formatted
    for col in visible_cols:
        if col not in numeric_cols:
            table[col] = table[col].map(latex_escape)

    align = "".join("r" if col in numeric_cols else "l" for col in visible_cols)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in visible_cols) + r" \\")
    lines.append(r"\midrule")
    previous = None
    separator_cols = separator_cols or []
    separator_after_cols = separator_after_cols or []
    for _, row in table.iterrows():
        if previous is not None and separator_cols:
            before = tuple(str(previous.get(col, "")) for col in separator_cols)
            current = tuple(str(row.get(col, "")) for col in separator_cols)
            if before != current:
                after_before = tuple(str(previous.get(col, "")) for col in separator_after_cols)
                after_current = tuple(str(row.get(col, "")) for col in separator_after_cols)
                lines.append(r"\midrule" if after_before != after_current else r"\hdashline")
        lines.append(" & ".join(str(row[c]) for c in visible_cols) + r" \\")
        previous = row
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def make_tables(buildings: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    no_occ_combined = []
    occ_combined = []
    for month in ("February", "March"):
        month_short = "Feb." if month == "February" else "Mar."
        no_occ = mean_no_occupant_table(buildings, month)
        no_occ_for_combined = no_occ.copy()
        no_occ_for_combined.insert(0, "Month", month_short)
        no_occ_combined.append(no_occ_for_combined)
        csv_path = ANALYSIS / f"{PREFIX}_no_occupant_mean_building_kpis_{month.lower()}.csv"
        tex_path = LATEX_TABLE_DIR / f"{PREFIX}_no_occupant_mean_building_kpis_{month.lower()}.tex"
        no_occ.to_csv(csv_path, index=False)
        format_table_with_winners(
            no_occ.rename(columns={"controller": "Controller"}),
            tex_path,
            digits=2,
            winner_exclude={"Controller": {"Baseline"}},
        )
        paths[f"no_occ_{month.lower()}"] = tex_path

        occ = mean_occupant_controller_table(buildings, month)
        occ_for_combined = occ.copy()
        occ_for_combined.insert(0, "Month", month_short)
        occ_combined.append(occ_for_combined)
        csv_path = ANALYSIS / f"{PREFIX}_occupant_controller_mean_kpis_{month.lower()}.csv"
        tex_path = LATEX_TABLE_DIR / f"{PREFIX}_occupant_controller_mean_kpis_{month.lower()}.tex"
        occ.to_csv(csv_path, index=False)
        format_table_with_winners(
            occ.rename(columns={"occupant": "Occupant", "controller": "Controller"}),
            tex_path,
            group_col="Occupant",
            digits=2,
        )
        paths[f"occ_{month.lower()}"] = tex_path

    if no_occ_combined:
        combined = pd.concat(no_occ_combined, ignore_index=True)
        csv_path = LATEX_TABLE_DIR / f"{PREFIX}_no_occupant_mean_building_kpis_feb_mar.csv"
        tex_path = LATEX_TABLE_DIR / f"{PREFIX}_no_occupant_mean_building_kpis_feb_mar.tex"
        combined.to_csv(csv_path, index=False)
        format_table_with_winners(
            combined.rename(columns={"controller": "Controller"}),
            tex_path,
            group_col="Month",
            digits=2,
            winner_exclude={"Controller": {"Baseline"}},
            separator_cols=["Month"],
            separator_after_cols=["Month"],
        )
        paths["no_occ_feb_mar"] = tex_path

    selected_no_occ = selected_no_occupant_table(buildings)
    if not selected_no_occ.empty:
        csv_path = LATEX_TABLE_DIR / f"{PREFIX}_no_occupant_selected_building_kpis_feb_mar.csv"
        tex_path = LATEX_TABLE_DIR / f"{PREFIX}_no_occupant_selected_building_kpis_feb_mar.tex"
        selected_no_occ.to_csv(csv_path, index=False)
        format_table_with_winners(
            selected_no_occ,
            tex_path,
            group_cols=["Month", "Building"],
            digits=2,
            winner_exclude={"Controller": {"Baseline"}},
            separator_cols=["Month", "Building"],
            separator_after_cols=["Month"],
        )
        paths["no_occ_selected_feb_mar"] = tex_path

    if occ_combined:
        combined = pd.concat(occ_combined, ignore_index=True)
        csv_path = LATEX_TABLE_DIR / f"{PREFIX}_occupant_controller_mean_kpis_feb_mar.csv"
        tex_path = LATEX_TABLE_DIR / f"{PREFIX}_occupant_controller_mean_kpis_feb_mar.tex"
        combined.to_csv(csv_path, index=False)
        format_table_with_winners(
            combined.rename(columns={"occupant": "Occupant", "controller": "Controller"}),
            tex_path,
            group_cols=["Month", "Occupant"],
            digits=2,
            separator_cols=["Month", "Occupant"],
            separator_after_cols=["Month"],
        )
        paths["occ_feb_mar"] = tex_path
    return paths


def hist_bins(values: pd.Series, nbins: int = 7) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.linspace(0.0, 1.0, nbins + 1)
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-3)
        return np.linspace(vmin - pad, vmax + pad, 4)
    locator = mticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    return np.asarray(locator.tick_values(vmin, vmax), dtype=float)


def metric_hist_bins(metric: str, values: pd.Series, nbins: int = 7) -> np.ndarray:
    if metric in HISTOGRAM_FIXED_BINS:
        return np.asarray(HISTOGRAM_FIXED_BINS[metric], dtype=float)
    return hist_bins(values, nbins=nbins)


def hist_edge_label(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return ""
    if abs(value) >= 10000:
        text = f"{value / 1000.0:.1f}".rstrip("0").rstrip(".")
        return f"{text}k"
    if abs(value) >= 10:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def hist_bin_label(left: float, right: float) -> str:
    if np.isneginf(left):
        return f"<{hist_edge_label(right)}"
    if np.isposinf(right):
        return f">{hist_edge_label(left)}"
    return f"{hist_edge_label(left)}-{hist_edge_label(right)}"


def grouped_histogram_grid(
    data: pd.DataFrame,
    compare_col: str,
    compare_values: list[str],
    metrics: list[tuple[str, str]],
    title: str,
    path: Path,
    *,
    ncols: int = 2,
    fig_width: float = 15.5,
    row_height: float = 3.7,
) -> Path:
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, row_height * nrows), squeeze=False)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        if metric not in data.columns:
            ax.set_axis_off()
            continue
        metric_data = data[[compare_col, metric]].dropna()
        if metric_data.empty:
            ax.set_axis_off()
            continue
        bins = metric_hist_bins(metric, metric_data[metric], nbins=7)
        centers = np.arange(len(bins) - 1, dtype=float)
        groups = [v for v in compare_values if not metric_data[metric_data[compare_col].eq(v)].empty]
        width = 0.78 / max(1, len(groups))
        for i, value in enumerate(groups):
            raw = pd.to_numeric(metric_data.loc[metric_data[compare_col].eq(value), metric], errors="coerce").dropna()
            counts, _ = np.histogram(raw, bins=bins)
            x = centers - 0.39 + (i + 0.5) * width
            ax.bar(x, counts, width=width, color=COLOR.get(value, "0.4"), alpha=0.88, label=LABEL.get(value, value))
        tick_labels = [hist_bin_label(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        ax.set_xticks(centers, tick_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Buildings")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_axis_off()
    fig.suptitle(title, y=0.997, fontsize=13)
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.95), h_pad=1.65, w_pad=1.2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def table_vii_histogram_grid(data: pd.DataFrame, month: str, path: Path) -> Path:
    """Plot the no-occupant KPI distributions using exactly the Table VII columns."""
    ncols = 5
    nrows = int(np.ceil(len(TABLE_VII_HISTOGRAM_KPIS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(17.2, 6.8), squeeze=False)
    for ax, (metric, label) in zip(axes.ravel(), TABLE_VII_HISTOGRAM_KPIS):
        if metric not in data.columns:
            ax.set_axis_off()
            continue
        metric_data = data[["controller", metric]].dropna()
        if metric_data.empty:
            ax.set_axis_off()
            continue
        bins = metric_hist_bins(metric, metric_data[metric], nbins=6)
        centers = np.arange(len(bins) - 1, dtype=float)
        groups = [value for value in CONTROLLERS if not metric_data[metric_data["controller"].eq(value)].empty]
        width = 0.82 / max(1, len(groups))
        for i, value in enumerate(groups):
            raw = pd.to_numeric(
                metric_data.loc[metric_data["controller"].eq(value), metric],
                errors="coerce",
            ).dropna()
            counts, _ = np.histogram(raw, bins=bins)
            x = centers - 0.41 + (i + 0.5) * width
            ax.bar(
                x,
                counts,
                width=width,
                color=COLOR.get(value, "0.4"),
                alpha=0.88,
                label=LABEL.get(value, value),
            )
        tick_labels = [hist_bin_label(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
        ax.set_xticks(centers, tick_labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Buildings")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=6.7, ncol=2, frameon=False)
    for ax in axes.ravel()[len(TABLE_VII_HISTOGRAM_KPIS) :]:
        ax.set_axis_off()
    fig.suptitle(f"{month} no-occupant building KPI distributions matching Table VII", y=1.01, fontsize=13)
    fig.tight_layout(rect=(0.01, 0.02, 1.0, 0.95), h_pad=1.6, w_pad=1.0)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def make_histograms(buildings: pd.DataFrame) -> dict[str, Path]:
    paths = {}
    load_metrics = [
        ("electricity_consumption_total", "Elec. score"),
        ("net_electricity_consumption_kwh", "Building net electricity [kWh]"),
        ("daily_peak_average", "Daily peak"),
        ("all_time_peak_average", "Max peak"),
        ("ramping_average", "Ramping"),
        ("daily_one_minus_load_factor_average", "1-LF"),
        ("grid_efficiency_cost", "Grid"),
        ("ramping_up_kwh", "Ramping [kWh]"),
    ]
    comfort_metrics = [
        ("discomfort_cold_pct", "Cold discomfort [%]"),
        ("discomfort_hot_pct", "Hot discomfort [%]"),
        ("comfort_cost_pct", "Comfort cost [%]"),
        ("kelvin_hours", "Kelvin-hours"),
        ("Tin_max", "Maximum indoor temperature"),
    ]
    override_metrics = [("overrides", "Override events"), ("override_hours", "Override hours")]
    for month in ("February", "March"):
        no_occ = buildings[
            buildings["month"].eq(month)
            & buildings["mode"].eq("no_occupant")
            & buildings["occupant"].eq("none")
            & buildings["controller"].isin(CONTROLLERS)
        ].copy()
        no_occ["discomfort_cold_pct"] = no_occ["discomfort_cold_proportion"] * 100.0
        no_occ["discomfort_hot_pct"] = no_occ["discomfort_hot_proportion"] * 100.0
        no_occ["comfort_cost_pct"] = no_occ["comfort_cost"] * 100.0
        paths[f"no_occ_{month.lower()}_table_vii"] = table_vii_histogram_grid(
            no_occ,
            month,
            FIG_DIR / f"{month.lower()}_no_occupant_table_vii_histograms.png",
        )
        mpc_occ = buildings[
            buildings["month"].eq(month)
            & buildings["mode"].eq("bspline_tdyn")
            & buildings["controller"].eq("MPC")
            & buildings["occupant"].isin(OCCUPANTS)
        ].copy()
        mpc_occ["discomfort_cold_pct"] = mpc_occ["discomfort_cold_proportion"] * 100.0
        mpc_occ["discomfort_hot_pct"] = mpc_occ["discomfort_hot_proportion"] * 100.0
        mpc_occ["comfort_cost_pct"] = mpc_occ["comfort_cost"] * 100.0
        paths[f"mpc_occ_{month.lower()}_load"] = grouped_histogram_grid(
            mpc_occ,
            "occupant",
            OCCUPANTS,
            load_metrics,
            f"{month} MPC B-spline $T^{{dyn}}$ load KPI distributions by occupant",
            FIG_DIR / f"{month.lower()}_mpc_bspline_tdyn_load_hist_by_occupant.png",
        )
        paths[f"mpc_occ_{month.lower()}_comfort"] = grouped_histogram_grid(
            mpc_occ,
            "occupant",
            OCCUPANTS,
            comfort_metrics + override_metrics,
            f"{month} MPC B-spline $T^{{dyn}}$ comfort/override KPI distributions by occupant",
            FIG_DIR / f"{month.lower()}_mpc_bspline_tdyn_comfort_hist_by_occupant.png",
        )
    mpc_occ_deployment = buildings[
        buildings["month"].isin(["February", "March"])
        & buildings["mode"].eq("bspline_tdyn")
        & buildings["controller"].eq("MPC")
        & buildings["occupant"].isin(OCCUPANTS)
    ].copy()
    mpc_occ_deployment["discomfort_cold_pct"] = mpc_occ_deployment["discomfort_cold_proportion"] * 100.0
    mpc_occ_deployment["discomfort_hot_pct"] = mpc_occ_deployment["discomfort_hot_proportion"] * 100.0
    mpc_occ_deployment["comfort_cost_pct"] = mpc_occ_deployment["comfort_cost"] * 100.0
    deployment_metrics = [metric for metric, _ in load_metrics + comfort_metrics + override_metrics]
    sum_metrics = {
        "net_electricity_consumption_kwh",
        "ramping_up_kwh",
        "kelvin_hours",
        "overrides",
        "override_hours",
    }
    max_metrics = {"Tin_max"}
    agg = {}
    for metric in deployment_metrics:
        if metric not in mpc_occ_deployment.columns:
            continue
        if metric in sum_metrics:
            agg[metric] = "sum"
        elif metric in max_metrics:
            agg[metric] = "max"
        else:
            agg[metric] = "mean"
    mpc_occ_deployment = (
        mpc_occ_deployment.groupby(["occupant", "building_id"], as_index=False)
        .agg(agg)
    )
    paths["mpc_occ_deployment_load"] = grouped_histogram_grid(
        mpc_occ_deployment,
        "occupant",
        OCCUPANTS,
        load_metrics,
        "February-March MPC B-spline $T^{dyn}$ deployment load KPI distributions by occupant",
        FIG_DIR / "feb_mar_mpc_bspline_tdyn_load_hist_by_occupant.png",
    )
    paths["mpc_occ_deployment_comfort"] = grouped_histogram_grid(
        mpc_occ_deployment,
        "occupant",
        OCCUPANTS,
        comfort_metrics + override_metrics,
        "February-March MPC B-spline $T^{dyn}$ deployment comfort/override KPI distributions by occupant",
        FIG_DIR / "feb_mar_mpc_bspline_tdyn_comfort_hist_by_occupant.png",
    )
    deployment_table_metrics = [
        ("electricity_consumption_total", "Elec. score"),
        ("daily_peak_average", "Daily peak"),
        ("all_time_peak_average", "Max peak"),
        ("ramping_average", "Ramping"),
        ("daily_one_minus_load_factor_average", "1-LF"),
        ("grid_efficiency_cost", "Grid"),
        ("discomfort_cold_pct", "Cold [%]"),
        ("discomfort_hot_pct", "Hot [%]"),
        ("comfort_cost_pct", "Comfort [%]"),
        ("kelvin_hours", "KH"),
        ("overrides", "Overrides"),
    ]
    paths["mpc_occ_deployment_kpis"] = grouped_histogram_grid(
        mpc_occ_deployment,
        "occupant",
        OCCUPANTS,
        deployment_table_metrics,
        "February-March MPC B-spline $T^{dyn}$ deployment KPI distributions by occupant",
        FIG_DIR / "feb_mar_mpc_bspline_tdyn_table_viii_hist_by_occupant.png",
        ncols=4,
        fig_width=17.2,
        row_height=3.0,
    )
    return paths


def representative_rows(buildings: pd.DataFrame) -> pd.DataFrame:
    no_occ = buildings[
        buildings["mode"].eq("no_occupant")
        & buildings["occupant"].eq("none")
        & buildings["month"].isin(["February", "March"])
    ].copy()
    summary = (
        no_occ.dropna(subset=["discomfort_proportion"])
        .groupby(["building_id", "building"], as_index=False)
        .agg(mean_no_occ_discomfort=("discomfort_proportion", "mean"))
        .sort_values(["mean_no_occ_discomfort", "building_id"])
        .reset_index(drop=True)
    )
    summary["discomfort_rank"] = np.arange(1, len(summary) + 1)
    rows: list[pd.DataFrame] = []
    for selection, selection_rank, building_id, reason in REPRESENTATIVE_BUILDINGS:
        row = summary[summary["building_id"].eq(building_id)].copy()
        if row.empty:
            label = building_id.replace("building_", "Building ")
            row = pd.DataFrame(
                {
                    "building_id": [building_id],
                    "building": [label],
                    "mean_no_occ_discomfort": [np.nan],
                    "discomfort_rank": [np.nan],
                }
            )
        row["selection"] = selection
        row["selection_rank"] = selection_rank
        row["selection_reason"] = reason
        rows.append(row)
    rep = pd.concat(rows, ignore_index=True)
    return rep[
        [
            "building_id",
            "building",
            "selection",
            "selection_rank",
            "discomfort_rank",
            "selection_reason",
            "mean_no_occ_discomfort",
        ]
    ].copy()


def case_rows_for_plot(
    buildings: pd.DataFrame,
    *,
    controller: str,
    mode: str,
    occupant: str,
    building_id: str,
) -> pd.DataFrame:
    rows = buildings[
        buildings["controller"].eq(controller)
        & buildings["mode"].eq(mode)
        & buildings["occupant"].eq(occupant)
        & buildings["building_id"].eq(building_id)
        & buildings["month"].isin(["February", "March"])
    ].copy()
    rows["_month_order"] = rows["month"].map({"February": 0, "March": 1})
    return rows.sort_values("_month_order")


def set_local_comfort_ylim(ax, values: list[np.ndarray], *, min_span: float = 5.0) -> None:
    finite = []
    for value in values:
        arr = np.asarray(value, dtype=float).ravel()
        finite.append(arr[np.isfinite(arr)])
    finite = [arr for arr in finite if len(arr)]
    if not finite:
        return
    merged = np.concatenate(finite)
    low = float(np.min(merged))
    high = float(np.max(merged))
    span = max(high - low, float(min_span))
    pad = max(0.4, 0.08 * span)
    ax.set_ylim(low - pad, high + pad)


def close_mean_daily_profile(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "x" not in data.columns:
        return data
    start = float(data["x"].min())
    first = data[np.isclose(data["x"], start)]
    if first.empty:
        return data
    closure = first.iloc[[0]].copy()
    closure["x"] = start + MEAN_DAILY_MONTH_SPAN
    return pd.concat([data, closure], ignore_index=True).sort_values("x")


def configure_mean_daily_xaxis(ax) -> None:
    march_start = MEAN_DAILY_MONTH_SPAN + MEAN_DAILY_MONTH_GAP
    ticks = [
        0,
        6,
        12,
        18,
        24,
        march_start,
        march_start + 6,
        march_start + 12,
        march_start + 18,
        march_start + 24,
    ]
    labels = ["0", "6", "12", "18", "24", "0", "6", "12", "18", "24"]
    ax.axvline(MEAN_DAILY_MONTH_SPAN + MEAN_DAILY_MONTH_GAP / 2.0, color="0.25", lw=0.75, ls="--", alpha=0.6)
    ax.set_xlim(0, march_start + MEAN_DAILY_MONTH_SPAN)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def plot_continuous_case(ax, rows: pd.DataFrame, *, mean_daily: bool) -> None:
    if rows.empty:
        ax.set_axis_off()
        return
    offset = 0.0
    boundary = None
    plotted = False
    title_parts = []
    ylim_values: list[np.ndarray] = []
    for _, row in rows.iterrows():
        frame = selected_rollout(row)
        tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        ref, low, high = plot_reference_and_bounds(row, frame)
        tdyn = tdyn_values(frame)
        n = min(len(tin), len(ref), len(low), len(high), len(tdyn) if tdyn is not None else len(tin))
        x = np.arange(n, dtype=float)
        if mean_daily:
            month_start = 0.0 if row["month"] == "February" else MEAN_DAILY_MONTH_SPAN + MEAN_DAILY_MONTH_GAP
            x = (x % MEAN_DAILY_MONTH_SPAN) + month_start
        else:
            x += offset
        data = pd.DataFrame({"x": x[:n], "Tin": tin[:n], "ref": ref[:n], "low": low[:n], "high": high[:n]})
        if tdyn is not None:
            data["Tdyn"] = tdyn[:n]
        if mean_daily:
            data = data.groupby("x", as_index=False).mean(numeric_only=True)
            data = close_mean_daily_profile(data)
        ax.fill_between(
            data["x"],
            data["low"],
            data["high"],
            color="0.78",
            alpha=0.55,
            linewidth=0,
            label="comfort band" if not plotted else None,
        )
        ax.plot(data["x"], data["Tin"], lw=1.25, color="steelblue", label="Tin" if not plotted else None)
        ax.plot(data["x"], data["ref"], lw=1.15, color="tab:orange", label="setpoint" if not plotted else None)
        if "Tdyn" in data:
            ax.plot(data["x"], data["Tdyn"], lw=1.15, color="black", label="Tdyn" if not plotted else None)
        ylim_values.extend([data["Tin"].to_numpy(), data["ref"].to_numpy(), data["low"].to_numpy(), data["high"].to_numpy()])
        if "Tdyn" in data:
            ylim_values.append(data["Tdyn"].to_numpy())
        disc = row.get("discomfort_proportion", np.nan)
        title_parts.append(f"{str(row['month'])[:3]} {100.0 * float(disc):.1f}%")
        plotted = True
        if not mean_daily:
            offset += n
            if row["month"] == "February":
                boundary = offset
    if mean_daily:
        configure_mean_daily_xaxis(ax)
    elif boundary is not None:
        ax.axvline(boundary - 0.5, color="0.25", lw=0.75, ls="--", alpha=0.6)
    ax.grid(alpha=0.2)
    ax.set_title("; ".join(title_parts), fontsize=15)
    set_local_comfort_ylim(ax, ylim_values)


def harmonize_column_ylims(axes: np.ndarray, *, pad_fraction: float = 0.06) -> None:
    for c in range(axes.shape[1]):
        lows: list[float] = []
        highs: list[float] = []
        for ax in axes[:, c]:
            if not ax.has_data():
                continue
            low, high = ax.get_ylim()
            lows.append(float(low))
            highs.append(float(high))
        if not lows or not highs:
            continue
        low = min(lows)
        high = max(highs)
        span = max(high - low, 1.0)
        pad = pad_fraction * span
        for ax in axes[:, c]:
            if ax.has_data():
                ax.set_ylim(low - pad, high + pad)


def make_no_occupant_selected_stack(buildings: pd.DataFrame) -> dict[str, Path]:
    rep = representative_rows(buildings)
    paths = {}
    for kind, mean_daily in (("month", False), ("mean_daily", True)):
        fig, axes = plt.subplots(len(CONTROLLERS), 3, figsize=(17, 2.45 * len(CONTROLLERS)), sharex=False)
        axes = np.asarray(axes)
        for r, controller in enumerate(CONTROLLERS):
            for c, (_, b) in enumerate(rep.sort_values("selection_rank").iterrows()):
                ax = axes[r, c]
                rows = case_rows_for_plot(
                    buildings,
                    controller=controller,
                    mode="no_occupant",
                    occupant="none",
                    building_id=b["building_id"],
                )
                plot_continuous_case(ax, rows, mean_daily=mean_daily)
                if r == 0:
                    ax.set_title(
                        f"{b['selection']}: {b['building']} (rank {int(b['discomfort_rank'])})\n{ax.get_title()}",
                        fontsize=16,
                    )
                if c == 0:
                    ax.set_ylabel(f"{LABEL.get(controller, controller)}\nTin [C]", fontsize=16)
                ax.tick_params(labelsize=14)
        axes[0, 0].legend(fontsize=14)
        title_kind = "February/March mean daily" if mean_daily else "continuous February--March full"
        fig.suptitle(
            f"No-occupant representative buildings by controller - {title_kind} comfort",
            fontsize=21,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        path = FIG_DIR / f"thesis_no_occupant_selected_controllers_{kind}.png"
        fig.savefig(path, dpi=230, bbox_inches="tight")
        plt.close(fig)
        paths[kind] = path
    return paths


def make_no_occupant_selected_kpi_bars(buildings: pd.DataFrame) -> dict[str, Path]:
    rep = representative_rows(buildings).sort_values("selection_rank")
    paths: dict[str, Path] = {}
    load_metrics = CANONICAL_BUILDING_KPIS[:6]
    comfort_metrics = [
        ("discomfort_cold_proportion", "Cold [%]"),
        ("discomfort_hot_proportion", "Hot [%]"),
        ("comfort_cost", "Comfort [%]"),
    ]
    bar_width = 0.145
    controller_offsets = (np.arange(len(CONTROLLERS)) - (len(CONTROLLERS) - 1) / 2.0) * bar_width

    for month in ("February", "March"):
        month_rows = buildings[
            buildings["month"].eq(month)
            & buildings["mode"].eq("no_occupant")
            & buildings["occupant"].eq("none")
            & buildings["controller"].isin(CONTROLLERS)
            & buildings["building_id"].isin(rep["building_id"])
        ].copy()
        export_rows = month_rows.copy()
        for col in ["discomfort_cold_proportion", "discomfort_hot_proportion", "comfort_cost"]:
            export_rows[col] = 100.0 * pd.to_numeric(export_rows[col], errors="coerce")
        export_rows[
            [
                "month",
                "building_id",
                "building",
                "controller",
                *[metric for metric, _ in load_metrics],
                *[metric for metric, _ in comfort_metrics],
                "kelvin_hours",
            ]
        ].to_csv(
            ANALYSIS / f"{PREFIX}_no_occupant_selected_building_kpis_{month.lower()}.csv",
            index=False,
        )

        fig, axes = plt.subplots(
            2,
            len(rep),
            figsize=(11.6, 5.3),
            sharex=False,
            gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.42, "wspace": 0.48},
        )
        axes = np.asarray(axes)
        legend_handles = []
        load_x = np.arange(len(load_metrics), dtype=float)
        comfort_x = np.array([0.0, 1.0, 2.0], dtype=float)
        kh_x = 4.0

        for c, (_, b) in enumerate(rep.iterrows()):
            building_rows = month_rows[month_rows["building_id"].eq(b["building_id"])]
            load_ax = axes[0, c]
            comfort_ax = axes[1, c]
            kh_ax = comfort_ax.twinx()

            for i, controller in enumerate(CONTROLLERS):
                row = building_rows[building_rows["controller"].eq(controller)]
                if row.empty:
                    continue
                row = row.iloc[0]
                color = COLOR[controller]
                display = LABEL.get(controller, controller)
                load_values = [
                    pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                    for metric, _ in load_metrics
                ]
                bars = load_ax.bar(
                    load_x + controller_offsets[i],
                    load_values,
                    width=bar_width,
                    color=color,
                    label=display,
                )
                if c == 0:
                    legend_handles.append(bars[0])

                comfort_values = [
                    100.0 * pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                    for metric, _ in comfort_metrics
                ]
                comfort_ax.bar(
                    comfort_x + controller_offsets[i],
                    comfort_values,
                    width=bar_width,
                    color=color,
                )
                kh_value = pd.to_numeric(pd.Series([row.get("kelvin_hours")]), errors="coerce").iloc[0]
                kh_ax.bar(
                    kh_x + controller_offsets[i],
                    [kh_value],
                    width=bar_width,
                    color=color,
                    alpha=0.78,
                )

            title = f"{b['selection']}: {b['building']} (rank {int(b['discomfort_rank'])})"
            load_ax.set_title(title, fontsize=12)
            load_ax.set_xticks(load_x)
            load_ax.set_xticklabels([label for _, label in load_metrics], rotation=28, ha="right", fontsize=9)
            load_ax.tick_params(axis="y", labelsize=9)
            load_ax.grid(axis="y", alpha=0.22)
            load_ax.set_ylim(bottom=0)
            if c == 0:
                load_ax.set_ylabel("Normalized score\n(lower better)", fontsize=10)

            comfort_ax.set_xticks([*comfort_x, kh_x])
            comfort_ax.set_xticklabels([label for _, label in comfort_metrics] + ["Ref. KH"], rotation=28, ha="right", fontsize=9)
            comfort_ax.tick_params(axis="y", labelsize=9)
            comfort_ax.grid(axis="y", alpha=0.22)
            comfort_ax.set_ylim(bottom=0)
            comfort_ax.axvline(3.0, color="0.35", lw=0.75, ls="--", alpha=0.55)
            comfort_ax.set_xlim(-0.6, 4.55)
            kh_ax.set_xlim(comfort_ax.get_xlim())
            kh_ax.set_ylim(bottom=0)
            kh_ax.tick_params(axis="y", labelsize=8)
            if c == 0:
                comfort_ax.set_ylabel("Comfort [%]\n(lower better)", fontsize=10)
            if c == len(rep) - 1:
                kh_ax.set_ylabel("Ref. KH\n(lower better)", fontsize=10)

        fig.legend(
            legend_handles,
            [LABEL.get(controller, controller) for controller in CONTROLLERS],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.025),
            ncol=len(CONTROLLERS),
            frameon=False,
            fontsize=10,
        )
        fig.suptitle(f"{month} no-occupant KPIs for the representative buildings", fontsize=14, y=1.075)
        path = FIG_DIR / f"thesis_no_occupant_selected_building_kpis_{month.lower()}.png"
        fig.savefig(path, dpi=260, bbox_inches="tight")
        plt.close(fig)
        paths[month.lower()] = path
    return paths


def make_no_occupant_selected_kpi_bar_summary(buildings: pd.DataFrame) -> Path:
    rep = representative_rows(buildings).sort_values("selection_rank")
    load_metrics = CANONICAL_BUILDING_KPIS[:6]
    comfort_metrics = [
        ("discomfort_cold_proportion", "Cold [%]"),
        ("discomfort_hot_proportion", "Hot [%]"),
        ("comfort_cost", "Comfort [%]"),
    ]
    bar_width = 0.145
    controller_offsets = (np.arange(len(CONTROLLERS)) - (len(CONTROLLERS) - 1) / 2.0) * bar_width
    load_x = np.arange(len(load_metrics), dtype=float)
    comfort_x = np.array([0.0, 1.0, 2.0], dtype=float)
    kh_x = 4.0

    fig, axes = plt.subplots(
        4,
        len(rep),
        figsize=(11.7, 8.8),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.0], "hspace": 0.58, "wspace": 0.48},
    )
    axes = np.asarray(axes)
    legend_handles = []

    for month_idx, month in enumerate(("February", "March")):
        month_rows = buildings[
            buildings["month"].eq(month)
            & buildings["mode"].eq("no_occupant")
            & buildings["occupant"].eq("none")
            & buildings["controller"].isin(CONTROLLERS)
            & buildings["building_id"].isin(rep["building_id"])
        ].copy()
        load_row = 2 * month_idx
        comfort_row = load_row + 1

        for c, (_, b) in enumerate(rep.iterrows()):
            building_rows = month_rows[month_rows["building_id"].eq(b["building_id"])]
            load_ax = axes[load_row, c]
            comfort_ax = axes[comfort_row, c]
            kh_ax = comfort_ax.twinx()

            for i, controller in enumerate(CONTROLLERS):
                row = building_rows[building_rows["controller"].eq(controller)]
                if row.empty:
                    continue
                row = row.iloc[0]
                color = COLOR[controller]
                display = LABEL.get(controller, controller)
                load_values = [
                    pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                    for metric, _ in load_metrics
                ]
                bars = load_ax.bar(
                    load_x + controller_offsets[i],
                    load_values,
                    width=bar_width,
                    color=color,
                    label=display,
                )
                if month_idx == 0 and c == 0:
                    legend_handles.append(bars[0])

                comfort_values = [
                    100.0 * pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
                    for metric, _ in comfort_metrics
                ]
                comfort_ax.bar(
                    comfort_x + controller_offsets[i],
                    comfort_values,
                    width=bar_width,
                    color=color,
                )
                kh_value = pd.to_numeric(pd.Series([row.get("kelvin_hours")]), errors="coerce").iloc[0]
                kh_ax.bar(
                    kh_x + controller_offsets[i],
                    [kh_value],
                    width=bar_width,
                    color=color,
                    alpha=0.78,
                )

            if month_idx == 0:
                load_ax.set_title(
                    f"{b['selection']}: {b['building']} (rank {int(b['discomfort_rank'])})",
                    fontsize=11,
                )
            load_ax.set_xticks(load_x)
            load_ax.set_xticklabels([label for _, label in load_metrics], rotation=28, ha="right", fontsize=8.5)
            load_ax.tick_params(axis="y", labelsize=8.5)
            load_ax.grid(axis="y", alpha=0.22)
            load_ax.set_ylim(bottom=0)
            if c == 0:
                load_ax.set_ylabel(f"{month}\nNormalized score\n(lower better)", fontsize=9.5)

            comfort_ax.set_xticks([*comfort_x, kh_x])
            comfort_ax.set_xticklabels([label for _, label in comfort_metrics] + ["Ref. KH"], rotation=28, ha="right", fontsize=8.5)
            comfort_ax.tick_params(axis="y", labelsize=8.5)
            comfort_ax.grid(axis="y", alpha=0.22)
            comfort_ax.set_ylim(bottom=0)
            comfort_ax.axvline(3.0, color="0.35", lw=0.75, ls="--", alpha=0.55)
            comfort_ax.set_xlim(-0.6, 4.55)
            kh_ax.set_xlim(comfort_ax.get_xlim())
            kh_ax.set_ylim(bottom=0)
            kh_ax.tick_params(axis="y", labelsize=7.5)
            if c == 0:
                comfort_ax.set_ylabel(f"{month}\nComfort [%]\n(lower better)", fontsize=9.5)
            if c == len(rep) - 1:
                kh_ax.set_ylabel("Ref. KH\n(lower better)", fontsize=9.5)

    fig.legend(
        legend_handles,
        [LABEL.get(controller, controller) for controller in CONTROLLERS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(CONTROLLERS),
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle("Selected-building no-occupant KPIs by controller", fontsize=13, y=1.045)
    path = FIG_DIR / "thesis_no_occupant_selected_building_kpis_feb_mar.png"
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return path


def make_mpc_occupant_selected_stack(buildings: pd.DataFrame) -> dict[str, Path]:
    rep = representative_rows(buildings)
    paths = {}
    for kind, mean_daily in (("month", False), ("mean_daily", True)):
        fig, axes = plt.subplots(len(OCCUPANTS), 3, figsize=(17, 2.45 * len(OCCUPANTS)), sharex=False)
        axes = np.asarray(axes)
        for r, occupant in enumerate(OCCUPANTS):
            for c, (_, b) in enumerate(rep.sort_values("selection_rank").iterrows()):
                ax = axes[r, c]
                rows = case_rows_for_plot(
                    buildings,
                    controller="MPC",
                    mode="bspline_tdyn",
                    occupant=occupant,
                    building_id=b["building_id"],
                )
                plot_continuous_case(ax, rows, mean_daily=mean_daily)
                if r == 0:
                    ax.set_title(
                        f"{b['selection']}: {b['building']} (rank {int(b['discomfort_rank'])})\n{ax.get_title()}",
                        fontsize=16,
                    )
                if c == 0:
                    ax.set_ylabel(f"{LABEL[occupant]}\nTin [C]", fontsize=16)
                ax.tick_params(labelsize=14)
        axes[0, 0].legend(fontsize=14)
        title_kind = "February/March mean daily" if mean_daily else "continuous February--March full"
        fig.suptitle(
            f"MPC B-spline $T^{{dyn}}$ representative buildings by occupant - {title_kind} comfort",
            fontsize=21,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        path = FIG_DIR / f"thesis_mpc_occupant_selected_{kind}.png"
        fig.savefig(path, dpi=230, bbox_inches="tight")
        plt.close(fig)
        paths[kind] = path
    return paths


def load_series_from_rows(rows: pd.DataFrame) -> np.ndarray | None:
    loads = []
    for _, row in rows.iterrows():
        frame = selected_rollout(row)
        for col in ("net_electricity_consumption", "electricity_consumption", "heating_electricity_consumption"):
            if col in frame.columns:
                loads.append(pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
                break
    if not loads:
        return None
    n = min(len(v) for v in loads)
    return np.sum([v[:n] for v in loads], axis=0)


def make_mpc_occupant_daily_load(buildings: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(12.5, 4.8))
    curve_selectors = {
        "no occupant": lambda month: buildings[
            buildings["controller"].eq("MPC")
            & buildings["month"].eq(month)
            & buildings["mode"].eq("no_occupant")
            & buildings["occupant"].eq("none")
        ]
    }
    for occ in OCCUPANTS:
        curve_selectors[LABEL[occ]] = (
            lambda month, occ=occ: buildings[
                buildings["controller"].eq("MPC")
                & buildings["month"].eq(month)
                & buildings["mode"].eq("bspline_tdyn")
                & buildings["occupant"].eq(occ)
            ]
        )
    for name, selector in curve_selectors.items():
        month_series = []
        for month in ("February", "March"):
            series = load_series_from_rows(selector(month))
            if series is not None:
                month_series.append(series)
        if not month_series:
            continue
        series = np.concatenate(month_series)
        daily = pd.DataFrame({"hour": np.arange(len(series)) % 24, "load": series}).groupby("hour")["load"].mean()
        ax.plot(daily.index, daily.values, marker="o", lw=1.5, label=name)
    ax.set_title("MPC mean hourly aggregate load by occupant - February-March deployment")
    ax.set_ylabel("Mean aggregate electricity [kWh]")
    ax.set_xlabel("Hour of day")
    ax.grid(alpha=0.25)
    ax.set_xticks(range(24))
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    path = FIG_DIR / "thesis_mpc_occupant_mean_hourly_load_feb_mar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_outputs_to_latex(paths: dict[str, Path]) -> None:
    LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        if path is None or not Path(path).exists() or Path(path).suffix.lower() not in {".png", ".pdf"}:
            continue
        shutil.copy2(path, LATEX_FIG_DIR / Path(path).name)


def notebook_table(df: pd.DataFrame, digits: int = 2) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return out.to_markdown(index=False)


def append_notebooks(cells: list[nbf.NotebookNode]) -> None:
    for path in NOTEBOOKS:
        nb = nbf.read(path, as_version=4)
        nb.cells = [cell for cell in nb.cells if MARKER not in str(cell.get("source", ""))]
        nb.cells.extend(cells)
        nbf.write(nb, path)
        print(f"Appended {path}")


def image_cell(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n### {title}\n\n![](../{path.as_posix()})")


def patch_latex() -> None:
    tex_path = LATEX_DIR / "conference_101719.tex"
    text = tex_path.read_text()
    start_marker = "% BEGIN CE1_THESIS_FINAL_RESULTS"
    end_marker = "% END CE1_THESIS_FINAL_RESULTS"
    block = rf"""
{start_marker}

\subsection{{No-Occupant Controller KPIs}}
Tables~\ref{{tab:no_occ_feb_kpis}} and~\ref{{tab:no_occ_mar_kpis}}
summarize the mean building-level KPIs for the no-occupant case. Winners
are bolded column-wise. The representative buildings used in
Fig.~\ref{{fig:no_occ_rep_month}} and~\ref{{fig:no_occ_rep_daily}} are chosen
from the no-occupant diagnosis and then reused unchanged in the occupant
analysis. Building 12 is a good-comfort, stable-fit case; Building 15 is a
middle-tail case; and Building 8 is the difficult case because it combines poor
RC fit with poor no-occupant comfort for both MPC and SAC. Some poor-RC
buildings still produce acceptable comfort in closed loop, so the bad example is
chosen from the overlap between poor model fit and poor controller comfort.

\begin{{table*}}[t]
\centering
\caption{{No-occupant mean building KPIs in February. Lower is better.}}
\label{{tab:no_occ_feb_kpis}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{\input{{tables/{PREFIX}_no_occupant_mean_building_kpis_february.tex}}}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\caption{{No-occupant mean building KPIs in March. Lower is better.}}
\label{{tab:no_occ_mar_kpis}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{\input{{tables/{PREFIX}_no_occupant_mean_building_kpis_march.tex}}}}
\end{{table*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/results/thesis_no_occupant_selected_controllers_month.png}}
\caption{{Continuous February--March comfort for the selected good, median,
and bad no-occupant buildings. Controllers are stacked row-wise. The orange
line is the canonical scheduled setpoint from the building dataset for every
controller. Each building column uses a shared y-axis so the controller rows
can be compared directly.}}
\label{{fig:no_occ_rep_month}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/results/thesis_no_occupant_selected_controllers_mean_daily.png}}
\caption{{Representative mean daily comfort for the same no-occupant
buildings. The vertical dashed line separates the February and March daily
profiles; each side uses its own 0--24 h x-axis. The orange line is again the
same scheduled setpoint for each controller row. Each building column uses a
shared y-axis.}}
\label{{fig:no_occ_rep_daily}}
\end{{figure*}}

\begin{{table*}}[t]
\centering
\caption{{Selected-building no-occupant KPIs in February and March. The same
Table~\ref{{tab:no_occ_kpis}} KPI columns are reported for the three representative buildings.
Winners are bolded within each month and building, excluding the baseline.
Lower is better.}}
\label{{tab:no_occ_selected_kpis}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{\input{{tables/{PREFIX}_no_occupant_selected_building_kpis_feb_mar.tex}}}}
\end{{table*}}

Figures~\ref{{fig:no_occ_rep_month}} and~\ref{{fig:no_occ_rep_daily}} make the
controller differences more explicit than the KPI table alone. The CityLearn
baseline is not a realistic heat-pump controller in this comparison; it
provides the ideal thermal load required to meet the scheduled setpoint, rather
than directly controlling the heat-pump action. For Buildings 12 and 15, this
produces almost zero deviation from the scheduled setpoint and zero discomfort.
The exception is the difficult Building 8, where the baseline still has
$1.6\%$ discomfort in February. This indicates that Building 8 has sufficiently
complex thermal dynamics that even the ideal-load reference leaves a small
setpoint-tracking deviation.

Safe-PI/RBC tracks the time-biased reference $T^{{ref}}$ rather than the
canonical scheduled setpoint. In the good Building 12, it shows the intended
pre-heating and pre-cooling behavior in the corresponding time windows, but
still leaves temperature outside the comfort band, with an average discomfort
of about $18.4\%$ over the two deployment months. In the median Building 15,
the mean daily plot again shows pre-heating and pre-cooling, but with a
persistent upward temperature offset; the resulting discomfort is slightly
above $50\%$ on average. In the difficult Building 8, no clear pre-heating or
pre-cooling pattern remains. The behavior is dominated by overheating, with
discomfort above $66\%$ in both months.

MPC and GNU-RL show very similar qualitative behavior on the three
representative buildings, with MPC slightly better in comfort. In Buildings 12
and 15, the indoor temperature visibly approaches approximately $+1~^\circ$C
and $-1~^\circ$C relative to the scheduled setpoint during the pre-heating and
pre-cooling windows, respectively. Both controllers therefore obtain zero, or
nearly zero, discomfort on the good and median buildings. For Building 8, the
pre-heating and pre-cooling pattern is much less visible, but the controllers
still keep comfort within a usable range: MPC has roughly $6\%$ mean
discomfort over February and March, while GNU-RL is about two percentage points
worse. The better MPC behavior on this difficult building is attributed to the
Kalman filter correction in the 2R2C state estimate. The RC error remains large
enough to prevent clean pre-heating and pre-cooling, but the receding-horizon
optimization is re-solved at every time step and can compensate enough to keep
comfort at acceptable levels.

SAC shows a different failure mode. In the good and median buildings, the
January-only policy produces systematic underheating during the February
deployment, although it still follows the broad pre-heating and pre-cooling
pattern. After additional February training, the policy adapts to the new
weather conditions and the discomfort that was around a few percent in February
is almost eliminated in March. This suggests that the RL policy needs to have
seen weather conditions close to the deployment distribution; with only January
data, it under-supplies heat in February. Finally, Building 8 remains difficult
for SAC. Its February discomfort is about $55\%$, almost entirely from
underheating, and although the March value falls to about $17\%$, it remains
worse than MPC and GNU-RL. Thus, complex thermal dynamics remain difficult even
for a model-free neural controller. Avoiding an explicit RC model does not make
the underlying closed-loop control problem easy; if the building response is
hard, the mapping that RL must learn is hard as well.

\subsection{{Occupant Experiments}}
The occupant experiments test the B-spline occupancy gate and dynamic comfort
reference under stochastic thermostat overrides. Controllers are first compared
with and without $T^{{dyn}}$, and MPC is then used to isolate occupant-type
effects.

\subsubsection{{Controller comparison}}
To evaluate the combined occupancy-prediction and adaptive-temperature-tracking
mechanism, occupant-in-the-loop rollouts were run once with the scheduled
setpoint and once with the B-spline occupancy gate and $T^{{dyn}}$. Thermostat
overrides were counted from the occupant feedback model. Figure~\ref{{fig:override_counts}}
reports total new override events across the 25 buildings, while
Table~\ref{{tab:occ_kpis}} reports mean building-level KPIs for the B-spline
$T^{{dyn}}$ case. With the dynamic reference, MPC has the fewest overrides for
all months and occupants, followed by GNU-RL. SAC improves in March after
additional February training but remains third. Safe-PI/RBC has the most
overrides and the weakest override reduction, about $14\%$ on average compared
with $78\%$ for MPC and $72\%$ for GNU-RL, consistent with its unstable
bang-bang behavior.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.49\textwidth]{{figures/results/feb_override_counts.png}}
\hfill
\includegraphics[width=0.49\textwidth]{{figures/results/mar_override_counts.png}}
\caption{{Total occupant override events in February (left) and March (right),
summed over the 25 buildings for each occupant type, with and without the
B-spline dynamic comfort reference.}}
\label{{fig:override_counts}}
\end{{figure*}}

\begin{{table*}}[t]
\centering
\caption{{B-spline $T^{{dyn}}$ occupant-controller KPIs in February and March.
Load-shape columns are mean normalized building-level scores; comfort and
override columns are mean building-level values. Winners are bolded within each
month and occupant.}}
\label{{tab:occ_kpis}}
\scriptsize
\resizebox{{\textwidth}}{{!}}{{\input{{tables/{PREFIX}_occupant_controller_mean_kpis_feb_mar.tex}}}}
\end{{table*}}

Table~\ref{{tab:occ_kpis}} shows the same controller separation observed in the
no-occupant experiment. Safe-PI/RBC consumes the least energy but has the worst
daily peak, $1-\mathrm{{LF}}$, ramping, and override count. SAC has strong peak
scores and often low tabulated discomfort; in February this is mostly cold
discomfort, indicating underheating, and it improves in March after seeing more
similar weather. Comfort percentage is less reliable here because the B-spline
gate is tuned for high recall, so override counts are the safest direct comfort
indicator. KH is also informative because it is computed against the active
predicted reference. Under these indicators, MPC is the most reliable:
SAC has about twice the MPC KH in February and $1.4$ times in March, while
Safe-PI/RBC has about four times the MPC KH in February and more than three
times in March. MPC also gives the best grid score in both months and usually
the best or near-best ramping, $1-\mathrm{{LF}}$, and override behavior. The rest
of the occupant analysis therefore uses MPC and merges February--March into one
deployment period.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.92\textwidth]{{figures/results/thesis_mpc_occupant_mean_hourly_load_feb_mar.png}}
\caption{{MPC mean hourly aggregate load for the no-occupant case and the four
occupant types over the combined February--March deployment period.}}
\label{{fig:mpc_occupant_load}}
\end{{figure*}}

\subsubsection{{MPC occupant comparison}}
For MPC, occupant type mainly changes the active reference. The tolerant
occupant has the fewest overrides, about $6.1$ per building-month, followed by
cold-preferring at $16.9$, hot-preferring at $30.4$, and sensitive at $37.5$.
KH is lowest for cold-preferring ($460.9$), then tolerant ($516.5$), sensitive
($607.5$), and hot-preferring ($706.8$), showing that high preferred
temperatures are harder to satisfy during the heating season. The same pattern
appears in Fig.~\ref{{fig:mpc_occupant_load}}: hot and sensitive occupants require
more continuous heat-pump action, giving the flattest aggregate profiles and the
lowest $1-\mathrm{{LF}}$ values, $0.86$ and $0.89$.

Figures~\ref{{fig:mpc_occ_rep_month}} and~\ref{{fig:mpc_occ_rep_daily}} show the
building-level mechanism. For the good Building 12, MPC follows the desired
pre-heating and pre-cooling pattern. The odd February sensitive-occupant case is
a high-demand case, not a SAC-like training artifact: mean $T_{{in}}$ is
$22.09^\circ$C against active reference $22.96^\circ$C and $T^{{dyn}}=23.00^\circ$C,
the gate is active $98.5\%$ of the time, and mean action is already $0.93$.
In March the same case has mean $T_{{in}}=22.99^\circ$C, reference
$22.97^\circ$C, and action $0.39$. Building 15 remains mostly hot-uncomfortable:
the gate predicts occupancy in $81.8\%$ of steps versus $79.3\%$ actual
occupancy, with only $2.5\%$ false positives and zero false negatives, so the
failure is mainly the high dynamic reference interacting with building response.
Building 8 remains difficult because of complex thermal dynamics and poor RC
fit; even the 2R2C model has about $5.9^\circ$C March holdout one-step RMSE, so
midnight pre-cooling is not consistently recovered.

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/results/thesis_mpc_occupant_selected_month.png}}
\caption{{Continuous February--March MPC comfort for the representative
buildings, stacked by occupant type. The orange line is the active
occupancy-gated reference, the black line is the underlying dynamic reference,
and the gray area marks the dynamic-reference $\pm2~^\circ$C band. Each subplot
uses a local y-axis.}}
\label{{fig:mpc_occ_rep_month}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/results/thesis_mpc_occupant_selected_mean_daily.png}}
\caption{{Representative mean daily MPC comfort for the same buildings and
occupant types. The vertical dashed line separates February from March, and
each side uses its own 0--24 h x-axis. The active reference follows the
schedule when the B-spline gate predicts no occupancy and follows Tdyn when
occupancy is predicted. Each subplot uses a local y-axis, and the gray area
marks the dynamic-reference $\pm2~^\circ$C band.}}
\label{{fig:mpc_occ_rep_daily}}
\end{{figure*}}

{end_marker}
"""
    if start_marker in text and end_marker in text:
        before, rest = text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        text = before + block + after
    else:
        old = (
            "Fig.~\\ref{fig:no_occ_rep_month} and~\\ref{fig:no_occ_rep_daily} are chosen\n"
            "once from the full final result table: Building 12 has the lowest aggregate\n"
            "discomfort, Building 20 is closest to the median aggregate discomfort, and\n"
            "Building 15 has the highest aggregate discomfort. This keeps the qualitative\n"
            "comfort comparison consistent across controllers."
        )
        new = (
            "Fig.~\\ref{fig:no_occ_rep_month} and~\\ref{fig:no_occ_rep_daily} are chosen\n"
            "from the no-occupant diagnosis and then reused unchanged in the occupant\n"
            "analysis. Building 12 is a good-comfort, stable-fit case; Building 15 is a\n"
            "middle-tail case; and Building 8 is the difficult case because it combines poor\n"
            "RC fit with poor no-occupant comfort for both MPC and SAC. Some poor-RC\n"
            "buildings still produce acceptable comfort in closed loop, so the bad example is\n"
            "chosen from the overlap between poor model fit and poor controller comfort."
        )
        if old in text:
            text = text.replace(old, new, 1)
        else:
            print("LaTeX representative-building paragraph already updated or not found; leaving section structure unchanged.")
    tex_path.write_text(text)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    buildings, districts = load_artifacts()

    table_paths = make_tables(buildings)
    hist_paths = make_histograms(buildings)
    no_occ_selected = make_no_occupant_selected_stack(buildings)
    mpc_occ_selected = make_mpc_occupant_selected_stack(buildings)
    mpc_occ_load = make_mpc_occupant_daily_load(buildings)

    all_paths = {
        **hist_paths,
        **{f"no_occ_selected_{k}": v for k, v in no_occ_selected.items()},
        **{f"mpc_occ_selected_{k}": v for k, v in mpc_occ_selected.items()},
        "mpc_occ_load": mpc_occ_load,
    }
    copy_outputs_to_latex(all_paths)
    patch_latex()

    buildings.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_augmented.csv", index=False)
    districts.to_csv(ANALYSIS / f"{PREFIX}_district_kpis_augmented.csv", index=False)

    feb_no_occ = mean_no_occupant_table(buildings, "February")
    mar_no_occ = mean_no_occupant_table(buildings, "March")
    selected_no_occ = selected_no_occupant_table(buildings)
    feb_occ = mean_occupant_controller_table(buildings, "February")
    mar_occ = mean_occupant_controller_table(buildings, "March")

    cells = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

This section condenses the final result artifacts into thesis-ready tables and
figures. It does not rerun any controller.

Representative buildings are fixed once across the full result set:

- **Building 12**: good no-occupant comfort and stable RC fit.
- **Building 15**: middle no-occupant comfort tail.
- **Building 8**: poor RC fit and poor no-occupant comfort for MPC and SAC.

The no-occupant comparison includes CityLearn baseline, Safe-PI/RBC, MPC,
adaptive GNU-RL, and SAC. The occupant-analysis figures then focus on MPC after
the controller comparison, because MPC is the most consistent comfort controller
under B-spline $T^{{dyn}}$. The occupant plots reuse the same three buildings so the
occupant effect is not confounded by changing examples.

Saved augmented tables:

- `analysis/{PREFIX}_building_kpis_augmented.csv`
- `analysis/{PREFIX}_district_kpis_augmented.csv`
- `analysis/{PREFIX}_no_occupant_mean_building_kpis_february.csv`
- `analysis/{PREFIX}_no_occupant_mean_building_kpis_march.csv`
- `analysis/{PREFIX}_occupant_controller_mean_kpis_february.csv`
- `analysis/{PREFIX}_occupant_controller_mean_kpis_march.csv`
"""
        ),
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### No-Occupant Mean Building KPIs - February\n\n"
            + notebook_table(feb_no_occ.rename(columns={"controller": "controller"}), digits=2)
        ),
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### No-Occupant Mean Building KPIs - March\n\n"
            + notebook_table(mar_no_occ.rename(columns={"controller": "controller"}), digits=2)
        ),
        image_cell(no_occ_selected["month"], "No-Occupant Representative Buildings - Full Feb/March"),
        image_cell(no_occ_selected["mean_daily"], "No-Occupant Representative Buildings - Mean Daily"),
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### No-Occupant Selected-Building KPIs\n\n"
            + notebook_table(selected_no_occ, digits=2)
        ),
        image_cell(hist_paths["no_occ_february_table_vii"], "No-Occupant February Table VII KPI Histograms"),
        image_cell(hist_paths["no_occ_march_table_vii"], "No-Occupant March Table VII KPI Histograms"),
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### Occupant Controller Mean KPIs - February\n\n"
            + notebook_table(feb_occ, digits=1)
        ),
        nbf.v4.new_markdown_cell(
            f"<!-- {MARKER} -->\n### Occupant Controller Mean KPIs - March\n\n"
            + notebook_table(mar_occ, digits=1)
        ),
        image_cell(mpc_occ_load, "MPC Occupant Mean Hourly Load - February-March Deployment"),
        image_cell(mpc_occ_selected["month"], "MPC Occupant Representative Buildings - Full Feb/March"),
        image_cell(mpc_occ_selected["mean_daily"], "MPC Occupant Representative Buildings - Mean Daily"),
        image_cell(hist_paths["mpc_occ_deployment_kpis"], "MPC Occupant February-March KPI Histograms"),
    ]
    append_notebooks(cells)

    print("Generated thesis final result assets:")
    for key, path in sorted(all_paths.items()):
        print(f"  {key}: {path}")
    for key, path in sorted(table_paths.items()):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
