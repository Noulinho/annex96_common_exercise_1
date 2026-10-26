#!/usr/bin/env python
"""Append SAC occupant-probability reward comparison to 08_sac_results only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.occupants.metrics import kelvin_hours, occupant_probability_comfort_band, override_hours
from src.rlmpc_pipeline import FITTED_OCCUPANT_PARAMS, Occupant


PREFIX = "ce1_occ_D010702_sac_occprob"
MARKER = f"## CE1 SAC Occupant Override-Probability Reward - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOK = Path('experiments/analysis/08_sac_results.ipynb')

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]
OCCUPANT_MODELS = {name: Occupant(**params) for name, params in FITTED_OCCUPANT_PARAMS.items()}
MODES = ["without_tdyn", "bspline_tdyn"]
MONTHS = {
    "feb": {"display": "February", "start": 744, "end": 1415},
    "mar": {"display": "March", "start": 1416, "end": 2159},
}

RUNS = [
    {
        "key": "old",
        "display": "SAC old Tref guard",
        "labels": {
            "feb": {
                "without_tdyn": "ce1_occ_D010702_sac_jan20_feb_without_tdyn",
                "bspline_tdyn": "ce1_occ_D010702_sac_jan20_feb_bspline_tdyn",
            },
            "mar": {
                "without_tdyn": "ce1_occ_D010702_sac_feb10_mar_without_tdyn",
                "bspline_tdyn": "ce1_occ_D010702_sac_feb10_mar_bspline_tdyn",
            },
        },
    },
    {
        "key": "occprob",
        "display": "SAC override-prob reward",
        "labels": {
            "feb": {
                "without_tdyn": "ce1_occ_D010702_sac_occprob_jan20_feb_without_tdyn",
                "bspline_tdyn": "ce1_occ_D010702_sac_occprob_jan20_feb_bspline_tdyn",
            },
            "mar": {
                "without_tdyn": "ce1_occ_D010702_sac_occprob_feb10_mar_without_tdyn",
                "bspline_tdyn": "ce1_occ_D010702_sac_occprob_feb10_mar_bspline_tdyn",
            },
        },
    },
]


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def raw_dir(building_id: str, case: str) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building_id / "occupant_present" / case


def rollout_path(building: dict, case: str) -> Path:
    return raw_dir(building["result_id"], case) / "rollout.csv"


def available_case(run: dict, month: str, mode: str, building: dict) -> str | None:
    case = run["labels"][month][mode]
    return case if rollout_path(building, case).exists() else None


def read_rollout(building: dict, case: str) -> pd.DataFrame:
    return pd.read_csv(rollout_path(building, case))


def read_summary(building: dict, case: str) -> pd.Series | None:
    path = RESULTS / "summaries" / "sac_merlin" / building["result_id"] / "occupant_present" / "summary.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path)
    rows = data[data["case"].astype(str) == case]
    return None if rows.empty else rows.iloc[-1]


def read_kpis(building: dict, case: str) -> dict[str, float]:
    path = raw_dir(building["result_id"], case) / "kpis.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    if "level" in data.columns:
        data = data[data["level"].astype(str) == "district"]
    return {
        str(row["cost_function"]): float(row["value"])
        for _, row in data.iterrows()
        if pd.notna(row.get("value"))
    }


def num(row: pd.Series | None, key: str, default=np.nan) -> float:
    if row is None or key not in row or pd.isna(row[key]):
        return float(default)
    return float(row[key])


def load(frame: pd.DataFrame) -> np.ndarray:
    for column in ("net_electricity_consumption", "electricity_consumption", "heating_electricity_consumption"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raise ValueError("Rollout has no electricity column.")


def time(frame: pd.DataFrame) -> np.ndarray:
    if "time_step" in frame.columns:
        return pd.to_numeric(frame["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(frame), dtype=int)


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


def rollout_summary(frame: pd.DataFrame, occupant_name: str, summary: pd.Series | None) -> dict[str, float]:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    occupant = OCCUPANT_MODELS[occupant_name]
    lower, upper = occupant_probability_comfort_band(occupant)

    feedback = None
    if "occupant_feedback" in frame.columns:
        feedback = pd.to_numeric(frame["occupant_feedback"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        override_count = float(np.count_nonzero(np.abs(feedback) > 0.0))
    else:
        override_count = num(summary, "override_count")

    base = baseline_setpoint(frame)
    eff = None
    try:
        eff = effective_setpoint(frame)
    except ValueError:
        pass
    if base is not None and eff is not None:
        override_h = override_hours(base, eff)
        if np.isnan(override_count):
            n = min(len(base), len(eff))
            active = np.abs(eff[:n] - base[:n]) > 1e-6
            starts = active & np.concatenate(([True], ~active[:-1]))
            override_count = float(np.count_nonzero(starts))
    elif feedback is not None:
        override_h = float(np.count_nonzero(np.abs(feedback) > 0.0))
    else:
        override_h = num(summary, "override_hours")

    return {
        "kelvin_hours": kelvin_hours(tin, lower=lower, upper=upper),
        "overrides": override_count,
        "override_hours": override_h,
        "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
        "tdyn_mean": float(pd.to_numeric(frame["tdyn"], errors="coerce").mean()) if "tdyn" in frame.columns else np.nan,
        "tdyn_final": float(pd.to_numeric(frame["tdyn"], errors="coerce").iloc[-1])
        if "tdyn" in frame.columns and len(frame)
        else np.nan,
        "reward_total": float(pd.to_numeric(frame.get("reward", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        if "reward" in frame.columns
        else num(summary, "reward_total"),
    }


def building_table() -> pd.DataFrame:
    rows = []
    for month in MONTHS:
        for run in RUNS:
            for mode in MODES:
                for building in BUILDINGS:
                    occ = occupant_for_building(building)
                    case = available_case(run, month, mode, building)
                    if case is None:
                        continue
                    roll = read_rollout(building, case)
                    summ = read_summary(building, case)
                    vals = read_kpis(building, case)
                    derived = rollout_summary(roll, occ, summ)
                    rows.append(
                        {
                            "month": MONTHS[month]["display"],
                            "controller": run["display"],
                            "mode": mode,
                            "building": building["display"],
                            "building_id": building["result_id"],
                            "occupant": occ,
                            "energy": vals.get("electricity_consumption_total", num(summ, "electricity_consumption_total")),
                            "disc_%": 100.0 * vals.get("discomfort_proportion", num(summ, "discomfort_proportion")),
                            "cold_%": 100.0 * vals.get("discomfort_cold_proportion", num(summ, "discomfort_cold_proportion")),
                            "hot_%": 100.0 * vals.get("discomfort_hot_proportion", num(summ, "discomfort_hot_proportion")),
                            "KH": derived["kelvin_hours"],
                            "daily_peak": vals.get("daily_peak_average", num(summ, "daily_peak_average")),
                            "ramping": vals.get("ramping_average", np.nan),
                            "overrides": derived["overrides"],
                            "override_hours": derived["override_hours"],
                            "Tin_max": derived["Tin_max"],
                            "tdyn_mean": derived["tdyn_mean"],
                            "tdyn_final": derived["tdyn_final"],
                            "reward_total": derived["reward_total"],
                        }
                    )
    return pd.DataFrame(rows)


def mean_table(buildings: pd.DataFrame) -> pd.DataFrame:
    cols = ["energy", "disc_%", "cold_%", "hot_%", "KH", "daily_peak", "ramping", "overrides", "override_hours", "Tin_max", "tdyn_final"]
    return buildings.groupby(["month", "controller", "mode"], as_index=False)[cols].mean(numeric_only=True)


def district_frames() -> list[pd.DataFrame]:
    frames = []
    for month in MONTHS:
        for run in RUNS:
            for mode in MODES:
                loads = []
                t0 = None
                for building in BUILDINGS:
                    case = available_case(run, month, mode, building)
                    if case is None:
                        continue
                    roll = read_rollout(building, case)
                    values = load(roll)
                    t = time(roll)
                    if t0 is None:
                        t0 = t
                    n = min(len(t0), len(values))
                    t0 = t0[:n]
                    loads.append(values[:n])
                if not loads:
                    continue
                n = min(len(values) for values in loads)
                frames.append(
                    pd.DataFrame(
                        {
                            "month": MONTHS[month]["display"],
                            "time_step": np.asarray(t0[:n], dtype=int),
                            "load": np.sum([values[:n] for values in loads], axis=0),
                            "controller": run["display"],
                            "mode": mode,
                            "series": f"{run['display']} | {mode}",
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


def bias_for_time(t: np.ndarray) -> np.ndarray:
    hour_label = (np.asarray(t, dtype=int) % 24) + 1
    preheat = np.isin(hour_label, [1, 2, 3, 4, 5, 14, 15, 16, 17])
    reduce = np.isin(hour_label, [7, 8, 9, 10, 18, 19, 20, 21, 22])
    return np.where(preheat, 1.0, np.where(reduce, -1.0, 0.0))


def make_comfort_grid(month: str, run: dict, mode: str) -> Path | None:
    if not any(available_case(run, month, mode, building) is not None for building in BUILDINGS):
        return None
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        case = available_case(run, month, mode, building)
        occ = occupant_for_building(building)
        if case is None:
            ax.set_axis_off()
            continue
        roll = read_rollout(building, case)
        t = time(roll)
        tin = pd.to_numeric(roll["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = ref_setpoint(roll)
        eff = effective_setpoint(roll)
        n = min(len(t), len(tin), len(tsp), len(eff))
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], eff[:n], lw=0.55, ls="--", color="black", label="effective setpoint")
        ax.plot(t[:n], tsp[:n], lw=0.65, ls="-.", color="tab:green", label="tracking reference")
        ax.plot(t[:n], tsp[:n] + bias_for_time(t[:n]), lw=0.55, ls=":", color="tab:orange", label="reference+bias")
        ax.fill_between(t[:n], tsp[:n] - 2.0, tsp[:n] + 2.0, color="gray", alpha=0.12)
        ax.set_title(f"{building['display']} | {occ}", fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} {mode} | {MONTHS[month]['display']} comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{month}_{run['key']}_{mode}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_district_daily_plot(frames: list[pd.DataFrame], month: str) -> Path | None:
    data = [frame for frame in frames if frame["month"].iloc[0] == MONTHS[month]["display"]]
    if not data:
        return None
    combined = pd.concat(data, ignore_index=True)
    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["series", "hour"], as_index=False)["load"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("series", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.2, label=name)
    ax.set_title(f"SAC mean hourly district load - {MONTHS[month]['display']}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = FIG_DIR / f"{month}_sac_daily_profile.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def sac_training_table() -> pd.DataFrame:
    rows = []
    for month, label_prefix in [("feb", "jan20_feb"), ("mar", "feb10_mar")]:
        for run in RUNS:
            for mode in MODES:
                case = run["labels"][month][mode]
                for building in BUILDINGS:
                    path = raw_dir(building["result_id"], case) / "online_training_rollout.csv"
                    if not path.exists():
                        continue
                    data = pd.read_csv(path)
                    if data.empty or not {"episode", "raw_reward", "reward"}.issubset(data.columns):
                        continue
                    grouped = data.groupby("episode", as_index=False).agg(
                        raw_reward_mean=("raw_reward", "mean"),
                        reward_mean=("reward", "mean"),
                        action_mean=("action", "mean"),
                    )
                    grouped["month"] = MONTHS[month]["display"]
                    grouped["controller"] = run["display"]
                    grouped["mode"] = mode
                    grouped["building"] = building["display"]
                    rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    data = pd.concat(rows, ignore_index=True)
    return data.groupby(["month", "controller", "mode", "episode"], as_index=False).mean(numeric_only=True)


def make_training_reward_plot(rewards: pd.DataFrame) -> Path | None:
    if rewards.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for (month, controller, mode), group in rewards.groupby(["month", "controller", "mode"], sort=False):
        ax.plot(group["episode"], group["raw_reward_mean"], marker="o", lw=1.2, label=f"{month} | {controller} | {mode}")
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_title("SAC online raw reward by episode - per-step mean")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean raw reward per step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = FIG_DIR / "sac_occprob_training_reward.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


TABLE_STYLE = """
<style>
.ce1-table-wrap { max-width: 100%; overflow-x: auto; margin: 0.25rem 0 0.9rem 0; }
.ce1-table { border-collapse: collapse; font-size: 11px; line-height: 1.15; width: auto; }
.ce1-table th,.ce1-table td { border: 1px solid #666; padding: 3px 6px; white-space: nowrap; }
.ce1-table th { font-weight: 600; }
.ce1-table td { text-align: right; }
.ce1-table td:first-child,.ce1-table td:nth-child(2),.ce1-table td:nth-child(3),.ce1-table td:nth-child(4) { text-align: left; }
</style>
""".strip()


def table_display(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.{digits}f}")
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
    html = html.replace("<td>", '<td style="border:1px solid #666;padding:3px 6px;white-space:nowrap;">')
    return (
        '<div class="ce1-table-wrap" '
        'style="max-width:100%;overflow-x:auto;margin:0.25rem 0 0.9rem 0;">'
        f"{html}</div>"
    )


def compact_mean(means: pd.DataFrame) -> pd.DataFrame:
    cols = ["month", "controller", "mode", "energy", "disc_%", "cold_%", "hot_%", "KH", "daily_peak", "overrides", "Tin_max", "tdyn_final"]
    return means[[col for col in cols if col in means.columns]]


def compact_district(districts: pd.DataFrame) -> pd.DataFrame:
    out = districts.copy()
    out["total_MWh"] = out["total_kwh"] / 1000.0
    cols = ["month", "controller", "mode", "total_MWh", "peak_kwh", "daily_profile_peak_kwh", "daily_profile_peak_hour", "load_factor"]
    return out[[col for col in cols if col in out.columns]]


def worst_rows(buildings: pd.DataFrame, month_name: str, n: int = 15) -> pd.DataFrame:
    cols = ["controller", "mode", "building", "occupant", "energy", "disc_%", "cold_%", "hot_%", "overrides", "Tin_max"]
    data = buildings[buildings["month"] == month_name].sort_values("disc_%", ascending=False).head(n)
    return data[[col for col in cols if col in data.columns]]


def image_cell(path: Path, title: str):
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{title}](../{path.as_posix()})")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    buildings = building_table()
    if buildings.empty:
        raise SystemExit("No occupant probability SAC artifacts found.")
    means = mean_table(buildings)
    frames = district_frames()
    districts = district_kpis(frames)
    rewards = sac_training_table()

    buildings.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all.csv", index=False)
    means.to_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv", index=False)
    districts.to_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv", index=False)
    if not rewards.empty:
        rewards.to_csv(ANALYSIS / f"{PREFIX}_sac_reward_by_episode.csv", index=False)

    comfort_paths = []
    for month in MONTHS:
        for run in RUNS:
            for mode in MODES:
                path = make_comfort_grid(month, run, mode)
                if path is not None:
                    comfort_paths.append((month, run, mode, path))
    district_paths = [(month, make_district_daily_plot(frames, month)) for month in MONTHS]
    reward_path = make_training_reward_plot(rewards)

    available_months = ", ".join(sorted(buildings["month"].unique()))
    cells = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}
{TABLE_STYLE}

SAC-only ablation for the occupant-loop overheating failure. The old SAC rows use the previous `Tref +/- 2 C` normalized guard reward. The new rows replace the guard terms by fitted expected override probabilities:

`p_cold(Tin) = min(0.002*exp(b*max(Tpref - Tin, 0)), p_max)`

`p_hot(Tin) = min(0.002*exp(b*max(Tin - Tpref, 0)), p_max)`

`D_occ = 0.1*((Tin - Tref)/2)^2 + 0.7*p_cold(Tin) + 0.2*p_hot(Tin)`

`reward = -D_occ`

For `bspline_tdyn`, `Tref` is the occupancy-gated Zhang-style dynamic reference; for `without_tdyn`, it is the active thermostat reference with occupant overrides in the loop. The probability curve uses each fitted occupant's `Tpref`, `b`, and `p_max`.

Available months in this section: `{available_months}`.

Mean building KPIs:

{table_display(compact_mean(means), digits=3)}

District aggregate KPIs:

{table_display(compact_district(districts), digits=3)}

Saved compact tables:

- `analysis/{PREFIX}_building_kpis_all.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_sac_reward_by_episode.csv`
"""
        )
    ]

    for month, path in district_paths:
        if path is not None:
            cells.append(image_cell(path, f"{MONTHS[month]['display']} SAC mean hourly district load"))
    for _, run, mode, path in comfort_paths:
        cells.append(image_cell(path, f"{run['display']} {mode} comfort"))
    if reward_path is not None:
        cells.append(image_cell(reward_path, "SAC training raw reward"))

    for month_name in sorted(buildings["month"].unique()):
        cells.append(
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n### {month_name} Worst SAC Comfort Rows\n\n"
                f"Top 15 by discomfort shown here. Full per-building KPIs are saved in "
                f"`analysis/{PREFIX}_building_kpis_all.csv`.\n\n"
                f"{TABLE_STYLE}\n\n{table_display(worst_rows(buildings, month_name), digits=3)}"
            )
        )

    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [cell for cell in nb.cells if MARKER not in str(cell.get("source", ""))]
    nb.cells.extend(cells)
    nbf.write(nb, NOTEBOOK)
    print(f"Appended {NOTEBOOK}")


if __name__ == "__main__":
    main()
