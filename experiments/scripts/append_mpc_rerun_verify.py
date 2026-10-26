from __future__ import annotations

import math
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SCENARIO = "no_occupant"
ORIGINAL_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman"
VERIFY_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_mpc_rerun_verify"
RAW_ROOT = ROOT / "results" / "raw" / "mpc"
SUMMARY_CSV = ROOT / "results" / "summaries" / "mpc" / f"{SCENARIO}_summary_all_buildings.csv"
FIG_DIR = ROOT / "figures" / "mpc_rerun_verify"
ANALYSIS_DIR = ROOT / "analysis"
NOTEBOOK_PATH = ANALYSIS_DIR / "02_mpc_results.ipynb"


def building_ids() -> list[str]:
    return [f"building_{i:02d}" for i in range(1, 26)]


def raw_dir(building_id: str, label: str) -> Path:
    return RAW_ROOT / building_id / SCENARIO / label


def read_rollout(building_id: str, label: str) -> pd.DataFrame:
    path = raw_dir(building_id, label) / "rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_kpis(building_id: str, label: str) -> pd.DataFrame:
    path = raw_dir(building_id, label) / "kpis.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_baseline_district() -> pd.DataFrame:
    path = ROOT / "results" / "raw" / "citylearn_baseline" / SCENARIO / "baseline" / "district_rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0),
            "controller": "CityLearn baseline",
        }
    )


def read_mpc_district() -> pd.DataFrame:
    loads = []
    time = None
    for bid in building_ids():
        rollout = read_rollout(bid, VERIFY_LABEL)
        values = pd.to_numeric(rollout["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        current_time = pd.to_numeric(rollout["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if time is None:
            time = current_time
        n = min(len(time), len(current_time), len(values))
        time = time[:n]
        loads.append(values[:n])
    if time is None or not loads:
        raise RuntimeError("No MPC rollouts found.")
    n = min(len(time), *(len(values) for values in loads))
    baseline = read_baseline_district()
    return pd.DataFrame(
        {
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([values[:n] for values in loads], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "controller": "Fresh MPC rerun",
        }
    )


def mean_hourly_profile(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(hour=frame["time_step"].to_numpy(dtype=int) % 24)
        .groupby("hour", as_index=False)[["load", "target"]]
        .mean(numeric_only=True)
    )


def pivot_kpis(kpis: pd.DataFrame) -> dict[str, float]:
    rows = kpis[kpis["level"].eq("building")]
    return rows.set_index("cost_function")["value"].to_dict()


def collect_verify_summary() -> pd.DataFrame:
    combined_summary = pd.read_csv(SUMMARY_CSV) if SUMMARY_CSV.exists() else pd.DataFrame()
    combined_summary = combined_summary[combined_summary.get("case", pd.Series(dtype=str)).eq(VERIFY_LABEL)]
    rows = []
    for i, bid in enumerate(building_ids(), start=1):
        kpi_map = pivot_kpis(read_kpis(bid, VERIFY_LABEL))
        rollout = read_rollout(bid, VERIFY_LABEL)
        summary_row = combined_summary[combined_summary.get("building_id", pd.Series(dtype=str)).eq(bid)]
        summary_map = summary_row.iloc[0].to_dict() if not summary_row.empty else {}
        rows.append(
            {
                "building": f"Building {i}",
                "building_id": bid,
                "electricity_consumption_total": summary_map.get(
                    "electricity_consumption_total",
                    kpi_map.get("electricity_consumption_total", np.nan),
                ),
                "discomfort_%": 100.0
                * summary_map.get("discomfort_proportion", kpi_map.get("discomfort_proportion", np.nan)),
                "cold_%": 100.0
                * summary_map.get(
                    "discomfort_cold_proportion",
                    kpi_map.get("discomfort_cold_proportion", np.nan),
                ),
                "hot_%": 100.0
                * summary_map.get(
                    "discomfort_hot_proportion",
                    kpi_map.get("discomfort_hot_proportion", np.nan),
                ),
                "kelvin_hours": summary_map.get("kelvin_hours", np.nan),
                "daily_peak_average": summary_map.get("daily_peak_average", np.nan),
                "tin_min": rollout["indoor_temperature"].min(),
                "tin_max": rollout["indoor_temperature"].max(),
            }
        )
    return pd.DataFrame(rows)


def compare_original_vs_verify() -> pd.DataFrame:
    rows = []
    for i, bid in enumerate(building_ids(), start=1):
        old_rollout = read_rollout(bid, ORIGINAL_LABEL)
        new_rollout = read_rollout(bid, VERIFY_LABEL)
        common_cols = [
            c
            for c in old_rollout.columns.intersection(new_rollout.columns)
            if pd.api.types.is_numeric_dtype(old_rollout[c])
            and pd.api.types.is_numeric_dtype(new_rollout[c])
        ]
        n = min(len(old_rollout), len(new_rollout))
        rollout_max_abs_diff = 0.0
        for col in common_cols:
            diff = (old_rollout[col].iloc[:n].to_numpy() - new_rollout[col].iloc[:n].to_numpy())
            if diff.size:
                rollout_max_abs_diff = max(rollout_max_abs_diff, float(np.nanmax(np.abs(diff))))

        old_kpis = read_kpis(bid, ORIGINAL_LABEL)
        new_kpis = read_kpis(bid, VERIFY_LABEL)
        merged = old_kpis.merge(
            new_kpis,
            on=["cost_function", "name", "level"],
            suffixes=("_original", "_rerun"),
        )
        kpi_diff = merged["value_original"].to_numpy() - merged["value_rerun"].to_numpy()
        kpi_max_abs_diff = float(np.nanmax(np.abs(kpi_diff))) if kpi_diff.size else math.nan
        rows.append(
            {
                "building": f"Building {i}",
                "rollout_max_abs_diff": rollout_max_abs_diff,
                "kpi_max_abs_diff": kpi_max_abs_diff,
                "rollout_rows_original": len(old_rollout),
                "rollout_rows_rerun": len(new_rollout),
            }
        )
    return pd.DataFrame(rows)


def plot_comfort_grid(summary: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{VERIFY_LABEL}_comfort_all_buildings.png"
    fig, axes = plt.subplots(5, 5, figsize=(20, 13), sharex=True, sharey=False)
    axes = axes.ravel()

    for i, bid in enumerate(building_ids(), start=1):
        ax = axes[i - 1]
        df = read_rollout(bid, VERIFY_LABEL)
        t = np.arange(len(df))
        ax.fill_between(t, df["tmin"], df["tmax"], color="0.85", alpha=0.7, label="+/- comfort band")
        ax.plot(t, df["indoor_temperature"], color="#1f77b4", lw=0.8, label="Indoor temperature")
        ax.plot(t, df["scheduled_setpoint"], color="black", lw=0.7, ls="--", label="True setpoint")
        ax.plot(t, df["tracking_setpoint"], color="#ff7f0e", lw=0.7, label="Biased target")
        row = summary[summary["building"].eq(f"Building {i}")].iloc[0]
        ax.set_title(f"B{i}: {row['discomfort_%']:.1f}% discomfort", fontsize=8)
        ax.grid(True, alpha=0.25)
        if i == 1:
            ax.legend(fontsize=7, loc="upper left")
        if (i - 1) % 5 == 0:
            ax.set_ylabel("degC", fontsize=8)
        if i > 20:
            ax.set_xlabel("February step", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("MPC rerun verification | 2R2C+Kalman | February comfort", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_building_one() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{VERIFY_LABEL}_building_01_comfort.png"
    df = read_rollout("building_01", VERIFY_LABEL)
    t = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(t, df["tmin"], df["tmax"], color="0.85", alpha=0.7, label="+/- comfort band")
    ax.plot(t, df["indoor_temperature"], color="#1f77b4", lw=1.0, label="Indoor temperature")
    ax.plot(t, df["scheduled_setpoint"], color="black", lw=0.9, ls="--", label="True setpoint")
    ax.plot(t, df["tracking_setpoint"], color="#ff7f0e", lw=0.9, label="Biased target")
    ax.set_title("Building 1 | fresh MPC rerun | February comfort")
    ax.set_xlabel("February step")
    ax.set_ylabel("Temperature [degC]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_daily_mean_load_profile() -> tuple[Path, pd.DataFrame]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{VERIFY_LABEL}_district_mean_hourly_profile.png"
    baseline = read_baseline_district()
    mpc = read_mpc_district()

    baseline_daily = mean_hourly_profile(baseline)
    mpc_daily = mean_hourly_profile(mpc)
    profile = pd.concat(
        [
            baseline_daily.assign(controller="CityLearn baseline"),
            mpc_daily.assign(controller="Fresh MPC rerun"),
        ],
        ignore_index=True,
    )
    profile.to_csv(ANALYSIS_DIR / f"{VERIFY_LABEL}_district_mean_hourly_profile.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(baseline_daily["hour"], baseline_daily["load"], marker="o", lw=1.4, label="CityLearn baseline")
    ax.plot(mpc_daily["hour"], mpc_daily["load"], marker="o", lw=1.4, label="Fresh MPC rerun")
    ax.plot(mpc_daily["hour"], mpc_daily["target"], marker="o", lw=1.0, ls="--", color="black", label="CE1 district target")
    ax.set_title("Mean hourly district load - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, profile


def markdown_table(df: pd.DataFrame, columns: list[str], rows: int | None = None) -> str:
    sub = df.loc[:, columns].copy()
    if rows is not None:
        sub = sub.head(rows)
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            sub[col] = sub[col].map(lambda x: f"{x:.6g}" if pd.notna(x) else "")
    return sub.to_markdown(index=False)


def append_notebook(summary: pd.DataFrame, diff: pd.DataFrame, fig_grid: Path, fig_b1: Path, fig_daily: Path, profile: pd.DataFrame) -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    title = f"## MPC Rerun Verification - `{VERIFY_LABEL}`"
    nb.cells = [
        cell
        for cell in nb.cells
        if not (cell.cell_type == "markdown" and isinstance(cell.source, str) and title in cell.source)
    ]

    mean = summary[
        [
            "electricity_consumption_total",
            "discomfort_%",
            "cold_%",
            "hot_%",
            "kelvin_hours",
            "daily_peak_average",
            "tin_min",
            "tin_max",
        ]
    ].mean(numeric_only=True)
    mean_df = pd.DataFrame([{"controller": "Fresh MPC rerun", **mean.to_dict()}])

    worst = summary.sort_values("discomfort_%", ascending=False)
    diff_summary = pd.DataFrame(
        [
            {
                "check": "max rollout abs diff vs stored MPC",
                "value": diff["rollout_max_abs_diff"].max(),
            },
            {
                "check": "max KPI abs diff vs stored MPC",
                "value": diff["kpi_max_abs_diff"].max(),
            },
        ]
    )

    summary_path = ANALYSIS_DIR / f"{VERIFY_LABEL}_building_kpis.csv"
    diff_path = ANALYSIS_DIR / f"{VERIFY_LABEL}_original_diff.csv"
    profile_path = ANALYSIS_DIR / f"{VERIFY_LABEL}_district_mean_hourly_profile.csv"
    summary.to_csv(summary_path, index=False)
    diff.to_csv(diff_path, index=False)

    rel_grid = Path("..") / fig_grid.relative_to(ROOT)
    rel_b1 = Path("..") / fig_b1.relative_to(ROOT)
    rel_daily = Path("..") / fig_daily.relative_to(ROOT)
    daily_pivot = profile.pivot(index="hour", columns="controller", values="load").reset_index()
    target = profile[profile["controller"].eq("Fresh MPC rerun")][["hour", "target"]]
    daily_pivot = daily_pivot.merge(target, on="hour", how="left").rename(columns={"target": "CE1 target"})

    source = f"""{title}

Fresh rerun of **only** the MPC controller, using the recovered working setup:

`2R2C + Kalman`, January RC fits, February deployment, `GridStress=baseline_daily_mean_exp`, exponential beta `2.0`, `--raw-power-cost`, `power_scale=5`, `comfort_weight=0.5`, `shift_weight=0.5`, and the time-bias guard comfort cost.

The stage objective is:

`J_k = 0.5 * [3*((Tin_k - (Tsp_k + bias_k))/2)^2 + 60*(max(Tsp_k - 2 - Tin_k, 0)/1.5)^2 + 15*(max(Tin_k - Tsp_k - 2, 0)/2)^2] + 0.5 * GridStress_k * P_heat_k / 5`

Mean building KPIs:

{markdown_table(mean_df, ["controller", "electricity_consumption_total", "discomfort_%", "cold_%", "hot_%", "kelvin_hours", "daily_peak_average", "tin_min", "tin_max"])}

Worst comfort buildings in the fresh rerun:

{markdown_table(worst, ["building", "discomfort_%", "cold_%", "hot_%", "electricity_consumption_total", "tin_min", "tin_max"], rows=8)}

Fresh rerun vs stored `{ORIGINAL_LABEL}`:

{markdown_table(diff_summary, ["check", "value"])}

Saved tables:

- `{summary_path.relative_to(ROOT)}`
- `{diff_path.relative_to(ROOT)}`
- `{profile_path.relative_to(ROOT)}`

Mean hourly district load:

{markdown_table(daily_pivot, ["hour", "CityLearn baseline", "Fresh MPC rerun", "CE1 target"])}

Clean comfort plots from the fresh rerun:

![Fresh MPC comfort all buildings]({rel_grid.as_posix()})

![Fresh MPC Building 1 comfort]({rel_b1.as_posix()})

Daily mean load profile:

![Fresh MPC mean hourly district load]({rel_daily.as_posix()})
"""
    nb.cells.append(nbformat.v4.new_markdown_cell(source))
    nbformat.write(nb, NOTEBOOK_PATH)


def main() -> None:
    summary = collect_verify_summary()
    diff = compare_original_vs_verify()
    fig_grid = plot_comfort_grid(summary)
    fig_b1 = plot_building_one()
    fig_daily, profile = plot_daily_mean_load_profile()
    append_notebook(summary, diff, fig_grid, fig_b1, fig_daily, profile)

    print("Mean building KPIs:")
    print(
        summary[
            [
                "electricity_consumption_total",
                "discomfort_%",
                "cold_%",
                "hot_%",
                "kelvin_hours",
                "daily_peak_average",
                "tin_min",
                "tin_max",
            ]
        ]
        .mean()
        .to_string()
    )
    print("\nWorst comfort buildings:")
    print(
        summary.sort_values("discomfort_%", ascending=False)
        .head(8)[
            [
                "building",
                "discomfort_%",
                "cold_%",
                "hot_%",
                "electricity_consumption_total",
                "tin_min",
                "tin_max",
            ]
        ]
        .to_string(index=False)
    )
    print("\nFresh-vs-stored diff:")
    print(diff[["building", "rollout_max_abs_diff", "kpi_max_abs_diff"]].to_string(index=False))
    print(f"\nSaved {fig_grid}")
    print(f"Saved {fig_b1}")
    print(f"Saved {fig_daily}")
    print(f"Appended {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
