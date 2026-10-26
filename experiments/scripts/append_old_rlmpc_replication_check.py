#!/usr/bin/env python3
"""Append the old RLMPC exact replication check to analysis notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries


OLD_LABEL = "ce1_time_bias_guard_1c_1r1c"
REP_LABEL = "ce1_time_bias_guard_1c_1r1c_rlmpc_replicate"
MARKER = f"## Old GNU-RL/RLMPC Replication Check - `{REP_LABEL}`"

ANALYSIS_DIR = Path("analysis")
RESULTS_DIR = Path("results")
FIG_DIR = ANALYSIS_DIR / f"{REP_LABEL}_check_figures"
BUILDINGS = dataset_building_entries(DATASET_NAME)
NOTEBOOKS = [
    ANALYSIS_DIR / "02_mpc_results.ipynb",
    ANALYSIS_DIR / "04_rlmpc_results.ipynb",
    ANALYSIS_DIR / "08_sac_results.ipynb",
    ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
    ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
    ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
]


def _raw_dir(building_id: str, label: str) -> Path:
    return RESULTS_DIR / "raw" / "rlmpc" / building_id / "no_occupant" / label


def _read_kpi_values(building_id: str, label: str) -> dict[str, float]:
    path = _raw_dir(building_id, label) / "kpis.csv"
    data = pd.read_csv(path)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    district = data[data.get("level", "") == "district"].copy()
    if district.empty:
        district = data
    return {
        str(row["cost_function"]): float(row["value"])
        for _, row in district.iterrows()
        if pd.notna(row["value"])
    }


def _per_building_kpis() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        vals = _read_kpi_values(building["result_id"], REP_LABEL)
        rows.append(
            {
                "building": building["display"],
                "energy": vals.get("electricity_consumption_total", np.nan),
                "disc_%": 100.0 * vals.get("discomfort_proportion", np.nan),
                "cold_%": 100.0 * vals.get("discomfort_cold_proportion", np.nan),
                "hot_%": 100.0 * vals.get("discomfort_hot_proportion", np.nan),
                "cold_delta_avg": vals.get("discomfort_cold_delta_average", np.nan),
                "hot_delta_avg": vals.get("discomfort_hot_delta_average", np.nan),
                "daily_peak": vals.get("daily_peak_average", np.nan),
                "ramping": vals.get("ramping_average", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS_DIR / f"{REP_LABEL}_per_building_kpis.csv", index=False)
    return out


def _markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data._"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(data.columns)
    rows = [headers] + data.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |", *[fmt(r) for r in rows[1:]]])


def _image(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.as_posix()})")


def _make_all25_comfort_overlay() -> Path:
    path = FIG_DIR / "comfort_overlay_all25_old_vs_replicate.png"
    fig, axes = plt.subplots(5, 5, figsize=(24, 16), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        bid = building["result_id"]
        old = pd.read_csv(_raw_dir(bid, OLD_LABEL) / "rollout.csv")
        rep = pd.read_csv(_raw_dir(bid, REP_LABEL) / "rollout.csv")
        x = old["time_step"].to_numpy() if "time_step" in old else np.arange(len(old))
        old_tin = pd.to_numeric(old["indoor_temperature"], errors="coerce").to_numpy(float)
        rep_tin = pd.to_numeric(rep["indoor_temperature"], errors="coerce").to_numpy(float)
        tsp = pd.to_numeric(old["effective_setpoint"], errors="coerce").to_numpy(float)
        tmin = pd.to_numeric(old["tmin"], errors="coerce").to_numpy(float)
        tmax = pd.to_numeric(old["tmax"], errors="coerce").to_numpy(float)
        n = min(len(x), len(old_tin), len(rep_tin), len(tsp), len(tmin), len(tmax))
        ax.plot(x[:n], old_tin[:n], lw=0.75, color="tab:blue", label="old Tin")
        ax.plot(x[:n], rep_tin[:n], lw=0.75, ls="--", color="tab:red", label="replicate Tin")
        ax.plot(x[:n], tsp[:n], lw=0.65, color="tab:orange", label="effective setpoint")
        ax.fill_between(x[:n], tmin[:n], tmax[:n], color="0.85", alpha=0.45, label="comfort band")
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=7, loc="best")
    for ax in axes[len(BUILDINGS) :]:
        ax.axis("off")
    fig.suptitle("Old GNU-RL/RLMPC vs replicate: all 25 comfort rollouts", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _mean_hourly_diff_table() -> pd.DataFrame:
    path = ANALYSIS_DIR / f"{REP_LABEL}_mean_hourly_district_profile_diff.csv"
    data = pd.read_csv(path)
    cols = ["hour", "net_old", "net_replicate", "net_diff", "heat_old", "heat_replicate", "heat_diff"]
    return data[cols].copy()


def _summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verdict = pd.read_csv(ANALYSIS_DIR / f"{REP_LABEL}_replication_verdict.csv")
    compact = pd.read_csv(ANALYSIS_DIR / f"{REP_LABEL}_compact_kpi_diff.csv")
    csv_diff = pd.read_csv(ANALYSIS_DIR / f"{REP_LABEL}_csv_diff_summary.csv")
    file_summary = (
        csv_diff.groupby("file", as_index=False)
        .agg(exact=("exact_dataframe_equal", "all"), max_abs_numeric_diff=("max_abs_numeric_diff", "max"))
        .sort_values("file")
    )
    json_diff = pd.read_csv(ANALYSIS_DIR / f"{REP_LABEL}_metrics_json_diff.csv")
    json_changed = json_diff[json_diff["changed"].astype(bool)].copy()
    hourly = _mean_hourly_diff_table()
    return verdict, compact, file_summary, json_changed, hourly


def _append_once(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for idx, cell in enumerate(notebook.cells):
        if MARKER in str(cell.get("source", "")):
            notebook.cells[idx : idx + len(cells)] = cells
            nbf.write(notebook, path)
            print(f"Updated {MARKER} in {path}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {MARKER} to {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all25_path = _make_all25_comfort_overlay()
    verdict, compact, file_summary, json_changed, hourly = _summary_tables()
    per_building = _per_building_kpis()
    mean_row = per_building.drop(columns=["building"]).mean(numeric_only=True).to_frame("mean").T
    mean_row.insert(0, "controller", "GNU-RL/RLMPC old replicated")

    json_note = json_changed[["building", "key", "old", "replicate"]].head(5)

    markdown = f"""{MARKER}

This section checks whether the old working GNU-RL/RLMPC time-bias experiment can be reproduced exactly.

Old label: `{OLD_LABEL}`

Replicate label: `{REP_LABEL}`

Recipe used for the rerun:

- Controller: GNU-RL/RLMPC, old 1R1C pipeline.
- Offline checkpoints: `results/models/rlmpc/offline/building_*/rlmpc__no_occupant.pt`.
- RC model: `results/models/rc/1state_rc_{{short_name}}_january.json`.
- Horizon: `6`.
- January PI dataset bootstrap, then February deployment.
- Fixed matched balanced inner objective, so PPO cost-parameter updates are skipped.
- Grid stress: `baseline_district_daily_mean_exp2_norm`.
- Reward/cost shape:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

`J_k = 0.5*D_k + 0.5*GridStress_k*P_heat,k/5`, `reward_k = -J_k`

with `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and true-setpoint guard `Tsp +/- 2 C`.

Replication verdict:

{_markdown_table(verdict, digits=6)}

Saved CSV exactness check:

{_markdown_table(file_summary, digits=6)}

Mean building KPIs for the replicated old RLMPC run:

{_markdown_table(mean_row, digits=4)}

Compact KPI old-vs-replicate diff:

{_markdown_table(compact, digits=6)}

Per-building KPIs:

{_markdown_table(per_building, digits=4)}

Metric JSON changed rows are only the expected `case` string change. First rows:

{_markdown_table(json_note, digits=4)}

Mean hourly district profile old-vs-replicate:

{_markdown_table(hourly, digits=6)}

Saved audit files:

- `analysis/{REP_LABEL}_replication_verdict.csv`
- `analysis/{REP_LABEL}_csv_diff_summary.csv`
- `analysis/{REP_LABEL}_kpi_diff.csv`
- `analysis/{REP_LABEL}_metrics_json_diff.csv`
- `analysis/{REP_LABEL}_mean_hourly_district_profile_diff.csv`
- `analysis/{REP_LABEL}_per_building_kpis.csv`
"""

    image_cells = [
        _image(all25_path, "All 25 old RLMPC vs replicate comfort rollouts"),
        _image(FIG_DIR / "comfort_overlay_representative_buildings.png", "Representative comfort overlay old vs replicate"),
        _image(FIG_DIR / "mean_hourly_district_load_old_vs_replicate.png", "Mean hourly district load old vs replicate"),
        _image(FIG_DIR / "mean_hourly_district_load_delta.png", "Mean hourly district load delta"),
    ]
    cells = [nbf.v4.new_markdown_cell(markdown), *image_cells]

    for notebook in NOTEBOOKS:
        if notebook.exists():
            _append_once(notebook, cells)


if __name__ == "__main__":
    main()
