#!/usr/bin/env python
"""Append deterministic January SAC evaluation comfort plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries


PREFIX = "ce1_occ_D010702_sac_noreset_jan20_janeval_det"
MARKER = f"## CE1 SAC No-Reset Deterministic January Evaluation - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOK = Path('experiments/analysis/08_sac_results.ipynb')
SUMMARY_CSV = ANALYSIS / "ce1_occ_D010702_sac_noreset_jan20_deterministic_january_eval_summary.csv"
BY_BUILDING_CSV = ANALYSIS / "ce1_occ_D010702_sac_noreset_jan20_deterministic_january_eval_by_building.csv"

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]

RUNS = [
    {
        "mode": "det_jan_no_occupant",
        "scenario": "no_occupant",
        "label": f"{PREFIX}_no_occupant",
        "title": "SAC no occupant, deterministic actor | January",
    },
    {
        "mode": "det_jan_bspline_tdyn",
        "scenario": "occupant_present",
        "label": f"{PREFIX}_bspline_tdyn",
        "title": "SAC occupant B-spline Tdyn, deterministic actor | January",
    },
]

TABLE_STYLE = """
<style>
.ce1-compact-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 0.25rem 0 0.9rem 0;
}
.ce1-compact-table-wrap table {
  border-collapse: collapse;
  font-size: 11px;
  line-height: 1.15;
  width: auto;
}
.ce1-compact-table-wrap th,
.ce1-compact-table-wrap td {
  border: 1px solid #666;
  padding: 3px 6px;
  white-space: nowrap;
}
.ce1-compact-table-wrap th {
  font-weight: 600;
}
</style>
""".strip()


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def occupant_for_display(display: str, mode: str) -> str:
    if mode == "det_jan_no_occupant":
        return "none"
    number = int(str(display).split()[-1])
    return OCCUPANTS[(number - 1) % len(OCCUPANTS)]


def raw_dir(building: dict, run: dict) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building["result_id"] / run["scenario"] / run["label"]


def read_rollout(building: dict, run: dict) -> pd.DataFrame:
    path = raw_dir(building, run) / "rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compact_table(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return f'<div class="ce1-compact-table-wrap">{out.to_html(index=False, escape=True, border=0)}</div>'


def series(frame: pd.DataFrame, *names: str) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), np.nan, dtype=float)


def make_comfort_grid(run: dict) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()

    for ax, building in zip(axes, BUILDINGS):
        data = read_rollout(building, run)
        x = series(data, "time_step")
        if np.isnan(x).all():
            x = np.arange(len(data), dtype=float)
        tin = series(data, "indoor_temperature", "tin_before_action")
        setpoint = series(data, "effective_setpoint", "tsp_before_action", "baseline_setpoint")

        n = min(len(x), len(tin), len(setpoint))
        ax.plot(x[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(x[:n], setpoint[:n], lw=0.75, color="tab:orange", label="setpoint")

        if run["mode"] == "det_jan_bspline_tdyn":
            tdyn = series(data, "tdyn")
            n_tdyn = min(n, len(tdyn))
            ax.plot(x[:n_tdyn], tdyn[:n_tdyn], lw=0.9, color="black", label="Tdyn")

        occ = "none" if run["scenario"] == "no_occupant" else occupant_for_building(building)
        ax.set_title(f"{building['display']} | {occ}", fontsize=7)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=7)
    fig.suptitle(run["title"], fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['mode']}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def deterministic_deployment_check() -> pd.DataFrame:
    checks = [
        ("February", "no_occupant_base", "no_occupant", "ce1_final_no_occ_D010702_sac_safepi9_a1"),
        ("February", "no_occupant_bad_cont", "no_occupant", "ce1_final_no_occ_D010702_sac_safepi9_bad10_a1"),
        ("February", "without_tdyn", "occupant_present", "ce1_occ_D010702_sac_jan20_feb_without_tdyn"),
        ("February", "bspline_tdyn", "occupant_present", "ce1_occ_D010702_sac_jan20_feb_bspline_tdyn"),
        ("March", "no_occupant", "no_occupant", "ce1_final_no_occ_D010702_sac_feb10_mar_a1"),
        ("March", "without_tdyn", "occupant_present", "ce1_occ_D010702_sac_feb10_mar_without_tdyn"),
        ("March", "bspline_tdyn", "occupant_present", "ce1_occ_D010702_sac_feb10_mar_bspline_tdyn"),
    ]
    rows = []
    for month, mode, scenario, label in checks:
        found = 0
        rows_count = 0
        max_diff = 0.0
        phases: set[str] = set()
        for building in BUILDINGS:
            path = RESULTS / "raw" / "sac_merlin" / building["result_id"] / scenario / label / "rollout.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            found += 1
            rows_count += len(frame)
            if "phase" in frame.columns:
                phases.update(frame["phase"].dropna().astype(str).unique().tolist())
            if {"policy_action", "policy_mean_action"}.issubset(frame.columns):
                diff = (frame["policy_action"] - frame["policy_mean_action"]).abs().max()
                max_diff = max(max_diff, float(diff))
        rows.append(
            {
                "month": month,
                "mode": mode,
                "buildings_found": found,
                "rows": rows_count,
                "max_abs_policy_action_minus_mean": max_diff,
                "phases": ", ".join(sorted(phases)),
            }
        )
    return pd.DataFrame(rows)


def append_once(cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [cell for cell in nb.cells if MARKER not in cell.get("source", "")]
    nb.cells.extend(cells)
    nbf.write(nb, NOTEBOOK)


def image_cell(path: Path, alt: str) -> nbf.NotebookNode:
    rel = Path("..") / path
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{alt}]({rel.as_posix()})")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_paths = [make_comfort_grid(run) for run in RUNS]
    summary = pd.read_csv(SUMMARY_CSV)
    by_building = pd.read_csv(BY_BUILDING_CSV)
    if "occupant" not in by_building.columns:
        by_building["occupant"] = [
            occupant_for_display(row.building, row.mode) for row in by_building[["building", "mode"]].itertuples()
        ]
    deployment_check = deterministic_deployment_check()
    deployment_check.to_csv(ANALYSIS / f"{PREFIX}_deployment_determinism_check.csv", index=False)

    show_cols = [
        "mode",
        "reward_mean",
        "action_mean",
        "action_std",
        "mae_to_setpoint",
        "mae_to_tracking_target",
        "cold_pct",
        "hot_pct",
        "tin_min",
        "tin_max",
        "policy_minus_mean_max_abs",
    ]
    compact_summary = summary[[col for col in show_cols if col in summary.columns]]
    worst = (
        by_building.sort_values(["reward_mean", "mae_to_setpoint"], ascending=[True, False])
        .head(10)[
            [
                "mode",
                "building",
                "occupant",
                "reward_mean",
                "action_mean",
                "mae_to_setpoint",
                "cold_pct",
                "hot_pct",
                "tin_min",
                "tin_max",
            ]
        ]
    )

    cells: list[nbf.NotebookNode] = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

Deterministic January evaluation of the final no-reset SAC actors.

This is not the stochastic training rollout. These plots use the saved `rollout.csv` files from deterministic actor evaluation: `policy_action == policy_mean_action`.

For the B-spline run, only three lines are shown: `Tin`, active setpoint, and full black `Tdyn`.

{TABLE_STYLE}

Deterministic January summary:

{compact_table(compact_summary, digits=4)}

Worst deterministic January building rows:

{compact_table(worst, digits=4)}

SAC deployment determinism check for the current comparison cell:

{compact_table(deployment_check, digits=4)}

Saved tables:

- `{SUMMARY_CSV.as_posix()}`
- `{BY_BUILDING_CSV.as_posix()}`
- `analysis/{PREFIX}_deployment_determinism_check.csv`
"""
        )
    ]
    cells.append(image_cell(plot_paths[0], "Deterministic January SAC no occupant comfort"))
    cells.append(image_cell(plot_paths[1], "Deterministic January SAC B-spline Tdyn comfort"))
    append_once(cells)
    print(f"Appended deterministic January SAC plots to {NOTEBOOK}")
    for path in plot_paths:
        print(path)
    print(ANALYSIS / f"{PREFIX}_deployment_determinism_check.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
