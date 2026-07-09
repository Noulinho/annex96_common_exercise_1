#!/usr/bin/env python
"""Append 2R2C DiffMPC-vs-explicit-MPC action-parity diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIG_DIR = ROOT / "figures" / "ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate"
MARKER = "## CE1 2R2C+Kalman DiffMPC Action-Parity Gate"

STRICT_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_rlmpc_parity_h24_west_guard"
SOFT_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_rlmpc_parity_h24_west_guard0"

RUNS = [
    {
        "label": STRICT_LABEL,
        "controller": "DiffMPC h24, future Tsp, Kalman+w, exact guard penalty",
        "short": "exact guard",
    },
    {
        "label": SOFT_LABEL,
        "controller": "DiffMPC h24, future Tsp, Kalman+w, no linear guard penalty",
        "short": "guard penalty 0",
    },
]


def _read_detail(label: str) -> pd.DataFrame:
    path = ANALYSIS / f"{label}_action_parity.csv"
    data = pd.read_csv(path)
    data = data[data["building"].ne("building")].copy()
    numeric = [
        "step",
        "time_step",
        "u_mpc",
        "u_diffmpc",
        "action_error",
        "abs_action_error",
        "tin_obs",
        "tin_mpc_rollout",
        "tin_replay_abs_error",
        "setpoint",
        "shift_signal",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["u_mpc", "u_diffmpc", "action_error", "abs_action_error"])


def _read_summary(label: str) -> pd.DataFrame:
    path = ANALYSIS / f"{label}_action_parity_summary.csv"
    data = pd.read_csv(path)
    data = data[data["building"].ne("building")].copy()
    for column in data.columns:
        if column not in {"building", "building_id"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _read_aggregate(label: str) -> dict[str, float]:
    path = ANALYSIS / f"{label}_action_parity_aggregate.json"
    return json.loads(path.read_text())


def _markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data._"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    rows = [list(data.columns)] + data.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |"

    return "\n".join(
        [fmt(rows[0]), "| " + " | ".join("-" * width for width in widths) + " |", *[fmt(row) for row in rows[1:]]]
    )


def _image(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.relative_to(ROOT).as_posix()})")


def _combined_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_rows = []
    worst_rows = []
    for run in RUNS:
        agg = _read_aggregate(run["label"])
        aggregate_rows.append(
            {
                "controller": run["short"],
                "buildings": agg["buildings"],
                "steps": agg["steps"],
                "action_mae_mean": agg["action_mae_mean"],
                "action_rmse_mean": agg["action_rmse_mean"],
                "action_max_abs_global": agg["action_max_abs_global"],
                "qpth_warnings": agg["qpth_warning_count_total"],
                "horizon_min": agg["checkpoint_horizon_min"],
                "horizon_max": agg["checkpoint_horizon_max"],
                "guard_penalty": agg["time_bias_guard_penalty"],
            }
        )
        summary = _read_summary(run["label"])
        worst = summary.sort_values("action_mae", ascending=False).head(8).copy()
        worst.insert(0, "controller", run["short"])
        worst_rows.append(
            worst[
                [
                    "controller",
                    "building",
                    "steps",
                    "action_mae",
                    "action_rmse",
                    "action_max_abs",
                    "qpth_warning_count",
                ]
            ]
        )
    aggregate = pd.DataFrame(aggregate_rows)
    worst_buildings = pd.concat(worst_rows, ignore_index=True)
    aggregate.to_csv(ANALYSIS / "ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate_summary.csv", index=False)
    worst_buildings.to_csv(
        ANALYSIS / "ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate_worst_buildings.csv",
        index=False,
    )
    return aggregate, worst_buildings


def _plot_action_mae() -> Path:
    summaries = []
    for run in RUNS:
        summary = _read_summary(run["label"])[["building", "action_mae"]].copy()
        summary["controller"] = run["short"]
        summary["building_num"] = summary["building"].str.extract(r"(\d+)").astype(int)
        summaries.append(summary)
    data = pd.concat(summaries, ignore_index=True).sort_values("building_num")
    pivot = data.pivot(index="building_num", columns="controller", values="action_mae")

    path = FIG_DIR / "action_mae_by_building.png"
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.38
    for i, column in enumerate(pivot.columns):
        ax.bar(x + (i - 0.5) * width, pivot[column].to_numpy(dtype=float), width=width, label=column)
    ax.axhline(0.05, color="black", lw=1.0, ls="--", alpha=0.7, label="0.05 action MAE target")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Building")
    ax.set_ylabel("Mean absolute action error")
    ax.set_title("DiffMPC first-action parity against explicit MPC")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_qpth_warnings() -> Path:
    summaries = []
    for run in RUNS:
        summary = _read_summary(run["label"])[["building", "qpth_warning_count"]].copy()
        summary["controller"] = run["short"]
        summary["building_num"] = summary["building"].str.extract(r"(\d+)").astype(int)
        summaries.append(summary)
    data = pd.concat(summaries, ignore_index=True).sort_values("building_num")
    pivot = data.pivot(index="building_num", columns="controller", values="qpth_warning_count")

    path = FIG_DIR / "qpth_warnings_by_building.png"
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.38
    for i, column in enumerate(pivot.columns):
        ax.bar(x + (i - 0.5) * width, pivot[column].to_numpy(dtype=float), width=width, label=column)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Building")
    ax.set_ylabel("qpth inaccurate-solution warnings")
    ax.set_title("Solver warning count during same-state action replay")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_scatter() -> Path:
    path = FIG_DIR / "mpc_vs_diffmpc_action_scatter.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    for ax, run in zip(axes, RUNS):
        data = _read_detail(run["label"])
        step = max(1, len(data) // 4500)
        sample = data.iloc[::step]
        ax.scatter(sample["u_mpc"], sample["u_diffmpc"], s=6, alpha=0.25, edgecolors="none")
        ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--")
        ax.set_title(run["short"])
        ax.set_xlabel("Explicit MPC action")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("DiffMPC action")
    fig.suptitle("Same-state first action: explicit MPC vs DiffMPC")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_action_traces() -> Path:
    path = FIG_DIR / "building1_first_week_action_trace.png"
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.8), sharex=True, sharey=True)
    for ax, run in zip(axes, RUNS):
        data = _read_detail(run["label"])
        data = data[data["building_id"].eq("building_01")].head(168)
        ax.plot(data["step"], data["u_mpc"], label="explicit MPC", lw=1.5)
        ax.plot(data["step"], data["u_diffmpc"], label="DiffMPC", lw=1.2)
        ax.set_title(run["short"])
        ax.set_ylabel("action")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("February replay step")
    fig.suptitle("Building 1 same-state first-action trace")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _append_once(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if MARKER in str(cell.get("source", "")):
            notebook.cells[index : index + len(cells)] = cells
            nbf.write(notebook, path)
            print(f"Updated {MARKER} in {path.relative_to(ROOT)}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {MARKER} to {path.relative_to(ROOT)}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    aggregate, worst = _combined_tables()
    mae_plot = _plot_action_mae()
    warning_plot = _plot_qpth_warnings()
    scatter_plot = _plot_scatter()
    trace_plot = _plot_action_traces()

    overview = f"""{MARKER}

Action-parity check before treating RLMPC as a fair comparator to explicit MPC.

Both DiffMPC variants here use:

- horizon `24`
- the future February setpoint sequence across the horizon, not current-setpoint repetition
- the same fitted 2R2C+Kalman checkpoints cloned to horizon 24
- the same time-biased target schedule and `baseline_district_daily_mean_exp2_norm` grid-stress profile
- the same Kalman observer plus explicit-MPC-style residual correction gain `0.2`
- same-state replay: the environment is stepped with explicit MPC actions, and DiffMPC is queried on those exact states

The strict row also applies the explicit-MPC-style large guard slack penalty (`100000`). The second row removes only that linear guard penalty to test whether the parity failure is mainly qpth numerical instability.

Action-parity aggregate:

{_markdown_table(aggregate, digits=4)}

Worst buildings by first-action MAE:

{_markdown_table(worst, digits=4)}

Verdict: this parity gate fails. Even after horizon 24, future setpoint lookahead, the matched grid-stress term, and the Kalman/residual update, DiffMPC first actions do not match explicit MPC. The strict exact-guard formulation also creates many qpth inaccurate-solution warnings. Therefore the current RLMPC deployment should not be presented as a fair same-cost comparison against the explicit MPC controller yet.

Saved tables:

- `analysis/ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate_summary.csv`
- `analysis/ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate_worst_buildings.csv`
"""
    cells = [
        nbf.v4.new_markdown_cell(overview),
        _image(mae_plot, "Action MAE by building"),
        _image(warning_plot, "qpth warnings by building"),
        _image(scatter_plot, "Explicit MPC vs DiffMPC first-action scatter"),
        _image(trace_plot, "Building 1 first-week same-state action trace"),
    ]
    for notebook in [ANALYSIS / "02_mpc_results.ipynb", ANALYSIS / "04_rlmpc_results.ipynb"]:
        _append_once(notebook, cells)

    print("Action-parity aggregate:")
    print(aggregate.to_string(index=False))
    print("\nWorst buildings:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
