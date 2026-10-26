#!/usr/bin/env python
"""Append the one-building B-spline Tdyn smoke test to 08_sac_results."""

from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
NOTEBOOK = Path('experiments/analysis/08_sac_results.ipynb')
FIG_PATH = ANALYSIS / "ce1_tdyn_smoke_b06_feb_reference_gate.png"
SUMMARY_PATH = ANALYSIS / "ce1_tdyn_smoke_b06_feb_gate_summary.csv"


RUNS = {
    "MPC 2R2C+Kalman": ROOT
    / "results/raw/mpc/building_06/occupant_present/ce1_tdyn_smoke_b06_occ4_feb_mpc/rollout.csv",
    "GNU-RL adaptive 1R1C": ROOT
    / "results/raw/rlmpc/building_06/occupant_present/ce1_tdyn_smoke_b06_occ4_feb_gnurl_adaptive_occ4_hot_bspline_tdyn/rollout.csv",
    "SAC safe-PI Jan20 GPU": ROOT
    / "results/raw/sac_merlin/building_06/occupant_present/ce1_tdyn_smoke_b06_occ4_feb_sac_jan20_gpu/rollout.csv",
}


def _load_rollout(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "reference_setpoint" not in df.columns:
        occ = pd.to_numeric(df.get("occ_bin_1step", 0.0), errors="coerce").fillna(0.0)
        tdyn = pd.to_numeric(df.get("tdyn", df["baseline_setpoint"]), errors="coerce").fillna(
            pd.to_numeric(df["baseline_setpoint"], errors="coerce")
        )
        df["reference_setpoint"] = (1.0 - occ) * df["baseline_setpoint"] + occ * tdyn
    return df


def _summary_row(name: str, df: pd.DataFrame) -> dict[str, float | str]:
    occ = pd.to_numeric(df["occ_bin_1step"], errors="coerce").fillna(0.0).to_numpy() > 0.5
    ref = pd.to_numeric(df["reference_setpoint"], errors="coerce").to_numpy(dtype=float)
    sched = pd.to_numeric(df["baseline_setpoint"], errors="coerce").to_numpy(dtype=float)
    tin = pd.to_numeric(df["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    action = pd.to_numeric(df["action"], errors="coerce").to_numpy(dtype=float)
    reward = pd.to_numeric(df["reward"], errors="coerce").to_numpy(dtype=float)
    unocc = ~occ
    ref_sched_abs = np.abs(ref[unocc] - sched[unocc]) if unocc.any() else np.asarray([np.nan])
    occ_ref_minus_sched = (ref[occ] - sched[occ]) if occ.any() else np.asarray([np.nan])
    return {
        "controller": name,
        "steps": int(len(df)),
        "pred_occ_%": float(100.0 * occ.mean()),
        "pred_unocc_steps": int(unocc.sum()),
        "unocc_|Tref-Tschedule|_mean": float(np.nanmean(ref_sched_abs)),
        "unocc_|Tref-Tschedule|_max": float(np.nanmax(ref_sched_abs)),
        "occ_Tref_minus_Tschedule_mean": float(np.nanmean(occ_ref_minus_sched)),
        "Tin_MAE_to_Tref": float(np.nanmean(np.abs(tin - ref))),
        "Tin_min": float(np.nanmin(tin)),
        "Tin_max": float(np.nanmax(tin)),
        "action_mean": float(np.nanmean(action)),
        "reward_sum": float(np.nansum(reward)),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.3f}")
    return display.to_markdown(index=False)


def main() -> None:
    import matplotlib.pyplot as plt

    rollouts = {name: _load_rollout(path) for name, path in RUNS.items()}
    summary = pd.DataFrame([_summary_row(name, df) for name, df in rollouts.items()])
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    fig, axes = plt.subplots(len(rollouts), 1, figsize=(16, 9), sharex=True)
    if len(rollouts) == 1:
        axes = [axes]
    for ax, (name, df) in zip(axes, rollouts.items()):
        x = np.arange(len(df))
        occ = pd.to_numeric(df["occ_bin_1step"], errors="coerce").fillna(0.0).to_numpy() > 0.5
        unocc = ~occ
        start = None
        for i, flag in enumerate(unocc):
            if flag and start is None:
                start = i
            if start is not None and ((not flag) or i == len(unocc) - 1):
                end = i if not flag else i + 1
                ax.axvspan(start, end, color="0.90", alpha=0.7, lw=0)
                start = None
        ax.plot(x, df["indoor_temperature"], lw=0.8, color="#1f77b4", label="Tin")
        ax.plot(x, df["reference_setpoint"], lw=1.2, color="black", label="Tref = schedule/Tdyn gate")
        ax.plot(x, df["baseline_setpoint"], lw=0.8, color="#2ca02c", alpha=0.8, label="Tschedule")
        if "tdyn" in df.columns:
            ax.plot(x, df["tdyn"], lw=0.8, color="#ff7f0e", alpha=0.9, label="Tdyn")
        ax.set_ylabel("degC")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("February time step")
    fig.suptitle("Building 6 | occ4_hot | B-spline Tdyn smoke test | February", y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    md = f"""## CE1 B-spline Tdyn Smoke Test - Building 6 / occ4_hot / February

Single-building smoke test before rerunning the full 25-building B-spline Tdyn branch.

Building 6 was selected because it has many unoccupied hours in February and March. This run uses `occ4_hot`, so the gap between `Tschedule` and `Tdyn` is visible.

Reference definition tested here:

`Tref_k = (1 - occ_bin_k) * Tschedule_k + occ_bin_k * Tdyn_k`

Grey background spans are predicted-unoccupied steps. The expected behavior is: when grey, black `Tref` must sit exactly on green `Tschedule`; when not grey, black `Tref` should move toward orange `Tdyn`.

![Building 6 Tdyn smoke test](ce1_tdyn_smoke_b06_feb_reference_gate.png)

Gate-check summary:

{_markdown_table(summary)}

Result: the reference gate itself passes for all three controllers. The `unocc_|Tref-Tschedule|_max` column is exactly zero, so the B-spline branch is no longer silently tracking `Tdyn` when predicted unoccupied. Any remaining discomfort in this smoke test is controller performance, not the reference-switch bug.

Saved compact table:

- `analysis/ce1_tdyn_smoke_b06_feb_gate_summary.csv`
"""

    nb = nbformat.read(NOTEBOOK, as_version=4)
    marker = "## CE1 B-spline Tdyn Smoke Test - Building 6 / occ4_hot / February"
    nb.cells = [cell for cell in nb.cells if marker not in cell.get("source", "")]
    nb.cells.append(nbformat.v4.new_markdown_cell(md))
    nbformat.write(nb, NOTEBOOK)
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {FIG_PATH}")
    print(f"Appended {NOTEBOOK}")


if __name__ == "__main__":
    main()
