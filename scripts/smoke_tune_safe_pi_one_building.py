#!/usr/bin/env python
"""Small Safe-PI Kp/Ki smoke test for one building.

Runs no-occupant February rollouts for a grid of PI gains, compares against
the current default, and saves a comfort plot. This is intentionally not a
notebook appender.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from scripts.run_ce1_thesis_rbc_safepi import run_one


def _kpi(path: Path, name: str) -> float:
    frame = pd.read_csv(path)
    rows = frame[(frame["level"].eq("district")) & (frame["cost_function"].eq(name))]
    return float(rows["value"].iloc[0]) if not rows.empty else float("nan")


def _label(kp: float, ki: float) -> str:
    return f"kp{kp:g}_ki{ki:g}".replace(".", "p")


def run_grid(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    building = next(
        b for b in dataset_building_entries(DATASET_NAME) if int(b["index"]) == int(args.building)
    )
    output_root_base = Path(args.output_root)
    summary_root_base = Path(args.summary_root)
    rows = []

    for kp in args.kp:
        for ki in args.ki:
            tag = _label(kp, ki)
            rollout_path = run_one(
                building=building,
                month=args.month,
                mode="no_occupant",
                occupant_name=None,
                seed=args.seed,
                output_root=output_root_base / tag,
                summary_root=summary_root_base / tag,
                alpha_occ=0.5,
                safe_pi_cold_margin=args.safe_pi_cold_margin,
                safe_pi_hot_margin=args.safe_pi_hot_margin,
                safe_pi_cold_min_action=args.safe_pi_cold_min_action,
                safe_pi_cold_kp=args.safe_pi_cold_kp,
                safe_pi_hot_max_action=args.safe_pi_hot_max_action,
                safe_pi_ramp_limit=args.safe_pi_ramp_limit,
                pi_kp=kp,
                pi_ki=ki,
            )
            rollout = pd.read_csv(rollout_path)
            kpis_path = rollout_path.parent / "kpis.csv"
            tracking = rollout["tracking_setpoint"].to_numpy(float)
            tin = rollout["indoor_temperature"].to_numpy(float)
            rows.append(
                {
                    "tag": tag,
                    "kp": kp,
                    "ki": ki,
                    "rollout_path": str(rollout_path),
                    "reward_sum": float(rollout["reward"].sum()),
                    "reward_mean": float(rollout["reward"].mean()),
                    "action_mean": float(rollout["action"].mean()),
                    "mae_to_tracking": float(np.mean(np.abs(tin - tracking))),
                    "tin_min": float(np.min(tin)),
                    "tin_max": float(np.max(tin)),
                    "energy": _kpi(kpis_path, "electricity_consumption_total"),
                    "discomfort": _kpi(kpis_path, "discomfort_proportion"),
                    "cold": _kpi(kpis_path, "discomfort_cold_proportion"),
                    "hot": _kpi(kpis_path, "discomfort_hot_proportion"),
                    "daily_peak": _kpi(kpis_path, "daily_peak_average"),
                    "ramping": _kpi(kpis_path, "ramping_average"),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["reward_mean", "mae_to_tracking", "discomfort"], ascending=[False, True, True]
    )
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_path, index=False)
    return results, summary_path


def make_plot(results: pd.DataFrame, args: argparse.Namespace) -> Path:
    default = results[(np.isclose(results["kp"], 0.2)) & (np.isclose(results["ki"], 0.005))]
    if default.empty:
        default = results.iloc[[0]]
    best = results.iloc[[0]]

    cases = [("Default Safe-PI", default.iloc[0]), ("Tuned Safe-PI", best.iloc[0])]
    fig, axes = plt.subplots(len(cases), 1, figsize=(14, 6), sharex=True, constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]

    for ax, (title, row) in zip(axes, cases):
        rollout = pd.read_csv(row["rollout_path"])
        x = rollout["time_step"].to_numpy()
        tin = rollout["indoor_temperature"].to_numpy(float)
        sp = rollout["tracking_setpoint"].to_numpy(float)
        ax.plot(x, tin, lw=1.0, label="Tin")
        ax.plot(x, sp, lw=1.2, label="Tref + bias")
        ax.fill_between(x, sp - 2.0, sp + 2.0, color="0.85", alpha=0.55, label="+/-2 C")
        ax.set_title(
            f"{title}: Kp={row['kp']:.3g}, Ki={row['ki']:.3g}, "
            f"reward={row['reward_mean']:.3f}, disc={100*row['discomfort']:.1f}%, "
            f"MAE={row['mae_to_tracking']:.2f} C"
        )
        ax.set_ylabel("Temperature [C]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=9)

    axes[-1].set_xlabel(f"{args.month.title()} time step")
    path = Path(args.figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--building", type=int, default=17)
    parser.add_argument("--month", choices=["jan", "feb", "mar"], default="feb")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--kp", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--ki", nargs="+", type=float, default=[0.0, 0.001, 0.003, 0.005, 0.01])
    parser.add_argument("--safe-pi-cold-margin", type=float, default=1.0)
    parser.add_argument("--safe-pi-hot-margin", type=float, default=1.0)
    parser.add_argument("--safe-pi-cold-min-action", type=float, default=0.25)
    parser.add_argument("--safe-pi-cold-kp", type=float, default=0.25)
    parser.add_argument("--safe-pi-hot-max-action", type=float, default=0.0)
    parser.add_argument("--safe-pi-ramp-limit", type=float, default=None)
    parser.add_argument("--output-root", default="results/raw/rbc_safe_pi_kpki_smoke")
    parser.add_argument("--summary-root", default="results/summaries/rbc_safe_pi_kpki_smoke")
    parser.add_argument(
        "--summary-csv",
        default="analysis/safe_pi_kpki_smoke_building17_feb_results.csv",
    )
    parser.add_argument(
        "--figure",
        default="analysis/figures/safe_pi_kpki_smoke_building17_feb_comfort.png",
    )
    args = parser.parse_args()

    results, summary_path = run_grid(args)
    figure_path = make_plot(results, args)
    cols = [
        "tag",
        "kp",
        "ki",
        "reward_mean",
        "mae_to_tracking",
        "discomfort",
        "cold",
        "hot",
        "energy",
        "daily_peak",
        "ramping",
        "action_mean",
        "tin_min",
        "tin_max",
    ]
    print(results[cols].head(10).to_string(index=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
