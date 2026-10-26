#!/usr/bin/env python
"""Append Building 17 SAC ablation results to the SAC notebook."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


NOTEBOOK = Path("experiments/analysis/08_sac_results.ipynb")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures/ce1_sac_b17_three_ablation")
MARKER = "## CE1 SAC Building 17 Three Separate Fix Tests"
BUILDING_ID = "building_17"
SCENARIO = "no_occupant"

RUNS = [
    (
        "Vanilla norm, auto alpha",
        "ce1_occ_D010702_sac_janfeb20_mar_no_occupant_b17_vanilla_norm",
    ),
    (
        "Checkpoint-selected actor",
        "ce1_occ_D010702_sac_janfeb20_mar_no_occupant_b17_checkpoint_eval",
    ),
    (
        "Fixed alpha",
        "ce1_occ_D010702_sac_janfeb20_mar_no_occupant_b17_fixed_alpha",
    ),
    (
        "Residual safe-PI",
        "ce1_occ_D010702_sac_janfeb20_mar_no_occupant_b17_residual_safepi",
    ),
]


def run_dir(label: str) -> Path:
    return Path("results/raw/sac_merlin") / BUILDING_ID / SCENARIO / label


def load_manifest(label: str) -> dict:
    return json.loads((run_dir(label) / "manifest.json").read_text())


def load_rollout(label: str) -> pd.DataFrame:
    return pd.read_csv(run_dir(label) / "rollout.csv")


def load_online(label: str) -> pd.DataFrame:
    path = run_dir(label) / "online_training_rollout.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_row(name: str, label: str) -> dict:
    manifest = load_manifest(label)
    metrics = manifest.get("metrics", {})
    config = manifest.get("config", {})
    rollout = load_rollout(label)
    tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
    action = pd.to_numeric(rollout["action"], errors="coerce")
    policy_action = pd.to_numeric(rollout.get("policy_action", action), errors="coerce")
    reward = pd.to_numeric(rollout["reward"], errors="coerce")
    return {
        "controller": name,
        "energy": metrics.get("electricity_consumption_total", np.nan),
        "disc_%": 100.0 * metrics.get("discomfort_proportion", np.nan),
        "cold_%": 100.0 * metrics.get("discomfort_cold_proportion", np.nan),
        "hot_%": 100.0 * metrics.get("discomfort_hot_proportion", np.nan),
        "KH": metrics.get("kelvin_hours", np.nan),
        "daily_peak": metrics.get("daily_peak_average", np.nan),
        "Tin_min": float(tin.min()),
        "Tin_max": float(tin.max()),
        "action_mean": float(action.mean()),
        "policy_action_mean": float(policy_action.mean()),
        "reward_mean": float(reward.mean()),
        "auto_alpha": bool(config.get("automatic_entropy_tuning", False)),
        "residual_safe_pi": bool(metrics.get("residual_safe_pi", False)),
        "selected_ep": metrics.get("checkpoint_selected_episode", np.nan),
    }


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_comfort_grid() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
    for ax, (name, label) in zip(axes.flat, RUNS):
        rollout = load_rollout(label)
        x = rollout["time_step"] if "time_step" in rollout else np.arange(len(rollout))
        tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
        tsp = pd.to_numeric(rollout["effective_setpoint"], errors="coerce")
        ax.plot(x, tin, lw=1.0, label="Tin")
        ax.plot(x, tsp, lw=1.2, label="setpoint")
        ax.fill_between(x, tsp - 2.0, tsp + 2.0, color="0.7", alpha=0.18, label="+/-2 C")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_ylabel("Temperature [C]")
    axes[-1, 0].set_xlabel("March time step")
    axes[-1, 1].set_xlabel("March time step")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Building 17 | March deterministic deployment comfort")
    fig.tight_layout()
    out = FIG_DIR / "building17_march_comfort_ablation_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def make_training_reward_plot() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, label in RUNS:
        online = load_online(label)
        if online.empty or "raw_reward" not in online:
            continue
        sac = online[online["phase"].eq("sac_training")].copy()
        if sac.empty:
            continue
        curve = sac.groupby("episode", as_index=False)["raw_reward"].mean()
        ax.plot(curve["episode"], curve["raw_reward"], marker="o", ms=3, lw=1.4, label=name)
    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.set_title("Building 17 SAC training reward - raw per-step mean")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean raw reward per step")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "building17_training_reward_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def make_checkpoint_plot() -> Path | None:
    path = run_dir("ce1_occ_D010702_sac_janfeb20_mar_no_occupant_b17_checkpoint_eval") / "checkpoint_eval_summary.csv"
    if not path.exists():
        return None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    ax1.plot(df["episode"], 100.0 * df["discomfort_proportion"], marker="o", lw=1.3, label="deployment discomfort")
    ax1.set_xlabel("Training episode checkpoint")
    ax1.set_ylabel("March discomfort [%]")
    ax1.grid(alpha=0.25)
    best = df.loc[df["discomfort_proportion"].idxmin()]
    ax1.axvline(float(best["episode"]), color="tab:red", ls="--", lw=1.0, label=f"selected ep {int(best['episode'])}")
    ax2 = ax1.twinx()
    ax2.plot(df["episode"], df["reward_total"], color="tab:green", marker="s", ms=3, lw=1.0, alpha=0.75, label="reward total")
    ax2.set_ylabel("March reward total")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    fig.suptitle("Building 17 checkpoint-selection validation trace")
    fig.tight_layout()
    out = FIG_DIR / "building17_checkpoint_eval_trace.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = pd.DataFrame([metric_row(name, label) for name, label in RUNS])
    out_csv = ANALYSIS / "ce1_sac_b17_three_ablation_kpis.csv"
    rows.to_csv(out_csv, index=False)
    comfort_fig = make_comfort_grid()
    reward_fig = make_training_reward_plot()
    checkpoint_fig = make_checkpoint_plot()

    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [
        cell
        for cell in nb.cells
        if MARKER not in "".join(cell.get("source", ""))
    ]

    rel_csv = Path(os.path.relpath(out_csv, NOTEBOOK.parent))
    rel_comfort = Path(os.path.relpath(comfort_fig, NOTEBOOK.parent))
    rel_reward = Path(os.path.relpath(reward_fig, NOTEBOOK.parent))
    checkpoint_text = ""
    if checkpoint_fig is not None:
        rel_checkpoint = Path(os.path.relpath(checkpoint_fig, NOTEBOOK.parent))
        checkpoint_text = f"\n![Building 17 checkpoint validation]({rel_checkpoint.as_posix()})\n"

    text = f"""<!-- {MARKER} -->
{MARKER}

Building 17 no-occupant ablation after the no-reset/vanilla-normalization run failed. All rows use the same normalized comfort reward:

`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`, `reward=-D_k`.

Training/evaluation protocol: safe-PI first episode, then SAC reward training on January+February for 20 total episodes, deterministic March deployment. No training-temperature reset, no robust normalization, no observation clipping, no reward clipping.

The three separate tests are:

- checkpoint-selected actor: evaluate each episode checkpoint deterministically on March and deploy the lowest-discomfort checkpoint;
- fixed alpha: disable automatic entropy tuning and keep fixed SAC alpha;
- residual safe-PI: use `action = clip(safe_pi_action + 0.5*(policy_action - 0.5), 0, 1)`.

March KPI comparison:

{markdown_table(rows)}

Saved table: `{rel_csv.as_posix()}`
"""
    nb.cells.extend(
        [
            nbf.v4.new_markdown_cell(text),
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n![Building 17 March comfort]({rel_comfort.as_posix()})"
            ),
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n![Building 17 training reward]({rel_reward.as_posix()})"
            ),
            nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->{checkpoint_text}") if checkpoint_text else None,
        ]
    )
    nb.cells = [cell for cell in nb.cells if cell is not None]
    nbf.write(nb, NOTEBOOK)
    print(f"Appended {MARKER} to {NOTEBOOK}")
    print(out_csv)
    print(comfort_fig)
    print(reward_fig)
    if checkpoint_fig:
        print(checkpoint_fig)


if __name__ == "__main__":
    main()
