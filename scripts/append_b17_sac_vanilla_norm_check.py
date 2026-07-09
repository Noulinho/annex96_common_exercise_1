#!/usr/bin/env python
"""Append Building 17 SAC no-normalization sanity check to SAC notebook."""

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


NOTEBOOK = Path("analysis/08_sac_results.ipynb")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures/ce1_sac_b17_vanilla_norm")
MARKER = "## CE1 SAC B17 No Robust/Clip Normalization Check"

BUILDING_ID = "building_17"
OLD_LABEL = "ce1_occ_D010702_sac_janfeb20_mar_no_occupant"
NEW_LABEL = "ce1_occ_D010702_sac_janfeb20_mar_no_occupant_b17_vanilla_norm"


def run_dir(label: str) -> Path:
    return Path("results/raw/sac_merlin") / BUILDING_ID / "no_occupant" / label


def load_manifest(label: str) -> dict:
    path = run_dir(label) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_rollout(label: str) -> pd.DataFrame:
    path = run_dir(label) / "rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_row(label: str, controller: str) -> dict:
    manifest = load_manifest(label)
    metrics = manifest.get("metrics", {})
    config = manifest.get("config", {})
    rollout = load_rollout(label)
    tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
    action = pd.to_numeric(rollout["action"], errors="coerce")
    reward = pd.to_numeric(rollout["reward"], errors="coerce")
    return {
        "controller": controller,
        "energy": metrics.get("electricity_consumption_total", np.nan),
        "disc_%": 100.0 * metrics.get("discomfort_proportion", np.nan),
        "cold_%": 100.0 * metrics.get("discomfort_cold_proportion", np.nan),
        "hot_%": 100.0 * metrics.get("discomfort_hot_proportion", np.nan),
        "KH": metrics.get("kelvin_hours", np.nan),
        "daily_peak": metrics.get("daily_peak_average", np.nan),
        "ramping": metrics.get("ramping_average", np.nan),
        "Tin_min": float(tin.min()),
        "Tin_max": float(tin.max()),
        "action_mean": float(action.mean()),
        "reward_mean": float(reward.mean()),
        "robust_norm": bool(config.get("robust_normalization", False)),
        "obs_clip": config.get("normalized_observation_clip", None),
        "reward_clip": config.get("normalized_reward_clip", None),
        "auto_entropy": bool(config.get("automatic_entropy_tuning", False)),
    }


def make_comfort_plot() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rollout = load_rollout(NEW_LABEL)
    x = rollout["time_step"] if "time_step" in rollout else np.arange(len(rollout))
    tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce")
    if "effective_setpoint" in rollout:
        tsp = pd.to_numeric(rollout["effective_setpoint"], errors="coerce")
    else:
        tsp = pd.to_numeric(rollout["baseline_setpoint"], errors="coerce")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(x, tin, lw=1.25, label="Tin")
    ax.plot(x, tsp, lw=1.5, label="setpoint")
    ax.fill_between(x, tsp - 2.0, tsp + 2.0, color="0.7", alpha=0.2, label="+/-2 C")
    ax.set_title("Building 17 | SAC no robust/clip normalization | March comfort")
    ax.set_xlabel("March time step")
    ax.set_ylabel("Temperature [C]")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "building17_sac_vanilla_norm_march_comfort.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = pd.DataFrame(
        [
            metric_row(OLD_LABEL, "Jan+Feb20 SAC with robust/clipped norm"),
            metric_row(NEW_LABEL, "Jan+Feb20 SAC vanilla norm"),
        ]
    )
    out_csv = ANALYSIS / "ce1_sac_b17_vanilla_norm_check.csv"
    rows.to_csv(out_csv, index=False)
    fig_path = make_comfort_plot()

    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [
        cell
        for cell in nb.cells
        if MARKER not in "".join(cell.get("source", ""))
    ]

    rel_fig = Path(os.path.relpath(fig_path, NOTEBOOK.parent))
    rel_csv = Path(os.path.relpath(out_csv, NOTEBOOK.parent))
    text = f"""<!-- {MARKER} -->
{MARKER}

Building 17 sanity check for the Jan+Feb SAC training run. This rerun keeps the same reward and safe-PI first episode, keeps automatic entropy tuning, but removes the active normalization/clip additions:

`robust_normalization=False`, `normalized_observation_clip=None`, `normalized_reward_clip=None`.

Reward/cost is still:

`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`, `reward=-D_k`.

March KPI comparison:

{markdown_table(rows)}

Saved table: `{rel_csv.as_posix()}`
"""
    nb.cells.extend(
        [
            nbf.v4.new_markdown_cell(text),
            nbf.v4.new_markdown_cell(
                f"<!-- {MARKER} -->\n![Building 17 March comfort]({rel_fig.as_posix()})"
            ),
        ]
    )
    nbf.write(nb, NOTEBOOK)
    print(f"Appended {MARKER} to {NOTEBOOK}")
    print(out_csv)
    print(fig_path)


if __name__ == "__main__":
    main()
