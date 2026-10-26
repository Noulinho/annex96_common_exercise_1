#!/usr/bin/env python
"""Append final-January SAC training comfort plots for the no-reset run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries


PREFIX = "ce1_occ_D010702_sac_noreset_jan20_feb"
MARKER = f"## CE1 SAC No-Reset January Episode 19 Comfort - `{PREFIX}`"
RESULTS = Path("results")
ANALYSIS = Path("analysis")
FIG_DIR = Path("figures") / PREFIX
NOTEBOOK = Path('experiments/analysis/08_sac_results.ipynb')

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]

RUNS = [
    {
        "mode": "no_occupant",
        "scenario": "no_occupant",
        "display": "SAC no occupant, no reset",
        "label": f"{PREFIX}_no_occupant",
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "display": "SAC occupant B-spline Tdyn, no reset",
        "label": f"{PREFIX}_bspline_tdyn",
    },
]

TABLE_STYLE = """
<style>
.ce1-small-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 0.25rem 0 0.9rem 0;
}
</style>
""".strip()


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def raw_dir(building: dict, run: dict) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building["result_id"] / run["scenario"] / run["label"]


def load_training(building: dict, run: dict) -> pd.DataFrame:
    path = raw_dir(building, run) / "online_training_rollout.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def final_episode(frame: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    episode = int(pd.to_numeric(frame["episode"], errors="coerce").max())
    data = frame[pd.to_numeric(frame["episode"], errors="coerce") == episode].copy()
    return episode, data


def step_axis(frame: pd.DataFrame) -> np.ndarray:
    if "absolute_time_step" in frame.columns:
        return pd.to_numeric(frame["absolute_time_step"], errors="coerce").to_numpy(dtype=float)
    return np.arange(len(frame), dtype=float)


def table_display(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    html = out.to_html(index=False, escape=True, border=0)
    html = html.replace(
        "<table ",
        '<table style="border-collapse:collapse;font-size:11px;line-height:1.15;width:auto;" ',
        1,
    )
    html = html.replace(
        "<th>",
        '<th style="border:1px solid #666;padding:3px 6px;white-space:nowrap;font-weight:600;">',
    )
    html = html.replace(
        "<td>",
        '<td style="border:1px solid #666;padding:3px 6px;white-space:nowrap;">',
    )
    return (
        '<div class="ce1-small-table-wrap" style="max-width:100%;overflow-x:auto;">'
        f"{html}</div>"
    )


def make_comfort_grid(run: dict) -> tuple[int, Path]:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    episode_seen = None

    for ax, building in zip(axes, BUILDINGS):
        frame = load_training(building, run)
        episode, data = final_episode(frame)
        episode_seen = episode

        x = step_axis(data)
        tin = pd.to_numeric(data["tin_before_action"], errors="coerce").to_numpy(dtype=float)
        tsp = pd.to_numeric(data["tsp_before_action"], errors="coerce").to_numpy(dtype=float)

        n = min(len(x), len(tin), len(tsp))
        ax.plot(x[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(x[:n], tsp[:n], lw=0.75, color="tab:orange", label="setpoint")

        if run["mode"] == "bspline_tdyn" and "tdyn" in data.columns:
            tdyn = pd.to_numeric(data["tdyn"], errors="coerce").to_numpy(dtype=float)
            n_tdyn = min(n, len(tdyn))
            ax.plot(x[:n_tdyn], tdyn[:n_tdyn], lw=0.85, color="black", label="Tdyn")

        occ = "none" if run["mode"] == "no_occupant" else occupant_for_building(building)
        ax.set_title(f"{building['display']} | {occ}", fontsize=7)
        ax.grid(alpha=0.2)

    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | January training episode {episode_seen} comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['mode']}_january_episode{episode_seen}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return int(episode_seen or 0), path


def episode_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        for building in BUILDINGS:
            frame = load_training(building, run)
            episode, data = final_episode(frame)
            tin = pd.to_numeric(data["tin_before_action"], errors="coerce")
            tsp = pd.to_numeric(data["tsp_before_action"], errors="coerce")
            reward = pd.to_numeric(data["raw_reward"], errors="coerce")
            action = pd.to_numeric(data["action"], errors="coerce")
            valid = tin.notna() & tsp.notna()
            err = tin[valid] - tsp[valid]
            rows.append(
                {
                    "mode": run["mode"],
                    "episode": episode,
                    "building": building["display"],
                    "occupant": "none" if run["mode"] == "no_occupant" else occupant_for_building(building),
                    "raw_reward_mean": float(reward.mean()),
                    "action_mean": float(action.mean()),
                    "mae_to_setpoint": float(err.abs().mean()),
                    "cold_%": float((tin[valid] < tsp[valid] - 2.0).mean() * 100.0),
                    "hot_%": float((tin[valid] > tsp[valid] + 2.0).mean() * 100.0),
                    "Tin_min": float(tin.min()),
                    "Tin_max": float(tin.max()),
                    "setpoint_mean": float(tsp.mean()),
                }
            )
    return pd.DataFrame(rows)


def image_cell(path: Path, alt: str) -> nbf.NotebookNode:
    rel = Path("..") / path
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{alt}]({rel.as_posix()})")


def append_once(cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.read(NOTEBOOK, as_version=4)
    nb.cells = [cell for cell in nb.cells if MARKER not in cell.get("source", "")]
    nb.cells.extend(cells)
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plot_info = [make_comfort_grid(run) for run in RUNS]
    summary = episode_summary()
    summary_path = ANALYSIS / f"{PREFIX}_january_episode19_comfort_summary.csv"
    summary.to_csv(summary_path, index=False)

    mean_table = summary.groupby(["mode", "episode"], as_index=False).agg(
        raw_reward_mean=("raw_reward_mean", "mean"),
        action_mean=("action_mean", "mean"),
        mae_to_setpoint=("mae_to_setpoint", "mean"),
        cold_pct=("cold_%", "mean"),
        hot_pct=("hot_%", "mean"),
        tin_min=("Tin_min", "min"),
        tin_max=("Tin_max", "max"),
    )
    worst = summary.sort_values(["cold_%", "mae_to_setpoint"], ascending=False).head(10)

    cells: list[nbf.NotebookNode] = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

Final January SAC training episode comfort check for the no-reset experiment.

These plots read `online_training_rollout.csv` directly and use the state SAC saw at each step: `tin_before_action` and `tsp_before_action`. For the B-spline run, the black line is `Tdyn`.

{TABLE_STYLE}

Mean final-January episode summary:

{table_display(mean_table, digits=3)}

Worst final-January cold rows:

{table_display(worst[["mode", "building", "occupant", "raw_reward_mean", "action_mean", "mae_to_setpoint", "cold_%", "hot_%", "Tin_min", "Tin_max"]], digits=3)}

Saved full table:

- `{summary_path.as_posix()}`
"""
        )
    ]
    for (episode, path), run in zip(plot_info, RUNS):
        cells.append(image_cell(path, f"{run['display']} January episode {episode} comfort"))

    append_once(cells)
    print(f"Appended {NOTEBOOK}")
    print(f"Saved {summary_path}")
    for _, path in plot_info:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
