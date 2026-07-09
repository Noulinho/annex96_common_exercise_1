#!/usr/bin/env python
"""Append Tdyn occupancy-gate sanity plots to the SAC results notebook only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.rlmpc_pipeline import FITTED_OCCUPANT_PARAMS


PREFIX = "ce1_tdyn_gate_sanity"
MARKER = f"## Tdyn Gate Sanity Check - `{PREFIX}`"
NOTEBOOK = Path("analysis/08_sac_results.ipynb")
FIG_DIR = Path("figures") / PREFIX
ANALYSIS = Path("analysis")

BUILDINGS = dataset_building_entries(DATASET_NAME)
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]

MPC_GATE_CASE = "ce1_occ_gatecheck_mpc_mar"
SAC_FINAL_CASE = "ce1_occ_D010702_sac_occprob_feb10_mar_bspline_tdyn"


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def raw_path(controller: str, building_id: str, case: str, filename: str) -> Path:
    return Path("results") / "raw" / controller / building_id / "occupant_present" / case / filename


def shade_unoccupied(ax, time, occ_now) -> None:
    if occ_now is None:
        return
    time = np.asarray(time)
    occ = np.asarray(occ_now, dtype=float) > 0.0
    if len(time) == 0 or len(time) != len(occ):
        return
    start = None
    for idx, occupied in enumerate(occ):
        if not occupied and start is None:
            start = idx
        if (occupied or idx == len(occ) - 1) and start is not None:
            end = idx if occupied else idx + 1
            ax.axvspan(time[start], time[min(end - 1, len(time) - 1)], color="0.88", alpha=0.35, lw=0)
            start = None


def make_mpc_gate_plot() -> tuple[Path, pd.DataFrame]:
    building = next(b for b in BUILDINGS if int(b["index"]) == 8)
    path = raw_path("mpc", building["result_id"], MPC_GATE_CASE, "rollout.csv")
    frame = pd.read_csv(path)
    frame["controller_reference"] = (frame["tmin"] + frame["tmax"]) / 2.0

    unoccupied = frame["occ_now"].astype(float) <= 0.0
    occupied = ~unoccupied
    summary = pd.DataFrame(
        [
            {
                "building": building["display"],
                "occupant": "occ4_hot",
                "Tpref": FITTED_OCCUPANT_PARAMS["occ4_hot"]["T_pref"],
                "month": "March",
                "unoccupied_steps": int(unoccupied.sum()),
                "unocc_ref_vs_schedule_mae": float(
                    (frame.loc[unoccupied, "controller_reference"] - frame.loc[unoccupied, "baseline_setpoint"])
                    .abs()
                    .mean()
                ),
                "occupied_ref_vs_tdyn_mae": float(
                    (frame.loc[occupied, "controller_reference"] - frame.loc[occupied, "tdyn"]).abs().mean()
                ),
            }
        ]
    )

    fig, ax = plt.subplots(figsize=(13, 4.0))
    t = frame["time_step"].to_numpy()
    shade_unoccupied(ax, t, frame["occ_now"].to_numpy())
    ax.plot(t, frame["indoor_temperature"], lw=1.0, color="steelblue", label="Tin")
    ax.plot(t, frame["controller_reference"], lw=1.15, color="tab:orange", label="active setpoint")
    ax.plot(t, frame["tdyn"], lw=1.15, color="black", label="Tdyn")
    ax.set_title("Building 8 | occ4_hot | patched MPC bspline_tdyn active-setpoint gate | March")
    ax.set_xlabel("March time step")
    ax.set_ylabel("Temperature [C]")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "mpc_building8_occ4_march_gatecheck.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out, summary


def make_sac_reward_diagnostics(train_summary: pd.DataFrame) -> pd.DataFrame:
    """Compact reward decomposition for the existing final February SAC episode."""

    params = {
        "occ1_tolerant": {"T_pref": 21.861014, "b": 1.198849, "max_prob": 0.4},
        "occ2_sensitive": {"T_pref": 23.035896, "b": 3.451184, "max_prob": 0.4},
        "occ3_cold": {"T_pref": 19.957358, "b": 2.618482, "max_prob": 0.4},
        "occ4_hot": {"T_pref": 23.909844, "b": 2.570147, "max_prob": 0.4},
    }
    rows = []
    for building in BUILDINGS:
        occupant = occupant_for_building(building)
        path = raw_path("sac_merlin", building["result_id"], SAC_FINAL_CASE, "online_training_rollout.csv")
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        episode = int(frame["episode"].max())
        ep = frame[frame["episode"] == episode].copy().reset_index(drop=True)
        if ep.empty or "raw_reward" not in ep or "tin_before_action" not in ep or "tsp_before_action" not in ep:
            continue

        # CityLearn calculates the reward after env.step(). Approximate that post-step
        # temperature with the next row's pre-action temperature for a readable diagnostic.
        if len(ep) > 1:
            tin_reward = pd.to_numeric(ep["tin_before_action"].shift(-1).iloc[:-1], errors="coerce").to_numpy()
            tref = pd.to_numeric(ep["tsp_before_action"].iloc[:-1], errors="coerce").to_numpy()
            raw_reward = pd.to_numeric(ep["raw_reward"].iloc[:-1], errors="coerce").to_numpy()
        else:
            tin_reward = pd.to_numeric(ep["tin_before_action"], errors="coerce").to_numpy()
            tref = pd.to_numeric(ep["tsp_before_action"], errors="coerce").to_numpy()
            raw_reward = pd.to_numeric(ep["raw_reward"], errors="coerce").to_numpy()

        p = params[occupant]
        tpref = float(p["T_pref"])
        b = float(p["b"])
        pmax = float(p["max_prob"])
        p_cold = np.minimum(0.002 * np.exp(np.minimum(50.0, b * np.maximum(tpref - tin_reward, 0.0))), pmax)
        p_hot = np.minimum(0.002 * np.exp(np.minimum(50.0, b * np.maximum(tin_reward - tpref, 0.0))), pmax)
        track = ((tin_reward - tref) / 2.0) ** 2
        approx_cost = 0.1 * track + 0.7 * p_cold + 0.2 * p_hot

        rows.append(
            {
                "building": building["display"],
                "occupant": occupant,
                "raw_reward_mean": float(np.nanmean(raw_reward)),
                "approx_reward_mean": float(np.nanmean(-approx_cost)),
                "w_track": float(np.nanmean(0.1 * track)),
                "w_cold_prob": float(np.nanmean(0.7 * p_cold)),
                "w_hot_prob": float(np.nanmean(0.2 * p_hot)),
                "|Tin_next-ref|": float(np.nanmean(np.abs(tin_reward - tref))),
                "Tin_std": float(np.nanstd(tin_reward)),
                "Tin_max": float(np.nanmax(tin_reward)),
                "reset_%": float(100.0 * ep["temperature_reset_applied"].astype(bool).mean())
                if "temperature_reset_applied" in ep
                else np.nan,
            }
        )

    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        return diagnostics
    return diagnostics.sort_values("raw_reward_mean").head(10).reset_index(drop=True)


def make_sac_training_grid() -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    rows = []
    for ax, building in zip(axes, BUILDINGS):
        path = raw_path("sac_merlin", building["result_id"], SAC_FINAL_CASE, "online_training_rollout.csv")
        if not path.exists():
            ax.set_axis_off()
            continue
        frame = pd.read_csv(path)
        episode = int(frame["episode"].max())
        ep = frame[frame["episode"] == episode].copy()
        if ep.empty:
            ax.set_axis_off()
            continue
        t = ep["time_step"].to_numpy()
        ax.plot(t, ep["tin_before_action"], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t, ep["tsp_before_action"], lw=0.75, color="tab:orange", label="setpoint")
        if "tdyn" in ep.columns:
            ax.plot(t, ep["tdyn"], lw=0.75, color="black", label="Tdyn")
        ax.set_title(f"{building['display']} | {occupant_for_building(building)}", fontsize=7)
        ax.grid(alpha=0.2)
        rows.append(
            {
                "building": building["display"],
                "episode": episode,
                "reward_mean": float(ep["reward"].mean()),
                "raw_reward_mean": float(ep["raw_reward"].mean()),
                "tin_mean": float(ep["tin_before_action"].mean()),
                "tin_max": float(ep["tin_before_action"].max()),
                "action_mean": float(ep["action"].mean()),
            }
        )
    axes[0].legend(fontsize=7)
    fig.suptitle("SAC override-prob bspline_tdyn | final February training episode comfort", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "sac_occprob_bspline_tdyn_final_feb_training_ep29_all25.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, pd.DataFrame(rows)


def make_sac_deploy_grid() -> tuple[Path, pd.DataFrame]:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    rows = []
    for ax, building in zip(axes, BUILDINGS):
        path = raw_path("sac_merlin", building["result_id"], SAC_FINAL_CASE, "rollout.csv")
        if not path.exists():
            ax.set_axis_off()
            continue
        frame = pd.read_csv(path)
        t = frame["time_step"].to_numpy()
        ax.plot(t, frame["indoor_temperature"], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t, frame["effective_setpoint"], lw=0.75, color="tab:orange", label="setpoint")
        if "tdyn" in frame.columns:
            ax.plot(t, frame["tdyn"], lw=0.75, color="black", label="Tdyn")
        ax.set_title(f"{building['display']} | {occupant_for_building(building)}", fontsize=7)
        ax.grid(alpha=0.2)
        rows.append(
            {
                "building": building["display"],
                "reward_mean": float(frame["reward"].mean()),
                "tin_mean": float(frame["indoor_temperature"].mean()),
                "tin_max": float(frame["indoor_temperature"].max()),
                "hot_gt_26_frac": float((frame["indoor_temperature"] > 26.0).mean()),
                "action_mean": float(frame["action"].mean()),
            }
        )
    axes[0].legend(fontsize=7)
    fig.suptitle("SAC override-prob bspline_tdyn | March deployment comfort", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "sac_occprob_bspline_tdyn_march_deployment_all25.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, pd.DataFrame(rows)


def table(data: pd.DataFrame, digits: int = 4) -> str:
    if data.empty:
        return "_No data found._"
    return data.to_markdown(index=False, floatfmt=f".{digits}f")


def image_cell(path: Path, alt: str) -> nbf.NotebookNode:
    rel = Path("..") / path
    return nbf.v4.new_markdown_cell(f"<!-- {MARKER} -->\n![{alt}]({rel.as_posix()})")


def append_notebook(cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(NOTEBOOK, as_version=4)
    kept = [cell for cell in notebook.cells if MARKER not in str(cell.get("source", ""))]
    notebook.cells = kept + cells
    nbf.write(notebook, NOTEBOOK)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    mpc_plot, mpc_summary = make_mpc_gate_plot()
    train_plot, train_summary = make_sac_training_grid()
    deploy_plot, deploy_summary = make_sac_deploy_grid()
    reward_diag = make_sac_reward_diagnostics(train_summary)

    mpc_summary.to_csv(ANALYSIS / f"{PREFIX}_mpc_building8_gate_summary.csv", index=False)
    train_summary.to_csv(ANALYSIS / f"{PREFIX}_sac_final_feb_training_episode_summary.csv", index=False)
    deploy_summary.to_csv(ANALYSIS / f"{PREFIX}_sac_march_deployment_summary.csv", index=False)
    reward_diag.to_csv(ANALYSIS / f"{PREFIX}_sac_final_feb_reward_diagnostics.csv", index=False)

    worst_train = train_summary.sort_values("tin_max", ascending=False).head(8)
    worst_deploy = deploy_summary.sort_values("tin_max", ascending=False).head(8)

    intro = nbf.v4.new_markdown_cell(
        f"""{MARKER}

Small sanity section before rerunning the full occupant experiments.

MPC gate check uses Building 8 because it has `occ4_hot` and the most unoccupied hours among the occ4 buildings. The plot intentionally shows only the essential curves: `Tin`, the active/controller setpoint `(Tmin+Tmax)/2`, and the latent `Tdyn`. Grey background spans are actual unoccupied steps.

MPC gate summary:

{table(mpc_summary, digits=6)}

The SAC plots use the existing final override-probability SAC artifact `{SAC_FINAL_CASE}`. These artifacts were generated before the actual-occupancy gate fix, so they are diagnostic plots of the already-trained policy, not a fresh rerun.

Final February training episode, hottest rows:

{table(worst_train, digits=4)}

Why the final February SAC reward can look good while the comfort trace is noisy: this is the override-probability reward, not a hard `Tref +/- 2 C` guard. The probability terms are clipped at each occupant's fitted `p_max=0.4`, the hot-probability term has only weight `0.2`, and the reward is calculated on the post-action state. Compact final-episode decomposition for the worst reward rows:

{table(reward_diag, digits=4)}

March deployment, hottest rows:

{table(worst_deploy, digits=4)}

Saved compact CSVs:

- `analysis/{PREFIX}_mpc_building8_gate_summary.csv`
- `analysis/{PREFIX}_sac_final_feb_training_episode_summary.csv`
- `analysis/{PREFIX}_sac_march_deployment_summary.csv`
- `analysis/{PREFIX}_sac_final_feb_reward_diagnostics.csv`
"""
    )

    append_notebook(
        [
            intro,
            image_cell(mpc_plot, "MPC Building 8 occupancy gate check"),
            image_cell(train_plot, "SAC final February training episode comfort"),
            image_cell(deploy_plot, "SAC March deployment comfort"),
        ]
    )
    print(f"Updated {NOTEBOOK}")
    print(mpc_plot)
    print(train_plot)
    print(deploy_plot)


if __name__ == "__main__":
    main()
