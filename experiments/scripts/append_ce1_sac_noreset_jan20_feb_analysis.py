#!/usr/bin/env python
"""Append CE1 no-reset SAC January-training / February-deployment analysis."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries


PREFIX = os.environ.get("CE1_SAC_NORESET_PREFIX", "ce1_occ_D010702_sac_noreset_jan20_feb")
MARKER_TITLE = os.environ.get("CE1_SAC_NORESET_MARKER_TITLE", "CE1 SAC No-Reset Jan20 Feb")
MARKER = f"## {MARKER_TITLE} - `{PREFIX}`"
REWARD_TEXT = os.environ.get(
    "CE1_SAC_NORESET_REWARD_TEXT",
    "`D_k = 0.1*((Tin-(Tref+bias))/2)^2 + 0.7*(max(Tref-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tref-2,0)/2)^2`, `reward = -D_k`.",
)
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
        "display": os.environ.get("CE1_SAC_NORESET_NO_OCC_DISPLAY", "SAC no occupant, no reset"),
        "label": f"{PREFIX}_no_occupant",
    },
    {
        "mode": "bspline_tdyn",
        "scenario": "occupant_present",
        "display": os.environ.get("CE1_SAC_NORESET_TDYN_DISPLAY", "SAC occupant B-spline Tdyn, no reset"),
        "label": f"{PREFIX}_bspline_tdyn",
    },
]

TABLE_STYLE = """
<style>
.ce1-table-wrap {
  max-width: 100%;
  overflow-x: auto;
  margin: 0.25rem 0 0.9rem 0;
}
.ce1-table {
  border-collapse: collapse;
  font-size: 11px;
  line-height: 1.15;
  width: auto;
}
.ce1-table th,
.ce1-table td {
  border: 1px solid #666;
  padding: 3px 6px;
  white-space: nowrap;
}
.ce1-table th {
  font-weight: 600;
}
</style>
""".strip()


def occupant_for_building(building: dict) -> str:
    return OCCUPANTS[(int(building["index"]) - 1) % len(OCCUPANTS)]


def raw_dir(building: dict, run: dict) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building["result_id"] / run["scenario"] / run["label"]


def summary_path(building: dict, run: dict) -> Path:
    return RESULTS / "summaries" / "sac_merlin" / building["result_id"] / run["scenario"] / "summary.csv"


def read_summary(building: dict, run: dict) -> pd.Series | None:
    path = summary_path(building, run)
    if not path.exists():
        return None
    data = pd.read_csv(path)
    if "case" not in data.columns:
        return None
    rows = data[data["case"].astype(str) == run["label"]]
    if rows.empty:
        return None
    return rows.iloc[-1]


def read_kpis(building: dict, run: dict) -> dict[str, float]:
    path = raw_dir(building, run) / "kpis.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    if "level" in data.columns:
        data = data[data["level"].astype(str) == "district"]
    out: dict[str, float] = {}
    for _, row in data.iterrows():
        value = pd.to_numeric(row.get("value"), errors="coerce")
        if pd.notna(value):
            out[str(row["cost_function"])] = float(value)
    return out


def num(row: pd.Series | None, key: str, default=np.nan) -> float:
    if row is None or key not in row or pd.isna(row[key]):
        return float(default)
    return float(row[key])


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_rollout(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "rollout.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_training(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "online_training_rollout.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_losses(building: dict, run: dict) -> pd.DataFrame | None:
    path = raw_dir(building, run) / "sac_training_losses.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def time_axis(frame: pd.DataFrame) -> np.ndarray:
    if "time_step" in frame.columns:
        return pd.to_numeric(frame["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(frame), dtype=int)


def setpoint(frame: pd.DataFrame) -> np.ndarray:
    for column in ("effective_setpoint", "baseline_setpoint", "scheduled_setpoint", "tmin"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("Rollout has no setpoint column.")


def setpoint_kelvin_hours_from_rollout(frame: pd.DataFrame) -> float:
    tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
    tsp = setpoint(frame)
    n = min(len(tin), len(tsp))
    if n == 0:
        return np.nan
    valid = np.isfinite(tin[:n]) & np.isfinite(tsp[:n])
    if not np.any(valid):
        return np.nan
    return float(np.sum(np.abs(tin[:n][valid] - tsp[:n][valid])))


def make_comfort_grid(run: dict) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        frame = load_rollout(building, run)
        if frame is None or frame.empty:
            ax.set_axis_off()
            continue
        t = time_axis(frame)
        tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = setpoint(frame)
        n = min(len(t), len(tin), len(tsp))
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.75, color="tab:orange", label="setpoint")
        if run["mode"] == "bspline_tdyn" and "tdyn" in frame.columns:
            tdyn = pd.to_numeric(frame["tdyn"], errors="coerce").to_numpy(dtype=float)
            n_tdyn = min(n, len(tdyn))
            ax.plot(t[:n_tdyn], tdyn[:n_tdyn], lw=0.85, color="black", label="Tdyn")
        occ = "none" if run["mode"] == "no_occupant" else occupant_for_building(building)
        ax.set_title(f"{building['display']} | {occ}", fontsize=7)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | February comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['mode']}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def building_kpis() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for building in BUILDINGS:
            frame = load_rollout(building, run)
            if frame is None:
                continue
            kpis = read_kpis(building, run)
            summary = read_summary(building, run)
            tin = pd.to_numeric(frame["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            reward = pd.to_numeric(frame.get("reward", pd.Series(dtype=float)), errors="coerce")
            rows.append(
                {
                    "mode": run["mode"],
                    "building": building["display"],
                    "occupant": "none" if run["mode"] == "no_occupant" else occupant_for_building(building),
                    "energy": kpis.get("electricity_consumption_total", num(summary, "electricity_consumption_total")),
                    "disc_%": 100.0 * kpis.get("discomfort_proportion", num(summary, "discomfort_proportion")),
                    "cold_%": 100.0 * kpis.get("discomfort_cold_proportion", num(summary, "discomfort_cold_proportion")),
                    "hot_%": 100.0 * kpis.get("discomfort_hot_proportion", num(summary, "discomfort_hot_proportion")),
                    "KH": num(summary, "kelvin_hours"),
                    "KH_setpoint": num(summary, "setpoint_kelvin_hours", setpoint_kelvin_hours_from_rollout(frame)),
                    "daily_peak": kpis.get("daily_peak_average", num(summary, "daily_peak_average")),
                    "ramping": kpis.get("ramping_average", np.nan),
                    "Tin_max": float(np.nanmax(tin)) if len(tin) else np.nan,
                    "reward_sum": float(reward.fillna(0.0).sum()) if not reward.empty else num(summary, "reward_total"),
                    "reward_mean": float(reward.mean()) if not reward.empty else np.nan,
                    "reset_steps": num(summary, "training_temperature_reset_steps", 0.0),
                    "dropped_steps": num(summary, "replay_filter_dropped_steps", 0.0),
                    "safe_pi_steps": num(summary, "online_safe_pi_steps", np.nan),
                    "sac_steps": num(summary, "online_sac_steps", np.nan),
                }
            )
    return pd.DataFrame(rows)


def reward_by_episode() -> pd.DataFrame:
    frames = []
    for run in RUNS:
        for building in BUILDINGS:
            data = load_training(building, run)
            if data is None or data.empty or "episode" not in data.columns:
                continue
            raw_col = "raw_reward" if "raw_reward" in data.columns else "reward"
            data = data.copy()
            data["raw_reward_plot"] = pd.to_numeric(data[raw_col], errors="coerce")
            data["reward_plot"] = pd.to_numeric(data.get("reward", data[raw_col]), errors="coerce")
            data["action_plot"] = pd.to_numeric(data.get("action", pd.Series(dtype=float)), errors="coerce")
            if "temperature_reset_applied" in data.columns:
                data["reset_plot"] = bool_series(data["temperature_reset_applied"]).astype(float)
            else:
                data["reset_plot"] = 0.0
            if "replay_transition_kept" in data.columns:
                data["dropped_plot"] = (~bool_series(data["replay_transition_kept"])).astype(float)
            else:
                data["dropped_plot"] = 0.0
            grouped = data.groupby("episode", as_index=False).agg(
                raw_reward_mean=("raw_reward_plot", "mean"),
                raw_reward_min=("raw_reward_plot", "min"),
                raw_reward_max=("raw_reward_plot", "max"),
                reward_mean=("reward_plot", "mean"),
                action_mean=("action_plot", "mean"),
                reset_fraction=("reset_plot", "mean"),
                dropped_fraction=("dropped_plot", "mean"),
            )
            grouped["mode"] = run["mode"]
            grouped["building"] = building["display"]
            frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def losses_by_episode() -> pd.DataFrame:
    frames = []
    for run in RUNS:
        for building in BUILDINGS:
            data = load_losses(building, run)
            if data is None or data.empty or "episode" not in data.columns:
                continue
            keep = [c for c in ("soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha", "alpha_loss") if c in data.columns]
            if not keep:
                continue
            grouped = data.groupby("episode", as_index=False)[keep].mean(numeric_only=True)
            grouped["mode"] = run["mode"]
            grouped["building"] = building["display"]
            frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_reward_plot(rewards: pd.DataFrame) -> Path | None:
    if rewards.empty:
        return None
    grouped = rewards.groupby(["mode", "episode"], as_index=False).agg(
        raw_reward_mean=("raw_reward_mean", "mean"),
        raw_reward_p10=("raw_reward_mean", lambda x: float(np.nanpercentile(x, 10))),
        raw_reward_p90=("raw_reward_mean", lambda x: float(np.nanpercentile(x, 90))),
    )
    fig, ax = plt.subplots(figsize=(10.5, 5))
    for mode, group in grouped.groupby("mode", sort=False):
        label = "no occupant" if mode == "no_occupant" else "occupant B-spline Tdyn"
        x = group["episode"].to_numpy(dtype=float)
        mean = group["raw_reward_mean"].to_numpy(dtype=float)
        ax.plot(x, mean, marker="o", lw=1.4, label=label)
        ax.fill_between(
            x,
            group["raw_reward_p10"].to_numpy(dtype=float),
            group["raw_reward_p90"].to_numpy(dtype=float),
            alpha=0.12,
        )
    ax.axhline(0.0, color="0.35", lw=0.8)
    ax.set_title("SAC online reward by episode - no temperature reset")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean raw reward per step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "sac_reward_by_episode_noreset.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_loss_plot(losses: pd.DataFrame) -> Path | None:
    if losses.empty or "policy_loss" not in losses.columns:
        return None
    grouped = losses.groupby(["mode", "episode"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for mode, group in grouped.groupby("mode", sort=False):
        label = "no occupant" if mode == "no_occupant" else "occupant B-spline Tdyn"
        if "soft_q1_loss" in group:
            axes[0].plot(group["episode"], group["soft_q1_loss"], marker="o", lw=1.0, label=label)
        axes[1].plot(group["episode"], group["policy_loss"], marker="o", lw=1.0, label=label)
    axes[0].set_title("Soft Q1 loss")
    axes[1].set_title("Policy loss")
    for ax in axes:
        ax.set_xlabel("Training episode")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("SAC loss means by episode - no temperature reset")
    fig.tight_layout()
    path = FIG_DIR / "sac_losses_by_episode_noreset.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def table_display(df: pd.DataFrame, digits: int = 3) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    html = out.to_html(index=False, escape=True, border=0, classes="ce1-table")
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
        '<div class="ce1-table-wrap" '
        'style="max-width:100%;overflow-x:auto;margin:0.25rem 0 0.9rem 0;">'
        f"{html}</div>"
    )


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

    btab = building_kpis()
    rewards = reward_by_episode()
    losses = losses_by_episode()

    btab.to_csv(ANALYSIS / f"{PREFIX}_building_kpis_all.csv", index=False)
    rewards.to_csv(ANALYSIS / f"{PREFIX}_sac_reward_by_episode.csv", index=False)
    losses.to_csv(ANALYSIS / f"{PREFIX}_sac_losses_by_episode.csv", index=False)

    mean_cols = ["energy", "disc_%", "cold_%", "hot_%", "KH", "KH_setpoint", "daily_peak", "ramping", "Tin_max", "reward_mean"]
    mean_table = btab.groupby("mode", as_index=False)[mean_cols].mean(numeric_only=True)
    reset_table = btab.groupby("mode", as_index=False).agg(
        reset_steps=("reset_steps", "sum"),
        dropped_steps=("dropped_steps", "sum"),
        safe_pi_steps=("safe_pi_steps", "sum"),
        sac_steps=("sac_steps", "sum"),
    )
    worst = btab.sort_values(["disc_%", "Tin_max"], ascending=False).head(15)
    final_reward = (
        rewards.sort_values("episode")
        .groupby("mode", as_index=False)
        .tail(1)
        [["mode", "episode", "raw_reward_mean", "reward_mean", "action_mean", "reset_fraction", "dropped_fraction"]]
        if not rewards.empty
        else pd.DataFrame()
    )

    comfort_paths = [make_comfort_grid(run) for run in RUNS]
    reward_path = make_reward_plot(rewards)
    loss_path = make_loss_plot(losses)

    cells: list[nbf.NotebookNode] = [
        nbf.v4.new_markdown_cell(
            f"""{MARKER}

Rerun of the SAC January-training / February-deployment experiment with the temperature reset intervention disabled.

Pipeline:

`safe-PI` bootstrap for episode 0, then 19 SAC reward-training episodes in January, then February deployment.

The reward is exactly the normalized-D comfort reward:

{REWARD_TEXT}

Run labels:

`{PREFIX}_no_occupant`

`{PREFIX}_bspline_tdyn`

Temperature-reset flags are not used in these runs; the reset check below should stay at zero.

Mean building KPIs:

{TABLE_STYLE}

{table_display(mean_table, digits=3)}

Temperature reset / replay-drop check:

{table_display(reset_table, digits=0)}

Final online reward summary:

{table_display(final_reward, digits=4)}

Worst February comfort rows:

{table_display(worst[["mode", "building", "occupant", "energy", "disc_%", "cold_%", "hot_%", "Tin_max", "reward_mean"]], digits=3)}

Saved full tables:

- `analysis/{PREFIX}_building_kpis_all.csv`
- `analysis/{PREFIX}_sac_reward_by_episode.csv`
- `analysis/{PREFIX}_sac_losses_by_episode.csv`
"""
        )
    ]
    for path, run in zip(comfort_paths, RUNS):
        cells.append(image_cell(path, f"{run['display']} February comfort"))
    if reward_path is not None:
        cells.append(image_cell(reward_path, "SAC reward by episode no reset"))
    if loss_path is not None:
        cells.append(image_cell(loss_path, "SAC losses by episode no reset"))

    append_once(cells)
    print(f"Appended {NOTEBOOK}")
    print(f"Saved figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
