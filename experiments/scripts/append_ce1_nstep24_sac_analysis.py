#!/usr/bin/env python
"""Append n-step-24 SAC results to 08_sac_results.ipynb.

Compares ce1_nstep24_2r2c_kalman_a1 against:
  - MPC reference: ce1_time_bias_guard_1c_2r2c_kalman
  - Best previous SAC: ce1_time_bias_guard_1c_2r2c_kalman_bc_mpc_sac10_es_a1
  - Comfort-only SAC: ce1_comfort_only_sac_a1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_analysis import aggregate_controller_rollouts
from src.rewards import load_simple_v4_price_profile, time_bias_setpoint_profile


MPC_LABEL      = "ce1_time_bias_guard_1c_2r2c_kalman"
PREV_SAC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_bc_mpc_sac10_es_a1"
CO_SAC_LABEL   = "ce1_comfort_only_sac_a1"
SAC_LABEL      = "ce1_nstep24_2r2c_kalman_a1"
SAC_A02_LABEL  = "ce1_nstep24_2r2c_kalman_a02"

FIG_DIR      = Path("figures") / "ce1_nstep24_sac_multi"
ANALYSIS_DIR = Path("analysis")
RESULTS      = Path("results")
BUILDINGS    = dataset_building_entries(DATASET_NAME)
FEB_START    = 744

RUNS = [
    {"key": "mpc",       "controller": "mpc",        "label": MPC_LABEL,      "display": "MPC 2R2C+Kalman",                        "comfort_slug": "mpc"},
    {"key": "sac_prev",  "controller": "sac_merlin",  "label": PREV_SAC_LABEL, "display": "SAC BC+reward (PI+BC+10, 1-step)",        "comfort_slug": "sac_prev"},
    {"key": "sac_co",    "controller": "sac_merlin",  "label": CO_SAC_LABEL,   "display": "SAC comfort-only (1-step)",               "comfort_slug": "sac_co"},
    {"key": "sac_n24",   "controller": "sac_merlin",  "label": SAC_LABEL,      "display": "SAC n-step-24 (PI+9 SAC, α=0.8)",        "comfort_slug": "sac_n24"},
    {"key": "sac_n24a02","controller": "sac_merlin",  "label": SAC_A02_LABEL,  "display": "SAC n-step-24 (PI+9 SAC, α=0.2)",        "comfort_slug": "sac_n24a02"},
]

KPI_CASES = [
    ("mpc", "baseline", "Baseline"),
    *[(r["controller"], r["label"], r["display"]) for r in RUNS],
]

WEIGHTS = {
    "comfort_weight": 0.5,
    "comfort_track_weight": 3.0,
    "comfort_track_scale": 2.0,
    "comfort_cold_weight": 100.0,
    "comfort_cold_scale": 1.5,
    "comfort_hot_weight": 20.0,
    "comfort_hot_scale": 2.0,
    "guard_band": 2.0,
    "bias_amplitude": 1.0,
    "shift_weight": 0.0,
    "power_scale": 5.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _rollout_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label / "rollout.csv"


def _summary_path(controller: str) -> Path:
    return RESULTS / "summaries" / controller / "no_occupant_summary_all_buildings.csv"


def _building_summary_path(controller: str, building_id: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / "summary.csv"


def _case_summary(controller: str, case: str) -> pd.DataFrame:
    per_building = []
    for b in BUILDINGS:
        p = _building_summary_path(controller, b["result_id"])
        if not p.exists():
            continue
        df = _read(p)
        sub = df[df["case"] == case]
        if not sub.empty:
            per_building.append(sub)
    if per_building:
        data = pd.concat(per_building, ignore_index=True)
        if "building_id" in data.columns:
            data = data.drop_duplicates(subset=["building_id", "case"], keep="last")
        return data
    agg_path = _summary_path(controller)
    if agg_path.exists():
        agg = _read(agg_path)
        return agg[agg["case"] == case].copy()
    return pd.DataFrame()


def _load_rollouts(controller: str, label: str) -> dict[str, pd.DataFrame]:
    return {b["result_id"]: _read(_rollout_path(controller, b["result_id"], label)) for b in BUILDINGS}


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for col in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if col in data.columns:
            return pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
    raise ValueError("No setpoint column found.")


def _time_step(data: pd.DataFrame) -> np.ndarray:
    if "time_step" in data.columns:
        return pd.to_numeric(data["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(data), dtype=int)


def _heating(data: pd.DataFrame) -> np.ndarray:
    col = "heating_electricity_consumption" if "heating_electricity_consumption" in data.columns else "action"
    return np.maximum(pd.to_numeric(data[col], errors="coerce").fillna(0.0).to_numpy(dtype=float), 0.0)


def _bias_and_stress(n: int) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(time_bias_setpoint_profile(simulation_start=FEB_START, simulation_end=FEB_START + n - 1,
                                                  amplitude=WEIGHTS["bias_amplitude"]), dtype=float)
    stress = np.asarray(load_simple_v4_price_profile(simulation_start=FEB_START, exp_beta=2.0), dtype=float)
    if stress.size < n:
        stress = np.resize(stress, n)
    return bias[:n], stress[:n]


def _controller_kpis() -> pd.DataFrame:
    frames = []
    for controller, case, display in KPI_CASES:
        sub = _case_summary(controller, case)
        if sub.empty:
            continue
        sub = sub.copy()
        sub["controller"] = display
        frames.append(sub)
    data = pd.concat(frames, ignore_index=True)
    data["discomfort_%"] = 100.0 * pd.to_numeric(data["discomfort_proportion"], errors="coerce")
    data["cold_%"]       = 100.0 * pd.to_numeric(data["discomfort_cold_proportion"], errors="coerce")
    data["hot_%"]        = 100.0 * pd.to_numeric(data["discomfort_hot_proportion"], errors="coerce")
    cols = ["electricity_consumption_total", "discomfort_%", "cold_%", "hot_%", "kelvin_hours", "daily_peak_average"]
    if "ramping_average" in data.columns:
        cols.append("ramping_average")
    out = data.groupby("controller", as_index=False)[cols].mean()
    order = {d: i for i, (_, _, d) in enumerate(KPI_CASES)}
    out["_order"] = out["controller"].map(order)
    return out.sort_values("_order").drop(columns="_order")


def _baseline_district() -> pd.DataFrame:
    data = _read(RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv")
    return pd.DataFrame({"time_step": np.arange(len(data)), "load": data["net_electricity_consumption"].to_numpy(float),
                         "target": data["district_target"].to_numpy(float), "controller": "CityLearn baseline"})


def _district_frame(run: dict) -> pd.DataFrame:
    frame, missing = aggregate_controller_rollouts(controller=run["controller"], case=run["label"], buildings=BUILDINGS)
    if missing:
        raise FileNotFoundError(f"Missing {run['display']} rollouts: {missing[:3]}")
    baseline = _baseline_district()
    n = min(len(frame), len(baseline))
    return pd.DataFrame({"time_step": frame["time_step"].to_numpy(int)[:n],
                         "load": frame["net_electricity_consumption"].to_numpy(float)[:n],
                         "target": baseline["target"].to_numpy(float)[:n],
                         "controller": run["display"]})


def _district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0]
    bl = baseline["load"].to_numpy(float)
    rows = []
    for frame in frames:
        load = frame["load"].to_numpy(float)
        tgt  = frame["target"].to_numpy(float)
        n = min(len(load), len(tgt))
        load, tgt = load[:n], tgt[:n]
        hrly = pd.DataFrame({"load": load, "hour": np.arange(n) % 24}).groupby("hour")["load"].mean()
        rows.append({"controller": frame["controller"].iloc[0],
                     "total_vs_citylearn": float(np.sum(load) / max(float(np.sum(bl)), 1e-9)),
                     "peak_vs_citylearn":  float(np.max(load)  / max(float(np.max(bl)),  1e-9)),
                     "daily_profile_peak_hour": int(hrly.idxmax()),
                     "ramping_vs_citylearn": float(np.sum(np.abs(np.diff(load))) / max(float(np.sum(np.abs(np.diff(bl)))), 1e-9)),
                     "above_target_kwh": float(np.sum(np.maximum(load - tgt, 0.0)))})
    return pd.DataFrame(rows)


def _make_comfort_grid(run: dict, rollouts: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, b in zip(axes, BUILDINGS):
        data = rollouts[b["result_id"]]
        ts = _time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(float)
        tsp = _setpoint(data)
        n = min(len(ts), len(tin), len(tsp))
        ts, tin, tsp = ts[:n], tin[:n], tsp[:n]
        bias, _ = _bias_and_stress(n)
        ax.plot(ts, tin, lw=0.75, color="steelblue", label="T_in")
        ax.plot(ts, tsp, lw=0.65, ls="--", color="black", label="Tsp")
        ax.plot(ts, tsp + bias, lw=0.65, ls=":", color="tab:orange", label="Tsp+bias")
        ax.fill_between(ts, tsp - WEIGHTS["guard_band"], tsp + WEIGHTS["guard_band"], alpha=0.10, color="green")
        ax.set_title(b["display"], fontsize=8)
        ax.set_ylim(14, 26)
        ax.grid(alpha=0.2)
    for ax in axes[len(BUILDINGS):]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} — time-biased target ±2 C guard", fontsize=12)
    fig.tight_layout()
    path = FIG_DIR / f"{run['comfort_slug']}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    target = frames[0][["time_step", "target"]]

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, grp in combined.groupby("controller", sort=False):
        ax.plot(grp["time_step"], grp["load"], lw=1.0, label=name)
    ax.plot(target["time_step"], target["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_xlabel("February time step"); ax.set_ylabel("District electricity [kWh]")
    ax.set_title(f"District load — {SAC_LABEL}"); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    ts_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(ts_path, dpi=140, bbox_inches="tight"); plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    tdail = target.assign(hour=target["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, grp in daily.groupby("controller", sort=False):
        ax.plot(grp["hour"], grp["load"], marker="o", lw=1.4, label=name)
    ax.plot(tdail["hour"], tdail["target"], color="black", marker="o", lw=1.1, ls="--", label="CE1 target")
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean district [kWh]")
    ax.set_title("Mean hourly district load — February"); ax.set_xticks(range(24)); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    dp_path = FIG_DIR / "district_mean_daily_profile.png"
    fig.savefig(dp_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return ts_path, dp_path


def _load_online_rewards(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for b in BUILDINGS:
            p = RESULTS / "raw" / run["controller"] / b["result_id"] / "no_occupant" / run["label"] / "online_training_rollout.csv"
            if not p.exists():
                continue
            data = _read(p)
            data["controller"] = run["display"]
            data["building"] = b["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _make_training_plot(online_rewards: pd.DataFrame) -> Path | None:
    if online_rewards.empty:
        return None
    episode_reward = (
        online_rewards.groupby(["controller", "building", "episode"], as_index=False)["reward"].sum()
        .groupby(["controller", "episode"], as_index=False)
        .agg(mean=("reward", "mean"), mx=("reward", "max"), mn=("reward", "min"))
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, grp in episode_reward.groupby("controller", sort=False):
        ax.plot(grp["episode"], grp["mean"], marker="o", lw=1.4, label=f"{name} mean")
        ax.fill_between(grp["episode"], grp["mn"], grp["mx"], alpha=0.12)
    ax.set_title("Online training reward by episode — n-step-24 vs baselines")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Episode reward sum")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "online_training_rewards.png"
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def _load_sac_losses(runs: list[dict]) -> pd.DataFrame:
    frames = []
    for run in runs:
        for b in BUILDINGS:
            p = RESULTS / "raw" / run["controller"] / b["result_id"] / "no_occupant" / run["label"] / "sac_training_losses.csv"
            if not p.exists():
                continue
            data = _read(p)
            data["controller"] = run["display"]
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _make_loss_plot(sac_losses: pd.DataFrame) -> Path | None:
    if sac_losses.empty or "alpha" not in sac_losses.columns:
        return None
    summary = sac_losses.groupby(["controller", "episode"], as_index=False).agg(
        policy_loss_mean=("policy_loss", "mean"),
        alpha_mean=("alpha", "mean"),
        reward_mean=("reward_batch_mean", "mean") if "reward_batch_mean" in sac_losses else ("policy_loss", "mean"),
    ).sort_values(["controller", "episode"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, grp in summary.groupby("controller", sort=False):
        axes[0].plot(grp["episode"], grp["policy_loss_mean"], marker="o", lw=1.2, label=name)
        axes[1].plot(grp["episode"], grp["alpha_mean"], marker="o", lw=1.2, label=name)
    axes[0].set_ylabel("Policy loss"); axes[1].set_ylabel("Alpha (entropy coeff)")
    axes[1].set_xlabel("Training episode")
    for ax in axes:
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle("SAC training diagnostics by episode")
    fig.tight_layout()
    path = FIG_DIR / "sac_training_losses.png"
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def _markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    data = df.copy()
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            data[col] = data[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(data.columns)
    rows = [headers] + data.astype(str).values.tolist()
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(headers))]
    def fmt(row):
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
    return "\n".join([fmt(headers), "| " + " | ".join("-" * w for w in widths) + " |",
                      *[fmt(r) for r in rows[1:]]])


def _append_once(path: Path, marker: str, cells: list) -> None:
    nb = nbf.read(path, as_version=4)
    for idx, cell in enumerate(nb.cells):
        if marker in str(cell.get("source", "")):
            nb.cells[idx:idx + len(cells)] = cells
            nbf.write(nb, path)
            print(f"Updated '{marker}' in {path}")
            return
    nb.cells.extend(cells)
    nbf.write(nb, path)
    print(f"Appended '{marker}' to {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rollouts = {run["key"]: _load_rollouts(run["controller"], run["label"]) for run in RUNS}
    kpis = _controller_kpis()
    kpis.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_controller_kpis.csv", index=False)

    district_frames = [_baseline_district(), *[_district_frame(run) for run in RUNS]]
    district_kpis = _district_kpis(district_frames)
    district_kpis.to_csv(ANALYSIS_DIR / f"{SAC_LABEL}_district_kpis.csv", index=False)
    ts_path, dp_path = _make_district_plots(district_frames)

    comfort_paths = {run["key"]: _make_comfort_grid(run, rollouts[run["key"]]) for run in RUNS}

    online_rewards = _load_online_rewards(RUNS)
    sac_losses     = _load_sac_losses(RUNS)
    reward_path    = _make_training_plot(online_rewards)
    loss_path      = _make_loss_plot(sac_losses)

    overview_md = f"""## CE1 N-step-24 SAC — `{SAC_LABEL}` / `{SAC_A02_LABEL}`

Architecture: **24-step return horizon** (γ^24 ≈ 0.786 bootstrap discount). No behaviour cloning. MERLIN-style: 1 PI exploration episode → 9 SAC reward episodes. January training → February deployment. Standard 2R2C+Kalman RC model. Two α variants: α=0.8 (CityLearn default, causes oscillation) and α=0.2 (lower entropy → stable heating).

The n-step return directly connects the action at time t to rewards 24 steps ahead, so the cold penalty from underheat (which lands ~1–2 h later) is included in the Q-target in **one** gradient step rather than requiring 24 levels of bootstrapping.

Q-target: `R_24 + γ^24 * V(s_{{t+24}})` where `R_24 = Σ_{{k=0}}^{{23}} γ^k * r_{{t+k}}`.

Reward (comfort only, no energy/shift):
`J_k = 0.5 * [3*((T_in-(Tsp+bias))/2)^2 + 100*(cold_guard/1.5)^2 + 20*(hot_guard/2)^2]`

Mean building KPIs:

{_markdown_table(kpis)}

District KPIs:

{_markdown_table(district_kpis)}
"""

    marker = f"## CE1 N-step-24 SAC — `{SAC_LABEL}` / `{SAC_A02_LABEL}`"
    cells = [nbf.v4.new_markdown_cell(overview_md)]

    for run in RUNS:
        cells.append(nbf.v4.new_markdown_cell(
            f"### {run['display']}\n"
            f"![Comfort grid](../{comfort_paths[run['key']].as_posix()})"
        ))

    cells.append(nbf.v4.new_markdown_cell(
        f"### District load\n"
        f"![Timeseries](../{ts_path.as_posix()})\n\n"
        f"![Daily profile](../{dp_path.as_posix()})"
    ))

    extra_paths = [p for p in [reward_path, loss_path] if p is not None]
    if extra_paths:
        cells.append(nbf.v4.new_markdown_cell(
            "### Training diagnostics\n" +
            "\n\n".join(f"![{p.stem}](../{p.as_posix()})" for p in extra_paths)
        ))

    nb_path = ANALYSIS_DIR / "08_sac_results.ipynb"
    if nb_path.exists():
        _append_once(nb_path, marker, cells)
    else:
        print(f"Notebook not found: {nb_path}")


if __name__ == "__main__":
    main()
