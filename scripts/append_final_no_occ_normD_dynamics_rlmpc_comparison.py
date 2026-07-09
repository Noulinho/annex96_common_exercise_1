#!/usr/bin/env python3
"""Append final no-occupant normalized-D GNU-RL dynamics-adaptation comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

import scripts.append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis as base
import scripts.append_final_no_occ_normD_controller_comparison as final


PREFIX = "ce1_final_no_occ_normD010702_dynamics_rlmpc_comparison"
MARKER = f"## Final No-Occupant Normalized-D MPC/SAC/GNU-RL Dynamics Adaptation - `{PREFIX}`"

MPC_LABEL = "ce1_final_no_occ_D010702_mpc_2r2c_kalman"
RLMPC_FIXED_LABEL = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_fixed"
RLMPC_ADAPTIVE_LABEL = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6_adaptive_dyn"
SAC_BASE_LABEL = "ce1_final_no_occ_D010702_sac_safepi9_a1"
SAC_BAD_LABEL = "ce1_final_no_occ_D010702_sac_safepi9_bad10_a1"

NOTEBOOKS = [
    base.ANALYSIS_DIR / "02_mpc_results.ipynb",
    base.ANALYSIS_DIR / "04_rlmpc_results.ipynb",
    base.ANALYSIS_DIR / "08_sac_results.ipynb",
    base.ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
    base.ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
    base.ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
]


def configure_base() -> None:
    base.FIG_DIR = Path("figures") / PREFIX
    base.MARKER = MARKER
    base.MPC_LABEL = MPC_LABEL
    base.RLMPC_LABEL = RLMPC_FIXED_LABEL
    base.SAC_BASE_LABEL = SAC_BASE_LABEL
    base.SAC_BAD_LABEL = SAC_BAD_LABEL
    base.RUNS = [
        {
            "key": "mpc",
            "display": "MPC 2R2C+Kalman normalized D",
            "controller": "mpc",
            "label": MPC_LABEL,
            "hybrid": False,
        },
        {
            "key": "rlmpc_fixed",
            "display": "GNU-RL fixed 1R1C H6 normalized D",
            "controller": "rlmpc",
            "label": RLMPC_FIXED_LABEL,
            "hybrid": False,
        },
        {
            "key": "rlmpc_adaptive",
            "display": "GNU-RL adaptive-dynamics 1R1C H6 normalized D",
            "controller": "rlmpc",
            "label": RLMPC_ADAPTIVE_LABEL,
            "hybrid": False,
        },
        {
            "key": "sac",
            "display": "SAC safe-PI normalized reward (9 ep + bad +10)",
            "controller": "sac_merlin",
            "label": SAC_BASE_LABEL,
            "hybrid": True,
        },
    ]
    base.SHORT_CONTROLLER_NAMES = {
        "CityLearn baseline": "CityLearn",
        base.RUNS[0]["display"]: "MPC",
        base.RUNS[1]["display"]: "GNU-RL fixed",
        base.RUNS[2]["display"]: "GNU-RL dyn-adapt",
        base.RUNS[3]["display"]: "SAC hybrid",
    }
    base.WEIGHTS = {
        "comfort_weight": 1.0,
        "shift_weight": 0.0,
        "track_weight": 0.1,
        "cold_weight": 0.7,
        "hot_weight": 0.2,
        "track_scale": 2.0,
        "cold_scale": 1.5,
        "hot_scale": 2.0,
        "guard_band": 2.0,
        "power_scale": 5.0,
        "bias_amplitude": 1.0,
    }


def markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    return base._markdown_table(df, digits=digits, max_rows=max_rows)


def short(df: pd.DataFrame) -> pd.DataFrame:
    return base._short_controller_names(df)


def image(path: Path, title: str) -> nbf.NotebookNode:
    return base._image(path, title)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_metrics(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _rlmpc_dynamics_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in [base.RUNS[1], base.RUNS[2]]:
        label = str(run["label"])
        for building in base.BUILDINGS:
            building_id = str(building["result_id"])
            metrics = _read_metrics(
                base.RESULTS
                / "summaries"
                / "rlmpc"
                / building_id
                / "no_occupant"
                / f"{label}_metrics.json"
            )
            hist = _safe_read_csv(base._raw_dir("rlmpc", building_id, label) / "dynamics_history.csv")
            first = hist.iloc[0].to_dict() if len(hist) else {}
            final_row = hist.iloc[-1].to_dict() if len(hist) else {}
            rows.append(
                {
                    "controller": run["display"],
                    "building": building["display"],
                    "building_id": building_id,
                    "adaptive_dynamics": bool(metrics.get("adaptive_dynamics", False)),
                    "dynamics_rows": int(len(hist)),
                    "dynamics_epochs_run": metrics.get("dynamics_epochs_run", max(0, len(hist) - 2)),
                    "train_rmse_before_c": metrics.get(
                        "dynamics_train_rmse_c_before",
                        first.get("train_rmse_c", np.nan),
                    ),
                    "train_rmse_after_c": metrics.get(
                        "dynamics_train_rmse_c_after",
                        final_row.get("train_rmse_c", np.nan),
                    ),
                    "val_rmse_before_c": metrics.get(
                        "dynamics_val_rmse_c_before",
                        first.get("val_rmse_c", np.nan),
                    ),
                    "val_rmse_after_c": metrics.get(
                        "dynamics_val_rmse_c_after",
                        final_row.get("val_rmse_c", np.nan),
                    ),
                    "C_before": metrics.get("C_before", np.nan),
                    "C_after": metrics.get("C_after_dynamics", np.nan),
                    "Rm_before": metrics.get("Rm_before", np.nan),
                    "Rm_after": metrics.get("Rm_after_dynamics", np.nan),
                    "Rout_before": metrics.get("Rout_before", np.nan),
                    "Rout_after": metrics.get("Rout_after_dynamics", np.nan),
                    "Pnom_before": metrics.get("Pnom_before", np.nan),
                    "Pnom_after": metrics.get("Pnom_after_dynamics", np.nan),
                    "Tm_before": metrics.get("Tm_before", np.nan),
                    "Tm_after": metrics.get("Tm_after_dynamics", np.nan),
                    "cop_a_before": metrics.get("cop_a_before", np.nan),
                    "cop_a_after": metrics.get("cop_a_after_dynamics", np.nan),
                    "cop_b_before": metrics.get("cop_b_before", np.nan),
                    "cop_b_after": metrics.get("cop_b_after_dynamics", np.nan),
                }
            )
    return pd.DataFrame(rows)


def _rlmpc_dynamics_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    rows["val_rmse_delta_c"] = rows["val_rmse_after_c"] - rows["val_rmse_before_c"]
    rows["train_rmse_delta_c"] = rows["train_rmse_after_c"] - rows["train_rmse_before_c"]
    out = (
        rows.groupby("controller", as_index=False)
        .agg(
            adaptive_dynamics=("adaptive_dynamics", "max"),
            dynamics_epochs_run=("dynamics_epochs_run", "mean"),
            train_rmse_before_c=("train_rmse_before_c", "mean"),
            train_rmse_after_c=("train_rmse_after_c", "mean"),
            train_rmse_delta_c=("train_rmse_delta_c", "mean"),
            val_rmse_before_c=("val_rmse_before_c", "mean"),
            val_rmse_after_c=("val_rmse_after_c", "mean"),
            val_rmse_delta_c=("val_rmse_delta_c", "mean"),
        )
    )
    return short(out)


def _make_rlmpc_dynamics_plot(rows: pd.DataFrame) -> Path:
    path = base.FIG_DIR / "rlmpc_adaptive_dynamics_rmse.png"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if rows.empty:
        ax.text(0.5, 0.5, "No GNU-RL dynamics rows", ha="center", va="center")
        ax.axis("off")
    else:
        plot = short(rows.copy())
        plot = plot[plot["adaptive_dynamics"].astype(bool)].copy()
        if plot.empty:
            ax.text(0.5, 0.5, "No adaptive-dynamics rows", ha="center", va="center")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return path
        plot["building_num"] = plot["building"].str.extract(r"(\d+)").astype(float)
        plot = plot.sort_values("building_num")
        x = np.arange(len(plot))
        width = 0.35
        ax.bar(x - width / 2, plot["val_rmse_before_c"], width, label="before", alpha=0.55)
        ax.bar(x + width / 2, plot["val_rmse_after_c"], width, label="after", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(plot["building"], rotation=70, ha="right", fontsize=8)
        ax.set_ylabel("Validation one-step Tin RMSE [C]")
        ax.set_title("GNU-RL adaptive dynamics: one-step model error before/after")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def append_once(path: Path, cells: list[nbf.NotebookNode]) -> None:
    base._append_once(path, cells)


def main() -> None:
    configure_base()
    base.FIG_DIR.mkdir(parents=True, exist_ok=True)
    base.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    building_kpis = base._building_kpis()
    mean_kpis = base._mean_building_kpis(building_kpis)
    district_frames = [base._baseline_district(), *[base._district_frame(run) for run in base.RUNS]]
    district_kpis = base._district_kpis(district_frames)
    district_hourly = final.save_district_mean_hourly(district_frames)
    term_by_building, hourly = base._term_diagnostics()
    term_summary = (
        term_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    bad_delta = base._sac_bad_delta()
    rewards, losses = base._sac_training_frames()
    dynamics_rows = _rlmpc_dynamics_rows()
    dynamics_summary = _rlmpc_dynamics_summary(dynamics_rows)

    prefix = PREFIX
    building_kpis.to_csv(base.ANALYSIS_DIR / f"{prefix}_building_kpis_all25.csv", index=False)
    final.compact_building_table(building_kpis).to_csv(
        base.ANALYSIS_DIR / f"{prefix}_building_kpis_all25_compact.csv", index=False
    )
    mean_kpis.to_csv(base.ANALYSIS_DIR / f"{prefix}_mean_building_kpis.csv", index=False)
    district_kpis.to_csv(base.ANALYSIS_DIR / f"{prefix}_district_kpis.csv", index=False)
    district_hourly.to_csv(base.ANALYSIS_DIR / f"{prefix}_district_mean_hourly_profile.csv", index=False)
    term_by_building.to_csv(base.ANALYSIS_DIR / f"{prefix}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(base.ANALYSIS_DIR / f"{prefix}_term_diagnostics.csv", index=False)
    hourly.to_csv(base.ANALYSIS_DIR / f"{prefix}_hourly_bias_response.csv", index=False)
    bad_delta.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_bad_delta.csv", index=False)
    rewards.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_online_training_rewards_raw.csv", index=False)
    losses.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_training_losses_raw.csv", index=False)
    dynamics_rows.to_csv(base.ANALYSIS_DIR / f"{prefix}_rlmpc_dynamics_by_building.csv", index=False)
    dynamics_summary.to_csv(base.ANALYSIS_DIR / f"{prefix}_rlmpc_dynamics_summary.csv", index=False)

    comfort_paths = {run["key"]: base._make_comfort_grid(run) for run in base.RUNS}
    safe_pi_path = base._make_safe_pi_grid(base._safe_pi_january_rows())
    district_load_path, district_daily_path = base._make_district_plots(district_frames)
    terms_path, response_path = base._make_term_plots(term_summary, hourly)
    training_paths, reward_summary, loss_summary = base._make_sac_training_plots(rewards, losses)
    reward_mean_path = final.make_final_reward_plot(reward_summary)
    bad_delta_path = base._make_bad_delta_plot(bad_delta)
    dynamics_path = _make_rlmpc_dynamics_plot(dynamics_rows)

    reward_summary.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_online_reward_summary.csv", index=False)
    loss_summary.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_loss_summary.csv", index=False)

    term_mean = base._pivot_terms(term_summary, "mean")
    term_max = base._pivot_terms(term_summary, "max")
    bad_delta_display, still_bad_display = final.sac_bad_tables(bad_delta)
    all_building_display = final.compact_building_table(building_kpis)
    reward_display = final.sac_reward_table(reward_summary)
    loss_core, loss_alpha = final.sac_loss_tables(loss_summary)

    overview = f"""{MARKER}

Final no-occupant comfort-only comparison with the fixed and adaptive GNU-RL/RLMPC variants.

Controller labels:

`{MPC_LABEL}`: MPC with January-only 2R2C+Kalman RC fit, horizon 24, deployed in February.

`{RLMPC_FIXED_LABEL}`: fixed GNU-RL/RLMPC using the completed 1R1C imitation-learned RC policy, horizon 6, and the same normalized comfort-only inner MPC cost. PPO is collected but the inner cost weights stay fixed at `(0.1, 0.7, 0.2)`, so the online cost update is skipped.

`{RLMPC_ADAPTIVE_LABEL}`: adaptive-dynamics GNU-RL/RLMPC with the same 1R1C horizon-6 pipeline and same fixed reward/cost, but after the stochastic February rollout it updates the differentiable RC dynamics from one-step `Tin(k+1)` prediction error. The comfort objective is not changed.

`{SAC_BASE_LABEL}` plus `{SAC_BAD_LABEL}`: SAC safe-PI bootstrap on January, then 9 reward-training episodes. Buildings with February discomfort above 10% after episode 9 were resumed for 10 more reward-training episodes. The final SAC row is the hybrid final deployment policy.

Cost/reward used here:

`D_k = 0.1*((Tin-(Tsp+bias))/2)^2 + 0.7*(max(Tsp-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tsp-2,0)/2)^2`

`J_k = D_k`, `reward = -D_k`.

There is no GridStress term, no electricity term, no peak term, and no ramp term. The three comfort weights sum to 1.0. `bias=+1 C` in the preheat windows, `bias=-1 C` in the reduce windows, and cold/hot guard penalties are measured against the true setpoint with a `Tsp +/- 2 C` guard.

Mean building comfort KPIs:

{markdown_table(final.mean_comfort_table(mean_kpis), digits=3)}

Mean building operation KPIs:

{markdown_table(final.mean_ops_table(mean_kpis), digits=3)}

District aggregate KPIs - load shape:

{markdown_table(final.district_shape_table(district_kpis), digits=3)}

District aggregate KPIs - tracking:

{markdown_table(final.district_tracking_table(district_kpis), digits=3)}

Weighted term diagnostics - mean:

{markdown_table(term_mean, digits=4)}

Weighted term diagnostics - max:

{markdown_table(term_max, digits=3)}

GNU-RL fixed/adaptive dynamics summary:

{markdown_table(dynamics_summary, digits=4)}

All-building KPI table, 25 buildings x 4 controllers:

{markdown_table(all_building_display, digits=3)}

SAC bad-building continuation deltas:

{markdown_table(bad_delta_display, digits=2)}

SAC buildings still above 10% discomfort after the bad-building continuation:

{markdown_table(still_bad_display, digits=2)}
"""
    if not reward_display.empty:
        overview += f"""

SAC online reward summary - per-step means:

{markdown_table(reward_display, digits=3)}
"""
    if not loss_core.empty:
        overview += f"""

SAC critic/policy loss summary, final episodes:

{markdown_table(loss_core, digits=3)}

SAC entropy/alpha loss summary, final episodes:

{markdown_table(loss_alpha, digits=3)}
"""
    overview += f"""

Saved compact tables:

- `analysis/{prefix}_building_kpis_all25.csv`
- `analysis/{prefix}_building_kpis_all25_compact.csv`
- `analysis/{prefix}_mean_building_kpis.csv`
- `analysis/{prefix}_district_kpis.csv`
- `analysis/{prefix}_district_mean_hourly_profile.csv`
- `analysis/{prefix}_term_diagnostics.csv`
- `analysis/{prefix}_term_diagnostics_by_building.csv`
- `analysis/{prefix}_rlmpc_dynamics_by_building.csv`
- `analysis/{prefix}_rlmpc_dynamics_summary.csv`
- `analysis/{prefix}_sac_bad_delta.csv`
- `analysis/{prefix}_sac_online_reward_summary.csv`
- `analysis/{prefix}_sac_loss_summary.csv`
"""

    comfort_titles = {
        "mpc": "MPC plot_comfort all buildings",
        "rlmpc_fixed": "Fixed GNU-RL plot_comfort all buildings",
        "rlmpc_adaptive": "Adaptive GNU-RL plot_comfort all buildings",
        "sac": "SAC final hybrid plot_comfort all buildings",
    }
    image_cells = [
        *[image(comfort_paths[key], comfort_titles[key]) for key in ["mpc", "rlmpc_fixed", "rlmpc_adaptive", "sac"]],
        image(safe_pi_path, "SAC safe PI January episode 0 comfort"),
        image(district_load_path, "District load time series"),
        image(district_daily_path, "Mean hourly district load profile"),
        image(terms_path, "Weighted stage-cost terms"),
        image(response_path, "Hourly time-bias response"),
        image(dynamics_path, "GNU-RL adaptive dynamics one-step RMSE"),
        image(bad_delta_path, "SAC bad-building continuation discomfort"),
        image(reward_mean_path, "SAC online reward per-step mean"),
        *[image(path, path.stem.replace("_", " ")) for path in training_paths],
    ]
    cells = [nbf.v4.new_markdown_cell(overview), *image_cells]

    for notebook in NOTEBOOKS:
        if notebook.exists():
            append_once(notebook, cells)

    print(f"Done: {PREFIX}")


if __name__ == "__main__":
    main()
