#!/usr/bin/env python3
"""Append final no-occupant normalized-D MPC/RLMPC/SAC comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

import scripts.append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis as base


PREFIX = "ce1_final_no_occ_normD010702_controller_comparison"
MARKER = f"## Final No-Occupant Normalized-D MPC/RLMPC/SAC - `{PREFIX}`"

MPC_LABEL = "ce1_final_no_occ_D010702_mpc_2r2c_kalman"
RLMPC_LABEL = "ce1_final_no_occ_D010702_rlmpc_1r1c_h6"
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
    base.RLMPC_LABEL = RLMPC_LABEL
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
            "key": "rlmpc",
            "display": "GNU-RL/RLMPC 1R1C H6 normalized reward",
            "controller": "rlmpc",
            "label": RLMPC_LABEL,
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
        base.RUNS[1]["display"]: "RLMPC",
        base.RUNS[2]["display"]: "SAC hybrid",
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


def compact_building_table(building_kpis: pd.DataFrame) -> pd.DataFrame:
    out = short(
        building_kpis[
            [
                "controller",
                "building",
                "electricity_consumption_total",
                "discomfort_%",
                "cold_%",
                "hot_%",
                "kelvin_hours",
                "daily_peak_average",
                "ramping_average",
                "feb_tin_max",
                "source_label",
            ]
        ]
    ).copy()
    return out.rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
            "feb_tin_max": "Tin_max",
            "source_label": "label",
        }
    )


def mean_comfort_table(mean_kpis: pd.DataFrame) -> pd.DataFrame:
    return short(
        mean_kpis[
            [
                "controller",
                "electricity_consumption_total",
                "discomfort_%",
                "cold_%",
                "hot_%",
                "kelvin_hours",
            ]
        ]
    ).rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
        }
    )


def mean_ops_table(mean_kpis: pd.DataFrame) -> pd.DataFrame:
    return short(
        mean_kpis[["controller", "daily_peak_average", "ramping_average", "feb_tin_max"]]
    ).rename(
        columns={
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
            "feb_tin_max": "Tin_max",
        }
    )


def district_shape_table(district_kpis: pd.DataFrame) -> pd.DataFrame:
    return short(
        district_kpis[
            [
                "controller",
                "total_vs_citylearn",
                "peak_vs_citylearn",
                "daily_profile_peak_kwh",
                "daily_profile_peak_hour",
            ]
        ]
    ).rename(
        columns={
            "total_vs_citylearn": "total_vs_CL",
            "peak_vs_citylearn": "peak_vs_CL",
            "daily_profile_peak_kwh": "daily_peak_kWh",
            "daily_profile_peak_hour": "daily_peak_hr",
        }
    )


def district_tracking_table(district_kpis: pd.DataFrame) -> pd.DataFrame:
    return short(
        district_kpis[["controller", "ramping_vs_citylearn", "target_rmse_kwh", "above_target_kwh"]]
    ).rename(
        columns={
            "ramping_vs_citylearn": "ramp_vs_CL",
            "target_rmse_kwh": "target_RMSE",
            "above_target_kwh": "above_target",
        }
    )


def sac_bad_tables(bad_delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bad_delta.empty:
        return pd.DataFrame(), pd.DataFrame()
    display = bad_delta[
        [
            "building",
            "discomfort_%_plus9",
            "discomfort_%_bad10",
            "delta_discomfort_%",
            "cold_%_bad10",
            "hot_%_bad10",
            "energy_bad10",
        ]
    ].rename(
        columns={
            "discomfort_%_plus9": "disc_ep9",
            "discomfort_%_bad10": "disc_final",
            "delta_discomfort_%": "delta_disc",
            "cold_%_bad10": "cold_final",
            "hot_%_bad10": "hot_final",
            "energy_bad10": "energy_final",
        }
    )
    still_bad = display[display["disc_final"] > 10.0].copy()
    return display, still_bad


def sac_reward_table(reward_summary: pd.DataFrame) -> pd.DataFrame:
    if reward_summary.empty:
        return pd.DataFrame()
    out = reward_summary[["episode", "raw_reward_mean", "reward_mean", "reset_fraction", "action_mean"]].copy()
    out["reset_%"] = 100.0 * out.pop("reset_fraction")
    return out.rename(columns={"raw_reward_mean": "raw_mean", "reward_mean": "learning_mean"})[
        ["episode", "raw_mean", "learning_mean", "reset_%", "action_mean"]
    ]


def sac_loss_tables(loss_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if loss_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    tail = loss_summary.tail(12).copy()
    core_cols = [
        "episode",
        "soft_q1_loss_mean",
        "soft_q1_loss_max",
        "soft_q2_loss_mean",
        "soft_q2_loss_max",
        "policy_loss_mean",
        "policy_loss_max",
    ]
    alpha_cols = ["episode", "alpha_loss_mean", "alpha_loss_max", "alpha_mean", "alpha_max"]
    core = tail[[column for column in core_cols if column in tail.columns]].rename(
        columns={
            "soft_q1_loss_mean": "q1_mean",
            "soft_q1_loss_max": "q1_max",
            "soft_q2_loss_mean": "q2_mean",
            "soft_q2_loss_max": "q2_max",
            "policy_loss_mean": "pi_mean",
            "policy_loss_max": "pi_max",
        }
    )
    alpha = tail[[column for column in alpha_cols if column in tail.columns]].rename(
        columns={
            "alpha_loss_mean": "alpha_loss_mean",
            "alpha_loss_max": "alpha_loss_max",
            "alpha_mean": "alpha_mean",
            "alpha_max": "alpha_max",
        }
    )
    return core, alpha


def save_district_mean_hourly(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    daily = (
        combined.assign(hour=combined["time_step"] % 24)
        .groupby(["controller", "hour"], as_index=False)["load"]
        .mean()
    )
    return short(daily)


def make_final_reward_plot(reward_summary: pd.DataFrame) -> Path:
    path = base.FIG_DIR / "sac_online_training_reward_mean_only_readable.png"
    if reward_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No SAC reward rows", ha="center", va="center")
        ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(
            reward_summary["episode"],
            reward_summary["raw_reward_mean"],
            marker="o",
            lw=1.5,
            label="Raw reward mean per step",
        )
        if not np.allclose(
            reward_summary["raw_reward_mean"],
            reward_summary["reward_mean"],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ):
            ax.plot(
                reward_summary["episode"],
                reward_summary["reward_mean"],
                marker="o",
                lw=1.2,
                ls="--",
                label="Learning reward mean per step",
            )
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("SAC online training reward by episode - per-step mean")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean reward per step")
        ax.grid(alpha=0.25)
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
    district_hourly = save_district_mean_hourly(district_frames)
    term_by_building, hourly = base._term_diagnostics()
    term_summary = (
        term_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    bad_delta = base._sac_bad_delta()
    rewards, losses = base._sac_training_frames()

    prefix = PREFIX
    building_kpis.to_csv(base.ANALYSIS_DIR / f"{prefix}_building_kpis_all25.csv", index=False)
    compact_building_table(building_kpis).to_csv(
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

    comfort_paths = {run["key"]: base._make_comfort_grid(run) for run in base.RUNS}
    safe_pi_path = base._make_safe_pi_grid(base._safe_pi_january_rows())
    district_load_path, district_daily_path = base._make_district_plots(district_frames)
    terms_path, response_path = base._make_term_plots(term_summary, hourly)
    training_paths, reward_summary, loss_summary = base._make_sac_training_plots(rewards, losses)
    reward_mean_path = make_final_reward_plot(reward_summary)
    bad_delta_path = base._make_bad_delta_plot(bad_delta)

    reward_summary.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_online_reward_summary.csv", index=False)
    loss_summary.to_csv(base.ANALYSIS_DIR / f"{prefix}_sac_loss_summary.csv", index=False)

    term_mean = base._pivot_terms(term_summary, "mean")
    term_max = base._pivot_terms(term_summary, "max")
    bad_delta_display, still_bad_display = sac_bad_tables(bad_delta)
    all_building_display = compact_building_table(building_kpis)
    reward_display = sac_reward_table(reward_summary)
    loss_core, loss_alpha = sac_loss_tables(loss_summary)

    overview = f"""{MARKER}

Final no-occupant comfort-only comparison after dropping the GridStress/load term.

Controller labels:

`{MPC_LABEL}`: MPC with January-only 2R2C+Kalman RC fit, horizon 24, deployed in February.

`{RLMPC_LABEL}`: GNU-RL/RLMPC using the 1R1C horizon-6 pipeline that previously gave the stable RLMPC behavior, with the same normalized comfort reward.

`{SAC_BASE_LABEL}` plus `{SAC_BAD_LABEL}`: SAC safe-PI bootstrap on January, then 9 reward-training episodes. Buildings with February discomfort above 10% after episode 9 were resumed for 10 more reward-training episodes. The final SAC row is the hybrid final deployment policy. This SAC run was launched outside the sandbox so PyTorch could use CUDA.

Cost/reward used here:

`D_k = 0.1*((Tin-(Tsp+bias))/2)^2 + 0.7*(max(Tsp-2-Tin,0)/1.5)^2 + 0.2*(max(Tin-Tsp-2,0)/2)^2`

`J_k = D_k`, `reward = -D_k`.

There is no GridStress term, no electricity term, no peak term, and no ramp term in this final no-occupant objective. The three displayed weights sum to 1.0. `bias=+1 C` in the preheat windows, `bias=-1 C` in the reduce windows, and cold/hot guard penalties are measured against the true setpoint with a `Tsp +/- 2 C` guard.

Mean building comfort KPIs:

{markdown_table(mean_comfort_table(mean_kpis), digits=3)}

Mean building operation KPIs:

{markdown_table(mean_ops_table(mean_kpis), digits=3)}

District aggregate KPIs - load shape:

{markdown_table(district_shape_table(district_kpis), digits=3)}

District aggregate KPIs - tracking:

{markdown_table(district_tracking_table(district_kpis), digits=3)}

Weighted term diagnostics - mean:

{markdown_table(term_mean, digits=4)}

Weighted term diagnostics - max:

{markdown_table(term_max, digits=3)}

All-building KPI table, 25 buildings x 3 controllers:

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
- `analysis/{prefix}_sac_bad_delta.csv`
- `analysis/{prefix}_sac_online_reward_summary.csv`
- `analysis/{prefix}_sac_loss_summary.csv`
"""

    image_cells = [
        image(comfort_paths["mpc"], "MPC plot_comfort all buildings"),
        image(comfort_paths["rlmpc"], "RLMPC plot_comfort all buildings"),
        image(comfort_paths["sac"], "SAC final hybrid plot_comfort all buildings"),
        image(safe_pi_path, "SAC safe PI January episode 0 comfort"),
        image(district_load_path, "District load time series"),
        image(district_daily_path, "Mean hourly district load profile"),
        image(terms_path, "Weighted stage-cost terms"),
        image(response_path, "Hourly time-bias response"),
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
