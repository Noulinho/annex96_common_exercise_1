#!/usr/bin/env python3
"""Append current-best MPC, replicated old RLMPC, and latest SAC comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

import scripts.append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis as base


PREFIX = "ce1_current_best_mpc_oldrlmpc_sac_comparison"
MARKER = f"## Current Best MPC / Old RLMPC / Latest SAC Comparison - `{PREFIX}`"

MPC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman"
RLMPC_LABEL = "ce1_time_bias_guard_1c_1r1c_rlmpc_replicate"
SAC_BASE_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi_plus9_fair_a1"
SAC_BAD_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi_plus9_bad10_fair_a1"

NOTEBOOKS = [
    base.ANALYSIS_DIR / "02_mpc_results.ipynb",
    base.ANALYSIS_DIR / "04_rlmpc_results.ipynb",
    base.ANALYSIS_DIR / "08_sac_results.ipynb",
    base.ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
    base.ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
    base.ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
]


def _configure_base() -> None:
    base.FIG_DIR = Path("figures") / PREFIX
    base.MARKER = MARKER
    base.MPC_LABEL = MPC_LABEL
    base.RLMPC_LABEL = RLMPC_LABEL
    base.SAC_BASE_LABEL = SAC_BASE_LABEL
    base.SAC_BAD_LABEL = SAC_BAD_LABEL
    base.RUNS = [
        {
            "key": "mpc",
            "display": "MPC 2R2C+Kalman working",
            "controller": "mpc",
            "label": MPC_LABEL,
            "hybrid": False,
        },
        {
            "key": "rlmpc",
            "display": "GNU-RL/RLMPC old 1R1C replicated",
            "controller": "rlmpc",
            "label": RLMPC_LABEL,
            "hybrid": False,
        },
        {
            "key": "sac",
            "display": "SAC latest safe-PI hybrid",
            "controller": "sac_merlin",
            "label": SAC_BASE_LABEL,
            "hybrid": True,
        },
    ]
    base.SHORT_CONTROLLER_NAMES = {
        "CityLearn baseline": "CityLearn",
        base.RUNS[0]["display"]: "MPC",
        base.RUNS[1]["display"]: "Old RLMPC",
        base.RUNS[2]["display"]: "SAC hybrid",
    }


def _markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    return base._markdown_table(df, digits=digits, max_rows=max_rows)


def _short(df: pd.DataFrame) -> pd.DataFrame:
    return base._short_controller_names(df)


def _image(path: Path, title: str) -> nbf.NotebookNode:
    return base._image(path, title)


def _term_diagnostics_realized_heat() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    hourly_rows = []
    w = base.WEIGHTS
    for run in base.RUNS:
        for building in base.BUILDINGS:
            data = base._read(base._rollout_path(run, building["result_id"]))
            tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            tsp = base._setpoint(data)
            heat = base._heating(data)
            time_step = base._time_step(data)
            n = min(len(tin), len(tsp), len(heat), len(time_step))
            tin, tsp, heat, time_step = tin[:n], tsp[:n], heat[:n], time_step[:n]
            bias, stress = base._bias_and_stress(n)
            tref = tsp + bias
            cold_guard = np.maximum(tsp - w["guard_band"] - tin, 0.0)
            hot_guard = np.maximum(tin - tsp - w["guard_band"], 0.0)
            track = w["comfort_weight"] * w["track_weight"] * ((tin - tref) / w["track_scale"]) ** 2
            cold = w["comfort_weight"] * w["cold_weight"] * (cold_guard / w["cold_scale"]) ** 2
            hot = w["comfort_weight"] * w["hot_weight"] * (hot_guard / w["hot_scale"]) ** 2
            shift = w["shift_weight"] * stress * heat / w["power_scale"]
            stage = track + cold + hot + shift
            guard_violation = (cold_guard > 0.0) | (hot_guard > 0.0)
            for term, values in {
                "tracking_target": track,
                "guard_cold": cold,
                "guard_hot": hot,
                "time_weighted_heat": shift,
                "stage_cost": stage,
                "mae_to_true_tsp_c": np.abs(tin - tsp),
                "mae_to_biased_tref_c": np.abs(tin - tref),
            }.items():
                rows.append(
                    {
                        "controller": run["display"],
                        "building": building["display"],
                        "term": term,
                        "mean": float(np.nanmean(values)),
                        "max": float(np.nanmax(values)),
                    }
                )
            rows.append(
                {
                    "controller": run["display"],
                    "building": building["display"],
                    "term": "guard_violation_%",
                    "mean": float(100.0 * np.nanmean(guard_violation)),
                    "max": float(100.0 * np.nanmax(guard_violation)),
                }
            )
            hourly_rows.append(
                pd.DataFrame(
                    {
                        "controller": run["display"],
                        "hour": time_step % 24,
                        "tin_minus_tsp": tin - tsp,
                        "action_proxy_heat": heat,
                        "guard_violation": guard_violation.astype(float),
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def _make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path, pd.DataFrame]:
    combined = pd.concat(frames, ignore_index=True)
    base.FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=base.SHORT_CONTROLLER_NAMES.get(name, name))
    ax.plot(frames[0]["time_step"], frames[0]["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("District load - February")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    load_path = base.FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = frames[0].assign(hour=frames[0]["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.4, label=base.SHORT_CONTROLLER_NAMES.get(name, name))
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Mean hourly district load - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    daily_path = base.FIG_DIR / "district_mean_hourly_profile.png"
    fig.savefig(daily_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path, daily


def _mpc_sanity_table() -> pd.DataFrame:
    path = base.ANALYSIS_DIR / "ce1_time_bias_guard_1c_2r2c_kalman_mpc_rerun_verify_original_diff.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    cols = ["building", "rollout_max_abs_diff", "kpi_max_abs_diff", "rollout_rows_original", "rollout_rows_rerun"]
    return data[cols].head(25)


def _rlmpc_replication_verdict() -> pd.DataFrame:
    path = base.ANALYSIS_DIR / "ce1_time_bias_guard_1c_1r1c_rlmpc_replicate_replication_verdict.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _latest_sac_bad_table(bad_delta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            "discomfort_%_plus9": "disc_plus9",
            "discomfort_%_bad10": "disc_bad10",
            "delta_discomfort_%": "delta_disc",
            "cold_%_bad10": "cold_bad10",
            "hot_%_bad10": "hot_bad10",
            "energy_bad10": "energy",
        }
    )
    still_bad = display[display["disc_bad10"] > 10.0].copy()
    return display, still_bad


def main() -> None:
    _configure_base()
    base.FIG_DIR.mkdir(parents=True, exist_ok=True)
    base.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    building_kpis = base._building_kpis()
    mean_kpis = base._mean_building_kpis(building_kpis)
    district_frames = [base._baseline_district(), *[base._district_frame(run) for run in base.RUNS]]
    district_kpis = base._district_kpis(district_frames)
    term_by_building, hourly = _term_diagnostics_realized_heat()
    term_summary = (
        term_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    bad_delta = base._sac_bad_delta()
    rewards, losses = base._sac_training_frames()

    building_kpis.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_building_kpis.csv", index=False)
    mean_kpis.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_mean_building_kpis.csv", index=False)
    district_kpis.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_district_kpis.csv", index=False)
    term_by_building.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_term_diagnostics.csv", index=False)
    hourly.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_hourly_bias_response.csv", index=False)
    bad_delta.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_bad_delta.csv", index=False)
    rewards.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_online_training_rewards_raw.csv", index=False)
    losses.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_training_losses_raw.csv", index=False)

    comfort_paths = {run["key"]: base._make_comfort_grid(run) for run in base.RUNS}
    safe_pi_path = base._make_safe_pi_grid(base._safe_pi_january_rows())
    district_load_path, district_daily_path, district_daily = _make_district_plots(district_frames)
    district_daily.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_district_mean_hourly_profile.csv", index=False)
    terms_path, response_path = base._make_term_plots(term_summary, hourly)
    training_paths, reward_summary, loss_summary = base._make_sac_training_plots(rewards, losses)
    bad_delta_path = base._make_bad_delta_plot(bad_delta)
    reward_summary.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_online_reward_summary.csv", index=False)
    loss_summary.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_loss_summary.csv", index=False)

    mean_comfort = _short(
        mean_kpis[["controller", "electricity_consumption_total", "discomfort_%", "cold_%", "hot_%", "kelvin_hours"]]
    ).rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
        }
    )
    mean_ops = _short(mean_kpis[["controller", "daily_peak_average", "ramping_average", "feb_tin_max"]]).rename(
        columns={"daily_peak_average": "daily_peak", "ramping_average": "ramping", "feb_tin_max": "Tin_max"}
    )
    district_shape = _short(
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
    district_tracking = _short(
        district_kpis[["controller", "ramping_vs_citylearn", "target_rmse_kwh", "above_target_kwh"]]
    ).rename(
        columns={
            "ramping_vs_citylearn": "ramp_vs_CL",
            "target_rmse_kwh": "target_RMSE",
            "above_target_kwh": "above_target",
        }
    )
    term_mean = base._pivot_terms(term_summary, "mean")
    term_max = base._pivot_terms(term_summary, "max")
    sac_bad_display, sac_still_bad = _latest_sac_bad_table(bad_delta)
    mpc_sanity = _mpc_sanity_table()
    rlm_verdict = _rlmpc_replication_verdict()

    reward_display = pd.DataFrame()
    if not reward_summary.empty:
        reward_display = reward_summary[["episode", "raw_reward_mean", "reward_mean", "reset_fraction", "action_mean"]].copy()
        reward_display["reset_%"] = 100.0 * reward_display.pop("reset_fraction")
        reward_display = reward_display.rename(columns={"raw_reward_mean": "raw_mean", "reward_mean": "replay_mean"})

    loss_display = pd.DataFrame()
    alpha_display = pd.DataFrame()
    if not loss_summary.empty:
        loss_display = loss_summary.tail(10)[
            [
                "episode",
                "soft_q1_loss_mean",
                "soft_q1_loss_max",
                "soft_q2_loss_mean",
                "soft_q2_loss_max",
                "policy_loss_mean",
                "policy_loss_max",
            ]
        ].rename(
            columns={
                "soft_q1_loss_mean": "q1_mean",
                "soft_q1_loss_max": "q1_max",
                "soft_q2_loss_mean": "q2_mean",
                "soft_q2_loss_max": "q2_max",
                "policy_loss_mean": "pi_mean",
                "policy_loss_max": "pi_max",
            }
        )
        alpha_cols = [c for c in ["episode", "alpha_loss_mean", "alpha_loss_max", "alpha_mean", "alpha_max"] if c in loss_summary.columns]
        alpha_display = loss_summary.tail(10)[alpha_cols].copy() if alpha_cols else pd.DataFrame()

    overview = f"""{MARKER}

This is the requested full visual comparison using the current artifacts you pointed to:

`{MPC_LABEL}`: the working recovered MPC run, January-only 2R2C+Kalman RC fit, February deployment.

`{RLMPC_LABEL}`: the old GNU-RL/RLMPC run replicated exactly under a fresh label. This is the old 1R1C/horizon-6 RLMPC recipe; it is included because it reproduces the old load-shifting behavior exactly.

`{SAC_BASE_LABEL}` plus `{SAC_BAD_LABEL}`: latest SAC safe-PI run and bad-building continuation. The SAC row is a hybrid: bad+10 policies where those artifacts exist, plus9 policies elsewhere.

Important sanity note: this is a **current-best artifact comparison**, not a same-model fairness claim. MPC and latest SAC are 2R2C+Kalman-family artifacts; the RLMPC row is the old 1R1C replicated controller.

Cost/reward accounting used for diagnostics below:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

`J_real,k = 0.5*D_k + 0.5*GridStress_k*P_heat,real,k/5`, `reward = -J_real,k`.

`bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and `GridStress=baseline_district_daily_mean_exp2_norm`.

MPC recovery sanity check:

{_markdown_table(mpc_sanity, digits=6)}

Old RLMPC replication verdict:

{_markdown_table(rlm_verdict, digits=6)}

Mean building comfort KPIs:

{_markdown_table(mean_comfort, digits=3)}

Mean building operation KPIs:

{_markdown_table(mean_ops, digits=3)}

District aggregate KPIs - load shape:

{_markdown_table(district_shape, digits=3)}

District aggregate KPIs - tracking:

{_markdown_table(district_tracking, digits=3)}

Weighted term diagnostics using realized heating electricity - mean:

{_markdown_table(term_mean, digits=3)}

Weighted term diagnostics using realized heating electricity - max:

{_markdown_table(term_max, digits=3)}

SAC bad-building continuation deltas:

{_markdown_table(sac_bad_display, digits=2)}

SAC buildings still above 10% discomfort after bad +10:

{_markdown_table(sac_still_bad, digits=2)}
"""
    if not reward_display.empty:
        overview += f"""

SAC online reward summary:

{_markdown_table(reward_display, digits=3)}
"""
    if not loss_display.empty:
        overview += f"""

SAC critic/policy loss summary, final 10 episodes:

{_markdown_table(loss_display, digits=3)}
"""
    if not alpha_display.empty:
        overview += f"""

SAC entropy/alpha loss summary, final 10 episodes:

{_markdown_table(alpha_display, digits=3)}
"""
    overview += f"""

Saved compact tables:

- `analysis/{PREFIX}_building_kpis.csv`
- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_term_diagnostics.csv`
- `analysis/{PREFIX}_district_mean_hourly_profile.csv`
- `analysis/{PREFIX}_sac_online_reward_summary.csv`
- `analysis/{PREFIX}_sac_loss_summary.csv`
"""

    image_cells = [
        _image(district_load_path, "District load time series"),
        _image(district_daily_path, "District mean hourly profile"),
        _image(comfort_paths["mpc"], "MPC comfort all buildings"),
        _image(comfort_paths["rlmpc"], "Old replicated GNU-RL/RLMPC comfort all buildings"),
        _image(comfort_paths["sac"], "Latest SAC hybrid comfort all buildings"),
        _image(safe_pi_path, "SAC safe PI January episode 0 comfort"),
        _image(terms_path, "Weighted reward terms"),
        _image(response_path, "Hourly time-bias response"),
        _image(bad_delta_path, "SAC bad-building continuation discomfort"),
        *[_image(path, path.stem.replace("_", " ")) for path in training_paths],
    ]
    cells = [nbf.v4.new_markdown_cell(overview), *image_cells]

    for notebook in NOTEBOOKS:
        if notebook.exists():
            base._append_once(notebook, cells)

    print("Done.")


if __name__ == "__main__":
    main()
