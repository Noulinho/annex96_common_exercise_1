#!/usr/bin/env python
"""Append CE1 2R2C h24 heating-reward GNU-RL replicate comparison."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

import experiments.scripts.append_ce1_fair_2r2c_kalman_rlmpc_sac_analysis as base


PREFIX = "ce1_time_bias_guard_1c_2r2c_kalman_h24_heating_g0_replicate"
MARKER = "## CE1 2R2C+Kalman h24 GNU-RL Heating-Reward Replicate"

MPC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman"
RLMPC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_rlmpc_h24_heating_g0_replicate"
SAC_BASE_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi_plus9_fair_a1"
SAC_BAD_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi_plus9_bad10_fair_a1"


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
            "display": "MPC 2R2C+Kalman",
            "controller": "mpc",
            "label": MPC_LABEL,
            "hybrid": False,
        },
        {
            "key": "rlmpc",
            "display": "GNU-RL/RLMPC 2R2C+Kalman h24 heating-g0",
            "controller": "rlmpc",
            "label": RLMPC_LABEL,
            "hybrid": False,
        },
        {
            "key": "sac",
            "display": "SAC safe-PI heating reward (plus9 + bad10)",
            "controller": "sac_merlin",
            "label": SAC_BASE_LABEL,
            "hybrid": True,
        },
    ]
    base.SHORT_CONTROLLER_NAMES = {
        "CityLearn baseline": "CityLearn",
        base.RUNS[0]["display"]: "MPC",
        base.RUNS[1]["display"]: "GNU-RL",
        base.RUNS[2]["display"]: "SAC",
    }


def _summary_row(controller: str, building_id: str, label: str) -> pd.Series:
    path = base.RESULTS / "summaries" / controller / building_id / "no_occupant" / "summary.csv"
    if not path.exists():
        return pd.Series(dtype=object)
    data = base._read(path)
    rows = data[data["case"].astype(str) == label]
    return rows.tail(1).iloc[0] if not rows.empty else pd.Series(dtype=object)


def _metric_json(controller: str, building_id: str, label: str) -> dict[str, object]:
    path = base.RESULTS / "summaries" / controller / building_id / "no_occupant" / f"{label}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _recipe_metadata_check() -> pd.DataFrame:
    rows = []
    for building in base.BUILDINGS:
        building_id = building["result_id"]
        rlm = _summary_row("rlmpc", building_id, RLMPC_LABEL)
        sac_label = base._sac_final_label(building_id)
        sac = _metric_json("sac_merlin", building_id, sac_label)
        mpc = _metric_json("mpc", building_id, MPC_LABEL)
        h24 = Path(
            f"results/models/rlmpc/offline/{building_id}/ce1_time_bias_guard_1c_2r2c_kalman_h24.pt"
        ).exists()
        rows.append(
            {
                "building": building["display"],
                "building_id": building_id,
                "rlmpc_model": rlm.get("model_type", ""),
                "rlmpc_observer": rlm.get("two_state_observer", ""),
                "h24_checkpoint": bool(h24),
                "rlmpc_reward_src": rlm.get("reward_power_source", ""),
                "sac_label": sac_label,
                "sac_reward_src": sac.get("reward_power_source", ""),
                "mpc_gridstress": mpc.get("grid_stress_mode", ""),
                "rlmpc_gridstress": rlm.get("grid_stress_mode", ""),
                "sac_gridstress": sac.get("grid_stress_mode", ""),
                "rlmpc_guard_pen": rlm.get("matched_mpc_time_bias_guard_penalty", 0.0),
            }
        )
    return pd.DataFrame(rows)


def _term_diagnostics_heating() -> tuple[pd.DataFrame, pd.DataFrame]:
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
                        "hour": (time_step + base.FEB_START) % 24,
                        "tin_minus_tsp": tin - tsp,
                        # Keep the helper's expected column name; values are realized heating electricity here.
                        "action_proxy_heat": heat,
                        "guard_violation": guard_violation.astype(float),
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def _qpth_note() -> str:
    return (
        "The run emitted repeated qpth inaccurate-solution warnings during the h24 2R2C rollout. "
        "The command was run interactively, not tee'd to a log, so the exact count is not available."
    )


def _markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    return base._markdown_table(df, digits=digits, max_rows=max_rows)


def _short(df: pd.DataFrame) -> pd.DataFrame:
    return base._short_controller_names(df)


def main() -> None:
    _configure_base()
    base.FIG_DIR.mkdir(parents=True, exist_ok=True)
    base.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    building_kpis = base._building_kpis()
    mean_kpis = base._mean_building_kpis(building_kpis)
    district_frames = [base._baseline_district(), *[base._district_frame(run) for run in base.RUNS]]
    district_kpis = base._district_kpis(district_frames)
    term_by_building, hourly = _term_diagnostics_heating()
    term_summary = (
        term_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    metadata_check = _recipe_metadata_check()
    rewards, losses = base._sac_training_frames()
    bad_delta = base._sac_bad_delta()

    building_kpis.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_building_kpis.csv", index=False)
    mean_kpis.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_mean_building_kpis.csv", index=False)
    district_kpis.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_district_kpis.csv", index=False)
    term_by_building.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_term_diagnostics.csv", index=False)
    hourly.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_hourly_bias_response.csv", index=False)
    metadata_check.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_metadata_check.csv", index=False)
    bad_delta.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_bad_delta.csv", index=False)
    rewards.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_online_training_rewards_raw.csv", index=False)
    losses.to_csv(base.ANALYSIS_DIR / f"{PREFIX}_sac_training_losses_raw.csv", index=False)

    comfort_paths = {run["key"]: base._make_comfort_grid(run) for run in base.RUNS}
    safe_pi_path = base._make_safe_pi_grid(base._safe_pi_january_rows())
    district_load_path, district_daily_path = base._make_district_plots(district_frames)
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

    bad_display = pd.DataFrame()
    if not bad_delta.empty:
        bad_display = bad_delta[
            [
                "building",
                "discomfort_%_plus9",
                "discomfort_%_bad10",
                "delta_discomfort_%",
                "cold_%_bad10",
                "hot_%_bad10",
            ]
        ].rename(
            columns={
                "discomfort_%_plus9": "disc_plus9",
                "discomfort_%_bad10": "disc_bad10",
                "delta_discomfort_%": "delta_disc",
                "cold_%_bad10": "cold_bad10",
                "hot_%_bad10": "hot_bad10",
            }
        )

    meta_display = metadata_check.head(10).copy()
    reward_display = pd.DataFrame()
    if not reward_summary.empty:
        reward_display = reward_summary[["episode", "raw_reward_mean", "reward_mean", "reset_fraction", "action_mean"]].copy()
        reward_display["reset_%"] = 100.0 * reward_display.pop("reset_fraction")
        reward_display = reward_display.rename(columns={"raw_reward_mean": "raw_mean", "reward_mean": "replay_mean"})

    overview = f"""{MARKER} - `{PREFIX}`

This is the requested replication of the earlier working GNU-RL recipe with only the RLMPC model/horizon changed to 2R2C+Kalman and horizon 24.

Labels:

`{MPC_LABEL}`: existing recovered MPC 2R2C+Kalman February rollout.

`{RLMPC_LABEL}`: new GNU-RL/RLMPC run from `ce1_time_bias_guard_1c_2r2c_kalman_h24.pt` checkpoints, Kalman observer, fixed matched balanced inner cost, `power_source=heating`, `raw_power_cost`, `baseline_district_daily_mean_exp2_norm`, and `time_bias_guard_penalty=0`.

`{SAC_BASE_LABEL}` plus `{SAC_BAD_LABEL}`: latest existing heating-reward SAC artifacts; SAC was not rerun. The final SAC row uses bad+10 policies where those artifacts exist and plus9 policies elsewhere.

Cost/reward recipe:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

RLMPC inner cost:

`J_inner,k = 0.5*D_k + 0.5*GridStress_k*(p_RC2,b*u_k)/5`

RLMPC/SAC logged reward:

`reward_k = -[0.5*D_k + 0.5*GridStress_k*P_heat,real,k/5]`

`bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and `GridStress=baseline_district_daily_mean_exp2_norm`.

Numerical note: {_qpth_note()}

Metadata sanity check, first 10 buildings:

{_markdown_table(meta_display, digits=4)}

Mean building comfort KPIs:

{_markdown_table(mean_comfort, digits=3)}

Mean building operation KPIs:

{_markdown_table(mean_ops, digits=3)}

District aggregate KPIs - load shape:

{_markdown_table(district_shape, digits=3)}

District aggregate KPIs - tracking:

{_markdown_table(district_tracking, digits=3)}

Weighted reward-term diagnostics using realized heating electricity - mean:

{_markdown_table(term_mean, digits=3)}

Weighted reward-term diagnostics using realized heating electricity - max:

{_markdown_table(term_max, digits=3)}

SAC bad-building continuation deltas:

{_markdown_table(bad_display, digits=2)}
"""
    if not reward_display.empty:
        overview += f"""

SAC online reward summary, compact mean view:

{_markdown_table(reward_display, digits=3)}
"""

    image_cells = [
        base._image(comfort_paths["mpc"], "MPC comfort all buildings"),
        base._image(comfort_paths["rlmpc"], "GNU-RL RLMPC comfort all buildings"),
        base._image(comfort_paths["sac"], "SAC comfort all buildings"),
        base._image(safe_pi_path, "SAC safe PI January episode 0 comfort"),
        base._image(district_load_path, "District load time series"),
        base._image(district_daily_path, "District mean hourly profile"),
        base._image(terms_path, "Weighted reward terms"),
        base._image(response_path, "Hourly time-bias response"),
        base._image(bad_delta_path, "SAC bad-building continuation discomfort"),
        *[base._image(path, path.stem.replace("_", " ")) for path in training_paths],
    ]
    cells = [nbf.v4.new_markdown_cell(overview), *image_cells]

    for notebook in [
        base.ANALYSIS_DIR / "02_mpc_results.ipynb",
        base.ANALYSIS_DIR / "04_rlmpc_results.ipynb",
        base.ANALYSIS_DIR / "08_sac_results.ipynb",
        base.ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
        base.ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
        base.ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
    ]:
        if notebook.exists():
            base._append_once(notebook, cells)
    print("Done.")


if __name__ == "__main__":
    main()
