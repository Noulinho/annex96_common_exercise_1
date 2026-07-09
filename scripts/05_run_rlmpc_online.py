#!/usr/bin/env python
"""Online Gnu-RL/RLMPC rollout and optional dynamics adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rlmpc_pipeline import (
    OccupantComfortReward,
    RLMPCExperiment,
    run_online_rlmpc_no_occupant,
    run_online_rlmpc_occupants,
    run_online_2r2c_rlmpc_no_occupant,
)
from src.buildings import DATASET_NAME, building_metadata, format_building_path, resolve_building_entries
from src.district_target import future_stress_from_profile, load_ce1_district_target_profiles
from src.experiment_normalization import temperature_error_scale
from src.rewards import (
    BalancedComfortEnergyReward,
    SimpleComfortEnergyReward,
    daily_window_shift_profile,
    load_grid_stress_profile,
    load_simple_v4_price_profile,
    time_bias_setpoint_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="no_occupant",
        choices=["no_occupant", "occupant_present", "occupant_present_peak_flattening"],
    )
    parser.add_argument(
        "--occupant",
        default="all",
        choices=["all", "occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"],
    )
    parser.add_argument(
        "--occupant-mode",
        default="both",
        choices=["without_tdyn", "bspline_tdyn", "both"],
        help="For occupant_present: run standard occupant overrides, B-spline gated adaptive T_dyn, or both.",
    )
    parser.add_argument(
        "--tdyn-variant",
        default=None,
        choices=["feedback_only", "present_only_no_bspline", "bspline_dynamic_comfort"],
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Offline checkpoint path or template. Defaults to the per-building no_occupant checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-template",
        default="results/models/rlmpc/offline/{building_id}/{slug}.pt",
        help="Per-building offline checkpoint template. Supports {slug} plus building placeholders.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single January PI dataset path or template. Defaults to the per-building collected dataset.",
    )
    parser.add_argument(
        "--dataset-template",
        default="results/datasets/january_pi/{building_id}/january_pi_dataset.npz",
        help="Per-building January PI dataset path template.",
    )
    parser.add_argument("--simulation-start", type=int, default=744,
                        help="First time step of the deployment period (default: 744 = February).")
    parser.add_argument("--simulation-end", type=int, default=1415,
                        help="Last time step of the deployment period (default: 1415 = February).")
    parser.add_argument("--output-dir", default="results/raw/rlmpc")
    parser.add_argument("--summary-dir", default="results/summaries/rlmpc")
    parser.add_argument(
        "--rc-model",
        default="results/models/rc/1state_rc_{short_name}_january.json",
        help="RC model path/template used to match the explicit MPC power term.",
    )
    parser.add_argument(
        "--building",
        default="all",
        help="Building selector: all, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument(
        "--occupancy-building",
        default="same",
        help="B-spline occupancy source. Use 'same' for the controlled building, or any building selector.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--lr-ppo", type=float, default=5e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--reward", default="occupant_comfort", choices=["comfort", "occupant_comfort", "balanced", "simple_v4"])
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--comfort-weight", type=float, default=0.5)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--peak-weight", type=float, default=3.0)
    parser.add_argument("--ramp-weight", type=float, default=1.0)
    parser.add_argument("--shift-weight", type=float, default=0.0)
    parser.add_argument("--preheat-weight", type=float, default=0.0)
    parser.add_argument("--preheat-horizon", type=int, default=6)
    parser.add_argument("--comfort-band", type=float, default=2.0)
    parser.add_argument(
        "--single-setpoint",
        action="store_true",
        help="Track the heating setpoint as the single comfort reference.",
    )
    parser.add_argument(
        "--comfort-mode",
        default="band",
        choices=["band", "setpoint", "asymmetric", "barrier", "time_bias_guard"],
        help="Balanced reward/MPC comfort term: soft band violation, exact setpoint tracking, or asymmetric setpoint tradeoff.",
    )
    parser.add_argument("--comfort-track-weight", type=float, default=3.0)
    parser.add_argument("--comfort-cold-weight", type=float, default=60.0)
    parser.add_argument("--comfort-hot-weight", type=float, default=15.0)
    parser.add_argument(
        "--adaptive-balanced-weights",
        action="store_true",
        help=(
            "For matched balanced no-occupant RLMPC, train softmax comfort weights "
            "(track/cold/hot) with PPO instead of freezing the exact MPC weights."
        ),
    )
    parser.add_argument(
        "--adaptive-dynamics",
        action="store_true",
        help=(
            "Keep the objective fixed and update the differentiable RC dynamics "
            "parameters online through the PPO policy objective, Gnu-RL style."
        ),
    )
    parser.add_argument("--dynamics-epochs", type=int, default=5)
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-minibatch", type=int, default=64)
    parser.add_argument("--dynamics-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dynamics-validation-fraction", type=float, default=0.2)
    parser.add_argument("--dynamics-patience", type=int, default=3)
    parser.add_argument("--comfort-cold-tolerance", type=float, default=0.5)
    parser.add_argument("--comfort-hot-tolerance", type=float, default=1.0)
    parser.add_argument("--comfort-track-scale", type=float, default=2.0)
    parser.add_argument("--comfort-cold-scale", type=float, default=1.5)
    parser.add_argument("--comfort-hot-scale", type=float, default=2.0)
    parser.add_argument("--comfort-cold-k", type=float, default=3.0,
                        help="Exponential growth rate for cold barrier (barrier mode only).")
    parser.add_argument("--comfort-hot-k", type=float, default=1.0,
                        help="Exponential growth rate for hot barrier (barrier mode only).")
    parser.add_argument(
        "--time-bias-amplitude",
        type=float,
        default=1.0,
        help="Setpoint bias magnitude in degC for --comfort-mode time_bias_guard.",
    )
    parser.add_argument(
        "--time-bias-guard-band",
        type=float,
        default=2.0,
        help="Comfort guard around the true setpoint for --comfort-mode time_bias_guard.",
    )
    parser.add_argument(
        "--time-bias-guard-penalty",
        type=float,
        default=0.0,
        help="Linear soft-constraint penalty on DiffMPC time-bias guard slack.",
    )
    parser.add_argument(
        "--comfort-scale",
        type=float,
        default=None,
        help="Scale for squared comfort error. Defaults to comfort_band**2 for band mode or sigma_T**2 for setpoint mode.",
    )
    parser.add_argument(
        "--temperature-scale-mode",
        default="setpoint_std",
        choices=[
            "error_std",
            "error_iqr",
            "error_range",
            "error_minmax",
            "setpoint_std",
            "setpoint_iqr",
            "setpoint_range",
            "setpoint_minmax",
            "indoor_std",
            "indoor_iqr",
            "indoor_range",
            "indoor_minmax",
            "temperature_minmax",
            "dataset_minmax",
            "full_dataset_minmax",
            "observation_minmax",
            "observation_space_minmax",
            "obs_minmax",
        ],
        help="How to normalize setpoint error when --comfort-mode setpoint and --comfort-scale is omitted.",
    )
    parser.add_argument("--power-scale", type=float, default=5.0)
    parser.add_argument("--power-source", default="heating", choices=["net", "heating", "action_proxy"])
    parser.add_argument(
        "--grid-stress-rollout",
        default=None,
        help="Optional rollout CSV used to build a pseudo-price/grid-stress load-shifting signal.",
    )
    parser.add_argument("--grid-stress-column", default="net_electricity_consumption")
    parser.add_argument(
        "--grid-stress-mode",
        default="normalized",
        choices=["normalized", "percentile", "daily_top_k"],
    )
    parser.add_argument("--grid-stress-percentile", type=float, default=80.0)
    parser.add_argument("--grid-stress-daily-top-k", type=int, default=4)
    parser.add_argument(
        "--load-shape-profile",
        default="ce1_stress",
        choices=["ce1_stress", "daily_windows", "baseline_daily_mean_exp"],
        help="Source for the shift term when no grid-stress rollout is supplied.",
    )
    parser.add_argument(
        "--district-target-weight",
        type=float,
        default=0.0,
        help="Weight on CE1 district-load target tracking. Uses the real district_target.csv.",
    )
    parser.add_argument(
        "--district-target-scale",
        type=float,
        default=None,
        help="Normalization for district target error. Defaults to max absolute target value.",
    )
    parser.add_argument(
        "--district-target-scale-mode",
        default="target_max",
        choices=[
            "target_max",
            "baseline_error_std",
            "baseline_error_iqr",
            "baseline_error_range",
            "baseline_error_maxabs",
            "controllable_error_minmax",
            "controllable_error_maxabs",
        ],
        help="Scale CE1 target error by target magnitude or baseline-vs-target error statistics.",
    )
    parser.add_argument(
        "--district-target-mode",
        default="tracking",
        choices=["tracking", "peak_only"],
        help="District target mode for reward and RLMPC inner objective.",
    )
    parser.add_argument(
        "--raw-power-cost",
        action="store_true",
        help="Use RC heat power / power_scale in the RLMPC cost instead of normalized action u.",
    )
    parser.add_argument("--alpha-occ", type=float, default=0.5)
    parser.add_argument("--tdyn-init-mode", default="schedule", choices=["schedule", "pref"])
    parser.add_argument("--delta-up", type=float, default=0.5)
    parser.add_argument("--delta-down", type=float, default=0.5)
    parser.add_argument("--drift-to-pref", type=float, default=0.05)
    parser.add_argument("--tdyn-min", type=float, default=18.0)
    parser.add_argument("--tdyn-max", type=float, default=26.0)
    parser.add_argument(
        "--tsp-lower-bias",
        type=float,
        default=0.5,
        help="simple_v4 reward: shift tracking target Tsp - bias (°C).",
    )
    parser.add_argument(
        "--rc-model-type",
        default="1r1c",
        choices=["1r1c", "2r2c"],
        help="Which offline checkpoint type to load: 1-state (default) or 2R2C.",
    )
    parser.add_argument(
        "--tm-penalty-weight",
        type=float,
        default=0.1,
        help="2R2C only: penalty weight on (Tm - Tsp)² relative to the Tin weight. "
             "Prevents cold-mass drag on Tin. 0 = disabled (old behaviour).",
    )
    parser.add_argument(
        "--two-state-observer",
        default="propagate",
        choices=["propagate", "kalman"],
        help="2R2C only: latent Tm observer used during online rollout.",
    )
    parser.add_argument("--kalman-process-var", type=float, default=0.05)
    parser.add_argument("--kalman-measurement-var", type=float, default=0.25)
    parser.add_argument("--kalman-initial-var", type=float, default=4.0)
    return parser


def _has_format_fields(path: str | Path) -> bool:
    text = str(path)
    return "{" in text and "}" in text


def _format_path(template: str | Path, building: dict, experiment: RLMPCExperiment) -> Path:
    values = dict(building)
    values.update(
        {
            "building_id": building["result_id"],
            "building_name": building["name"],
            "building": building["name"],
            "slug": experiment.result_slug,
        }
    )
    return Path(str(template).format(**values))


def _resolve_path(
    *,
    explicit: str | Path | None,
    template: str | Path,
    building: dict,
    experiment: RLMPCExperiment,
    multiple_buildings: bool,
    label: str,
) -> Path:
    value = template if explicit is None else explicit
    if multiple_buildings and explicit is not None and not _has_format_fields(explicit):
        raise ValueError(
            f"--{label} points to one fixed path while --building selects multiple buildings. "
            f"Use --{label}-template or include a placeholder such as {{building_id}}."
        )
    return _format_path(value, building, experiment)


def main() -> None:
    args = build_parser().parse_args()
    import pandas as pd

    # Online occupant cases all start from the same offline imitation checkpoint.
    experiment = RLMPCExperiment(controller="rlmpc", scenario="no_occupant")
    rc_model_type = getattr(args, "rc_model_type", "1r1c")

    # Build the exp-normalized district-baseline daily load profile used as a pseudo-price.
    _baseline_daily_mean_exp_profile = None
    if args.reward == "simple_v4" or args.load_shape_profile == "baseline_daily_mean_exp":
        _baseline_daily_mean_exp_profile = load_simple_v4_price_profile(simulation_start=744, exp_beta=2.0)
        if not _baseline_daily_mean_exp_profile:
            print("WARNING: district baseline not found; GridStress = 1.0 (flat)")

    if args.reward == "balanced":
        reward_function = BalancedComfortEnergyReward
        reward_function_kwargs = None
    elif args.reward == "simple_v4":
        reward_function = SimpleComfortEnergyReward
        reward_function_kwargs = None  # filled per-building with shift_profile below
    elif args.reward == "occupant_comfort":
        reward_function = OccupantComfortReward
        reward_function_kwargs = None
    else:
        reward_function = None
        reward_function_kwargs = None
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    multiple = len(buildings) > 1
    all_summary_rows = []

    for building in buildings:
        if rc_model_type == "2r2c":
            slug_2r2c = experiment.result_slug + "_2r2c"
            if args.checkpoint is not None:
                checkpoint = Path(args.checkpoint)
            else:
                checkpoint = Path(
                    args.checkpoint_template.format(
                        building_id=building["result_id"],
                        building_name=building["name"],
                        building=building["name"],
                        slug=slug_2r2c,
                    )
                )
        else:
            checkpoint = _resolve_path(
                explicit=args.checkpoint,
                template=args.checkpoint_template,
                building=building,
                experiment=experiment,
                multiple_buildings=multiple,
                label="checkpoint",
            )
        dataset_path = _resolve_path(
            explicit=args.dataset,
            template=args.dataset_template,
            building=building,
            experiment=experiment,
            multiple_buildings=multiple,
            label="dataset",
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing offline RL-MPC checkpoint for {building['display']}: {checkpoint}. "
                "Run scripts/04_train_rlmpc_offline.py first."
            )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing January PI dataset for {building['display']}: {dataset_path}. "
                "Run scripts/00_collect_january_pi_data.py first."
            )

        building_name = str(building["name"])
        if args.occupancy_building.lower() == "same":
            occupancy_building_name = building_name
        else:
            occupancy_entries = resolve_building_entries(args.occupancy_building, DATASET_NAME)
            if len(occupancy_entries) != 1:
                raise ValueError("--occupancy-building must resolve to one building, not 'all'.")
            occupancy_building_name = str(occupancy_entries[0]["name"])
        metadata = building_metadata(building, occupancy_building_name=occupancy_building_name)
        setpoint_bias_profile = None
        if args.reward == "balanced":
            if args.comfort_mode in {"asymmetric", "time_bias_guard"}:
                comfort_scale = 1.0 if args.comfort_scale is None else float(args.comfort_scale)
                temperature_scale = float(args.comfort_track_scale)
            elif args.comfort_scale is None and args.comfort_mode == "setpoint":
                temperature_scale = temperature_error_scale(
                    building_name=building_name,
                    simulation_start=0,
                    simulation_end=1415 if "minmax" in args.temperature_scale_mode else 743,
                    dataset_name=DATASET_NAME,
                    mode=args.temperature_scale_mode,
                )
                comfort_scale = temperature_scale**2
            else:
                comfort_scale = (
                    float(args.comfort_band) ** 2
                    if args.comfort_scale is None
                    else float(args.comfort_scale)
                )
                temperature_scale = float(comfort_scale**0.5)
            metadata["comfort_mode"] = args.comfort_mode
            metadata["temperature_scale_mode"] = args.temperature_scale_mode
            metadata["temperature_error_scale"] = float(temperature_scale)
            metadata["comfort_scale"] = float(comfort_scale)
            metadata["comfort_track_weight"] = float(args.comfort_track_weight)
            metadata["comfort_cold_weight"] = float(args.comfort_cold_weight)
            metadata["comfort_hot_weight"] = float(args.comfort_hot_weight)
            metadata["comfort_cold_tolerance"] = float(args.comfort_cold_tolerance)
            metadata["comfort_hot_tolerance"] = float(args.comfort_hot_tolerance)
            metadata["comfort_track_scale"] = float(args.comfort_track_scale)
            metadata["comfort_cold_scale"] = float(args.comfort_cold_scale)
            metadata["comfort_hot_scale"] = float(args.comfort_hot_scale)
            metadata["comfort_cold_k"] = float(args.comfort_cold_k)
            metadata["comfort_hot_k"] = float(args.comfort_hot_k)
            metadata["single_setpoint"] = bool(args.single_setpoint)
            metadata["reward_power_source"] = str(args.power_source)
            metadata["reward_power_scale"] = float(args.power_scale)
            if args.comfort_mode == "time_bias_guard":
                setpoint_bias_profile = time_bias_setpoint_profile(
                    simulation_start=args.simulation_start,
                    simulation_end=args.simulation_end,
                    amplitude=args.time_bias_amplitude,
                )
                metadata["time_bias_amplitude"] = float(args.time_bias_amplitude)
                metadata["time_bias_guard_band"] = float(args.time_bias_guard_band)
                metadata["setpoint_bias_profile_mean"] = float(
                    sum(setpoint_bias_profile) / max(len(setpoint_bias_profile), 1)
                )
            reward_function_kwargs = dict(
                comfort_weight=args.comfort_weight,
                energy_weight=args.energy_weight,
                peak_weight=args.peak_weight,
                ramp_weight=args.ramp_weight,
                shift_weight=args.shift_weight,
                preheat_weight=args.preheat_weight,
                comfort_band=args.comfort_band,
                comfort_scale=comfort_scale,
                comfort_mode=args.comfort_mode,
                comfort_track_weight=args.comfort_track_weight,
                comfort_cold_weight=args.comfort_cold_weight,
                comfort_hot_weight=args.comfort_hot_weight,
                comfort_cold_tolerance=args.comfort_cold_tolerance,
                comfort_hot_tolerance=args.comfort_hot_tolerance,
                comfort_track_scale=args.comfort_track_scale,
                comfort_cold_scale=args.comfort_cold_scale,
                comfort_hot_scale=args.comfort_hot_scale,
                comfort_cold_k=args.comfort_cold_k,
                comfort_hot_k=args.comfort_hot_k,
                single_setpoint=args.single_setpoint,
                power_scale=args.power_scale,
                power_source=args.power_source,
                setpoint_bias_profile=setpoint_bias_profile,
                time_bias_guard_band=args.time_bias_guard_band,
            )
        elif args.reward == "simple_v4":
            reward_function_kwargs = dict(
                comfort_weight=args.comfort_weight,
                energy_weight=args.energy_weight,
                comfort_band=args.comfort_band,
                tsp_lower_bias=args.tsp_lower_bias,
                power_scale=args.power_scale,
                shift_profile=_baseline_daily_mean_exp_profile or None,
            )
            metadata["simple_v4_tsp_lower_bias"] = float(args.tsp_lower_bias)
            metadata["simple_price_profile"] = "baseline_district_daily_mean_exp2_norm"
            if _baseline_daily_mean_exp_profile:
                metadata["simple_price_profile_min"] = float(min(_baseline_daily_mean_exp_profile))
                metadata["simple_price_profile_max"] = float(max(_baseline_daily_mean_exp_profile))
        else:
            reward_function_kwargs = None
        grid_stress_profile = _baseline_daily_mean_exp_profile if args.reward == "simple_v4" else None
        future_grid_stress_profile = None
        if args.grid_stress_rollout is not None:
            grid_stress_path = format_building_path(args.grid_stress_rollout, building)
            grid_stress_profile = load_grid_stress_profile(
                grid_stress_path,
                column=args.grid_stress_column,
                mode=args.grid_stress_mode,
                percentile=args.grid_stress_percentile,
                daily_top_k=args.grid_stress_daily_top_k,
            )
            metadata["grid_stress_rollout"] = str(grid_stress_path)
            metadata["grid_stress_mode"] = args.grid_stress_mode
            metadata["grid_stress_profile_mean"] = float(sum(grid_stress_profile) / max(len(grid_stress_profile), 1))
            if reward_function_kwargs is not None:
                reward_function_kwargs = dict(reward_function_kwargs)
                reward_function_kwargs["shift_profile"] = grid_stress_profile
        elif float(args.shift_weight) != 0.0 and args.load_shape_profile == "daily_windows":
            grid_stress_profile = daily_window_shift_profile(
                simulation_start=744,
                simulation_end=1415,
            )
            metadata["grid_stress_mode"] = "daily_windows_signed"
            metadata["grid_stress_profile_mean"] = float(sum(grid_stress_profile) / max(len(grid_stress_profile), 1))
            metadata["grid_stress_profile_min"] = float(min(grid_stress_profile)) if grid_stress_profile else 0.0
            metadata["grid_stress_profile_max"] = float(max(grid_stress_profile)) if grid_stress_profile else 0.0
            if reward_function_kwargs is not None:
                reward_function_kwargs = dict(reward_function_kwargs)
                reward_function_kwargs["shift_profile"] = grid_stress_profile
        elif float(args.shift_weight) != 0.0 and args.load_shape_profile == "baseline_daily_mean_exp":
            grid_stress_profile = _baseline_daily_mean_exp_profile or None
            metadata["grid_stress_mode"] = "baseline_district_daily_mean_exp2_norm"
            metadata["grid_stress_profile_mean"] = float(sum(grid_stress_profile) / max(len(grid_stress_profile), 1)) if grid_stress_profile else 0.0
            metadata["grid_stress_profile_min"] = float(min(grid_stress_profile)) if grid_stress_profile else 0.0
            metadata["grid_stress_profile_max"] = float(max(grid_stress_profile)) if grid_stress_profile else 0.0
            if reward_function_kwargs is not None:
                reward_function_kwargs = dict(reward_function_kwargs)
                reward_function_kwargs["shift_profile"] = grid_stress_profile
        district_profiles = None
        needs_district_profiles = (
            float(args.district_target_weight) != 0.0
            or (
                (float(args.shift_weight) != 0.0 or float(args.preheat_weight) != 0.0)
                and grid_stress_profile is None
            )
        )
        if needs_district_profiles:
            district_profiles = load_ce1_district_target_profiles(
                building_name=building_name,
                simulation_start=744,
                simulation_end=1415,
                dataset_name=DATASET_NAME,
                target_scale=args.district_target_scale,
                target_scale_mode=args.district_target_scale_mode,
            )
            metadata["district_target_weight"] = float(args.district_target_weight)
            metadata["district_target_mode"] = args.district_target_mode
            metadata["district_target_scale"] = float(district_profiles.target_scale)
            metadata["district_target_scale_mode"] = str(district_profiles.target_scale_mode)
            metadata["district_target_mean"] = float(sum(district_profiles.target) / max(len(district_profiles.target), 1))
            metadata["district_background_mean"] = float(
                sum(district_profiles.background_without_controlled_heating)
                / max(len(district_profiles.background_without_controlled_heating), 1)
            )
            metadata["controlled_heating_nominal_power"] = float(district_profiles.heating_nominal_power)
            if grid_stress_profile is None and (
                float(args.shift_weight) != 0.0 or float(args.preheat_weight) != 0.0
            ):
                grid_stress_profile = district_profiles.district_stress_profile()
                metadata["grid_stress_mode"] = "ce1_baseline_above_target"
                metadata["grid_stress_profile_mean"] = float(
                    sum(grid_stress_profile) / max(len(grid_stress_profile), 1)
                )
                if reward_function_kwargs is not None:
                    reward_function_kwargs = dict(reward_function_kwargs)
                    reward_function_kwargs["shift_profile"] = grid_stress_profile
            if float(args.preheat_weight) != 0.0:
                future_grid_stress_profile = district_profiles.future_stress_profile(args.preheat_horizon)
                metadata["preheat_weight"] = float(args.preheat_weight)
                metadata["preheat_horizon"] = int(args.preheat_horizon)
                metadata["future_grid_stress_profile_mean"] = float(
                    sum(future_grid_stress_profile) / max(len(future_grid_stress_profile), 1)
                )
                if reward_function_kwargs is not None:
                    reward_function_kwargs = dict(reward_function_kwargs)
                    reward_function_kwargs["future_shift_profile"] = future_grid_stress_profile
            if reward_function_kwargs is not None:
                reward_function_kwargs = dict(reward_function_kwargs)
                reward_function_kwargs.update(
                    district_profiles.as_reward_kwargs(
                        target_weight=args.district_target_weight,
                        target_mode=args.district_target_mode,
                    )
                )
        if (
            float(args.preheat_weight) != 0.0
            and future_grid_stress_profile is None
            and grid_stress_profile is not None
        ):
            future_grid_stress_profile = future_stress_from_profile(
                grid_stress_profile,
                horizon=args.preheat_horizon,
            )
            metadata["preheat_weight"] = float(args.preheat_weight)
            metadata["preheat_horizon"] = int(args.preheat_horizon)
            metadata["future_grid_stress_profile_mean"] = float(
                sum(future_grid_stress_profile) / max(len(future_grid_stress_profile), 1)
            )
            if reward_function_kwargs is not None:
                reward_function_kwargs = dict(reward_function_kwargs)
                reward_function_kwargs["future_shift_profile"] = future_grid_stress_profile
        matched_mpc_cost = None
        if args.reward == "balanced":
            rc_model_path = format_building_path(args.rc_model, building)
            if not rc_model_path.exists():
                raise FileNotFoundError(f"Missing RC model for matched RLMPC cost: {rc_model_path}")
            rc_params = json.loads(rc_model_path.read_text())
            matched_mpc_cost = {
                "comfort_weight": args.comfort_weight,
                "energy_weight": args.energy_weight,
                "peak_weight": args.peak_weight,
                "ramp_weight": args.ramp_weight,
                "shift_weight": args.shift_weight,
                "preheat_weight": args.preheat_weight,
                "district_target_weight": args.district_target_weight,
                "district_target_scale": 1.0 if district_profiles is None else district_profiles.target_scale,
                "district_target_mode": args.district_target_mode,
                "comfort_band": args.comfort_band,
                "comfort_scale": comfort_scale,
                "comfort_mode": args.comfort_mode,
                "comfort_track_weight": args.comfort_track_weight,
                "comfort_cold_weight": args.comfort_cold_weight,
                "comfort_hot_weight": args.comfort_hot_weight,
                "adaptive_weights": bool(args.adaptive_balanced_weights),
                "comfort_cold_tolerance": args.comfort_cold_tolerance,
                "comfort_hot_tolerance": args.comfort_hot_tolerance,
                "comfort_track_scale": args.comfort_track_scale,
                "comfort_cold_scale": args.comfort_cold_scale,
                "comfort_hot_scale": args.comfort_hot_scale,
                "time_bias_guard_band": args.time_bias_guard_band,
                "time_bias_guard_penalty": args.time_bias_guard_penalty,
                "power_scale": args.power_scale,
                "power_per_unit_action": float(rc_params["power_per_unit_action"]),
                "normalize_terms": not args.raw_power_cost,
            }
            metadata["rlmpc_inner_cost"] = "matched_balanced_mpc"
            metadata["rlmpc_adaptive_balanced_weights"] = bool(args.adaptive_balanced_weights)
            metadata["rlmpc_matched_rc_model"] = str(rc_model_path)
            if args.power_source == "action_proxy" and reward_function_kwargs is not None:
                reward_function_kwargs = dict(reward_function_kwargs)
                reward_function_kwargs["action_power_coefficients"] = [float(rc_params["power_per_unit_action"])]
                metadata["reward_action_proxy_rc_model"] = str(rc_model_path)
                metadata["reward_action_proxy_power_per_unit_action"] = float(rc_params["power_per_unit_action"])
        output_dir = Path(args.output_dir) / building["result_id"]
        summary_dir = Path(args.summary_dir) / building["result_id"]
        seed = int(args.seed + 1000 * (int(building["index"]) - 1))
        training_seed = int(args.training_seed + 1000 * (int(building["index"]) - 1))

        print(f"\n=== {building['display']}: {building_name} ===")
        print(f"Offline checkpoint: {checkpoint}")
        print(f"Baseline dataset: {dataset_path}")
        if args.scenario == "occupant_present":
            print(f"Occupancy model source: {occupancy_building_name}")

        if args.scenario == "no_occupant":
            if rc_model_type == "2r2c":
                case = run_online_2r2c_rlmpc_no_occupant(
                    checkpoint_path=checkpoint,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    summary_dir=summary_dir,
                    seed=seed,
                    training_seed=training_seed,
                    reward_function=reward_function,
                    reward_function_kwargs=reward_function_kwargs,
                    device=args.device,
                    sigma=args.sigma,
                    clip_eps=args.clip_eps,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch,
                    lr_ppo=args.lr_ppo,
                    gamma=args.gamma,
                    max_steps=args.max_steps,
                    label=args.label,
                    building_name=building_name,
                    building_metadata_dict=metadata,
                    stress_profile=grid_stress_profile,
                    setpoint_bias_profile=setpoint_bias_profile,
                    matched_mpc_cost=matched_mpc_cost,
                    tm_penalty_weight=args.tm_penalty_weight,
                    two_state_observer=args.two_state_observer,
                    kalman_process_var=args.kalman_process_var,
                    kalman_measurement_var=args.kalman_measurement_var,
                    kalman_initial_var=args.kalman_initial_var,
                    sim_start=args.simulation_start,
                    sim_end=args.simulation_end,
                )
            else:
                case = run_online_rlmpc_no_occupant(
                    checkpoint_path=checkpoint,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    summary_dir=summary_dir,
                    seed=seed,
                    training_seed=training_seed,
                    reward_function=reward_function,
                    reward_function_kwargs=reward_function_kwargs,
                    device=args.device,
                    sigma=args.sigma,
                    clip_eps=args.clip_eps,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.minibatch,
                    lr_ppo=args.lr_ppo,
                    gamma=args.gamma,
                    adaptive_dynamics=args.adaptive_dynamics,
                    dynamics_epochs=args.dynamics_epochs,
                    dynamics_lr=args.dynamics_lr,
                    dynamics_minibatch_size=args.dynamics_minibatch,
                    dynamics_max_grad_norm=args.dynamics_max_grad_norm,
                    dynamics_validation_fraction=args.dynamics_validation_fraction,
                    dynamics_patience=args.dynamics_patience,
                    max_steps=args.max_steps,
                    sim_start=args.simulation_start,
                    sim_end=args.simulation_end,
                    label=args.label,
                    building_name=building_name,
                    building_metadata=metadata,
                    matched_mpc_cost=matched_mpc_cost,
                    stress_profile=grid_stress_profile,
                    future_stress_profile=future_grid_stress_profile,
                    setpoint_bias_profile=setpoint_bias_profile,
                    district_target_profile=None if district_profiles is None else district_profiles.target,
                    district_background_profile=(
                        None if district_profiles is None else district_profiles.background_without_controlled_heating
                    ),
                    district_action_power_profile=None if district_profiles is None else district_profiles.heating_action_power,
                )
            all_summary_rows.append(case["metrics"])
        elif args.scenario == "occupant_present":
            cases = run_online_rlmpc_occupants(
                checkpoint_path=checkpoint,
                dataset_path=dataset_path,
                output_dir=output_dir,
                summary_dir=summary_dir,
                seed=seed,
                training_seed=training_seed,
                reward_function=reward_function,
                reward_function_kwargs=reward_function_kwargs,
                device=args.device,
                sigma=args.sigma,
                clip_eps=args.clip_eps,
                ppo_epochs=args.ppo_epochs,
                minibatch_size=args.minibatch,
                lr_ppo=args.lr_ppo,
                gamma=args.gamma,
                adaptive_dynamics=args.adaptive_dynamics,
                dynamics_epochs=args.dynamics_epochs,
                dynamics_lr=args.dynamics_lr,
                dynamics_minibatch_size=args.dynamics_minibatch,
                dynamics_max_grad_norm=args.dynamics_max_grad_norm,
                dynamics_validation_fraction=args.dynamics_validation_fraction,
                dynamics_patience=args.dynamics_patience,
                max_steps=args.max_steps,
                label_prefix=args.label,
                occupant=args.occupant,
                occupant_mode=args.occupant_mode,
                alpha_occ=args.alpha_occ,
                T_dyn_init_mode=args.tdyn_init_mode,
                delta_up=args.delta_up,
                delta_down=args.delta_down,
                drift_to_pref=args.drift_to_pref,
                T_dyn_min=args.tdyn_min,
                T_dyn_max=args.tdyn_max,
                sim_start=args.simulation_start,
                sim_end=args.simulation_end,
                building_name=building_name,
                occupancy_building_name=occupancy_building_name,
                building_metadata=metadata,
                matched_mpc_cost=matched_mpc_cost,
                stress_profile=grid_stress_profile,
                future_stress_profile=future_grid_stress_profile,
                setpoint_bias_profile=setpoint_bias_profile,
                district_target_profile=None if district_profiles is None else district_profiles.target,
                district_background_profile=(
                    None if district_profiles is None else district_profiles.background_without_controlled_heating
                ),
                district_action_power_profile=None if district_profiles is None else district_profiles.heating_action_power,
            )
            all_summary_rows.extend(case["metrics"] for case in cases.values())
        else:
            raise SystemExit("occupant_present_peak_flattening is not wired yet.")

    if len(buildings) > 1:
        combined = pd.DataFrame(all_summary_rows)
        combined_path = Path(args.summary_dir) / f"{args.scenario}_summary_all_buildings.csv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"\nSaved combined summary to {combined_path}")


if __name__ == "__main__":
    main()
