#!/usr/bin/env python
"""Run MERLIN-style SAC training and February evaluation.

For every case this follows the MERLIN deployment-strategy shape:

- First January episode: PITemperatureController acts and fills replay.
- Remaining January episodes: SAC acts and learns with CityLearn SAC updates.
- February: deterministic trained-policy evaluation.
- Occupant runs append one-step B-spline occupancy forecasts to SAC observations.
- Occupant runs can be repeated with ComfortReward, DynamicComfortReward,
  OccupantFeedbackReward, and FixedComfortReward.
- Results: February deterministic trained-policy rollout, saved with the same
  `rollout.csv`, `kpis.csv`, `district_kpis.csv`, and summary JSON structure as
  the MPC and GNU-RL pipelines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import DATASET_NAME, building_metadata, format_building_path, resolve_building_entries
from src.district_target import future_stress_from_profile, load_ce1_district_target_profiles
from src.rewards import (
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
        default="all",
        choices=["without_tdyn", "bspline_tdyn", "both", "bspline_observation", "all"],
        help="Run legacy T_dyn modes, the B-spline observation reward matrix, or all supported SAC occupant cases.",
    )
    parser.add_argument("--output-dir", default="results/raw/sac_merlin")
    parser.add_argument("--summary-dir", default="results/summaries/sac_merlin")
    parser.add_argument("--model-dir", default="results/models/sac_merlin")
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
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="Short smoke-test limit.")
    parser.add_argument(
        "--reward",
        default="all",
        choices=[
            "comfort",
            "dynamic_comfort",
            "feedback",
            "fixed_comfort",
            "occupant_comfort",
            "occupant_prob",
            "balanced",
            "lagrangian",
            "simple_v4",
            "both",
            "all",
        ],
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument(
        "--automatic-entropy-tuning",
        action="store_true",
        help="Learn SAC entropy coefficient alpha instead of keeping --alpha fixed.",
    )
    parser.add_argument(
        "--entropy-lr",
        type=float,
        default=None,
        help="Learning rate for automatic entropy tuning. Defaults to --lr.",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=None,
        help="Target entropy for automatic entropy tuning. Defaults to -action_dim.",
    )
    parser.add_argument("--robust-normalization", action="store_true")
    parser.add_argument("--robust-normalization-low", type=float, default=5.0)
    parser.add_argument("--robust-normalization-high", type=float, default=95.0)
    parser.add_argument("--normalized-observation-clip", type=float, default=None)
    parser.add_argument("--normalized-reward-clip", type=float, default=None)
    parser.add_argument(
        "--observation-filter",
        default=None,
        choices=[None, "none", "mpc_core"],
        help="Optional SAC observation filter. 'mpc_core' keeps only thermal/weather/reference features.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for SAC actor/critic updates.",
    )
    parser.add_argument(
        "--action-scaling-coefficient",
        type=float,
        default=0.5,
        help="Scale applied inside CityLearn SAC actor. 0.5 caps learned actions at half the action range; 1.0 allows the full action range.",
    )
    parser.add_argument(
        "--jumpstart-controller",
        default="pi",
        choices=["pi", "safe_pi", "mpc"],
        help="Controller used for the first January replay-filling episode.",
    )
    parser.add_argument(
        "--bc-teacher-controller",
        default="none",
        choices=["none", "mpc"],
        help="Offline teacher used for behavior-cloning labels before February deployment.",
    )
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument("--bc-validation-fraction", type=float, default=0.2)
    parser.add_argument("--bc-early-stopping-patience", type=int, default=5)
    parser.add_argument("--bc-min-delta", type=float, default=1e-6)
    parser.add_argument("--mpc-rc-model", default="results/models/rc/1state_rc_{short_name}_january.json")
    parser.add_argument(
        "--mpc-rc-model-type",
        default="1r1c",
        choices=["1r1c", "2r2c"],
        help="RC model used by the MPC jump-start teacher.",
    )
    parser.add_argument(
        "--mpc-rc-model-2r2c",
        default="results/models/rc/2r2c_{short_name}_january.json",
        help="2R2C model path template used by --jumpstart-controller mpc and --mpc-rc-model-type 2r2c.",
    )
    parser.add_argument(
        "--mpc-two-state-observer",
        default="propagate",
        choices=["propagate", "kalman"],
        help="2R2C MPC teacher observer used for jump-start and behavior-cloning labels.",
    )
    parser.add_argument("--mpc-kalman-process-var", type=float, default=0.05)
    parser.add_argument("--mpc-kalman-measurement-var", type=float, default=0.25)
    parser.add_argument("--mpc-kalman-initial-var", type=float, default=4.0)
    parser.add_argument("--training-episodes", type=int, default=10)
    parser.add_argument(
        "--resume-agent-template",
        default=None,
        help=(
            "Optional saved SAC agent path template to resume from. Supports "
            "{building_id}, {short_name}, and {label}."
        ),
    )
    parser.add_argument(
        "--resume-episode-offset",
        type=int,
        default=0,
        help="Episode number offset used when logging resumed online training episodes.",
    )
    parser.add_argument(
        "--resume-replay-keep-last",
        type=int,
        default=None,
        help=(
            "When resuming a saved SAC agent, keep only the last N replay transitions "
            "from the saved buffer before collecting new online data. Use 0 for a hard "
            "month-boundary replay reset."
        ),
    )
    parser.add_argument(
        "--early-stop-reward-slope",
        action="store_true",
        help="Stop SAC training when recent per-step reward has plateaued.",
    )
    parser.add_argument("--early-stop-min-episodes", type=int, default=10)
    parser.add_argument("--early-stop-window", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument(
        "--early-stop-slope-abs",
        type=float,
        default=0.25,
        help="Absolute recent reward-per-step slope threshold for plateau detection.",
    )
    parser.add_argument("--n-step-horizon", type=int, default=1,
                        help="N-step return horizon for Q-target. 1 = standard 1-step Bellman.")
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
        help="Balanced reward comfort term.",
    )
    parser.add_argument("--comfort-track-weight", type=float, default=3.0)
    parser.add_argument("--comfort-cold-weight", type=float, default=60.0)
    parser.add_argument("--comfort-hot-weight", type=float, default=15.0)
    parser.add_argument("--comfort-cold-tolerance", type=float, default=0.5)
    parser.add_argument("--comfort-hot-tolerance", type=float, default=1.0)
    parser.add_argument("--comfort-track-scale", type=float, default=2.0)
    parser.add_argument("--comfort-cold-scale", type=float, default=1.5)
    parser.add_argument("--comfort-hot-scale", type=float, default=2.0)
    parser.add_argument(
        "--comfort-guard-exponent",
        type=float,
        default=2.0,
        help="Power applied to asymmetric/time-bias guard violations after normalization.",
    )
    parser.add_argument(
        "--occupant-probability-base",
        type=float,
        default=0.002,
        help="Base probability in the expected occupant override reward.",
    )
    parser.add_argument(
        "--no-occupant-probability-clip",
        action="store_true",
        help="Do not clip expected override probabilities at the fitted occupant max_prob.",
    )
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
        "--comfort-scale",
        type=float,
        default=None,
        help="Scale for squared comfort-band violation. Defaults to comfort_band**2.",
    )
    parser.add_argument("--power-scale", type=float, default=5.0)
    parser.add_argument("--power-source", default="heating", choices=["net", "heating", "action_proxy"])
    parser.add_argument(
        "--power-scale-mode",
        default="fixed",
        choices=["fixed", "nominal_heating"],
        help="Use a fixed power scale or each building's nominal heating power for SAC reward normalization.",
    )
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
        help="Scale CE1 target error by target magnitude or controllable error statistics.",
    )
    parser.add_argument(
        "--district-target-mode",
        default="tracking",
        choices=["tracking", "peak_only"],
        help="Reward target mode.",
    )
    parser.add_argument(
        "--heat-budget-weight",
        type=float,
        default=0.0,
        help="Weight on rolling heat-budget floor against the baseline/PI heating profile.",
    )
    parser.add_argument(
        "--heat-budget-window",
        type=int,
        default=24,
        help="Rolling window length for the heat-budget floor.",
    )
    parser.add_argument(
        "--heat-budget-tolerance",
        type=float,
        default=0.05,
        help="Allowed fractional missing heat before the heat-budget floor is penalized.",
    )
    parser.add_argument(
        "--heat-budget-temperature-gate-margin",
        type=float,
        default=None,
        help="If set, apply heat-budget floor only when indoor temperature is at least this many deg C below setpoint.",
    )
    parser.add_argument(
        "--pi-prior-weight",
        type=float,
        default=0.0,
        help="Weight on staying near PI/baseline heating away from district stress.",
    )
    parser.add_argument(
        "--pi-prior-temperature-gate-margin",
        type=float,
        default=None,
        help="If set, allow below-PI heating for free when indoor temperature is above setpoint plus this margin.",
    )
    parser.add_argument(
        "--structured-exploration-start",
        type=float,
        default=0.0,
        help="Probability of replacing SAC actions with structured random actions at the first SAC episode.",
    )
    parser.add_argument(
        "--structured-exploration-end",
        type=float,
        default=0.0,
        help="Final structured-exploration probability at the last SAC episode.",
    )
    parser.add_argument(
        "--structured-exploration-cold-min-action",
        type=float,
        default=0.5,
        help="Minimum sampled action for structured exploration when Tin <= Tsp.",
    )
    parser.add_argument("--structured-exploration-beta-alpha", type=float, default=0.7)
    parser.add_argument("--structured-exploration-beta-beta", type=float, default=0.7)
    parser.add_argument(
        "--comfort-shield-margin",
        type=float,
        default=None,
        help="If set, force a PI-style heating floor when Tin is more than this many degC below Tsp.",
    )
    parser.add_argument("--comfort-shield-min-action", type=float, default=0.35)
    parser.add_argument("--comfort-shield-kp", type=float, default=0.25)
    parser.add_argument(
        "--comfort-shield-eval",
        action="store_true",
        help="Apply the same cold comfort shield during deterministic February evaluation.",
    )
    parser.add_argument(
        "--comfort-shield-exploration",
        action="store_true",
        help="Also apply the comfort shield during PI exploration episodes (prevents LSTM crashes filling the replay buffer).",
    )
    parser.add_argument("--safe-pi-cold-margin", type=float, default=1.0)
    parser.add_argument("--safe-pi-hot-margin", type=float, default=1.0)
    parser.add_argument("--safe-pi-cold-min-action", type=float, default=0.25)
    parser.add_argument("--safe-pi-cold-kp", type=float, default=0.25)
    parser.add_argument("--safe-pi-hot-max-action", type=float, default=0.0)
    parser.add_argument("--safe-pi-random-prob", type=float, default=0.15)
    parser.add_argument("--safe-pi-random-low", type=float, default=0.0)
    parser.add_argument("--safe-pi-random-high", type=float, default=0.55)
    parser.add_argument("--safe-pi-ramp-limit", type=float, default=None)
    parser.add_argument(
        "--residual-safe-pi",
        action="store_true",
        help="Use safe-PI as the base action and train SAC on a bounded residual around it.",
    )
    parser.add_argument(
        "--residual-safe-pi-scale",
        type=float,
        default=0.5,
        help="Residual action scale: final_action = clip(safe_pi + scale*(policy_action - 0.5), 0, 1).",
    )
    parser.add_argument(
        "--checkpoint-select-by-eval",
        action="store_true",
        help="After each training episode, deterministically evaluate on the deployment window and keep the best policy.",
    )
    parser.add_argument(
        "--checkpoint-select-metric",
        default="discomfort_proportion",
        help="Metric minimized when --checkpoint-select-by-eval is active.",
    )
    parser.add_argument(
        "--training-temperature-reset",
        action="store_true",
        help="During January training only, reset runaway LSTM indoor temperatures back near the setpoint.",
    )
    parser.add_argument("--training-temperature-reset-cold-margin", type=float, default=4.0)
    parser.add_argument("--training-temperature-reset-hot-margin", type=float, default=4.0)
    parser.add_argument("--training-temperature-reset-target-margin", type=float, default=1.5)
    parser.add_argument("--training-temperature-reset-absolute-min", type=float, default=10.0)
    parser.add_argument("--training-temperature-reset-absolute-max", type=float, default=30.0)
    parser.add_argument("--training-temperature-reset-hidden", action="store_true", default=True)
    parser.add_argument("--no-training-temperature-reset-hidden", dest="training_temperature_reset_hidden", action="store_false")
    parser.add_argument("--training-temperature-reset-keep-transition", dest="training_temperature_reset_drop_transition", action="store_false")
    parser.add_argument(
        "--training-temperature-reset-replay-mode",
        choices=["drop", "keep", "penalty", "penalty_terminal"],
        default="drop",
        help="How to handle transitions that trigger a training temperature reset.",
    )
    parser.add_argument(
        "--training-temperature-reset-penalty",
        type=float,
        default=-1000.0,
        help="Learning reward stored for reset transitions when replay mode is penalty_terminal.",
    )
    parser.set_defaults(training_temperature_reset_drop_transition=True)
    parser.add_argument("--replay-filter-temperature-min", type=float, default=None)
    parser.add_argument("--replay-filter-temperature-max", type=float, default=None)
    parser.add_argument("--replay-filter-temperature-delta-max", type=float, default=None)
    parser.add_argument("--replay-filter-reward-abs-max", type=float, default=None)
    parser.add_argument(
        "--lagrange-init",
        type=float,
        default=1.0,
        help="Initial Lagrange multiplier λ (lagrangian reward only).",
    )
    parser.add_argument(
        "--lagrange-lr",
        type=float,
        default=0.05,
        help="Dual update learning rate η_λ (lagrangian reward only).",
    )
    parser.add_argument(
        "--lagrange-update-period",
        type=int,
        default=96,
        help="Steps between λ updates during SAC training (lagrangian reward only).",
    )
    parser.add_argument(
        "--comfort-budget",
        type=float,
        default=0.05,
        help="Acceptable mean cold violation in °C per step ε (lagrangian reward only).",
    )
    parser.add_argument(
        "--lagrange-min",
        type=float,
        default=0.0,
        help="Minimum value for λ (lagrangian reward only).",
    )
    parser.add_argument(
        "--lagrange-max",
        type=float,
        default=100.0,
        help="Maximum value for λ (lagrangian reward only).",
    )
    parser.add_argument("--train-sim-start", type=int, default=0,
                        help="First time step of the SAC training period (default: 0 = January).")
    parser.add_argument("--train-sim-end", type=int, default=743,
                        help="Last time step of the SAC training period (default: 743 = January).")
    parser.add_argument("--eval-sim-start", type=int, default=744,
                        help="First time step of the deployment/evaluation period (default: 744 = February).")
    parser.add_argument("--eval-sim-end", type=int, default=1415,
                        help="Last time step of the deployment/evaluation period (default: 1415 = February).")
    parser.add_argument(
        "--rc-rollout-horizons",
        nargs="+",
        type=int,
        default=[],
        help="Append 1R1C forward-rollout features to SAC observations at these horizons (hours). "
             "E.g. --rc-rollout-horizons 4 8 adds cold-deficit and heat-effect at 4h and 8h.",
    )
    parser.add_argument(
        "--rc-rollout-model-dir",
        default="results/models/rc",
        help="Directory containing 1state_rc_{short_name}_january.json files.",
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
        default=0.0,
        help="simple_v4 reward: shift tracking target Tsp - bias (°C). "
             "0 = track actual Tsp (fixed default); 0.5 = old behaviour (underheats).",
    )
    parser.add_argument(
        "--cold-only",
        action="store_true",
        default=False,
        help="simple_v4 reward: one-sided comfort penalty (cold only). "
             "Default False = symmetric, matching MPC stage cost.",
    )
    return parser


_MPC_MODULE = None


def _load_mpc_module():
    global _MPC_MODULE
    if _MPC_MODULE is not None:
        return _MPC_MODULE
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "03_run_mpc.py"
    spec = importlib.util.spec_from_file_location("ce1_mpc_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MPC runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MPC_MODULE = module
    return module


def _load_mpc_rc_models(args: argparse.Namespace, building: dict) -> tuple[dict, dict | None, dict]:
    mpc_module = _load_mpc_module()
    rc_model_path = mpc_module.resolve_rc_model_path(args.mpc_rc_model, building)
    if not rc_model_path.exists():
        raise FileNotFoundError(f"Missing MPC jump-start RC model for {building['display']}: {rc_model_path}")
    rc_params = json.loads(rc_model_path.read_text())
    metadata = {
        "jumpstart_mpc_rc_model": str(rc_model_path),
        "jumpstart_mpc_rc_model_type": str(args.mpc_rc_model_type),
    }
    rc2_params = None
    if args.mpc_rc_model_type == "2r2c":
        rc2_path = Path(format_building_path(args.mpc_rc_model_2r2c, building))
        if not rc2_path.exists():
            raise FileNotFoundError(f"Missing MPC jump-start 2R2C model for {building['display']}: {rc2_path}")
        rc2_params = json.loads(rc2_path.read_text())
        metadata["jumpstart_mpc_rc_model_2r2c"] = str(rc2_path)
    return rc_params, rc2_params, metadata


def _save_mpc_jumpstart_rollout(
    *,
    output_dir: str | Path,
    building: dict,
    label: str,
    teacher_result: dict,
    filename: str = "mpc_jumpstart_rollout.csv",
) -> Path:
    mpc_module = _load_mpc_module()
    path = Path(output_dir) / building["result_id"] / "no_occupant" / label / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    history = mpc_module.rollout_history_dataframe(teacher_result["env"], teacher_result)
    history.to_csv(path, index=False)
    return path


def main() -> None:
    args = build_parser().parse_args()

    import pandas as pd
    from src.sac_merlin_pipeline import (
        JAN_END,
        JAN_START,
        FEB_END,
        FEB_START,
        MerlinSACConfig,
        collect_bc_expert_dataset,
        fit_bspline_occupancy_model,
        fitted_occupants,
        reward_function_from_mode,
        run_merlin_sac_no_occupant,
    )

    config = MerlinSACConfig(
        lr=args.lr,
        alpha=args.alpha,
        automatic_entropy_tuning=args.automatic_entropy_tuning,
        entropy_lr=args.entropy_lr,
        target_entropy=args.target_entropy,
        robust_normalization=args.robust_normalization,
        robust_normalization_low=args.robust_normalization_low,
        robust_normalization_high=args.robust_normalization_high,
        normalized_observation_clip=args.normalized_observation_clip,
        normalized_reward_clip=args.normalized_reward_clip,
        observation_filter=args.observation_filter,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
        action_scaling_coefficient=args.action_scaling_coefficient,
        n_step_horizon=args.n_step_horizon,
        device=None if args.device == "auto" else args.device,
    )
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    all_summary_rows = []

    for building in buildings:
        building_name = str(building["name"])
        if args.occupancy_building.lower() == "same":
            occupancy_building_name = building_name
        else:
            occupancy_entries = resolve_building_entries(args.occupancy_building, DATASET_NAME)
            if len(occupancy_entries) != 1:
                raise ValueError("--occupancy-building must resolve to one building, not 'all'.")
            occupancy_building_name = str(occupancy_entries[0]["name"])
        metadata = building_metadata(building, occupancy_building_name=occupancy_building_name)
        baseline_daily_mean_exp_train = None
        baseline_daily_mean_exp_eval = None
        if args.reward == "simple_v4" or args.load_shape_profile == "baseline_daily_mean_exp":
            baseline_daily_mean_exp_train = load_simple_v4_price_profile(simulation_start=JAN_START, exp_beta=2.0)
            baseline_daily_mean_exp_eval = load_simple_v4_price_profile(simulation_start=FEB_START, exp_beta=2.0)
            if not baseline_daily_mean_exp_eval:
                print("WARNING: district baseline not found; GridStress = 1.0 (flat)")
        grid_stress_profile = None
        grid_stress_profile_train = None
        grid_stress_profile_eval = None
        future_grid_stress_profile_train = None
        future_grid_stress_profile_eval = None
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
            grid_stress_profile_train = grid_stress_profile
            grid_stress_profile_eval = grid_stress_profile
        elif float(args.shift_weight) != 0.0 and args.load_shape_profile == "daily_windows":
            grid_stress_profile_train = daily_window_shift_profile(
                simulation_start=JAN_START,
                simulation_end=JAN_END,
            )
            grid_stress_profile_eval = daily_window_shift_profile(
                simulation_start=FEB_START,
                simulation_end=FEB_END,
            )
            grid_stress_profile = grid_stress_profile_eval
            metadata["grid_stress_mode"] = "daily_windows_signed"
            metadata["grid_stress_profile_mean"] = float(
                sum(grid_stress_profile_eval) / max(len(grid_stress_profile_eval), 1)
            )
            metadata["grid_stress_profile_min"] = (
                float(min(grid_stress_profile_eval)) if grid_stress_profile_eval else 0.0
            )
            metadata["grid_stress_profile_max"] = (
                float(max(grid_stress_profile_eval)) if grid_stress_profile_eval else 0.0
            )
        elif float(args.shift_weight) != 0.0 and args.load_shape_profile == "baseline_daily_mean_exp":
            grid_stress_profile_train = baseline_daily_mean_exp_train or None
            grid_stress_profile_eval = baseline_daily_mean_exp_eval or None
            grid_stress_profile = grid_stress_profile_eval
            metadata["grid_stress_mode"] = "baseline_district_daily_mean_exp2_norm"
            metadata["grid_stress_profile_mean"] = (
                float(sum(grid_stress_profile_eval) / max(len(grid_stress_profile_eval), 1))
                if grid_stress_profile_eval else 0.0
            )
            metadata["grid_stress_profile_min"] = (
                float(min(grid_stress_profile_eval)) if grid_stress_profile_eval else 0.0
            )
            metadata["grid_stress_profile_max"] = (
                float(max(grid_stress_profile_eval)) if grid_stress_profile_eval else 0.0
            )
        district_profiles = None
        district_profiles_train = None
        needs_district_profiles = (
            float(args.district_target_weight) != 0.0
            or float(args.heat_budget_weight) != 0.0
            or float(args.pi_prior_weight) != 0.0
            or (
                (float(args.shift_weight) != 0.0 or float(args.preheat_weight) != 0.0)
                and grid_stress_profile_eval is None
            )
            or args.power_scale_mode == "nominal_heating"
        )
        if needs_district_profiles:
            district_profiles = load_ce1_district_target_profiles(
                building_name=building_name,
                simulation_start=FEB_START,
                simulation_end=FEB_END,
                dataset_name=DATASET_NAME,
                target_scale=args.district_target_scale,
                target_scale_mode=args.district_target_scale_mode,
            )
            district_profiles_train = load_ce1_district_target_profiles(
                building_name=building_name,
                simulation_start=JAN_START,
                simulation_end=JAN_END,
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
            metadata["baseline_controlled_heating_mean"] = float(
                sum(district_profiles.baseline_controlled_heating)
                / max(len(district_profiles.baseline_controlled_heating), 1)
            )
            metadata["controlled_heating_nominal_power"] = float(district_profiles.heating_nominal_power)
            if grid_stress_profile_eval is None and (
                float(args.shift_weight) != 0.0 or float(args.preheat_weight) != 0.0
            ):
                grid_stress_profile_train = district_profiles_train.district_stress_profile()
                grid_stress_profile_eval = district_profiles.district_stress_profile()
                grid_stress_profile = grid_stress_profile_eval
                metadata["grid_stress_mode"] = "ce1_baseline_above_target"
                metadata["grid_stress_profile_mean"] = float(
                    sum(grid_stress_profile_eval) / max(len(grid_stress_profile_eval), 1)
                )
            if float(args.preheat_weight) != 0.0:
                future_grid_stress_profile_train = district_profiles_train.future_stress_profile(args.preheat_horizon)
                future_grid_stress_profile_eval = district_profiles.future_stress_profile(args.preheat_horizon)
                metadata["preheat_weight"] = float(args.preheat_weight)
                metadata["preheat_horizon"] = int(args.preheat_horizon)
                metadata["future_grid_stress_profile_mean"] = float(
                    sum(future_grid_stress_profile_eval) / max(len(future_grid_stress_profile_eval), 1)
                )
        if (
            float(args.preheat_weight) != 0.0
            and future_grid_stress_profile_eval is None
            and grid_stress_profile_eval is not None
        ):
            future_grid_stress_profile_train = future_stress_from_profile(
                grid_stress_profile_train,
                horizon=args.preheat_horizon,
            )
            future_grid_stress_profile_eval = future_stress_from_profile(
                grid_stress_profile_eval,
                horizon=args.preheat_horizon,
            )
            metadata["preheat_weight"] = float(args.preheat_weight)
            metadata["preheat_horizon"] = int(args.preheat_horizon)
            metadata["future_grid_stress_profile_mean"] = float(
                sum(future_grid_stress_profile_eval) / max(len(future_grid_stress_profile_eval), 1)
            )
        effective_power_scale = float(args.power_scale)
        if args.power_scale_mode == "nominal_heating":
            if district_profiles is None:
                raise ValueError("--power-scale-mode nominal_heating requires district profiles.")
            effective_power_scale = float(district_profiles.heating_nominal_power)
        metadata["reward_power_source"] = args.power_source
        metadata["reward_power_scale"] = float(effective_power_scale)
        metadata["reward_power_scale_mode"] = args.power_scale_mode
        action_proxy_power = None
        if args.power_source == "action_proxy":
            proxy_template = args.mpc_rc_model_2r2c if args.mpc_rc_model_type == "2r2c" else args.mpc_rc_model
            proxy_path = Path(format_building_path(proxy_template, building))
            if not proxy_path.exists():
                raise FileNotFoundError(f"Missing action-proxy RC model for {building['display']}: {proxy_path}")
            proxy_params = json.loads(proxy_path.read_text())
            action_proxy_power = float(proxy_params["power_per_unit_action"])
            metadata["reward_action_proxy_rc_model"] = str(proxy_path)
            metadata["reward_action_proxy_power_per_unit_action"] = action_proxy_power
        output_dir = Path(args.output_dir) / building["result_id"]
        summary_dir = Path(args.summary_dir) / building["result_id"]
        model_dir = Path(args.model_dir) / building["result_id"]
        seed = int(args.seed + 1000 * (int(building["index"]) - 1))
        training_seed = int(args.training_seed + 1000 * (int(building["index"]) - 1))

        print(f"\n=== {building['display']}: {building_name} ===")
        if args.scenario == "occupant_present":
            print(f"Occupancy model source: {occupancy_building_name}")

        if args.reward == "simple_v4":
            metadata["simple_price_profile"] = "baseline_district_daily_mean_exp2_norm"
            metadata["simple_v4_tsp_lower_bias"] = float(args.tsp_lower_bias)
            metadata["simple_v4_ramp_weight"] = float(args.ramp_weight)
            metadata["simple_v4_cold_only"] = bool(args.cold_only)

        if args.scenario == "no_occupant":
            reward_function_kwargs = None
            eval_reward_function_kwargs = None
            setpoint_bias_profile_train = None
            if args.reward == "simple_v4":
                # Use stronger comfort weight for simple_v4 (0.5 is too weak vs energy term).
                # Explicit --comfort-weight overrides this; otherwise default to 1.5.
                _v4_comfort_weight = args.comfort_weight if args.comfort_weight != 0.5 else 1.5
                reward_function_kwargs = dict(
                    comfort_weight=_v4_comfort_weight,
                    energy_weight=args.energy_weight,
                    comfort_band=args.comfort_band,
                    tsp_lower_bias=args.tsp_lower_bias,
                    ramp_weight=args.ramp_weight,
                    power_scale=effective_power_scale,
                    shift_profile=baseline_daily_mean_exp_train or None,
                    cold_only=args.cold_only,
                )
                eval_reward_function_kwargs = dict(reward_function_kwargs)
                eval_reward_function_kwargs["shift_profile"] = baseline_daily_mean_exp_eval or None
            elif args.reward in {"balanced", "lagrangian"}:
                comfort_scale = 1.0 if args.comfort_mode in {"asymmetric", "time_bias_guard"} and args.comfort_scale is None else (
                    float(args.comfort_band) ** 2
                    if args.comfort_scale is None
                    else float(args.comfort_scale)
                )
                setpoint_bias_profile_train = None
                setpoint_bias_profile_eval = None
                if args.comfort_mode == "time_bias_guard":
                    setpoint_bias_profile_train = time_bias_setpoint_profile(
                        simulation_start=args.train_sim_start,
                        simulation_end=args.train_sim_end,
                        amplitude=args.time_bias_amplitude,
                    )
                    setpoint_bias_profile_eval = time_bias_setpoint_profile(
                        simulation_start=args.eval_sim_start,
                        simulation_end=args.eval_sim_end,
                        amplitude=args.time_bias_amplitude,
                    )
                    metadata["time_bias_amplitude"] = float(args.time_bias_amplitude)
                    metadata["time_bias_guard_band"] = float(args.time_bias_guard_band)
                    metadata["setpoint_bias_profile_mean"] = float(
                        sum(setpoint_bias_profile_eval) / max(len(setpoint_bias_profile_eval), 1)
                    )
                metadata["comfort_mode"] = args.comfort_mode
                metadata["comfort_scale"] = float(comfort_scale)
                metadata["comfort_track_weight"] = float(args.comfort_track_weight)
                metadata["comfort_cold_weight"] = float(args.comfort_cold_weight)
                metadata["comfort_hot_weight"] = float(args.comfort_hot_weight)
                metadata["comfort_cold_tolerance"] = float(args.comfort_cold_tolerance)
                metadata["comfort_hot_tolerance"] = float(args.comfort_hot_tolerance)
                metadata["comfort_track_scale"] = float(args.comfort_track_scale)
                metadata["comfort_cold_scale"] = float(args.comfort_cold_scale)
                metadata["comfort_hot_scale"] = float(args.comfort_hot_scale)
                metadata["comfort_guard_exponent"] = float(args.comfort_guard_exponent)
                metadata["comfort_cold_k"] = float(args.comfort_cold_k)
                metadata["comfort_hot_k"] = float(args.comfort_hot_k)
                metadata["single_setpoint"] = bool(args.single_setpoint)
                metadata["heat_budget_weight"] = float(args.heat_budget_weight)
                metadata["heat_budget_window"] = int(args.heat_budget_window)
                metadata["heat_budget_tolerance"] = float(args.heat_budget_tolerance)
                metadata["heat_budget_temperature_gate_margin"] = (
                    None
                    if args.heat_budget_temperature_gate_margin is None
                    else float(args.heat_budget_temperature_gate_margin)
                )
                metadata["pi_prior_weight"] = float(args.pi_prior_weight)
                metadata["pi_prior_temperature_gate_margin"] = (
                    None
                    if args.pi_prior_temperature_gate_margin is None
                    else float(args.pi_prior_temperature_gate_margin)
                )
                metadata["automatic_entropy_tuning"] = bool(args.automatic_entropy_tuning)
                metadata["entropy_lr"] = None if args.entropy_lr is None else float(args.entropy_lr)
                metadata["target_entropy"] = None if args.target_entropy is None else float(args.target_entropy)
                metadata["robust_normalization"] = bool(args.robust_normalization)
                metadata["normalized_observation_clip"] = (
                    None if args.normalized_observation_clip is None else float(args.normalized_observation_clip)
                )
                metadata["normalized_reward_clip"] = (
                    None if args.normalized_reward_clip is None else float(args.normalized_reward_clip)
                )
                metadata["observation_filter"] = args.observation_filter or "none"
                metadata["structured_exploration_start"] = float(args.structured_exploration_start)
                metadata["structured_exploration_end"] = float(args.structured_exploration_end)
                metadata["structured_exploration_cold_min_action"] = float(
                    args.structured_exploration_cold_min_action
                )
                metadata["structured_exploration_beta_alpha"] = float(args.structured_exploration_beta_alpha)
                metadata["structured_exploration_beta_beta"] = float(args.structured_exploration_beta_beta)
                metadata["comfort_shield_margin"] = (
                    None if args.comfort_shield_margin is None else float(args.comfort_shield_margin)
                )
                metadata["comfort_shield_min_action"] = float(args.comfort_shield_min_action)
                metadata["comfort_shield_kp"] = float(args.comfort_shield_kp)
                metadata["comfort_shield_eval"] = bool(args.comfort_shield_eval)
                metadata["safe_pi_cold_margin"] = float(args.safe_pi_cold_margin)
                metadata["safe_pi_hot_margin"] = float(args.safe_pi_hot_margin)
                metadata["safe_pi_random_prob"] = float(args.safe_pi_random_prob)
                metadata["residual_safe_pi"] = bool(args.residual_safe_pi)
                metadata["residual_safe_pi_scale"] = float(args.residual_safe_pi_scale)
                metadata["checkpoint_select_by_eval"] = bool(args.checkpoint_select_by_eval)
                metadata["checkpoint_select_metric"] = str(args.checkpoint_select_metric)
                metadata["training_temperature_reset"] = bool(args.training_temperature_reset)
                metadata["training_temperature_reset_cold_margin"] = float(args.training_temperature_reset_cold_margin)
                metadata["training_temperature_reset_hot_margin"] = float(args.training_temperature_reset_hot_margin)
                metadata["training_temperature_reset_target_margin"] = float(args.training_temperature_reset_target_margin)
                metadata["training_temperature_reset_absolute_min"] = (
                    None
                    if args.training_temperature_reset_absolute_min is None
                    else float(args.training_temperature_reset_absolute_min)
                )
                metadata["training_temperature_reset_absolute_max"] = (
                    None
                    if args.training_temperature_reset_absolute_max is None
                    else float(args.training_temperature_reset_absolute_max)
                )
                metadata["training_temperature_reset_hidden"] = bool(args.training_temperature_reset_hidden)
                metadata["training_temperature_reset_drop_transition"] = bool(
                    args.training_temperature_reset_drop_transition
                )
                metadata["training_temperature_reset_replay_mode"] = str(
                    args.training_temperature_reset_replay_mode
                )
                metadata["training_temperature_reset_penalty"] = float(
                    -abs(float(args.training_temperature_reset_penalty))
                )
                metadata["early_stop_reward_slope"] = bool(args.early_stop_reward_slope)
                metadata["early_stop_min_episodes"] = int(args.early_stop_min_episodes)
                metadata["early_stop_window"] = int(args.early_stop_window)
                metadata["early_stop_patience"] = int(args.early_stop_patience)
                metadata["early_stop_slope_abs"] = float(args.early_stop_slope_abs)
                metadata["replay_filter_temperature_min"] = (
                    None if args.replay_filter_temperature_min is None else float(args.replay_filter_temperature_min)
                )
                metadata["replay_filter_temperature_max"] = (
                    None if args.replay_filter_temperature_max is None else float(args.replay_filter_temperature_max)
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
                    comfort_guard_exponent=args.comfort_guard_exponent,
                    comfort_cold_k=args.comfort_cold_k,
                    comfort_hot_k=args.comfort_hot_k,
                    single_setpoint=args.single_setpoint,
                    power_scale=effective_power_scale,
                    power_source=args.power_source,
                    action_power_coefficients=(
                        None if action_proxy_power is None else [action_proxy_power]
                    ),
                    setpoint_bias_profile=setpoint_bias_profile_train,
                    time_bias_guard_band=args.time_bias_guard_band,
                    shift_profile=grid_stress_profile_train,
                    future_shift_profile=future_grid_stress_profile_train,
                    heat_budget_weight=args.heat_budget_weight,
                    heat_budget_window=args.heat_budget_window,
                    heat_budget_tolerance=args.heat_budget_tolerance,
                    heat_budget_temperature_gate_margin=args.heat_budget_temperature_gate_margin,
                    pi_prior_weight=args.pi_prior_weight,
                    pi_prior_temperature_gate_margin=args.pi_prior_temperature_gate_margin,
                )
                eval_reward_function_kwargs = dict(reward_function_kwargs)
                eval_reward_function_kwargs["setpoint_bias_profile"] = setpoint_bias_profile_eval
                eval_reward_function_kwargs["shift_profile"] = grid_stress_profile_eval
                eval_reward_function_kwargs["future_shift_profile"] = future_grid_stress_profile_eval
                if args.reward == "lagrangian":
                    lagrange_kwargs = dict(
                        lagrange_multiplier=args.lagrange_init,
                        lagrange_lr=args.lagrange_lr,
                        comfort_constraint_budget=args.comfort_budget,
                        lagrange_update_period=args.lagrange_update_period,
                        lagrange_min=args.lagrange_min,
                        lagrange_max=args.lagrange_max,
                    )
                    reward_function_kwargs.update(lagrange_kwargs)
                    eval_reward_function_kwargs.update(lagrange_kwargs)
                    metadata["lagrange_init"] = float(args.lagrange_init)
                    metadata["lagrange_lr"] = float(args.lagrange_lr)
                    metadata["comfort_budget"] = float(args.comfort_budget)
                    metadata["lagrange_update_period"] = int(args.lagrange_update_period)
                if district_profiles is not None:
                    reward_function_kwargs["heat_budget_profile"] = district_profiles_train.baseline_controlled_heating
                    reward_function_kwargs["pi_prior_profile"] = district_profiles_train.baseline_controlled_heating
                    eval_reward_function_kwargs["heat_budget_profile"] = district_profiles.baseline_controlled_heating
                    eval_reward_function_kwargs["pi_prior_profile"] = district_profiles.baseline_controlled_heating
                    if float(args.district_target_weight) != 0.0:
                        reward_function_kwargs.update(
                            district_profiles_train.as_reward_kwargs(
                                target_weight=args.district_target_weight,
                                target_mode=args.district_target_mode,
                            )
                        )
                        eval_reward_function_kwargs.update(
                            district_profiles.as_reward_kwargs(
                                target_weight=args.district_target_weight,
                                target_mode=args.district_target_mode,
                            )
                        )
                    eval_reward_function_kwargs["shift_profile"] = grid_stress_profile_eval
                    eval_reward_function_kwargs["future_shift_profile"] = future_grid_stress_profile_eval
            _reward_mode = args.reward if args.reward not in {"all", "both"} else "comfort"
            if args.reward == "simple_v4":
                _reward_cls = SimpleComfortEnergyReward
            else:
                _reward_cls = reward_function_from_mode(_reward_mode)
            jumpstart_actions = None
            bc_expert_observations = None
            bc_expert_actions = None
            resume_agent_path = None
            if args.resume_agent_template:
                resume_agent_path = Path(
                    str(args.resume_agent_template).format(
                        building=building["name"],
                        building_name=building["name"],
                        building_id=building["result_id"],
                        index=building["index"],
                        short_name=building["short_name"],
                        label=args.label,
                    )
                )
                metadata["resume_agent_path"] = str(resume_agent_path)
                metadata["resume_episode_offset"] = int(args.resume_episode_offset)
                metadata["resume_replay_keep_last"] = (
                    None if args.resume_replay_keep_last is None else int(args.resume_replay_keep_last)
                )
            needs_mpc_teacher = args.jumpstart_controller == "mpc" or args.bc_teacher_controller == "mpc"
            if needs_mpc_teacher:
                if args.reward not in {"balanced", "lagrangian", "simple_v4"}:
                    raise ValueError("MPC teacher currently supports balanced/lagrangian/simple_v4 no-occupant runs.")
                rc_params, rc2_params, mpc_metadata = _load_mpc_rc_models(args, building)
                metadata.update(mpc_metadata)
                mpc_module = _load_mpc_module()
                if args.reward == "simple_v4":
                    teacher_objective_kwargs = dict(
                        controller_name=f"MPC jump-start teacher | {building['display']}",
                        objective_mode="simple",
                        w_comfort=_v4_comfort_weight,
                        w_energy=args.energy_weight,
                        comfort_band=args.comfort_band,
                        simple_tsp_lower_bias=args.tsp_lower_bias,
                        grid_stress_profile=baseline_daily_mean_exp_train or None,
                    )
                    teacher_cold_penalty = 1e5
                    teacher_hot_penalty = 1e3
                else:
                    teacher_objective_kwargs = dict(
                        controller_name=f"MPC jump-start teacher | {building['display']}",
                        objective_mode="balanced",
                        w_energy=args.energy_weight,
                        w_comfort=args.comfort_weight,
                        lambda_u=args.peak_weight,
                        lambda_du=args.ramp_weight,
                        shift_weight=args.shift_weight,
                        preheat_weight=args.preheat_weight,
                        use_smooth_du=True,
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
                        comfort_guard_exponent=args.comfort_guard_exponent,
                        comfort_cold_k=args.comfort_cold_k,
                        comfort_hot_k=args.comfort_hot_k,
                        power_scale=effective_power_scale,
                        setpoint_bias_profile=setpoint_bias_profile_train,
                        time_bias_guard_band=args.time_bias_guard_band,
                        normalize_cost_terms=True,
                        grid_stress_profile=grid_stress_profile_train,
                        future_grid_stress_profile=future_grid_stress_profile_train,
                        district_target_weight=args.district_target_weight,
                        district_target_profile=None if district_profiles_train is None else district_profiles_train.target,
                        district_background_profile=(
                            None
                            if district_profiles_train is None
                            else district_profiles_train.background_without_controlled_heating
                        ),
                        district_action_power_profile=(
                            None if district_profiles_train is None else district_profiles_train.heating_action_power
                        ),
                        district_target_scale=1.0 if district_profiles_train is None else district_profiles_train.target_scale,
                        district_target_mode=args.district_target_mode,
                    )
                    teacher_cold_penalty = 1e5
                    teacher_hot_penalty = 1e5 if args.comfort_mode == "time_bias_guard" else 1e3
                print("Running January MPC teacher for SAC labels...")
                teacher_result = mpc_module.run_february_mpc(
                    rc_params=rc_params,
                    rc2_params=rc2_params,
                    cold_penalty=teacher_cold_penalty,
                    hot_penalty=teacher_hot_penalty,
                    seed=seed,
                    max_steps=args.max_steps,
                    sim_start=args.train_sim_start,
                    sim_end=args.train_sim_end,
                    building_name=building_name,
                    reward_function=_reward_cls,
                    reward_function_kwargs=reward_function_kwargs,
                    two_state_observer=args.mpc_two_state_observer,
                    kalman_process_var=args.mpc_kalman_process_var,
                    kalman_measurement_var=args.mpc_kalman_measurement_var,
                    kalman_initial_var=args.mpc_kalman_initial_var,
                    **teacher_objective_kwargs,
                )
                teacher_actions = [float(v) for v in teacher_result["u_hist"]]
                teacher_path = _save_mpc_jumpstart_rollout(
                    output_dir=args.output_dir,
                    building=building,
                    label=args.label,
                    teacher_result=teacher_result,
                    filename=(
                        "mpc_jumpstart_rollout.csv"
                        if args.jumpstart_controller == "mpc"
                        else "mpc_bc_teacher_rollout.csv"
                    ),
                )
                metadata["mpc_teacher_rollout"] = str(teacher_path)
                metadata["mpc_teacher_reward_total"] = float(teacher_result["reward_hist"].sum())
                metadata["mpc_teacher_action_mean"] = float(sum(teacher_actions) / max(len(teacher_actions), 1))
                metadata["mpc_teacher_action_max"] = float(max(teacher_actions)) if teacher_actions else 0.0
                if args.jumpstart_controller == "mpc":
                    jumpstart_actions = teacher_actions
                    metadata["jumpstart_mpc_rollout"] = str(teacher_path)
                if args.bc_teacher_controller == "mpc":
                    bc_dataset = collect_bc_expert_dataset(
                        actions=teacher_actions,
                        seed=seed,
                        reward_function=_reward_cls,
                        reward_function_kwargs=reward_function_kwargs,
                        building_name=building_name,
                        max_steps=args.max_steps,
                        sim_start=args.train_sim_start,
                        sim_end=args.train_sim_end,
                    )
                    bc_expert_observations = bc_dataset["observations"]
                    bc_expert_actions = bc_dataset["actions"]
                    metadata["bc_teacher_controller"] = "mpc"
                    metadata["bc_expert_samples"] = int(len(bc_expert_actions))
                    metadata["bc_epochs_requested"] = int(args.bc_epochs)
                    metadata["bc_batch_size_requested"] = int(args.bc_batch_size)
                    metadata["bc_lr_requested"] = float(args.bc_lr)
                    metadata["bc_validation_fraction"] = float(args.bc_validation_fraction)
                    metadata["bc_early_stopping_patience"] = int(args.bc_early_stopping_patience)
                    metadata["bc_min_delta"] = float(args.bc_min_delta)
            case = run_merlin_sac_no_occupant(
                output_dir=output_dir,
                summary_dir=summary_dir,
                model_dir=model_dir,
                seed=seed,
                reward_function=_reward_cls,
                reward_function_kwargs=reward_function_kwargs,
                eval_reward_function_kwargs=eval_reward_function_kwargs,
                max_steps=args.max_steps,
                label=args.label,
                config=config,
                training_episodes=args.training_episodes,
                building_name=building_name,
                building_metadata=metadata,
                jumpstart_controller=args.jumpstart_controller,
                jumpstart_actions=jumpstart_actions,
                bc_expert_observations=bc_expert_observations,
                bc_expert_actions=bc_expert_actions,
                bc_epochs=args.bc_epochs,
                bc_batch_size=args.bc_batch_size,
                bc_lr=args.bc_lr,
                bc_validation_fraction=args.bc_validation_fraction,
                bc_early_stopping_patience=args.bc_early_stopping_patience,
                bc_min_delta=args.bc_min_delta,
                structured_exploration_start=args.structured_exploration_start,
                structured_exploration_end=args.structured_exploration_end,
                structured_exploration_cold_min_action=args.structured_exploration_cold_min_action,
                structured_exploration_beta_alpha=args.structured_exploration_beta_alpha,
                structured_exploration_beta_beta=args.structured_exploration_beta_beta,
                comfort_shield_margin=args.comfort_shield_margin,
                comfort_shield_min_action=args.comfort_shield_min_action,
                comfort_shield_kp=args.comfort_shield_kp,
                comfort_shield_eval=args.comfort_shield_eval,
                comfort_shield_exploration=args.comfort_shield_exploration,
                safe_pi_cold_margin=args.safe_pi_cold_margin,
                safe_pi_hot_margin=args.safe_pi_hot_margin,
                safe_pi_cold_min_action=args.safe_pi_cold_min_action,
                safe_pi_cold_kp=args.safe_pi_cold_kp,
                safe_pi_hot_max_action=args.safe_pi_hot_max_action,
                safe_pi_random_prob=args.safe_pi_random_prob,
                safe_pi_random_low=args.safe_pi_random_low,
                safe_pi_random_high=args.safe_pi_random_high,
                safe_pi_ramp_limit=args.safe_pi_ramp_limit,
                residual_safe_pi=args.residual_safe_pi,
                residual_safe_pi_scale=args.residual_safe_pi_scale,
                training_temperature_reset=args.training_temperature_reset,
                training_temperature_reset_cold_margin=args.training_temperature_reset_cold_margin,
                training_temperature_reset_hot_margin=args.training_temperature_reset_hot_margin,
                training_temperature_reset_target_margin=args.training_temperature_reset_target_margin,
                training_temperature_reset_absolute_min=args.training_temperature_reset_absolute_min,
                training_temperature_reset_absolute_max=args.training_temperature_reset_absolute_max,
                training_temperature_reset_hidden=args.training_temperature_reset_hidden,
                training_temperature_reset_drop_transition=args.training_temperature_reset_drop_transition,
                training_temperature_reset_replay_mode=args.training_temperature_reset_replay_mode,
                training_temperature_reset_penalty=args.training_temperature_reset_penalty,
                replay_filter_temperature_min=args.replay_filter_temperature_min,
                replay_filter_temperature_max=args.replay_filter_temperature_max,
                replay_filter_temperature_delta_max=args.replay_filter_temperature_delta_max,
                replay_filter_reward_abs_max=args.replay_filter_reward_abs_max,
                rc_rollout_horizons=tuple(args.rc_rollout_horizons) if args.rc_rollout_horizons else None,
                rc_rollout_model_dir=args.rc_rollout_model_dir,
                train_sim_start=args.train_sim_start,
                train_sim_end=args.train_sim_end,
                eval_sim_start=args.eval_sim_start,
                eval_sim_end=args.eval_sim_end,
                resume_agent_path=resume_agent_path,
                resume_episode_offset=args.resume_episode_offset,
                resume_replay_keep_last=args.resume_replay_keep_last,
                early_stop_reward_slope=args.early_stop_reward_slope,
                early_stop_min_episodes=args.early_stop_min_episodes,
                early_stop_window=args.early_stop_window,
                early_stop_patience=args.early_stop_patience,
                early_stop_slope_abs=args.early_stop_slope_abs,
                checkpoint_select_by_eval=args.checkpoint_select_by_eval,
                checkpoint_select_metric=args.checkpoint_select_metric,
            )
            all_summary_rows.append(case["metrics"])
        elif args.scenario == "occupant_present":
            if args.reward not in {"balanced", "lagrangian", "occupant_prob"}:
                raise ValueError("Final occupant SAC experiment expects --reward balanced, lagrangian, or occupant_prob.")
            comfort_scale = 1.0 if args.comfort_mode in {"asymmetric", "time_bias_guard"} and args.comfort_scale is None else (
                float(args.comfort_band) ** 2
                if args.comfort_scale is None
                else float(args.comfort_scale)
            )
            setpoint_bias_profile_train = None
            setpoint_bias_profile_eval = None
            if args.comfort_mode == "time_bias_guard":
                setpoint_bias_profile_train = time_bias_setpoint_profile(
                    simulation_start=args.train_sim_start,
                    simulation_end=args.train_sim_end,
                    amplitude=args.time_bias_amplitude,
                )
                setpoint_bias_profile_eval = time_bias_setpoint_profile(
                    simulation_start=args.eval_sim_start,
                    simulation_end=args.eval_sim_end,
                    amplitude=args.time_bias_amplitude,
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
                comfort_guard_exponent=args.comfort_guard_exponent,
                comfort_cold_k=args.comfort_cold_k,
                comfort_hot_k=args.comfort_hot_k,
                single_setpoint=args.single_setpoint,
                power_scale=effective_power_scale,
                power_source=args.power_source,
                action_power_coefficients=(None if action_proxy_power is None else [action_proxy_power]),
                setpoint_bias_profile=setpoint_bias_profile_train,
                time_bias_guard_band=args.time_bias_guard_band,
                shift_profile=grid_stress_profile_train,
                future_shift_profile=future_grid_stress_profile_train,
                heat_budget_weight=args.heat_budget_weight,
                heat_budget_window=args.heat_budget_window,
                heat_budget_tolerance=args.heat_budget_tolerance,
                heat_budget_temperature_gate_margin=args.heat_budget_temperature_gate_margin,
                pi_prior_weight=args.pi_prior_weight,
                pi_prior_temperature_gate_margin=args.pi_prior_temperature_gate_margin,
                occupant_probability_base=args.occupant_probability_base,
                occupant_probability_clip=not args.no_occupant_probability_clip,
            )
            eval_reward_function_kwargs = dict(reward_function_kwargs)
            eval_reward_function_kwargs["setpoint_bias_profile"] = setpoint_bias_profile_eval
            eval_reward_function_kwargs["shift_profile"] = grid_stress_profile_eval
            eval_reward_function_kwargs["future_shift_profile"] = future_grid_stress_profile_eval
            _reward_cls = reward_function_from_mode(args.reward)
            jumpstart_actions = None
            bc_expert_observations = None
            bc_expert_actions = None
            resume_agent_path = None
            if args.resume_agent_template:
                resume_agent_path = Path(
                    str(args.resume_agent_template).format(
                        building=building["name"],
                        building_name=building["name"],
                        building_id=building["result_id"],
                        index=building["index"],
                        short_name=building["short_name"],
                        label=args.label,
                    )
                )
            occupants = fitted_occupants()
            if args.occupant != "all":
                occupants = {args.occupant: occupants[args.occupant]}
            mode_names = []
            if args.occupant_mode in ("without_tdyn", "both"):
                mode_names.append("without_tdyn")
            if args.occupant_mode in ("bspline_tdyn", "both"):
                mode_names.append("bspline_tdyn")
            for occ_name, occ in occupants.items():
                for mode_name in mode_names:
                    occupancy_model = None
                    occupancy_series = None
                    if mode_name == "bspline_tdyn":
                        occupancy_model, occupancy_series = fit_bspline_occupancy_model(
                            seed=seed,
                            building_name=occupancy_building_name,
                        )
                    case_label = args.label
                    if args.occupant == "all" or args.occupant_mode == "both":
                        case_label = f"{args.label}_{occ_name}_{mode_name}"
                    case_reward_function_kwargs = reward_function_kwargs
                    case_eval_reward_function_kwargs = eval_reward_function_kwargs
                    if args.reward == "occupant_prob":
                        case_reward_function_kwargs = dict(reward_function_kwargs)
                        case_eval_reward_function_kwargs = dict(eval_reward_function_kwargs)
                        case_reward_function_kwargs["occupant"] = deepcopy(occ)
                        case_eval_reward_function_kwargs["occupant"] = deepcopy(occ)
                    case = run_merlin_sac_no_occupant(
                        output_dir=output_dir,
                        summary_dir=summary_dir,
                        model_dir=model_dir,
                        seed=seed,
                        reward_function=_reward_cls,
                        reward_function_kwargs=case_reward_function_kwargs,
                        eval_reward_function_kwargs=case_eval_reward_function_kwargs,
                        max_steps=args.max_steps,
                        label=case_label,
                        config=config,
                        training_episodes=args.training_episodes,
                        building_name=building_name,
                        building_metadata={
                            **metadata,
                            "occupant": occ_name,
                            "occupant_mode": mode_name,
                            "occupancy_building_name": occupancy_building_name,
                        },
                        occupant=deepcopy(occ),
                        occupant_mode=mode_name,
                        occupancy_model=occupancy_model,
                        occupancy_series=occupancy_series,
                        alpha_occ=args.alpha_occ,
                        T_dyn_init_mode=args.tdyn_init_mode,
                        delta_up=args.delta_up,
                        delta_down=args.delta_down,
                        drift_to_pref=args.drift_to_pref,
                        T_dyn_min=args.tdyn_min,
                        T_dyn_max=args.tdyn_max,
                        save_scenario="occupant_present",
                        jumpstart_controller=args.jumpstart_controller,
                        jumpstart_actions=jumpstart_actions,
                        bc_expert_observations=bc_expert_observations,
                        bc_expert_actions=bc_expert_actions,
                        bc_epochs=args.bc_epochs,
                        bc_batch_size=args.bc_batch_size,
                        bc_lr=args.bc_lr,
                        bc_validation_fraction=args.bc_validation_fraction,
                        bc_early_stopping_patience=args.bc_early_stopping_patience,
                        bc_min_delta=args.bc_min_delta,
                        structured_exploration_start=args.structured_exploration_start,
                        structured_exploration_end=args.structured_exploration_end,
                        structured_exploration_cold_min_action=args.structured_exploration_cold_min_action,
                        structured_exploration_beta_alpha=args.structured_exploration_beta_alpha,
                        structured_exploration_beta_beta=args.structured_exploration_beta_beta,
                        comfort_shield_margin=args.comfort_shield_margin,
                        comfort_shield_min_action=args.comfort_shield_min_action,
                        comfort_shield_kp=args.comfort_shield_kp,
                        comfort_shield_hot_margin=getattr(args, "comfort_shield_hot_margin", None),
                        comfort_shield_hot_max_action=getattr(args, "comfort_shield_hot_max_action", 0.0),
                        comfort_shield_eval=args.comfort_shield_eval,
                        comfort_shield_exploration=args.comfort_shield_exploration,
                        safe_pi_cold_margin=args.safe_pi_cold_margin,
                        safe_pi_hot_margin=args.safe_pi_hot_margin,
                        safe_pi_cold_min_action=args.safe_pi_cold_min_action,
                        safe_pi_cold_kp=args.safe_pi_cold_kp,
                        safe_pi_hot_max_action=args.safe_pi_hot_max_action,
                        safe_pi_random_prob=args.safe_pi_random_prob,
                        safe_pi_random_low=args.safe_pi_random_low,
                        safe_pi_random_high=args.safe_pi_random_high,
                        safe_pi_ramp_limit=args.safe_pi_ramp_limit,
                        residual_safe_pi=args.residual_safe_pi,
                        residual_safe_pi_scale=args.residual_safe_pi_scale,
                        training_temperature_reset=args.training_temperature_reset,
                        training_temperature_reset_cold_margin=args.training_temperature_reset_cold_margin,
                        training_temperature_reset_hot_margin=args.training_temperature_reset_hot_margin,
                        training_temperature_reset_target_margin=args.training_temperature_reset_target_margin,
                        training_temperature_reset_absolute_min=args.training_temperature_reset_absolute_min,
                        training_temperature_reset_absolute_max=args.training_temperature_reset_absolute_max,
                        training_temperature_reset_hidden=args.training_temperature_reset_hidden,
                        training_temperature_reset_drop_transition=args.training_temperature_reset_drop_transition,
                        training_temperature_reset_replay_mode=args.training_temperature_reset_replay_mode,
                        training_temperature_reset_penalty=args.training_temperature_reset_penalty,
                        replay_filter_temperature_min=args.replay_filter_temperature_min,
                        replay_filter_temperature_max=args.replay_filter_temperature_max,
                        replay_filter_temperature_delta_max=args.replay_filter_temperature_delta_max,
                        replay_filter_reward_abs_max=args.replay_filter_reward_abs_max,
                        rc_rollout_horizons=tuple(args.rc_rollout_horizons) if args.rc_rollout_horizons else None,
                        rc_rollout_model_dir=args.rc_rollout_model_dir,
                        train_sim_start=args.train_sim_start,
                        train_sim_end=args.train_sim_end,
                        eval_sim_start=args.eval_sim_start,
                        eval_sim_end=args.eval_sim_end,
                        resume_agent_path=resume_agent_path,
                        resume_episode_offset=args.resume_episode_offset,
                        resume_replay_keep_last=args.resume_replay_keep_last,
                        early_stop_reward_slope=args.early_stop_reward_slope,
                        early_stop_min_episodes=args.early_stop_min_episodes,
                        early_stop_window=args.early_stop_window,
                        early_stop_patience=args.early_stop_patience,
                        early_stop_slope_abs=args.early_stop_slope_abs,
                        checkpoint_select_by_eval=args.checkpoint_select_by_eval,
                        checkpoint_select_metric=args.checkpoint_select_metric,
                    )
                    all_summary_rows.append(case["metrics"])
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
