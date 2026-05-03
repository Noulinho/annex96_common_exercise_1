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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sac_merlin_pipeline import (
    MerlinSACConfig,
    reward_function_from_mode,
    run_merlin_sac_no_occupant,
    run_merlin_sac_occupants,
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
        choices=["without_tdyn", "bspline_tdyn", "both", "bspline_observation"],
        help="Kept for backward compatibility; SAC occupant runs now use B-spline forecasts as observations only.",
    )
    parser.add_argument("--output-dir", default="results/raw/sac_merlin")
    parser.add_argument("--summary-dir", default="results/summaries/sac_merlin")
    parser.add_argument("--model-dir", default="results/models/sac_merlin")
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
            "both",
            "all",
        ],
    )
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=2)
    parser.add_argument("--training-episodes", type=int, default=10)
    parser.add_argument("--alpha-occ", type=float, default=0.5)
    parser.add_argument("--tdyn-init-mode", default="schedule", choices=["schedule", "pref"])
    parser.add_argument("--delta-up", type=float, default=0.5)
    parser.add_argument("--delta-down", type=float, default=0.5)
    parser.add_argument("--drift-to-pref", type=float, default=0.01)
    parser.add_argument("--tdyn-min", type=float, default=18.0)
    parser.add_argument("--tdyn-max", type=float, default=26.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = MerlinSACConfig(
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
    )
    if args.scenario == "no_occupant":
        run_merlin_sac_no_occupant(
            output_dir=args.output_dir,
            summary_dir=args.summary_dir,
            model_dir=args.model_dir,
            seed=args.seed,
            reward_function=reward_function_from_mode("comfort"),
            max_steps=args.max_steps,
            label=args.label,
            config=config,
            training_episodes=args.training_episodes,
        )
    elif args.scenario == "occupant_present":
        run_merlin_sac_occupants(
            output_dir=args.output_dir,
            summary_dir=args.summary_dir,
            model_dir=args.model_dir,
            seed=args.seed,
            training_seed=args.training_seed,
            max_steps=args.max_steps,
            occupant=args.occupant,
            occupant_mode=args.occupant_mode,
            reward_mode=args.reward,
            config=config,
            training_episodes=args.training_episodes,
            alpha_occ=args.alpha_occ,
            T_dyn_init_mode=args.tdyn_init_mode,
            delta_up=args.delta_up,
            delta_down=args.delta_down,
            drift_to_pref=args.drift_to_pref,
            T_dyn_min=args.tdyn_min,
            T_dyn_max=args.tdyn_max,
        )
    else:
        raise SystemExit("occupant_present_peak_flattening is not wired yet.")


if __name__ == "__main__":
    main()
