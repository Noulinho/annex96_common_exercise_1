#!/usr/bin/env python
"""Online Gnu-RL/RLMPC cost-parameter optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rlmpc_pipeline import (
    OccupantComfortReward,
    RLMPCExperiment,
    offline_checkpoint_path,
    run_online_rlmpc_no_occupant,
    run_online_rlmpc_occupants,
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="notebooks/january_pi_dataset.npz")
    parser.add_argument("--output-dir", default="results/raw/rlmpc")
    parser.add_argument("--summary-dir", default="results/summaries/rlmpc")
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
    parser.add_argument("--reward", default="occupant_comfort", choices=["comfort", "occupant_comfort"])
    parser.add_argument("--alpha-occ", type=float, default=0.5)
    parser.add_argument("--tdyn-init-mode", default="schedule", choices=["schedule", "pref"])
    parser.add_argument("--delta-up", type=float, default=0.5)
    parser.add_argument("--delta-down", type=float, default=0.5)
    parser.add_argument("--drift-to-pref", type=float, default=0.0)
    parser.add_argument("--tdyn-min", type=float, default=18.0)
    parser.add_argument("--tdyn-max", type=float, default=26.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Online occupant cases all start from the same offline imitation checkpoint.
    experiment = RLMPCExperiment(controller="rlmpc", scenario="no_occupant")
    checkpoint = Path(args.checkpoint) if args.checkpoint else offline_checkpoint_path(experiment)
    reward_function = OccupantComfortReward if args.reward == "occupant_comfort" else None

    if args.scenario == "no_occupant":
        run_online_rlmpc_no_occupant(
            checkpoint_path=checkpoint,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            summary_dir=args.summary_dir,
            seed=args.seed,
            training_seed=args.training_seed,
            reward_function=reward_function,
            device=args.device,
            sigma=args.sigma,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch,
            lr_ppo=args.lr_ppo,
            gamma=args.gamma,
            max_steps=args.max_steps,
            label="baseline",
        )
    elif args.scenario == "occupant_present":
        run_online_rlmpc_occupants(
            checkpoint_path=checkpoint,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            summary_dir=args.summary_dir,
            seed=args.seed,
            training_seed=args.training_seed,
            reward_function=reward_function,
            device=args.device,
            sigma=args.sigma,
            clip_eps=args.clip_eps,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch,
            lr_ppo=args.lr_ppo,
            gamma=args.gamma,
            max_steps=args.max_steps,
            occupant=args.occupant,
            occupant_mode=args.occupant_mode,
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
