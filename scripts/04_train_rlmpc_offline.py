#!/usr/bin/env python
"""Offline imitation stage for Gnu-RL/RLMPC.

This is the script version of the working notebook section:

1. load the January PI rollout dataset,
2. train the differentiable MPC by imitating actions and one-step temperature,
3. save the learned DiffMPC checkpoint plus normalization bundle,
4. optionally run the pretrained raw policy on January as a sanity check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rlmpc_pipeline import (
    OccupantComfortReward,
    RLMPCExperiment,
    offline_bundle_path,
    offline_checkpoint_path,
    run_january_raw_rollout,
    train_offline_diffmpc,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="no_occupant",
        choices=["no_occupant", "occupant_present", "occupant_present_peak_flattening"],
    )
    parser.add_argument("--occupant", default=None)
    parser.add_argument(
        "--tdyn-variant",
        default=None,
        choices=["feedback_only", "present_only_no_bspline", "bspline_dynamic_comfort"],
    )
    parser.add_argument("--dataset", default="notebooks/january_pi_dataset.npz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="Training/shuffle seed, matching notebook rng=0 default.")
    parser.add_argument("--rollout-seed", type=int, default=49)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-dyn", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--rho-u", type=float, default=1.0)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=1.0)
    parser.add_argument("--q-reg", type=float, default=1e-4)
    parser.add_argument("--qp-max-iter", type=int, default=200)
    parser.add_argument("--qp-eps", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=None, help="Short smoke-test subset of the dataset.")
    parser.add_argument("--observation-preset", default="auto", choices=["auto", "notebook_mpc_28"])
    parser.add_argument(
        "--reward",
        default="occupant_comfort",
        choices=["comfort", "occupant_comfort"],
        help="Reward used only for the optional raw January rollout check.",
    )
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = RLMPCExperiment(
        controller="rlmpc",
        scenario=args.scenario,
        occupant_label=args.occupant,
        tdyn_variant=args.tdyn_variant,
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else offline_checkpoint_path(experiment)
    bundle_path = Path(args.bundle) if args.bundle else offline_bundle_path(experiment)

    train_offline_diffmpc(
        dataset_path=args.dataset,
        checkpoint_path=checkpoint_path,
        bundle_path=bundle_path,
        horizon=args.horizon,
        eta=args.eta,
        rho_u=args.rho_u,
        u_min=args.u_min,
        u_max=args.u_max,
        q_reg=args.q_reg,
        qp_max_iter=args.qp_max_iter,
        qp_eps=args.qp_eps,
        epochs=args.epochs,
        lr=args.lr,
        lambda_dyn=args.lambda_dyn,
        batch_size=args.batch_size,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        max_samples=args.max_samples,
        observation_preset=args.observation_preset,
    )

    if not args.skip_rollout:
        reward_function = OccupantComfortReward if args.reward == "occupant_comfort" else None
        run_january_raw_rollout(
            checkpoint_path=checkpoint_path,
            seed=args.rollout_seed,
            reward_function=reward_function,
            device=args.device,
            max_steps=args.max_rollout_steps,
        )


if __name__ == "__main__":
    main()
