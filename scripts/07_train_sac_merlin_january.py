#!/usr/bin/env python
"""January PI jump-start for MERLIN-style CityLearn SAC.

This fills CityLearn SACRBC's replay buffer by letting the built-in
PITemperatureController act during January. It intentionally does not run an
SB3 prefill or a separate offline gradient-training phase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sac_merlin_pipeline import MerlinSACConfig, OccupantComfortReward, train_january_pi_jumpstart


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/models/sac_merlin/january_pi_agent.pkl")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=None, help="Short smoke-test limit.")
    parser.add_argument("--reward", default="occupant_comfort", choices=["comfort", "occupant_comfort"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reward_function = OccupantComfortReward if args.reward == "occupant_comfort" else None
    config = MerlinSACConfig(
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
    )
    info = train_january_pi_jumpstart(
        output_path=args.output,
        seed=args.seed,
        reward_function=reward_function,
        max_steps=args.max_steps,
        config=config,
    )
    print("January PI jump-start finished")
    print(f"Steps: {info['steps']}")
    print(f"Replay size: {info['replay_size']}")
    print(f"Normalized/trained: {info['normalized']}")
    print(f"Saved agent: {info['agent_path']}")


if __name__ == "__main__":
    main()
