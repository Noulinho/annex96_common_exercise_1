#!/usr/bin/env python
"""January PI jump-start for MERLIN-style CityLearn SAC.

This fills CityLearn SACRBC's replay buffer by letting the built-in
PITemperatureController act during January. It intentionally does not run an
SB3 prefill or a separate offline gradient-training phase.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import DATASET_NAME, building_metadata, format_building_path, resolve_building_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--building",
        default="all",
        help="Building selector: all, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Single output path or template. For multiple buildings, include placeholders or use the default.",
    )
    parser.add_argument(
        "--output-template",
        default="results/models/sac_merlin/{building_id}/january_pi_agent.pkl",
        help="Per-building output path template.",
    )
    parser.add_argument("--summary", default="results/models/sac_merlin/january_jumpstart_summary.csv")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=None, help="Short smoke-test limit.")
    parser.add_argument("--reward", default="occupant_comfort", choices=["comfort", "occupant_comfort"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=2)
    return parser


def _has_format_fields(path: str | Path) -> bool:
    text = str(path)
    return "{" in text and "}" in text


def _resolve_output_path(args: argparse.Namespace, building: dict, multiple_buildings: bool) -> Path:
    value = args.output_template if args.output is None else args.output
    if multiple_buildings and args.output is not None and not _has_format_fields(args.output):
        raise ValueError(
            "--output points to one fixed path while --building selects multiple buildings. "
            "Use --output-template or include a placeholder such as {building_id}."
        )
    return format_building_path(value, building)


def main() -> None:
    args = build_parser().parse_args()
    from src.sac_merlin_pipeline import MerlinSACConfig, OccupantComfortReward, train_january_pi_jumpstart

    reward_function = OccupantComfortReward if args.reward == "occupant_comfort" else None
    config = MerlinSACConfig(
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        update_per_time_step=args.updates_per_step,
    )
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    multiple = len(buildings) > 1
    rows = []

    for building in buildings:
        output_path = _resolve_output_path(args, building, multiple)
        seed = int(args.seed + 1000 * (int(building["index"]) - 1))
        print(f"\n=== {building['display']}: {building['name']} ===")
        info = train_january_pi_jumpstart(
            output_path=output_path,
            seed=seed,
            reward_function=reward_function,
            max_steps=args.max_steps,
            config=config,
            building_name=str(building["name"]),
        )
        print("January PI jump-start finished")
        print(f"Steps: {info['steps']}")
        print(f"Replay size: {info['replay_size']}")
        print(f"Normalized/trained: {info['normalized']}")
        print(f"Saved agent: {info['agent_path']}")
        rows.append(
            {
                **building_metadata(building),
                "seed": seed,
                "steps": info["steps"],
                "reward_total": info["reward_total"],
                "replay_size": info["replay_size"],
                "normalized": info["normalized"],
                "agent_path": info["agent_path"],
                "model_path": info["model_path"],
            }
        )

    if rows:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved January jump-start summary to {summary_path}")


if __name__ == "__main__":
    main()
