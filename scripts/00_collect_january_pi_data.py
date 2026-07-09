#!/usr/bin/env python
"""Collect January PI-controller datasets for one or more dataset buildings.

These datasets feed both:

- one-state RC fitting for the hand-written MPC, and
- offline imitation learning for the differentiable MPC/RL-MPC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import DATASET_NAME, building_metadata, dataset_dir, format_building_path, resolve_building_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect January PI-controller transition datasets for all selected buildings."
    )
    parser.add_argument(
        "--building",
        default="all",
        help="Building selector: all, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument(
        "--output-template",
        default="results/datasets/january_pi/{building_id}/january_pi_dataset.npz",
        help="Output NPZ path template. Supports {building_id}, {short_name}, {index}, and {building_name}.",
    )
    parser.add_argument(
        "--summary",
        default="results/datasets/january_pi/summary.json",
        help="Summary JSON path for multi-building collection.",
    )
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--start-time-step", type=int, default=0)
    parser.add_argument("--end-time-step", type=int, default=743)
    parser.add_argument("--max-steps", type=int, default=None, help="Short smoke-test limit.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _schema_path() -> Path:
    return dataset_dir(DATASET_NAME) / "schema.json"


def _collect_one(building: dict, args: argparse.Namespace) -> dict:
    import numpy as np
    from citylearn.agents.rbc import PITemperatureController

    from src.data.collect import collect_trajectories
    from src.envs import EnvConfig, make_citylearn_env

    output_path = format_building_path(args.output_template, building)
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and not args.overwrite:
        print(f"Skipping {building['display']} because {output_path} already exists. Use --overwrite to refresh.")
        return {
            **building_metadata(building),
            "dataset_path": str(output_path),
            "metadata_path": str(metadata_path),
            "status": "skipped_existing",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed + 1000 * (int(building["index"]) - 1))
    config = EnvConfig(
        schema_path=_schema_path(),
        dataset_dir=dataset_dir(DATASET_NAME),
        buildings=[str(building["name"])],
        start_time_step=int(args.start_time_step),
        end_time_step=int(args.end_time_step),
        random_seed=seed,
        central_agent=True,
        active_actions=("heating_device",),
    )
    env = make_citylearn_env(config)
    controller = PITemperatureController(env)

    if args.max_steps is None:
        _, dataset = collect_trajectories(env, controller, save_path=output_path, seed=seed)
    else:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = controller.predict(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_obs)
            dones.append(done)
            obs = next_obs
            steps += 1
            if steps >= int(args.max_steps):
                break
        dataset = {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }
        np.savez_compressed(output_path, **dataset)

    row = {
        **building_metadata(building),
        "dataset_path": str(output_path),
        "metadata_path": str(metadata_path),
        "status": "collected",
        "seed": seed,
        "start_time_step": int(args.start_time_step),
        "end_time_step": int(args.end_time_step),
        "steps": int(dataset["states"].shape[0]),
        "state_shape": list(dataset["states"].shape),
        "action_shape": list(dataset["actions"].shape),
        "reward_total": float(np.asarray(dataset["rewards"], dtype=np.float32).sum()),
    }
    metadata_path.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"{building['display']}: saved {row['steps']} transitions to {output_path} "
        f"(reward_total={row['reward_total']:.3f})"
    )
    return row


def main() -> None:
    args = build_parser().parse_args()
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    rows = [_collect_one(building, args) for building in buildings]
    if len(rows) > 1:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Saved collection summary to {summary_path}")


if __name__ == "__main__":
    main()
