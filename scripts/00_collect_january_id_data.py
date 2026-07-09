#!/usr/bin/env python
"""Collect richer January identification rollouts for fragile RC fits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import DATASET_NAME, building_metadata, dataset_dir, format_building_path, resolve_building_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect January PI rollouts with block-wise safe excitation. The dataset keeps the same "
            "(state, action, reward, next_state, done) NPZ layout as the original January PI data."
        )
    )
    parser.add_argument(
        "--building",
        default="5,8,11,14,19",
        help="Building selector: all, comma list, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument(
        "--output-template",
        default="results/datasets/january_id/{building_id}/january_pi_prbs_dataset.npz",
        help="Output NPZ path template. Supports {building_id}, {short_name}, {index}, and {building_name}.",
    )
    parser.add_argument("--summary", default="results/datasets/january_id/summary.json")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--start-time-step", type=int, default=0)
    parser.add_argument("--end-time-step", type=int, default=743)
    parser.add_argument("--max-steps", type=int, default=None, help="Short smoke-test limit per episode.")
    parser.add_argument("--block-hours", type=int, default=6)
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--comfort-guard", type=float, default=2.5)
    parser.add_argument("--cold-recovery-action", type=float, default=0.8)
    parser.add_argument("--hot-relief-action", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _schema_path() -> Path:
    return dataset_dir(DATASET_NAME) / "schema.json"


def _as_scalar_action(action) -> float:
    import numpy as np

    arr = np.asarray(action, dtype=float)
    return float(arr.reshape(-1)[0])


def _replace_first_action(action, value: float):
    import numpy as np

    arr = np.asarray(action, dtype=float).copy()
    arr.reshape(-1)[0] = float(np.clip(value, 0.0, 1.0))
    return arr


class ExcitedPIController:
    """PI controller plus safe block excitation for RC identification."""

    def __init__(
        self,
        env,
        *,
        seed: int,
        block_hours: int,
        amplitude: float,
        comfort_guard: float,
        cold_recovery_action: float,
        hot_relief_action: float,
    ) -> None:
        import numpy as np
        from citylearn.agents.rbc import PITemperatureController

        self.pi = PITemperatureController(env)
        self.rng = np.random.default_rng(seed)
        self.block_hours = max(int(block_hours), 1)
        self.amplitude = float(amplitude)
        self.comfort_guard = float(comfort_guard)
        self.cold_recovery_action = float(cold_recovery_action)
        self.hot_relief_action = float(hot_relief_action)
        names = list(env.unwrapped.buildings[0].active_observations)
        self.indoor_idx = names.index("indoor_dry_bulb_temperature")
        self.heat_sp_idx = names.index("indoor_dry_bulb_temperature_heating_set_point")
        self.step = 0
        self._block_signal: dict[int, float] = {}

    def _signal(self, step: int) -> float:
        block = int(step // self.block_hours)
        if block not in self._block_signal:
            levels = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self._block_signal[block] = float(self.amplitude * self.rng.choice(levels))
        return self._block_signal[block]

    def predict(self, obs):
        import numpy as np

        base = self.pi.predict(obs)
        base_scalar = _as_scalar_action(base)
        obs_arr = np.asarray(obs, dtype=float)
        obs_1d = obs_arr[0] if obs_arr.ndim == 2 else obs_arr.reshape(-1)
        tin = float(obs_1d[self.indoor_idx])
        tsp = float(obs_1d[self.heat_sp_idx])

        action = base_scalar + self._signal(self.step)
        if tin < tsp - self.comfort_guard:
            action = max(action, base_scalar, self.cold_recovery_action)
        elif tin > tsp + self.comfort_guard:
            action = min(action, base_scalar, self.hot_relief_action)

        self.step += 1
        return _replace_first_action(base, float(np.clip(action, 0.0, 1.0)))


def _collect_one(building: dict, args: argparse.Namespace) -> dict:
    import numpy as np

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
    states, actions, rewards, next_states, dones = [], [], [], [], []
    base_seed = int(args.seed + 1000 * (int(building["index"]) - 1))

    for episode in range(int(args.episodes)):
        episode_seed = int(base_seed + episode)
        config = EnvConfig(
            schema_path=_schema_path(),
            dataset_dir=dataset_dir(DATASET_NAME),
            buildings=[str(building["name"])],
            start_time_step=int(args.start_time_step),
            end_time_step=int(args.end_time_step),
            random_seed=episode_seed,
            central_agent=True,
            active_actions=("heating_device",),
        )
        env = make_citylearn_env(config)
        controller = ExcitedPIController(
            env,
            seed=episode_seed,
            block_hours=int(args.block_hours),
            amplitude=float(args.amplitude),
            comfort_guard=float(args.comfort_guard),
            cold_recovery_action=float(args.cold_recovery_action),
            hot_relief_action=float(args.hot_relief_action),
        )
        obs, _ = env.reset(seed=episode_seed)
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
            if args.max_steps is not None and steps >= int(args.max_steps):
                break

    dataset = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }
    np.savez_compressed(output_path, **dataset)

    action_values = np.asarray(dataset["actions"], dtype=float).reshape(len(states), -1)[:, 0]
    row = {
        **building_metadata(building),
        "dataset_path": str(output_path),
        "metadata_path": str(metadata_path),
        "status": "collected",
        "seed": base_seed,
        "episodes": int(args.episodes),
        "start_time_step": int(args.start_time_step),
        "end_time_step": int(args.end_time_step),
        "steps": int(dataset["states"].shape[0]),
        "state_shape": list(dataset["states"].shape),
        "action_shape": list(dataset["actions"].shape),
        "action_mean": float(np.mean(action_values)),
        "action_std": float(np.std(action_values)),
        "action_min": float(np.min(action_values)),
        "action_max": float(np.max(action_values)),
        "reward_total": float(np.asarray(dataset["rewards"], dtype=np.float32).sum()),
        "block_hours": int(args.block_hours),
        "amplitude": float(args.amplitude),
        "comfort_guard": float(args.comfort_guard),
    }
    metadata_path.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"{building['display']}: saved {row['steps']} excited-ID transitions to {output_path} "
        f"(action mean={row['action_mean']:.3f}, std={row['action_std']:.3f})"
    )
    return row


def main() -> None:
    args = build_parser().parse_args()
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    rows = [_collect_one(building, args) for building in buildings]
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Saved ID collection summary to {summary_path}")


if __name__ == "__main__":
    main()
