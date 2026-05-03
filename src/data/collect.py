"""Rollout collection helpers used by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def run_env(env: Any, agent: Any) -> Any:
    obs, _ = env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        actions = agent.predict(obs)
        obs, _, terminated, truncated, _ = env.step(actions)

    return env


def collect_trajectories(env: Any, agent: Any, save_path: str | Path | None = None, seed: int | None = None):
    """Collect `(state, action, reward, next_state, done)` transitions."""

    states, actions, rewards, next_states, dones = [], [], [], [], []
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(done)

        obs = next_obs

    dataset = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }

    if save_path is not None:
        np.savez_compressed(save_path, **dataset)

    return env, dataset

