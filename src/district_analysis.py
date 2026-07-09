"""District-level analysis helpers for CE1 controller comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries, dataset_dir, schema_path


FEB_START = 744
FEB_END = 1415


def run_citylearn_baseline_district(
    *,
    output_dir: str | Path = "results/raw/citylearn_baseline/no_occupant/baseline",
    summary_dir: str | Path = "results/summaries/citylearn_baseline/no_occupant",
    dataset_name: str = DATASET_NAME,
    simulation_start: int = FEB_START,
    simulation_end: int = FEB_END,
    seed: int = 49,
) -> dict[str, Path]:
    """Run CityLearn's no-control BaselineAgent for the full district."""

    from citylearn.agents.base import BaselineAgent
    from citylearn.citylearn import CityLearnEnv

    output_dir = Path(output_dir)
    summary_dir = Path(summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    data_dir = dataset_dir(dataset_name)
    env = CityLearnEnv(
        str(schema_path(dataset_name)),
        root_directory=str(data_dir),
        central_agent=True,
        simulation_start_time_step=int(simulation_start),
        simulation_end_time_step=int(simulation_end),
        random_seed=int(seed),
    )
    agent = BaselineAgent(env)
    observations, _ = env.reset(seed=seed)
    terminated = truncated = False
    rewards: list[float] = []

    while not (terminated or truncated):
        actions = agent.predict(observations)
        observations, reward, terminated, truncated, _ = env.step(actions)
        rewards.append(float(np.sum(reward)))

    raw_district_load = np.asarray(env.net_electricity_consumption, dtype=float)
    raw_target = np.asarray(env.buildings[0].district_load_target.district_load_target, dtype=float)
    reward_array = np.asarray(rewards, dtype=float)
    n = min(len(raw_district_load), len(raw_target), len(reward_array))
    district_load = raw_district_load[:n]
    target = raw_target[:n]
    reward_array = reward_array[:n]
    history = pd.DataFrame(
        {
            "time_step": np.arange(n, dtype=int),
            "net_electricity_consumption": district_load,
            "district_target": target,
            "reward": reward_array,
        }
    )

    optional_series = {
        "net_electricity_consumption_without_storage": "net_electricity_consumption_without_storage",
        "net_electricity_consumption_without_storage_and_partial_load": (
            "net_electricity_consumption_without_storage_and_partial_load"
        ),
        "net_electricity_consumption_without_storage_and_partial_load_and_pv": (
            "net_electricity_consumption_without_storage_and_partial_load_and_pv"
        ),
    }
    for column, attr in optional_series.items():
        try:
            values = np.asarray(getattr(env, attr), dtype=float)
        except Exception:
            continue
        history[column] = values[:n]

    rollout_path = output_dir / "district_rollout.csv"
    kpis_path = output_dir / "kpis.csv"
    summary_path = summary_dir / "baseline_metrics.json"
    history.to_csv(rollout_path, index=False)
    kpis = env.evaluate().copy()
    kpis.to_csv(kpis_path, index=False)
    metrics = {
        "case": "citylearn_baseline",
        "reward_total": float(np.sum(rewards)),
        "district_energy_sum": float(np.sum(district_load)),
        "district_peak": float(np.max(district_load)) if n else 0.0,
        "district_target_rmse_kwh": float(np.sqrt(np.mean((district_load - target) ** 2))) if n else 0.0,
        "district_target_mae_kwh": float(np.mean(np.abs(district_load - target))) if n else 0.0,
        "district_above_target_sum_kwh": float(np.sum(np.maximum(district_load - target, 0.0))) if n else 0.0,
        "simulation_start": int(simulation_start),
        "simulation_end": int(simulation_end),
        "dataset": dataset_name,
    }
    summary_path.write_text(__import__("json").dumps(metrics, indent=2) + "\n")
    return {"rollout": rollout_path, "kpis": kpis_path, "summary": summary_path}


def load_or_run_citylearn_baseline_district(
    *,
    output_dir: str | Path = "results/raw/citylearn_baseline/no_occupant/baseline",
    summary_dir: str | Path = "results/summaries/citylearn_baseline/no_occupant",
    force: bool = False,
    **kwargs,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    rollout_path = output_dir / "district_rollout.csv"
    if force or not rollout_path.exists():
        run_citylearn_baseline_district(output_dir=output_dir, summary_dir=summary_dir, **kwargs)
    return pd.read_csv(rollout_path)


def aggregate_controller_rollouts(
    *,
    controller: str,
    case: str,
    scenario: str = "no_occupant",
    results_root: str | Path = "results",
    buildings: Iterable[dict] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Sum per-building rollout load into one district-level profile."""

    results_root = Path(results_root)
    buildings = list(dataset_building_entries() if buildings is None else buildings)
    series = []
    missing: list[Path] = []

    for building in buildings:
        path = results_root / "raw" / controller / building["result_id"] / scenario / case / "rollout.csv"
        if not path.exists():
            missing.append(path)
            continue
        frame = pd.read_csv(path)
        if "net_electricity_consumption" not in frame.columns:
            missing.append(path)
            continue
        series.append(pd.to_numeric(frame["net_electricity_consumption"], errors="coerce").fillna(0.0).to_numpy())

    if missing or len(series) != len(buildings):
        return pd.DataFrame(), missing

    n = min(len(s) for s in series)
    district_load = np.sum(np.stack([s[:n] for s in series], axis=0), axis=0)
    return pd.DataFrame(
        {
            "time_step": np.arange(n, dtype=int),
            "net_electricity_consumption": district_load,
        }
    ), []


def district_target_kpis(load, target) -> dict[str, float]:
    load = np.asarray(load, dtype=float)
    target = np.asarray(target, dtype=float)
    n = min(len(load), len(target))
    load = load[:n]
    target = target[:n]
    error = load - target
    above = np.maximum(error, 0.0)
    return {
        "energy_sum_kwh": float(np.sum(load)),
        "peak_kwh": float(np.max(load)) if n else 0.0,
        "target_rmse_kwh": float(np.sqrt(np.mean(error**2))) if n else 0.0,
        "target_mae_kwh": float(np.mean(np.abs(error))) if n else 0.0,
        "above_target_sum_kwh": float(np.sum(above)) if n else 0.0,
        "above_target_peak_kwh": float(np.max(above)) if n else 0.0,
        "above_target_fraction": float(np.mean(above > 1e-9)) if n else 0.0,
        "ramping_abs_sum_kwh": float(np.sum(np.abs(np.diff(load)))) if n > 1 else 0.0,
    }
