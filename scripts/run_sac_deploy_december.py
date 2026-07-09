#!/usr/bin/env python
"""Deploy a trained SAC agent on December without retraining.

Loads the agent state dict saved by 08_run_sac_merlin_online.py and runs a
deterministic evaluation on the December period (steps 8016–8759).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import DATASET_NAME, building_metadata, format_building_path, resolve_building_entries

DEC_START = 8016
DEC_END = 8759


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--train-label", required=True)
    p.add_argument("--deploy-label", required=True)
    p.add_argument("--building", default="all")
    p.add_argument("--output-dir", default="results/raw/sac_merlin")
    p.add_argument("--summary-dir", default="results/summaries/sac_merlin")
    p.add_argument("--model-dir", default="results/models/sac_merlin")
    p.add_argument("--seed", type=int, default=49)
    p.add_argument("--action-scaling-coefficient", type=float, default=1.0)
    # Reward (must match training)
    p.add_argument("--comfort-mode", default="time_bias_guard")
    p.add_argument("--comfort-weight", type=float, default=0.5)
    p.add_argument("--comfort-track-weight", type=float, default=3.0)
    p.add_argument("--comfort-cold-weight", type=float, default=60.0)
    p.add_argument("--comfort-hot-weight", type=float, default=15.0)
    p.add_argument("--comfort-track-scale", type=float, default=2.0)
    p.add_argument("--comfort-cold-scale", type=float, default=1.5)
    p.add_argument("--comfort-hot-scale", type=float, default=2.0)
    p.add_argument("--time-bias-amplitude", type=float, default=1.0)
    p.add_argument("--time-bias-guard-band", type=float, default=2.0)
    p.add_argument("--energy-weight", type=float, default=0.5)
    p.add_argument("--shift-weight", type=float, default=0.5)
    p.add_argument("--power-scale", type=float, default=5.0)
    p.add_argument("--ramp-weight", type=float, default=0.0)
    return p


def _restore_agent(payload: dict, agent) -> None:
    """Restore network weights and normalization from saved payload into agent."""
    import torch

    for i, net in enumerate(agent.policy_net):
        net.load_state_dict(payload["policy_net"][i])
    for i in range(len(agent.soft_q_net1)):
        agent.soft_q_net1[i].load_state_dict(payload["soft_q_net1"][i])
        agent.soft_q_net2[i].load_state_dict(payload["soft_q_net2"][i])
        agent.target_soft_q_net1[i].load_state_dict(payload["target_soft_q_net1"][i])
        agent.target_soft_q_net2[i].load_state_dict(payload["target_soft_q_net2"][i])

    agent.norm_mean = payload["norm_mean"]
    agent.norm_std = payload["norm_std"]
    agent.r_norm_mean = payload["r_norm_mean"]
    agent.r_norm_std = payload["r_norm_std"]
    agent.normalized = list(payload["normalized"])
    agent.time_step = int(payload["time_step"])
    agent.standardize_start_time_step = int(payload["standardize_start_time_step"])
    agent.end_exploration_time_step = int(payload["end_exploration_time_step"])

    if payload.get("automatic_entropy_tuning") and hasattr(agent, "log_alpha"):
        import math
        for i, val in enumerate(payload.get("log_alpha", [])):
            if val is not None and i < len(agent.log_alpha) and agent.log_alpha[i] is not None:
                with torch.no_grad():
                    agent.log_alpha[i].fill_(float(val))
    alpha_val = payload.get("alpha")
    if alpha_val is not None and not math.isnan(float(alpha_val)):
        agent.alpha = float(alpha_val)


def main() -> None:
    args = build_parser().parse_args()

    import math
    from src.sac_merlin_pipeline import (
        MerlinSACConfig,
        _evaluate_trained_agent_on_february,
        build_merlin_sac_agent,
        make_env,
        save_result,
        set_experiment_seed,
        summarize_run,
    )
    from src.rewards import BalancedComfortEnergyReward, time_bias_setpoint_profile, load_simple_v4_price_profile

    buildings = resolve_building_entries(args.building, DATASET_NAME)

    setpoint_bias_profile = time_bias_setpoint_profile(
        simulation_start=DEC_START,
        simulation_end=DEC_END,
        amplitude=args.time_bias_amplitude,
    )
    shift_profile = load_simple_v4_price_profile(simulation_start=DEC_START, exp_beta=2.0)

    reward_kwargs = dict(
        comfort_weight=args.comfort_weight,
        energy_weight=args.energy_weight,
        comfort_mode=args.comfort_mode,
        comfort_track_weight=args.comfort_track_weight,
        comfort_cold_weight=args.comfort_cold_weight,
        comfort_hot_weight=args.comfort_hot_weight,
        comfort_track_scale=args.comfort_track_scale,
        comfort_cold_scale=args.comfort_cold_scale,
        comfort_hot_scale=args.comfort_hot_scale,
        time_bias_guard_band=args.time_bias_guard_band,
        ramp_weight=args.ramp_weight,
        shift_weight=args.shift_weight,
        power_scale=args.power_scale,
        setpoint_bias_profile=setpoint_bias_profile,
        shift_profile=shift_profile,
    )

    config = MerlinSACConfig(action_scaling_coefficient=args.action_scaling_coefficient)

    all_rows = []
    for building in buildings:
        building_name = str(building["name"])
        seed = int(args.seed + 1000 * (int(building["index"]) - 1))
        set_experiment_seed(seed)

        pkl_path = (
            Path(args.model_dir) / building["result_id"]
            / f"{args.train_label}_jan_feb_online_agent.pkl"
        )
        if not pkl_path.exists():
            print(f"SKIP {building['display']}: {pkl_path} not found")
            continue

        print(f"\n=== {building['display']} ===")
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)

        # Build a fresh env just to size the agent networks
        probe_env = make_env(
            sim_start=DEC_START,
            sim_end=DEC_END,
            seed=seed,
            reward_function=BalancedComfortEnergyReward,
            reward_function_kwargs=reward_kwargs,
            building_name=building_name,
        )
        exploration_steps = int(payload["standardize_start_time_step"])
        agent = build_merlin_sac_agent(probe_env, exploration_steps=exploration_steps, seed=seed, config=config)
        _restore_agent(payload, agent)

        result = _evaluate_trained_agent_on_february(
            agent,
            seed=seed,
            reward_function=BalancedComfortEnergyReward,
            reward_function_kwargs=reward_kwargs,
            label=args.deploy_label,
            building_name=building_name,
            sim_start=DEC_START,
            sim_end=DEC_END,
        )

        meta = building_metadata(building)
        metrics = summarize_run(args.deploy_label, result, online_info=meta)
        save_result(
            result,
            output_dir=Path(args.output_dir) / building["result_id"],
            summary_dir=Path(args.summary_dir) / building["result_id"],
            scenario="no_occupant",
            label=args.deploy_label,
            metrics=metrics,
        )
        all_rows.append(metrics)
        print(f"  discomfort={100*metrics['discomfort_proportion']:.2f}%  cold={100*metrics['discomfort_cold_proportion']:.2f}%")

    if len(all_rows) > 1:
        import pandas as pd
        combined = pd.DataFrame(all_rows)
        out = Path(args.summary_dir) / "no_occupant_summary_all_buildings.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            existing = pd.read_csv(out)
            existing = existing[existing["case"] != args.deploy_label]
            combined = pd.concat([existing, combined], ignore_index=True)
        combined.to_csv(out, index=False)
        print(f"\nSaved combined summary to {out}")


if __name__ == "__main__":
    main()
