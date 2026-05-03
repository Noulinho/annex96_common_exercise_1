"""MERLIN-style SAC jump-start pipeline.

This module follows the MERLIN adaptation pattern using CityLearn's SACRBC:

1. Temperature PI controls the exploration/jump-start period.
2. SAC stores those transitions in its replay buffer.
3. SAC switches to its learned policy and trains online with CityLearn's
   built-in SAC update loop.

No SB3 replay injection and no separate offline gradient pretraining are used.
"""

from __future__ import annotations

import json
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Union

import numpy as np
from citylearn.reward_function import ComfortReward

from src.envs import EnvConfig, make_citylearn_env
from src.occupants.comfort import OccupantFeedbackReward
from src.occupants.wrappers import BSplineOccupancyObservationWrapper, OccupantEnvWrapper
from src.rlmpc_pipeline import (
    fit_bspline_occupancy_model,
    fitted_occupants,
    forecast_occupancy_from_bsplines,
)


DATASET_NAME = "annex96_ce1_vt_neighborhood"
CONTROL_BUILDING_NAME = "resstock-amy2018-2021-release-1-247942"
JAN_START = 0
JAN_END = 743
FEB_START = 744
FEB_END = 1415


@dataclass(frozen=True)
class MerlinSACConfig:
    hidden_dimension: tuple[int, int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    lr: float = 0.005
    alpha: float = 0.8
    batch_size: int = 256
    replay_buffer_capacity: int = 100_000
    action_scaling_coefficient: float = 0.5
    reward_scaling: float = 5.0
    update_per_time_step: int = 2
    pi_kp: float = 0.2
    pi_ki: float = 0.005
    pi_deadband: float = 0.5


class FixedComfortReward(ComfortReward):
    """Four-region comfort reward centered on a fixed occupant T_pref."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        occupant=None,
        band: float = None,
        lower_exponent: float = None,
        higher_exponent: float = None,
    ):
        super().__init__(
            env_metadata=env_metadata,
            band=band,
            lower_exponent=lower_exponent,
            higher_exponent=higher_exponent,
        )
        self.occupant = deepcopy(occupant)

    def _pref(self, default_sp: float) -> float:
        if self.occupant is None:
            return float(default_sp)
        if hasattr(self.occupant, "T_pref"):
            return float(self.occupant.T_pref)
        if hasattr(self.occupant, "preferred_temperature"):
            return float(self.occupant.preferred_temperature)
        return float(default_sp)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        for observation in observations:
            indoor_temperature = float(observation["indoor_dry_bulb_temperature"])

            scheduled_sp = float(observation.get("indoor_dry_bulb_temperature_heating_set_point", 21.0))
            set_point = self._pref(scheduled_sp)
            band = self.band if self.band is not None else float(observation["comfort_band"])
            lower_bound = set_point - band
            upper_bound = set_point + band
            delta = abs(indoor_temperature - set_point)

            if indoor_temperature < lower_bound:
                reward = -(delta**self.lower_exponent)
            elif lower_bound <= indoor_temperature < set_point:
                reward = 0.0
            elif set_point <= indoor_temperature <= upper_bound:
                reward = -delta
            else:
                reward = -(delta**self.higher_exponent)

            reward_list.append(float(reward))

        return [sum(reward_list)] if self.central_agent else reward_list


class DynamicComfortReward(ComfortReward):
    """ComfortReward with a persistent occupant-feedback-driven T_dyn target."""

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        occupant=None,
        band: float = None,
        lower_exponent: float = None,
        higher_exponent: float = None,
        t_dyn_init_mode: str = "schedule",
        delta_up: float = 0.5,
        delta_down: float = 0.5,
        drift_to_pref: float = 0.01,
        t_dyn_min: float = 18.0,
        t_dyn_max: float = 26.0,
    ):
        super().__init__(
            env_metadata=env_metadata,
            band=band,
            lower_exponent=lower_exponent,
            higher_exponent=higher_exponent,
        )
        self.occupant = deepcopy(occupant)
        self.t_dyn_init_mode = t_dyn_init_mode
        self.delta_up = float(delta_up)
        self.delta_down = float(delta_down)
        self.drift_to_pref = float(drift_to_pref)
        self.t_dyn_min = float(t_dyn_min)
        self.t_dyn_max = float(t_dyn_max)
        self._t_dyn = None

        if self.env_metadata is not None and self.env_metadata.get("buildings") is not None:
            self.reset()

    def reset(self):
        n_buildings = 0 if self.env_metadata is None else len(self.env_metadata["buildings"])
        self._t_dyn = [None] * n_buildings

    def _pref(self, default_sp: float) -> float:
        if self.occupant is None:
            return float(default_sp)
        return float(self.occupant.T_pref)

    def _init_t_dyn_if_needed(self, i: int, observation: Mapping[str, Union[int, float]]):
        if self._t_dyn[i] is not None:
            return

        heating_sp = float(observation.get("indoor_dry_bulb_temperature_heating_set_point", 21.0))
        if self.t_dyn_init_mode == "schedule":
            t_dyn = heating_sp
        elif self.t_dyn_init_mode == "pref":
            t_dyn = self._pref(heating_sp)
        else:
            raise ValueError("t_dyn_init_mode must be 'schedule' or 'pref'")

        self._t_dyn[i] = float(np.clip(t_dyn, self.t_dyn_min, self.t_dyn_max))

    def _update_t_dyn(self, i: int, observation: Mapping[str, Union[int, float]]):
        self._init_t_dyn_if_needed(i, observation)

        heating_sp = float(observation.get("indoor_dry_bulb_temperature_heating_set_point", 21.0))
        t_pref = self._pref(heating_sp)

        if self.drift_to_pref > 0.0:
            self._t_dyn[i] = (1.0 - self.drift_to_pref) * self._t_dyn[i] + self.drift_to_pref * t_pref

        feedback = float(observation.get("occupant_feedback_signal", 0.0))
        if feedback > 0.0:
            self._t_dyn[i] += self.delta_up
        elif feedback < 0.0:
            self._t_dyn[i] -= self.delta_down

        self._t_dyn[i] = float(np.clip(self._t_dyn[i], self.t_dyn_min, self.t_dyn_max))

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        if self._t_dyn is None or len(self._t_dyn) != len(observations):
            self.reset()

        reward_list = []
        for i, observation in enumerate(observations):
            self._update_t_dyn(i, observation)

            heating_demand = float(observation.get("heating_demand", 0.0))
            cooling_demand = float(observation.get("cooling_demand", 0.0))
            heating = heating_demand > cooling_demand
            hvac_mode = int(observation["hvac_mode"])
            indoor_temperature = float(observation["indoor_dry_bulb_temperature"])

            set_point = float(self._t_dyn[i])
            band = self.band if self.band is not None else float(observation["comfort_band"])
            lower_bound = set_point - band
            upper_bound = set_point + band
            delta = abs(indoor_temperature - set_point)

            if hvac_mode in [1, 2]:
                if indoor_temperature < lower_bound:
                    exponent = self.lower_exponent if hvac_mode == 2 else self.higher_exponent
                    reward = -(delta**exponent)
                elif lower_bound <= indoor_temperature < set_point:
                    reward = 0.0 if heating else -delta
                elif set_point <= indoor_temperature <= upper_bound:
                    reward = -delta if heating else 0.0
                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(delta**exponent)
            else:
                if indoor_temperature < lower_bound:
                    exponent = self.lower_exponent if heating else self.higher_exponent
                    reward = -(delta**exponent)
                elif lower_bound <= indoor_temperature <= upper_bound:
                    reward = 0.0
                else:
                    exponent = self.higher_exponent if heating else self.lower_exponent
                    reward = -(delta**exponent)

            reward_list.append(float(reward))

        return [sum(reward_list)] if self.central_agent else reward_list


OccupantComfortReward = DynamicComfortReward


def set_experiment_seed(seed: int) -> int:
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def dataset_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / "data" / "datasets" / DATASET_NAME
    return dataset_dir / "schema.json", dataset_dir


def make_env(*, sim_start: int, sim_end: int, seed: int, reward_function=None, reward_function_kwargs=None):
    schema_path, dataset_dir = dataset_paths()
    config = EnvConfig(
        schema_path=schema_path,
        dataset_dir=dataset_dir,
        buildings=[CONTROL_BUILDING_NAME],
        start_time_step=int(sim_start),
        end_time_step=int(sim_end),
        random_seed=int(seed),
        central_agent=True,
        active_actions=("heating_device",),
    )
    overrides: dict[str, Any] = {}
    if reward_function is not None:
        overrides["reward_function"] = reward_function
    if reward_function_kwargs is not None:
        overrides["reward_function_kwargs"] = reward_function_kwargs
    return make_citylearn_env(config, **overrides)


def build_merlin_sac_agent(env, *, exploration_steps: int, seed: int, config: MerlinSACConfig):
    """Create CityLearn SACRBC with PI exploration and MERLIN-like parameters."""

    from citylearn.agents.rbc import PITemperatureController
    from citylearn.agents.sac import SACRBC

    # The local CityLearn RLC constructor has the historical misspelling
    # `action_scaling_coefficienct`; use it so the value is actually applied.
    return SACRBC(
        env,
        rbc=PITemperatureController(
            env,
            kp=config.pi_kp,
            ki=config.pi_ki,
            temp_deadband=config.pi_deadband,
            random_seed=int(seed),
        ),
        hidden_dimension=list(config.hidden_dimension),
        discount=config.discount,
        tau=config.tau,
        lr=config.lr,
        alpha=config.alpha,
        batch_size=config.batch_size,
        replay_buffer_capacity=config.replay_buffer_capacity,
        standardize_start_time_step=int(exploration_steps),
        end_exploration_time_step=max(int(exploration_steps) - 1, 0),
        action_scaling_coefficienct=config.action_scaling_coefficient,
        reward_scaling=config.reward_scaling,
        update_per_time_step=config.update_per_time_step,
        random_seed=int(seed),
    )


def _empty_kpis():
    import pandas as pd

    return pd.DataFrame(columns=["cost_function", "value", "name", "level"])


def _get_citylearn_kpi(district_kpis, name: str):
    rows = district_kpis[district_kpis["cost_function"] == name]["value"]
    return float(rows.iloc[0]) if len(rows) > 0 else None


def reward_function_from_mode(reward_mode: str):
    if reward_mode == "comfort":
        from citylearn.reward_function import ComfortReward

        return ComfortReward
    if reward_mode == "dynamic_comfort":
        return DynamicComfortReward
    if reward_mode == "fixed_comfort":
        return FixedComfortReward
    if reward_mode == "feedback":
        return OccupantFeedbackReward
    raise ValueError("reward_mode must be 'comfort', 'dynamic_comfort', 'fixed_comfort', or 'feedback'.")


def _building_arrays(env, n_steps: int | None = None) -> dict[str, np.ndarray]:
    building = env.unwrapped.buildings[0]
    n = int(env.unwrapped.time_step + 1 if n_steps is None else n_steps)
    n = max(0, n)

    def arr(values):
        return np.asarray(values[:n], dtype=float)

    data = {
        "time_step": np.arange(n, dtype=int),
        "indoor_temperature": arr(building.indoor_dry_bulb_temperature),
        "baseline_setpoint": arr(
            building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
        ),
        "effective_setpoint": arr(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point),
        "cooling_setpoint": arr(building.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point),
    }

    if hasattr(building, "net_electricity_consumption"):
        data["net_electricity_consumption"] = arr(building.net_electricity_consumption)

    return data


def rollout_history_dataframe(result: dict[str, Any]):
    import pandas as pd

    n_steps = len(result["Tin_hist"])
    history = pd.DataFrame({
        "time_step": np.arange(n_steps, dtype=int),
        "indoor_temperature": result["Tin_hist"],
        "baseline_setpoint": result["baseline_setpoints"],
        "effective_setpoint": result["effective_setpoints"],
        "cooling_setpoint": result["cooling_setpoints"],
        "action": result["action_hist"],
        "reward": result["reward_hist"],
        "phase": result["phase_hist"],
    })
    if "controlled_action_hist" in result:
        history["controlled_action"] = result["controlled_action_hist"]
    if "net_electricity_consumption" in result:
        history["net_electricity_consumption"] = result["net_electricity_consumption"]
    optional = {
        "tdyn": "T_dyn_hist",
        "occupant_feedback": "feedback_hist",
        "occ_now": "occ_now_hist",
        "occ_hat_1step": "occ_hat_hist",
        "occ_bin_1step": "occ_bin_hist",
    }
    for column, key in optional.items():
        if key in result and len(result[key]) == n_steps:
            history[column] = np.asarray(result[key], dtype=float)
    return history


def summarize_run(label: str, result: dict[str, Any], *, online_info: dict[str, Any] | None = None):
    summary = {
        "case": label,
        "override_count": int(result.get("override_count", 0)),
        "reward_total": float(np.asarray(result["reward_hist"], dtype=float).sum()) if len(result["reward_hist"]) else 0.0,
        "electricity_consumption_total": _get_citylearn_kpi(result["district_kpis"], "electricity_consumption_total"),
        "discomfort_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_proportion"),
        "discomfort_cold_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_cold_proportion"),
        "discomfort_hot_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_hot_proportion"),
        "daily_peak_average": _get_citylearn_kpi(result["district_kpis"], "daily_peak_average"),
    }
    if online_info:
        summary.update(online_info)
    if "target_temperature" in result:
        summary["target_temperature"] = float(result["target_temperature"])
    if "predicted_occupied_fraction" in result:
        summary["predicted_occupied_fraction"] = float(result["predicted_occupied_fraction"])
    if "T_dyn_hist" in result and len(result["T_dyn_hist"]):
        summary["tdyn_mean"] = float(np.asarray(result["T_dyn_hist"], dtype=float).mean())
        summary["tdyn_final"] = float(np.asarray(result["T_dyn_hist"], dtype=float)[-1])
    return summary


def _tdyn_initial_value(env, occupant, *, init_mode: str, lower_bound: float, upper_bound: float) -> float:
    building = env.unwrapped.buildings[0]
    scheduled = float(
        building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
            building.time_step
        ]
    )
    if init_mode == "schedule":
        value = scheduled
    elif init_mode == "pref":
        value = float(occupant.T_pref)
    else:
        raise ValueError("T_dyn_init_mode must be 'schedule' or 'pref'.")
    return float(np.clip(value, lower_bound, upper_bound))


def _update_tdyn_from_feedback(
    value: float,
    feedback: float,
    occupant,
    *,
    delta_up: float,
    delta_down: float,
    drift_to_pref: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    if drift_to_pref > 0.0:
        value = (1.0 - drift_to_pref) * float(value) + drift_to_pref * float(occupant.T_pref)
    if int(feedback) > 0:
        value += float(delta_up)
    elif int(feedback) < 0:
        value -= float(delta_down)
    return float(np.clip(value, lower_bound, upper_bound))


def _tdyn_gate(
    *,
    occupancy_model,
    occupancy_series,
    absolute_time_step: int,
    alpha_occ: float,
) -> tuple[float, float, float]:
    current_occupancy = int(occupancy_series[min(int(absolute_time_step), len(occupancy_series) - 1)])
    occ_hat, occ_bin = forecast_occupancy_from_bsplines(
        occupancy_model,
        current_occupancy=current_occupancy,
        time_step=int(absolute_time_step),
        horizon=1,
        alpha=float(alpha_occ),
    )
    now = float(current_occupancy)
    hat = float(occ_hat[0]) if len(occ_hat) else now
    binary = float(occ_bin[0]) if len(occ_bin) else float(now > alpha_occ)
    return now, hat, binary


def _reward_tdyn(env) -> float | None:
    reward_function = getattr(env.unwrapped, "reward_function", None)
    values = getattr(reward_function, "_t_dyn", None)
    if values is None or not len(values) or values[0] is None:
        return None
    return float(values[0])


def _apply_tdyn_to_observations(env, observations, reference: float):
    building = env.unwrapped.buildings[0]
    time_step = int(building.time_step)
    building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[time_step] = float(reference)
    building.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[time_step] = float(reference)

    names = list(building.active_observations)
    obs_array = np.asarray(observations[0], dtype=float).copy()
    for obs_name in (
        "indoor_dry_bulb_temperature_heating_set_point",
        "indoor_dry_bulb_temperature_cooling_set_point",
    ):
        if obs_name in names:
            obs_array[names.index(obs_name)] = float(reference)
    return [obs_array]


def _save_agent_artifact(agent, path: str | Path) -> Path:
    """Save the serializable SAC state without pickling the CityLearn env."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_step": int(agent.time_step),
        "standardize_start_time_step": int(agent.standardize_start_time_step),
        "end_exploration_time_step": int(agent.end_exploration_time_step),
        "normalized": list(agent.normalized),
        "norm_mean": agent.norm_mean,
        "norm_std": agent.norm_std,
        "r_norm_mean": agent.r_norm_mean,
        "r_norm_std": agent.r_norm_std,
        "replay_buffer": [buffer.buffer for buffer in agent.replay_buffer],
        "replay_position": [int(buffer.position) for buffer in agent.replay_buffer],
        "soft_q_net1": [net.state_dict() for net in agent.soft_q_net1],
        "soft_q_net2": [net.state_dict() for net in agent.soft_q_net2],
        "target_soft_q_net1": [net.state_dict() for net in agent.target_soft_q_net1],
        "target_soft_q_net2": [net.state_dict() for net in agent.target_soft_q_net2],
        "policy_net": [net.state_dict() for net in agent.policy_net],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def _load_agent_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def train_january_pi_jumpstart(
    *,
    output_path: str | Path = "results/models/sac_merlin/january_pi_agent.pkl",
    seed: int = 49,
    reward_function=ComfortReward,
    max_steps: int | None = None,
    config: MerlinSACConfig = MerlinSACConfig(),
) -> dict[str, Any]:
    """Run January PI exploration and save the SACRBC agent with replay."""

    set_experiment_seed(seed)
    env = make_env(sim_start=JAN_START, sim_end=JAN_END, seed=seed, reward_function=reward_function)
    exploration_steps = JAN_END - JAN_START + 1
    agent = build_merlin_sac_agent(env, exploration_steps=exploration_steps, seed=seed, config=config)

    observations, _ = env.reset(seed=seed)
    terminated = truncated = False
    rewards = []
    steps = 0

    while not (terminated or truncated):
        actions = agent.predict(observations, deterministic=False)
        next_observations, reward, terminated, truncated, _ = env.step(actions)
        agent.update(observations, actions, reward, next_observations, terminated=terminated, truncated=truncated)
        rewards.append(float(np.sum(reward)))
        observations = next_observations
        steps += 1
        if max_steps is not None and steps >= int(max_steps):
            break

    agent_path = _save_agent_artifact(agent, output_path)
    model_path = Path(agent_path).with_suffix(".zip")
    try:
        agent.save_models(str(model_path))
    except Exception:
        model_path = None

    return {
        "agent": agent,
        "agent_path": str(agent_path),
        "model_path": None if model_path is None else str(model_path),
        "steps": steps,
        "reward_total": float(np.sum(rewards)),
        "replay_size": int(len(agent.replay_buffer[0])),
        "normalized": bool(agent.normalized[0]),
    }


def _evaluate_trained_agent_on_february(
    agent,
    *,
    seed: int,
    reward_function,
    reward_function_kwargs: dict[str, Any] | None = None,
    max_steps: int | None = None,
    occupant=None,
    mode: str = "bspline_observation",
    occupancy_model=None,
    occupancy_series=None,
    alpha_occ: float = 0.5,
    T_dyn_init_mode: str = "schedule",
    delta_up: float = 0.5,
    delta_down: float = 0.5,
    drift_to_pref: float = 0.01,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
    label: str = "SAC",
) -> dict[str, Any]:
    set_experiment_seed(seed)
    env = make_env(
        sim_start=FEB_START,
        sim_end=FEB_END,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
    )
    if occupant is not None:
        env = OccupantEnvWrapper(env, deepcopy(occupant))
    if occupancy_model is not None and occupancy_series is not None:
        env = BSplineOccupancyObservationWrapper(
            env,
            occupancy_model=occupancy_model,
            occupancy_series=occupancy_series,
            forecast_fn=forecast_occupancy_from_bsplines,
            alpha=alpha_occ,
        )
    observations, _ = env.reset(seed=seed)
    agent.reset()
    use_sac_policy = all(bool(v) for v in agent.normalized)
    if not use_sac_policy and hasattr(agent, "rbc"):
        # Short smoke tests can stop before the January jump-start reaches the
        # SAC standardization point. Full runs should always use SAC here.
        agent.rbc.reset()

    terminated = truncated = False
    rewards: list[float] = []
    actions_hist: list[float] = []
    controlled_actions_hist: list[float] = []
    phase_hist: list[str] = []
    feedback_hist: list[float] = []
    tdyn_hist: list[float] = []
    occ_now_hist: list[float] = []
    occ_hat_hist: list[float] = []
    occ_bin_hist: list[float] = []
    steps = 0
    T_dyn = None
    if occupant is not None and mode == "bspline_tdyn":
        T_dyn = _tdyn_initial_value(
            env,
            occupant,
            init_mode=T_dyn_init_mode,
            lower_bound=T_dyn_min,
            upper_bound=T_dyn_max,
        )

    while not (terminated or truncated):
        if occupant is not None and mode == "bspline_tdyn":
            absolute_time_step = FEB_START + steps
            occ_now, occ_hat, occ_bin = _tdyn_gate(
                occupancy_model=occupancy_model,
                occupancy_series=occupancy_series,
                absolute_time_step=absolute_time_step,
                alpha_occ=alpha_occ,
            )
            scheduled = float(
                env.unwrapped.buildings[0].energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                    env.unwrapped.buildings[0].time_step
                ]
            )
            reference = float(T_dyn) if occ_bin > 0.0 else scheduled
            observations_for_agent = _apply_tdyn_to_observations(env, observations, reference)
        else:
            observations_for_agent = observations
            occ_now = occ_hat = occ_bin = np.nan
        if occupancy_model is not None and occupancy_series is not None:
            absolute_time_step = FEB_START + steps
            occ_now = float(occupancy_series[min(int(absolute_time_step), len(occupancy_series) - 1)])
            occ_hat = float(np.asarray(observations_for_agent[0], dtype=float)[-2])
            occ_bin = float(np.asarray(observations_for_agent[0], dtype=float)[-1])

        if use_sac_policy:
            actions = agent.predict(observations_for_agent, deterministic=True)
            phase = "deterministic_eval"
        else:
            actions = agent.rbc.predict(observations_for_agent)
            phase = "pi_eval_untrained_smoke"
        next_observations, reward, terminated, truncated, _ = env.step(actions)
        rewards.append(float(np.sum(reward)))
        actions_hist.append(float(np.asarray(actions, dtype=float).reshape(-1)[0]))
        controlled_actions_hist.append(float(getattr(env, "last_controlled_action", actions_hist[-1])))
        phase_hist.append(phase)
        feedback = float(getattr(env, "last_feedback", 0.0))
        feedback_hist.append(feedback)
        if occupant is not None and mode == "bspline_tdyn":
            T_dyn = _update_tdyn_from_feedback(
                float(T_dyn),
                feedback,
                occupant,
                delta_up=delta_up,
                delta_down=delta_down,
                drift_to_pref=drift_to_pref,
                lower_bound=T_dyn_min,
                upper_bound=T_dyn_max,
            )
            tdyn_hist.append(float(T_dyn))
            occ_now_hist.append(float(occ_now))
            occ_hat_hist.append(float(occ_hat))
            occ_bin_hist.append(float(occ_bin))
        else:
            reward_tdyn = _reward_tdyn(env)
            if reward_tdyn is not None:
                tdyn_hist.append(reward_tdyn)
            if occupancy_model is not None and occupancy_series is not None:
                occ_now_hist.append(float(occ_now))
                occ_hat_hist.append(float(occ_hat))
                occ_bin_hist.append(float(occ_bin))
        observations = next_observations
        steps += 1
        if max_steps is not None and steps >= int(max_steps):
            break

    try:
        kpis = env.unwrapped.evaluate().copy()
        district_kpis = kpis[kpis["level"] == "district"].copy()
    except Exception:
        kpis = _empty_kpis()
        district_kpis = kpis.copy()

    arrays = _building_arrays(env, n_steps=len(rewards))
    result = {
        "name": label,
        "env": env,
        "kpis": kpis,
        "district_kpis": district_kpis,
        "Tin_hist": arrays["indoor_temperature"],
        "baseline_setpoints": arrays["baseline_setpoint"],
        "effective_setpoints": arrays["effective_setpoint"],
        "cooling_setpoints": arrays["cooling_setpoint"],
        "reward_hist": np.asarray(rewards, dtype=float),
        "action_hist": np.asarray(actions_hist, dtype=float),
        "controlled_action_hist": np.asarray(controlled_actions_hist, dtype=float),
        "phase_hist": np.asarray(phase_hist, dtype=object),
        "override_count": int(getattr(env, "override_count", 0)),
    }
    if "net_electricity_consumption" in arrays:
        result["net_electricity_consumption"] = arrays["net_electricity_consumption"]
    if occupant is not None:
        result["feedback_hist"] = np.asarray(feedback_hist, dtype=float)
        result["target_temperature"] = float(occupant.T_pref)
    if len(tdyn_hist):
        result["T_dyn_hist"] = np.asarray(tdyn_hist, dtype=float)
    if len(occ_bin_hist):
        result["occ_now_hist"] = np.asarray(occ_now_hist, dtype=float)
        result["occ_hat_hist"] = np.asarray(occ_hat_hist, dtype=float)
        result["occ_bin_hist"] = np.asarray(occ_bin_hist, dtype=float)
        result["predicted_occupied_fraction"] = float(np.mean(occ_bin_hist))
    return result


def run_merlin_sac_no_occupant(
    *,
    output_dir: str | Path = "results/raw/sac_merlin",
    summary_dir: str | Path = "results/summaries/sac_merlin",
    model_dir: str | Path = "results/models/sac_merlin",
    seed: int = 49,
    reward_function=ComfortReward,
    max_steps: int | None = None,
    label: str = "baseline",
    config: MerlinSACConfig = MerlinSACConfig(),
    training_episodes: int = 10,
) -> dict[str, Any]:
    """Run MERLIN-style PI jump-start, SAC training, then February evaluation.

    The first January episode is controlled by Temperature PI and fills the SAC
    replay buffer. SAC then trains for the remaining January episodes before a
    deterministic February evaluation. This mirrors MERLIN's deployment
    strategy structure more closely than online training directly on the test
    period.
    """

    set_experiment_seed(seed)
    env = make_env(sim_start=JAN_START, sim_end=JAN_END, seed=seed, reward_function=reward_function)
    exploration_steps = FEB_START - JAN_START
    agent = build_merlin_sac_agent(env, exploration_steps=exploration_steps, seed=seed, config=config)

    online_rewards = []
    phase_hist = []
    action_hist = []
    episode_hist = []
    absolute_time_step_hist = []
    step_limit_reached = False
    total_steps = 0

    for episode in range(int(training_episodes)):
        observations, _ = env.reset(seed=seed)
        terminated = truncated = False
        episode_step = 0

        while not (terminated or truncated):
            phase = "pi_exploration" if agent.time_step < exploration_steps else "sac_training"
            actions = agent.predict(observations, deterministic=False)
            next_observations, reward, terminated, truncated, _ = env.step(actions)
            agent.update(observations, actions, reward, next_observations, terminated=terminated, truncated=truncated)

            online_rewards.append(float(np.sum(reward)))
            phase_hist.append(phase)
            action_hist.append(float(np.asarray(actions, dtype=float).reshape(-1)[0]))
            episode_hist.append(int(episode))
            absolute_time_step_hist.append(int(JAN_START + episode_step))
            observations = next_observations
            episode_step += 1
            total_steps += 1
            if max_steps is not None and total_steps >= int(max_steps):
                step_limit_reached = True
                break

        if step_limit_reached:
            break

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    agent_path = _save_agent_artifact(agent, model_dir / f"{label}_jan_feb_online_agent.pkl")
    model_zip = model_dir / f"{label}_jan_feb_online_agent.zip"
    try:
        agent.save_models(str(model_zip))
    except Exception:
        model_zip = None

    result = _evaluate_trained_agent_on_february(
        agent,
        seed=seed,
        reward_function=reward_function,
        max_steps=max_steps,
    )

    online_info = {
        "jumpstart_controller": "PITemperatureController",
        "exploration_steps": int(exploration_steps),
        "training_episodes": int(training_episodes),
        "online_train_steps": int(len(online_rewards)),
        "online_train_reward_total": float(np.sum(online_rewards)),
        "online_replay_size": int(len(agent.replay_buffer[0])),
        "online_normalized": bool(agent.normalized[0]),
        "online_pi_steps": int(sum(1 for p in phase_hist if p == "pi_exploration")),
        "online_sac_steps": int(sum(1 for p in phase_hist if p == "sac_training")),
        "agent_pickle": str(agent_path),
        "agent_model_zip": None if model_zip is None else str(model_zip),
    }
    metrics = summarize_run(label, result, online_info=online_info)
    save_result(
        result,
        output_dir=Path(output_dir),
        summary_dir=Path(summary_dir),
        scenario="no_occupant",
        label=label,
        metrics=metrics,
    )

    online_history = {
        "time_step": np.arange(len(online_rewards), dtype=int),
        "episode": episode_hist,
        "absolute_time_step": absolute_time_step_hist,
        "phase": phase_hist,
        "action": action_hist,
        "reward": online_rewards,
    }
    import pandas as pd

    output_case_dir = Path(output_dir) / "no_occupant" / label
    pd.DataFrame(online_history).to_csv(output_case_dir / "online_training_rollout.csv", index=False)

    summary_path = Path(summary_dir) / "no_occupant" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)

    manifest = {
        "method": "CityLearn SACRBC MERLIN-style PI jump-start",
        "source_equivalence": {
            "exploration_policy": "Temperature PI via CityLearn PITemperatureController",
            "rl_agent": "CityLearn SACRBC",
            "separate_offline_gradient_pretraining": False,
            "training_period": "January episodes",
            "evaluation_period": "February deterministic rollout",
            "online_update_per_time_step": config.update_per_time_step,
        },
        "config": {
            **config.__dict__,
            "hidden_dimension": list(config.hidden_dimension),
        },
        "metrics": metrics,
    }
    (output_case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("MERLIN-style SAC run finished")
    print(f"PI exploration steps: {online_info['online_pi_steps']}")
    print(f"SAC training steps: {online_info['online_sac_steps']}")
    print(f"Replay size: {online_info['online_replay_size']}")
    print(f"February reward: {metrics['reward_total']:.6f}")
    print(result["district_kpis"][["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    print(f"Saved rollout to {output_case_dir / 'rollout.csv'}")
    print(f"Saved summary to {summary_path}")
    return {"agent": agent, "result": result, "metrics": metrics, "online_info": online_info}


def _train_merlin_agent_on_january(
    *,
    seed: int,
    training_seed: int,
    reward_function,
    reward_function_kwargs: dict[str, Any] | None = None,
    config: MerlinSACConfig,
    training_episodes: int,
    max_steps: int | None = None,
    occupant=None,
    mode: str = "bspline_observation",
    occupancy_model=None,
    occupancy_series=None,
    alpha_occ: float = 0.5,
    T_dyn_init_mode: str = "schedule",
    delta_up: float = 0.5,
    delta_down: float = 0.5,
    drift_to_pref: float = 0.01,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
):
    import pandas as pd

    set_experiment_seed(training_seed)
    env = make_env(
        sim_start=JAN_START,
        sim_end=JAN_END,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
    )
    if occupant is not None:
        env = OccupantEnvWrapper(env, deepcopy(occupant))
    if occupancy_model is not None and occupancy_series is not None:
        env = BSplineOccupancyObservationWrapper(
            env,
            occupancy_model=occupancy_model,
            occupancy_series=occupancy_series,
            forecast_fn=forecast_occupancy_from_bsplines,
            alpha=alpha_occ,
        )

    exploration_steps = JAN_END - JAN_START + 1
    agent = build_merlin_sac_agent(env, exploration_steps=exploration_steps, seed=seed, config=config)
    rewards = []
    phases = []
    actions_hist = []
    controlled_actions_hist = []
    episodes = []
    absolute_time_steps = []
    feedback_hist = []
    tdyn_hist = []
    occ_bin_hist = []
    total_steps = 0
    step_limit_reached = False

    for episode in range(int(training_episodes)):
        observations, _ = env.reset(seed=seed)
        terminated = truncated = False
        episode_step = 0
        T_dyn = None
        if occupant is not None and mode == "bspline_tdyn":
            T_dyn = _tdyn_initial_value(
                env,
                occupant,
                init_mode=T_dyn_init_mode,
                lower_bound=T_dyn_min,
                upper_bound=T_dyn_max,
            )

        while not (terminated or truncated):
            if occupant is not None and mode == "bspline_tdyn":
                absolute_time_step = JAN_START + episode_step
                _, _, occ_bin = _tdyn_gate(
                    occupancy_model=occupancy_model,
                    occupancy_series=occupancy_series,
                    absolute_time_step=absolute_time_step,
                    alpha_occ=alpha_occ,
                )
                scheduled = float(
                    env.unwrapped.buildings[0].energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                        env.unwrapped.buildings[0].time_step
                    ]
                )
                reference = float(T_dyn) if occ_bin > 0.0 else scheduled
                observations_for_agent = _apply_tdyn_to_observations(env, observations, reference)
            else:
                observations_for_agent = observations
                occ_bin = np.nan
            if occupancy_model is not None and occupancy_series is not None:
                absolute_time_step = JAN_START + episode_step
                occ_now = float(occupancy_series[min(int(absolute_time_step), len(occupancy_series) - 1)])
                obs_array = np.asarray(observations_for_agent[0], dtype=float)
                occ_hat = float(obs_array[-2])
                occ_bin = float(obs_array[-1])
            else:
                occ_now = occ_hat = np.nan

            phase = "pi_exploration" if agent.time_step < exploration_steps else "sac_training"
            actions = agent.predict(observations_for_agent, deterministic=False)
            next_observations, reward, terminated, truncated, _ = env.step(actions)
            feedback = float(getattr(env, "last_feedback", 0.0))

            if occupant is not None and mode == "bspline_tdyn":
                T_dyn = _update_tdyn_from_feedback(
                    float(T_dyn),
                    feedback,
                    occupant,
                    delta_up=delta_up,
                    delta_down=delta_down,
                    drift_to_pref=drift_to_pref,
                    lower_bound=T_dyn_min,
                    upper_bound=T_dyn_max,
                )
                next_observations_for_agent = next_observations
                if not (terminated or truncated):
                    next_abs = JAN_START + min(episode_step + 1, JAN_END - JAN_START)
                    _, _, next_occ_bin = _tdyn_gate(
                        occupancy_model=occupancy_model,
                        occupancy_series=occupancy_series,
                        absolute_time_step=next_abs,
                        alpha_occ=alpha_occ,
                    )
                    scheduled_next = float(
                        env.unwrapped.buildings[0].energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            env.unwrapped.buildings[0].time_step
                        ]
                    )
                    next_reference = float(T_dyn) if next_occ_bin > 0.0 else scheduled_next
                    next_observations_for_agent = _apply_tdyn_to_observations(env, next_observations, next_reference)
            else:
                next_observations_for_agent = next_observations

            agent.update(
                observations_for_agent,
                actions,
                reward,
                next_observations_for_agent,
                terminated=terminated,
                truncated=truncated,
            )

            rewards.append(float(np.sum(reward)))
            phases.append(phase)
            actions_hist.append(float(np.asarray(actions, dtype=float).reshape(-1)[0]))
            controlled_actions_hist.append(float(getattr(env, "last_controlled_action", actions_hist[-1])))
            episodes.append(int(episode))
            absolute_time_steps.append(int(JAN_START + episode_step))
            feedback_hist.append(feedback)
            if occupant is not None and mode == "bspline_tdyn":
                tdyn_hist.append(float(T_dyn))
                occ_bin_hist.append(float(occ_bin))
            else:
                reward_tdyn = _reward_tdyn(env)
                if reward_tdyn is not None:
                    tdyn_hist.append(float(reward_tdyn))
                if occupancy_model is not None and occupancy_series is not None:
                    occ_bin_hist.append(float(occ_bin))
            observations = next_observations_for_agent
            episode_step += 1
            total_steps += 1

            if max_steps is not None and total_steps >= int(max_steps):
                step_limit_reached = True
                break

        if step_limit_reached:
            break

    history = pd.DataFrame(
        {
            "time_step": np.arange(len(rewards), dtype=int),
            "episode": episodes,
            "absolute_time_step": absolute_time_steps,
            "phase": phases,
            "action": actions_hist,
            "controlled_action": controlled_actions_hist,
            "reward": rewards,
            "occupant_feedback": feedback_hist,
        }
    )
    if tdyn_hist:
        history["tdyn"] = tdyn_hist
    if occ_bin_hist:
        history["occ_bin_1step"] = occ_bin_hist

    info = {
        "jumpstart_controller": "PITemperatureController",
        "exploration_steps": int(exploration_steps),
        "training_episodes": int(training_episodes),
        "online_train_steps": int(len(rewards)),
        "online_train_reward_total": float(np.sum(rewards)),
        "online_replay_size": int(len(agent.replay_buffer[0])),
        "online_normalized": bool(agent.normalized[0]),
        "online_pi_steps": int(sum(1 for p in phases if p == "pi_exploration")),
        "online_sac_steps": int(sum(1 for p in phases if p == "sac_training")),
    }
    return agent, info, history


def _run_merlin_sac_occupant_case(
    *,
    label: str,
    occupant,
    mode: str,
    reward_mode: str,
    output_dir: Path,
    summary_dir: Path,
    model_dir: Path,
    seed: int,
    training_seed: int,
    max_steps: int | None,
    config: MerlinSACConfig,
    training_episodes: int,
    alpha_occ: float,
    T_dyn_init_mode: str,
    delta_up: float,
    delta_down: float,
    drift_to_pref: float,
    T_dyn_min: float,
    T_dyn_max: float,
) -> dict[str, Any]:
    import pandas as pd

    reward_function = reward_function_from_mode(reward_mode)
    reward_function_kwargs = None
    if reward_mode == "dynamic_comfort":
        reward_function_kwargs = {
            "occupant": deepcopy(occupant),
            "t_dyn_init_mode": T_dyn_init_mode,
            "delta_up": delta_up,
            "delta_down": delta_down,
            "drift_to_pref": drift_to_pref,
            "t_dyn_min": T_dyn_min,
            "t_dyn_max": T_dyn_max,
        }
    elif reward_mode == "fixed_comfort":
        reward_function_kwargs = {
            "occupant": deepcopy(occupant),
        }
    occupancy_model, occupancy_series = fit_bspline_occupancy_model(seed=seed)

    agent, online_info, training_history = _train_merlin_agent_on_january(
        seed=seed,
        training_seed=training_seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        config=config,
        training_episodes=training_episodes,
        max_steps=max_steps,
        occupant=deepcopy(occupant),
        mode=mode,
        occupancy_model=occupancy_model,
        occupancy_series=occupancy_series,
        alpha_occ=alpha_occ,
        T_dyn_init_mode=T_dyn_init_mode,
        delta_up=delta_up,
        delta_down=delta_down,
        drift_to_pref=drift_to_pref,
        T_dyn_min=T_dyn_min,
        T_dyn_max=T_dyn_max,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    agent_path = _save_agent_artifact(agent, model_dir / f"{label}_jan_training_agent.pkl")
    model_zip = model_dir / f"{label}_jan_training_agent.zip"
    try:
        agent.save_models(str(model_zip))
    except Exception:
        model_zip = None

    result = _evaluate_trained_agent_on_february(
        agent,
        seed=int(seed + 10_000),
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        max_steps=max_steps,
        occupant=deepcopy(occupant),
        mode=mode,
        occupancy_model=occupancy_model,
        occupancy_series=occupancy_series,
        alpha_occ=alpha_occ,
        T_dyn_init_mode=T_dyn_init_mode,
        delta_up=delta_up,
        delta_down=delta_down,
        drift_to_pref=drift_to_pref,
        T_dyn_min=T_dyn_min,
        T_dyn_max=T_dyn_max,
        label=label,
    )

    online_info = {
        **online_info,
        "reward_mode": reward_mode,
        "occupant_mode": mode,
        "T_dyn_init_mode": T_dyn_init_mode,
        "agent_pickle": str(agent_path),
        "agent_model_zip": None if model_zip is None else str(model_zip),
    }
    metrics = summarize_run(label, result, online_info=online_info)
    save_result(
        result,
        output_dir=output_dir,
        summary_dir=summary_dir,
        scenario="occupant_present",
        label=label,
        metrics=metrics,
    )

    output_case_dir = output_dir / "occupant_present" / label
    training_history.to_csv(output_case_dir / "online_training_rollout.csv", index=False)
    manifest = {
        "method": "CityLearn SACRBC MERLIN-style PI jump-start occupant case",
        "source_equivalence": {
            "exploration_policy": "Temperature PI via CityLearn PITemperatureController",
            "rl_agent": "CityLearn SACRBC",
            "training_period": "January episodes",
            "evaluation_period": "February deterministic rollout",
            "reward_mode": reward_mode,
            "occupant_mode": mode,
            "T_dyn_init_mode": T_dyn_init_mode,
            "online_update_per_time_step": config.update_per_time_step,
        },
        "config": {
            **config.__dict__,
            "hidden_dimension": list(config.hidden_dimension),
        },
        "metrics": metrics,
    }
    (output_case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Reward:", metrics["reward_total"])
    print("Overrides:", metrics["override_count"])
    if "tdyn_final" in metrics:
        print(f"T_dyn mean/final: {metrics['tdyn_mean']:.3f} / {metrics['tdyn_final']:.3f}")
    if not result["district_kpis"].empty:
        print(result["district_kpis"][["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return {"agent": agent, "result": result, "metrics": metrics}


def run_merlin_sac_occupants(
    *,
    output_dir: str | Path = "results/raw/sac_merlin",
    summary_dir: str | Path = "results/summaries/sac_merlin",
    model_dir: str | Path = "results/models/sac_merlin",
    seed: int = 49,
    training_seed: int = 0,
    max_steps: int | None = None,
    occupant: str = "all",
    occupant_mode: str = "both",
    reward_mode: str = "all",
    config: MerlinSACConfig = MerlinSACConfig(),
    training_episodes: int = 10,
    alpha_occ: float = 0.5,
    T_dyn_init_mode: str = "schedule",
    delta_up: float = 0.5,
    delta_down: float = 0.5,
    drift_to_pref: float = 0.01,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
) -> dict[str, Any]:
    import pandas as pd

    output_dir = Path(output_dir)
    summary_dir = Path(summary_dir)
    model_dir = Path(model_dir)

    occupants = fitted_occupants()
    if occupant != "all":
        occupants = {occupant: occupants[occupant]}

    mode = "bspline_observation"
    if reward_mode == "all":
        reward_modes = ["comfort", "dynamic_comfort", "feedback", "fixed_comfort"]
    elif reward_mode == "both":
        reward_modes = ["comfort", "dynamic_comfort", "feedback"]
    elif reward_mode == "occupant_comfort":
        reward_modes = ["dynamic_comfort"]
    else:
        reward_modes = [reward_mode]
    add_reward_suffix = len(reward_modes) > 1
    suffix = {
        "comfort": "comfort_reward",
        "dynamic_comfort": "dynamic_reward",
        "feedback": "feedback_reward",
        "fixed_comfort": "fixed_comfort_reward",
    }

    all_results: dict[str, Any] = {}
    summary_rows = []
    for occ_idx, (name, occ) in enumerate(occupants.items(), start=1):
        for reward_name in reward_modes:
            label = f"{name}_{suffix[reward_name]}" if add_reward_suffix else name
            print(f"\n{label}")
            case = _run_merlin_sac_occupant_case(
                label=label,
                occupant=deepcopy(occ),
                mode=mode,
                reward_mode=reward_name,
                output_dir=output_dir,
                summary_dir=summary_dir,
                model_dir=model_dir,
                seed=int(seed + occ_idx),
                training_seed=int(training_seed + occ_idx),
                max_steps=max_steps,
                config=config,
                training_episodes=training_episodes,
                alpha_occ=alpha_occ,
                T_dyn_init_mode=T_dyn_init_mode,
                delta_up=delta_up,
                delta_down=delta_down,
                drift_to_pref=drift_to_pref,
                T_dyn_min=T_dyn_min,
                T_dyn_max=T_dyn_max,
            )
            all_results[label] = case
            summary_rows.append(case["metrics"])

    summary_path = summary_dir / "occupant_present" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"Saved summary to {summary_path}")
    return all_results


def run_merlin_jumpstart_only_february(
    *,
    output_dir: str | Path = "results/raw/sac_merlin",
    seed: int = 49,
    reward_function=ComfortReward,
    max_steps: int | None = None,
    label: str = "jumpstart_only_february",
    config: MerlinSACConfig = MerlinSACConfig(),
) -> dict[str, Any]:
    """Run the January PI jump-start through February without SAC learning.

    This is a diagnostic rollout: the SACRBC wrapper is still used, but the
    exploration/standardization boundary is placed after the simulated period,
    so the Temperature-PI jump-start controller remains in charge and the SAC
    policy is never trained or evaluated. The saved rollout is February only,
    matching the layout used by the result notebook.
    """

    set_experiment_seed(seed)
    env = make_env(sim_start=JAN_START, sim_end=FEB_END, seed=seed, reward_function=reward_function)

    # Keep the agent in PI exploration mode for the entire Jan+Feb diagnostic.
    no_training_boundary = FEB_END - JAN_START + 2
    agent = build_merlin_sac_agent(
        env,
        exploration_steps=no_training_boundary,
        seed=seed,
        config=config,
    )

    observations, _ = env.reset(seed=seed)
    terminated = truncated = False
    rewards: list[float] = []
    actions_hist: list[float] = []
    phase_hist: list[str] = []
    step = 0

    while not (terminated or truncated):
        actions = agent.predict(observations, deterministic=False)
        next_observations, reward, terminated, truncated, _ = env.step(actions)
        agent.update(observations, actions, reward, next_observations, terminated=terminated, truncated=truncated)

        rewards.append(float(np.sum(reward)))
        actions_hist.append(float(np.asarray(actions, dtype=float).reshape(-1)[0]))
        phase_hist.append("pi_jumpstart_only")
        observations = next_observations
        step += 1
        if max_steps is not None and step >= int(max_steps):
            break

    feb_start_idx = max(0, FEB_START - JAN_START)
    feb_end_idx = len(rewards)
    arrays = _building_arrays(env, n_steps=feb_end_idx)

    result = {
        "Tin_hist": arrays["indoor_temperature"][feb_start_idx:feb_end_idx],
        "baseline_setpoints": arrays["baseline_setpoint"][feb_start_idx:feb_end_idx],
        "effective_setpoints": arrays["effective_setpoint"][feb_start_idx:feb_end_idx],
        "cooling_setpoints": arrays["cooling_setpoint"][feb_start_idx:feb_end_idx],
        "reward_hist": np.asarray(rewards[feb_start_idx:feb_end_idx], dtype=float),
        "action_hist": np.asarray(actions_hist[feb_start_idx:feb_end_idx], dtype=float),
        "phase_hist": np.asarray(phase_hist[feb_start_idx:feb_end_idx], dtype=object),
        "override_count": int(getattr(env, "override_count", 0)),
    }
    if "net_electricity_consumption" in arrays:
        result["net_electricity_consumption"] = arrays["net_electricity_consumption"][feb_start_idx:feb_end_idx]

    output_case_dir = Path(output_dir) / "no_occupant" / label
    output_case_dir.mkdir(parents=True, exist_ok=True)
    rollout_history_dataframe(result).to_csv(output_case_dir / "rollout.csv", index=False)
    manifest = {
        "method": "CityLearn SACRBC Temperature-PI jump-start only",
        "source_equivalence": {
            "exploration_policy": "Temperature PI via CityLearn PITemperatureController",
            "rl_agent": "CityLearn SACRBC",
            "separate_offline_gradient_pretraining": False,
            "february_sac_online_learning": False,
        },
        "diagnostic": {
            "saved_rollout": "February only",
            "january_warm_start_steps": int(feb_start_idx),
            "february_pi_steps": int(len(result["reward_hist"])),
            "standardize_start_time_step": int(agent.standardize_start_time_step),
            "end_exploration_time_step": int(agent.end_exploration_time_step),
            "sac_normalized": bool(agent.normalized[0]),
        },
        "config": {
            **config.__dict__,
            "hidden_dimension": list(config.hidden_dimension),
        },
    }
    (output_case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("MERLIN jump-start-only February diagnostic finished")
    print(f"January PI warm-start steps: {feb_start_idx}")
    print(f"February PI steps saved: {len(result['reward_hist'])}")
    print(f"Saved rollout to {output_case_dir / 'rollout.csv'}")
    return {"agent": agent, "result": result, "manifest": manifest}


def save_result(
    result: dict[str, Any],
    *,
    output_dir: Path,
    summary_dir: Path,
    scenario: str,
    label: str,
    metrics: dict[str, Any],
) -> None:
    output_case_dir = output_dir / scenario / label
    summary_case_dir = summary_dir / scenario
    output_case_dir.mkdir(parents=True, exist_ok=True)
    summary_case_dir.mkdir(parents=True, exist_ok=True)

    rollout_history_dataframe(result).to_csv(output_case_dir / "rollout.csv", index=False)
    result["kpis"].to_csv(output_case_dir / "kpis.csv", index=False)
    result["district_kpis"].to_csv(output_case_dir / "district_kpis.csv", index=False)
    (summary_case_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
