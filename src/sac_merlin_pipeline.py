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
import io
import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Union

import numpy as np
from gymnasium import spaces
from citylearn.reward_function import ComfortReward, RewardFunction

from src.envs import EnvConfig, make_citylearn_env
from src.occupants.comfort import OccupantFeedbackReward
from src.occupants.metrics import (
    kelvin_hours,
    occupant_probability_comfort_band,
    override_hours,
    setpoint_kelvin_hours,
)
from src.occupants.tdyn import TDynConfig, update_tdyn_reference
from src.occupants.wrappers import BSplineOccupancyObservationWrapper, OccupantEnvWrapper
from src.rewards import BalancedComfortEnergyReward, LagrangianDistrictReward
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
    automatic_entropy_tuning: bool = False
    entropy_lr: float | None = None
    target_entropy: float | None = None
    robust_normalization: bool = False
    robust_normalization_low: float = 5.0
    robust_normalization_high: float = 95.0
    normalized_observation_clip: float | None = None
    normalized_reward_clip: float | None = None
    observation_filter: str | None = None
    batch_size: int = 256
    replay_buffer_capacity: int = 100_000
    action_scaling_coefficient: float = 0.5
    reward_scaling: float = 5.0
    update_per_time_step: int = 2
    n_step_horizon: int = 1
    device: str | None = None
    pi_kp: float = 0.2
    pi_ki: float = 0.005
    pi_deadband: float = 0.5


MPC_CORE_OBSERVATIONS = (
    "hour",
    "outdoor_dry_bulb_temperature",
    "direct_solar_irradiance",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_heating_set_point",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "comfort_band",
    "hvac_mode",
    "bspline_occupancy_probability_1",
    "bspline_occupied_1",
)


class ObservationSubsetWrapper:
    """Expose a compact SAC observation vector while preserving the base env."""

    def __init__(self, env, names: tuple[str, ...]):
        self.env = env
        self._requested_names = tuple(names)
        self._indices: list[list[int]] = []
        self.observation_names = []
        self.observation_space = []

        for obs_names, obs_space in zip(env.observation_names, env.observation_space):
            selected = [name for name in self._requested_names if name in obs_names]
            if not selected:
                raise ValueError("ObservationSubsetWrapper selected no observations.")
            indices = [list(obs_names).index(name) for name in selected]
            self._indices.append(indices)
            self.observation_names.append(selected)
            low = np.asarray(obs_space.low, dtype=np.float32)[indices]
            high = np.asarray(obs_space.high, dtype=np.float32)[indices]
            self.observation_space.append(spaces.Box(low=low, high=high, dtype=np.float32))

        self.action_space = env.action_space
        if hasattr(env, "action_names"):
            self.action_names = env.action_names

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _filter(self, observations):
        return [
            np.asarray(observation, dtype=np.float32)[indices]
            for observation, indices in zip(observations, self._indices)
        ]

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        return self._filter(observations), info

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)
        return self._filter(observations), reward, terminated, truncated, info

    def close(self):
        return self.env.close()


def _apply_observation_filter(env, observation_filter: str | None):
    mode = "none" if observation_filter is None else str(observation_filter).strip().lower()
    if mode in {"", "none", "full", "all"}:
        return env
    if mode == "mpc_core":
        return ObservationSubsetWrapper(env, MPC_CORE_OBSERVATIONS)
    raise ValueError(f"Unknown SAC observation filter: {observation_filter}")


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
        drift_to_pref: float = 0.05,
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

        feedback = float(observation.get("occupant_feedback_signal", 0.0))
        self._t_dyn[i] = update_tdyn_reference(
            self._t_dyn[i],
            feedback,
            preferred_temperature=t_pref,
            config=TDynConfig(
                delta_up=self.delta_up,
                delta_down=self.delta_down,
                drift_to_pref=self.drift_to_pref,
                lower_bound=self.t_dyn_min,
                upper_bound=self.t_dyn_max,
                anchor_feedback_to_pref=True,
            ),
        )

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


class ExpectedOccupantOverrideReward(RewardFunction):
    """Track the active reference and penalize fitted occupant override risk.

    The tracking reference is the active setpoint seen by the controller unless
    a caller explicitly supplies an override through set_current_reference_setpoints.
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        occupant=None,
        comfort_track_weight: float = 0.1,
        comfort_cold_weight: float = 0.7,
        comfort_hot_weight: float = 0.2,
        comfort_track_scale: float = 2.0,
        occupant_probability_base: float = 0.002,
        occupant_probability_clip: bool = True,
        single_setpoint: bool = True,
        **kwargs,
    ):
        super().__init__(env_metadata=env_metadata, **kwargs)
        self.occupant = deepcopy(occupant)
        self.comfort_track_weight = float(comfort_track_weight)
        self.comfort_cold_weight = float(comfort_cold_weight)
        self.comfort_hot_weight = float(comfort_hot_weight)
        self.comfort_track_scale = max(float(comfort_track_scale), 1e-6)
        self.occupant_probability_base = max(float(occupant_probability_base), 0.0)
        self.occupant_probability_clip = bool(occupant_probability_clip)
        self.single_setpoint = bool(single_setpoint)
        self._current_reference_setpoints: list[float | None] = []
        self.reset()

    def reset(self):
        if self.env_metadata is None:
            self._current_reference_setpoints = []
        else:
            self._current_reference_setpoints = [None for _ in self.env_metadata["buildings"]]

    def set_current_reference_setpoints(self, setpoints) -> None:
        self._current_reference_setpoints = [
            None if value is None else float(value)
            for value in setpoints
        ]

    def _occupant_pref(self, fallback: float) -> float:
        if self.occupant is None:
            return float(fallback)
        if hasattr(self.occupant, "T_pref"):
            return float(self.occupant.T_pref)
        if hasattr(self.occupant, "preferred_temperature"):
            return float(self.occupant.preferred_temperature)
        return float(fallback)

    def _occupant_sensitivity(self) -> float:
        if self.occupant is None:
            return 1.0
        if hasattr(self.occupant, "b"):
            return float(self.occupant.b)
        if hasattr(self.occupant, "sensitivity"):
            return float(self.occupant.sensitivity)
        return 1.0

    def _occupant_max_probability(self) -> float:
        if self.occupant is None:
            return 1.0
        if hasattr(self.occupant, "max_prob"):
            return float(self.occupant.max_prob)
        if hasattr(self.occupant, "max_probability"):
            return float(self.occupant.max_probability)
        return 0.4

    def _setpoint(self, observation: Mapping[str, Union[int, float]], index: int | None = None) -> float:
        if (
            index is not None
            and index < len(self._current_reference_setpoints)
            and self._current_reference_setpoints[index] is not None
        ):
            return float(self._current_reference_setpoints[index])
        if "override_setpoint" in observation:
            return float(observation["override_setpoint"])
        if self.single_setpoint:
            return float(observation["indoor_dry_bulb_temperature_heating_set_point"])

        hvac_mode = int(observation.get("hvac_mode", 0))
        if hvac_mode == 1:
            return float(observation["indoor_dry_bulb_temperature_cooling_set_point"])
        if hvac_mode == 2:
            return float(observation["indoor_dry_bulb_temperature_heating_set_point"])
        heating = float(observation.get("heating_demand", 0.0))
        cooling = float(observation.get("cooling_demand", 0.0))
        if cooling > heating:
            return float(observation["indoor_dry_bulb_temperature_cooling_set_point"])
        return float(observation["indoor_dry_bulb_temperature_heating_set_point"])

    def _override_probability(self, delta: float) -> float:
        exponent = min(50.0, max(0.0, self._occupant_sensitivity() * float(delta)))
        probability = self.occupant_probability_base * float(np.exp(exponent))
        if self.occupant_probability_clip:
            probability = min(probability, self._occupant_max_probability())
        return float(probability)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = []
        for i, observation in enumerate(observations):
            tin = float(observation["indoor_dry_bulb_temperature"])
            tref = self._setpoint(observation, i)
            tpref = self._occupant_pref(tref)
            track = ((tin - tref) / self.comfort_track_scale) ** 2
            p_cold = self._override_probability(max(tpref - tin, 0.0))
            p_hot = self._override_probability(max(tin - tpref, 0.0))
            cost = (
                self.comfort_track_weight * track
                + self.comfort_cold_weight * p_cold
                + self.comfort_hot_weight * p_hot
            )
            reward_list.append(float(-cost))

        return [float(np.sum(reward_list))] if self.central_agent else reward_list


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


def make_env(
    *,
    sim_start: int,
    sim_end: int,
    seed: int,
    reward_function=None,
    reward_function_kwargs=None,
    building_name: str = CONTROL_BUILDING_NAME,
):
    schema_path, dataset_dir = dataset_paths()
    config = EnvConfig(
        schema_path=schema_path,
        dataset_dir=dataset_dir,
        buildings=[building_name],
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
        automatic_entropy_tuning=config.automatic_entropy_tuning,
        entropy_lr=config.entropy_lr,
        target_entropy=config.target_entropy,
        robust_normalization=config.robust_normalization,
        robust_normalization_low=config.robust_normalization_low,
        robust_normalization_high=config.robust_normalization_high,
        normalized_observation_clip=config.normalized_observation_clip,
        normalized_reward_clip=config.normalized_reward_clip,
        batch_size=config.batch_size,
        replay_buffer_capacity=config.replay_buffer_capacity,
        standardize_start_time_step=int(exploration_steps),
        end_exploration_time_step=max(int(exploration_steps) - 1, 0),
        action_scaling_coefficienct=config.action_scaling_coefficient,
        reward_scaling=config.reward_scaling,
        update_per_time_step=config.update_per_time_step,
        n_step_horizon=config.n_step_horizon,
        device=config.device,
        random_seed=int(seed),
    )


def collect_bc_expert_dataset(
    *,
    actions: list[float] | np.ndarray,
    seed: int,
    reward_function=None,
    reward_function_kwargs: dict[str, Any] | None = None,
    building_name: str = CONTROL_BUILDING_NAME,
    max_steps: int | None = None,
    sim_start: int = JAN_START,
    sim_end: int = JAN_END,
) -> dict[str, np.ndarray]:
    """Replay a teacher action sequence on the training period and record SAC observations."""

    env = make_env(
        sim_start=sim_start,
        sim_end=sim_end,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
    )
    obs, _ = env.reset(seed=seed)
    teacher_actions = np.asarray(actions, dtype=float).reshape(-1)
    observations = []
    action_hist = []
    reward_hist = []
    absolute_time_step = []
    terminated = truncated = False
    step_count = 0

    while not (terminated or truncated) and step_count < teacher_actions.size:
        action = float(np.clip(teacher_actions[step_count], 0.0, 1.0))
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        observations.append(np.asarray(o, dtype=float))
        action_hist.append(action)
        absolute_time_step.append(int(JAN_START + step_count))
        obs, reward, terminated, truncated, _ = env.step([[action]])
        reward_hist.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))
        step_count += 1
        if max_steps is not None and step_count >= int(max_steps):
            break

    return {
        "observations": np.asarray(observations, dtype=float),
        "actions": np.asarray(action_hist, dtype=float).reshape(-1, 1),
        "rewards": np.asarray(reward_hist, dtype=float),
        "absolute_time_step": np.asarray(absolute_time_step, dtype=int),
    }


def _observation_value(env, observations, name: str, default: float = np.nan) -> float:
    building = env.unwrapped.buildings[0]
    values = np.asarray(observations[0] if isinstance(observations, (list, tuple)) else observations, dtype=float)
    names = _observation_names_for_values(env, values)
    if name in names:
        return float(values[names.index(name)])
    if name == "indoor_dry_bulb_temperature":
        return float(building.energy_simulation.indoor_dry_bulb_temperature[building.time_step])
    if name == "indoor_dry_bulb_temperature_heating_set_point":
        return float(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[building.time_step])
    return float(default)


def _observation_names_for_values(env, values) -> list[str]:
    """Return the observation-name list that matches the current observation vector."""

    values = np.asarray(values)
    wrapper_names = getattr(env, "observation_names", None)
    if wrapper_names:
        names = list(wrapper_names[0])
        if len(names) == values.size:
            return names

    building = env.unwrapped.buildings[0]
    names = list(getattr(building, "active_observations", []))
    if len(names) == values.size:
        return names

    return names[: values.size]


def _policy_distribution_diagnostics(agent, observations) -> dict[str, float]:
    if not getattr(agent, "normalized", [False])[0]:
        return {
            "policy_mean_action": np.nan,
            "policy_raw_mean": np.nan,
            "policy_log_std": np.nan,
            "policy_std": np.nan,
        }
    try:
        import torch

        obs = observations[0] if isinstance(observations, (list, tuple)) else observations
        encoded = agent.get_encoded_observations(0, obs)
        normalized = agent.get_normalized_observations(0, encoded)
        tensor = torch.FloatTensor(normalized).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            raw_mean, log_std = agent.policy_net[0].forward(tensor)
            mean_action = torch.tanh(raw_mean) * agent.policy_net[0].action_scale + agent.policy_net[0].action_bias
        return {
            "policy_mean_action": float(mean_action.detach().cpu().numpy().reshape(-1)[0]),
            "policy_raw_mean": float(raw_mean.detach().cpu().numpy().reshape(-1)[0]),
            "policy_log_std": float(log_std.detach().cpu().numpy().reshape(-1)[0]),
            "policy_std": float(log_std.exp().detach().cpu().numpy().reshape(-1)[0]),
        }
    except Exception:
        return {
            "policy_mean_action": np.nan,
            "policy_raw_mean": np.nan,
            "policy_log_std": np.nan,
            "policy_std": np.nan,
        }


def _structured_exploration_rate(
    episode: int,
    *,
    first_sac_episode: int,
    last_episode: int,
    start: float,
    end: float,
) -> float:
    if episode < first_sac_episode:
        return 0.0
    denom = max(int(last_episode) - int(first_sac_episode), 1)
    frac = np.clip((int(episode) - int(first_sac_episode)) / denom, 0.0, 1.0)
    return float((1.0 - frac) * float(start) + frac * float(end))


def _apply_training_action_fixes(
    *,
    action: float,
    env,
    observations,
    rng: np.random.Generator,
    phase: str,
    episode: int,
    training_episodes: int,
    structured_exploration_start: float,
    structured_exploration_end: float,
    structured_exploration_cold_min_action: float,
    structured_exploration_beta_alpha: float,
    structured_exploration_beta_beta: float,
    comfort_shield_margin: float | None,
    comfort_shield_min_action: float,
    comfort_shield_kp: float,
    comfort_shield_hot_margin: float | None,
    comfort_shield_hot_max_action: float,
    enable_structured_exploration: bool,
    enable_comfort_shield: bool,
) -> dict[str, Any]:
    tin = _observation_value(env, observations, "indoor_dry_bulb_temperature")
    tsp = _observation_value(env, observations, "indoor_dry_bulb_temperature_heating_set_point", default=20.0)
    raw_action = float(np.clip(action, 0.0, 1.0))
    final_action = raw_action
    exploration_rate = 0.0
    exploration_action = np.nan
    exploration_applied = False
    shield_action = np.nan
    shield_applied = False

    if enable_structured_exploration and phase == "sac_training":
        exploration_rate = _structured_exploration_rate(
            episode,
            first_sac_episode=1,
            last_episode=max(int(training_episodes) - 1, 1),
            start=structured_exploration_start,
            end=structured_exploration_end,
        )
        if float(rng.random()) < exploration_rate:
            if np.isfinite(tin) and np.isfinite(tsp) and tin <= tsp:
                exploration_action = float(rng.uniform(float(structured_exploration_cold_min_action), 1.0))
            else:
                exploration_action = float(
                    rng.beta(
                        max(float(structured_exploration_beta_alpha), 1e-6),
                        max(float(structured_exploration_beta_beta), 1e-6),
                    )
                )
            final_action = float(np.clip(exploration_action, 0.0, 1.0))
            exploration_applied = True

    if enable_comfort_shield and comfort_shield_margin is not None:
        cold_error = float(tsp - tin) if np.isfinite(tin) and np.isfinite(tsp) else -np.inf
        if cold_error > float(comfort_shield_margin):
            shield_action = max(
                float(comfort_shield_min_action),
                float(comfort_shield_kp) * cold_error,
            )
            shield_action = float(np.clip(shield_action, 0.0, 1.0))
            shielded = max(final_action, shield_action)
            shield_applied = bool(shielded > final_action + 1e-12)
            final_action = float(shielded)
        hot_margin = (
            float(comfort_shield_margin)
            if comfort_shield_hot_margin is None
            else float(comfort_shield_hot_margin)
        )
        hot_error = float(tin - tsp) if np.isfinite(tin) and np.isfinite(tsp) else -np.inf
        if hot_error > hot_margin:
            hot_cap = float(np.clip(comfort_shield_hot_max_action, 0.0, 1.0))
            capped = min(final_action, hot_cap)
            shield_applied = shield_applied or bool(capped < final_action - 1e-12)
            final_action = float(capped)
            shield_action = final_action

    if shield_applied and exploration_applied:
        source = "structured_exploration+comfort_shield"
    elif shield_applied:
        source = "comfort_shield"
    elif exploration_applied:
        source = "structured_exploration"
    else:
        source = phase

    return {
        "raw_action": raw_action,
        "action": final_action,
        "tin": tin,
        "tsp": tsp,
        "exploration_rate": exploration_rate,
        "structured_exploration_applied": exploration_applied,
        "structured_exploration_action": exploration_action,
        "comfort_shield_applied": shield_applied,
        "comfort_shield_action": shield_action,
        "action_source": source,
    }


def _apply_safe_pi_exploration_action(
    *,
    action: float,
    env,
    observations,
    rng: np.random.Generator,
    cold_margin: float,
    hot_margin: float,
    cold_min_action: float,
    cold_kp: float,
    hot_max_action: float,
    random_prob: float,
    random_low: float,
    random_high: float,
    ramp_limit: float | None,
    previous_action: float | None,
) -> dict[str, Any]:
    tin = _observation_value(env, observations, "indoor_dry_bulb_temperature")
    tsp = _observation_value(env, observations, "indoor_dry_bulb_temperature_heating_set_point", default=20.0)
    raw_action = float(np.clip(action, 0.0, 1.0))
    final_action = raw_action
    reasons: list[str] = []
    random_action = np.nan

    if (
        float(random_prob) > 0.0
        and np.isfinite(tin)
        and np.isfinite(tsp)
        and (tsp - float(cold_margin)) <= tin <= (tsp + float(hot_margin))
        and float(rng.random()) < float(random_prob)
    ):
        low = float(np.clip(random_low, 0.0, 1.0))
        high = float(np.clip(max(random_high, low), 0.0, 1.0))
        random_action = float(rng.uniform(low, high))
        final_action = random_action
        reasons.append("random_excitation")

    if np.isfinite(tin) and np.isfinite(tsp):
        cold_error = float(tsp - float(cold_margin) - tin)
        if cold_error > 0.0:
            floor = float(cold_min_action) + float(cold_kp) * cold_error
            final_action = max(final_action, float(np.clip(floor, 0.0, 1.0)))
            reasons.append("cold_floor")

        hot_error = float(tin - tsp - float(hot_margin))
        if hot_error > 0.0:
            final_action = min(final_action, float(np.clip(hot_max_action, 0.0, 1.0)))
            reasons.append("hot_cap")

    if ramp_limit is not None and previous_action is not None and np.isfinite(previous_action):
        limit = abs(float(ramp_limit))
        final_action = float(np.clip(final_action, previous_action - limit, previous_action + limit))
        reasons.append("ramp_limit")

    final_action = float(np.clip(final_action, 0.0, 1.0))
    applied = bool(abs(final_action - raw_action) > 1e-12 or np.isfinite(random_action))
    return {
        "raw_action": raw_action,
        "action": final_action,
        "tin": tin,
        "tsp": tsp,
        "exploration_rate": float(random_prob),
        "structured_exploration_applied": False,
        "structured_exploration_action": np.nan,
        "comfort_shield_applied": applied,
        "comfort_shield_action": final_action if applied else np.nan,
        "safe_pi_applied": applied,
        "safe_pi_action": final_action,
        "safe_pi_random_action": random_action,
        "safe_pi_reason": "+".join(reasons) if reasons else "pi",
        "action_source": "safe_pi_" + ("+".join(reasons) if reasons else "pi"),
    }


def _clone_agent_eval_state(agent) -> dict[str, Any]:
    """Snapshot SAC policy/normalization state so validation rollouts are non-mutating."""

    import torch

    state: dict[str, Any] = {
        "time_step": int(agent.time_step),
        "standardize_start_time_step": int(agent.standardize_start_time_step),
        "end_exploration_time_step": int(agent.end_exploration_time_step),
        "normalized": deepcopy(agent.normalized),
        "norm_mean": deepcopy(agent.norm_mean),
        "norm_std": deepcopy(agent.norm_std),
        "r_norm_mean": deepcopy(agent.r_norm_mean),
        "r_norm_std": deepcopy(agent.r_norm_std),
        "alpha": float(getattr(agent, "alpha", np.nan)),
        "log_alpha": [],
    }
    for name in [
        "soft_q_net1",
        "soft_q_net2",
        "target_soft_q_net1",
        "target_soft_q_net2",
        "policy_net",
    ]:
        nets = getattr(agent, name)
        state[name] = [
            None if net is None else {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            for net in nets
        ]
    for value in getattr(agent, "log_alpha", []):
        state["log_alpha"].append(None if value is None else float(value.detach().cpu().item()))
    # Keep torch imported here so static linters do not complain about the lazy import above.
    _ = torch
    return state


def _restore_agent_eval_state(agent, state: dict[str, Any]) -> None:
    """Restore a state captured by _clone_agent_eval_state."""

    agent.time_step = int(state["time_step"])
    agent.standardize_start_time_step = int(state["standardize_start_time_step"])
    agent.end_exploration_time_step = int(state["end_exploration_time_step"])
    for name in ["normalized", "norm_mean", "norm_std", "r_norm_mean", "r_norm_std"]:
        setattr(agent, name, deepcopy(state[name]))
    for name in [
        "soft_q_net1",
        "soft_q_net2",
        "target_soft_q_net1",
        "target_soft_q_net2",
        "policy_net",
    ]:
        nets = getattr(agent, name)
        for index, net_state in enumerate(state.get(name, [])):
            if net_state is None or index >= len(nets) or nets[index] is None:
                continue
            nets[index].load_state_dict({k: v.to(agent.device) for k, v in net_state.items()})
    if "alpha" in state:
        agent.alpha = float(state["alpha"])
    for index, value in enumerate(state.get("log_alpha", [])):
        if value is None or index >= len(getattr(agent, "log_alpha", [])) or agent.log_alpha[index] is None:
            continue
        with __import__("torch").no_grad():
            agent.log_alpha[index].data.fill_(float(value))


def _apply_residual_safe_pi_action(
    *,
    policy_action: float,
    base_action: float,
    env,
    observations,
    rng: np.random.Generator,
    residual_scale: float,
    cold_margin: float,
    hot_margin: float,
    cold_min_action: float,
    cold_kp: float,
    hot_max_action: float,
    ramp_limit: float | None,
    previous_action: float | None,
) -> dict[str, Any]:
    """Use safe PI as the base action and let SAC learn a bounded residual."""

    base_info = _apply_safe_pi_exploration_action(
        action=float(base_action),
        env=env,
        observations=observations,
        rng=rng,
        cold_margin=cold_margin,
        hot_margin=hot_margin,
        cold_min_action=cold_min_action,
        cold_kp=cold_kp,
        hot_max_action=hot_max_action,
        random_prob=0.0,
        random_low=0.0,
        random_high=0.0,
        ramp_limit=ramp_limit,
        previous_action=previous_action,
    )
    raw_policy_action = float(np.clip(policy_action, 0.0, 1.0))
    residual_delta = float(residual_scale) * (raw_policy_action - 0.5)
    final_action = float(np.clip(float(base_info["action"]) + residual_delta, 0.0, 1.0))
    return {
        "raw_action": raw_policy_action,
        "action": final_action,
        "tin": float(base_info["tin"]),
        "tsp": float(base_info["tsp"]),
        "exploration_rate": 0.0,
        "structured_exploration_applied": False,
        "structured_exploration_action": np.nan,
        "comfort_shield_applied": bool(abs(final_action - float(base_info["action"])) > 1e-12),
        "comfort_shield_action": final_action,
        "safe_pi_applied": bool(base_info.get("safe_pi_applied", False)),
        "safe_pi_action": float(base_info["action"]),
        "safe_pi_random_action": np.nan,
        "safe_pi_reason": str(base_info.get("safe_pi_reason", "pi")),
        "residual_safe_pi_base_action": float(base_info["action"]),
        "residual_safe_pi_delta": residual_delta,
        "action_source": f"residual_safe_pi_{base_info.get('safe_pi_reason', 'pi')}",
    }


def _reset_pi_integral_if_needed(controller, reason: str) -> None:
    if not reason or reason == "pi" or reason == "random_excitation":
        return
    if hasattr(controller, "integral_errors"):
        controller.integral_errors = {}


def _transition_safe_for_replay(
    *,
    env,
    observations,
    next_observations,
    reward,
    temperature_min: float | None,
    temperature_max: float | None,
    temperature_delta_max: float | None,
    reward_abs_max: float | None,
) -> tuple[bool, str]:
    tin = _observation_value(env, observations, "indoor_dry_bulb_temperature")
    next_tin = _observation_value(env, next_observations, "indoor_dry_bulb_temperature")
    reward_value = float(np.sum(reward))
    values = [tin, next_tin, reward_value]
    if not all(np.isfinite(v) for v in values):
        return False, "non_finite"
    if temperature_min is not None and min(tin, next_tin) < float(temperature_min):
        return False, "temperature_below_min"
    if temperature_max is not None and max(tin, next_tin) > float(temperature_max):
        return False, "temperature_above_max"
    if temperature_delta_max is not None and abs(next_tin - tin) > float(temperature_delta_max):
        return False, "temperature_jump"
    if reward_abs_max is not None and abs(reward_value) > float(reward_abs_max):
        return False, "reward_outlier"
    return True, "kept"


def _set_temperature_in_observation(env, observations, temperature: float):
    values = observations[0] if isinstance(observations, (list, tuple)) else observations
    names = _observation_names_for_values(env, values)
    if "indoor_dry_bulb_temperature" not in names:
        return observations

    idx = names.index("indoor_dry_bulb_temperature")
    if isinstance(observations, tuple):
        obs_out = [np.asarray(o, dtype=float).copy() for o in observations]
        obs_out[0][idx] = float(temperature)
        return tuple(obs_out)
    if isinstance(observations, list):
        obs_out = [np.asarray(o, dtype=float).copy() for o in observations]
        obs_out[0][idx] = float(temperature)
        return obs_out

    obs_out = np.asarray(observations, dtype=float).copy()
    obs_out[idx] = float(temperature)
    return obs_out


def _apply_training_temperature_reset(
    *,
    env,
    next_observations,
    cold_margin: float,
    hot_margin: float,
    target_margin: float,
    absolute_min: float | None,
    absolute_max: float | None,
    reset_hidden: bool,
) -> tuple[Any, dict[str, Any]]:
    """Keep January training rollouts inside the LSTM's sane thermal range.

    This is deliberately training-only. The deployment rollout is untouched; the
    reset just prevents one bad exploration action from poisoning every later
    replay transition through a runaway recurrent thermal state.
    """

    building = env.unwrapped.buildings[0]
    tin = _observation_value(env, next_observations, "indoor_dry_bulb_temperature")
    tsp = _observation_value(
        env,
        next_observations,
        "indoor_dry_bulb_temperature_heating_set_point",
        default=20.0,
    )
    info = {
        "temperature_reset_applied": False,
        "temperature_reset_reason": "",
        "temperature_reset_raw_tin": float(tin),
        "temperature_reset_tin": float(tin),
    }
    if not (np.isfinite(tin) and np.isfinite(tsp)):
        safe_tin = float(tsp if np.isfinite(tsp) else 20.0)
        reason = "non_finite"
    else:
        trigger_low = float(tsp) - float(cold_margin)
        trigger_high = float(tsp) + float(hot_margin)
        if absolute_min is not None:
            trigger_low = max(trigger_low, float(absolute_min))
        if absolute_max is not None:
            trigger_high = min(trigger_high, float(absolute_max))

        if trigger_low <= float(tin) <= trigger_high:
            return next_observations, info

        target_low = float(tsp) - float(target_margin)
        target_high = float(tsp) + float(target_margin)
        if absolute_min is not None:
            target_low = max(target_low, float(absolute_min))
        if absolute_max is not None:
            target_high = min(target_high, float(absolute_max))
        safe_tin = float(np.clip(float(tin), target_low, target_high))
        reason = "below_training_guard" if float(tin) < trigger_low else "above_training_guard"

    building.energy_simulation.indoor_dry_bulb_temperature[building.time_step] = safe_tin
    if hasattr(building, "dynamics") and getattr(building.dynamics, "_model_input", None) is not None:
        try:
            ix = building.dynamics.input_observation_names.index("indoor_dry_bulb_temperature")
            low = float(building.dynamics.input_normalization_minimum[ix])
            high = float(building.dynamics.input_normalization_maximum[ix])
            if high > low:
                building.dynamics._model_input[ix][-1] = (safe_tin - low) / (high - low)
            if reset_hidden and hasattr(building.dynamics, "init_hidden"):
                building.dynamics._hidden_state = building.dynamics.init_hidden(1)
        except Exception:
            pass

    safe_observations = _set_temperature_in_observation(env, next_observations, safe_tin)
    info.update(
        {
            "temperature_reset_applied": True,
            "temperature_reset_reason": reason,
            "temperature_reset_tin": safe_tin,
        }
    )
    return safe_observations, info


def _normalize_replay_buffer_for_agent(agent, index: int = 0) -> None:
    if agent.normalized[index] or len(agent.replay_buffer[index]) == 0:
        return
    if hasattr(agent, "standardize_replay_buffer"):
        agent.standardize_replay_buffer(index)
        return
    X = np.array([j[0] for j in agent.replay_buffer[index].buffer], dtype=float)
    agent.norm_mean[index] = np.nanmean(X, axis=0)
    agent.norm_std[index] = np.nanstd(X, axis=0) + 1e-5
    R = np.array([j[2] for j in agent.replay_buffer[index].buffer], dtype=float)
    agent.r_norm_mean[index] = np.nanmean(R, dtype=float)
    agent.r_norm_std[index] = np.nanstd(R, dtype=float) / agent.reward_scaling + 1e-5
    agent.replay_buffer[index].buffer = [
        (
            np.hstack(agent.get_normalized_observations(index, o).reshape(1, -1)[0]),
            a,
            agent.get_normalized_reward(index, r),
            np.hstack(agent.get_normalized_observations(index, n).reshape(1, -1)[0]),
            d,
        )
        for o, a, r, n, d in agent.replay_buffer[index].buffer
    ]
    agent.normalized[index] = True


def _sac_policy_ready_for_prediction(agent, index: int = 0) -> bool:
    if bool(agent.normalized[index]) and agent.norm_mean[index] is not None and agent.norm_std[index] is not None:
        return True
    if len(agent.replay_buffer[index]) >= int(agent.batch_size):
        _normalize_replay_buffer_for_agent(agent, index=index)
        return bool(agent.normalized[index]) and agent.norm_mean[index] is not None and agent.norm_std[index] is not None
    return False


def behavior_clone_sac_actor(
    agent,
    *,
    expert_observations: np.ndarray,
    expert_actions: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    validation_fraction: float = 0.2,
    early_stopping_patience: int = 5,
    min_delta: float = 1e-6,
    seed: int = 49,
) -> list[dict[str, float]]:
    """Supervised actor pretraining against teacher actions."""

    import torch

    if int(epochs) <= 0:
        return []

    set_experiment_seed(seed)
    index = 0
    _normalize_replay_buffer_for_agent(agent, index=index)
    raw_obs = np.asarray(expert_observations, dtype=float)
    actions = np.asarray(expert_actions, dtype=float).reshape(-1, agent.action_dimension[index])
    if raw_obs.shape[0] != actions.shape[0]:
        raise ValueError("expert_observations and expert_actions must have the same length.")
    if raw_obs.shape[0] == 0:
        raise ValueError("Behavior-cloning expert dataset is empty.")

    encoded = np.asarray([agent.get_encoded_observations(index, o) for o in raw_obs], dtype=float)
    if not agent.normalized[index]:
        agent.norm_mean[index] = np.nanmean(encoded, axis=0)
        agent.norm_std[index] = np.nanstd(encoded, axis=0) + 1e-5
        agent.r_norm_mean[index] = 0.0
        agent.r_norm_std[index] = 1.0
        agent.normalized[index] = True
    normalized = (encoded - agent.norm_mean[index]) / agent.norm_std[index]
    action_low = np.asarray(agent.action_space[index].low, dtype=float).reshape(1, -1)
    action_high = np.asarray(agent.action_space[index].high, dtype=float).reshape(1, -1)
    actions = np.clip(actions, action_low, action_high)

    device = agent.device
    x = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    y = torch.as_tensor(actions, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(agent.policy_net[index].parameters(), lr=float(lr))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    n = int(x.shape[0])
    batch_size = int(max(1, min(batch_size, n)))
    val_count = int(round(n * max(0.0, min(float(validation_fraction), 0.8))))
    if n - val_count < 1:
        val_count = 0
    split_order = torch.randperm(n, generator=generator)
    val_idx = split_order[:val_count].to(device)
    train_idx = split_order[val_count:].to(device)
    if train_idx.numel() == 0:
        train_idx = split_order.to(device)
        val_idx = torch.empty(0, dtype=torch.long, device=device)
    best_val_loss = float("inf")
    best_state = None
    stale_epochs = 0
    patience = max(0, int(early_stopping_patience))
    history = []

    for epoch in range(int(epochs)):
        order = train_idx[torch.randperm(int(train_idx.numel()), generator=generator).to(device)]
        losses = []
        mae_values = []
        max_abs_values = []
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start:start + batch_size]
            xb = x[idx]
            yb = y[idx]
            _, _, pred = agent.policy_net[index].sample(xb)
            err = pred - yb
            loss = torch.mean(err**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            mae_values.append(float(torch.mean(torch.abs(err)).detach().cpu().item()))
            max_abs_values.append(float(torch.max(torch.abs(err)).detach().cpu().item()))
        with torch.no_grad():
            _, _, train_pred = agent.policy_net[index].sample(x[train_idx])
            train_err = train_pred - y[train_idx]
            train_full_loss = float(torch.mean(train_err**2).detach().cpu().item())
            train_full_mae = float(torch.mean(torch.abs(train_err)).detach().cpu().item())
            train_full_max_abs = float(torch.max(torch.abs(train_err)).detach().cpu().item())
            if val_idx.numel() > 0:
                _, _, val_pred = agent.policy_net[index].sample(x[val_idx])
                val_err = val_pred - y[val_idx]
                val_loss = float(torch.mean(val_err**2).detach().cpu().item())
                val_mae = float(torch.mean(torch.abs(val_err)).detach().cpu().item())
                val_max_abs = float(torch.max(torch.abs(val_err)).detach().cpu().item())
            else:
                val_loss = train_full_loss
                val_mae = train_full_mae
                val_max_abs = train_full_max_abs

        improved = val_loss < best_val_loss - float(min_delta)
        if improved:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in agent.policy_net[index].state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": int(epoch + 1),
                "bc_loss_mean": float(np.mean(losses)),
                "bc_loss_max": float(np.max(losses)),
                "bc_mae_mean": float(np.mean(mae_values)),
                "bc_mae_max": float(np.max(mae_values)),
                "bc_abs_error_max": float(np.max(max_abs_values)),
                "bc_train_loss": train_full_loss,
                "bc_train_mae": train_full_mae,
                "bc_train_abs_error_max": train_full_max_abs,
                "bc_val_loss": val_loss,
                "bc_val_mae": val_mae,
                "bc_val_abs_error_max": val_max_abs,
                "bc_best_val_loss": float(best_val_loss),
                "bc_improved": bool(improved),
                "bc_stale_epochs": int(stale_epochs),
                "bc_early_stopped": bool(patience > 0 and stale_epochs >= patience),
                "bc_train_samples": int(train_idx.numel()),
                "bc_val_samples": int(val_idx.numel()),
            }
        )
        if patience > 0 and stale_epochs >= patience:
            break

    if best_state is not None:
        agent.policy_net[index].load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return history


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
    if reward_mode == "balanced":
        return BalancedComfortEnergyReward
    if reward_mode == "lagrangian":
        return LagrangianDistrictReward
    if reward_mode == "occupant_prob":
        return ExpectedOccupantOverrideReward
    if reward_mode == "dynamic_comfort":
        return DynamicComfortReward
    if reward_mode == "occupant_comfort":
        return OccupantComfortReward
    if reward_mode == "fixed_comfort":
        return FixedComfortReward
    if reward_mode == "feedback":
        return OccupantFeedbackReward
    raise ValueError(
        "reward_mode must be 'comfort', 'dynamic_comfort', 'fixed_comfort', "
        "'feedback', 'balanced', 'lagrangian', or 'occupant_prob'."
    )


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
    if hasattr(building, "heating_electricity_consumption"):
        data["heating_electricity_consumption"] = arr(building.heating_electricity_consumption)

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
    rollout_optional = {
        "policy_action": "policy_action_hist",
        "policy_mean_action": "policy_mean_action_hist",
        "policy_raw_mean": "policy_raw_mean_hist",
        "policy_log_std": "policy_log_std_hist",
        "policy_std": "policy_std_hist",
        "action_source": "action_source_hist",
        "comfort_shield_applied": "comfort_shield_applied_hist",
        "comfort_shield_action": "comfort_shield_action_hist",
        "tin_before_action": "tin_before_action_hist",
        "tsp_before_action": "tsp_before_action_hist",
    }
    for column, key in rollout_optional.items():
        if key in result and len(result[key]) == n_steps:
            history[column] = np.asarray(result[key])
    if "net_electricity_consumption" in result:
        history["net_electricity_consumption"] = result["net_electricity_consumption"]
    if "heating_electricity_consumption" in result:
        history["heating_electricity_consumption"] = result["heating_electricity_consumption"]
    optional = {
        "reference_setpoint": "reference_setpoints",
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
    comfort_low = float(result.get("comfort_band_lower", 20.0))
    comfort_high = float(result.get("comfort_band_upper", 24.0))
    summary = {
        "case": label,
        "override_count": int(result.get("override_count", 0)),
        "override_hours": override_hours(
            result.get("baseline_setpoints", []),
            result.get("effective_setpoints", []),
        ),
        "reward_total": float(np.asarray(result["reward_hist"], dtype=float).sum()) if len(result["reward_hist"]) else 0.0,
        "kelvin_hours": kelvin_hours(result.get("Tin_hist", []), lower=comfort_low, upper=comfort_high),
        "setpoint_kelvin_hours": setpoint_kelvin_hours(
            result.get("Tin_hist", []),
            result.get("effective_setpoints", []),
        ),
        "electricity_consumption_total": _get_citylearn_kpi(result["district_kpis"], "electricity_consumption_total"),
        "discomfort_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_proportion"),
        "discomfort_cold_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_cold_proportion"),
        "discomfort_hot_proportion": _get_citylearn_kpi(result["district_kpis"], "discomfort_hot_proportion"),
        "daily_peak_average": _get_citylearn_kpi(result["district_kpis"], "daily_peak_average"),
    }
    if online_info:
        summary.update(online_info)
    if "building_name" in result:
        summary["building_name"] = result["building_name"]
    if "target_temperature" in result:
        summary["target_temperature"] = float(result["target_temperature"])
    if "comfort_band_lower" in result:
        summary["comfort_band_lower"] = float(result["comfort_band_lower"])
    if "comfort_band_upper" in result:
        summary["comfort_band_upper"] = float(result["comfort_band_upper"])
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
    return update_tdyn_reference(
        value,
        feedback,
        preferred_temperature=float(occupant.T_pref),
        config=TDynConfig(
            delta_up=float(delta_up),
            delta_down=float(delta_down),
            drift_to_pref=float(drift_to_pref),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            anchor_feedback_to_pref=True,
        ),
    )


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


def _actual_occupancy_gate(occ_now: float, alpha_occ: float) -> float:
    return float(float(occ_now) > float(alpha_occ))


def _blend_tdyn_reference(scheduled: float, T_dyn: float, occupancy_gate: float) -> float:
    gate = float(np.clip(occupancy_gate, 0.0, 1.0))
    return (1.0 - gate) * float(scheduled) + gate * float(T_dyn)


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

    obs_array = np.asarray(observations[0], dtype=float).copy()
    names = _observation_names_for_values(env, obs_array)
    if "indoor_dry_bulb_temperature" in names:
        tin = float(obs_array[names.index("indoor_dry_bulb_temperature")])
    else:
        tin = float(building.energy_simulation.indoor_dry_bulb_temperature[time_step])
    for obs_name in (
        "indoor_dry_bulb_temperature_heating_set_point",
        "indoor_dry_bulb_temperature_cooling_set_point",
    ):
        if obs_name in names:
            obs_array[names.index(obs_name)] = float(reference)
    for obs_name in (
        "indoor_dry_bulb_temperature_heating_delta",
        "indoor_dry_bulb_temperature_cooling_delta",
    ):
        if obs_name in names:
            obs_array[names.index(obs_name)] = tin - float(reference)
    return [obs_array]


def _set_reward_reference_setpoint(env, reference: float | None) -> None:
    reward_function = getattr(env.unwrapped, "reward_function", None)
    if reward_function is None:
        return
    if hasattr(reward_function, "set_current_reference_setpoints"):
        reward_function.set_current_reference_setpoints([None if reference is None else float(reference)])
    if reference is None:
        if hasattr(reward_function, "_t_dyn"):
            reward_function._t_dyn = [None]
    else:
        reward_function._t_dyn = [float(reference)]


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
        "automatic_entropy_tuning": bool(getattr(agent, "automatic_entropy_tuning", False)),
        "alpha": float(getattr(agent, "alpha", np.nan)),
        "log_alpha": [
            None if value is None else float(value.detach().cpu().item())
            for value in getattr(agent, "log_alpha", [])
        ],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def _load_agent_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _load_agent_artifact(agent, path: str | Path) -> dict[str, Any]:
    """Restore a saved SAC state into a freshly constructed CityLearn agent."""

    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Older artifacts may contain CUDA tensors. Pickle dispatches those through
    # torch.storage._load_from_bytes, so temporarily map all tensor storages to
    # the current agent device while reading the payload.
    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(  # type: ignore[assignment]
        io.BytesIO(b),
        map_location=agent.device,
        weights_only=False,
    )
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    finally:
        torch.storage._load_from_bytes = original_loader  # type: ignore[assignment]

    agent.time_step = int(payload.get("time_step", agent.time_step))
    agent.standardize_start_time_step = int(
        payload.get("standardize_start_time_step", agent.standardize_start_time_step)
    )
    agent.end_exploration_time_step = int(
        payload.get("end_exploration_time_step", agent.end_exploration_time_step)
    )
    for name in ["normalized", "norm_mean", "norm_std", "r_norm_mean", "r_norm_std"]:
        if name in payload:
            setattr(agent, name, payload[name])

    for index, buffer in enumerate(payload.get("replay_buffer", [])):
        if index >= len(agent.replay_buffer):
            break
        agent.replay_buffer[index].buffer = list(buffer)
    for index, position in enumerate(payload.get("replay_position", [])):
        if index >= len(agent.replay_buffer):
            break
        agent.replay_buffer[index].position = int(position)

    for name in [
        "soft_q_net1",
        "soft_q_net2",
        "target_soft_q_net1",
        "target_soft_q_net2",
        "policy_net",
    ]:
        for index, state in enumerate(payload.get(name, [])):
            nets = getattr(agent, name)
            if index >= len(nets) or nets[index] is None:
                continue
            nets[index].load_state_dict({k: v.to(agent.device) for k, v in state.items()})

    if payload.get("automatic_entropy_tuning", False) and getattr(agent, "automatic_entropy_tuning", False):
        for index, value in enumerate(payload.get("log_alpha", [])):
            if value is None or index >= len(agent.log_alpha) or agent.log_alpha[index] is None:
                continue
            with torch.no_grad():
                agent.log_alpha[index].data.fill_(float(value))
    if "alpha" in payload:
        agent.alpha = float(payload["alpha"])

    return payload


def _trim_agent_replay_keep_last(agent, keep_last: int | None) -> list[dict[str, int]]:
    """Keep only the latest replay transitions after loading a resumed agent."""

    if keep_last is None:
        return []
    keep_last = int(keep_last)
    if keep_last < 0:
        raise ValueError("resume_replay_keep_last must be non-negative.")

    stats: list[dict[str, int]] = []
    for index, buffer in enumerate(agent.replay_buffer):
        before = len(buffer.buffer)
        if keep_last == 0:
            buffer.buffer = []
            buffer.position = 0
        elif before > keep_last:
            if before >= int(buffer.capacity):
                ordered = list(buffer.buffer[int(buffer.position):]) + list(buffer.buffer[: int(buffer.position)])
            else:
                ordered = list(buffer.buffer)
            buffer.buffer = ordered[-keep_last:]
            buffer.position = len(buffer.buffer) % int(buffer.capacity)
        else:
            buffer.position = len(buffer.buffer) % int(buffer.capacity)
        stats.append(
            {
                "index": int(index),
                "before": int(before),
                "after": int(len(buffer.buffer)),
                "position": int(buffer.position),
            }
        )
    return stats


def train_january_pi_jumpstart(
    *,
    output_path: str | Path = "results/models/sac_merlin/january_pi_agent.pkl",
    seed: int = 49,
    reward_function=ComfortReward,
    max_steps: int | None = None,
    config: MerlinSACConfig = MerlinSACConfig(),
    building_name: str = CONTROL_BUILDING_NAME,
) -> dict[str, Any]:
    """Run January PI exploration and save the SACRBC agent with replay."""

    set_experiment_seed(seed)
    env = make_env(
        sim_start=JAN_START,
        sim_end=JAN_END,
        seed=seed,
        reward_function=reward_function,
        building_name=building_name,
    )
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
    drift_to_pref: float = 0.05,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
    label: str = "SAC",
    building_name: str = CONTROL_BUILDING_NAME,
    comfort_shield_margin: float | None = None,
    comfort_shield_min_action: float = 0.35,
    comfort_shield_kp: float = 0.25,
    comfort_shield_hot_margin: float | None = None,
    comfort_shield_hot_max_action: float = 0.0,
    apply_comfort_shield: bool = False,
    residual_safe_pi: bool = False,
    residual_safe_pi_scale: float = 0.5,
    safe_pi_cold_margin: float = 1.0,
    safe_pi_hot_margin: float = 1.0,
    safe_pi_cold_min_action: float = 0.25,
    safe_pi_cold_kp: float = 0.25,
    safe_pi_hot_max_action: float = 0.0,
    safe_pi_ramp_limit: float | None = None,
    rc_rollout_horizons: tuple[int, ...] | None = None,
    rc_rollout_model_dir: str = "results/models/rc",
    observation_filter: str | None = None,
    sim_start: int = FEB_START,
    sim_end: int = FEB_END,
    mutate_agent_clock: bool = True,
) -> dict[str, Any]:
    set_experiment_seed(seed)
    env = make_env(
        sim_start=sim_start,
        sim_end=sim_end,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
    )
    if rc_rollout_horizons:
        from src.occupants.wrappers import RCRolloutObservationWrapper
        env = RCRolloutObservationWrapper(env, rc_model_dir=rc_rollout_model_dir, horizons=rc_rollout_horizons)
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
    env = _apply_observation_filter(env, observation_filter)
    observations, _ = env.reset(seed=seed)
    if bool(mutate_agent_clock):
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
    policy_action_hist: list[float] = []
    policy_mean_action_hist: list[float] = []
    policy_raw_mean_hist: list[float] = []
    policy_log_std_hist: list[float] = []
    policy_std_hist: list[float] = []
    action_source_hist: list[str] = []
    shield_applied_hist: list[bool] = []
    shield_action_hist: list[float] = []
    tin_before_action_hist: list[float] = []
    tsp_before_action_hist: list[float] = []
    phase_hist: list[str] = []
    feedback_hist: list[float] = []
    tdyn_hist: list[float] = []
    reference_hist: list[float] = []
    occ_now_hist: list[float] = []
    occ_hat_hist: list[float] = []
    occ_bin_hist: list[float] = []
    steps = 0
    previous_action: float | None = None
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
        reward_reference = None
        if occupant is not None and mode == "bspline_tdyn":
            absolute_time_step = int(sim_start) + steps
            occ_now, occ_hat, occ_bin = _tdyn_gate(
                occupancy_model=occupancy_model,
                occupancy_series=occupancy_series,
                absolute_time_step=absolute_time_step,
                alpha_occ=alpha_occ,
            )
            building = env.unwrapped.buildings[0]
            time_step = int(building.time_step)
            scheduled = float(
                building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                    time_step
                ]
            )
            reward_reference = _blend_tdyn_reference(scheduled, float(T_dyn), occ_bin)
            observations_for_agent = _apply_tdyn_to_observations(env, observations, reward_reference)
        else:
            observations_for_agent = observations
            occ_now = occ_hat = occ_bin = np.nan
        if occupant is None and occupancy_model is not None and occupancy_series is not None:
            absolute_time_step = int(sim_start) + steps
            occ_now = float(occupancy_series[min(int(absolute_time_step), len(occupancy_series) - 1)])
            occ_hat = float(np.asarray(observations_for_agent[0], dtype=float)[-2])
            occ_bin = float(np.asarray(observations_for_agent[0], dtype=float)[-1])

        if use_sac_policy:
            policy_stats = _policy_distribution_diagnostics(agent, observations_for_agent)
            raw_actions = agent.get_post_exploration_prediction(observations_for_agent, deterministic=True)
            phase = "deterministic_eval"
            if residual_safe_pi:
                base_actions = agent.rbc.predict(observations_for_agent)
                action_info = _apply_residual_safe_pi_action(
                    policy_action=float(np.asarray(raw_actions, dtype=float).reshape(-1)[0]),
                    base_action=float(np.asarray(base_actions, dtype=float).reshape(-1)[0]),
                    env=env,
                    observations=observations_for_agent,
                    rng=np.random.default_rng(seed + 10_000 + steps),
                    residual_scale=residual_safe_pi_scale,
                    cold_margin=safe_pi_cold_margin,
                    hot_margin=safe_pi_hot_margin,
                    cold_min_action=safe_pi_cold_min_action,
                    cold_kp=safe_pi_cold_kp,
                    hot_max_action=safe_pi_hot_max_action,
                    ramp_limit=safe_pi_ramp_limit,
                    previous_action=previous_action,
                )
            else:
                action_info = _apply_training_action_fixes(
                    action=float(np.asarray(raw_actions, dtype=float).reshape(-1)[0]),
                    env=env,
                    observations=observations_for_agent,
                    rng=np.random.default_rng(seed + 10_000 + steps),
                    phase=phase,
                    episode=0,
                    training_episodes=1,
                    structured_exploration_start=0.0,
                    structured_exploration_end=0.0,
                    structured_exploration_cold_min_action=0.0,
                    structured_exploration_beta_alpha=1.0,
                    structured_exploration_beta_beta=1.0,
                    comfort_shield_margin=comfort_shield_margin,
                    comfort_shield_min_action=comfort_shield_min_action,
                    comfort_shield_kp=comfort_shield_kp,
                    comfort_shield_hot_margin=comfort_shield_hot_margin,
                    comfort_shield_hot_max_action=comfort_shield_hot_max_action,
                    enable_structured_exploration=False,
                    enable_comfort_shield=apply_comfort_shield,
                )
            actions = [[float(action_info["action"])]]
            if bool(mutate_agent_clock):
                agent.actions = actions
                agent.next_time_step()
        else:
            actions = agent.rbc.predict(observations_for_agent)
            phase = "pi_eval_untrained_smoke"
            policy_stats = _policy_distribution_diagnostics(agent, observations_for_agent)
            action_info = _apply_training_action_fixes(
                action=float(np.asarray(actions, dtype=float).reshape(-1)[0]),
                env=env,
                observations=observations_for_agent,
                rng=np.random.default_rng(seed + 10_000 + steps),
                phase=phase,
                episode=0,
                training_episodes=1,
                structured_exploration_start=0.0,
                structured_exploration_end=0.0,
                structured_exploration_cold_min_action=0.0,
                structured_exploration_beta_alpha=1.0,
                structured_exploration_beta_beta=1.0,
                comfort_shield_margin=None,
                comfort_shield_min_action=comfort_shield_min_action,
                comfort_shield_kp=comfort_shield_kp,
                comfort_shield_hot_margin=None,
                comfort_shield_hot_max_action=comfort_shield_hot_max_action,
                enable_structured_exploration=False,
                enable_comfort_shield=False,
            )
        if occupant is not None and mode == "bspline_tdyn":
            _set_reward_reference_setpoint(env, reward_reference)
        next_observations, reward, terminated, truncated, _ = env.step(actions)
        rewards.append(float(np.sum(reward)))
        actions_hist.append(float(np.asarray(actions, dtype=float).reshape(-1)[0]))
        controlled_actions_hist.append(float(getattr(env, "last_controlled_action", actions_hist[-1])))
        policy_action_hist.append(float(action_info["raw_action"]))
        policy_mean_action_hist.append(float(policy_stats["policy_mean_action"]))
        policy_raw_mean_hist.append(float(policy_stats["policy_raw_mean"]))
        policy_log_std_hist.append(float(policy_stats["policy_log_std"]))
        policy_std_hist.append(float(policy_stats["policy_std"]))
        action_source_hist.append(str(action_info["action_source"]))
        shield_applied_hist.append(bool(action_info["comfort_shield_applied"]))
        shield_action_hist.append(float(action_info["comfort_shield_action"]))
        tin_before_action_hist.append(float(action_info["tin"]))
        tsp_before_action_hist.append(float(action_info["tsp"]))
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
            reference_hist.append(float(reward_reference))
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
        previous_action = float(np.asarray(actions, dtype=float).reshape(-1)[0])
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
        "building_name": building_name,
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
        "policy_action_hist": np.asarray(policy_action_hist, dtype=float),
        "policy_mean_action_hist": np.asarray(policy_mean_action_hist, dtype=float),
        "policy_raw_mean_hist": np.asarray(policy_raw_mean_hist, dtype=float),
        "policy_log_std_hist": np.asarray(policy_log_std_hist, dtype=float),
        "policy_std_hist": np.asarray(policy_std_hist, dtype=float),
        "action_source_hist": np.asarray(action_source_hist, dtype=object),
        "comfort_shield_applied_hist": np.asarray(shield_applied_hist, dtype=bool),
        "comfort_shield_action_hist": np.asarray(shield_action_hist, dtype=float),
        "tin_before_action_hist": np.asarray(tin_before_action_hist, dtype=float),
        "tsp_before_action_hist": np.asarray(tsp_before_action_hist, dtype=float),
        "phase_hist": np.asarray(phase_hist, dtype=object),
        "override_count": int(getattr(env, "override_count", 0)),
        "comfort_band_lower": float(occupant_probability_comfort_band(occupant)[0]),
        "comfort_band_upper": float(occupant_probability_comfort_band(occupant)[1]),
    }
    if "net_electricity_consumption" in arrays:
        result["net_electricity_consumption"] = arrays["net_electricity_consumption"]
    if "heating_electricity_consumption" in arrays:
        result["heating_electricity_consumption"] = arrays["heating_electricity_consumption"]
    if occupant is not None:
        result["feedback_hist"] = np.asarray(feedback_hist, dtype=float)
        result["target_temperature"] = float(occupant.T_pref)
    if len(tdyn_hist):
        result["T_dyn_hist"] = np.asarray(tdyn_hist, dtype=float)
    if len(reference_hist):
        result["reference_setpoints"] = np.asarray(reference_hist, dtype=float)
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
    reward_function_kwargs: dict[str, Any] | None = None,
    eval_reward_function_kwargs: dict[str, Any] | None = None,
    max_steps: int | None = None,
    label: str = "baseline",
    config: MerlinSACConfig = MerlinSACConfig(),
    training_episodes: int = 10,
    building_name: str = CONTROL_BUILDING_NAME,
    building_metadata: dict[str, Any] | None = None,
    occupant=None,
    occupant_mode: str = "without_tdyn",
    occupancy_model=None,
    occupancy_series=None,
    alpha_occ: float = 0.5,
    T_dyn_init_mode: str = "schedule",
    delta_up: float = 0.5,
    delta_down: float = 0.5,
    drift_to_pref: float = 0.05,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
    save_scenario: str = "no_occupant",
    jumpstart_controller: str = "pi",
    jumpstart_actions: list[float] | np.ndarray | None = None,
    bc_expert_observations: np.ndarray | None = None,
    bc_expert_actions: np.ndarray | None = None,
    bc_epochs: int = 0,
    bc_batch_size: int = 128,
    bc_lr: float = 1e-3,
    bc_validation_fraction: float = 0.2,
    bc_early_stopping_patience: int = 5,
    bc_min_delta: float = 1e-6,
    structured_exploration_start: float = 0.0,
    structured_exploration_end: float = 0.0,
    structured_exploration_cold_min_action: float = 0.5,
    structured_exploration_beta_alpha: float = 0.7,
    structured_exploration_beta_beta: float = 0.7,
    comfort_shield_margin: float | None = None,
    comfort_shield_min_action: float = 0.35,
    comfort_shield_kp: float = 0.25,
    comfort_shield_hot_margin: float | None = None,
    comfort_shield_hot_max_action: float = 0.0,
    comfort_shield_eval: bool = False,
    comfort_shield_exploration: bool = False,
    safe_pi_cold_margin: float = 1.0,
    safe_pi_hot_margin: float = 1.0,
    safe_pi_cold_min_action: float = 0.25,
    safe_pi_cold_kp: float = 0.25,
    safe_pi_hot_max_action: float = 0.0,
    safe_pi_random_prob: float = 0.15,
    safe_pi_random_low: float = 0.0,
    safe_pi_random_high: float = 0.55,
    safe_pi_ramp_limit: float | None = None,
    residual_safe_pi: bool = False,
    residual_safe_pi_scale: float = 0.5,
    training_temperature_reset: bool = False,
    training_temperature_reset_cold_margin: float = 4.0,
    training_temperature_reset_hot_margin: float = 4.0,
    training_temperature_reset_target_margin: float = 1.5,
    training_temperature_reset_absolute_min: float | None = 10.0,
    training_temperature_reset_absolute_max: float | None = 30.0,
    training_temperature_reset_hidden: bool = True,
    training_temperature_reset_drop_transition: bool = True,
    training_temperature_reset_replay_mode: str = "drop",
    training_temperature_reset_penalty: float = -1000.0,
    replay_filter_temperature_min: float | None = None,
    replay_filter_temperature_max: float | None = None,
    replay_filter_temperature_delta_max: float | None = None,
    replay_filter_reward_abs_max: float | None = None,
    rc_rollout_horizons: tuple[int, ...] | None = None,
    rc_rollout_model_dir: str = "results/models/rc",
    train_sim_start: int = JAN_START,
    train_sim_end: int = JAN_END,
    eval_sim_start: int = FEB_START,
    eval_sim_end: int = FEB_END,
    resume_agent_path: str | Path | None = None,
    resume_episode_offset: int = 0,
    resume_replay_keep_last: int | None = None,
    early_stop_reward_slope: bool = False,
    early_stop_min_episodes: int = 10,
    early_stop_window: int = 5,
    early_stop_patience: int = 3,
    early_stop_slope_abs: float = 0.25,
    checkpoint_select_by_eval: bool = False,
    checkpoint_select_metric: str = "discomfort_proportion",
) -> dict[str, Any]:
    """Run MERLIN-style jump-start, SAC training, then deployment-period evaluation.

    The first training episode is controlled by either Temperature PI or an
    externally supplied teacher action sequence and fills the SAC replay buffer.
    SAC then trains for the remaining training episodes before a deterministic
    evaluation on the deployment period. This mirrors MERLIN's deployment strategy
    structure more closely than online training directly on the test period.
    """

    set_experiment_seed(seed)
    env = make_env(
        sim_start=train_sim_start,
        sim_end=train_sim_end,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
    )
    if rc_rollout_horizons:
        from src.occupants.wrappers import RCRolloutObservationWrapper
        env = RCRolloutObservationWrapper(env, rc_model_dir=rc_rollout_model_dir, horizons=rc_rollout_horizons)
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
    env = _apply_observation_filter(env, config.observation_filter)
    jumpstart_controller = str(jumpstart_controller).strip().lower()
    if jumpstart_controller not in {"pi", "safe_pi", "mpc"}:
        raise ValueError("jumpstart_controller must be 'pi', 'safe_pi', or 'mpc'.")
    training_temperature_reset_replay_mode = str(training_temperature_reset_replay_mode).strip().lower()
    if training_temperature_reset_replay_mode not in {"drop", "keep", "penalty", "penalty_terminal"}:
        raise ValueError(
            "training_temperature_reset_replay_mode must be 'drop', 'keep', 'penalty', or 'penalty_terminal'."
        )
    # Backward compatibility for older callers that used the boolean drop flag.
    if not training_temperature_reset_drop_transition and training_temperature_reset_replay_mode == "drop":
        training_temperature_reset_replay_mode = "keep"
    exploration_steps = train_sim_end - train_sim_start + 1
    teacher_actions = None
    if jumpstart_controller == "mpc":
        if jumpstart_actions is None:
            raise ValueError("jumpstart_actions are required when jumpstart_controller='mpc'.")
        teacher_actions = np.asarray(jumpstart_actions, dtype=float).reshape(-1)
        if teacher_actions.size == 0:
            raise ValueError("MPC jump-start produced no teacher actions.")
        exploration_steps = min(exploration_steps, int(teacher_actions.size))
    agent = build_merlin_sac_agent(env, exploration_steps=exploration_steps, seed=seed, config=config)
    resume_payload = None
    resume_replay_trim_stats: list[dict[str, int]] = []
    if resume_agent_path is not None:
        resume_payload = _load_agent_artifact(agent, resume_agent_path)
        resume_replay_trim_stats = _trim_agent_replay_keep_last(agent, resume_replay_keep_last)
        # The saved CityLearn time_step belongs to the old training environment.
        # Reset the agent clock for the fresh training window while keeping the
        # learned networks, normalization, and entropy state. The replay buffer
        # may be trimmed above for month-boundary adaptation.
        agent.time_step = 0
        agent.standardize_start_time_step = 0
        agent.end_exploration_time_step = -1
        exploration_steps = 0

    _lagrange_rf = env.unwrapped.reward_function
    if not isinstance(_lagrange_rf, LagrangianDistrictReward):
        _lagrange_rf = None

    online_rewards = []
    phase_hist = []
    action_hist = []
    episode_hist = []
    absolute_time_step_hist = []
    sac_loss_hist = []
    bc_loss_hist = []
    policy_action_hist = []
    policy_mean_action_hist = []
    policy_raw_mean_hist = []
    policy_log_std_hist = []
    policy_std_hist = []
    action_source_hist = []
    exploration_rate_hist = []
    structured_exploration_applied_hist = []
    structured_exploration_action_hist = []
    comfort_shield_applied_hist = []
    comfort_shield_action_hist = []
    tin_before_action_hist = []
    tsp_before_action_hist = []
    safe_pi_applied_hist = []
    safe_pi_reason_hist = []
    temperature_reset_applied_hist = []
    temperature_reset_reason_hist = []
    temperature_reset_raw_tin_hist = []
    temperature_reset_tin_hist = []
    replay_kept_hist = []
    replay_filter_reason_hist = []
    raw_reward_hist = []
    replay_terminal_hist = []
    reset_penalty_applied_hist = []
    tdyn_hist = []
    reference_hist = []
    occ_bin_hist = []
    bc_completed = False
    step_limit_reached = False
    total_steps = 0
    previous_action: float | None = None
    rng = np.random.default_rng(int(seed) + 7531)
    structured_exploration_enabled = (
        float(structured_exploration_start) > 0.0
        or float(structured_exploration_end) > 0.0
    )
    comfort_shield_enabled = comfort_shield_margin is not None
    early_stop_history: list[dict[str, float]] = []
    early_stop_stale_windows = 0
    early_stop_triggered = False
    early_stop_episode: int | None = None
    early_stop_last_slope = np.nan
    checkpoint_eval_rows: list[dict[str, float | int | str]] = []
    best_checkpoint_state: dict[str, Any] | None = None
    best_checkpoint_metric = np.inf
    best_checkpoint_episode: int | None = None

    resume_episode_offset = int(resume_episode_offset)
    for episode_index in range(int(training_episodes)):
        episode_label = resume_episode_offset + int(episode_index)
        observations, _ = env.reset(seed=seed)
        terminated = truncated = False
        episode_step = 0
        previous_action = None
        episode_sac_rewards: list[float] = []
        episode_sac_raw_rewards: list[float] = []
        T_dyn = None
        if occupant is not None and str(occupant_mode).strip().lower() == "bspline_tdyn":
            T_dyn = _tdyn_initial_value(
                env,
                occupant,
                init_mode=T_dyn_init_mode,
                lower_bound=T_dyn_min,
                upper_bound=T_dyn_max,
            )

        while not (terminated or truncated):
            observations_for_agent = observations
            reward_reference = None
            occ_bin = np.nan
            if occupant is not None and str(occupant_mode).strip().lower() == "bspline_tdyn":
                absolute_time_step_for_tdyn = int(train_sim_start) + int(episode_step)
                occ_now_gate, _, occ_bin = _tdyn_gate(
                    occupancy_model=occupancy_model,
                    occupancy_series=occupancy_series,
                    absolute_time_step=absolute_time_step_for_tdyn,
                    alpha_occ=alpha_occ,
                )
                building = env.unwrapped.buildings[0]
                time_step = int(building.time_step)
                scheduled = float(
                    building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                        time_step
                    ]
                )
                reward_reference = _blend_tdyn_reference(scheduled, float(T_dyn), occ_bin)
                observations_for_agent = _apply_tdyn_to_observations(env, observations, reward_reference)
            use_teacher = (
                resume_payload is None
                and
                jumpstart_controller == "mpc"
                and episode_index == 0
                and teacher_actions is not None
                and episode_step < teacher_actions.size
            )
            phase = (
                "mpc_exploration"
                if use_teacher
                else (
                    "safe_pi_exploration"
                    if jumpstart_controller == "safe_pi" and agent.time_step < exploration_steps
                    else ("pi_exploration" if agent.time_step < exploration_steps else "sac_training")
                )
            )
            if phase == "sac_training" and not _sac_policy_ready_for_prediction(agent):
                phase = "safe_pi_bootstrap"
            replay_actions = None
            if use_teacher:
                actions = [[float(np.clip(teacher_actions[episode_step], 0.0, 1.0))]]
                replay_actions = actions
                agent.actions = actions
                agent.next_time_step()
                policy_stats = _policy_distribution_diagnostics(agent, observations_for_agent)
                action_info = _apply_training_action_fixes(
                    action=float(np.asarray(actions, dtype=float).reshape(-1)[0]),
                    env=env,
                    observations=observations_for_agent,
                    rng=rng,
                    phase=phase,
                    episode=episode_label,
                    training_episodes=training_episodes,
                    structured_exploration_start=structured_exploration_start,
                    structured_exploration_end=structured_exploration_end,
                    structured_exploration_cold_min_action=structured_exploration_cold_min_action,
                    structured_exploration_beta_alpha=structured_exploration_beta_alpha,
                    structured_exploration_beta_beta=structured_exploration_beta_beta,
                    comfort_shield_margin=None,
                    comfort_shield_min_action=comfort_shield_min_action,
                    comfort_shield_kp=comfort_shield_kp,
                    comfort_shield_hot_margin=None,
                    comfort_shield_hot_max_action=comfort_shield_hot_max_action,
                    enable_structured_exploration=False,
                    enable_comfort_shield=False,
                )
            else:
                if phase == "sac_training":
                    policy_stats = _policy_distribution_diagnostics(agent, observations_for_agent)
                    raw_actions = agent.get_post_exploration_prediction(observations_for_agent, deterministic=False)
                    if residual_safe_pi:
                        base_actions = agent.rbc.predict(observations_for_agent)
                        action_info = _apply_residual_safe_pi_action(
                            policy_action=float(np.asarray(raw_actions, dtype=float).reshape(-1)[0]),
                            base_action=float(np.asarray(base_actions, dtype=float).reshape(-1)[0]),
                            env=env,
                            observations=observations_for_agent,
                            rng=rng,
                            residual_scale=residual_safe_pi_scale,
                            cold_margin=safe_pi_cold_margin,
                            hot_margin=safe_pi_hot_margin,
                            cold_min_action=safe_pi_cold_min_action,
                            cold_kp=safe_pi_cold_kp,
                            hot_max_action=safe_pi_hot_max_action,
                            ramp_limit=safe_pi_ramp_limit,
                            previous_action=previous_action,
                        )
                        replay_actions = [[float(action_info["raw_action"])]]
                    else:
                        action_info = _apply_training_action_fixes(
                            action=float(np.asarray(raw_actions, dtype=float).reshape(-1)[0]),
                            env=env,
                            observations=observations_for_agent,
                            rng=rng,
                            phase=phase,
                            episode=episode_label,
                            training_episodes=training_episodes,
                            structured_exploration_start=structured_exploration_start,
                            structured_exploration_end=structured_exploration_end,
                            structured_exploration_cold_min_action=structured_exploration_cold_min_action,
                            structured_exploration_beta_alpha=structured_exploration_beta_alpha,
                            structured_exploration_beta_beta=structured_exploration_beta_beta,
                            comfort_shield_margin=comfort_shield_margin,
                            comfort_shield_min_action=comfort_shield_min_action,
                            comfort_shield_kp=comfort_shield_kp,
                            comfort_shield_hot_margin=comfort_shield_hot_margin,
                            comfort_shield_hot_max_action=comfort_shield_hot_max_action,
                            enable_structured_exploration=structured_exploration_enabled,
                            enable_comfort_shield=comfort_shield_enabled,
                        )
                        replay_actions = None
                    actions = [[float(action_info["action"])]]
                    if replay_actions is None:
                        replay_actions = actions
                    agent.actions = actions
                    agent.next_time_step()
                elif phase in {"safe_pi_exploration", "safe_pi_bootstrap"}:
                    if phase == "safe_pi_exploration":
                        actions = agent.predict(observations_for_agent, deterministic=False)
                    else:
                        actions = agent.rbc.predict(observations_for_agent)
                        agent.actions = actions
                        agent.next_time_step()
                    policy_stats = _policy_distribution_diagnostics(agent, observations_for_agent)
                    action_info = _apply_safe_pi_exploration_action(
                        action=float(np.asarray(actions, dtype=float).reshape(-1)[0]),
                        env=env,
                        observations=observations_for_agent,
                        rng=rng,
                        cold_margin=safe_pi_cold_margin,
                        hot_margin=safe_pi_hot_margin,
                        cold_min_action=safe_pi_cold_min_action,
                        cold_kp=safe_pi_cold_kp,
                        hot_max_action=safe_pi_hot_max_action,
                        random_prob=safe_pi_random_prob,
                        random_low=safe_pi_random_low,
                        random_high=safe_pi_random_high,
                        ramp_limit=safe_pi_ramp_limit,
                        previous_action=previous_action,
                    )
                    _reset_pi_integral_if_needed(agent.rbc, str(action_info.get("safe_pi_reason", "")))
                    actions = [[float(action_info["action"])]]
                    replay_actions = actions
                    agent.actions = actions
                else:
                    actions = agent.predict(observations_for_agent, deterministic=False)
                    replay_actions = actions
                    policy_stats = _policy_distribution_diagnostics(agent, observations_for_agent)
                    action_info = _apply_training_action_fixes(
                        action=float(np.asarray(actions, dtype=float).reshape(-1)[0]),
                        env=env,
                        observations=observations_for_agent,
                        rng=rng,
                        phase=phase,
                        episode=episode_label,
                        training_episodes=training_episodes,
                        structured_exploration_start=structured_exploration_start,
                        structured_exploration_end=structured_exploration_end,
                        structured_exploration_cold_min_action=structured_exploration_cold_min_action,
                        structured_exploration_beta_alpha=structured_exploration_beta_alpha,
                        structured_exploration_beta_beta=structured_exploration_beta_beta,
                        comfort_shield_margin=comfort_shield_margin if comfort_shield_exploration else None,
                        comfort_shield_min_action=comfort_shield_min_action,
                        comfort_shield_kp=comfort_shield_kp,
                        comfort_shield_hot_margin=comfort_shield_hot_margin,
                        comfort_shield_hot_max_action=comfort_shield_hot_max_action,
                        enable_structured_exploration=False,
                        enable_comfort_shield=comfort_shield_exploration and comfort_shield_enabled,
                    )
                    if bool(action_info["comfort_shield_applied"]):
                        actions = [[float(action_info["action"])]]
                        replay_actions = actions
                        agent.actions = actions
            if occupant is not None and str(occupant_mode).strip().lower() == "bspline_tdyn":
                _set_reward_reference_setpoint(env, reward_reference)
            next_observations, reward, terminated, truncated, _ = env.step(actions)
            raw_reward = [float(v) for v in np.asarray(reward, dtype=float).reshape(-1)]
            learning_reward = list(raw_reward)
            replay_terminated = bool(terminated)
            reset_info = {
                "temperature_reset_applied": False,
                "temperature_reset_reason": "",
                "temperature_reset_raw_tin": _observation_value(env, next_observations, "indoor_dry_bulb_temperature"),
                "temperature_reset_tin": _observation_value(env, next_observations, "indoor_dry_bulb_temperature"),
            }
            if training_temperature_reset and phase in {"safe_pi_exploration", "safe_pi_bootstrap", "pi_exploration", "sac_training"}:
                next_observations, reset_info = _apply_training_temperature_reset(
                    env=env,
                    next_observations=next_observations,
                    cold_margin=training_temperature_reset_cold_margin,
                    hot_margin=training_temperature_reset_hot_margin,
                    target_margin=training_temperature_reset_target_margin,
                    absolute_min=training_temperature_reset_absolute_min,
                    absolute_max=training_temperature_reset_absolute_max,
                    reset_hidden=training_temperature_reset_hidden,
                )
            next_observations_for_agent = next_observations
            if occupant is not None and str(occupant_mode).strip().lower() == "bspline_tdyn":
                feedback = float(getattr(env, "last_feedback", 0.0))
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
                if not (terminated or truncated):
                    absolute_next_time_step = int(train_sim_start) + int(episode_step) + 1
                    _, _, next_occ_bin = _tdyn_gate(
                        occupancy_model=occupancy_model,
                        occupancy_series=occupancy_series,
                        absolute_time_step=absolute_next_time_step,
                        alpha_occ=alpha_occ,
                    )
                    next_building = env.unwrapped.buildings[0]
                    next_time_step = int(next_building.time_step)
                    next_scheduled = float(
                        next_building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            next_time_step
                        ]
                    )
                    next_reference = _blend_tdyn_reference(next_scheduled, float(T_dyn), next_occ_bin)
                    next_observations_for_agent = _apply_tdyn_to_observations(env, next_observations, next_reference)
            reset_applied = bool(reset_info.get("temperature_reset_applied", False))
            reset_penalty_applied = False
            if (
                reset_applied
                and training_temperature_reset_replay_mode in {"penalty", "penalty_terminal"}
                and phase == "sac_training"
            ):
                reset_penalty = -abs(float(training_temperature_reset_penalty))
                if training_temperature_reset_replay_mode == "penalty_terminal":
                    learning_reward = [reset_penalty for _ in learning_reward]
                    replay_terminated = True
                else:
                    learning_reward = [float(v) + reset_penalty for v in learning_reward]
                reset_penalty_applied = True
            replay_keep, replay_reason = _transition_safe_for_replay(
                env=env,
                observations=observations,
                next_observations=next_observations,
                reward=learning_reward,
                temperature_min=replay_filter_temperature_min,
                temperature_max=replay_filter_temperature_max,
                temperature_delta_max=replay_filter_temperature_delta_max,
                reward_abs_max=replay_filter_reward_abs_max,
            )
            if reset_applied and training_temperature_reset_replay_mode == "penalty_terminal" and reset_penalty_applied:
                replay_keep = True
                replay_reason = "temperature_reset_penalty_terminal"
            elif reset_applied and training_temperature_reset_replay_mode == "penalty" and reset_penalty_applied:
                replay_keep = True
                replay_reason = "temperature_reset_penalty"
            elif reset_applied and training_temperature_reset_replay_mode in {"penalty", "penalty_terminal"}:
                replay_keep = False
                replay_reason = "temperature_reset_non_sac"
            elif reset_applied and training_temperature_reset_replay_mode == "drop":
                replay_keep = False
                replay_reason = "temperature_reset"
            if replay_keep:
                agent.update(
                    observations_for_agent,
                    replay_actions if replay_actions is not None else actions,
                    learning_reward,
                    next_observations_for_agent,
                    terminated=replay_terminated,
                    truncated=truncated,
                )
            else:
                agent.last_update_losses = []
            for loss_record in getattr(agent, "last_update_losses", []):
                sac_loss_hist.append(
                    {
                        "online_step": int(total_steps),
                        "episode": int(episode_label),
                        "absolute_time_step": int(train_sim_start + episode_step),
                        "phase": phase,
                        **loss_record,
                    }
                )
            if _lagrange_rf is not None and phase == "sac_training":
                _lagrange_rf.step_sac()
            if phase == "sac_training":
                episode_sac_rewards.append(float(np.sum(learning_reward)))
                episode_sac_raw_rewards.append(float(np.sum(raw_reward)))

            online_rewards.append(float(np.sum(learning_reward)))
            raw_reward_hist.append(float(np.sum(raw_reward)))
            phase_hist.append(phase)
            action_hist.append(float(np.asarray(actions, dtype=float).reshape(-1)[0]))
            policy_action_hist.append(float(action_info["raw_action"]))
            policy_mean_action_hist.append(float(policy_stats["policy_mean_action"]))
            policy_raw_mean_hist.append(float(policy_stats["policy_raw_mean"]))
            policy_log_std_hist.append(float(policy_stats["policy_log_std"]))
            policy_std_hist.append(float(policy_stats["policy_std"]))
            action_source_hist.append(str(action_info["action_source"]))
            exploration_rate_hist.append(float(action_info["exploration_rate"]))
            structured_exploration_applied_hist.append(bool(action_info["structured_exploration_applied"]))
            structured_exploration_action_hist.append(float(action_info["structured_exploration_action"]))
            comfort_shield_applied_hist.append(bool(action_info["comfort_shield_applied"]))
            comfort_shield_action_hist.append(float(action_info["comfort_shield_action"]))
            tin_before_action_hist.append(float(action_info["tin"]))
            tsp_before_action_hist.append(float(action_info["tsp"]))
            safe_pi_applied_hist.append(bool(action_info.get("safe_pi_applied", False)))
            safe_pi_reason_hist.append(str(action_info.get("safe_pi_reason", "")))
            temperature_reset_applied_hist.append(bool(reset_info.get("temperature_reset_applied", False)))
            temperature_reset_reason_hist.append(str(reset_info.get("temperature_reset_reason", "")))
            temperature_reset_raw_tin_hist.append(float(reset_info.get("temperature_reset_raw_tin", np.nan)))
            temperature_reset_tin_hist.append(float(reset_info.get("temperature_reset_tin", np.nan)))
            replay_kept_hist.append(bool(replay_keep))
            replay_filter_reason_hist.append(str(replay_reason))
            replay_terminal_hist.append(bool(replay_terminated))
            reset_penalty_applied_hist.append(bool(reset_penalty_applied))
            if occupant is not None and str(occupant_mode).strip().lower() == "bspline_tdyn":
                tdyn_hist.append(float(T_dyn))
                reference_hist.append(float(reward_reference))
                occ_bin_hist.append(float(occ_bin))
            episode_hist.append(int(episode_label))
            absolute_time_step_hist.append(int(train_sim_start + episode_step))
            observations = next_observations
            previous_action = float(np.asarray(actions, dtype=float).reshape(-1)[0])
            episode_step += 1
            total_steps += 1
            if max_steps is not None and total_steps >= int(max_steps):
                step_limit_reached = True
                break

        if step_limit_reached:
            break

        if episode_sac_rewards:
            early_stop_history.append(
                {
                    "episode": float(episode_label),
                    "reward_mean": float(np.mean(episode_sac_rewards)),
                    "raw_reward_mean": float(np.mean(episode_sac_raw_rewards)),
                    "sac_steps": float(len(episode_sac_rewards)),
                }
            )
            if bool(early_stop_reward_slope) and int(episode_index) + 1 >= int(early_stop_min_episodes):
                window = max(int(early_stop_window), 2)
                if len(early_stop_history) >= window:
                    recent = early_stop_history[-window:]
                    x = np.asarray([row["episode"] for row in recent], dtype=float)
                    y = np.asarray([row["raw_reward_mean"] for row in recent], dtype=float)
                    if np.all(np.isfinite(y)):
                        early_stop_last_slope = float(np.polyfit(x, y, 1)[0])
                        if abs(early_stop_last_slope) <= abs(float(early_stop_slope_abs)):
                            early_stop_stale_windows += 1
                        else:
                            early_stop_stale_windows = 0
                    if early_stop_stale_windows >= max(int(early_stop_patience), 1):
                        early_stop_triggered = True
                        early_stop_episode = int(episode_label)
                        break

        if (
            not bc_completed
            and resume_payload is None
            and int(bc_epochs) > 0
            and bc_expert_observations is not None
            and bc_expert_actions is not None
            and int(episode_index) == 0
        ):
            print(f"Running behavior cloning for {int(bc_epochs)} epochs...")
            bc_loss_hist = behavior_clone_sac_actor(
                agent,
                expert_observations=np.asarray(bc_expert_observations, dtype=float),
                expert_actions=np.asarray(bc_expert_actions, dtype=float),
                epochs=int(bc_epochs),
                batch_size=int(bc_batch_size),
                lr=float(bc_lr),
                validation_fraction=float(bc_validation_fraction),
                early_stopping_patience=int(bc_early_stopping_patience),
                min_delta=float(bc_min_delta),
                seed=seed,
            )
            bc_completed = True

        if bool(checkpoint_select_by_eval) and _sac_policy_ready_for_prediction(agent):
            validation_state = _clone_agent_eval_state(agent)
            validation_result = _evaluate_trained_agent_on_february(
                agent,
                seed=seed,
                reward_function=reward_function,
                reward_function_kwargs=(
                    reward_function_kwargs if eval_reward_function_kwargs is None else eval_reward_function_kwargs
                ),
                max_steps=max_steps,
                occupant=deepcopy(occupant) if occupant is not None else None,
                mode=str(occupant_mode).strip().lower(),
                occupancy_model=occupancy_model,
                occupancy_series=occupancy_series,
                alpha_occ=alpha_occ,
                T_dyn_init_mode=T_dyn_init_mode,
                delta_up=delta_up,
                delta_down=delta_down,
                drift_to_pref=drift_to_pref,
                T_dyn_min=T_dyn_min,
                T_dyn_max=T_dyn_max,
                building_name=building_name,
                comfort_shield_margin=comfort_shield_margin,
                comfort_shield_min_action=comfort_shield_min_action,
                comfort_shield_kp=comfort_shield_kp,
                comfort_shield_hot_margin=comfort_shield_hot_margin,
                comfort_shield_hot_max_action=comfort_shield_hot_max_action,
                apply_comfort_shield=bool(comfort_shield_eval),
                residual_safe_pi=bool(residual_safe_pi),
                residual_safe_pi_scale=float(residual_safe_pi_scale),
                safe_pi_cold_margin=safe_pi_cold_margin,
                safe_pi_hot_margin=safe_pi_hot_margin,
                safe_pi_cold_min_action=safe_pi_cold_min_action,
                safe_pi_cold_kp=safe_pi_cold_kp,
                safe_pi_hot_max_action=safe_pi_hot_max_action,
                safe_pi_ramp_limit=safe_pi_ramp_limit,
                rc_rollout_horizons=rc_rollout_horizons,
                rc_rollout_model_dir=rc_rollout_model_dir,
                observation_filter=config.observation_filter,
                sim_start=eval_sim_start,
                sim_end=eval_sim_end,
                mutate_agent_clock=False,
            )
            validation_summary = summarize_run(
                f"{label}_checkpoint_ep{int(episode_label)}",
                validation_result,
            )
            metric_value = float(validation_summary.get(str(checkpoint_select_metric), np.nan))
            checkpoint_eval_rows.append(
                {
                    "episode": int(episode_label),
                    "metric": str(checkpoint_select_metric),
                    "metric_value": metric_value,
                    "reward_total": float(validation_summary["reward_total"]),
                    "electricity_consumption_total": float(validation_summary["electricity_consumption_total"]),
                    "discomfort_proportion": float(validation_summary["discomfort_proportion"]),
                    "discomfort_cold_proportion": float(validation_summary["discomfort_cold_proportion"]),
                    "discomfort_hot_proportion": float(validation_summary["discomfort_hot_proportion"]),
                    "daily_peak_average": float(validation_summary["daily_peak_average"]),
                    "kelvin_hours": float(validation_summary["kelvin_hours"]),
                }
            )
            if np.isfinite(metric_value) and metric_value < best_checkpoint_metric:
                best_checkpoint_metric = metric_value
                best_checkpoint_episode = int(episode_label)
                best_checkpoint_state = validation_state
            _restore_agent_eval_state(agent, validation_state)

    if bool(checkpoint_select_by_eval) and best_checkpoint_state is not None:
        _restore_agent_eval_state(agent, best_checkpoint_state)

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
        reward_function_kwargs=(
            reward_function_kwargs if eval_reward_function_kwargs is None else eval_reward_function_kwargs
        ),
        max_steps=max_steps,
        occupant=deepcopy(occupant) if occupant is not None else None,
        mode=str(occupant_mode).strip().lower(),
        occupancy_model=occupancy_model,
        occupancy_series=occupancy_series,
        alpha_occ=alpha_occ,
        T_dyn_init_mode=T_dyn_init_mode,
        delta_up=delta_up,
        delta_down=delta_down,
        drift_to_pref=drift_to_pref,
        T_dyn_min=T_dyn_min,
        T_dyn_max=T_dyn_max,
        building_name=building_name,
        comfort_shield_margin=comfort_shield_margin,
        comfort_shield_min_action=comfort_shield_min_action,
        comfort_shield_kp=comfort_shield_kp,
        comfort_shield_hot_margin=comfort_shield_hot_margin,
        comfort_shield_hot_max_action=comfort_shield_hot_max_action,
        apply_comfort_shield=bool(comfort_shield_eval),
        residual_safe_pi=bool(residual_safe_pi),
        residual_safe_pi_scale=float(residual_safe_pi_scale),
        safe_pi_cold_margin=safe_pi_cold_margin,
        safe_pi_hot_margin=safe_pi_hot_margin,
        safe_pi_cold_min_action=safe_pi_cold_min_action,
        safe_pi_cold_kp=safe_pi_cold_kp,
        safe_pi_hot_max_action=safe_pi_hot_max_action,
        safe_pi_ramp_limit=safe_pi_ramp_limit,
        rc_rollout_horizons=rc_rollout_horizons,
        rc_rollout_model_dir=rc_rollout_model_dir,
        observation_filter=config.observation_filter,
        sim_start=eval_sim_start,
        sim_end=eval_sim_end,
    )

    online_info = {
        "jumpstart_controller": (
            "MPC teacher"
            if jumpstart_controller == "mpc"
            else (
                "Safe PITemperatureController"
                if jumpstart_controller == "safe_pi"
                else "PITemperatureController"
            )
        ),
        "exploration_steps": int(exploration_steps),
        "training_episodes": int(training_episodes),
        "training_episodes_completed": int(early_stop_episode if early_stop_triggered else training_episodes),
        "early_stop_reward_slope": bool(early_stop_reward_slope),
        "early_stop_triggered": bool(early_stop_triggered),
        "early_stop_episode": None if early_stop_episode is None else int(early_stop_episode),
        "early_stop_min_episodes": int(early_stop_min_episodes),
        "early_stop_window": int(early_stop_window),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_slope_abs": float(early_stop_slope_abs),
        "early_stop_last_slope": float(early_stop_last_slope) if np.isfinite(early_stop_last_slope) else None,
        "resume_agent_path": None if resume_agent_path is None else str(resume_agent_path),
        "resume_episode_offset": int(resume_episode_offset),
        "resume_loaded": bool(resume_payload is not None),
        "resume_optimizer_state_restored": False,
        "resume_source_time_step": None if resume_payload is None else int(resume_payload.get("time_step", -1)),
        "resume_source_replay_size": (
            None
            if resume_payload is None
            else int(len(resume_payload.get("replay_buffer", [[]])[0]))
        ),
        "resume_replay_keep_last": None if resume_replay_keep_last is None else int(resume_replay_keep_last),
        "resume_replay_after_trim_size": (
            None
            if not resume_replay_trim_stats
            else int(resume_replay_trim_stats[0]["after"])
        ),
        "resume_replay_trim_stats": resume_replay_trim_stats,
        "online_train_steps": int(len(online_rewards)),
        "online_train_reward_total": float(np.sum(online_rewards)),
        "online_replay_size": int(len(agent.replay_buffer[0])),
        "online_normalized": bool(agent.normalized[0]),
        "online_pi_steps": int(sum(1 for p in phase_hist if p == "pi_exploration")),
        "online_safe_pi_steps": int(sum(1 for p in phase_hist if p in {"safe_pi_exploration", "safe_pi_bootstrap"})),
        "online_safe_pi_bootstrap_steps": int(sum(1 for p in phase_hist if p == "safe_pi_bootstrap")),
        "online_mpc_steps": int(sum(1 for p in phase_hist if p == "mpc_exploration")),
        "online_sac_steps": int(sum(1 for p in phase_hist if p == "sac_training")),
        "replay_filter_kept_steps": int(sum(bool(v) for v in replay_kept_hist)),
        "replay_filter_dropped_steps": int(sum(not bool(v) for v in replay_kept_hist)),
        "replay_filter_dropped_fraction": float(
            np.mean([not bool(v) for v in replay_kept_hist]) if replay_kept_hist else 0.0
        ),
        "replay_filter_temperature_min": None if replay_filter_temperature_min is None else float(replay_filter_temperature_min),
        "replay_filter_temperature_max": None if replay_filter_temperature_max is None else float(replay_filter_temperature_max),
        "replay_filter_temperature_delta_max": None if replay_filter_temperature_delta_max is None else float(replay_filter_temperature_delta_max),
        "replay_filter_reward_abs_max": None if replay_filter_reward_abs_max is None else float(replay_filter_reward_abs_max),
        "structured_exploration_start": float(structured_exploration_start),
        "structured_exploration_end": float(structured_exploration_end),
        "structured_exploration_cold_min_action": float(structured_exploration_cold_min_action),
        "structured_exploration_steps": int(sum(bool(v) for v in structured_exploration_applied_hist)),
        "structured_exploration_fraction": float(
            np.mean(structured_exploration_applied_hist)
            if structured_exploration_applied_hist
            else 0.0
        ),
        "comfort_shield_margin": None if comfort_shield_margin is None else float(comfort_shield_margin),
        "comfort_shield_min_action": float(comfort_shield_min_action),
        "comfort_shield_kp": float(comfort_shield_kp),
        "comfort_shield_hot_margin": (
            None if comfort_shield_hot_margin is None else float(comfort_shield_hot_margin)
        ),
        "comfort_shield_hot_max_action": float(comfort_shield_hot_max_action),
        "comfort_shield_training_steps": int(sum(bool(v) for v in comfort_shield_applied_hist)),
        "comfort_shield_training_fraction": float(
            np.mean(comfort_shield_applied_hist)
            if comfort_shield_applied_hist
            else 0.0
        ),
        "comfort_shield_eval": bool(comfort_shield_eval),
        "comfort_shield_eval_steps": int(
            np.sum(result.get("comfort_shield_applied_hist", np.asarray([], dtype=bool)))
        ),
        "safe_pi_cold_margin": float(safe_pi_cold_margin),
        "safe_pi_hot_margin": float(safe_pi_hot_margin),
        "safe_pi_cold_min_action": float(safe_pi_cold_min_action),
        "safe_pi_cold_kp": float(safe_pi_cold_kp),
        "safe_pi_hot_max_action": float(safe_pi_hot_max_action),
        "safe_pi_random_prob": float(safe_pi_random_prob),
        "safe_pi_random_low": float(safe_pi_random_low),
        "safe_pi_random_high": float(safe_pi_random_high),
        "safe_pi_ramp_limit": None if safe_pi_ramp_limit is None else float(safe_pi_ramp_limit),
        "safe_pi_applied_steps": int(sum(bool(v) for v in safe_pi_applied_hist)),
        "safe_pi_applied_fraction": float(np.mean(safe_pi_applied_hist) if safe_pi_applied_hist else 0.0),
        "residual_safe_pi": bool(residual_safe_pi),
        "residual_safe_pi_scale": float(residual_safe_pi_scale),
        "checkpoint_select_by_eval": bool(checkpoint_select_by_eval),
        "checkpoint_select_metric": str(checkpoint_select_metric),
        "checkpoint_selected_episode": None if best_checkpoint_episode is None else int(best_checkpoint_episode),
        "checkpoint_selected_metric_value": (
            None if not np.isfinite(best_checkpoint_metric) else float(best_checkpoint_metric)
        ),
        "checkpoint_eval_count": int(len(checkpoint_eval_rows)),
        "training_temperature_reset": bool(training_temperature_reset),
        "training_temperature_reset_cold_margin": float(training_temperature_reset_cold_margin),
        "training_temperature_reset_hot_margin": float(training_temperature_reset_hot_margin),
        "training_temperature_reset_target_margin": float(training_temperature_reset_target_margin),
        "training_temperature_reset_absolute_min": (
            None if training_temperature_reset_absolute_min is None else float(training_temperature_reset_absolute_min)
        ),
        "training_temperature_reset_absolute_max": (
            None if training_temperature_reset_absolute_max is None else float(training_temperature_reset_absolute_max)
        ),
        "training_temperature_reset_hidden": bool(training_temperature_reset_hidden),
        "training_temperature_reset_drop_transition": bool(training_temperature_reset_drop_transition),
        "training_temperature_reset_replay_mode": str(training_temperature_reset_replay_mode),
        "training_temperature_reset_penalty": float(-abs(float(training_temperature_reset_penalty))),
        "training_temperature_reset_steps": int(sum(bool(v) for v in temperature_reset_applied_hist)),
        "training_temperature_reset_fraction": float(
            np.mean(temperature_reset_applied_hist) if temperature_reset_applied_hist else 0.0
        ),
        "training_temperature_reset_penalty_steps": int(sum(bool(v) for v in reset_penalty_applied_hist)),
        "training_temperature_reset_penalty_fraction": float(
            np.mean(reset_penalty_applied_hist) if reset_penalty_applied_hist else 0.0
        ),
        "automatic_entropy_tuning": bool(config.automatic_entropy_tuning),
        "entropy_lr": None if config.entropy_lr is None else float(config.entropy_lr),
        "target_entropy": None if config.target_entropy is None else float(config.target_entropy),
        "alpha_final": float(getattr(agent, "alpha", config.alpha)),
        "robust_normalization": bool(config.robust_normalization),
        "robust_normalization_low": float(config.robust_normalization_low),
        "robust_normalization_high": float(config.robust_normalization_high),
        "normalized_observation_clip": None if config.normalized_observation_clip is None else float(config.normalized_observation_clip),
        "normalized_reward_clip": None if config.normalized_reward_clip is None else float(config.normalized_reward_clip),
        "bc_epochs": int(bc_epochs),
        "bc_batch_size": int(bc_batch_size),
        "bc_lr": float(bc_lr),
        "bc_validation_fraction": float(bc_validation_fraction),
        "bc_early_stopping_patience": int(bc_early_stopping_patience),
        "bc_min_delta": float(bc_min_delta),
        "bc_expert_samples": int(0 if bc_expert_actions is None else np.asarray(bc_expert_actions).reshape(-1).size),
        "bc_final_loss_mean": None if not bc_loss_hist else float(bc_loss_hist[-1]["bc_loss_mean"]),
        "bc_final_loss_max": None if not bc_loss_hist else float(bc_loss_hist[-1]["bc_loss_max"]),
        "bc_final_val_loss": None if not bc_loss_hist else float(bc_loss_hist[-1]["bc_val_loss"]),
        "bc_best_val_loss": None if not bc_loss_hist else float(min(row["bc_val_loss"] for row in bc_loss_hist)),
        "bc_epochs_run": int(len(bc_loss_hist)),
        "agent_pickle": str(agent_path),
        "agent_model_zip": None if model_zip is None else str(model_zip),
        "uses_separate_eval_reward_kwargs": bool(eval_reward_function_kwargs is not None),
        "save_scenario": str(save_scenario),
        "occupant_mode": str(occupant_mode),
        "train_sim_start": int(train_sim_start),
        "train_sim_end": int(train_sim_end),
        "eval_sim_start": int(eval_sim_start),
        "eval_sim_end": int(eval_sim_end),
    }
    if occupant is not None and hasattr(occupant, "T_pref"):
        online_info["occupant_target_temperature"] = float(occupant.T_pref)
    if _lagrange_rf is not None:
        online_info["lagrange_final_lambda"] = float(_lagrange_rf.lagrange_multiplier)
        online_info["lagrange_trajectory"] = _lagrange_rf.lambda_trajectory
    if building_metadata:
        online_info.update(building_metadata)
    metrics = summarize_run(label, result, online_info=online_info)
    save_result(
        result,
        output_dir=Path(output_dir),
        summary_dir=Path(summary_dir),
        scenario=str(save_scenario),
        label=label,
        metrics=metrics,
    )

    online_history = {
        "time_step": np.arange(len(online_rewards), dtype=int),
        "episode": episode_hist,
        "absolute_time_step": absolute_time_step_hist,
        "phase": phase_hist,
        "action": action_hist,
        "policy_action": policy_action_hist,
        "policy_mean_action": policy_mean_action_hist,
        "policy_raw_mean": policy_raw_mean_hist,
        "policy_log_std": policy_log_std_hist,
        "policy_std": policy_std_hist,
        "action_source": action_source_hist,
        "exploration_rate": exploration_rate_hist,
        "structured_exploration_applied": structured_exploration_applied_hist,
        "structured_exploration_action": structured_exploration_action_hist,
        "comfort_shield_applied": comfort_shield_applied_hist,
        "comfort_shield_action": comfort_shield_action_hist,
        "tin_before_action": tin_before_action_hist,
        "tsp_before_action": tsp_before_action_hist,
        "safe_pi_applied": safe_pi_applied_hist,
        "safe_pi_reason": safe_pi_reason_hist,
        "temperature_reset_applied": temperature_reset_applied_hist,
        "temperature_reset_reason": temperature_reset_reason_hist,
        "temperature_reset_raw_tin": temperature_reset_raw_tin_hist,
        "temperature_reset_tin": temperature_reset_tin_hist,
        "replay_transition_kept": replay_kept_hist,
        "replay_filter_reason": replay_filter_reason_hist,
        "replay_terminal": replay_terminal_hist,
        "reset_penalty_applied": reset_penalty_applied_hist,
        "raw_reward": raw_reward_hist,
        "reward": online_rewards,
    }
    if tdyn_hist:
        online_history["tdyn"] = tdyn_hist
    if reference_hist:
        online_history["reference_setpoint"] = reference_hist
    if occ_bin_hist:
        online_history["occ_bin_1step"] = occ_bin_hist
    import pandas as pd

    output_case_dir = Path(output_dir) / str(save_scenario) / label
    pd.DataFrame(online_history).to_csv(output_case_dir / "online_training_rollout.csv", index=False)
    if bc_expert_observations is not None and bc_expert_actions is not None:
        np.savez_compressed(
            output_case_dir / "bc_expert_dataset.npz",
            observations=np.asarray(bc_expert_observations, dtype=float),
            actions=np.asarray(bc_expert_actions, dtype=float),
        )
    if bc_loss_hist:
        pd.DataFrame(bc_loss_hist).to_csv(output_case_dir / "bc_training_losses.csv", index=False)
    if sac_loss_hist:
        pd.DataFrame(sac_loss_hist).to_csv(output_case_dir / "sac_training_losses.csv", index=False)
    if checkpoint_eval_rows:
        pd.DataFrame(checkpoint_eval_rows).to_csv(output_case_dir / "checkpoint_eval_summary.csv", index=False)

    summary_path = Path(summary_dir) / str(save_scenario) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([metrics])
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing.get("case", pd.Series(dtype=str)) != metrics.get("case")]
        new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_csv(summary_path, index=False)

    manifest_jumpstart = (
        "MPC teacher"
        if jumpstart_controller == "mpc"
        else (
            "safe bounded Temperature PI via CityLearn PITemperatureController"
            if jumpstart_controller == "safe_pi"
            else "Temperature PI via CityLearn PITemperatureController"
        )
    )
    manifest = {
        "method": f"CityLearn SACRBC MERLIN-style {manifest_jumpstart} jump-start",
        "source_equivalence": {
            "exploration_policy": manifest_jumpstart,
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
    print(f"Safe PI exploration steps: {online_info['online_safe_pi_steps']}")
    print(f"Training temperature reset steps: {online_info['training_temperature_reset_steps']}")
    print(f"Replay dropped steps: {online_info['replay_filter_dropped_steps']}")
    print(f"BC epochs: {online_info['bc_epochs_run']}/{online_info['bc_epochs']}")
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
    drift_to_pref: float = 0.05,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
    building_name: str = CONTROL_BUILDING_NAME,
):
    import pandas as pd

    set_experiment_seed(training_seed)
    env = make_env(
        sim_start=JAN_START,
        sim_end=JAN_END,
        seed=seed,
        reward_function=reward_function,
        reward_function_kwargs=reward_function_kwargs,
        building_name=building_name,
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
                occ_now_gate, _, occ_bin = _tdyn_gate(
                    occupancy_model=occupancy_model,
                    occupancy_series=occupancy_series,
                    absolute_time_step=absolute_time_step,
                    alpha_occ=alpha_occ,
                )
                observations_for_agent = observations
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
                    next_observations_for_agent = next_observations
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
    building_name: str,
    occupancy_building_name: str,
    building_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    import pandas as pd

    reward_function = reward_function_from_mode(reward_mode)
    reward_function_kwargs = None
    if reward_mode in ("dynamic_comfort", "occupant_comfort"):
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
    occupancy_model, occupancy_series = fit_bspline_occupancy_model(seed=seed, building_name=occupancy_building_name)

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
        building_name=building_name,
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
        building_name=building_name,
    )

    online_info = {
        **online_info,
        "reward_mode": reward_mode,
        "occupant_mode": mode,
        "T_dyn_init_mode": T_dyn_init_mode,
        "agent_pickle": str(agent_path),
        "agent_model_zip": None if model_zip is None else str(model_zip),
    }
    if building_metadata:
        online_info.update(building_metadata)
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
    drift_to_pref: float = 0.05,
    T_dyn_min: float = 18.0,
    T_dyn_max: float = 26.0,
    building_name: str = CONTROL_BUILDING_NAME,
    occupancy_building_name: str | None = None,
    building_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import pandas as pd

    output_dir = Path(output_dir)
    summary_dir = Path(summary_dir)
    model_dir = Path(model_dir)

    occupants = fitted_occupants()
    if occupant != "all":
        occupants = {occupant: occupants[occupant]}

    suffix = {
        "comfort": "comfort_reward",
        "dynamic_comfort": "dynamic_reward",
        "feedback": "feedback_reward",
        "fixed_comfort": "fixed_comfort_reward",
        "occupant_comfort": "occupant_reward",
    }

    def observation_rewards() -> list[str]:
        if reward_mode == "all":
            return ["comfort", "dynamic_comfort", "feedback", "fixed_comfort"]
        if reward_mode == "both":
            return ["comfort", "dynamic_comfort", "feedback"]
        if reward_mode == "occupant_comfort":
            return ["dynamic_comfort"]
        return [reward_mode]

    def tdyn_rewards() -> list[str]:
        if reward_mode in ("all", "both"):
            return ["comfort", "occupant_comfort"]
        if reward_mode == "dynamic_comfort":
            return ["occupant_comfort"]
        if reward_mode in ("feedback", "fixed_comfort"):
            return ["comfort"]
        return [reward_mode]

    if occupant_mode == "all":
        mode_reward_plan = [
            ("bspline_observation", observation_rewards()),
            ("without_tdyn", tdyn_rewards()),
            ("bspline_tdyn", tdyn_rewards()),
        ]
    elif occupant_mode == "both":
        mode_reward_plan = [
            ("without_tdyn", tdyn_rewards()),
            ("bspline_tdyn", tdyn_rewards()),
        ]
    elif occupant_mode == "bspline_observation":
        mode_reward_plan = [("bspline_observation", observation_rewards())]
    else:
        mode_reward_plan = [(occupant_mode, tdyn_rewards())]

    all_results: dict[str, Any] = {}
    summary_rows = []
    occupancy_building_name = building_name if occupancy_building_name is None else occupancy_building_name
    for occ_idx, (name, occ) in enumerate(occupants.items(), start=1):
        for mode, reward_names in mode_reward_plan:
            add_reward_suffix = len(reward_names) > 1
            for reward_name in reward_names:
                if mode == "bspline_observation":
                    label = f"{name}_{suffix[reward_name]}" if add_reward_suffix else name
                else:
                    label = f"{name}_{mode}_{suffix[reward_name]}"
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
                    building_name=building_name,
                    occupancy_building_name=occupancy_building_name,
                    building_metadata=building_metadata,
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
    building_name: str = CONTROL_BUILDING_NAME,
) -> dict[str, Any]:
    """Run the January PI jump-start through February without SAC learning.

    This is a diagnostic rollout: the SACRBC wrapper is still used, but the
    exploration/standardization boundary is placed after the simulated period,
    so the Temperature-PI jump-start controller remains in charge and the SAC
    policy is never trained or evaluated. The saved rollout is February only,
    matching the layout used by the result notebook.
    """

    set_experiment_seed(seed)
    env = make_env(
        sim_start=JAN_START,
        sim_end=FEB_END,
        seed=seed,
        reward_function=reward_function,
        building_name=building_name,
    )

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
        "comfort_band_lower": 20.0,
        "comfort_band_upper": 24.0,
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
