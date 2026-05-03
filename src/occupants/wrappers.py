"""Environment wrappers for occupant interaction."""

from __future__ import annotations

from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OccupantEnvWrapper(gym.Wrapper):
    """Apply thermostat overrides and expose recent override feedback as state."""

    def __init__(
        self,
        env,
        occupant,
        feedback_horizon: int = 4,
        action_on_threshold: float = 0.05,
        action_on_value: float = 1.0,
        feedback_comfort_range: float | None = None,
    ):
        super().__init__(env)
        self.occupant = occupant
        self.feedback_horizon = max(1, int(feedback_horizon))
        # Kept for backward-compatible construction; thermostat overrides do
        # not classify or modify continuous HVAC actions.
        self.action_on_threshold = float(action_on_threshold)
        self.action_on_value = float(action_on_value)
        self.feedback_comfort_range = feedback_comfort_range
        self.feedback_buffer = deque([0.0] * self.feedback_horizon, maxlen=self.feedback_horizon)
        if hasattr(env, "observation_names"):
            self.observation_names = [
                list(names) + [f"occupant_feedback_buffer_{i}" for i in range(self.feedback_horizon)]
                for names in env.observation_names
            ]
        if hasattr(env, "action_names"):
            self.action_names = env.action_names
        self.observation_space = [
            spaces.Box(
                low=np.concatenate([
                    np.asarray(space.low, dtype=np.float32),
                    np.full(self.feedback_horizon, -1.0, dtype=np.float32),
                ]),
                high=np.concatenate([
                    np.asarray(space.high, dtype=np.float32),
                    np.ones(self.feedback_horizon, dtype=np.float32),
                ]),
                dtype=np.float32,
            )
            for space in env.observation_space
        ]
        self.action_space = env.action_space

    @staticmethod
    def _ensure_feedback_signal(building):
        full_setpoints = building.energy_simulation.__dict__[
            "_indoor_dry_bulb_temperature_heating_set_point"
        ]
        n_steps = len(full_setpoints)
        current = building.energy_simulation.__dict__.get("_occupant_feedback_signal")
        if current is None or len(current) != n_steps:
            building.energy_simulation.occupant_feedback_signal = np.zeros(n_steps, dtype=np.float32)
        return building.energy_simulation.__dict__["_occupant_feedback_signal"]

    @staticmethod
    def _absolute_index(building, episode_index: int) -> int:
        start = 0 if building.energy_simulation.start_time_step is None else building.energy_simulation.start_time_step
        return int(start) + int(episode_index)

    @staticmethod
    def _occupant_preference(occupant, default: float) -> float:
        if hasattr(occupant, "T_pref"):
            return float(occupant.T_pref)
        if hasattr(occupant, "preferred_temperature"):
            return float(occupant.preferred_temperature)
        return float(default)

    @staticmethod
    def _occupant_probability_cap(occupant) -> float:
        if hasattr(occupant, "max_prob"):
            return float(occupant.max_prob)
        if hasattr(occupant, "max_probability"):
            return float(occupant.max_probability)
        return 0.4

    def _feedback_features(self):
        return np.asarray(list(self.feedback_buffer), dtype=np.float32)

    def _augment(self, observations):
        feedback_features = self._feedback_features()
        return [
            np.concatenate([np.asarray(observation, dtype=np.float32), feedback_features])
            for observation in observations
        ]

    def _building_action_names(self):
        return list(self.action_names[0]) if hasattr(self, "action_names") else []

    @staticmethod
    def _copy_actions(action):
        if isinstance(action, np.ndarray):
            return [action.astype(np.float32, copy=True)]
        if len(action) > 0 and np.asarray(action[0]).ndim == 0:
            return [np.asarray(action, dtype=np.float32).copy()]
        return [np.asarray(a, dtype=np.float32).copy() for a in action]

    def _action_value(self, actions, action_name: str) -> float | None:
        action_names = self._building_action_names()
        if action_name not in action_names:
            return None
        return float(np.asarray(actions[0], dtype=float).reshape(-1)[action_names.index(action_name)])

    def _set_action_value(self, actions, action_name: str, value: float) -> None:
        action_names = self._building_action_names()
        if action_name in action_names:
            actions[0][action_names.index(action_name)] = float(value)

    def _available_hvac_action(self) -> str | None:
        action_names = self._building_action_names()
        for action_name in ("heating_device", "cooling_device", "cooling_or_heating_device"):
            if action_name in action_names:
                return action_name
        return None

    def _expected_hvac_state(self, building, time_step: int, indoor_temperature: float, set_point: float):
        action_name = self._available_hvac_action()
        if action_name is None:
            return False, 0

        outdoor_temperature = float(building.weather.outdoor_dry_bulb_temperature[time_step])

        if action_name == "heating_device":
            return indoor_temperature < set_point and outdoor_temperature < set_point, 1
        if action_name == "cooling_device":
            return indoor_temperature > set_point and outdoor_temperature > set_point, -1

        if indoor_temperature < set_point and outdoor_temperature < set_point:
            return True, 1
        if indoor_temperature > set_point and outdoor_temperature > set_point:
            return True, -1
        return False, 0

    def _is_hvac_on(self, actions, expected_direction: int = 0) -> bool:
        action_name = self._available_hvac_action()
        if action_name is None:
            return False
        action_value = self._action_value(actions, action_name)
        if action_value is None:
            return False
        if action_name == "cooling_or_heating_device":
            if expected_direction < 0:
                return action_value <= -self.action_on_threshold
            if expected_direction > 0:
                return action_value >= self.action_on_threshold
            return abs(action_value) >= self.action_on_threshold
        return action_value >= self.action_on_threshold

    def _feedback_probability(self, building, time_step: int, indoor_temperature: float, set_point: float) -> float:
        if self.feedback_comfort_range is None:
            comfort_range = float(building.energy_simulation.comfort_band[time_step])
        else:
            comfort_range = float(self.feedback_comfort_range)
        comfort_range = max(abs(comfort_range), 1.0)
        probability = ((float(indoor_temperature) - float(set_point)) / comfort_range) ** 2
        return float(min(probability, self._occupant_probability_cap(self.occupant)))

    def _simulate_feedback(self, building, actions, time_step: int, occupied: bool, set_point: float) -> tuple[float, int]:
        if not occupied:
            return 0.0, 0

        indoor_temperature = float(building.energy_simulation.indoor_dry_bulb_temperature[time_step])
        expected_on, expected_direction = self._expected_hvac_state(
            building,
            time_step,
            indoor_temperature,
            set_point,
        )
        if expected_on == self._is_hvac_on(actions, expected_direction):
            return 0.0, expected_direction

        if np.random.rand() >= self._feedback_probability(building, time_step, indoor_temperature, set_point):
            return 0.0, expected_direction

        return (1.0 if expected_on else -1.0), expected_direction

    def _apply_feedback_to_action(self, actions, feedback: float, expected_direction: int) -> None:
        action_name = self._available_hvac_action()
        if action_name is None or feedback == 0.0:
            return

        if feedback < 0.0:
            self._set_action_value(actions, action_name, 0.0)
            return

        if action_name == "cooling_or_heating_device":
            direction = -1.0 if expected_direction < 0 else 1.0
            self._set_action_value(actions, action_name, direction * self.action_on_value)
        else:
            self._set_action_value(actions, action_name, self.action_on_value)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.occupant.clear_override()
        self.override_count = 0
        self.last_feedback = 0.0
        self.last_agent_action = 0.0
        self.last_controlled_action = 0.0
        self.feedback_buffer = deque([0.0] * self.feedback_horizon, maxlen=self.feedback_horizon)

        building = self.env.unwrapped.buildings[0]
        self._ensure_feedback_signal(building).fill(0.0)
        self.effective_setpoints = [
            float(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[building.time_step])
        ]
        self.baseline_setpoints = [
            float(
                building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                    building.time_step
                ]
            )
        ]
        return self._augment(obs), info

    def step(self, action):
        building = self.env.unwrapped.buildings[0]
        current_ix = building.time_step
        indoor_temperature = float(building.energy_simulation.indoor_dry_bulb_temperature[current_ix])
        next_ix = min(
            current_ix + 1,
            len(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point) - 1,
        )

        occupied_next = float(building.energy_simulation.occupant_count[next_ix]) > 0.0
        scheduled_next_setpoint = float(
            building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[next_ix]
        )
        new_setpoint = float(
            self.occupant.step(
                indoor_temperature,
                scheduled_next_setpoint,
                occupied=occupied_next,
            )
        )

        feedback = 0.0
        if self.occupant.last_override_started:
            self.override_count += 1
            if new_setpoint > scheduled_next_setpoint:
                feedback = 1.0
            elif new_setpoint < scheduled_next_setpoint:
                feedback = -1.0

        controlled_action = self._copy_actions(action)
        self.last_agent_action = float(np.asarray(controlled_action, dtype=float).reshape(-1)[0])
        self.last_controlled_action = self.last_agent_action
        self.last_feedback = feedback
        self.feedback_buffer.appendleft(feedback)
        feedback_signal = self._ensure_feedback_signal(building)
        feedback_signal[self._absolute_index(building, next_ix)] = feedback

        building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[next_ix] = new_setpoint
        obs, reward, terminated, truncated, info = self.env.step(controlled_action)
        self.effective_setpoints.append(
            float(building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[building.time_step])
        )
        self.baseline_setpoints.append(
            float(
                building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                    building.time_step
                ]
            )
        )
        return self._augment(obs), reward, terminated, truncated, info


class OccupantReplayWrapper(OccupantEnvWrapper):
    """Named alias for replay-style occupant rollouts used during fitting."""


class BSplineOccupancyObservationWrapper(gym.Wrapper):
    """Append one-step B-spline occupancy forecast features to observations."""

    def __init__(self, env, occupancy_model, occupancy_series, forecast_fn, alpha: float = 0.5):
        super().__init__(env)
        self.occupancy_model = occupancy_model
        self.occupancy_series = occupancy_series
        self.forecast_fn = forecast_fn
        self.alpha = float(alpha)
        self.observation_names = [
            list(names) + ["bspline_occupancy_probability_1", "bspline_occupied_1"]
            for names in env.observation_names
        ]
        if hasattr(env.unwrapped, "action_names"):
            self.action_names = env.unwrapped.action_names
        self.observation_space = [
            spaces.Box(
                low=np.concatenate([np.asarray(space.low, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)]),
                high=np.concatenate([np.asarray(space.high, dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]),
                dtype=np.float32,
            )
            for space in env.observation_space
        ]
        self.action_space = env.action_space

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def _absolute_time_step(self) -> int:
        building = self.env.unwrapped.buildings[0]
        start = building.energy_simulation.start_time_step
        start = 0 if start is None else int(start)
        return start + int(building.time_step)

    def _forecast(self) -> tuple[float, float]:
        absolute_time_step = min(self._absolute_time_step(), len(self.occupancy_series) - 1)
        current_occupancy = int(self.occupancy_series[absolute_time_step])
        occ_hat, occ_bin = self.forecast_fn(
            self.occupancy_model,
            current_occupancy=current_occupancy,
            time_step=absolute_time_step,
            horizon=1,
            alpha=self.alpha,
        )
        hat = float(occ_hat[0]) if len(occ_hat) else float(current_occupancy)
        binary = float(occ_bin[0]) if len(occ_bin) else float(current_occupancy > self.alpha)
        return hat, binary

    def _augment(self, observations):
        hat, binary = self._forecast()
        return [
            np.concatenate([np.asarray(observation, dtype=np.float32), np.array([hat, binary], dtype=np.float32)])
            for observation in observations
        ]

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        return self._augment(observations), info

    def step(self, action):
        observations, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observations), reward, terminated, truncated, info
