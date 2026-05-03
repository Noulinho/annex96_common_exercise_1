"""Occupant comfort rewards and metrics."""

from __future__ import annotations

from collections import deque

import numpy as np
from citylearn.reward_function import RewardFunction


class OccupantFeedbackReward(RewardFunction):
    """Reward combining override feedback discomfort and HVAC energy."""

    def __init__(self, env_metadata, beta=0.75, epsilon=0.05, feedback_horizon=4):
        super().__init__(env_metadata)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.feedback_horizon = max(1, int(feedback_horizon))
        self.feedback_buffers = []
        self.energy_scales = []
        if self.env_metadata is not None and self.env_metadata.get("buildings") is not None:
            self.reset()

    def reset(self):
        building_count = 0 if self.env_metadata is None else len(self.env_metadata["buildings"])
        self.feedback_buffers = [
            deque([0.0] * self.feedback_horizon, maxlen=self.feedback_horizon)
            for _ in range(building_count)
        ]
        self.energy_scales = []
        for i in range(building_count):
            metadata = self.env_metadata["buildings"][i]
            nominal_power = float(metadata.get("heating_device", {}).get("nominal_power", 0.0) or 0.0)
            self.energy_scales.append(max(nominal_power, 1.0))

    def _discomfort_from_buffer(self, feedback_buffer):
        discomfort = 0.0
        horizon = max(1, self.feedback_horizon)
        for lag, feedback in enumerate(feedback_buffer):
            weight = np.e - np.exp((lag + 1) / horizon)
            discomfort += weight * abs(float(feedback))
        return float(discomfort)

    def calculate(self, observations):
        if len(self.feedback_buffers) != len(observations):
            self.reset()

        rewards = []
        for i, obs in enumerate(observations):
            feedback = float(np.sign(obs.get("occupant_feedback_signal", 0.0)))
            occupied = float(obs.get("occupant_count", 0.0)) > 0.0
            self.feedback_buffers[i].appendleft(feedback)

            if occupied and abs(feedback) > 0.0:
                discomfort_cost = self._discomfort_from_buffer(self.feedback_buffers[i])
            elif occupied:
                discomfort_cost = -self.epsilon
            else:
                discomfort_cost = 0.0

            hvac_electricity = max(
                float(obs.get("heating_electricity_consumption", obs.get("net_electricity_consumption", 0.0))),
                0.0,
            )
            energy_scale = self.energy_scales[i] if i < len(self.energy_scales) else 1.0
            energy_cost = hvac_electricity / max(energy_scale, 1.0)
            total_cost = self.beta * discomfort_cost + (1.0 - self.beta) * energy_cost
            rewards.append(-total_cost)

        return [sum(rewards)] if self.central_agent else rewards
