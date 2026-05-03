"""One-state RC model used by the occupant-adaptive MPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RC1R1CParams:
    cop_a: float
    cop_b: float
    mean_indoor_temperature: float
    A: float
    Bd0: float
    Bd1: float
    Bd2: float
    Bu: float
    power_per_unit_action: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class RC1R1CModel:
    def __init__(self, params: RC1R1CParams):
        self.params = params

    def cop(self, outdoor_temperature):
        p = self.params
        return np.clip(p.cop_a + p.cop_b * np.asarray(outdoor_temperature), 0.5, 6.0)

    def predict_next(self, indoor_temperature, outdoor_temperature, solar_irradiance, action, disturbance=0.0):
        p = self.params
        return (
            p.A * indoor_temperature
            + p.Bd0 * p.mean_indoor_temperature
            + p.Bd1 * outdoor_temperature
            + p.Bd2 * solar_irradiance
            + p.Bu * self.cop(outdoor_temperature) * action
            + disturbance
        )

    def rollout(self, initial_temperature, outdoor_temperature, solar_irradiance, action):
        outdoor_temperature = np.asarray(outdoor_temperature, dtype=float)
        solar_irradiance = np.asarray(solar_irradiance, dtype=float)
        action = np.asarray(action, dtype=float)

        predicted = np.zeros_like(outdoor_temperature, dtype=float)
        predicted[0] = float(initial_temperature)
        for k in range(len(predicted) - 1):
            predicted[k + 1] = self.predict_next(
                predicted[k],
                outdoor_temperature[k],
                solar_irradiance[k],
                action[k],
            )

        return predicted

