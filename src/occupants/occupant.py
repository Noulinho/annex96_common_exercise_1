"""Simple stochastic occupant override model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Occupant:
    preferred_temperature: float
    sensitivity: float
    max_probability: float = 0.4
    duration: int = 3

    def __post_init__(self):
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def prob_increase(self, indoor_temperature: float) -> float:
        delta = max(0.0, self.preferred_temperature - indoor_temperature)
        return min(0.002 * np.exp(self.sensitivity * delta), self.max_probability)

    def prob_decrease(self, indoor_temperature: float) -> float:
        delta = max(0.0, indoor_temperature - self.preferred_temperature)
        return min(0.002 * np.exp(self.sensitivity * delta), self.max_probability)

    @staticmethod
    def override_delta(probability: float) -> float:
        if probability < 0.2:
            return 0.5
        if probability < 0.4:
            return 1.0
        return 2.0

    def clear_override(self) -> None:
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def step(self, indoor_temperature: float, scheduled_setpoint: float, occupied: bool = True) -> float:
        self.last_override_started = False

        if not occupied:
            self.clear_override()
            return scheduled_setpoint

        if self.override_timer > 0:
            self.override_timer -= 1
            return float(self.override_setpoint)

        p_inc = self.prob_increase(indoor_temperature)
        p_dec = self.prob_decrease(indoor_temperature)
        draw = np.random.rand()

        if draw < p_inc:
            self.last_override_started = True
            self.override_timer = max(self.duration - 1, 0)
            self.override_setpoint = scheduled_setpoint + self.override_delta(p_inc)
        elif draw < p_inc + p_dec:
            self.last_override_started = True
            self.override_timer = max(self.duration - 1, 0)
            self.override_setpoint = scheduled_setpoint - self.override_delta(p_dec)
        else:
            self.override_setpoint = scheduled_setpoint

        return float(self.override_setpoint)

