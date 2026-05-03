"""Two-state RC exploration model.

This is intentionally separate from the main 1R1C MPC model so the extension
does not muddy the main experimental story.
"""

from __future__ import annotations

import numpy as np


def discretize_2r2c(params, dt_seconds: float = 3600.0):
    ram, rao, ca, cm, ku, ks, k0 = params
    ac = np.array(
        [
            [-(1 / (ca * ram) + 1 / (ca * rao)), 1 / (ca * ram)],
            [1 / (cm * ram), -1 / (cm * ram)],
        ],
        dtype=float,
    )
    bc = np.array(
        [
            [1 / (ca * rao), ks / ca, ku / ca, k0 / ca],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return np.eye(2) + dt_seconds * ac, dt_seconds * bc


def rollout_2r2c(params_fit: dict[str, float], initial_air_temperature, outdoor_temperature, irradiance, action):
    a, b = discretize_2r2c(
        [
            params_fit["Ram"],
            params_fit["Rao"],
            params_fit["Ca"],
            params_fit["Cm"],
            params_fit["ku"],
            params_fit["ks"],
            params_fit["k0"],
        ]
    )
    x = np.array([initial_air_temperature, initial_air_temperature], dtype=float)
    predicted = np.zeros_like(outdoor_temperature, dtype=float)

    for k in range(len(predicted)):
        u = np.array([outdoor_temperature[k], irradiance[k], action[k], 1.0])
        x = a @ x + b @ u
        predicted[k] = x[0]

    return predicted

