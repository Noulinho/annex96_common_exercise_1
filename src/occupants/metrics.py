"""Comfort metrics derived from occupant override models."""

from __future__ import annotations

import math

import numpy as np


def occupant_temperature_preference(occupant, default: float = 22.0) -> float:
    if occupant is None:
        return float(default)
    if hasattr(occupant, "T_pref"):
        return float(occupant.T_pref)
    if hasattr(occupant, "preferred_temperature"):
        return float(occupant.preferred_temperature)
    return float(default)


def occupant_temperature_sensitivity(occupant, default: float = 1.0) -> float:
    if occupant is None:
        return float(default)
    if hasattr(occupant, "b"):
        return float(occupant.b)
    if hasattr(occupant, "sensitivity"):
        return float(occupant.sensitivity)
    return float(default)


def occupant_probability_comfort_band(
    occupant,
    *,
    threshold: float = 0.02,
    base_probability: float = 0.002,
    default_band: tuple[float, float] = (20.0, 24.0),
) -> tuple[float, float]:
    """Band where increase and decrease override probabilities are below threshold."""

    if occupant is None:
        return tuple(map(float, default_band))

    sensitivity = occupant_temperature_sensitivity(occupant)
    if sensitivity <= 0.0 or threshold <= base_probability:
        return occupant_temperature_preference(occupant), occupant_temperature_preference(occupant)

    center = occupant_temperature_preference(occupant)
    half_width = math.log(float(threshold) / float(base_probability)) / sensitivity
    return float(center - half_width), float(center + half_width)


def kelvin_hours(
    temperatures,
    *,
    lower: float = 20.0,
    upper: float = 24.0,
) -> float:
    """Mean degree violation outside a fixed comfort band for hourly rollouts."""

    values = np.asarray(temperatures, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    cold = np.maximum(float(lower) - values, 0.0)
    hot = np.maximum(values - float(upper), 0.0)
    return float(np.mean(cold + hot))


def setpoint_kelvin_hours(
    temperatures,
    setpoints,
    *,
    timestep_hours: float = 1.0,
) -> float:
    """Total Kelvin-hours relative to the active setpoint.

    For hourly data, 2 hours at 26.5 C with a 25.0 C setpoint gives
    ``abs(26.5 - 25.0) * 2 = 3.0`` Kelvin-hours.
    """

    temp = np.asarray(temperatures, dtype=float)
    ref = np.asarray(setpoints, dtype=float)
    n = min(temp.size, ref.size)
    if n == 0:
        return 0.0

    temp = temp[:n]
    ref = ref[:n]
    valid = np.isfinite(temp) & np.isfinite(ref)
    if not np.any(valid):
        return 0.0

    return float(np.sum(np.abs(temp[valid] - ref[valid])) * float(timestep_hours))


def override_hours(baseline_setpoints, effective_setpoints, *, tolerance: float = 1e-6) -> float:
    """Count hourly rollout steps where the effective setpoint differs from baseline."""

    baseline = np.asarray(baseline_setpoints, dtype=float)
    effective = np.asarray(effective_setpoints, dtype=float)
    n = min(baseline.size, effective.size)
    if n == 0:
        return 0.0

    baseline = baseline[:n]
    effective = effective[:n]
    valid = np.isfinite(baseline) & np.isfinite(effective)
    return float(np.count_nonzero(np.abs(effective[valid] - baseline[valid]) > float(tolerance)))
