"""Helpers for thermostat override schedules."""

from __future__ import annotations

import numpy as np


def reset_future_heating_setpoints(building, start_index: int) -> float:
    """Restore the future heating setpoint schedule from the baseline schedule."""

    setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point
    baseline = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
    if len(setpoints) == 0:
        return float("nan")

    start_index = max(0, min(int(start_index), len(setpoints) - 1))
    setpoints[start_index:] = np.asarray(baseline[start_index:], dtype=setpoints.dtype)
    return float(setpoints[start_index])


def active_override_steps(occupant) -> int:
    """Return active schedule steps including the current next-step setpoint."""

    if getattr(occupant, "override_setpoint", None) is None:
        return 0
    return max(1, int(getattr(occupant, "override_timer", 0)) + 1)


def apply_heating_override_schedule(building, start_index: int, setpoint: float, steps: int) -> float:
    """Restore the future schedule, then write the active override window."""

    setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point
    if len(setpoints) == 0:
        return float("nan")

    start_index = max(0, min(int(start_index), len(setpoints) - 1))
    reset_future_heating_setpoints(building, start_index)

    steps = max(0, int(steps))
    if steps > 0:
        end_index = min(start_index + steps, len(setpoints))
        setpoints[start_index:end_index] = float(setpoint)

    return float(setpoints[start_index])
