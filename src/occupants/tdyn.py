"""Paper-style dynamic comfort reference construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TDynConfig:
    init_mode: str = "schedule"
    delta_up: float = 0.5
    delta_down: float = 0.5
    drift_to_pref: float = 0.05
    lower_bound: float = 18.0
    upper_bound: float = 26.0
    anchor_feedback_to_pref: bool = True


def update_tdyn_reference(
    value: float,
    feedback: int | float,
    *,
    preferred_temperature: float | None = None,
    config: TDynConfig = TDynConfig(),
) -> float:
    """Update the dynamic comfort reference from occupant feedback.

    Zhang-style OITL control uses a dynamic reference temperature that responds
    to real-time feedback while remaining tied to the personalized preference.
    When a fitted preference is available, feedback nudges T_dyn toward that
    preference instead of allowing repeated one-sided feedback to saturate at a
    global comfort bound.
    """

    value = float(value)
    pref = None if preferred_temperature is None else float(preferred_temperature)
    feedback = float(feedback)

    if config.anchor_feedback_to_pref and pref is not None:
        if feedback > 0.0 and value < pref:
            value = min(value + float(config.delta_up), pref)
        elif feedback < 0.0 and value > pref:
            value = max(value - float(config.delta_down), pref)

        if config.drift_to_pref > 0.0:
            value = value + float(config.drift_to_pref) * (pref - value)
    else:
        if config.drift_to_pref > 0.0 and pref is not None:
            value = value + float(config.drift_to_pref) * (pref - value)
        if feedback > 0.0:
            value += float(config.delta_up)
        elif feedback < 0.0:
            value -= float(config.delta_down)

    return float(np.clip(value, config.lower_bound, config.upper_bound))


def detect_override_starts_from_effective_setpoint(
    effective_setpoint,
    baseline_setpoint,
    *,
    duration: int = 3,
    eps: float = 0.25,
) -> np.ndarray:
    """Infer new warmer/cooler override starts from effective-vs-baseline setpoints."""

    effective = np.asarray(effective_setpoint, dtype=np.float32).reshape(-1)
    baseline = np.asarray(baseline_setpoint, dtype=np.float32).reshape(-1)
    n = min(len(effective), len(baseline))
    diff = effective[:n] - baseline[:n]

    starts = np.zeros(n, dtype=np.int64)
    cooldown = 0
    for t in range(n):
        if cooldown > 0:
            cooldown -= 1
            continue

        if abs(diff[t]) > eps:
            starts[t] = 1 if diff[t] > 0 else -1
            cooldown = max(int(duration) - 1, 0)

    return starts


def build_tdyn_from_override_starts(
    baseline_setpoint,
    override_starts,
    *,
    preferred_temperature: float | None = None,
    config: TDynConfig = TDynConfig(),
) -> np.ndarray:
    """Build persistent T_dyn exactly like the working MPC logic."""

    baseline = np.asarray(baseline_setpoint, dtype=np.float32).reshape(-1)
    starts = np.asarray(override_starts, dtype=np.int64).reshape(-1)
    n = min(len(baseline), len(starts))
    baseline = baseline[:n]
    starts = starts[:n]

    if config.init_mode == "schedule":
        tdyn = float(baseline[0])
    elif config.init_mode == "pref":
        if preferred_temperature is None:
            raise ValueError("preferred_temperature is required when init_mode='pref'.")
        tdyn = float(preferred_temperature)
    else:
        raise ValueError("TDynConfig.init_mode must be 'schedule' or 'pref'.")

    trace = np.zeros(n, dtype=np.float32)
    trace[0] = np.clip(tdyn, config.lower_bound, config.upper_bound)
    for t in range(1, n):
        trace[t] = update_tdyn_reference(
            float(trace[t - 1]),
            starts[t],
            preferred_temperature=preferred_temperature,
            config=config,
        )

    return trace


class TDynState:
    """Online persistent T_dyn state updated from real override feedback."""

    def __init__(self, initial_value: float, config: TDynConfig = TDynConfig(), preferred_temperature=None):
        self.config = config
        self.preferred_temperature = preferred_temperature
        self.value = float(np.clip(initial_value, config.lower_bound, config.upper_bound))

    def update(self, feedback: int | float) -> float:
        self.value = update_tdyn_reference(
            self.value,
            feedback,
            preferred_temperature=self.preferred_temperature,
            config=self.config,
        )
        return self.value
