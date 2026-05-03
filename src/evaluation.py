"""Evaluation metrics shared by RL, MPC, and RLMPC experiments."""

from __future__ import annotations

import numpy as np


def evaluate_rollout(history) -> dict[str, float]:
    """Compute presentation-level metrics from a rollout dataframe."""

    metrics: dict[str, float] = {}

    if "net_electricity_consumption" in history:
        y = history["net_electricity_consumption"].to_numpy(dtype=float)
        metrics["energy_consumption"] = float(np.nansum(y))
        metrics["peak_demand"] = float(np.nanmax(y))
        metrics["ramping"] = float(np.nansum(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
        metrics["peak_to_average_ratio"] = float(np.nanmax(y) / max(np.nanmean(y), 1e-9))

    if {"indoor_temperature", "preferred_temperature"}.issubset(history.columns):
        err = history["indoor_temperature"].to_numpy(dtype=float) - history["preferred_temperature"].to_numpy(dtype=float)
        metrics["comfort_mae"] = float(np.nanmean(np.abs(err)))
        metrics["comfort_rmse"] = float(np.sqrt(np.nanmean(err**2)))

    if "occupant_feedback_signal" in history:
        feedback = history["occupant_feedback_signal"].to_numpy(dtype=float)
        metrics["occupant_overrides"] = float(np.count_nonzero(feedback))
        metrics["override_magnitude"] = float(np.nansum(np.abs(feedback)))

    return metrics


def save_metrics(metrics: dict[str, float], path) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
