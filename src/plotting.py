"""Reusable plotting helpers for clean analysis notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_temperature_tracking(history: pd.DataFrame, ax=None):
    ax = ax or plt.subplots(figsize=(10, 4))[1]
    if "indoor_temperature" in history:
        ax.plot(history["indoor_temperature"].to_numpy(), label="Indoor temperature")
    if "preferred_temperature" in history:
        ax.plot(history["preferred_temperature"].to_numpy(), label="Preferred temperature")
    if "effective_setpoint" in history:
        ax.plot(history["effective_setpoint"].to_numpy(), label="Effective setpoint", alpha=0.8)
    ax.set_ylabel("Temperature [deg C]")
    ax.legend()
    return ax


def plot_controller_metrics(metrics: pd.DataFrame, ax=None):
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    metrics.plot(kind="bar", ax=ax)
    ax.set_xlabel("")
    ax.legend(title="Metric")
    return ax

