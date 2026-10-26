#!/usr/bin/env python
"""Append CE1 normalized-weight comfort/ramp/preheat results."""

from __future__ import annotations

from pathlib import Path

from experiments.scripts import append_ce1_asym_preheat_peakshift_analysis as base


LABEL = "ce1_phiT50_ramp20_preheat20_v1"

base.LABEL = LABEL
base.FIG_DIR = Path("figures") / LABEL
base.WEIGHTS = {
    "label": LABEL,
    "comfort_mode": "asymmetric",
    "comfort_weight": 0.50,
    "comfort_track_weight": 0.0,
    "comfort_cold_weight": 1.0,
    "comfort_hot_weight": 1.0,
    "comfort_cold_tolerance": 0.3,
    "comfort_hot_tolerance": 1.5,
    "comfort_track_scale": 2.0,
    "comfort_cold_scale": 2.0,
    "comfort_hot_scale": 2.0,
    "energy_weight": 0.0,
    "peak_weight": 0.05,
    "ramp_weight": 0.20,
    "shift_weight": 0.05,
    "preheat_weight": 0.20,
    "preheat_horizon": 6,
    "district_target_weight": 0.0,
    "district_target_mode": "peak_only",
    "district_target_scale_mode": "controllable_error_minmax",
}


if __name__ == "__main__":
    base.main()
