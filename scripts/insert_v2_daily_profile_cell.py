#!/usr/bin/env python
"""Insert missing mean daily district profile figure into the v2 MPC section of 02_mpc_results.ipynb."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.district_target import load_ce1_district_target_profiles

LABEL_V1 = "ce1_asym_hot_heavy_shift_v1"
LABEL_V2 = "ce1_asym_hot_heavy_shift_v2"
FIG_DIR = Path("figures") / LABEL_V2
BUILDINGS = dataset_building_entries(DATASET_NAME)
SCALE_MODE = "target_max"


def _rollout_path(label: str, building_id: str) -> Path:
    return Path("results") / "raw" / "mpc" / building_id / "no_occupant" / label / "rollout.csv"


def _district_series(label: str) -> pd.DataFrame:
    baseline_district = None
    target = None
    baseline_heating_sum = None
    controlled_heating_sum = None
    for building in BUILDINGS:
        profiles = load_ce1_district_target_profiles(
            building_name=building["name"],
            simulation_start=744,
            simulation_end=1415,
            dataset_name=DATASET_NAME,
            target_scale_mode=SCALE_MODE,
        )
        base = np.asarray(profiles.baseline_district_load, dtype=float)
        tgt = np.asarray(profiles.target, dtype=float)
        baseline_heat = np.asarray(profiles.baseline_controlled_heating, dtype=float)
        data = pd.read_csv(_rollout_path(label, building["result_id"]))
        controlled_heat = data["heating_electricity_consumption"].to_numpy(dtype=float)
        n = min(len(base), len(tgt), len(baseline_heat), len(controlled_heat))
        if baseline_district is None:
            baseline_district = base[:n]
            target = tgt[:n]
            baseline_heating_sum = np.zeros(n, dtype=float)
            controlled_heating_sum = np.zeros(n, dtype=float)
        else:
            n = min(n, len(baseline_district))
            baseline_district = baseline_district[:n]
            target = target[:n]
            baseline_heating_sum = baseline_heating_sum[:n]
            controlled_heating_sum = controlled_heating_sum[:n]
        baseline_heating_sum += baseline_heat[:n]
        controlled_heating_sum += controlled_heat[:n]
    controlled = baseline_district - baseline_heating_sum + controlled_heating_sum
    return pd.DataFrame({
        "time_step": np.arange(len(controlled), dtype=int),
        "load": controlled,
        "target": target,
        "baseline": baseline_district,
    })


def _baseline_district() -> pd.DataFrame:
    path = Path("results/raw/citylearn_baseline/no_occupant/baseline/district_rollout.csv")
    data = pd.read_csv(path)
    return pd.DataFrame({
        "time_step": np.arange(len(data), dtype=int),
        "load": data["net_electricity_consumption"].to_numpy(dtype=float),
        "target": data["district_target"].to_numpy(dtype=float),
    })


def _make_daily_profile_plot(series_v1, series_v2, baseline) -> Path:
    def hourly_mean(load, n=672):
        load = np.asarray(load[:n], dtype=float)
        hours = np.arange(n) % 24
        return np.array([load[hours == h].mean() for h in range(24)])

    n = min(len(series_v1), len(series_v2), len(baseline))
    h = np.arange(24)
    mean_v1 = hourly_mean(series_v1["load"].values, n)
    mean_v2 = hourly_mean(series_v2["load"].values, n)
    mean_base = hourly_mean(baseline["load"].values, n)
    mean_target = hourly_mean(series_v1["target"].values, n)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(h, mean_base, color="steelblue", marker="o", lw=1.5, label="CityLearn baseline")
    ax.plot(h, mean_v1, color="green", marker="o", lw=1.5, label=f"MPC {LABEL_V1}")
    ax.plot(h, mean_v2, color="darkorange", marker="o", lw=1.5, label=f"MPC {LABEL_V2}")
    ax.plot(h, mean_target, color="black", marker="o", lw=1.2, ls="--", label="CE1 district target")
    ax.set_title(f"Mean daily district profile — MPC v1 vs v2 ({LABEL_V2})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "mpc_mean_daily_profile.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading district series...")
    series_v1 = _district_series(LABEL_V1)
    series_v2 = _district_series(LABEL_V2)
    baseline = _baseline_district()

    fig_path = _make_daily_profile_plot(series_v1, series_v2, baseline)
    print(f"Saved figure: {fig_path}")

    nb_path = Path("analysis/02_mpc_results.ipynb")
    nb = nbf.read(nb_path, as_version=4)
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            f"![MPC mean daily district profile v1 vs v2](../{fig_path.as_posix()})"
        )
    )
    nbf.write(nb, nb_path)
    print(f"Inserted cell into {nb_path}")


if __name__ == "__main__":
    main()
