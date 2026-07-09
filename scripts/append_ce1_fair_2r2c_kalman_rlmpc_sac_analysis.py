#!/usr/bin/env python
"""Append fair CE1 2R2C+Kalman MPC/RLMPC/SAC comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from src.buildings import DATASET_NAME, dataset_building_entries
from src.occupants.metrics import kelvin_hours
from src.rewards import load_simple_v4_price_profile, time_bias_setpoint_profile


ANALYSIS_DIR = Path("analysis")
RESULTS = Path("results")
FIG_DIR = Path("figures") / "ce1_time_bias_guard_1c_2r2c_kalman_proxy_fair_p100k"
BUILDINGS = dataset_building_entries(DATASET_NAME)
FEB_START = 744

MPC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman"
RLMPC_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_rlmpc_h24_west_exact_proxy_fair_p100k"
SAC_BASE_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi9_proxy_fair_a1"
SAC_BAD_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi9_bad10_proxy_fair_a1"
MPC_REPLICATE_LABEL = "ce1_time_bias_guard_1c_2r2c_kalman_replicate"
MARKER = "## CE1 Exact-Proxy Fair 2R2C+Kalman MPC/RLMPC/SAC"

RUNS = [
    {
        "key": "mpc",
        "display": "MPC 2R2C+Kalman exact proxy cost",
        "controller": "mpc",
        "label": MPC_LABEL,
        "hybrid": False,
    },
    {
        "key": "rlmpc",
        "display": "RLMPC 2R2C+Kalman exact proxy reward",
        "controller": "rlmpc",
        "label": RLMPC_LABEL,
        "hybrid": False,
    },
    {
        "key": "sac",
        "display": "SAC safe-PI exact proxy reward (9 ep + bad +10)",
        "controller": "sac_merlin",
        "label": SAC_BASE_LABEL,
        "hybrid": True,
    },
]

SHORT_CONTROLLER_NAMES = {
    "CityLearn baseline": "CityLearn",
    RUNS[0]["display"]: "MPC",
    RUNS[1]["display"]: "RLMPC",
    RUNS[2]["display"]: "SAC hybrid",
}

WEIGHTS = {
    "comfort_weight": 0.5,
    "shift_weight": 0.5,
    "track_weight": 3.0,
    "cold_weight": 60.0,
    "hot_weight": 15.0,
    "track_scale": 2.0,
    "cold_scale": 1.5,
    "hot_scale": 2.0,
    "guard_band": 2.0,
    "power_scale": 5.0,
    "bias_amplitude": 1.0,
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _raw_dir(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "raw" / controller / building_id / "no_occupant" / label


def _summary_path(controller: str, building_id: str, label: str) -> Path:
    return RESULTS / "summaries" / controller / building_id / "no_occupant" / f"{label}_metrics.json"


def _bad_building_ids() -> set[str]:
    bad: set[str] = set()
    for building in BUILDINGS:
        building_id = building["result_id"]
        bad_kpis = _raw_dir("sac_merlin", building_id, SAC_BAD_LABEL) / "kpis.csv"
        if bad_kpis.exists():
            bad.add(building_id)
            continue
        base_kpis = _raw_dir("sac_merlin", building_id, SAC_BASE_LABEL) / "kpis.csv"
        if not base_kpis.exists():
            continue
        discomfort = _kpi_values(base_kpis).get("discomfort_proportion", np.nan)
        if np.isfinite(discomfort) and discomfort > 0.10:
            bad.add(building_id)
    return bad


def _sac_final_label(building_id: str) -> str:
    bad_kpis = _raw_dir("sac_merlin", building_id, SAC_BAD_LABEL) / "kpis.csv"
    return SAC_BAD_LABEL if building_id in _bad_building_ids() and bad_kpis.exists() else SAC_BASE_LABEL


def _rollout_path(run: dict[str, object], building_id: str) -> Path:
    label = str(run["label"])
    if bool(run.get("hybrid")):
        label = _sac_final_label(building_id)
    return _raw_dir(str(run["controller"]), building_id, label) / "rollout.csv"


def _kpis_path(run: dict[str, object], building_id: str) -> Path:
    label = str(run["label"])
    if bool(run.get("hybrid")):
        label = _sac_final_label(building_id)
    return _raw_dir(str(run["controller"]), building_id, label) / "kpis.csv"


def _kpi_values(path: Path) -> dict[str, float]:
    data = _read(path)
    district = data[data.get("level", "") == "district"].copy()
    if district.empty:
        district = data
    return {
        str(row["cost_function"]): float(row["value"])
        for _, row in district.iterrows()
        if pd.notna(row.get("value"))
    }


def _setpoint(data: pd.DataFrame) -> np.ndarray:
    for column in ("scheduled_setpoint", "baseline_setpoint", "effective_setpoint", "tmin"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    raise ValueError("No setpoint-like column found.")


def _time_step(data: pd.DataFrame) -> np.ndarray:
    if "time_step" in data.columns:
        return pd.to_numeric(data["time_step"], errors="coerce").fillna(0).to_numpy(dtype=int)
    return np.arange(len(data), dtype=int)


def _load(data: pd.DataFrame) -> np.ndarray:
    for column in ("net_electricity_consumption", "electricity_consumption", "heating_electricity_consumption"):
        if column in data.columns:
            return pd.to_numeric(data[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    raise ValueError("Rollout has no electricity-like column.")


def _heating(data: pd.DataFrame) -> np.ndarray:
    if "heating_electricity_consumption" in data.columns:
        values = pd.to_numeric(data["heating_electricity_consumption"], errors="coerce").fillna(0.0)
        return np.maximum(values.to_numpy(dtype=float), 0.0)
    values = pd.to_numeric(data["action"], errors="coerce").fillna(0.0)
    return np.maximum(values.to_numpy(dtype=float), 0.0)


def _rc2_power_per_unit_action(building: dict[str, object]) -> float:
    path = RESULTS / "models" / "rc" / f"2r2c_{building['short_name']}_january_kalman.json"
    params = json.loads(path.read_text())
    return float(params["power_per_unit_action"])


def _action_proxy_heat(data: pd.DataFrame, building: dict[str, object]) -> np.ndarray:
    action = pd.to_numeric(data["action"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.maximum(action, 0.0) * _rc2_power_per_unit_action(building)


def _numeric_max_abs_diff(left: pd.DataFrame, right: pd.DataFrame) -> float:
    common = [column for column in left.columns if column in right.columns]
    numeric = [
        column
        for column in common
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column])
    ]
    if not numeric:
        return 0.0
    n = min(len(left), len(right))
    return float(np.nanmax(np.abs(left.loc[: n - 1, numeric].to_numpy(dtype=float) - right.loc[: n - 1, numeric].to_numpy(dtype=float))))


def _mpc_recovery_check() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS[:4]:
        ref_dir = _raw_dir("mpc", building["result_id"], MPC_LABEL)
        replicate_dir = _raw_dir("mpc", building["result_id"], MPC_REPLICATE_LABEL)
        row = {"building": building["display"], "building_id": building["result_id"]}
        exact = True
        for stem in ("rollout", "kpis", "district_kpis"):
            ref_path = ref_dir / f"{stem}.csv"
            replicate_path = replicate_dir / f"{stem}.csv"
            if not ref_path.exists() or not replicate_path.exists():
                row[f"{stem}_max_abs_diff"] = np.nan
                row[f"{stem}_same_shape"] = False
                exact = False
                continue
            left = _read(ref_path)
            right = _read(replicate_path)
            max_diff = _numeric_max_abs_diff(left, right)
            same_shape = left.shape == right.shape
            row[f"{stem}_max_abs_diff"] = max_diff
            row[f"{stem}_same_shape"] = same_shape
            exact = exact and same_shape and max_diff == 0.0
        row["exact_zero_diff"] = exact
        rows.append(row)
    return pd.DataFrame(rows)


def _bias_and_stress(n: int) -> tuple[np.ndarray, np.ndarray]:
    bias = np.asarray(
        time_bias_setpoint_profile(
            simulation_start=FEB_START,
            simulation_end=FEB_START + n - 1,
            amplitude=WEIGHTS["bias_amplitude"],
        ),
        dtype=float,
    )
    stress = np.asarray(load_simple_v4_price_profile(simulation_start=FEB_START, exp_beta=2.0), dtype=float)
    if stress.size < n:
        stress = np.resize(stress, n)
    return bias[:n], stress[:n]


def _baseline_district() -> pd.DataFrame:
    data = _read(RESULTS / "raw" / "citylearn_baseline" / "no_occupant" / "baseline" / "district_rollout.csv")
    return pd.DataFrame(
        {
            "time_step": np.arange(len(data), dtype=int),
            "load": pd.to_numeric(data["net_electricity_consumption"], errors="coerce").fillna(0.0),
            "target": pd.to_numeric(data["district_target"], errors="coerce").fillna(0.0),
            "controller": "CityLearn baseline",
        }
    )


def _building_kpis() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for building in BUILDINGS:
            rollout = _read(_rollout_path(run, building["result_id"]))
            tin = pd.to_numeric(rollout["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            values = _kpi_values(_kpis_path(run, building["result_id"]))
            rows.append(
                {
                    "controller": run["display"],
                    "building": building["display"],
                    "building_id": building["result_id"],
                    "source_label": _sac_final_label(building["result_id"]) if run["key"] == "sac" else run["label"],
                    "electricity_consumption_total": values.get("electricity_consumption_total", np.nan),
                    "discomfort_%": 100.0 * values.get("discomfort_proportion", np.nan),
                    "cold_%": 100.0 * values.get("discomfort_cold_proportion", np.nan),
                    "hot_%": 100.0 * values.get("discomfort_hot_proportion", np.nan),
                    "kelvin_hours": kelvin_hours(tin, lower=20.0, upper=24.0),
                    "daily_peak_average": values.get("daily_peak_average", np.nan),
                    "ramping_average": values.get("ramping_average", np.nan),
                    "feb_tin_max": float(np.nanmax(tin)),
                }
            )
    return pd.DataFrame(rows)


def _mean_building_kpis(building_kpis: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "electricity_consumption_total",
        "discomfort_%",
        "cold_%",
        "hot_%",
        "kelvin_hours",
        "daily_peak_average",
        "ramping_average",
        "feb_tin_max",
    ]
    out = building_kpis.groupby("controller", as_index=False)[columns].mean(numeric_only=True)
    order = {run["display"]: i for i, run in enumerate(RUNS)}
    out["_order"] = out["controller"].map(order)
    return out.sort_values("_order").drop(columns="_order")


def _district_frame(run: dict[str, object]) -> pd.DataFrame:
    loads = []
    time = None
    for building in BUILDINGS:
        data = _read(_rollout_path(run, building["result_id"]))
        values = _load(data)
        if time is None:
            time = _time_step(data)
        n = min(len(time), len(values))
        time = time[:n]
        loads.append(values[:n])
    n = min(len(values) for values in loads)
    baseline = _baseline_district()
    return pd.DataFrame(
        {
            "time_step": np.asarray(time[:n], dtype=int),
            "load": np.sum([values[:n] for values in loads], axis=0),
            "target": baseline["target"].to_numpy(dtype=float)[:n],
            "controller": run["display"],
        }
    )


def _district_kpis(frames: list[pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0]["load"].to_numpy(dtype=float)
    base_total = max(float(np.sum(baseline)), 1e-9)
    base_peak = max(float(np.max(baseline)), 1e-9)
    base_ramp = max(float(np.sum(np.abs(np.diff(baseline)))), 1e-9)
    rows = []
    for frame in frames:
        load = frame["load"].to_numpy(dtype=float)
        target = frame["target"].to_numpy(dtype=float)
        n = min(len(load), len(target))
        load, target = load[:n], target[:n]
        hourly = pd.DataFrame({"hour": np.arange(n) % 24, "load": load}).groupby("hour")["load"].mean()
        ramp = float(np.sum(np.abs(np.diff(load)))) if n > 1 else 0.0
        rows.append(
            {
                "controller": frame["controller"].iloc[0],
                "total_kwh": float(np.sum(load)),
                "total_vs_citylearn": float(np.sum(load) / base_total),
                "peak_kwh": float(np.max(load)),
                "peak_vs_citylearn": float(np.max(load) / base_peak),
                "daily_profile_peak_kwh": float(hourly.max()),
                "daily_profile_peak_hour": int(hourly.idxmax()),
                "ramping_abs_kwh": ramp,
                "ramping_vs_citylearn": float(ramp / base_ramp),
                "target_rmse_kwh": float(np.sqrt(np.mean((load - target) ** 2))),
                "above_target_kwh": float(np.sum(np.maximum(load - target, 0.0))),
            }
        )
    return pd.DataFrame(rows)


def _term_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    hourly_rows = []
    for run in RUNS:
        for building in BUILDINGS:
            data = _read(_rollout_path(run, building["result_id"]))
            tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
            tsp = _setpoint(data)
            heat = _action_proxy_heat(data, building)
            time_step = _time_step(data)
            n = min(len(tin), len(tsp), len(heat), len(time_step))
            tin, tsp, heat, time_step = tin[:n], tsp[:n], heat[:n], time_step[:n]
            bias, stress = _bias_and_stress(n)
            tref = tsp + bias
            cold_guard = np.maximum(tsp - WEIGHTS["guard_band"] - tin, 0.0)
            hot_guard = np.maximum(tin - tsp - WEIGHTS["guard_band"], 0.0)
            track = WEIGHTS["comfort_weight"] * WEIGHTS["track_weight"] * ((tin - tref) / WEIGHTS["track_scale"]) ** 2
            cold = WEIGHTS["comfort_weight"] * WEIGHTS["cold_weight"] * (cold_guard / WEIGHTS["cold_scale"]) ** 2
            hot = WEIGHTS["comfort_weight"] * WEIGHTS["hot_weight"] * (hot_guard / WEIGHTS["hot_scale"]) ** 2
            shift = WEIGHTS["shift_weight"] * stress * heat / WEIGHTS["power_scale"]
            stage = track + cold + hot + shift
            guard_violation = (cold_guard > 0.0) | (hot_guard > 0.0)
            for term, values in {
                "tracking_target": track,
                "guard_cold": cold,
                "guard_hot": hot,
                "time_weighted_heat": shift,
                "stage_cost": stage,
                "mae_to_true_tsp_c": np.abs(tin - tsp),
                "mae_to_biased_tref_c": np.abs(tin - tref),
            }.items():
                rows.append(
                    {
                        "controller": run["display"],
                        "building": building["display"],
                        "term": term,
                        "mean": float(np.nanmean(values)),
                        "max": float(np.nanmax(values)),
                    }
                )
            rows.append(
                {
                    "controller": run["display"],
                    "building": building["display"],
                    "term": "guard_violation_%",
                    "mean": float(100.0 * np.nanmean(guard_violation)),
                    "max": float(100.0 * np.nanmax(guard_violation)),
                }
            )
            hourly_rows.append(
                pd.DataFrame(
                    {
                        "controller": run["display"],
                        "hour": (time_step + FEB_START) % 24,
                        "tin_minus_tsp": tin - tsp,
                        "action_proxy_heat": heat,
                        "guard_violation": guard_violation.astype(float),
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(hourly_rows, ignore_index=True)


def _rlmpc_summary_row(building_id: str) -> pd.Series:
    path = RESULTS / "summaries" / "rlmpc" / building_id / "no_occupant" / "summary.csv"
    if not path.exists():
        return pd.Series(dtype=object)
    data = _read(path)
    rows = data[data["case"].astype(str) == RLMPC_LABEL]
    return rows.tail(1).iloc[0] if not rows.empty else pd.Series(dtype=object)


def _sac_manifest(building_id: str, label: str) -> dict[str, object]:
    path = _raw_dir("sac_merlin", building_id, label) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("metrics", {})


def _exact_proxy_metadata_check() -> pd.DataFrame:
    rows = []
    for building in BUILDINGS:
        building_id = building["result_id"]
        rc2_power = _rc2_power_per_unit_action(building)
        rlm_row = _rlmpc_summary_row(building_id)
        sac_label = _sac_final_label(building_id)
        sac_metrics = _sac_manifest(building_id, sac_label)
        rlm_power = float(rlm_row.get("reward_action_proxy_power_per_unit_action", np.nan))
        sac_power = float(sac_metrics.get("reward_action_proxy_power_per_unit_action", np.nan))
        rows.append(
            {
                "building": building["display"],
                "building_id": building_id,
                "rc2_power_per_unit_action": rc2_power,
                "rlmpc_reward_power_source": rlm_row.get("reward_power_source", ""),
                "rlmpc_power_per_unit_action": rlm_power,
                "sac_source_label": sac_label,
                "sac_reward_power_source": sac_metrics.get("reward_power_source", ""),
                "sac_power_per_unit_action": sac_power,
                "rlmpc_coeff_abs_diff": abs(rlm_power - rc2_power) if np.isfinite(rlm_power) else np.nan,
                "sac_coeff_abs_diff": abs(sac_power - rc2_power) if np.isfinite(sac_power) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _sac_bad_delta() -> pd.DataFrame:
    rows = []
    bad_buildings = _bad_building_ids()
    for building in BUILDINGS:
        if building["result_id"] not in bad_buildings:
            continue
        bad_path = _raw_dir("sac_merlin", building["result_id"], SAC_BAD_LABEL) / "kpis.csv"
        if not bad_path.exists():
            continue
        base = _kpi_values(_raw_dir("sac_merlin", building["result_id"], SAC_BASE_LABEL) / "kpis.csv")
        resumed = _kpi_values(bad_path)
        row = {
            "building": building["display"],
            "building_id": building["result_id"],
            "discomfort_%_plus9": 100.0 * base.get("discomfort_proportion", np.nan),
            "discomfort_%_bad10": 100.0 * resumed.get("discomfort_proportion", np.nan),
            "cold_%_bad10": 100.0 * resumed.get("discomfort_cold_proportion", np.nan),
            "hot_%_bad10": 100.0 * resumed.get("discomfort_hot_proportion", np.nan),
            "energy_plus9": base.get("electricity_consumption_total", np.nan),
            "energy_bad10": resumed.get("electricity_consumption_total", np.nan),
        }
        row["delta_discomfort_%"] = row["discomfort_%_bad10"] - row["discomfort_%_plus9"]
        row["delta_energy"] = row["energy_bad10"] - row["energy_plus9"]
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("delta_discomfort_%") if not out.empty else out


def _safe_pi_january_rows() -> dict[str, pd.DataFrame]:
    out = {}
    for building in BUILDINGS:
        path = _raw_dir("sac_merlin", building["result_id"], SAC_BASE_LABEL) / "online_training_rollout.csv"
        data = _read(path)
        subset = data[(data["episode"] == 0) & (data["phase"].astype(str) == "safe_pi_exploration")].copy()
        out[building["result_id"]] = subset
    return out


def _sac_training_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reward_frames = []
    loss_frames = []
    labels = [
        (SAC_BASE_LABEL, set(b["result_id"] for b in BUILDINGS)),
        (SAC_BAD_LABEL, _bad_building_ids()),
    ]
    for label, allowed in labels:
        for building in BUILDINGS:
            if building["result_id"] not in allowed:
                continue
            case_dir = _raw_dir("sac_merlin", building["result_id"], label)
            reward_path = case_dir / "online_training_rollout.csv"
            if reward_path.exists():
                data = _read(reward_path)
                data["building"] = building["display"]
                data["building_id"] = building["result_id"]
                data["source_label"] = label
                reward_frames.append(data)
            loss_path = case_dir / "sac_training_losses.csv"
            if loss_path.exists():
                losses = _read(loss_path)
                losses["building"] = building["display"]
                losses["building_id"] = building["result_id"]
                losses["source_label"] = label
                loss_frames.append(losses)
    rewards = pd.concat(reward_frames, ignore_index=True) if reward_frames else pd.DataFrame()
    losses = pd.concat(loss_frames, ignore_index=True) if loss_frames else pd.DataFrame()
    return rewards, losses


def _make_comfort_grid(run: dict[str, object]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        data = _read(_rollout_path(run, building["result_id"]))
        t = _time_step(data)
        tin = pd.to_numeric(data["indoor_temperature"], errors="coerce").to_numpy(dtype=float)
        tsp = _setpoint(data)
        n = min(len(t), len(tin), len(tsp))
        bias, _ = _bias_and_stress(n)
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.65, ls="--", color="black", label="Tsp")
        ax.plot(t[:n], tsp[:n] + bias[:n], lw=0.65, ls=":", color="tab:orange", label="Tsp+bias")
        ax.fill_between(t[:n], tsp[:n] - 2.0, tsp[:n] + 2.0, color="gray", alpha=0.13)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{run['display']} | February comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / f"{run['key']}_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_safe_pi_grid(rows: dict[str, pd.DataFrame]) -> Path:
    fig, axes = plt.subplots(5, 5, figsize=(22, 15), sharex=True)
    axes = axes.ravel()
    for ax, building in zip(axes, BUILDINGS):
        data = rows[building["result_id"]]
        t = pd.to_numeric(data["time_step"], errors="coerce").to_numpy(dtype=float)
        tin = pd.to_numeric(data["tin_before_action"], errors="coerce").to_numpy(dtype=float)
        tsp = pd.to_numeric(data["tsp_before_action"], errors="coerce").to_numpy(dtype=float)
        n = min(len(t), len(tin), len(tsp))
        ax.plot(t[:n], tin[:n], lw=0.75, color="steelblue", label="Tin")
        ax.plot(t[:n], tsp[:n], lw=0.65, ls="--", color="black", label="Tsp")
        ax.fill_between(t[:n], tsp[:n] - 2.0, tsp[:n] + 2.0, color="gray", alpha=0.13)
        ax.set_title(building["display"], fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("SAC safe-PI January bootstrap episode 0 comfort", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "sac_safe_pi_january_episode0_comfort_all25.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _make_district_plots(frames: list[pd.DataFrame]) -> tuple[Path, Path]:
    combined = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(15, 5))
    for name, group in combined.groupby("controller", sort=False):
        ax.plot(group["time_step"], group["load"], lw=1.0, label=name)
    ax.plot(frames[0]["time_step"], frames[0]["target"], color="black", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Fair 2R2C+Kalman district load - February")
    ax.set_xlabel("February time step")
    ax.set_ylabel("District electricity [kWh]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    load_path = FIG_DIR / "district_load_timeseries.png"
    fig.savefig(load_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    daily = combined.assign(hour=combined["time_step"] % 24).groupby(["controller", "hour"], as_index=False)["load"].mean()
    target_daily = frames[0].assign(hour=frames[0]["time_step"] % 24).groupby("hour", as_index=False)["target"].mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    for name, group in daily.groupby("controller", sort=False):
        ax.plot(group["hour"], group["load"], marker="o", lw=1.3, label=name)
    ax.plot(target_daily["hour"], target_daily["target"], color="black", marker="o", lw=1.0, ls="--", label="CE1 target")
    ax.set_title("Mean hourly district load - February")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    daily_path = FIG_DIR / "district_mean_hourly_profile.png"
    fig.savefig(daily_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return load_path, daily_path


def _make_term_plots(term_summary: pd.DataFrame, hourly: pd.DataFrame) -> tuple[Path, Path]:
    terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost"]
    pivot = term_summary[term_summary["term"].isin(terms)].pivot(index="term", columns="controller", values="mean")
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.loc[terms].plot(kind="bar", ax=ax)
    ax.set_title("Mean weighted stage-cost terms")
    ax.set_ylabel("Mean weighted term value")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    terms_path = FIG_DIR / "weighted_terms_by_controller.png"
    fig.savefig(terms_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    hourly_mean = hourly.groupby(["controller", "hour"], as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for name, group in hourly_mean.groupby("controller", sort=False):
        axes[0].plot(group["hour"], group["tin_minus_tsp"], marker="o", lw=1.2, label=name)
        axes[1].plot(group["hour"], group["action_proxy_heat"], marker="o", lw=1.2, label=name)
        axes[2].plot(group["hour"], 100.0 * group["guard_violation"], marker="o", lw=1.2, label=name)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel("Tin - Tsp [C]")
    axes[1].set_ylabel("RC action proxy heat")
    axes[2].set_ylabel("Guard violation [%]")
    axes[2].set_xlabel("Hour of day")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Hourly response to +1/-1 C time-bias schedule")
    fig.tight_layout()
    response_path = FIG_DIR / "hourly_bias_response.png"
    fig.savefig(response_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return terms_path, response_path


def _make_sac_training_plots(rewards: pd.DataFrame, losses: pd.DataFrame) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    paths: list[Path] = []
    reward_summary = pd.DataFrame()
    loss_summary = pd.DataFrame()

    if not rewards.empty:
        sac_rewards = rewards[rewards["phase"].astype(str).str.contains("sac", case=False, na=False)].copy()
        sac_rewards["raw_reward_plot"] = pd.to_numeric(
            sac_rewards["raw_reward"] if "raw_reward" in sac_rewards.columns else sac_rewards["reward"],
            errors="coerce",
        )
        sac_rewards["reward_plot"] = pd.to_numeric(sac_rewards["reward"], errors="coerce")
        sac_rewards["reset_flag"] = sac_rewards.get("temperature_reset_applied", False)
        if sac_rewards["reset_flag"].dtype == object:
            sac_rewards["reset_flag"] = sac_rewards["reset_flag"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        else:
            sac_rewards["reset_flag"] = pd.to_numeric(sac_rewards["reset_flag"], errors="coerce").fillna(0.0)
        reward_summary = (
            sac_rewards.groupby(["episode"], as_index=False)
            .agg(
                raw_reward_mean=("raw_reward_plot", "mean"),
                raw_reward_min=("raw_reward_plot", "min"),
                raw_reward_max=("raw_reward_plot", "max"),
                reward_mean=("reward_plot", "mean"),
                reset_fraction=("reset_flag", "mean"),
                action_mean=("action", "mean"),
            )
            .sort_values("episode")
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(reward_summary["episode"], reward_summary["raw_reward_mean"], marker="o", lw=1.3, label="Raw reward mean")
        if not np.allclose(
            reward_summary["raw_reward_mean"],
            reward_summary["reward_mean"],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ):
            ax.plot(
                reward_summary["episode"],
                reward_summary["reward_mean"],
                marker="o",
                lw=1.1,
                ls="--",
                label="Replay reward mean",
            )
        mean_values = reward_summary[["raw_reward_mean", "reward_mean"]].to_numpy(dtype=float).ravel()
        mean_values = mean_values[np.isfinite(mean_values)]
        if mean_values.size:
            low = float(np.nanmin(mean_values))
            high = max(float(np.nanmax(mean_values)), 0.0)
            span = max(high - low, 1.0)
            ax.set_ylim(low - 0.08 * span, high + 0.08 * span)
        ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("SAC online training reward by episode - mean only")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean raw reward per step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIG_DIR / "sac_online_training_reward.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(reward_summary["episode"], 100.0 * reward_summary["reset_fraction"], marker="o", lw=1.3)
        ax.set_title("SAC training temperature-reset frequency")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Reset-applied steps [%]")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / "sac_training_reset_fraction.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not losses.empty:
        cols = [c for c in ["soft_q1_loss", "soft_q2_loss", "policy_loss", "alpha_loss", "alpha"] if c in losses.columns]
        loss_summary = losses.groupby("episode", as_index=False)[cols].agg(["mean", "max"])
        loss_summary.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c for c in loss_summary.columns.to_flat_index()]
        fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.8 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            ax.plot(loss_summary["episode"], loss_summary[f"{col}_mean"], marker="o", lw=1.2, label="mean")
            ax.plot(loss_summary["episode"], loss_summary[f"{col}_max"], marker="o", lw=0.9, ls="--", label="max")
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Training episode")
        fig.suptitle("SAC loss diagnostics")
        fig.tight_layout()
        path = FIG_DIR / "sac_training_losses.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths, reward_summary, loss_summary


def _make_bad_delta_plot(delta: pd.DataFrame) -> Path:
    path = FIG_DIR / "sac_bad_building_plus9_vs_bad10_discomfort.png"
    if delta.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No bad-building continuation rows", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return path
    plot = delta.sort_values("discomfort_%_plus9", ascending=False)
    x = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.38
    ax.bar(x - width / 2, plot["discomfort_%_plus9"], width, label="SAC plus9")
    ax.bar(x + width / 2, plot["discomfort_%_bad10"], width, label="SAC bad +10")
    ax.set_xticks(x)
    ax.set_xticklabels(plot["building"], rotation=45, ha="right")
    ax.set_ylabel("February discomfort [%]")
    ax.set_title("Bad-building SAC continuation effect")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _markdown_table(df: pd.DataFrame, digits: int = 4, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data._"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    headers = list(data.columns)
    rows = [headers] + data.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |"

    return "\n".join([fmt(headers), "| " + " | ".join("-" * width for width in widths) + " |", *[fmt(row) for row in rows[1:]]])


def _short_controller_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "controller" in out.columns:
        out["controller"] = out["controller"].map(SHORT_CONTROLLER_NAMES).fillna(out["controller"])
    return out


def _pivot_terms(term_summary: pd.DataFrame, value: str) -> pd.DataFrame:
    terms = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost", "guard_violation_%"]
    out = (
        term_summary[term_summary["term"].isin(terms)]
        .pivot(index="controller", columns="term", values=value)
        .reset_index()
    )
    out = _short_controller_names(out)
    return out.rename(
        columns={
            "tracking_target": "track",
            "guard_cold": "cold_guard",
            "guard_hot": "hot_guard",
            "time_weighted_heat": "grid_heat",
            "stage_cost": "stage",
            "guard_violation_%": "guard_%",
        }
    )


def _image(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.as_posix()})")


def _append_once(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if MARKER in str(cell.get("source", "")):
            notebook.cells[index:index + len(cells)] = cells
            nbf.write(notebook, path)
            print(f"Updated {MARKER} in {path}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {MARKER} to {path}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    building_kpis = _building_kpis()
    mean_kpis = _mean_building_kpis(building_kpis)
    district_frames = [_baseline_district(), *[_district_frame(run) for run in RUNS]]
    district_kpis = _district_kpis(district_frames)
    term_by_building, hourly = _term_diagnostics()
    term_summary = (
        term_by_building.groupby(["controller", "term"], as_index=False)
        .agg(mean=("mean", "mean"), max=("max", "max"))
        .sort_values(["controller", "term"])
    )
    bad_delta = _sac_bad_delta()
    rewards, losses = _sac_training_frames()
    recovery_check = _mpc_recovery_check()
    exact_proxy_check = _exact_proxy_metadata_check()

    prefix = "ce1_time_bias_guard_1c_2r2c_kalman_proxy_fair_p100k"
    building_kpis.to_csv(ANALYSIS_DIR / f"{prefix}_building_kpis.csv", index=False)
    mean_kpis.to_csv(ANALYSIS_DIR / f"{prefix}_mean_building_kpis.csv", index=False)
    district_kpis.to_csv(ANALYSIS_DIR / f"{prefix}_district_kpis.csv", index=False)
    term_by_building.to_csv(ANALYSIS_DIR / f"{prefix}_term_diagnostics_by_building.csv", index=False)
    term_summary.to_csv(ANALYSIS_DIR / f"{prefix}_term_diagnostics.csv", index=False)
    hourly.to_csv(ANALYSIS_DIR / f"{prefix}_hourly_bias_response.csv", index=False)
    bad_delta.to_csv(ANALYSIS_DIR / f"{prefix}_sac_bad_delta.csv", index=False)
    recovery_check.to_csv(ANALYSIS_DIR / f"{prefix}_mpc_replicate_check.csv", index=False)
    exact_proxy_check.to_csv(ANALYSIS_DIR / f"{prefix}_exact_proxy_metadata_check.csv", index=False)
    rewards.to_csv(ANALYSIS_DIR / f"{prefix}_sac_online_training_rewards_raw.csv", index=False)
    losses.to_csv(ANALYSIS_DIR / f"{prefix}_sac_training_losses_raw.csv", index=False)

    comfort_paths = {run["key"]: _make_comfort_grid(run) for run in RUNS}
    safe_pi_path = _make_safe_pi_grid(_safe_pi_january_rows())
    district_load_path, district_daily_path = _make_district_plots(district_frames)
    terms_path, response_path = _make_term_plots(term_summary, hourly)
    training_paths, reward_summary, loss_summary = _make_sac_training_plots(rewards, losses)
    bad_delta_path = _make_bad_delta_plot(bad_delta)
    reward_summary.to_csv(ANALYSIS_DIR / f"{prefix}_sac_online_reward_summary.csv", index=False)
    loss_summary.to_csv(ANALYSIS_DIR / f"{prefix}_sac_loss_summary.csv", index=False)

    final_bad = (
        bad_delta[bad_delta["discomfort_%_bad10"] > 10.0].copy()
        if "discomfort_%_bad10" in bad_delta.columns
        else pd.DataFrame()
    )
    ppo_skipped = []
    for building in BUILDINGS:
        path = _raw_dir("rlmpc", building["result_id"], RLMPC_LABEL) / "ppo_history.csv"
        if path.exists():
            hist = _read(path)
            ppo_skipped.append(bool(hist.get("skipped", pd.Series([False])).fillna(False).astype(bool).all()))
    ppo_note = "all RLMPC PPO updates were skipped" if ppo_skipped and all(ppo_skipped) else "some RLMPC PPO rows were not skipped"

    mean_comfort = _short_controller_names(
        mean_kpis[["controller", "electricity_consumption_total", "discomfort_%", "cold_%", "hot_%", "kelvin_hours"]]
    ).rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
        }
    )
    mean_ops = _short_controller_names(
        mean_kpis[["controller", "daily_peak_average", "ramping_average", "feb_tin_max"]]
    ).rename(
        columns={
            "daily_peak_average": "daily_peak",
            "ramping_average": "ramping",
            "feb_tin_max": "Tin_max",
        }
    )
    district_shape = _short_controller_names(
        district_kpis[
            [
                "controller",
                "total_vs_citylearn",
                "peak_vs_citylearn",
                "daily_profile_peak_kwh",
                "daily_profile_peak_hour",
            ]
        ]
    ).rename(
        columns={
            "total_vs_citylearn": "total_vs_CL",
            "peak_vs_citylearn": "peak_vs_CL",
            "daily_profile_peak_kwh": "daily_peak_kWh",
            "daily_profile_peak_hour": "daily_peak_hr",
        }
    )
    district_tracking = _short_controller_names(
        district_kpis[["controller", "ramping_vs_citylearn", "target_rmse_kwh", "above_target_kwh"]]
    ).rename(
        columns={
            "ramping_vs_citylearn": "ramp_vs_CL",
            "target_rmse_kwh": "target_RMSE",
            "above_target_kwh": "above_target",
        }
    )
    term_mean = _pivot_terms(term_summary, "mean")
    term_max = _pivot_terms(term_summary, "max")
    if bad_delta.empty:
        bad_delta_display = pd.DataFrame()
        final_bad_display = pd.DataFrame()
    else:
        bad_delta_display = bad_delta[
            [
                "building",
                "discomfort_%_plus9",
                "discomfort_%_bad10",
                "delta_discomfort_%",
                "cold_%_bad10",
                "hot_%_bad10",
            ]
        ].rename(
            columns={
                "discomfort_%_plus9": "disc_plus9",
                "discomfort_%_bad10": "disc_bad10",
                "delta_discomfort_%": "delta_disc",
                "cold_%_bad10": "cold_bad10",
                "hot_%_bad10": "hot_bad10",
            }
        )
        final_bad_display = final_bad[
            ["building", "discomfort_%_bad10", "cold_%_bad10", "hot_%_bad10", "energy_bad10"]
        ].rename(
            columns={
                "discomfort_%_bad10": "disc_bad10",
                "cold_%_bad10": "cold_bad10",
                "hot_%_bad10": "hot_bad10",
                "energy_bad10": "energy",
            }
        )
    recovery_display = recovery_check[
        [
            "building",
            "rollout_max_abs_diff",
            "kpis_max_abs_diff",
            "district_kpis_max_abs_diff",
            "exact_zero_diff",
        ]
    ].rename(
        columns={
            "rollout_max_abs_diff": "rollout_diff",
            "kpis_max_abs_diff": "kpi_diff",
            "district_kpis_max_abs_diff": "district_diff",
            "exact_zero_diff": "exact",
        }
    )
    recovery_display["exact"] = recovery_display["exact"].map(lambda value: "yes" if bool(value) else "no")
    proxy_display = exact_proxy_check.head(8).copy().rename(
        columns={
            "rc2_power_per_unit_action": "rc2_pu",
            "rlmpc_reward_power_source": "rlmpc_src",
            "rlmpc_power_per_unit_action": "rlmpc_pu",
            "sac_reward_power_source": "sac_src",
            "sac_power_per_unit_action": "sac_pu",
            "rlmpc_coeff_abs_diff": "rlmpc_diff",
            "sac_coeff_abs_diff": "sac_diff",
        }
    )
    proxy_display = proxy_display[
        ["building", "rc2_pu", "rlmpc_src", "rlmpc_pu", "rlmpc_diff", "sac_src", "sac_pu", "sac_diff"]
    ]
    reward_display = pd.DataFrame()
    if not reward_summary.empty:
        reward_display = reward_summary[["episode", "raw_reward_mean", "reward_mean", "reset_fraction", "action_mean"]].copy()
        reward_display["reset_%"] = 100.0 * reward_display.pop("reset_fraction")
        reward_display = reward_display.rename(
            columns={
                "raw_reward_mean": "raw_mean",
                "reward_mean": "replay_mean",
                "action_mean": "action_mean",
            }
        )
        reward_display = reward_display[["episode", "raw_mean", "replay_mean", "reset_%", "action_mean"]]
    loss_core = pd.DataFrame()
    loss_alpha = pd.DataFrame()
    if not loss_summary.empty:
        loss_tail = loss_summary.tail(10).copy()
        core_cols = [
            "episode",
            "soft_q1_loss_mean",
            "soft_q1_loss_max",
            "soft_q2_loss_mean",
            "soft_q2_loss_max",
            "policy_loss_mean",
            "policy_loss_max",
        ]
        alpha_cols = ["episode", "alpha_loss_mean", "alpha_loss_max", "alpha_mean", "alpha_max"]
        loss_core = loss_tail[[column for column in core_cols if column in loss_tail.columns]].rename(
            columns={
                "soft_q1_loss_mean": "q1_mean",
                "soft_q1_loss_max": "q1_max",
                "soft_q2_loss_mean": "q2_mean",
                "soft_q2_loss_max": "q2_max",
                "policy_loss_mean": "pi_mean",
                "policy_loss_max": "pi_max",
            }
        )
        loss_alpha = loss_tail[[column for column in alpha_cols if column in loss_tail.columns]].rename(
            columns={
                "alpha_loss_mean": "alpha_loss_mean",
                "alpha_loss_max": "alpha_loss_max",
                "alpha_mean": "alpha_mean",
                "alpha_max": "alpha_max",
            }
        )

    overview = f"""{MARKER} - `{prefix}`

Fair comparison labels:

`{MPC_LABEL}`: recovered MPC with January-only 2R2C+Kalman RC fit and `baseline_district_daily_mean_exp2_norm` grid stress.

`{RLMPC_LABEL}`: RLMPC/GNU-RL using the same 2R2C+Kalman model, horizon 24, Kalman/w_est observer, future setpoint/bias profiles, baseline-daily grid stress, action-proxy heat, and matched MPC inner objective. The artifact says {ppo_note}; this preserves cost identity instead of learning a different inner cost.

`{SAC_BASE_LABEL}` plus `{SAC_BAD_LABEL}`: SAC safe-PI bootstrap, exact same action-proxy reward/gridstress, actor range 0-1. The base SAC is one safe-PI January bootstrap plus 9 SAC reward episodes. Buildings with February discomfort above 10% were then resumed for 10 more January SAC episodes; the final SAC row is a hybrid using resumed policies for those buildings and base policies for the others.

MPC recovery sanity check:

{_markdown_table(recovery_display, digits=6)}

Cost/reward:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

Exact controller cost/reward:

`J_k = 0.5*D_k + 0.5*GridStress*(p_RC2,b*u_k)/5`, `reward = -J_k`.

`p_RC2,b` is each building's January 2R2C+Kalman `power_per_unit_action`. `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and `GridStress = baseline_district_daily_mean_exp2_norm`. The common term diagnostics below use the same action proxy for all three controllers.

MPC and RLMPC also include the do-mpc-style soft nonlinear guard constraints `Tin >= Tsp-2` and `Tin <= Tsp+2` with solver penalty `100000`. These constraints are part of the optimizer; the SAC reward and the weighted-term diagnostic tables report the displayed `J_k` terms.

Exact action-proxy metadata check:

{_markdown_table(proxy_display, digits=4)}

Full wide tables are saved as `{prefix}_*.csv`; the notebook view below is split into compact tables so the KPI columns stay readable.

Mean building comfort KPIs:

{_markdown_table(mean_comfort, digits=3)}

Mean building operation KPIs:

{_markdown_table(mean_ops, digits=3)}

District aggregate KPIs - load shape:

{_markdown_table(district_shape, digits=3)}

District aggregate KPIs - tracking:

{_markdown_table(district_tracking, digits=3)}

Weighted term diagnostics - mean:

{_markdown_table(term_mean, digits=3)}

Weighted term diagnostics - max:

{_markdown_table(term_max, digits=3)}

SAC bad-building continuation deltas:

{_markdown_table(bad_delta_display, digits=2)}

SAC buildings still above 10% discomfort after bad +10:

{_markdown_table(final_bad_display, digits=2)}
"""
    if not reward_summary.empty:
        overview += f"""

SAC online reward summary, compact mean view:

{_markdown_table(reward_display, digits=3)}
"""
    if not loss_summary.empty:
        overview += f"""

SAC critic/policy loss summary, final 10 episodes:

{_markdown_table(loss_core, digits=3)}

SAC entropy/alpha loss summary, final 10 episodes:

{_markdown_table(loss_alpha, digits=3)}
"""

    image_cells = [
        _image(comfort_paths["mpc"], "MPC comfort all buildings"),
        _image(comfort_paths["rlmpc"], "RLMPC comfort all buildings"),
        _image(comfort_paths["sac"], "SAC final hybrid comfort all buildings"),
        _image(safe_pi_path, "SAC safe PI January episode 0 comfort"),
        _image(district_load_path, "District load time series"),
        _image(district_daily_path, "District mean hourly profile"),
        _image(terms_path, "Weighted stage cost terms"),
        _image(response_path, "Hourly time-bias response"),
        _image(bad_delta_path, "SAC bad-building continuation discomfort"),
        *[_image(path, path.stem.replace("_", " ")) for path in training_paths],
    ]
    cells = [nbf.v4.new_markdown_cell(overview), *image_cells]

    for notebook in [
        ANALYSIS_DIR / "02_mpc_results.ipynb",
        ANALYSIS_DIR / "04_rlmpc_results.ipynb",
        ANALYSIS_DIR / "08_sac_results.ipynb",
        ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
        ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
        ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
    ]:
        if notebook.exists():
            _append_once(notebook, cells)

    print("Done.")


if __name__ == "__main__":
    main()
