#!/usr/bin/env python
"""Run the February MPC rollout from notebooks/my_mpc.ipynb.

This script intentionally mirrors the working notebook cell:

- fitted one-state RC/COP model loaded from JSON,
- same do-mpc dynamics and objective,
- same online offset/disturbance correction `w_est`,
- same February window 744-1415,
- same optional occupant wrapper with thermostat overrides.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="no_occupant",
        choices=["no_occupant", "occupant_present", "occupant_present_peak_flattening"],
    )
    parser.add_argument("--rc-model", default="results/models/rc/1state_rc_247942_january.json")
    parser.add_argument("--output-dir", default="results/raw/mpc")
    parser.add_argument("--summary-dir", default="results/summaries/mpc")
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional short smoke-test length.")
    parser.add_argument(
        "--occupant",
        default="all",
        choices=["all", "occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"],
        help="Occupant to run for occupant_present. Default runs all four.",
    )
    parser.add_argument(
        "--occupant-mode",
        default="both",
        choices=["without_tdyn", "bspline_tdyn", "both"],
        help="Run baseline occupant MPC, B-spline gated adaptive T_dyn MPC, or both.",
    )
    return parser


DATASET_NAME = "annex96_ce1_vt_neighborhood"
CONTROL_BUILDING_NAME = "resstock-amy2018-2021-release-1-247942"
OCCUPANCY_BUILDING_NAME = "resstock-amy2018-2021-release-1-20199"
FEB_START = 744
FEB_END = 1415

FITTED_OCCUPANT_PARAMS = {
    "occ1_tolerant": {"T_pref": 21.861014, "b": 1.198849, "max_prob": 0.4},
    "occ2_sensitive": {"T_pref": 23.035896, "b": 3.451184, "max_prob": 0.4},
    "occ3_cold": {"T_pref": 19.957358, "b": 2.618482, "max_prob": 0.4},
    "occ4_hot": {"T_pref": 23.909844, "b": 2.570147, "max_prob": 0.4},
}


def set_experiment_seed(seed: int) -> int:
    import numpy as np

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def make_env(*, sim_start: int, sim_end: int, seed: int, reward_function=None):
    from citylearn.citylearn import CityLearnEnv

    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / "data" / "datasets" / DATASET_NAME
    schema_path = dataset_dir / "schema.json"

    kwargs = dict(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=True,
        buildings=[CONTROL_BUILDING_NAME],
        simulation_start_time_step=int(sim_start),
        simulation_end_time_step=int(sim_end),
        random_seed=int(seed),
        active_actions=["heating_device"],
    )
    if reward_function is not None:
        kwargs["reward_function"] = reward_function

    return CityLearnEnv(**kwargs)


class OccupantComfortReward:
    """Same notebook reward wrapper around CityLearn ComfortReward."""

    def __new__(cls, *args, **kwargs):
        from citylearn.reward_function import ComfortReward

        class _OccupantComfortReward(ComfortReward):
            def calculate(self, observations):
                reward_list = []
                for o in observations:
                    if "override_setpoint" in o:
                        o["indoor_dry_bulb_temperature_heating_set_point"] = o["override_setpoint"]
                    reward_list.append(super().calculate([o])[0])
                return [sum(reward_list)] if self.central_agent else reward_list

        return _OccupantComfortReward(*args, **kwargs)


class Occupant:
    def __init__(self, T_pref, b, max_prob=0.4, duration=3):
        self.T_pref = T_pref
        self.b = b
        self.max_prob = max_prob
        self.duration = duration
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def prob_increase(self, Tin):
        import numpy as np

        d = max(0, self.T_pref - Tin)
        return min(0.002 * np.exp(self.b * d), self.max_prob)

    def prob_decrease(self, Tin):
        import numpy as np

        d = max(0, Tin - self.T_pref)
        return min(0.002 * np.exp(self.b * d), self.max_prob)

    @staticmethod
    def override_delta(p_override):
        if p_override < 0.2:
            return 0.5
        if p_override < 0.4:
            return 1.0
        return 2.0

    def clear_override(self):
        self.override_timer = 0
        self.override_setpoint = None
        self.last_override_started = False

    def step(self, Tin, scheduled_Tsp, occupied=True):
        import numpy as np

        self.last_override_started = False
        if not occupied:
            self.clear_override()
            return scheduled_Tsp

        if self.override_timer > 0:
            self.override_timer -= 1
            return self.override_setpoint

        p_inc = self.prob_increase(Tin)
        p_dec = self.prob_decrease(Tin)
        r = np.random.rand()

        if r < p_inc:
            new_sp = scheduled_Tsp + self.override_delta(p_inc)
        elif r < p_inc + p_dec:
            new_sp = scheduled_Tsp - self.override_delta(p_dec)
        else:
            self.override_setpoint = None
            return scheduled_Tsp

        self.override_setpoint = new_sp
        self.override_timer = max(self.duration - 1, 0)
        self.last_override_started = True
        return new_sp


class OccupantWrapper:
    def __new__(cls, *args, **kwargs):
        import gymnasium as gym

        class _OccupantWrapper(gym.Wrapper):
            def __init__(self, env, occupant):
                super().__init__(env)
                self.occupant = occupant

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                self.occupant.override_timer = 0
                self.occupant.override_setpoint = None
                self.occupant.last_override_started = False
                self.override_count = 0
                self.last_feedback = 0
                b = self.env.unwrapped.buildings[0]
                self.effective_setpoints = [
                    float(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[b.time_step])
                ]
                self.baseline_setpoints = [
                    float(
                        b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            b.time_step
                        ]
                    )
                ]
                return obs, info

            def step(self, action):
                b = self.env.unwrapped.buildings[0]
                current_ix = b.time_step
                Tin = float(b.energy_simulation.indoor_dry_bulb_temperature[current_ix])
                next_ix = min(
                    current_ix + 1,
                    len(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point) - 1,
                )
                occupied_next = float(b.energy_simulation.occupant_count[next_ix]) > 0.0
                scheduled_next_Tsp = float(
                    b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[next_ix]
                )

                if occupied_next:
                    new_Tsp = self.occupant.step(Tin, scheduled_next_Tsp, occupied=True)
                else:
                    new_Tsp = self.occupant.step(Tin, scheduled_next_Tsp, occupied=False)

                feedback = 0
                if self.occupant.last_override_started:
                    self.override_count += 1
                    if new_Tsp > scheduled_next_Tsp:
                        feedback = +1
                    elif new_Tsp < scheduled_next_Tsp:
                        feedback = -1

                self.last_feedback = feedback

                b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[next_ix] = new_Tsp
                obs, reward, terminated, truncated, info = self.env.step(action)

                self.effective_setpoints.append(
                    float(b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point[b.time_step])
                )
                self.baseline_setpoints.append(
                    float(
                        b.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control[
                            b.time_step
                        ]
                    )
                )
                return obs, reward, terminated, truncated, info

        return _OccupantWrapper(*args, **kwargs)


def observation_index_map(env) -> dict[str, int]:
    names = env.unwrapped.buildings[0].active_observations

    def idx(name):
        return names.index(name)

    return {
        "tin": idx("indoor_dry_bulb_temperature"),
        "tout": idx("outdoor_dry_bulb_temperature"),
        "solar": idx("direct_solar_irradiance"),
        "tout_p": [idx(f"outdoor_dry_bulb_temperature_predicted_{h}") for h in [1, 2, 3]],
        "solar_p": [idx(f"direct_solar_irradiance_predicted_{h}") for h in [1, 2, 3]],
        "tmin": idx("indoor_dry_bulb_temperature_heating_set_point"),
        "tmax": idx("indoor_dry_bulb_temperature_cooling_set_point"),
    }


def fitted_occupants() -> dict[str, Occupant]:
    return {name: Occupant(**params) for name, params in FITTED_OCCUPANT_PARAMS.items()}


def dataset_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    dataset_dir = repo / "data" / "datasets" / DATASET_NAME
    return dataset_dir / "schema.json", dataset_dir


def load_building_occupancy_series(building_name: str = OCCUPANCY_BUILDING_NAME):
    import pandas as pd

    schema_path, dataset_dir = dataset_paths()
    schema = json.loads(schema_path.read_text())
    csv_path = dataset_dir / schema["buildings"][building_name]["energy_simulation"]
    return pd.read_csv(csv_path)["occupant_count"].astype(int).to_numpy()


def fit_bspline_occupancy_model(*, seed: int, building_name: str = OCCUPANCY_BUILDING_NAME):
    from src.occupants.bspline_preferences import FixedBSplineIMCOccupancyPredictor

    occupancy_series = load_building_occupancy_series(building_name)
    occupancy_model = FixedBSplineIMCOccupancyPredictor(
        period=24,
        spline_degree=3,
        n_internal_knots=4,
        l2_phi=1e-3,
        min_samples_per_state=25,
        maxiter=1000,
        random_state=int(seed),
    )
    occupancy_model.fit(occupancy_series)
    return occupancy_model, occupancy_series


def forecast_occupancy_from_bsplines(occupancy_model, current_occupancy: int, time_step: int, horizon: int, alpha=0.5):
    import numpy as np

    occ_hat_h, _ = occupancy_model.predict_expected_occupancy(
        current_state=int(current_occupancy),
        start_t=int(time_step),
        horizon=int(horizon),
    )
    occ_bin_h = (occ_hat_h > float(alpha)).astype(float)
    return np.asarray(occ_hat_h, dtype=float), np.asarray(occ_bin_h, dtype=float)


def get_citylearn_kpi(district_kpis, name: str):
    rows = district_kpis[district_kpis["cost_function"] == name]["value"]
    return float(rows.iloc[0]) if len(rows) > 0 else None


def rollout_history_dataframe(env, result):
    import numpy as np
    import pandas as pd

    b = env.unwrapped.buildings[0]
    n = len(result["Tin_hist"])
    history = pd.DataFrame(
        {
            "time_step": np.arange(n, dtype=int),
            "indoor_temperature": result["Tin_hist"],
            "tmin": result["Tmin_hist"],
            "tmax": result["Tmax_hist"],
            "action": result["u_hist"],
            "price": result["price_hist"],
        }
    )
    if hasattr(env, "effective_setpoints"):
        history["effective_setpoint"] = np.asarray(env.effective_setpoints[:n], dtype=float)
    if hasattr(env, "baseline_setpoints"):
        history["baseline_setpoint"] = np.asarray(env.baseline_setpoints[:n], dtype=float)
    if hasattr(b, "net_electricity_consumption"):
        history["net_electricity_consumption"] = np.asarray(b.net_electricity_consumption[:n], dtype=float)
    optional_columns = {
        "tdyn": "T_dyn_hist",
        "occupant_feedback": "feedback_hist",
        "occ_now": "occ_now_hist",
        "occ_hat_1step": "occ_hat_hist",
        "occ_bin_1step": "occ_bin_hist",
    }
    for column, key in optional_columns.items():
        if key in result:
            history[column] = np.asarray(result[key][:n], dtype=float)
    return history


def run_february_mpc(
    *,
    controller_name,
    rc_params,
    objective_mode="baseline",
    fixed_band=None,
    w_energy=1.0,
    w_comfort=1.0,
    lambda_u=1e-3,
    lambda_du=0.0,
    use_smooth_du=False,
    cold_penalty=1e5,
    hot_penalty=1e3,
    occupant=None,
    reward_function=None,
    seed=49,
    max_steps=None,
):
    import casadi as ca
    import do_mpc
    import numpy as np

    set_experiment_seed(seed)

    A = rc_params["A"]
    Bd0 = rc_params["Bd0"]
    Bd1 = rc_params["Bd1"]
    Bd2 = rc_params["Bd2"]
    Bu = rc_params["Bu"]
    Tm = rc_params["mean_indoor_temperature"]
    cop_a = rc_params["cop_a"]
    cop_b = rc_params["cop_b"]
    power_per_unit_u = rc_params["power_per_unit_action"]

    model = do_mpc.model.Model("discrete")
    Tin_m = model.set_variable("_x", "Tin")
    if use_smooth_du:
        u_prev = model.set_variable("_x", "u_prev")
    else:
        u_prev = None

    u_m = model.set_variable("_u", "u")
    Tout_m = model.set_variable("_tvp", "Tout")
    Sol_m = model.set_variable("_tvp", "Sol")
    Price_m = model.set_variable("_tvp", "Price")
    Tsp_m = model.set_variable("_tvp", "Tsp")
    Tmin_m = model.set_variable("_tvp", "Tmin")
    Tmax_m = model.set_variable("_tvp", "Tmax")
    w_est = model.set_variable("_tvp", "w_est")

    COP_expr = cop_a + cop_b * Tout_m
    COP_expr = ca.fmin(ca.fmax(COP_expr, 0.5), 6.0)
    Tin_next_m = A * Tin_m + Bd0 * Tm + Bd1 * Tout_m + Bd2 * Sol_m + Bu * COP_expr * u_m + w_est

    model.set_expression("Tin_occ", Tin_next_m)
    model.set_rhs("Tin", Tin_next_m)
    if use_smooth_du:
        model.set_rhs("u_prev", u_m)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.supress_ipopt_output()
    setup_mpc = {
        "n_horizon": 24,
        "t_step": 1.0,
        "state_discretization": "discrete",
        "store_full_solution": True,
    }
    mpc.set_param(**setup_mpc)

    if objective_mode == "baseline":
        Tref = Tmin_m
        stage_cost = (Tin_m - Tref) ** 2 + lambda_u * u_m**2
        terminal_cost = (Tin_m - Tref) ** 2
    elif objective_mode == "energy_comfort":
        energy_cost = Price_m * power_per_unit_u * u_m
        comfort_cost = (Tin_m - Tsp_m) ** 2
        smooth_cost = lambda_u * u_m**2
        if use_smooth_du:
            smooth_cost = smooth_cost + lambda_du * (u_m - u_prev) ** 2
        stage_cost = w_energy * energy_cost + w_comfort * comfort_cost + smooth_cost
        terminal_cost = w_comfort * comfort_cost
    else:
        raise ValueError(f"Unknown objective_mode: {objective_mode}")

    if objective_mode == "baseline" and use_smooth_du:
        stage_cost = stage_cost + lambda_du * (u_m - u_prev) ** 2

    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    mpc.bounds["lower", "_u", "u"] = 0.0
    mpc.bounds["upper", "_u", "u"] = 1.0
    mpc.set_nl_cons("too_cold", Tmin_m - Tin_m, ub=0.0, soft_constraint=True, penalty_term_cons=cold_penalty)
    mpc.set_nl_cons("too_hot", Tin_m - Tmax_m, ub=0.0, soft_constraint=True, penalty_term_cons=hot_penalty)

    N = setup_mpc["n_horizon"]
    tvp_template = mpc.get_tvp_template()
    latest = {
        "Tout0": 0.0,
        "Sol0": 0.0,
        "Price0": 0.0,
        "Tsp0": 20.0,
        "Toutp": [0.0, 0.0, 0.0],
        "Solp": [0.0, 0.0, 0.0],
        "Pricep": [0.0, 0.0, 0.0],
        "Tspp": [20.0, 20.0, 20.0],
        "Tmin0": 18.0,
        "Tmax0": 24.0,
        "w_est": 0.0,
    }

    def tvp_fun(_t_now):
        for k in range(N + 1):
            if k == 0:
                Tout_f, Sol_f, Price_f, Tsp_f = (
                    latest["Tout0"],
                    latest["Sol0"],
                    latest["Price0"],
                    latest["Tsp0"],
                )
            elif k <= 3:
                Tout_f, Sol_f, Price_f, Tsp_f = (
                    latest["Toutp"][k - 1],
                    latest["Solp"][k - 1],
                    latest["Pricep"][k - 1],
                    latest["Tspp"][k - 1],
                )
            else:
                Tout_f, Sol_f, Price_f, Tsp_f = (
                    latest["Toutp"][-1],
                    latest["Solp"][-1],
                    latest["Pricep"][-1],
                    latest["Tspp"][-1],
                )

            tvp_template["_tvp", k, "Tout"] = float(Tout_f)
            tvp_template["_tvp", k, "Sol"] = float(Sol_f)
            tvp_template["_tvp", k, "Price"] = float(Price_f)
            tvp_template["_tvp", k, "Tsp"] = float(Tsp_f)
            tvp_template["_tvp", k, "Tmin"] = float(latest["Tmin0"])
            tvp_template["_tvp", k, "Tmax"] = float(latest["Tmax0"])
            tvp_template["_tvp", k, "w_est"] = float(latest["w_est"])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    env = make_env(sim_start=744, sim_end=1415, seed=seed, reward_function=reward_function)
    if occupant is not None:
        env = OccupantWrapper(env, occupant)

    obs, _ = env.reset(seed=seed)
    building = env.unwrapped.buildings[0]
    obs_idx = observation_index_map(env)
    mpc.reset_history()

    Tin_hist, Tmin_hist, Tmax_hist = [], [], []
    u_hist, price_hist, reward_hist = [], [], []
    w_est_val = 0.0
    gamma_w = 0.2
    prev_Tin = None
    prev_u = None
    prev_Tout = None
    prev_Sol = None
    last_u = 0.0
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        Tin0 = float(o[obs_idx["tin"]])
        Tout0 = float(o[obs_idx["tout"]])
        Sol0 = float(o[obs_idx["solar"]])
        time_step = building.time_step
        Price0 = float(building.pricing.electricity_pricing[time_step])
        scheduled_setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
        Tsp0 = float(scheduled_setpoints[time_step])
        Toutp = [float(o[j]) for j in obs_idx["tout_p"]]
        Solp = [float(o[j]) for j in obs_idx["solar_p"]]
        Pricep = [
            float(building.pricing.electricity_pricing_predicted_1[time_step]),
            float(building.pricing.electricity_pricing_predicted_2[time_step]),
            float(building.pricing.electricity_pricing_predicted_3[time_step]),
        ]
        Tspp = [float(scheduled_setpoints[min(time_step + h, len(scheduled_setpoints) - 1)]) for h in [1, 2, 3]]

        if fixed_band is None:
            Tmin0 = float(o[obs_idx["tmin"]])
            Tmax0 = float(o[obs_idx["tmax"]])
        else:
            Tmin0, Tmax0 = map(float, fixed_band)

        if prev_Tin is not None:
            cop_prev = np.clip(cop_a + cop_b * prev_Tout, 0.5, 6.0)
            Tin_pred = A * prev_Tin + Bd0 * Tm + Bd1 * prev_Tout + Bd2 * prev_Sol + Bu * cop_prev * prev_u + w_est_val
            err = Tin0 - Tin_pred
            w_est_val = w_est_val + gamma_w * err

        latest.update(
            {
                "Tout0": Tout0,
                "Sol0": Sol0,
                "Price0": Price0,
                "Tsp0": Tsp0,
                "Toutp": Toutp,
                "Solp": Solp,
                "Pricep": Pricep,
                "Tspp": Tspp,
                "Tmin0": Tmin0,
                "Tmax0": Tmax0,
                "w_est": w_est_val,
            }
        )

        x0 = np.array([[Tin0], [last_u]]) if use_smooth_du else np.array([[Tin0]])
        u0 = float(np.asarray(mpc.make_step(x0)).reshape(-1)[0])
        obs, reward, terminated, truncated, _ = env.step([[u0]])

        prev_Tin = Tin0
        prev_u = u0
        prev_Tout = Tout0
        prev_Sol = Sol0
        last_u = u0
        Tin_hist.append(Tin0)
        Tmin_hist.append(Tmin0)
        Tmax_hist.append(Tmax0)
        u_hist.append(u0)
        price_hist.append(Price0)
        reward_hist.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))

        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    if max_steps is None:
        kpis = env.unwrapped.evaluate().copy()
        district_kpis = kpis[kpis["level"] == "district"].copy()
    else:
        import pandas as pd

        kpis = pd.DataFrame(columns=["cost_function", "value", "level", "name"])
        district_kpis = kpis.copy()
    result = {
        "name": controller_name,
        "env": env,
        "kpis": kpis,
        "district_kpis": district_kpis,
        "Tin_hist": np.asarray(Tin_hist, dtype=float),
        "Tmin_hist": np.asarray(Tmin_hist, dtype=float),
        "Tmax_hist": np.asarray(Tmax_hist, dtype=float),
        "u_hist": np.asarray(u_hist, dtype=float),
        "price_hist": np.asarray(price_hist, dtype=float),
        "reward_hist": np.asarray(reward_hist, dtype=float),
        "override_count": int(getattr(env, "override_count", 0)),
    }

    print(f"{controller_name} run finished")
    print(f"Overrides: {result['override_count']}")
    print(district_kpis[["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return result


def run_february_bspline_tdyn_mpc(
    *,
    controller_name,
    rc_params,
    occupant,
    fixed_band=None,
    alpha_occ=0.5,
    lambda_u=1e-3,
    lambda_du=0.0,
    use_smooth_du=False,
    cold_penalty=1e5,
    hot_penalty=1e3,
    reward_function=None,
    seed=49,
    max_steps=None,
    T_dyn_init_mode="schedule",
    delta_up=0.5,
    delta_down=0.5,
    drift_to_pref=0.0,
    T_dyn_min=18.0,
    T_dyn_max=26.0,
):
    """Run B-spline gated paper-style adaptive T_dyn MPC.

    This keeps the same fitted RC dynamics and online offset correction as the
    notebook MPC. B-spline occupancy gates the comfort term, while the comfort
    reference is persistent T_dyn updated only when new occupant feedback starts.
    """

    import casadi as ca
    import do_mpc
    import numpy as np

    set_experiment_seed(seed)

    A = rc_params["A"]
    Bd0 = rc_params["Bd0"]
    Bd1 = rc_params["Bd1"]
    Bd2 = rc_params["Bd2"]
    Bu = rc_params["Bu"]
    Tm = rc_params["mean_indoor_temperature"]
    cop_a = rc_params["cop_a"]
    cop_b = rc_params["cop_b"]

    occupancy_model, occupancy_series = fit_bspline_occupancy_model(seed=seed)
    occupant_preference = float(occupant.T_pref)

    model = do_mpc.model.Model("discrete")
    Tin_m = model.set_variable("_x", "Tin")
    if use_smooth_du:
        u_prev = model.set_variable("_x", "u_prev")
    else:
        u_prev = None

    u_m = model.set_variable("_u", "u")
    Tout_m = model.set_variable("_tvp", "Tout")
    Sol_m = model.set_variable("_tvp", "Sol")
    Tdyn_m = model.set_variable("_tvp", "Tdyn")
    Tmin_m = model.set_variable("_tvp", "Tmin")
    Tmax_m = model.set_variable("_tvp", "Tmax")
    occ_bin_m = model.set_variable("_tvp", "occ_bin")
    occ_hat_m = model.set_variable("_tvp", "occ_hat")
    w_est = model.set_variable("_tvp", "w_est")

    COP_expr = cop_a + cop_b * Tout_m
    COP_expr = ca.fmin(ca.fmax(COP_expr, 0.5), 6.0)
    Tin_next_m = A * Tin_m + Bd0 * Tm + Bd1 * Tout_m + Bd2 * Sol_m + Bu * COP_expr * u_m + w_est

    model.set_expression("Tin_occ", Tin_next_m)
    model.set_rhs("Tin", Tin_next_m)
    if use_smooth_du:
        model.set_rhs("u_prev", u_m)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.settings.supress_ipopt_output()
    setup_mpc = {
        "n_horizon": 24,
        "t_step": 1.0,
        "state_discretization": "discrete",
        "store_full_solution": True,
    }
    mpc.set_param(**setup_mpc)

    stage_cost = occ_bin_m * (Tin_m - Tdyn_m) ** 2 + lambda_u * u_m**2
    terminal_cost = occ_bin_m * (Tin_m - Tdyn_m) ** 2
    if use_smooth_du:
        stage_cost = stage_cost + lambda_du * (u_m - u_prev) ** 2

    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    mpc.bounds["lower", "_u", "u"] = 0.0
    mpc.bounds["upper", "_u", "u"] = 1.0
    mpc.set_nl_cons("too_cold", Tmin_m - Tin_m, ub=0.0, soft_constraint=True, penalty_term_cons=cold_penalty)
    mpc.set_nl_cons("too_hot", Tin_m - Tmax_m, ub=0.0, soft_constraint=True, penalty_term_cons=hot_penalty)

    N = setup_mpc["n_horizon"]
    tvp_template = mpc.get_tvp_template()
    latest = {
        "Tout0": 0.0,
        "Sol0": 0.0,
        "Tdyn": 21.0,
        "Tmin0": 18.0,
        "Tmax0": 26.0,
        "occ_hat": np.zeros(N + 1, dtype=float),
        "occ_bin": np.zeros(N + 1, dtype=float),
        "Toutp": [0.0, 0.0, 0.0],
        "Solp": [0.0, 0.0, 0.0],
        "w_est": 0.0,
    }

    def tvp_fun(_t_now):
        for k in range(N + 1):
            if k == 0:
                Tout_f, Sol_f = latest["Tout0"], latest["Sol0"]
            elif k <= 3:
                Tout_f, Sol_f = latest["Toutp"][k - 1], latest["Solp"][k - 1]
            else:
                Tout_f, Sol_f = latest["Toutp"][-1], latest["Solp"][-1]

            tvp_template["_tvp", k, "Tout"] = float(Tout_f)
            tvp_template["_tvp", k, "Sol"] = float(Sol_f)
            tvp_template["_tvp", k, "Tdyn"] = float(latest["Tdyn"])
            tvp_template["_tvp", k, "Tmin"] = float(latest["Tmin0"])
            tvp_template["_tvp", k, "Tmax"] = float(latest["Tmax0"])
            tvp_template["_tvp", k, "occ_hat"] = float(latest["occ_hat"][k])
            tvp_template["_tvp", k, "occ_bin"] = float(latest["occ_bin"][k])
            tvp_template["_tvp", k, "w_est"] = float(latest["w_est"])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)
    mpc.setup()

    env = make_env(sim_start=FEB_START, sim_end=FEB_END, seed=seed, reward_function=reward_function)
    env = OccupantWrapper(env, occupant)

    obs, _ = env.reset(seed=seed)
    building = env.unwrapped.buildings[0]
    obs_idx = observation_index_map(env)
    mpc.reset_history()

    scheduled_setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
    scheduled0 = float(scheduled_setpoints[building.time_step])
    if T_dyn_init_mode == "schedule":
        T_dyn = scheduled0
    elif T_dyn_init_mode == "pref":
        T_dyn = occupant_preference
    else:
        raise ValueError("T_dyn_init_mode must be 'schedule' or 'pref'.")
    T_dyn = float(np.clip(T_dyn, T_dyn_min, T_dyn_max))

    Tin_hist, Tmin_hist, Tmax_hist = [], [], []
    u_hist, price_hist, reward_hist = [], [], []
    T_dyn_hist, feedback_hist = [], []
    occ_now_hist, occ_hat_hist, occ_bin_hist = [], [], []
    w_est_val = 0.0
    gamma_w = 0.2
    prev_Tin = None
    prev_u = None
    prev_Tout = None
    prev_Sol = None
    last_u = 0.0
    terminated = False
    truncated = False
    step_count = 0

    while not (terminated or truncated):
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        Tin0 = float(o[obs_idx["tin"]])
        Tout0 = float(o[obs_idx["tout"]])
        Sol0 = float(o[obs_idx["solar"]])
        time_step = int(building.time_step)
        Price0 = float(building.pricing.electricity_pricing[time_step])
        Toutp = [float(o[j]) for j in obs_idx["tout_p"]]
        Solp = [float(o[j]) for j in obs_idx["solar_p"]]

        if fixed_band is None:
            Tmin0, Tmax0 = 18.0, 26.0
        else:
            Tmin0, Tmax0 = map(float, fixed_band)

        if prev_Tin is not None:
            cop_prev = np.clip(cop_a + cop_b * prev_Tout, 0.5, 6.0)
            Tin_pred = A * prev_Tin + Bd0 * Tm + Bd1 * prev_Tout + Bd2 * prev_Sol + Bu * cop_prev * prev_u + w_est_val
            err = Tin0 - Tin_pred
            w_est_val = w_est_val + gamma_w * err

        if drift_to_pref > 0.0:
            T_dyn = (1.0 - drift_to_pref) * T_dyn + drift_to_pref * occupant_preference

        occ_now = int(occupancy_series[min(time_step, len(occupancy_series) - 1)])
        occ_hat_h, occ_bin_h = forecast_occupancy_from_bsplines(
            occupancy_model=occupancy_model,
            current_occupancy=occ_now,
            time_step=time_step,
            horizon=N,
            alpha=alpha_occ,
        )
        occ_hat_full = np.zeros(N + 1, dtype=float)
        occ_bin_full = np.zeros(N + 1, dtype=float)
        occ_hat_full[0] = float(occ_now)
        occ_bin_full[0] = float(occ_now > alpha_occ)
        occ_hat_full[1:] = occ_hat_h
        occ_bin_full[1:] = occ_bin_h

        latest.update(
            {
                "Tout0": Tout0,
                "Sol0": Sol0,
                "Tdyn": T_dyn,
                "Tmin0": Tmin0,
                "Tmax0": Tmax0,
                "occ_hat": occ_hat_full,
                "occ_bin": occ_bin_full,
                "Toutp": Toutp,
                "Solp": Solp,
                "w_est": w_est_val,
            }
        )

        x0 = np.array([[Tin0], [last_u]]) if use_smooth_du else np.array([[Tin0]])
        u0 = float(np.asarray(mpc.make_step(x0)).reshape(-1)[0])
        obs, reward, terminated, truncated, _ = env.step([[u0]])

        feedback = int(getattr(env, "last_feedback", 0))
        if feedback > 0:
            T_dyn += float(delta_up)
        elif feedback < 0:
            T_dyn -= float(delta_down)
        T_dyn = float(np.clip(T_dyn, T_dyn_min, T_dyn_max))

        prev_Tin = Tin0
        prev_u = u0
        prev_Tout = Tout0
        prev_Sol = Sol0
        last_u = u0
        Tin_hist.append(Tin0)
        Tmin_hist.append(Tmin0)
        Tmax_hist.append(Tmax0)
        u_hist.append(u0)
        price_hist.append(Price0)
        reward_hist.append(float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward))
        T_dyn_hist.append(float(T_dyn))
        feedback_hist.append(float(feedback))
        occ_now_hist.append(float(occ_now))
        occ_hat_hist.append(float(occ_hat_h[0]) if len(occ_hat_h) else 0.0)
        occ_bin_hist.append(float(occ_bin_h[0]) if len(occ_bin_h) else 0.0)

        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    if max_steps is None:
        kpis = env.unwrapped.evaluate().copy()
        district_kpis = kpis[kpis["level"] == "district"].copy()
    else:
        import pandas as pd

        kpis = pd.DataFrame(columns=["cost_function", "value", "level", "name"])
        district_kpis = kpis.copy()

    result = {
        "name": controller_name,
        "target_temperature": occupant_preference,
        "T_dyn_init_mode": T_dyn_init_mode,
        "env": env,
        "kpis": kpis,
        "district_kpis": district_kpis,
        "Tin_hist": np.asarray(Tin_hist, dtype=float),
        "Tmin_hist": np.asarray(Tmin_hist, dtype=float),
        "Tmax_hist": np.asarray(Tmax_hist, dtype=float),
        "u_hist": np.asarray(u_hist, dtype=float),
        "price_hist": np.asarray(price_hist, dtype=float),
        "reward_hist": np.asarray(reward_hist, dtype=float),
        "T_dyn_hist": np.asarray(T_dyn_hist, dtype=float),
        "feedback_hist": np.asarray(feedback_hist, dtype=float),
        "occ_now_hist": np.asarray(occ_now_hist, dtype=float),
        "occ_hat_hist": np.asarray(occ_hat_hist, dtype=float),
        "occ_bin_hist": np.asarray(occ_bin_hist, dtype=float),
        "predicted_occupied_fraction": float(np.mean(occ_bin_hist)) if len(occ_bin_hist) else 0.0,
        "override_count": int(getattr(env, "override_count", 0)),
    }

    print(f"{controller_name} run finished")
    print(f"Occupant T_pref: {occupant_preference:.3f} C")
    if len(result["T_dyn_hist"]):
        print(f"T_dyn mean/final: {result['T_dyn_hist'].mean():.3f} / {result['T_dyn_hist'][-1]:.3f} C")
    print(f"Predicted occupied fraction: {result['predicted_occupied_fraction']:.3f}")
    print(f"Overrides: {result['override_count']}")
    print(district_kpis[["cost_function", "value"]].reset_index(drop=True).to_string(index=False))
    return result


def summarize_run(label, result):
    district = result["district_kpis"]
    summary = {
        "case": label,
        "override_count": int(result.get("override_count", 0)),
        "reward_total": float(result["reward_hist"].sum()) if len(result["reward_hist"]) else 0.0,
        "electricity_consumption_total": get_citylearn_kpi(district, "electricity_consumption_total"),
        "discomfort_proportion": get_citylearn_kpi(district, "discomfort_proportion"),
        "discomfort_cold_proportion": get_citylearn_kpi(district, "discomfort_cold_proportion"),
        "discomfort_hot_proportion": get_citylearn_kpi(district, "discomfort_hot_proportion"),
        "daily_peak_average": get_citylearn_kpi(district, "daily_peak_average"),
    }
    if "target_temperature" in result:
        summary["target_temperature"] = float(result["target_temperature"])
    if "predicted_occupied_fraction" in result:
        summary["predicted_occupied_fraction"] = float(result["predicted_occupied_fraction"])
    if "T_dyn_hist" in result and len(result["T_dyn_hist"]):
        summary["tdyn_mean"] = float(result["T_dyn_hist"].mean())
        summary["tdyn_final"] = float(result["T_dyn_hist"][-1])
    return summary


def save_result(result, *, output_dir: Path, summary_dir: Path, scenario: str, label: str):
    output_dir = output_dir / scenario / label
    summary_dir = summary_dir / scenario
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    history = rollout_history_dataframe(result["env"], result)
    history.to_csv(output_dir / "rollout.csv", index=False)
    result["kpis"].to_csv(output_dir / "kpis.csv", index=False)
    result["district_kpis"].to_csv(output_dir / "district_kpis.csv", index=False)
    metrics = summarize_run(label, result)
    (summary_dir / f"{label}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    import pandas as pd

    args = build_parser().parse_args()
    if args.scenario == "occupant_present_peak_flattening":
        raise SystemExit("Peak flattening is not implemented yet in the notebook MPC.")

    rc_params = json.loads(Path(args.rc_model).read_text())
    output_dir = Path(args.output_dir)
    summary_dir = Path(args.summary_dir)
    summary_rows = []

    if args.scenario == "no_occupant":
        result = run_february_mpc(
            controller_name="Current MPC (baseline)",
            rc_params=rc_params,
            objective_mode="baseline",
            fixed_band=None,
            lambda_u=1e-3,
            lambda_du=0.0,
            use_smooth_du=False,
            cold_penalty=1e5,
            hot_penalty=1e3,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        summary_rows.append(save_result(result, output_dir=output_dir, summary_dir=summary_dir, scenario=args.scenario, label="baseline"))

    elif args.scenario == "occupant_present":
        occupants = fitted_occupants()
        if args.occupant != "all":
            occupants = {args.occupant: occupants[args.occupant]}

        for occ_idx, (name, occ) in enumerate(occupants.items(), start=1):
            if args.occupant_mode in ("without_tdyn", "both"):
                label = f"{name}_without_tdyn"
                print(f"\n{label}")
                result = run_february_mpc(
                    controller_name=f"Current MPC + {name} without T_dyn",
                    rc_params=rc_params,
                    objective_mode="baseline",
                    fixed_band=None,
                    lambda_u=1e-3,
                    lambda_du=0.0,
                    use_smooth_du=False,
                    cold_penalty=1e5,
                    hot_penalty=1e3,
                    occupant=deepcopy(occ),
                    reward_function=OccupantComfortReward,
                    seed=int(args.seed + occ_idx),
                    max_steps=args.max_steps,
                )
                summary_rows.append(
                    save_result(result, output_dir=output_dir, summary_dir=summary_dir, scenario=args.scenario, label=label)
                )

            if args.occupant_mode in ("bspline_tdyn", "both"):
                label = f"{name}_bspline_tdyn"
                print(f"\n{label}")
                result = run_february_bspline_tdyn_mpc(
                    controller_name=f"B-spline gated adaptive T_dyn MPC + {name}",
                    rc_params=rc_params,
                    occupant=deepcopy(occ),
                    fixed_band=None,
                    alpha_occ=0.5,
                    lambda_u=1e-3,
                    lambda_du=0.0,
                    use_smooth_du=False,
                    cold_penalty=1e5,
                    hot_penalty=1e3,
                    reward_function=OccupantComfortReward,
                    seed=int(args.seed + occ_idx),
                    max_steps=args.max_steps,
                    T_dyn_init_mode="schedule",
                    delta_up=0.5,
                    delta_down=0.5,
                    drift_to_pref=0.0,
                    T_dyn_min=18.0,
                    T_dyn_max=26.0,
                )
                summary_rows.append(
                    save_result(result, output_dir=output_dir, summary_dir=summary_dir, scenario=args.scenario, label=label)
                )

    summary = pd.DataFrame(summary_rows)
    summary_path = summary_dir / args.scenario / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
