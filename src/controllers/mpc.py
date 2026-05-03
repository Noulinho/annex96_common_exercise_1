"""Occupant-adaptive RC-MPC controller definitions."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in incomplete envs.
    np = None


def _require_numpy():
    if np is None:
        raise ImportError("OccupantAdaptiveRCMPC requires numpy. Install the project requirements first.")
    return np


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 24
    energy_weight: float = 1.0
    comfort_weight: float = 1.0
    peak_weight: float = 0.0
    action_lower_bound: float = 0.0
    action_upper_bound: float = 1.0
    lambda_u: float = 1e-3
    lambda_du: float = 0.0
    use_smooth_du: bool = False
    cold_penalty: float = 1e5
    hot_penalty: float = 1e3
    objective_mode: str = "energy_comfort"
    fixed_band: tuple[float, float] | None = None
    disturbance_gain: float = 0.2


@dataclass(frozen=True)
class MPCObservationIndices:
    indoor_temperature: int
    outdoor_temperature: int
    direct_solar_irradiance: int
    outdoor_temperature_forecast: tuple[int, int, int]
    direct_solar_irradiance_forecast: tuple[int, int, int]
    heating_setpoint: int
    cooling_setpoint: int

    @classmethod
    def from_env(cls, env) -> "MPCObservationIndices":
        names = env.unwrapped.buildings[0].active_observations

        def idx(name: str) -> int:
            return names.index(name)

        return cls(
            indoor_temperature=idx("indoor_dry_bulb_temperature"),
            outdoor_temperature=idx("outdoor_dry_bulb_temperature"),
            direct_solar_irradiance=idx("direct_solar_irradiance"),
            outdoor_temperature_forecast=tuple(
                idx(f"outdoor_dry_bulb_temperature_predicted_{h}") for h in (1, 2, 3)
            ),
            direct_solar_irradiance_forecast=tuple(
                idx(f"direct_solar_irradiance_predicted_{h}") for h in (1, 2, 3)
            ),
            heating_setpoint=idx("indoor_dry_bulb_temperature_heating_set_point"),
            cooling_setpoint=idx("indoor_dry_bulb_temperature_cooling_set_point"),
        )


class OccupantAdaptiveRCMPC:
    """do-mpc implementation of the notebook's learned RC-MPC controller.

    The notebook version relied on globals such as `A`, `Bd0`, `i_tin`, and
    `power_per_unit_u`. Here those values are read from the fitted RC model, the
    environment, and an explicit observation-index object.
    """

    def __init__(
        self,
        rc_model,
        env=None,
        config: MPCConfig | None = None,
        preference_model=None,
        observation_indices: MPCObservationIndices | None = None,
    ):
        self.rc_model = rc_model
        self.env = env
        self.config = config or MPCConfig()
        self.preference_model = preference_model
        self.observation_indices = observation_indices or (
            MPCObservationIndices.from_env(env) if env is not None else None
        )

        self._built = False
        self._model = None
        self.mpc = None
        self._latest: dict[str, Any] = {}
        if self.env is not None and self.observation_indices is not None:
            self._build_mpc()
            self.reset()

    @property
    def params(self):
        return self.rc_model.params

    def bind_env(self, env, observation_indices: MPCObservationIndices | None = None):
        """Attach an environment after construction and build the MPC solver."""

        self.env = env
        self.observation_indices = observation_indices or MPCObservationIndices.from_env(env)
        self._build_mpc()
        self.reset()
        return self

    def _build_mpc(self) -> None:
        try:
            import casadi as ca
            import do_mpc
        except ImportError as exc:
            raise ImportError(
                "OccupantAdaptiveRCMPC requires casadi and do-mpc. Install the "
                "project requirements before running MPC experiments."
            ) from exc

        cfg = self.config
        p = self.params

        model = do_mpc.model.Model("discrete")
        indoor_temperature = model.set_variable("_x", "Tin")
        if cfg.use_smooth_du:
            model.set_variable("_x", "u_prev")

        action = model.set_variable("_u", "u")
        outdoor_temperature = model.set_variable("_tvp", "Tout")
        solar = model.set_variable("_tvp", "Sol")
        price = model.set_variable("_tvp", "Price")
        preferred_temperature = model.set_variable("_tvp", "Tsp")
        lower_comfort_bound = model.set_variable("_tvp", "Tmin")
        upper_comfort_bound = model.set_variable("_tvp", "Tmax")
        disturbance = model.set_variable("_tvp", "w_est")

        cop = p.cop_a + p.cop_b * outdoor_temperature
        cop = ca.fmin(ca.fmax(cop, 0.5), 6.0)
        indoor_temperature_next = (
            p.A * indoor_temperature
            + p.Bd0 * p.mean_indoor_temperature
            + p.Bd1 * outdoor_temperature
            + p.Bd2 * solar
            + p.Bu * cop * action
            + disturbance
        )

        model.set_rhs("Tin", indoor_temperature_next)
        if cfg.use_smooth_du:
            model.set_rhs("u_prev", action)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=cfg.horizon,
            t_step=1.0,
            state_discretization="discrete",
            store_full_solution=True,
        )

        if cfg.objective_mode == "baseline":
            stage_cost = (indoor_temperature - lower_comfort_bound) ** 2 + cfg.lambda_u * action**2
            terminal_cost = (indoor_temperature - lower_comfort_bound) ** 2
        elif cfg.objective_mode == "energy_comfort":
            energy_cost = price * p.power_per_unit_action * action
            comfort_cost = (indoor_temperature - preferred_temperature) ** 2
            peak_cost = (p.power_per_unit_action * action) ** 2
            smooth_cost = cfg.lambda_u * action**2
            if cfg.use_smooth_du:
                smooth_cost = smooth_cost + cfg.lambda_du * (action - model.x["u_prev"]) ** 2
            stage_cost = (
                cfg.energy_weight * energy_cost
                + cfg.comfort_weight * comfort_cost
                + cfg.peak_weight * peak_cost
                + smooth_cost
            )
            terminal_cost = cfg.comfort_weight * comfort_cost
        else:
            raise ValueError(f"Unknown objective_mode: {cfg.objective_mode}")

        if cfg.objective_mode == "baseline" and cfg.use_smooth_du:
            stage_cost = stage_cost + cfg.lambda_du * (action - model.x["u_prev"]) ** 2

        mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
        mpc.bounds["lower", "_u", "u"] = cfg.action_lower_bound
        mpc.bounds["upper", "_u", "u"] = cfg.action_upper_bound
        mpc.set_nl_cons(
            "too_cold",
            lower_comfort_bound - indoor_temperature,
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=cfg.cold_penalty,
        )
        mpc.set_nl_cons(
            "too_hot",
            indoor_temperature - upper_comfort_bound,
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=cfg.hot_penalty,
        )

        tvp_template = mpc.get_tvp_template()
        self._latest = {
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
            for k in range(cfg.horizon + 1):
                if k == 0:
                    tout = self._latest["Tout0"]
                    sol = self._latest["Sol0"]
                    price_value = self._latest["Price0"]
                    setpoint = self._latest["Tsp0"]
                elif k <= 3:
                    tout = self._latest["Toutp"][k - 1]
                    sol = self._latest["Solp"][k - 1]
                    price_value = self._latest["Pricep"][k - 1]
                    setpoint = self._latest["Tspp"][k - 1]
                else:
                    tout = self._latest["Toutp"][-1]
                    sol = self._latest["Solp"][-1]
                    price_value = self._latest["Pricep"][-1]
                    setpoint = self._latest["Tspp"][-1]

                tvp_template["_tvp", k, "Tout"] = float(tout)
                tvp_template["_tvp", k, "Sol"] = float(sol)
                tvp_template["_tvp", k, "Price"] = float(price_value)
                tvp_template["_tvp", k, "Tsp"] = float(setpoint)
                tvp_template["_tvp", k, "Tmin"] = float(self._latest["Tmin0"])
                tvp_template["_tvp", k, "Tmax"] = float(self._latest["Tmax0"])
                tvp_template["_tvp", k, "w_est"] = float(self._latest["w_est"])

            return tvp_template

        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        self._model = model
        self.mpc = mpc
        self._built = True

    def reset(self) -> None:
        if not self._built:
            return
        self.mpc.reset_history()
        self.w_est_val = 0.0
        self.prev_indoor_temperature = None
        self.prev_action = None
        self.prev_outdoor_temperature = None
        self.prev_solar = None
        self.last_action = 0.0

    def predict(self, observation, deterministic: bool = True):
        if not self._built:
            raise RuntimeError("Call bind_env(env) or pass env=... before predict().")

        np_ = _require_numpy()
        obs = np_.asarray(
            observation[0] if isinstance(observation, (list, tuple)) else observation,
            dtype=np_.float32,
        ).reshape(-1)
        idx = self.observation_indices
        building = self.env.unwrapped.buildings[0]
        time_step = building.time_step

        indoor_temperature = float(obs[idx.indoor_temperature])
        outdoor_temperature = float(obs[idx.outdoor_temperature])
        solar = float(obs[idx.direct_solar_irradiance])
        outdoor_temperature_forecast = [float(obs[i]) for i in idx.outdoor_temperature_forecast]
        solar_forecast = [float(obs[i]) for i in idx.direct_solar_irradiance_forecast]

        scheduled_setpoints = building.energy_simulation.indoor_dry_bulb_temperature_heating_set_point_without_control
        preferred_now = self._preferred_temperature(time_step, scheduled_setpoints, obs, indoor_temperature)
        preferred_forecast = [
            self._preferred_temperature(
                min(time_step + h, len(scheduled_setpoints) - 1),
                scheduled_setpoints,
                obs,
                indoor_temperature,
            )
            for h in (1, 2, 3)
        ]

        if self.config.fixed_band is None:
            lower_comfort_bound = float(obs[idx.heating_setpoint])
            upper_comfort_bound = float(obs[idx.cooling_setpoint])
        else:
            lower_comfort_bound, upper_comfort_bound = map(float, self.config.fixed_band)

        if self.prev_indoor_temperature is not None:
            predicted = self.rc_model.predict_next(
                self.prev_indoor_temperature,
                self.prev_outdoor_temperature,
                self.prev_solar,
                self.prev_action,
                self.w_est_val,
            )
            error = indoor_temperature - predicted
            self.w_est_val = self.w_est_val + self.config.disturbance_gain * error

        self._latest.update(
            {
                "Tout0": outdoor_temperature,
                "Sol0": solar,
                "Price0": self._price_at(time_step, 0),
                "Tsp0": preferred_now,
                "Toutp": outdoor_temperature_forecast,
                "Solp": solar_forecast,
                "Pricep": [self._price_at(time_step, h) for h in (1, 2, 3)],
                "Tspp": preferred_forecast,
                "Tmin0": lower_comfort_bound,
                "Tmax0": upper_comfort_bound,
                "w_est": self.w_est_val,
            }
        )

        if self.config.use_smooth_du:
            x0 = np_.array([[indoor_temperature], [self.last_action]])
        else:
            x0 = np_.array([[indoor_temperature]])

        action = float(np_.asarray(self.mpc.make_step(x0)).reshape(-1)[0])
        action = float(np_.clip(action, self.config.action_lower_bound, self.config.action_upper_bound))

        self.prev_indoor_temperature = indoor_temperature
        self.prev_action = action
        self.prev_outdoor_temperature = outdoor_temperature
        self.prev_solar = solar
        self.last_action = action

        return [[action]]

    def _price_at(self, time_step: int, horizon: int) -> float:
        pricing = self.env.unwrapped.buildings[0].pricing
        if horizon == 0:
            return float(pricing.electricity_pricing[time_step])

        attr = f"electricity_pricing_predicted_{horizon}"
        if hasattr(pricing, attr):
            values = getattr(pricing, attr)
            return float(values[time_step])

        values = pricing.electricity_pricing
        return float(values[min(time_step + horizon, len(values) - 1)])

    def _preferred_temperature(self, time_step: int, scheduled_setpoints, obs, indoor_temperature: float) -> float:
        scheduled = float(scheduled_setpoints[time_step])
        if self.preference_model is None:
            return scheduled

        model = self.preference_model
        if callable(model):
            return float(
                model(
                    time_step=time_step,
                    observation=obs,
                    indoor_temperature=indoor_temperature,
                    scheduled_temperature=scheduled,
                )
            )
        if hasattr(model, "preferred_temperature"):
            return float(model.preferred_temperature)
        if hasattr(model, "T_pref"):
            return float(model.T_pref)
        if hasattr(model, "occupied_temperature"):
            return float(model.occupied_temperature)
        if not hasattr(model, "predict"):
            return scheduled

        predict = model.predict
        try:
            parameters = inspect.signature(predict).parameters
        except (TypeError, ValueError):
            parameters = {}

        kwargs = {}
        for name in parameters:
            if name in {"time_step", "t", "t_idx"}:
                kwargs[name] = time_step
            elif name in {"observation", "obs"}:
                kwargs[name] = obs
            elif name == "indoor_temperature":
                kwargs[name] = indoor_temperature
            elif name in {"scheduled_temperature", "scheduled_setpoint"}:
                kwargs[name] = scheduled

        if kwargs:
            return float(predict(**kwargs))

        # Constant-preference objects may expose a zero-argument predict().
        return float(predict())
