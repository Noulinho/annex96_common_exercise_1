"""Two-state 2R2C thermal model for MPC.

State: [Tin, Tm]  (indoor air temperature, thermal mass temperature)
Discrete-time dynamics (teacher-forcing parametrisation):

    Tin(k+1) = a11*Tin(k) + a12*Tm(k) + b_tout*Tout(k) + b_sol*Sol(k)
               + b_heat*COP(Tout)*u(k) + b0 + w_est
    Tm(k+1)  = a21*Tin(k) + a22*Tm(k)

The thermal time constants are:
    tau_air  = -1 / ln(a11) approximately, ignoring mass coupling
    tau_mass = -1 / ln(a22) hours  (should be >> tau_air for useful storage)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RC2R2CParams:
    cop_a: float
    cop_b: float
    a11: float
    a12: float
    a21: float
    a22: float
    b_tout: float
    b_sol: float
    b_heat: float
    b0: float
    power_per_unit_action: float
    fit_mse: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def tau_mass_hours(self) -> float:
        return -1.0 / np.log(max(self.a22, 1e-9))

    @property
    def tau_air_approx_hours(self) -> float:
        """Approximate air time constant (ignoring mass coupling)."""
        return -1.0 / np.log(max(self.a11, 1e-9))


class RC2R2CModel:
    def __init__(self, params: RC2R2CParams):
        self.params = params

    def cop(self, outdoor_temperature):
        p = self.params
        return np.clip(p.cop_a + p.cop_b * np.asarray(outdoor_temperature), 0.5, 6.0)

    def predict_next(self, tin, tm, outdoor_temperature, solar_irradiance, action, disturbance=0.0):
        p = self.params
        cop = float(np.clip(p.cop_a + p.cop_b * outdoor_temperature, 0.5, 6.0))
        tin_next = (
            p.a11 * tin
            + p.a12 * tm
            + p.b_tout * outdoor_temperature
            + p.b_sol * solar_irradiance
            + p.b_heat * cop * action
            + p.b0
            + disturbance
        )
        tm_next = p.a21 * tin + p.a22 * tm
        return float(tin_next), float(tm_next)

    def rollout(self, initial_temperature, outdoor_temperature, solar_irradiance, action):
        outdoor_temperature = np.asarray(outdoor_temperature, dtype=float)
        solar_irradiance = np.asarray(solar_irradiance, dtype=float)
        action = np.asarray(action, dtype=float)
        T = len(outdoor_temperature)
        tin_pred = np.zeros(T, dtype=float)
        tin = float(initial_temperature)
        tm = float(initial_temperature)
        for k in range(T):
            tin_pred[k] = tin
            tin, tm = self.predict_next(tin, tm, outdoor_temperature[k], solar_irradiance[k], action[k])
        return tin_pred


def save_2r2c_model(model: RC2R2CModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.params.to_dict(), indent=2) + "\n")


def load_2r2c_model(path: str | Path) -> RC2R2CModel:
    data = json.loads(Path(path).read_text())
    return RC2R2CModel(RC2R2CParams(**data))


# Legacy helpers kept for backwards compatibility ----------------------------

def discretize_2r2c(params, dt_seconds: float = 3600.0):
    ram, rao, ca, cm, ku, ks, k0 = params
    ac = np.array(
        [
            [-(1 / (ca * ram) + 1 / (ca * rao)), 1 / (ca * ram)],
            [1 / (cm * ram), -1 / (cm * ram)],
        ],
        dtype=float,
    )
    bc = np.array(
        [
            [1 / (ca * rao), ks / ca, ku / ca, k0 / ca],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return np.eye(2) + dt_seconds * ac, dt_seconds * bc


def rollout_2r2c(params_fit: dict[str, float], initial_air_temperature, outdoor_temperature, irradiance, action):
    a, b = discretize_2r2c(
        [
            params_fit["Ram"],
            params_fit["Rao"],
            params_fit["Ca"],
            params_fit["Cm"],
            params_fit["ku"],
            params_fit["ks"],
            params_fit["k0"],
        ]
    )
    x = np.array([initial_air_temperature, initial_air_temperature], dtype=float)
    predicted = np.zeros_like(outdoor_temperature, dtype=float)

    for k in range(len(predicted)):
        u = np.array([outdoor_temperature[k], irradiance[k], action[k], 1.0])
        x = a @ x + b @ u
        predicted[k] = x[0]

    return predicted
