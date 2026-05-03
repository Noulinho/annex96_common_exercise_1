"""Fit RC models from saved CityLearn rollout datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from src.models.rc_1r1c import RC1R1CModel, RC1R1CParams


def squeeze_citylearn_dataset(npz_path: str | Path):
    data = np.load(npz_path)
    x = data["states"]
    xp = data["next_states"]
    u = np.asarray(data["actions"])

    if x.ndim == 3:
        x = x[:, 0, :]
    if xp.ndim == 3:
        xp = xp[:, 0, :]
    if u.ndim == 3:
        u = u[:, 0, 0]
    elif u.ndim == 2:
        u = u[:, 0]

    return x, np.asarray(u, dtype=float), xp


def fit_1r1c_from_arrays(
    x,
    u,
    xp,
    *,
    indoor_idx: int,
    outdoor_idx: int,
    solar_idx: int,
    heating_demand_idx: int,
    heating_power_idx: int,
    ridge_alpha: float = 1e-3,
) -> RC1R1CModel:
    tin = x[:, indoor_idx].astype(float)
    tin_next = xp[:, indoor_idx].astype(float)
    tout = x[:, outdoor_idx].astype(float)
    solar = x[:, solar_idx].astype(float)
    heat_demand = x[:, heating_demand_idx].astype(float)
    heat_power = x[:, heating_power_idx].astype(float)

    cop_mask = heat_power > 0.1
    cop_reg = LinearRegression()
    cop_reg.fit(tout[cop_mask].reshape(-1, 1), heat_demand[cop_mask] / heat_power[cop_mask])

    cop_a = float(cop_reg.intercept_)
    cop_b = float(cop_reg.coef_[0])
    mean_tin = float(tin.mean())
    cop_all = np.clip(cop_a + cop_b * tout, 0.5, 6.0)

    phi = np.column_stack([tin, np.full_like(tin, mean_tin), tout, solar, cop_all * u])
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=False)
    ridge.fit(phi, tin_next - tin)
    alpha, bd0, bd1, bd2, bu = ridge.coef_

    power_mask = u > 1e-3
    if np.any(power_mask):
        power_per_unit_action = float(
            np.dot(u[power_mask], heat_power[power_mask]) / np.dot(u[power_mask], u[power_mask])
        )
    else:
        power_per_unit_action = 0.0

    return RC1R1CModel(
        RC1R1CParams(
            cop_a=cop_a,
            cop_b=cop_b,
            mean_indoor_temperature=mean_tin,
            A=float(1.0 + alpha),
            Bd0=float(bd0),
            Bd1=float(bd1),
            Bd2=float(bd2),
            Bu=float(bu),
            power_per_unit_action=power_per_unit_action,
        )
    )


def save_rc_model(model: RC1R1CModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.params.to_dict(), indent=2) + "\n")


def load_rc_model(path: str | Path) -> RC1R1CModel:
    data = json.loads(Path(path).read_text())
    return RC1R1CModel(RC1R1CParams(**data))
