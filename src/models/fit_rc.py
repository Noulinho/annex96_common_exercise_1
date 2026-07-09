"""Fit RC models from saved CityLearn rollout datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from src.models.rc_1r1c import RC1R1CModel, RC1R1CParams
from src.models.rc_2r2c import RC2R2CModel, RC2R2CParams


def squeeze_citylearn_dataset(npz_path: str | Path, *, return_dones: bool = False):
    data = np.load(npz_path)
    x = data["states"]
    xp = data["next_states"]
    u = np.asarray(data["actions"])
    dones = np.asarray(data["dones"]).reshape(-1) if "dones" in data.files else None

    if x.ndim == 3:
        x = x[:, 0, :]
    if xp.ndim == 3:
        xp = xp[:, 0, :]
    if u.ndim == 3:
        u = u[:, 0, 0]
    elif u.ndim == 2:
        u = u[:, 0]

    if return_dones:
        return x, np.asarray(u, dtype=float), xp, dones
    return x, np.asarray(u, dtype=float), xp


def _segments_from_dones(n: int, dones=None, *, min_length: int = 1) -> list[tuple[int, int]]:
    if dones is None:
        return [(0, n)] if n >= min_length else []

    done_arr = np.asarray(dones).reshape(-1).astype(bool)
    if len(done_arr) != n:
        return [(0, n)] if n >= min_length else []

    ends = list(np.flatnonzero(done_arr) + 1)
    segments: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        end_i = int(end)
        if end_i - start >= min_length:
            segments.append((start, end_i))
        start = end_i
    if n - start >= min_length:
        segments.append((start, n))

    return segments or ([(0, n)] if n >= min_length else [])


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


def fit_1r1c_bounded_robust_from_arrays(
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
    huber_scale: float = 1.0,
) -> RC1R1CModel:
    """Fit the 1R1C form with physical-ish bounds and robust residual loss."""

    from scipy.optimize import least_squares

    tin = x[:, indoor_idx].astype(float)
    tin_next = xp[:, indoor_idx].astype(float)
    tout = x[:, outdoor_idx].astype(float)
    solar = x[:, solar_idx].astype(float)
    heat_demand = x[:, heating_demand_idx].astype(float)
    heat_power = x[:, heating_power_idx].astype(float)

    cop_mask = (heat_power > 0.1) & np.isfinite(heat_power) & np.isfinite(heat_demand) & np.isfinite(tout)
    cop_reg = LinearRegression()
    if np.count_nonzero(cop_mask) >= 2:
        cop_reg.fit(tout[cop_mask].reshape(-1, 1), heat_demand[cop_mask] / heat_power[cop_mask])
        cop_a = float(cop_reg.intercept_)
        cop_b = float(cop_reg.coef_[0])
    else:
        cop_a = 3.0
        cop_b = 0.0

    mean_tin = float(np.nanmean(tin))
    cop_all = np.clip(cop_a + cop_b * tout, 0.5, 6.0)
    eff_heat = cop_all * u
    phi = np.column_stack([tin, np.full_like(tin, mean_tin), tout, solar, eff_heat])
    y = tin_next

    valid = np.isfinite(phi).all(axis=1) & np.isfinite(y)
    if np.count_nonzero(valid) < 24:
        raise ValueError("Not enough finite transitions to fit a bounded robust 1R1C model.")

    # Ridge on Tin(k+1)-Tin(k) is a cheap warm-start for the bounded nonlinear solve.
    warm = fit_1r1c_from_arrays(
        x,
        u,
        xp,
        indoor_idx=indoor_idx,
        outdoor_idx=outdoor_idx,
        solar_idx=solar_idx,
        heating_demand_idx=heating_demand_idx,
        heating_power_idx=heating_power_idx,
        ridge_alpha=ridge_alpha,
    ).params
    p0 = np.asarray([warm.A, warm.Bd0, warm.Bd1, warm.Bd2, warm.Bu], dtype=float)

    lower = np.asarray([0.05, -2.0, -1.0, -0.02, 0.0], dtype=float)
    upper = np.asarray([0.995, 2.0, 1.0, 0.02, 80.0], dtype=float)
    p0 = np.clip(p0, lower + 1e-8, upper - 1e-8)

    phi_valid = phi[valid]
    y_valid = y[valid]

    def residual(params: np.ndarray) -> np.ndarray:
        return phi_valid @ params - y_valid

    result = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=float(huber_scale),
        x_scale=np.asarray([0.5, 0.5, 0.2, 0.002, 10.0], dtype=float),
        max_nfev=10000,
    )
    A, bd0, bd1, bd2, bu = result.x

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
            A=float(A),
            Bd0=float(bd0),
            Bd1=float(bd1),
            Bd2=float(bd2),
            Bu=float(bu),
            power_per_unit_action=power_per_unit_action,
        )
    )


def evaluate_1r1c_fit_from_arrays(
    model: RC1R1CModel,
    x,
    u,
    xp,
    *,
    indoor_idx: int,
    outdoor_idx: int,
    solar_idx: int,
    dones=None,
) -> dict[str, float]:
    """Return one-step and free-run temperature fit metrics for a 1R1C model."""

    tin = x[:, indoor_idx].astype(float)
    tin_next = xp[:, indoor_idx].astype(float)
    tout = x[:, outdoor_idx].astype(float)
    solar = x[:, solar_idx].astype(float)
    u = np.asarray(u, dtype=float)

    one_step = model.predict_next(tin, tout, solar, u)
    err = one_step - tin_next
    valid = np.isfinite(err)
    if dones is None:
        segments = [(0, len(tin))]
    else:
        done_arr = np.asarray(dones).reshape(-1).astype(bool)
        ends = np.flatnonzero(done_arr) + 1
        starts = np.r_[0, ends[:-1]]
        segments = [(int(start), int(end)) for start, end in zip(starts, ends) if end - start >= 2]
        if not segments and len(tin) >= 2:
            segments = [(0, len(tin))]

    free_err_parts = []
    for start, end in segments:
        free = model.rollout(tin[start], tout[start:end], solar[start:end], u[start:end])
        free_err_parts.append(free[1:] - tin[start + 1:end])
    free_err = np.concatenate(free_err_parts) if free_err_parts else np.asarray([], dtype=float)
    free_valid = np.isfinite(free_err)

    def _rmse(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.sqrt(np.nanmean(values[mask] ** 2))) if np.any(mask) else float("nan")

    def _mae(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.nanmean(np.abs(values[mask]))) if np.any(mask) else float("nan")

    def _bias(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.nanmean(values[mask])) if np.any(mask) else float("nan")

    return {
        "one_step_rmse_c": _rmse(err, valid),
        "one_step_mae_c": _mae(err, valid),
        "one_step_bias_c": _bias(err, valid),
        "free_run_rmse_c": _rmse(free_err, free_valid),
        "free_run_mae_c": _mae(free_err, free_valid),
        "free_run_bias_c": _bias(free_err, free_valid),
    }


def fit_2r2c_from_arrays(
    x,
    u,
    xp,
    *,
    indoor_idx: int,
    outdoor_idx: int,
    solar_idx: int,
    heating_demand_idx: int,
    heating_power_idx: int,
    bounded: bool = False,
    dones=None,
    ridge_alpha: float = 1e-3,
    huber_scale: float = 2.0,
    rollout_horizon: int = 24,
    rollout_stride: int = 24,
    rollout_weight: float = 0.25,
) -> RC2R2CModel:
    """Fit a 2R2C thermal model using teacher-forcing hidden-state approach.

    Parametrises the discrete-time A matrix as:
        a11 = 1 - alpha - beta   (air self-decay)
        a12 = alpha              (mass → air coupling)
        a21 = gamma              (air → mass coupling)
        a22 = 1 - gamma          (mass self-decay)
    where alpha, beta, gamma > 0 and alpha+beta < 1, gamma < 1.

    The Tin update is:
        Tin(k+1) = a11*Tin(k) + a12*Tm(k) + b_tout*Tout(k) + b_sol*Sol(k)
                   + b_heat*COP(k)*u(k) + b0
    Tm evolves as a hidden state:
        Tm(k+1) = a21*Tin(k) + a22*Tm(k)
    """
    from scipy.optimize import minimize

    tin = x[:, indoor_idx].astype(float)
    tin_next = xp[:, indoor_idx].astype(float)
    tout = x[:, outdoor_idx].astype(float)
    solar = x[:, solar_idx].astype(float)
    heat_demand = x[:, heating_demand_idx].astype(float)
    heat_power = x[:, heating_power_idx].astype(float)

    cop_mask = (heat_power > 0.1) & np.isfinite(heat_power) & np.isfinite(heat_demand) & np.isfinite(tout)
    cop_reg = LinearRegression()
    if np.count_nonzero(cop_mask) >= 2:
        cop_reg.fit(tout[cop_mask].reshape(-1, 1), heat_demand[cop_mask] / heat_power[cop_mask])
        cop_a = float(cop_reg.intercept_)
        cop_b = float(cop_reg.coef_[0])
    else:
        cop_a = 3.0
        cop_b = 0.0
    cop_all = np.clip(cop_a + cop_b * tout, 0.5, 6.0)
    eff_heat = cop_all * u

    power_mask = u > 1e-3
    power_per_unit = float(
        np.dot(u[power_mask], heat_power[power_mask]) / np.dot(u[power_mask], u[power_mask])
    ) if np.any(power_mask) else 0.0

    T = len(tin)
    segments = _segments_from_dones(T, dones, min_length=1)

    # Initialise using 1R1C fit as warm-start
    phi0 = np.column_stack([tin, np.full(T, tin.mean()), tout, solar, eff_heat])
    valid0 = np.isfinite(phi0).all(axis=1) & np.isfinite(tin_next)
    if np.count_nonzero(valid0) < 24:
        raise ValueError("Not enough finite transitions to fit a 2R2C model.")
    r0 = Ridge(alpha=ridge_alpha, fit_intercept=False)
    r0.fit(phi0[valid0], tin_next[valid0] - tin[valid0])
    _alpha_1r1c, _bd0, _bd1_1r1c, _bd2_1r1c, _bu_1r1c = r0.coef_

    # alpha = mass-air coupling, beta = outdoor coupling, gamma = mass decay
    # Start with alpha=0.05 (slow mass coupling), beta from 1R1C fit, gamma small
    alpha0 = max(0.02, -_alpha_1r1c * 0.15)
    beta0 = max(0.02, _bd1_1r1c)
    gamma0 = max(0.005, alpha0 * 0.1)

    p0 = np.array([
        np.log(np.clip(alpha0, 1e-4, 0.49)),
        np.log(np.clip(gamma0, 1e-5, 0.49)),
        np.log(np.clip(beta0, 1e-4, 0.49)),
        float(_bd2_1r1c),
        float(_bu_1r1c),
        float(_bd0 * tin.mean()),
    ])

    def _unpack(params):
        log_alpha, log_gamma, log_beta, b_sol, b_heat, b0 = params
        alpha = np.exp(log_alpha)
        gamma = np.exp(log_gamma)
        beta = np.exp(log_beta)
        if alpha + beta >= 0.995 or gamma >= 0.995:
            return None
        a11 = 1.0 - alpha - beta
        a12 = alpha
        a21 = gamma
        a22 = 1.0 - gamma
        return a11, a12, a21, a22, beta, b_sol, b_heat, b0

    def _soft_l1_mean(residuals: np.ndarray) -> float:
        if residuals.size == 0:
            return 1e10
        scaled = residuals / float(huber_scale)
        return float(np.mean(2.0 * huber_scale**2 * (np.sqrt(1.0 + scaled**2) - 1.0)))

    def _teacher_residuals(params, *, return_tm: bool = False):
        unpacked = _unpack(params)
        if unpacked is None:
            return None if not return_tm else (None, None)
        a11, a12, a21, a22, beta, b_sol, b_heat, b0 = unpacked
        residuals: list[float] = []
        tm_seq = np.full(T, np.nan, dtype=float) if return_tm else None

        for start, end in segments:
            tm = float(tin[start])
            for k in range(start, end):
                if return_tm:
                    tm_seq[k] = tm
                if not (
                    np.isfinite(tin[k])
                    and np.isfinite(tm)
                    and np.isfinite(tout[k])
                    and np.isfinite(solar[k])
                    and np.isfinite(eff_heat[k])
                    and np.isfinite(tin_next[k])
                ):
                    tm = a21 * tin[k] + a22 * tm if np.isfinite(tin[k]) and np.isfinite(tm) else tm
                    continue
                tin_pred = a11 * tin[k] + a12 * tm + beta * tout[k] + b_sol * solar[k] + b_heat * eff_heat[k] + b0
                residuals.append(float(tin_pred - tin_next[k]))
                tm = a21 * tin[k] + a22 * tm

        res = np.asarray(residuals, dtype=float)
        return (res, tm_seq) if return_tm else res

    def _rollout_residuals(params, tm_seq: np.ndarray) -> np.ndarray:
        unpacked = _unpack(params)
        if unpacked is None:
            return np.asarray([1e5], dtype=float)
        a11, a12, a21, a22, beta, b_sol, b_heat, b0 = unpacked
        residuals: list[float] = []
        horizon = max(1, int(rollout_horizon))
        stride = max(1, int(rollout_stride))

        for start, end in segments:
            if end <= start:
                continue
            for window_start in range(start, end, stride):
                tin_state = float(tin[window_start])
                tm_state = float(tm_seq[window_start]) if np.isfinite(tm_seq[window_start]) else tin_state
                window_end = min(end, window_start + horizon)
                for k in range(window_start, window_end):
                    if not (
                        np.isfinite(tin_state)
                        and np.isfinite(tm_state)
                        and np.isfinite(tout[k])
                        and np.isfinite(solar[k])
                        and np.isfinite(eff_heat[k])
                        and np.isfinite(tin_next[k])
                    ):
                        continue
                    tin_pred = (
                        a11 * tin_state
                        + a12 * tm_state
                        + beta * tout[k]
                        + b_sol * solar[k]
                        + b_heat * eff_heat[k]
                        + b0
                    )
                    tm_next = a21 * tin_state + a22 * tm_state
                    if not np.isfinite(tin_pred) or abs(tin_pred) > 1e4:
                        return np.asarray([1e5], dtype=float)
                    residuals.append(float(tin_pred - tin_next[k]))
                    tin_state = float(tin_pred)
                    tm_state = float(tm_next)

        return np.asarray(residuals, dtype=float)

    def _objective(params):
        teacher, tm_seq = _teacher_residuals(params, return_tm=True)
        if teacher is None or teacher.size == 0:
            return 1e10
        teacher_loss = _soft_l1_mean(teacher)
        rollout = _rollout_residuals(params, tm_seq)
        rollout_loss = _soft_l1_mean(rollout)
        if not np.isfinite(teacher_loss) or not np.isfinite(rollout_loss):
            return 1e10
        return teacher_loss + float(rollout_weight) * rollout_loss

    # Bounds: keep gamma in (0.05, 0.5) → tau_mass in (1.4h, ~20h)
    # Lower bound raised from 0.005 to 0.05: prevents degenerate integrator mode
    # (γ=0.005 → τ_mass=200h makes the QP ill-conditioned in GNU-RL)
    if bounded:
        # Physical bounds on b_sol, b_heat, b0 prevent degenerate solutions on difficult buildings
        # (e.g. b0=-9.5 for Building 11 under the unconstrained fit)
        opt_bounds = [
            (np.log(0.002), np.log(0.7)),   # log_alpha
            (np.log(0.05),  np.log(0.5)),   # log_gamma
            (np.log(0.002), np.log(0.7)),   # log_beta
            (-0.001, 0.03),                  # b_sol: near-positive, small
            (0.0, 80.0),                     # b_heat: heating must warm
            (-5.0, 5.0),                     # b0: bounded thermal bias
        ]
        p0[3] = float(np.clip(p0[3], -0.001, 0.03))
        p0[4] = float(np.clip(p0[4], 0.0, 80.0))
        p0[5] = float(np.clip(p0[5], -5.0, 5.0))
    else:
        opt_bounds = [
            (np.log(0.002), np.log(0.7)),   # log_alpha
            (np.log(0.05),  np.log(0.5)),   # log_gamma: controls tau_mass
            (np.log(0.002), np.log(0.7)),   # log_beta
            (None, None),                    # b_sol
            (None, None),                    # b_heat
            (None, None),                    # b0
        ]
    p0[:3] = np.clip(
        p0[:3],
        [b[0] for b in opt_bounds[:3]],
        [b[1] for b in opt_bounds[:3]],
    )
    res = minimize(_objective, p0, method="L-BFGS-B", bounds=opt_bounds,
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
    # Retry from a few perturbations to avoid local minima
    for scales in [(0.5, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 0.5, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 0.5), (1.0, 1.0, 2.0)]:
        p_try = p0.copy()
        for i, scale in enumerate(scales):
            p_try[i] = np.log(np.exp(p_try[i]) * scale)
        p_try[:3] = np.clip(
            p_try[:3],
            [b[0] for b in opt_bounds[:3]],
            [b[1] for b in opt_bounds[:3]],
        )
        r2 = minimize(_objective, p_try, method="L-BFGS-B", bounds=opt_bounds,
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
        if r2.fun < res.fun:
            res = r2

    log_alpha, log_gamma, log_beta, b_sol, b_heat, b0 = res.x
    alpha = float(np.exp(log_alpha))
    gamma = float(np.exp(log_gamma))
    beta = float(np.exp(log_beta))

    teacher_res = _teacher_residuals(res.x)
    fit_mse = float(np.mean(teacher_res**2)) if teacher_res is not None and teacher_res.size else float("nan")

    return RC2R2CModel(RC2R2CParams(
        cop_a=cop_a,
        cop_b=cop_b,
        a11=1.0 - alpha - beta,
        a12=alpha,
        a21=gamma,
        a22=1.0 - gamma,
        b_tout=beta,
        b_sol=float(b_sol),
        b_heat=float(b_heat),
        b0=float(b0),
        power_per_unit_action=power_per_unit,
        fit_mse=fit_mse,
    ))


def evaluate_2r2c_fit_from_arrays(
    model: RC2R2CModel,
    x,
    u,
    xp,
    *,
    indoor_idx: int,
    outdoor_idx: int,
    solar_idx: int,
    dones=None,
    rollout_horizon: int = 24,
    rollout_stride: int = 24,
) -> dict[str, float]:
    """Return teacher-forced, rolling-horizon, and full free-run fit metrics."""

    tin = x[:, indoor_idx].astype(float)
    tin_next = xp[:, indoor_idx].astype(float)
    tout = x[:, outdoor_idx].astype(float)
    solar = x[:, solar_idx].astype(float)
    u = np.asarray(u, dtype=float)
    p = model.params
    segments = _segments_from_dones(len(tin), dones, min_length=1)

    one_step_err: list[float] = []
    horizon_err: list[float] = []
    free_err: list[float] = []
    tm_seq = np.full(len(tin), np.nan, dtype=float)

    for start, end in segments:
        tm = float(tin[start])
        for k in range(start, end):
            tm_seq[k] = tm
            tin_pred, _ = model.predict_next(tin[k], tm, tout[k], solar[k], u[k])
            one_step_err.append(float(tin_pred - tin_next[k]))
            tm = p.a21 * tin[k] + p.a22 * tm

    horizon = max(1, int(rollout_horizon))
    stride = max(1, int(rollout_stride))
    for start, end in segments:
        for window_start in range(start, end, stride):
            tin_state = float(tin[window_start])
            tm_state = float(tm_seq[window_start]) if np.isfinite(tm_seq[window_start]) else tin_state
            for k in range(window_start, min(end, window_start + horizon)):
                tin_pred, tm_next = model.predict_next(tin_state, tm_state, tout[k], solar[k], u[k])
                horizon_err.append(float(tin_pred - tin_next[k]))
                tin_state = tin_pred
                tm_state = tm_next

        tin_state = float(tin[start])
        tm_state = float(tin[start])
        for k in range(start, end):
            tin_pred, tm_next = model.predict_next(tin_state, tm_state, tout[k], solar[k], u[k])
            free_err.append(float(tin_pred - tin_next[k]))
            tin_state = tin_pred
            tm_state = tm_next

    one_step = np.asarray(one_step_err, dtype=float)
    horizon_values = np.asarray(horizon_err, dtype=float)
    free = np.asarray(free_err, dtype=float)

    def _finite(values: np.ndarray) -> np.ndarray:
        return values[np.isfinite(values)]

    def _rmse(values: np.ndarray) -> float:
        values = _finite(values)
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    def _mae(values: np.ndarray) -> float:
        values = _finite(values)
        return float(np.mean(np.abs(values))) if values.size else float("nan")

    def _bias(values: np.ndarray) -> float:
        values = _finite(values)
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "one_step_rmse_c": _rmse(one_step),
        "one_step_mae_c": _mae(one_step),
        "one_step_bias_c": _bias(one_step),
        "horizon_24h_rmse_c": _rmse(horizon_values),
        "horizon_24h_mae_c": _mae(horizon_values),
        "horizon_24h_bias_c": _bias(horizon_values),
        "free_run_rmse_c": _rmse(free),
        "free_run_mae_c": _mae(free),
        "free_run_bias_c": _bias(free),
    }


def evaluate_2r2c_kalman_fit_from_arrays(
    model: RC2R2CModel,
    x,
    u,
    xp,
    *,
    indoor_idx: int,
    outdoor_idx: int,
    solar_idx: int,
    dones=None,
    process_var: float = 0.05,
    measurement_var: float = 0.25,
    initial_var: float = 4.0,
    rollout_horizon: int = 24,
    rollout_stride: int = 24,
) -> dict[str, float]:
    """Evaluate 2R2C forecasts with a scalar Kalman observer for hidden Tm.

    Tin is directly observed. The filter uses the next observed Tin residual to
    correct the previous hidden mass estimate, then propagates that corrected
    mass state to the next control step.
    """

    tin = x[:, indoor_idx].astype(float)
    tin_next = xp[:, indoor_idx].astype(float)
    tout = x[:, outdoor_idx].astype(float)
    solar = x[:, solar_idx].astype(float)
    u = np.asarray(u, dtype=float)
    p = model.params
    segments = _segments_from_dones(len(tin), dones, min_length=1)

    tm_filtered = np.full(len(tin), np.nan, dtype=float)
    one_step_err: list[float] = []
    horizon_err: list[float] = []

    q = max(float(process_var), 1e-12)
    r = max(float(measurement_var), 1e-12)

    for start, end in segments:
        tm = float(tin[start])
        var = max(float(initial_var), 1e-12)
        for k in range(start, end):
            tm_filtered[k] = tm
            tin_pred, _ = model.predict_next(tin[k], tm, tout[k], solar[k], u[k])
            err = float(tin_pred - tin_next[k])
            one_step_err.append(err)

            innovation = float(tin_next[k] - tin_pred)
            innovation_var = p.a12 * p.a12 * var + r
            gain = 0.0 if innovation_var <= 1e-12 else var * p.a12 / innovation_var
            tm_post = tm + gain * innovation
            var_post = max((1.0 - gain * p.a12) * var, 1e-12)
            tm = p.a21 * tin[k] + p.a22 * tm_post
            var = p.a22 * p.a22 * var_post + q

    horizon = max(1, int(rollout_horizon))
    stride = max(1, int(rollout_stride))
    for start, end in segments:
        for window_start in range(start, end, stride):
            tin_state = float(tin[window_start])
            tm_state = float(tm_filtered[window_start]) if np.isfinite(tm_filtered[window_start]) else tin_state
            for k in range(window_start, min(end, window_start + horizon)):
                tin_pred, tm_next = model.predict_next(tin_state, tm_state, tout[k], solar[k], u[k])
                horizon_err.append(float(tin_pred - tin_next[k]))
                tin_state = tin_pred
                tm_state = tm_next

    one_step = np.asarray(one_step_err, dtype=float)
    horizon_values = np.asarray(horizon_err, dtype=float)

    def _finite(values: np.ndarray) -> np.ndarray:
        return values[np.isfinite(values)]

    def _rmse(values: np.ndarray) -> float:
        values = _finite(values)
        return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")

    def _mae(values: np.ndarray) -> float:
        values = _finite(values)
        return float(np.mean(np.abs(values))) if values.size else float("nan")

    def _bias(values: np.ndarray) -> float:
        values = _finite(values)
        return float(np.mean(values)) if values.size else float("nan")

    return {
        "kalman_one_step_rmse_c": _rmse(one_step),
        "kalman_one_step_mae_c": _mae(one_step),
        "kalman_one_step_bias_c": _bias(one_step),
        "kalman_horizon_24h_rmse_c": _rmse(horizon_values),
        "kalman_horizon_24h_mae_c": _mae(horizon_values),
        "kalman_horizon_24h_bias_c": _bias(horizon_values),
    }


def save_rc_model(model: RC1R1CModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.params.to_dict(), indent=2) + "\n")


def load_rc_model(path: str | Path) -> RC1R1CModel:
    data = json.loads(Path(path).read_text())
    return RC1R1CModel(RC1R1CParams(**data))
