"""Gnu-RL style differentiable MPC controller.

This module is the clean home for the `PaperDiffMPC` model currently developed
in `notebooks/citylearn_rlmpc_gnurllike_sac.ipynb`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DiffMPCConfig:
    horizon: int = 24
    eta: float = 10.0
    rho_u: float = 1.0
    dt: float = 1.0
    action_lower_bound: float = 0.0
    action_upper_bound: float = 1.0
    cop_lower_bound: float = 0.5
    cop_upper_bound: float = 6.0
    q_reg: float = 1e-4
    qp_max_iter: int = 200
    qp_eps: float = 1e-6

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class PaperDiffMPC:
    """Differentiable QP-MPC model from the Gnu-RL notebook.

    The implementation intentionally imports `torch` and `qpth` lazily so the
    project structure can be imported in lightweight environments. Instantiate
    this class only in an environment with the RLMPC dependencies installed.
    """

    def __new__(cls, *args, **kwargs):
        try:
            import torch
            import torch.nn as nn
            from qpth.qp import QPFunction
        except ImportError as exc:
            raise ImportError(
                "PaperDiffMPC requires torch and qpth. Activate the notebook "
                "environment or install the RLMPC dependencies first."
            ) from exc

        class _PaperDiffMPC(nn.Module):
            def __init__(
                self,
                config: DiffMPCConfig | None = None,
                *,
                T=None,
                eta=None,
                rho_u=None,
                dt=None,
                u_min=None,
                u_max=None,
                cop_min=None,
                cop_max=None,
                q_reg=None,
                qp_max_iter=None,
                qp_eps=None,
                device=None,
            ):
                super().__init__()
                cfg = config or DiffMPCConfig()
                self.T = int(cfg.horizon if T is None else T)
                self.eta = float(cfg.eta if eta is None else eta)
                self.rho_u = float(cfg.rho_u if rho_u is None else rho_u)
                self.dt = float(cfg.dt if dt is None else dt)
                self.u_min = float(cfg.action_lower_bound if u_min is None else u_min)
                self.u_max = float(cfg.action_upper_bound if u_max is None else u_max)
                self.cop_min = float(cfg.cop_lower_bound if cop_min is None else cop_min)
                self.cop_max = float(cfg.cop_upper_bound if cop_max is None else cop_max)
                self.q_reg = float(cfg.q_reg if q_reg is None else q_reg)
                max_iter = cfg.qp_max_iter if qp_max_iter is None else qp_max_iter
                eps = cfg.qp_eps if qp_eps is None else qp_eps

                self.nz = 2 * self.T
                self.qp = QPFunction(verbose=False, eps=eps, maxIter=max_iter)
                self.softplus = nn.Softplus()
                self.cost_mode = "tracking"
                self.balanced_comfort_weight = 0.5
                self.balanced_energy_weight = 1.0
                self.balanced_peak_weight = 3.0
                self.balanced_ramp_weight = 1.0
                self.balanced_shift_weight = 0.0
                self.balanced_preheat_weight = 0.0
                self.balanced_comfort_band = 2.0
                self.balanced_comfort_scale = 4.0
                self.balanced_comfort_mode = "band"
                self.balanced_comfort_track_weight = 3.0
                self.balanced_comfort_cold_weight = 60.0
                self.balanced_comfort_hot_weight = 15.0
                self.balanced_adaptive_weights = False
                self.balanced_comfort_cold_tolerance = 0.5
                self.balanced_comfort_hot_tolerance = 1.0
                self.balanced_comfort_track_scale = 2.0
                self.balanced_comfort_cold_scale = 1.5
                self.balanced_comfort_hot_scale = 2.0
                self.balanced_time_bias_guard_band = 2.0
                self.balanced_power_scale = 5.0
                self.balanced_power_per_unit_action = 1.0
                self.balanced_x_std = 1.0
                self.balanced_normalize_terms = False
                self.district_target_weight = 0.0
                self.district_target_scale = 1.0
                self.district_target_mode = "tracking"

                self.C_raw = nn.Parameter(torch.tensor(1.0))
                self.Rm_raw = nn.Parameter(torch.tensor(1.0))
                self.Rout_raw = nn.Parameter(torch.tensor(1.0))
                self.Aeff_raw = nn.Parameter(torch.tensor(0.1))
                self.Pnom_raw = nn.Parameter(torch.tensor(1.0))
                self.Tm = nn.Parameter(torch.tensor(0.0))
                self.cop_a = nn.Parameter(torch.tensor(1.5))
                self.cop_b = nn.Parameter(torch.tensor(0.03))

                eta_init = torch.tensor(max(self.eta, 1e-4), dtype=torch.float32)
                rho_u_init = torch.tensor(max(self.rho_u, 1e-4), dtype=torch.float32)
                self.q_track_raw = nn.Parameter(torch.log(torch.expm1(eta_init)))
                self.r_u_raw = nn.Parameter(torch.log(torch.expm1(rho_u_init)))
                self.balanced_weight_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
                # Kept only so older checkpoints with this key still load. The
                # controller should track the provided setpoint/T_dyn directly.
                self.sp_bias_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

                if device is not None:
                    self.to(device)

            def _pos(self, x):
                return self.softplus(x) + 1e-6

            def q_track(self):
                return self.softplus(self.q_track_raw) + 1e-4

            def r_u(self):
                return self.softplus(self.r_u_raw) + 1e-4

            def sp_bias(self):
                return torch.zeros((), device=self.sp_bias_raw.device, dtype=self.sp_bias_raw.dtype)

            def use_balanced_mpc_cost(
                self,
                *,
                comfort_weight=0.5,
                energy_weight=1.0,
                peak_weight=3.0,
                ramp_weight=1.0,
                shift_weight=0.0,
                preheat_weight=0.0,
                comfort_band=2.0,
                comfort_scale=4.0,
                comfort_mode="band",
                comfort_track_weight=3.0,
                comfort_cold_weight=60.0,
                comfort_hot_weight=15.0,
                adaptive_weights=False,
                comfort_cold_tolerance=0.5,
                comfort_hot_tolerance=1.0,
                comfort_track_scale=2.0,
                comfort_cold_scale=1.5,
                comfort_hot_scale=2.0,
                time_bias_guard_band=2.0,
                time_bias_guard_penalty=0.0,
                power_scale=5.0,
                power_per_unit_action=1.0,
                x_std=1.0,
                normalize_terms=False,
                district_target_weight=0.0,
                district_target_scale=1.0,
                district_target_mode="tracking",
            ):
                """Use the same stage-cost structure as the explicit MPC script.

                The QP state is normalized temperature, so the comfort band and
                slack cost are scaled back to physical degrees Celsius.
                """

                self.cost_mode = "balanced_mpc"
                self.balanced_comfort_weight = float(comfort_weight)
                self.balanced_energy_weight = float(energy_weight)
                self.balanced_peak_weight = float(peak_weight)
                self.balanced_ramp_weight = float(ramp_weight)
                self.balanced_shift_weight = float(shift_weight)
                self.balanced_preheat_weight = float(preheat_weight)
                self.balanced_comfort_band = float(comfort_band)
                self.balanced_comfort_scale = max(float(comfort_scale), 1e-6)
                self.balanced_comfort_mode = str(comfort_mode).strip().lower()
                if self.balanced_comfort_mode not in {"band", "setpoint", "asymmetric", "barrier", "time_bias_guard"}:
                    raise ValueError(
                        "comfort_mode must be 'band', 'setpoint', 'asymmetric', 'barrier', or 'time_bias_guard'."
                    )
                # The QP can only express quadratic costs, so 'barrier' is implemented
                # as asymmetric slack variables.  The exponential barrier only appears
                # in the BalancedComfortEnergyReward training signal, which drives the
                # learned cold_weight up until cold violations become negligible.
                if self.balanced_comfort_mode == "barrier":
                    self.balanced_comfort_mode = "asymmetric"
                self.balanced_comfort_track_weight = float(comfort_track_weight)
                self.balanced_comfort_cold_weight = float(comfort_cold_weight)
                self.balanced_comfort_hot_weight = float(comfort_hot_weight)
                self.balanced_adaptive_weights = bool(adaptive_weights)
                initial = torch.tensor(
                    [
                        max(float(comfort_track_weight), 1e-8),
                        max(float(comfort_cold_weight), 1e-8),
                        max(float(comfort_hot_weight), 1e-8),
                    ],
                    dtype=self.balanced_weight_logits.dtype,
                    device=self.balanced_weight_logits.device,
                )
                initial = initial / torch.clamp(initial.sum(), min=1e-8)
                with torch.no_grad():
                    self.balanced_weight_logits.copy_(torch.log(initial))
                self.balanced_weight_logits.requires_grad = self.balanced_adaptive_weights
                self.balanced_comfort_cold_tolerance = max(float(comfort_cold_tolerance), 0.0)
                self.balanced_comfort_hot_tolerance = max(float(comfort_hot_tolerance), 0.0)
                self.balanced_comfort_track_scale = max(float(comfort_track_scale), 1e-6)
                self.balanced_comfort_cold_scale = max(float(comfort_cold_scale), 1e-6)
                self.balanced_comfort_hot_scale = max(float(comfort_hot_scale), 1e-6)
                self.balanced_time_bias_guard_band = max(float(time_bias_guard_band), 0.0)
                self.balanced_time_bias_guard_penalty = max(float(time_bias_guard_penalty), 0.0)
                self.balanced_power_scale = max(float(power_scale), 1e-6)
                self.balanced_power_per_unit_action = float(power_per_unit_action)
                self.balanced_x_std = max(float(x_std), 1e-6)
                self.balanced_normalize_terms = bool(normalize_terms)
                self.district_target_weight = float(district_target_weight)
                self.district_target_scale = max(float(district_target_scale), 1e-6)
                self.district_target_mode = str(district_target_mode).strip().lower()
                if self.district_target_mode not in {"tracking", "peak_only"}:
                    raise ValueError("district_target_mode must be 'tracking' or 'peak_only'.")
                return self

            def balanced_weight_vector(self):
                import torch

                if getattr(self, "balanced_adaptive_weights", False):
                    return torch.softmax(self.balanced_weight_logits, dim=0)
                return torch.as_tensor(
                    [
                        self.balanced_comfort_track_weight,
                        self.balanced_comfort_cold_weight,
                        self.balanced_comfort_hot_weight,
                    ],
                    device=self.balanced_weight_logits.device,
                    dtype=self.balanced_weight_logits.dtype,
                )

            def _cop(self, tout):
                return torch.clamp(self.cop_a + self.cop_b * tout, self.cop_min, self.cop_max)

            def _zoh(self, ac, dt):
                a = torch.exp(ac * dt)
                eps = 1e-8
                phi = torch.where(torch.abs(ac) > eps, (a - 1.0) / ac, torch.full_like(ac, dt))
                return a, phi

            def _build_tracking_qp(self, x0, d_seq, sp_seq, w_seq, q_mult_seq=None):
                dev = x0.device
                dtype = x0.dtype
                batch_size = x0.shape[0]
                horizon = self.T
                nz = self.nz

                c = self._pos(self.C_raw)
                rm = self._pos(self.Rm_raw)
                rout = self._pos(self.Rout_raw)
                aeff = self._pos(self.Aeff_raw)
                pnom = self._pos(self.Pnom_raw)

                ac = -(1.0 / (rm * c) + 1.0 / (rout * c))
                a, phi = self._zoh(ac, self.dt)

                tout = d_seq[:, :, 0]
                isol = d_seq[:, :, 1]
                tm = self.Tm.view(1, 1).repeat(batch_size, horizon)

                bd0 = phi * (1.0 / (rm * c))
                bd1 = phi * (1.0 / (rout * c))
                bd2 = phi * (aeff / c)
                bu = phi * (self._cop(tout) * pnom / c)
                rhs = bd0 * tm + bd1 * tout + bd2 * isol + w_seq.squeeze(-1)

                q = self.q_track()
                r = self.r_u()
                sp_ref = sp_seq.squeeze(-1)
                if q_mult_seq is None:
                    q_mult = torch.ones(batch_size, horizon, device=dev, dtype=dtype)
                else:
                    if q_mult_seq.ndim == 2:
                        q_mult = q_mult_seq
                    else:
                        q_mult = q_mult_seq.squeeze(-1)
                    if q_mult.shape[0] == 1 and batch_size > 1:
                        q_mult = q_mult.repeat(batch_size, 1)
                    q_mult = torch.clamp(q_mult.to(device=dev, dtype=dtype), min=0.0)
                q_weights = q * q_mult

                q_mat = torch.zeros(batch_size, nz, nz, device=dev, dtype=dtype)
                q_mat[:, 0:horizon, 0:horizon] = torch.diag_embed(q_weights)
                q_mat[:, horizon : 2 * horizon, horizon : 2 * horizon] = (
                    torch.eye(horizon, device=dev, dtype=dtype) * r
                )
                q_mat = q_mat + self.q_reg * torch.eye(nz, device=dev, dtype=dtype).unsqueeze(0)

                p_vec = torch.zeros(batch_size, nz, device=dev, dtype=dtype)
                p_vec[:, 0:horizon] = -q_weights * sp_ref

                aeq = torch.zeros(batch_size, horizon, nz, device=dev, dtype=dtype)
                beq = torch.zeros(batch_size, horizon, device=dev, dtype=dtype)
                for t in range(horizon):
                    aeq[:, t, t] = 1.0
                    aeq[:, t, horizon + t] = -bu[:, t]
                    if t == 0:
                        beq[:, t] = a * x0.squeeze(-1) + rhs[:, t]
                    else:
                        aeq[:, t, t - 1] = -a
                        beq[:, t] = rhs[:, t]

                g = torch.zeros(batch_size, 2 * horizon, nz, device=dev, dtype=dtype)
                h = torch.zeros(batch_size, 2 * horizon, device=dev, dtype=dtype)
                for t in range(horizon):
                    g[:, t, horizon + t] = 1.0
                    h[:, t] = self.u_max
                    g[:, horizon + t, horizon + t] = -1.0
                    h[:, horizon + t] = -self.u_min

                return q_mat, p_vec, g, h, aeq, beq

            def _build_balanced_mpc_qp(
                self,
                x0,
                d_seq,
                sp_seq,
                w_seq,
                u_prev=None,
                stress_seq=None,
                district_target_seq=None,
                district_background_seq=None,
                district_action_power_seq=None,
                future_stress_seq=None,
                setpoint_bias_seq=None,
            ):
                dev = x0.device
                dtype = x0.dtype
                batch_size = x0.shape[0]
                horizon = self.T
                comfort_mode = self.balanced_comfort_mode
                setpoint_comfort = comfort_mode == "setpoint"
                asymmetric_comfort = comfort_mode == "asymmetric"
                time_bias_comfort = comfort_mode == "time_bias_guard"
                weighted_slack_comfort = comfort_mode in {"asymmetric", "time_bias_guard"}
                slack_comfort = comfort_mode in {"band", "asymmetric", "time_bias_guard"}
                has_district_profiles = (
                    float(self.district_target_weight) != 0.0
                    and district_target_seq is not None
                    and district_background_seq is not None
                    and district_action_power_seq is not None
                )
                target_peak_only = has_district_profiles and self.district_target_mode == "peak_only"

                x_slice = slice(0, horizon)
                u_slice = slice(horizon, 2 * horizon)
                var_end = 2 * horizon
                cold_slice = hot_slice = target_slice = None
                if slack_comfort:
                    cold_slice = slice(var_end, var_end + horizon)
                    var_end += horizon
                    hot_slice = slice(var_end, var_end + horizon)
                    var_end += horizon
                if target_peak_only:
                    target_slice = slice(var_end, var_end + horizon)
                    var_end += horizon
                nz = var_end

                c = self._pos(self.C_raw)
                rm = self._pos(self.Rm_raw)
                rout = self._pos(self.Rout_raw)
                aeff = self._pos(self.Aeff_raw)
                pnom = self._pos(self.Pnom_raw)

                ac = -(1.0 / (rm * c) + 1.0 / (rout * c))
                a, phi = self._zoh(ac, self.dt)

                tout = d_seq[:, :, 0]
                isol = d_seq[:, :, 1]
                tm = self.Tm.view(1, 1).repeat(batch_size, horizon)

                bd0 = phi * (1.0 / (rm * c))
                bd1 = phi * (1.0 / (rout * c))
                bd2 = phi * (aeff / c)
                bu = phi * (self._cop(tout) * pnom / c)
                rhs = bd0 * tm + bd1 * tout + bd2 * isol + w_seq.squeeze(-1)

                q_mat = torch.zeros(batch_size, nz, nz, device=dev, dtype=dtype)
                p_vec = torch.zeros(batch_size, nz, device=dev, dtype=dtype)

                x_std = torch.as_tensor(self.balanced_x_std, device=dev, dtype=dtype)
                comfort_weight = torch.as_tensor(self.balanced_comfort_weight, device=dev, dtype=dtype)
                energy_weight = torch.as_tensor(self.balanced_energy_weight, device=dev, dtype=dtype)
                peak_weight = torch.as_tensor(self.balanced_peak_weight, device=dev, dtype=dtype)
                ramp_weight = torch.as_tensor(self.balanced_ramp_weight, device=dev, dtype=dtype)
                shift_weight = torch.as_tensor(self.balanced_shift_weight, device=dev, dtype=dtype)
                preheat_weight = torch.as_tensor(self.balanced_preheat_weight, device=dev, dtype=dtype)
                target_weight = torch.as_tensor(self.district_target_weight, device=dev, dtype=dtype)
                target_scale = torch.as_tensor(self.district_target_scale, device=dev, dtype=dtype)
                power_coeff = torch.as_tensor(
                    1.0
                    if self.balanced_normalize_terms
                    else self.balanced_power_per_unit_action / self.balanced_power_scale,
                    device=dev,
                    dtype=dtype,
                )

                sp_ref = sp_seq.squeeze(-1)
                if setpoint_bias_seq is None:
                    setpoint_bias = torch.zeros(batch_size, horizon, device=dev, dtype=dtype)
                else:
                    if setpoint_bias_seq.ndim == 1:
                        setpoint_bias = setpoint_bias_seq.view(1, horizon).repeat(batch_size, 1)
                    elif setpoint_bias_seq.ndim == 2:
                        setpoint_bias = setpoint_bias_seq
                    else:
                        setpoint_bias = setpoint_bias_seq.squeeze(-1)
                    if setpoint_bias.shape[0] == 1 and batch_size > 1:
                        setpoint_bias = setpoint_bias.repeat(batch_size, 1)
                    setpoint_bias = setpoint_bias.to(device=dev, dtype=dtype)
                track_ref = sp_ref + setpoint_bias

                # qpth solves 0.5 z'Qz + p'z. Use 2*w on the diagonal to get w*z^2.
                if setpoint_comfort:
                    track_coeff = comfort_weight * x_std * x_std / self.balanced_comfort_scale
                    eye_h = torch.eye(horizon, device=dev, dtype=dtype)
                    q_mat[:, x_slice, x_slice] = q_mat[:, x_slice, x_slice] + eye_h * (2.0 * track_coeff)
                    p_vec[:, x_slice] = p_vec[:, x_slice] - 2.0 * track_coeff * sp_ref
                elif weighted_slack_comfort:
                    track_scale = torch.as_tensor(self.balanced_comfort_track_scale, device=dev, dtype=dtype)
                    if getattr(self, "balanced_adaptive_weights", False):
                        alpha = torch.softmax(self.balanced_weight_logits.to(device=dev, dtype=dtype), dim=0)
                        track_weight, cold_weight, hot_weight = alpha[0], alpha[1], alpha[2]
                    else:
                        track_weight = torch.as_tensor(self.balanced_comfort_track_weight, device=dev, dtype=dtype)
                        cold_weight = torch.as_tensor(self.balanced_comfort_cold_weight, device=dev, dtype=dtype)
                        hot_weight = torch.as_tensor(self.balanced_comfort_hot_weight, device=dev, dtype=dtype)
                    cold_scale = torch.as_tensor(self.balanced_comfort_cold_scale, device=dev, dtype=dtype)
                    hot_scale = torch.as_tensor(self.balanced_comfort_hot_scale, device=dev, dtype=dtype)
                    track_coeff = comfort_weight * track_weight * x_std * x_std / (track_scale * track_scale)
                    eye_h = torch.eye(horizon, device=dev, dtype=dtype)
                    q_mat[:, x_slice, x_slice] = q_mat[:, x_slice, x_slice] + eye_h * (2.0 * track_coeff)
                    p_vec[:, x_slice] = p_vec[:, x_slice] - 2.0 * track_coeff * track_ref
                    assert cold_slice is not None and hot_slice is not None
                    cold_diag = 2.0 * comfort_weight * cold_weight * x_std * x_std / (cold_scale * cold_scale)
                    hot_diag = 2.0 * comfort_weight * hot_weight * x_std * x_std / (hot_scale * hot_scale)
                    q_mat[:, cold_slice, cold_slice] = torch.eye(horizon, device=dev, dtype=dtype) * cold_diag
                    q_mat[:, hot_slice, hot_slice] = torch.eye(horizon, device=dev, dtype=dtype) * hot_diag
                    if time_bias_comfort and self.balanced_time_bias_guard_penalty > 0.0:
                        guard_penalty = torch.as_tensor(
                            self.balanced_time_bias_guard_penalty * self.balanced_x_std,
                            device=dev,
                            dtype=dtype,
                        )
                        p_vec[:, cold_slice] = p_vec[:, cold_slice] + guard_penalty
                        p_vec[:, hot_slice] = p_vec[:, hot_slice] + guard_penalty
                else:
                    slack_diag = 2.0 * comfort_weight * x_std * x_std / self.balanced_comfort_scale
                    assert cold_slice is not None and hot_slice is not None
                    q_mat[:, cold_slice, cold_slice] = torch.eye(horizon, device=dev, dtype=dtype) * slack_diag
                    q_mat[:, hot_slice, hot_slice] = torch.eye(horizon, device=dev, dtype=dtype) * slack_diag

                u_diag = 2.0 * peak_weight * power_coeff * power_coeff
                q_mat[:, u_slice, u_slice] = torch.eye(horizon, device=dev, dtype=dtype) * u_diag
                p_vec[:, u_slice] = energy_weight * power_coeff
                if stress_seq is None:
                    stress = torch.zeros(batch_size, horizon, device=dev, dtype=dtype)
                else:
                    if stress_seq.ndim == 1:
                        stress = stress_seq.view(1, horizon).repeat(batch_size, 1)
                    elif stress_seq.ndim == 2:
                        stress = stress_seq
                    else:
                        stress = stress_seq.squeeze(-1)
                    if stress.shape[0] == 1 and batch_size > 1:
                        stress = stress.repeat(batch_size, 1)
                    stress = stress.to(device=dev, dtype=dtype)
                p_vec[:, u_slice] = p_vec[:, u_slice] + shift_weight * stress * power_coeff
                if future_stress_seq is None:
                    future_stress = torch.zeros(batch_size, horizon, device=dev, dtype=dtype)
                else:
                    if future_stress_seq.ndim == 1:
                        future_stress = future_stress_seq.view(1, horizon).repeat(batch_size, 1)
                    elif future_stress_seq.ndim == 2:
                        future_stress = future_stress_seq
                    else:
                        future_stress = future_stress_seq.squeeze(-1)
                    if future_stress.shape[0] == 1 and batch_size > 1:
                        future_stress = future_stress.repeat(batch_size, 1)
                    future_stress = torch.clamp(future_stress.to(device=dev, dtype=dtype), min=0.0)
                preheat_signal = torch.clamp(1.0 - stress, min=0.0) * future_stress
                p_vec[:, u_slice] = p_vec[:, u_slice] - preheat_weight * preheat_signal * power_coeff

                district_target = district_background = district_action_power = None
                if has_district_profiles:
                    def _profile_tensor(values):
                        if values.ndim == 1:
                            out = values.view(1, horizon).repeat(batch_size, 1)
                        elif values.ndim == 2:
                            out = values
                        else:
                            out = values.squeeze(-1)
                        if out.shape[0] == 1 and batch_size > 1:
                            out = out.repeat(batch_size, 1)
                        return out.to(device=dev, dtype=dtype)

                    district_target = _profile_tensor(district_target_seq)
                    district_background = _profile_tensor(district_background_seq)
                    district_action_power = torch.clamp(_profile_tensor(district_action_power_seq), min=0.0)
                    scale_sq = target_scale * target_scale
                    if target_peak_only:
                        assert target_slice is not None
                        q_mat[:, target_slice, target_slice] = (
                            torch.eye(horizon, device=dev, dtype=dtype) * (2.0 * target_weight / scale_sq)
                        )
                    else:
                        q_mat[:, u_slice, u_slice] = q_mat[:, u_slice, u_slice] + torch.diag_embed(
                            2.0 * target_weight * district_action_power * district_action_power / scale_sq
                        )
                        p_vec[:, u_slice] = p_vec[:, u_slice] + (
                            2.0
                            * target_weight
                            * district_action_power
                            * (district_background - district_target)
                            / scale_sq
                        )

                if u_prev is None:
                    u_prev_vec = torch.zeros(batch_size, device=dev, dtype=dtype)
                else:
                    if not torch.is_tensor(u_prev):
                        u_prev_vec = torch.as_tensor(u_prev, device=dev, dtype=dtype)
                    else:
                        u_prev_vec = u_prev.to(device=dev, dtype=dtype)
                    u_prev_vec = u_prev_vec.reshape(-1)
                    if u_prev_vec.numel() == 1 and batch_size > 1:
                        u_prev_vec = u_prev_vec.repeat(batch_size)

                for t in range(horizon):
                    u_idx = horizon + t
                    q_mat[:, u_idx, u_idx] = q_mat[:, u_idx, u_idx] + 2.0 * ramp_weight
                    if t == 0:
                        p_vec[:, u_idx] = p_vec[:, u_idx] - 2.0 * ramp_weight * u_prev_vec
                    else:
                        u_prev_idx = horizon + t - 1
                        q_mat[:, u_idx, u_prev_idx] = q_mat[:, u_idx, u_prev_idx] - 2.0 * ramp_weight
                        q_mat[:, u_prev_idx, u_idx] = q_mat[:, u_prev_idx, u_idx] - 2.0 * ramp_weight
                        q_mat[:, u_prev_idx, u_prev_idx] = q_mat[:, u_prev_idx, u_prev_idx] + 2.0 * ramp_weight

                q_mat = q_mat + self.q_reg * torch.eye(nz, device=dev, dtype=dtype).unsqueeze(0)

                aeq = torch.zeros(batch_size, horizon, nz, device=dev, dtype=dtype)
                beq = torch.zeros(batch_size, horizon, device=dev, dtype=dtype)
                for t in range(horizon):
                    aeq[:, t, t] = 1.0
                    aeq[:, t, horizon + t] = -bu[:, t]
                    if t == 0:
                        beq[:, t] = a * x0.squeeze(-1) + rhs[:, t]
                    else:
                        aeq[:, t, t - 1] = -a
                        beq[:, t] = rhs[:, t]

                n_ineq = 2 * horizon
                if slack_comfort:
                    n_ineq += 4 * horizon
                if target_peak_only:
                    n_ineq += 2 * horizon
                g = torch.zeros(batch_size, n_ineq, nz, device=dev, dtype=dtype)
                h = torch.zeros(batch_size, n_ineq, device=dev, dtype=dtype)
                if asymmetric_comfort:
                    cold_tol_norm = torch.as_tensor(
                        self.balanced_comfort_cold_tolerance / self.balanced_x_std,
                        device=dev,
                        dtype=dtype,
                    )
                    hot_tol_norm = torch.as_tensor(
                        self.balanced_comfort_hot_tolerance / self.balanced_x_std,
                        device=dev,
                        dtype=dtype,
                    )
                    lower = sp_ref - cold_tol_norm
                    upper = sp_ref + hot_tol_norm
                elif time_bias_comfort:
                    guard_norm = torch.as_tensor(
                        self.balanced_time_bias_guard_band / self.balanced_x_std,
                        device=dev,
                        dtype=dtype,
                    )
                    lower = sp_ref - guard_norm
                    upper = sp_ref + guard_norm
                else:
                    band_norm = torch.as_tensor(
                        self.balanced_comfort_band / self.balanced_x_std,
                        device=dev,
                        dtype=dtype,
                    )
                    lower = sp_ref - band_norm
                    upper = sp_ref + band_norm

                row = 0
                for t in range(horizon):
                    g[:, row, horizon + t] = 1.0
                    h[:, row] = self.u_max
                    row += 1
                    g[:, row, horizon + t] = -1.0
                    h[:, row] = -self.u_min
                    row += 1
                    if not slack_comfort:
                        continue
                    assert cold_slice is not None and hot_slice is not None
                    g[:, row, cold_slice.start + t] = -1.0
                    h[:, row] = 0.0
                    row += 1
                    g[:, row, hot_slice.start + t] = -1.0
                    h[:, row] = 0.0
                    row += 1
                    # lower - x <= s_cold  ->  -x - s_cold <= -lower
                    g[:, row, t] = -1.0
                    g[:, row, cold_slice.start + t] = -1.0
                    h[:, row] = -lower[:, t]
                    row += 1
                    # x - upper <= s_hot  ->  x - s_hot <= upper
                    g[:, row, t] = 1.0
                    g[:, row, hot_slice.start + t] = -1.0
                    h[:, row] = upper[:, t]
                    row += 1
                    if target_peak_only:
                        assert target_slice is not None
                        assert district_target is not None
                        assert district_background is not None
                        assert district_action_power is not None
                        g[:, row, target_slice.start + t] = -1.0
                        h[:, row] = 0.0
                        row += 1
                        g[:, row, horizon + t] = district_action_power[:, t]
                        g[:, row, target_slice.start + t] = -1.0
                        h[:, row] = district_target[:, t] - district_background[:, t]
                        row += 1

                if target_peak_only and not slack_comfort:
                    assert target_slice is not None
                    assert district_target is not None
                    assert district_background is not None
                    assert district_action_power is not None
                    for t in range(horizon):
                        g[:, row, target_slice.start + t] = -1.0
                        h[:, row] = 0.0
                        row += 1
                        g[:, row, horizon + t] = district_action_power[:, t]
                        g[:, row, target_slice.start + t] = -1.0
                        h[:, row] = district_target[:, t] - district_background[:, t]
                        row += 1

                return q_mat, p_vec, g, h, aeq, beq

            def _build_qp(
                self,
                x0,
                d_seq,
                sp_seq,
                w_seq,
                q_mult_seq=None,
                u_prev=None,
                stress_seq=None,
                district_target_seq=None,
                district_background_seq=None,
                district_action_power_seq=None,
                future_stress_seq=None,
                setpoint_bias_seq=None,
            ):
                if self.cost_mode == "balanced_mpc":
                    return self._build_balanced_mpc_qp(
                        x0,
                        d_seq,
                        sp_seq,
                        w_seq,
                        u_prev=u_prev,
                        stress_seq=stress_seq,
                        district_target_seq=district_target_seq,
                        district_background_seq=district_background_seq,
                        district_action_power_seq=district_action_power_seq,
                        future_stress_seq=future_stress_seq,
                        setpoint_bias_seq=setpoint_bias_seq,
                    )
                return self._build_tracking_qp(x0, d_seq, sp_seq, w_seq, q_mult_seq=q_mult_seq)

            def forward(
                self,
                x0,
                d_seq,
                sp_seq,
                w_seq=None,
                q_mult_seq=None,
                u_prev=None,
                stress_seq=None,
                district_target_seq=None,
                district_background_seq=None,
                district_action_power_seq=None,
                future_stress_seq=None,
                setpoint_bias_seq=None,
            ):
                if x0.ndim == 1:
                    x0 = x0.view(1, 1)

                batch_size = x0.shape[0]
                dev = x0.device
                dtype = x0.dtype

                if d_seq.ndim == 2:
                    d_seq = d_seq.unsqueeze(0).repeat(batch_size, 1, 1)
                if sp_seq.ndim == 2:
                    sp_seq = sp_seq.unsqueeze(0).repeat(batch_size, 1, 1)
                if w_seq is None:
                    w_seq = torch.zeros(batch_size, self.T, 1, device=dev, dtype=dtype)

                q_mat, p_vec, g, h, aeq, beq = self._build_qp(
                    x0,
                    d_seq,
                    sp_seq,
                    w_seq,
                    q_mult_seq=q_mult_seq,
                    u_prev=u_prev,
                    stress_seq=stress_seq,
                    district_target_seq=district_target_seq,
                    district_background_seq=district_background_seq,
                    district_action_power_seq=district_action_power_seq,
                    future_stress_seq=future_stress_seq,
                    setpoint_bias_seq=setpoint_bias_seq,
                )
                z = self.qp(q_mat, p_vec, g, h, aeq, beq)
                x_seq = z[:, : self.T].unsqueeze(-1)
                u_seq = z[:, self.T : 2 * self.T].unsqueeze(-1)
                u0 = torch.clamp(u_seq[:, 0, :], self.u_min, self.u_max)
                return u0, u_seq, x_seq

        instance = _PaperDiffMPC(*args, **kwargs)
        return instance


def freeze_dynamics_parameters(mpc):
    """Freeze learned RC dynamics and leave online cost parameters trainable."""

    for parameter in [
        mpc.C_raw,
        mpc.Rm_raw,
        mpc.Rout_raw,
        mpc.Aeff_raw,
        mpc.Pnom_raw,
        mpc.Tm,
        mpc.cop_a,
        mpc.cop_b,
    ]:
        parameter.requires_grad = False

    adaptive_balanced = (
        getattr(mpc, "cost_mode", "tracking") == "balanced_mpc"
        and bool(getattr(mpc, "balanced_adaptive_weights", False))
    )
    mpc.q_track_raw.requires_grad = getattr(mpc, "cost_mode", "tracking") != "balanced_mpc"
    mpc.r_u_raw.requires_grad = getattr(mpc, "cost_mode", "tracking") != "balanced_mpc"
    if hasattr(mpc, "balanced_weight_logits"):
        mpc.balanced_weight_logits.requires_grad = adaptive_balanced
    mpc.sp_bias_raw.requires_grad = False
    return mpc


def trainable_cost_parameters(mpc):
    """Return current online cost parameters.

    The matched-MPC balanced cost uses fixed MPC weights, so PPO has no inner
    cost parameters to update in that mode.
    """

    if getattr(mpc, "cost_mode", "tracking") == "balanced_mpc":
        if bool(getattr(mpc, "balanced_adaptive_weights", False)) and hasattr(mpc, "balanced_weight_logits"):
            return [mpc.balanced_weight_logits]
        return []
    return [mpc.q_track_raw, mpc.r_u_raw]


def freeze_cost_parameters(mpc):
    """Freeze MPC objective parameters before online dynamics adaptation."""

    mpc.q_track_raw.requires_grad = False
    mpc.r_u_raw.requires_grad = False
    if hasattr(mpc, "balanced_weight_logits"):
        mpc.balanced_weight_logits.requires_grad = False
    mpc.sp_bias_raw.requires_grad = False
    return mpc


def trainable_dynamics_parameters(mpc):
    """Return the differentiable RC dynamics parameters for online adaptation."""

    parameters = [
        mpc.C_raw,
        mpc.Rm_raw,
        mpc.Rout_raw,
        mpc.Aeff_raw,
        mpc.Pnom_raw,
        mpc.Tm,
        mpc.cop_a,
        mpc.cop_b,
    ]
    for parameter in parameters:
        parameter.requires_grad = True
    return parameters


def diffmpc_param_snapshot(mpc) -> dict[str, float]:
    """Return report-friendly parameter values."""

    weights = None
    if hasattr(mpc, "balanced_weight_logits"):
        try:
            weights = mpc.balanced_weight_vector().detach().cpu()
        except Exception:
            weights = None
    return {
        "C": float(mpc._pos(mpc.C_raw).detach().cpu()),
        "Rm": float(mpc._pos(mpc.Rm_raw).detach().cpu()),
        "Rout": float(mpc._pos(mpc.Rout_raw).detach().cpu()),
        "Aeff": float(mpc._pos(mpc.Aeff_raw).detach().cpu()),
        "Pnom": float(mpc._pos(mpc.Pnom_raw).detach().cpu()),
        "Tm": float(mpc.Tm.detach().cpu()),
        "cop_a": float(mpc.cop_a.detach().cpu()),
        "cop_b": float(mpc.cop_b.detach().cpu()),
        "q_track": float(mpc.q_track().detach().cpu()),
        "r_u": float(mpc.r_u().detach().cpu()),
        "sp_bias": float(mpc.sp_bias().detach().cpu()),
        "cost_mode": getattr(mpc, "cost_mode", "tracking"),
        "balanced_comfort_weight": float(getattr(mpc, "balanced_comfort_weight", float("nan"))),
        "balanced_energy_weight": float(getattr(mpc, "balanced_energy_weight", float("nan"))),
        "balanced_peak_weight": float(getattr(mpc, "balanced_peak_weight", float("nan"))),
        "balanced_ramp_weight": float(getattr(mpc, "balanced_ramp_weight", float("nan"))),
        "balanced_shift_weight": float(getattr(mpc, "balanced_shift_weight", float("nan"))),
        "balanced_preheat_weight": float(getattr(mpc, "balanced_preheat_weight", float("nan"))),
        "balanced_comfort_band": float(getattr(mpc, "balanced_comfort_band", float("nan"))),
        "balanced_comfort_scale": float(getattr(mpc, "balanced_comfort_scale", float("nan"))),
        "balanced_comfort_mode": getattr(mpc, "balanced_comfort_mode", "band"),
        "balanced_comfort_track_weight": float(getattr(mpc, "balanced_comfort_track_weight", float("nan"))),
        "balanced_comfort_cold_weight": float(getattr(mpc, "balanced_comfort_cold_weight", float("nan"))),
        "balanced_comfort_hot_weight": float(getattr(mpc, "balanced_comfort_hot_weight", float("nan"))),
        "balanced_adaptive_weights": bool(getattr(mpc, "balanced_adaptive_weights", False)),
        "alpha_track": float(weights[0]) if weights is not None else float("nan"),
        "alpha_cold": float(weights[1]) if weights is not None else float("nan"),
        "alpha_hot": float(weights[2]) if weights is not None else float("nan"),
        "balanced_comfort_cold_tolerance": float(getattr(mpc, "balanced_comfort_cold_tolerance", float("nan"))),
        "balanced_comfort_hot_tolerance": float(getattr(mpc, "balanced_comfort_hot_tolerance", float("nan"))),
        "balanced_comfort_track_scale": float(getattr(mpc, "balanced_comfort_track_scale", float("nan"))),
        "balanced_comfort_cold_scale": float(getattr(mpc, "balanced_comfort_cold_scale", float("nan"))),
        "balanced_comfort_hot_scale": float(getattr(mpc, "balanced_comfort_hot_scale", float("nan"))),
        "balanced_power_scale": float(getattr(mpc, "balanced_power_scale", float("nan"))),
        "balanced_power_per_unit_action": float(getattr(mpc, "balanced_power_per_unit_action", float("nan"))),
        "balanced_normalize_terms": bool(getattr(mpc, "balanced_normalize_terms", False)),
        "district_target_weight": float(getattr(mpc, "district_target_weight", float("nan"))),
        "district_target_scale": float(getattr(mpc, "district_target_scale", float("nan"))),
        "district_target_mode": getattr(mpc, "district_target_mode", "tracking"),
    }


class TwoStateDiffMPC:
    """Differentiable 2-state QP-MPC using fixed 2R2C thermal parameters.

    Dynamics (physical units, °C):
        Tin(t+1) = a11*Tin + a12*Tm + b_tout*Tout + b_sol*Sol + b_heat*COP(Tout)*u + b0
        Tm(t+1)  = a21*Tin + a22*Tm

    QP variables: z = [Tin_1..Tin_T, Tm_1..Tm_T, u_0..u_{T-1}]  (nz = 3*T)
    Cost: sum_t [ q_track*(Tin_t - sp_t)² + r_u*u_t² + energy_w*gs_t*u_t ]
    Subject to: dynamics equalities, 0 <= u_t <= 1.

    Only q_track_raw and r_u_raw are learnable; all dynamics parameters are fixed.
    The mass temperature Tm is maintained as an internal running buffer and must
    be reset at the start of each episode.
    """

    def __new__(cls, *args, **kwargs):
        try:
            import torch
            import torch.nn as nn
            from qpth.qp import QPFunction
        except ImportError as exc:
            raise ImportError(
                "TwoStateDiffMPC requires torch and qpth."
            ) from exc

        class _TwoStateDiffMPC(nn.Module):
            def __init__(
                self,
                *,
                a11: float,
                a12: float,
                a21: float,
                a22: float,
                b_tout: float,
                b_sol: float,
                b_heat: float,
                b0: float,
                cop_a: float,
                cop_b: float,
                T: int = 6,
                u_min: float = 0.0,
                u_max: float = 1.0,
                q_reg: float = 1e-4,
                qp_max_iter: int = 200,
                qp_eps: float = 1e-6,
                eta: float = 10.0,
                rho_u: float = 1.0,
                tm_penalty_weight: float = 0.0,
            ):
                super().__init__()
                self.T = int(T)
                self.u_min = float(u_min)
                self.u_max = float(u_max)
                self.q_reg = float(q_reg)
                self.cop_min = 0.5
                self.cop_max = 6.0

                # Fixed dynamics (not learnable)
                self.register_buffer("a11", torch.tensor(float(a11)))
                self.register_buffer("a12", torch.tensor(float(a12)))
                self.register_buffer("a21", torch.tensor(float(a21)))
                self.register_buffer("a22", torch.tensor(float(a22)))
                self.register_buffer("b_tout", torch.tensor(float(b_tout)))
                self.register_buffer("b_sol", torch.tensor(float(b_sol)))
                self.register_buffer("b_heat", torch.tensor(float(b_heat)))
                self.register_buffer("b0", torch.tensor(float(b0)))
                self.register_buffer("_cop_a", torch.tensor(float(cop_a)))
                self.register_buffer("_cop_b", torch.tensor(float(cop_b)))

                # Running mass temperature (not a parameter, reset per episode)
                self.register_buffer("tm_buffer", torch.tensor(20.0))

                # Learnable cost weights
                softplus = nn.Softplus()
                eta_t = torch.tensor(max(float(eta), 1e-4))
                rho_t = torch.tensor(max(float(rho_u), 1e-4))
                self.q_track_raw = nn.Parameter(torch.log(torch.expm1(eta_t)))
                self.r_u_raw = nn.Parameter(torch.log(torch.expm1(rho_t)))
                self.softplus = softplus
                self.tm_penalty_weight = float(tm_penalty_weight)

                self.qp = QPFunction(verbose=False, eps=qp_eps, maxIter=qp_max_iter)
                self.cost_mode = "tracking"
                self.balanced_comfort_weight = 0.5
                self.balanced_energy_weight = 1.0
                self.balanced_peak_weight = 0.0
                self.balanced_ramp_weight = 0.0
                self.balanced_shift_weight = 0.0
                self.balanced_comfort_band = 2.0
                self.balanced_comfort_scale = 4.0
                self.balanced_comfort_mode = "time_bias_guard"
                self.balanced_comfort_track_weight = 3.0
                self.balanced_comfort_cold_weight = 60.0
                self.balanced_comfort_hot_weight = 15.0
                self.balanced_comfort_cold_tolerance = 0.5
                self.balanced_comfort_hot_tolerance = 1.0
                self.balanced_comfort_track_scale = 2.0
                self.balanced_comfort_cold_scale = 1.5
                self.balanced_comfort_hot_scale = 2.0
                self.balanced_time_bias_guard_band = 2.0
                self.balanced_power_scale = 5.0
                self.balanced_power_per_unit_action = 1.0
                self.balanced_normalize_terms = False

            def _pos(self, x):
                return self.softplus(x) + 1e-6

            def q_track(self):
                return self._pos(self.q_track_raw)

            def r_u(self):
                return self._pos(self.r_u_raw)

            def use_balanced_mpc_cost(
                self,
                *,
                comfort_weight=0.5,
                energy_weight=0.0,
                peak_weight=0.0,
                ramp_weight=0.0,
                shift_weight=0.5,
                preheat_weight=0.0,
                district_target_weight=0.0,
                district_target_scale=1.0,
                district_target_mode="tracking",
                comfort_band=2.0,
                comfort_scale=1.0,
                comfort_mode="time_bias_guard",
                comfort_track_weight=3.0,
                comfort_cold_weight=60.0,
                comfort_hot_weight=15.0,
                adaptive_weights=False,
                comfort_cold_tolerance=0.5,
                comfort_hot_tolerance=1.0,
                comfort_track_scale=2.0,
                comfort_cold_scale=1.5,
                comfort_hot_scale=2.0,
                time_bias_guard_band=2.0,
                time_bias_guard_penalty=0.0,
                power_scale=5.0,
                power_per_unit_action=1.0,
                x_std=1.0,
                normalize_terms=False,
            ):
                """Use the same quadratic/slack objective as the explicit MPC stage cost."""

                self.cost_mode = "balanced_mpc"
                self.balanced_comfort_weight = float(comfort_weight)
                self.balanced_energy_weight = float(energy_weight)
                self.balanced_peak_weight = float(peak_weight)
                self.balanced_ramp_weight = float(ramp_weight)
                self.balanced_shift_weight = float(shift_weight)
                self.balanced_comfort_band = float(comfort_band)
                self.balanced_comfort_scale = max(float(comfort_scale), 1e-6)
                self.balanced_comfort_mode = str(comfort_mode).strip().lower()
                if self.balanced_comfort_mode not in {"band", "setpoint", "asymmetric", "barrier", "time_bias_guard"}:
                    raise ValueError(
                        "comfort_mode must be 'band', 'setpoint', 'asymmetric', 'barrier', or 'time_bias_guard'."
                    )
                if self.balanced_comfort_mode == "barrier":
                    self.balanced_comfort_mode = "asymmetric"
                self.balanced_comfort_track_weight = float(comfort_track_weight)
                self.balanced_comfort_cold_weight = float(comfort_cold_weight)
                self.balanced_comfort_hot_weight = float(comfort_hot_weight)
                self.balanced_adaptive_weights = bool(adaptive_weights)
                self.balanced_comfort_cold_tolerance = max(float(comfort_cold_tolerance), 0.0)
                self.balanced_comfort_hot_tolerance = max(float(comfort_hot_tolerance), 0.0)
                self.balanced_comfort_track_scale = max(float(comfort_track_scale), 1e-6)
                self.balanced_comfort_cold_scale = max(float(comfort_cold_scale), 1e-6)
                self.balanced_comfort_hot_scale = max(float(comfort_hot_scale), 1e-6)
                self.balanced_time_bias_guard_band = max(float(time_bias_guard_band), 0.0)
                self.balanced_time_bias_guard_penalty = max(float(time_bias_guard_penalty), 0.0)
                self.balanced_power_scale = max(float(power_scale), 1e-6)
                self.balanced_power_per_unit_action = float(power_per_unit_action)
                self.balanced_normalize_terms = bool(normalize_terms)
                return self

            def _cop(self, tout):
                import torch
                return torch.clamp(self._cop_a + self._cop_b * tout, self.cop_min, self.cop_max)

            def reset_mass_temperature(self, tm_init: float):
                self.tm_buffer.fill_(float(tm_init))

            def update_mass_temperature(self, tin: float, tm: float) -> float:
                """Advance mass temperature one step using 2R2C dynamics."""
                tm_next = float(self.a21) * tin + float(self.a22) * tm
                self.tm_buffer.fill_(tm_next)
                return tm_next

            def _build_balanced_mpc_qp(
                self,
                tin0,
                tm0,
                d_seq,
                sp_seq,
                stress_seq=None,
                setpoint_bias_seq=None,
                w_seq=None,
            ):
                import torch

                dev = tin0.device
                dtype = tin0.dtype
                batch = tin0.shape[0]
                T = self.T
                comfort_mode = self.balanced_comfort_mode
                setpoint_comfort = comfort_mode == "setpoint"
                asymmetric_comfort = comfort_mode == "asymmetric"
                time_bias_comfort = comfort_mode == "time_bias_guard"
                weighted_slack_comfort = comfort_mode in {"asymmetric", "time_bias_guard"}
                slack_comfort = comfort_mode in {"band", "asymmetric", "time_bias_guard"}

                x_slice = slice(0, T)
                y_slice = slice(T, 2 * T)
                u_slice = slice(2 * T, 3 * T)
                var_end = 3 * T
                cold_slice = hot_slice = None
                if slack_comfort:
                    cold_slice = slice(var_end, var_end + T)
                    var_end += T
                    hot_slice = slice(var_end, var_end + T)
                    var_end += T
                nz = var_end

                tout = d_seq[:, :, 0]
                isol = d_seq[:, :, 1]
                cop_seq = self._cop(tout)
                b_heat_eff = self.b_heat * cop_seq
                if w_seq is None:
                    w_est = torch.zeros(batch, T, device=dev, dtype=dtype)
                else:
                    if w_seq.ndim == 1:
                        w_est = w_seq.view(1, T).repeat(batch, 1)
                    elif w_seq.ndim == 2:
                        w_est = w_seq
                    else:
                        w_est = w_seq.squeeze(-1)
                    if w_est.shape[0] == 1 and batch > 1:
                        w_est = w_est.repeat(batch, 1)
                    w_est = w_est.to(device=dev, dtype=dtype)

                Q = torch.zeros(batch, nz, nz, device=dev, dtype=dtype)
                p = torch.zeros(batch, nz, device=dev, dtype=dtype)

                comfort_weight = torch.as_tensor(self.balanced_comfort_weight, device=dev, dtype=dtype)
                energy_weight = torch.as_tensor(self.balanced_energy_weight, device=dev, dtype=dtype)
                peak_weight = torch.as_tensor(self.balanced_peak_weight, device=dev, dtype=dtype)
                ramp_weight = torch.as_tensor(self.balanced_ramp_weight, device=dev, dtype=dtype)
                shift_weight = torch.as_tensor(self.balanced_shift_weight, device=dev, dtype=dtype)
                power_coeff = torch.as_tensor(
                    1.0
                    if self.balanced_normalize_terms
                    else self.balanced_power_per_unit_action / self.balanced_power_scale,
                    device=dev,
                    dtype=dtype,
                )

                sp_ref = sp_seq.squeeze(-1)
                if setpoint_bias_seq is None:
                    setpoint_bias = torch.zeros(batch, T, device=dev, dtype=dtype)
                else:
                    if setpoint_bias_seq.ndim == 1:
                        setpoint_bias = setpoint_bias_seq.view(1, T).repeat(batch, 1)
                    elif setpoint_bias_seq.ndim == 2:
                        setpoint_bias = setpoint_bias_seq
                    else:
                        setpoint_bias = setpoint_bias_seq.squeeze(-1)
                    if setpoint_bias.shape[0] == 1 and batch > 1:
                        setpoint_bias = setpoint_bias.repeat(batch, 1)
                    setpoint_bias = setpoint_bias.to(device=dev, dtype=dtype)
                track_ref = sp_ref + setpoint_bias

                eye_h = torch.eye(T, device=dev, dtype=dtype)
                if setpoint_comfort:
                    track_coeff = comfort_weight / self.balanced_comfort_scale
                    Q[:, x_slice, x_slice] = Q[:, x_slice, x_slice] + eye_h * (2.0 * track_coeff)
                    p[:, x_slice] = p[:, x_slice] - 2.0 * track_coeff * sp_ref
                elif weighted_slack_comfort:
                    track_scale = torch.as_tensor(self.balanced_comfort_track_scale, device=dev, dtype=dtype)
                    track_weight = torch.as_tensor(self.balanced_comfort_track_weight, device=dev, dtype=dtype)
                    cold_weight = torch.as_tensor(self.balanced_comfort_cold_weight, device=dev, dtype=dtype)
                    hot_weight = torch.as_tensor(self.balanced_comfort_hot_weight, device=dev, dtype=dtype)
                    cold_scale = torch.as_tensor(self.balanced_comfort_cold_scale, device=dev, dtype=dtype)
                    hot_scale = torch.as_tensor(self.balanced_comfort_hot_scale, device=dev, dtype=dtype)
                    track_coeff = comfort_weight * track_weight / (track_scale * track_scale)
                    Q[:, x_slice, x_slice] = Q[:, x_slice, x_slice] + eye_h * (2.0 * track_coeff)
                    p[:, x_slice] = p[:, x_slice] - 2.0 * track_coeff * track_ref
                    assert cold_slice is not None and hot_slice is not None
                    Q[:, cold_slice, cold_slice] = eye_h * (
                        2.0 * comfort_weight * cold_weight / (cold_scale * cold_scale)
                    )
                    Q[:, hot_slice, hot_slice] = eye_h * (
                        2.0 * comfort_weight * hot_weight / (hot_scale * hot_scale)
                    )
                    if time_bias_comfort and self.balanced_time_bias_guard_penalty > 0.0:
                        guard_penalty = torch.as_tensor(
                            self.balanced_time_bias_guard_penalty,
                            device=dev,
                            dtype=dtype,
                        )
                        p[:, cold_slice] = p[:, cold_slice] + guard_penalty
                        p[:, hot_slice] = p[:, hot_slice] + guard_penalty
                else:
                    assert cold_slice is not None and hot_slice is not None
                    slack_diag = 2.0 * comfort_weight / self.balanced_comfort_scale
                    Q[:, cold_slice, cold_slice] = eye_h * slack_diag
                    Q[:, hot_slice, hot_slice] = eye_h * slack_diag

                Q[:, u_slice, u_slice] = eye_h * (2.0 * peak_weight * power_coeff * power_coeff)
                p[:, u_slice] = energy_weight * power_coeff
                if stress_seq is not None:
                    if stress_seq.ndim == 1:
                        stress = stress_seq.view(1, T).repeat(batch, 1)
                    elif stress_seq.ndim == 2:
                        stress = stress_seq
                    else:
                        stress = stress_seq.squeeze(-1)
                    if stress.shape[0] == 1 and batch > 1:
                        stress = stress.repeat(batch, 1)
                    p[:, u_slice] = p[:, u_slice] + shift_weight * stress.to(dev, dtype) * power_coeff

                for t in range(T):
                    u_idx = u_slice.start + t
                    Q[:, u_idx, u_idx] = Q[:, u_idx, u_idx] + 2.0 * ramp_weight
                    if t > 0:
                        u_prev_idx = u_slice.start + t - 1
                        Q[:, u_idx, u_prev_idx] = Q[:, u_idx, u_prev_idx] - 2.0 * ramp_weight
                        Q[:, u_prev_idx, u_idx] = Q[:, u_prev_idx, u_idx] - 2.0 * ramp_weight
                        Q[:, u_prev_idx, u_prev_idx] = Q[:, u_prev_idx, u_prev_idx] + 2.0 * ramp_weight

                Q = Q + self.q_reg * torch.eye(nz, device=dev, dtype=dtype).unsqueeze(0)

                neq = 2 * T
                A_eq = torch.zeros(batch, neq, nz, device=dev, dtype=dtype)
                b_eq = torch.zeros(batch, neq, device=dev, dtype=dtype)
                rhs_tin = self.b_tout * tout + self.b_sol * isol + self.b0 + w_est
                for t in range(T):
                    tin_row = 2 * t
                    tm_row = 2 * t + 1
                    A_eq[:, tin_row, x_slice.start + t] = 1.0
                    A_eq[:, tin_row, u_slice.start + t] = -b_heat_eff[:, t]
                    if t == 0:
                        b_eq[:, tin_row] = self.a11 * tin0 + self.a12 * tm0 + rhs_tin[:, t]
                        A_eq[:, tm_row, y_slice.start + t] = 1.0
                        b_eq[:, tm_row] = self.a21 * tin0 + self.a22 * tm0
                    else:
                        A_eq[:, tin_row, x_slice.start + t - 1] = -self.a11
                        A_eq[:, tin_row, y_slice.start + t - 1] = -self.a12
                        b_eq[:, tin_row] = rhs_tin[:, t]
                        A_eq[:, tm_row, y_slice.start + t] = 1.0
                        A_eq[:, tm_row, x_slice.start + t - 1] = -self.a21
                        A_eq[:, tm_row, y_slice.start + t - 1] = -self.a22

                n_ineq = 2 * T
                if slack_comfort:
                    n_ineq += 4 * T
                G = torch.zeros(batch, n_ineq, nz, device=dev, dtype=dtype)
                h = torch.zeros(batch, n_ineq, device=dev, dtype=dtype)
                if asymmetric_comfort:
                    lower = sp_ref - self.balanced_comfort_cold_tolerance
                    upper = sp_ref + self.balanced_comfort_hot_tolerance
                elif time_bias_comfort:
                    lower = sp_ref - self.balanced_time_bias_guard_band
                    upper = sp_ref + self.balanced_time_bias_guard_band
                else:
                    lower = sp_ref - self.balanced_comfort_band
                    upper = sp_ref + self.balanced_comfort_band

                row = 0
                for t in range(T):
                    G[:, row, u_slice.start + t] = 1.0
                    h[:, row] = self.u_max
                    row += 1
                    G[:, row, u_slice.start + t] = -1.0
                    h[:, row] = -self.u_min
                    row += 1
                    if not slack_comfort:
                        continue
                    assert cold_slice is not None and hot_slice is not None
                    G[:, row, cold_slice.start + t] = -1.0
                    h[:, row] = 0.0
                    row += 1
                    G[:, row, hot_slice.start + t] = -1.0
                    h[:, row] = 0.0
                    row += 1
                    G[:, row, x_slice.start + t] = -1.0
                    G[:, row, cold_slice.start + t] = -1.0
                    h[:, row] = -lower[:, t]
                    row += 1
                    G[:, row, x_slice.start + t] = 1.0
                    G[:, row, hot_slice.start + t] = -1.0
                    h[:, row] = upper[:, t]
                    row += 1

                return Q, p, G, h, A_eq, b_eq

            def _build_qp(
                self,
                tin0,
                tm0,
                d_seq,
                sp_seq,
                stress_seq=None,
                setpoint_bias_seq=None,
                w_seq=None,
            ):
                """Build the QP matrices for the 2-state tracking problem.

                Args:
                    tin0: (batch,) indoor temperature at t=0 (physical °C)
                    tm0:  (batch,) mass temperature at t=0 (physical °C)
                    d_seq: (batch, T, 2) → [Tout, Solar] in physical units
                    sp_seq: (batch, T, 1) → setpoint in physical °C
                    stress_seq: (batch, T) or None → GridStress values
                """
                import torch

                if self.cost_mode == "balanced_mpc":
                    return self._build_balanced_mpc_qp(
                        tin0,
                        tm0,
                        d_seq,
                        sp_seq,
                        stress_seq=stress_seq,
                        setpoint_bias_seq=setpoint_bias_seq,
                        w_seq=w_seq,
                    )

                dev = tin0.device
                dtype = tin0.dtype
                batch = tin0.shape[0]
                T = self.T
                nz = 3 * T  # [Tin_1..T, Tm_1..T, u_0..T-1]

                x_slice = slice(0, T)
                y_slice = slice(T, 2 * T)
                u_slice = slice(2 * T, 3 * T)

                tout = d_seq[:, :, 0]   # (batch, T)
                isol = d_seq[:, :, 1]   # (batch, T)
                cop_seq = self._cop(tout)  # (batch, T)
                if w_seq is None:
                    w_est = torch.zeros(batch, T, device=dev, dtype=dtype)
                else:
                    if w_seq.ndim == 1:
                        w_est = w_seq.view(1, T).repeat(batch, 1)
                    elif w_seq.ndim == 2:
                        w_est = w_seq
                    else:
                        w_est = w_seq.squeeze(-1)
                    if w_est.shape[0] == 1 and batch > 1:
                        w_est = w_est.repeat(batch, 1)
                    w_est = w_est.to(device=dev, dtype=dtype)

                b_heat_eff = self.b_heat * cop_seq  # (batch, T)

                q = self.q_track()
                r = self.r_u()

                # --- Cost matrix Q (nz x nz) and linear term p (nz) ---
                Q = torch.zeros(batch, nz, nz, device=dev, dtype=dtype)
                p = torch.zeros(batch, nz, device=dev, dtype=dtype)

                # Comfort: 0.5 * 2q * (Tin - sp)²  (qpth convention: 0.5 z'Qz + p'z)
                Q[:, x_slice, x_slice] = torch.eye(T, device=dev, dtype=dtype).unsqueeze(0) * (2.0 * q)
                sp_ref = sp_seq.squeeze(-1)  # (batch, T)
                p[:, x_slice] = -2.0 * q * sp_ref

                # Thermal mass penalty: encourages Tm to track Tsp, preventing cold-mass drag on Tin
                if self.tm_penalty_weight > 0:
                    q_tm = q * self.tm_penalty_weight
                    Q[:, y_slice, y_slice] = torch.eye(T, device=dev, dtype=dtype).unsqueeze(0) * (2.0 * q_tm)
                    p[:, y_slice] = -2.0 * q_tm * sp_ref

                # Action regularisation
                Q[:, u_slice, u_slice] = torch.eye(T, device=dev, dtype=dtype).unsqueeze(0) * (2.0 * r)

                # Linear energy/stress term on u
                if stress_seq is not None:
                    if stress_seq.ndim == 1:
                        stress = stress_seq.view(1, T).repeat(batch, 1)
                    else:
                        stress = stress_seq
                    p[:, u_slice] = p[:, u_slice] + stress.to(dev, dtype)

                Q = Q + self.q_reg * torch.eye(nz, device=dev, dtype=dtype).unsqueeze(0)

                # --- Equality constraints: A_eq z = b_eq (2T constraints) ---
                neq = 2 * T
                A_eq = torch.zeros(batch, neq, nz, device=dev, dtype=dtype)
                b_eq = torch.zeros(batch, neq, device=dev, dtype=dtype)

                rhs_tin = (
                    self.b_tout * tout + self.b_sol * isol + self.b0 + w_est
                )  # (batch, T) — excludes state coupling

                for t in range(T):
                    tin_row = 2 * t
                    tm_row = 2 * t + 1

                    # Tin dynamics: Tin_{t+1} - b_heat*cop[t]*u_t = a11*Tin_t + a12*Tm_t + rhs_t
                    A_eq[:, tin_row, t] = 1.0            # Tin_{t+1}
                    A_eq[:, tin_row, 2 * T + t] = -b_heat_eff[:, t]  # -b_heat*cop*u_t

                    if t == 0:
                        b_eq[:, tin_row] = self.a11 * tin0 + self.a12 * tm0 + rhs_tin[:, t]
                        b_eq[:, tm_row] = self.a21 * tin0 + self.a22 * tm0
                    else:
                        A_eq[:, tin_row, t - 1] = -self.a11    # -a11*Tin_t
                        A_eq[:, tin_row, T + t - 1] = -self.a12  # -a12*Tm_t
                        b_eq[:, tin_row] = rhs_tin[:, t]

                        # Tm dynamics: Tm_{t+1} - a21*Tin_t - a22*Tm_t = 0
                        A_eq[:, tm_row, T + t] = 1.0           # Tm_{t+1}
                        A_eq[:, tm_row, t - 1] = -self.a21    # -a21*Tin_t
                        A_eq[:, tm_row, T + t - 1] = -self.a22  # -a22*Tm_t

                    if t == 0:
                        # Tm_1 = a21*tin0 + a22*tm0 (no variables, pure rhs)
                        A_eq[:, tm_row, T] = 1.0  # Tm_1

                # --- Inequality: action bounds G z <= h (2T constraints) ---
                n_ineq = 2 * T
                G = torch.zeros(batch, n_ineq, nz, device=dev, dtype=dtype)
                h = torch.zeros(batch, n_ineq, device=dev, dtype=dtype)
                for t in range(T):
                    G[:, t, 2 * T + t] = 1.0
                    h[:, t] = self.u_max
                    G[:, T + t, 2 * T + t] = -1.0
                    h[:, T + t] = -self.u_min

                return Q, p, G, h, A_eq, b_eq

            def forward(self, tin0, tm0, d_seq, sp_seq, stress_seq=None, setpoint_bias_seq=None, w_seq=None):
                """Solve the 2R2C tracking QP and return (u_opt, tin_pred, tm_pred).

                Args:
                    tin0: (batch, 1) current indoor temperature (physical °C)
                    tm0:  (batch, 1) current mass temperature (physical °C)
                    d_seq: (batch, T, 2) disturbance sequence [Tout, Solar]
                    sp_seq: (batch, T, 1) setpoint sequence (physical °C)
                    stress_seq: (batch, T) GridStress profile (optional)

                Returns:
                    u_opt: (batch,) optimal first action
                    tin_seq: (batch, T) predicted Tin sequence
                    tm_seq: (batch, T) predicted Tm sequence
                """
                import torch

                tin0_v = tin0.view(-1)
                tm0_v = tm0.view(-1)
                Q, p, G, h, A_eq, b_eq = self._build_qp(
                    tin0_v,
                    tm0_v,
                    d_seq,
                    sp_seq,
                    stress_seq,
                    setpoint_bias_seq=setpoint_bias_seq,
                    w_seq=w_seq,
                )
                z = self.qp(Q, p, G, h, A_eq, b_eq)
                T = self.T
                tin_seq = z[:, :T]
                tm_seq = z[:, T:2 * T]
                u_seq = z[:, 2 * T:]
                u_opt = torch.clamp(u_seq[:, 0], self.u_min, self.u_max)
                return u_opt, tin_seq, tm_seq

        instance = _TwoStateDiffMPC(*args, **kwargs)
        return instance


def twostatediffmpc_param_snapshot(mpc) -> dict[str, float]:
    """Return cost parameter values for a TwoStateDiffMPC."""
    return {
        "q_track": float(mpc.q_track().detach().cpu()),
        "r_u": float(mpc.r_u().detach().cpu()),
        "a11": float(mpc.a11.detach().cpu()),
        "a12": float(mpc.a12.detach().cpu()),
        "a21": float(mpc.a21.detach().cpu()),
        "a22": float(mpc.a22.detach().cpu()),
        "b_heat": float(mpc.b_heat.detach().cpu()),
        "b0": float(mpc.b0.detach().cpu()),
        "sp_bias": 0.0,  # compatibility with 1-state snapshot
    }
