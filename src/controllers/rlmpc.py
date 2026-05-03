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
                self.sp_bias_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
                self.sp_bias_max = 2.0

                if device is not None:
                    self.to(device)

            def _pos(self, x):
                return self.softplus(x) + 1e-6

            def q_track(self):
                return self.softplus(self.q_track_raw) + 1e-4

            def r_u(self):
                return self.softplus(self.r_u_raw) + 1e-4

            def sp_bias(self):
                return self.sp_bias_max * torch.tanh(self.sp_bias_raw)

            def _cop(self, tout):
                return torch.clamp(self.cop_a + self.cop_b * tout, self.cop_min, self.cop_max)

            def _zoh(self, ac, dt):
                a = torch.exp(ac * dt)
                eps = 1e-8
                phi = torch.where(torch.abs(ac) > eps, (a - 1.0) / ac, torch.full_like(ac, dt))
                return a, phi

            def _build_qp(self, x0, d_seq, sp_seq, w_seq, q_mult_seq=None):
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
                sp_biased = sp_seq.squeeze(-1) + self.sp_bias()
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
                p_vec[:, 0:horizon] = -q_weights * sp_biased

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

            def forward(self, x0, d_seq, sp_seq, w_seq=None, q_mult_seq=None):
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

                q_mat, p_vec, g, h, aeq, beq = self._build_qp(x0, d_seq, sp_seq, w_seq, q_mult_seq)
                z = self.qp(q_mat, p_vec, g, h, aeq, beq)
                x_seq = z[:, : self.T].unsqueeze(-1)
                u_seq = z[:, self.T :].unsqueeze(-1)
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

    mpc.q_track_raw.requires_grad = True
    mpc.r_u_raw.requires_grad = True
    mpc.sp_bias_raw.requires_grad = True
    return mpc


def diffmpc_param_snapshot(mpc) -> dict[str, float]:
    """Return report-friendly parameter values."""

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
    }
