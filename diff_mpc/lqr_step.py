import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr

from collections import namedtuple

import time

from . import util, mpc
from .pnqp import pnqp

LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs'
)


# --- REPLACE the entire legacy class LQRStep(Function) with this ---

class _LQRStepFn(Function):
    """
    New-style autograd Function wrapper for LQRStep.

    IMPORTANT:
    - forward/backward are @staticmethod with ctx
    - all configuration is passed as non-tensor args and stored on ctx
    - gradients are returned ONLY for tensor inputs (x_init, C, c, F, f)
    """

    @staticmethod
    def forward(
        ctx,
        x_init, C, c, F, f,
        current_x, current_u,
        # --- config (non-tensor) ---
        n_state, n_ctrl, T,
        u_lower, u_upper,
        u_zero_I, delta_u,
        linesearch_decay, max_linesearch_iter,
        true_cost, true_dynamics,
        delta_space, verbose,
        back_eps, no_op_forward
    ):
        # Store config on ctx
        ctx.n_state = n_state
        ctx.n_ctrl = n_ctrl
        ctx.T = T
        ctx.u_lower = u_lower
        ctx.u_upper = u_upper
        ctx.u_zero_I = u_zero_I
        ctx.delta_u = delta_u
        ctx.linesearch_decay = linesearch_decay
        ctx.max_linesearch_iter = max_linesearch_iter
        ctx.true_cost = true_cost
        ctx.true_dynamics = true_dynamics
        ctx.delta_space = delta_space
        ctx.verbose = verbose
        ctx.back_eps = back_eps
        ctx.no_op_forward = no_op_forward

        # Cache the current trajectory (needed for delta-space backward)
        ctx.current_x = current_x
        ctx.current_u = current_u

        if no_op_forward:
            # Save tensors for backward
            ctx.save_for_backward(x_init, C, c, F, f, current_x, current_u)
            return current_x, current_u

        if delta_space:
            assert current_x is not None
            assert current_u is not None

            # Taylor-expand objective to do backward in delta space.
            c_back = []
            for t in range(T):
                xt = current_x[t]
                ut = current_u[t]
                xut = torch.cat((xt, ut), 1)
                c_back.append(util.bmv(C[t], xut) + c[t])
            c_back = torch.stack(c_back)
            f_back = None
        else:
            raise AssertionError("Only delta_space=True is implemented in this codebase.")

        Ks, ks, back_out = _LQRStepFn._lqr_backward(ctx, C, c_back, F, f_back)
        new_x, new_u, for_out = _LQRStepFn._lqr_forward(ctx, x_init, C, c, F, f, Ks, ks)

        # stash for the wrapper Module to read after apply()
        _LQRStepFn.last_back_out = back_out
        _LQRStepFn.last_for_out  = for_out

        # Save tensors for backward
        ctx.save_for_backward(x_init, C, c, F, f, new_x, new_u)

        # Save non-tensor outputs (for debugging/metrics)
        ctx.back_out = back_out
        ctx.for_out = for_out

        return new_x, new_u

    @staticmethod
    def backward(ctx, dl_dx, dl_du):
        start = time.time()
        x_init, C, c, F, f, new_x, new_u = ctx.saved_tensors

        T = ctx.T
        n_state = ctx.n_state

        # Build r = [dl/dx, dl/du] stacked over horizon
        r = []
        for t in range(T):
            rt = torch.cat((dl_dx[t], dl_du[t]), 1)
            r.append(rt)
        r = torch.stack(r)

        # Active-set indicator if box constraints exist
        if ctx.u_lower is None:
            I = None
        else:
            I = (torch.abs(new_u - ctx.u_lower) <= 1e-8) | (torch.abs(new_u - ctx.u_upper) <= 1e-8)

        dx_init0 = torch.zeros_like(x_init)

        _mpc = mpc.MPC(
            ctx.n_state, ctx.n_ctrl, ctx.T,
            u_zero_I=I,
            u_init=None,
            lqr_iter=1,
            verbose=-1,
            n_batch=C.size(1),
            delta_u=None,
            exit_unconverged=False,
            eps=ctx.back_eps,
        )

        dx, du, _ = _mpc(dx_init0, mpc.QuadCost(C, -r), mpc.LinDx(F, None))
        dx, du = dx.data, du.data

        dxu = torch.cat((dx, du), 2)
        xu = torch.cat((new_x, new_u), 2)

        # dC
        dC = torch.zeros_like(C)
        for t in range(T):
            xut = torch.cat((new_x[t], new_u[t]), 1)
            dxut = dxu[t]
            dCt = -0.5 * (util.bger(dxut, xut) + util.bger(xut, dxut))
            dC[t] = dCt

        # dc
        dc = -dxu

        # Lambda recursion
        lams = []
        prev_lam = None
        for t in range(T - 1, -1, -1):
            Ct_xx = C[t, :, :n_state, :n_state]
            Ct_xu = C[t, :, :n_state, n_state:]
            ct_x = c[t, :, :n_state]
            xt = new_x[t]
            ut = new_u[t]

            lamt = util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                lamt = lamt + util.bmv(Fxt, prev_lam)
            lams.append(lamt)
            prev_lam = lamt
        lams = list(reversed(lams))

        # dLambda recursion
        dlams = []
        prev_dlam = None
        for t in range(T - 1, -1, -1):
            dCt_xx = C[t, :, :n_state, :n_state]
            dCt_xu = C[t, :, :n_state, n_state:]
            drt_x = -r[t, :, :n_state]
            dxt = dx[t]
            dut = du[t]

            dlamt = util.bmv(dCt_xx, dxt) + util.bmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = F[t, :, :, :n_state].transpose(1, 2)
                dlamt = dlamt + util.bmv(Fxt, prev_dlam)
            dlams.append(dlamt)
            prev_dlam = dlamt
        dlams = torch.stack(list(reversed(dlams)))

        # dF
        dF = torch.zeros_like(F)
        for t in range(T - 1):
            xut = xu[t]
            lamt = lams[t + 1]
            dxut = dxu[t]
            dlamt = dlams[t + 1]
            dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))

        # df
        if f is not None and f.nelement() > 0:
            _dlams = dlams[1:]
            assert _dlams.shape == f.shape
            df = -_dlams
        else:
            # Must return a tensor (or None) corresponding to input f
            df = torch.tensor([], device=x_init.device)

        dx_init = -dlams[0]

        ctx.backward_time = time.time() - start

        # Return gradients for inputs:
        # forward(ctx, x_init, C, c, F, f, current_x, current_u, ...config...)
        return (
            dx_init,     # x_init
            dC,          # C
            dc,          # c
            dF,          # F
            df,          # f
            None, None,  # current_x, current_u
            # --- all config args: None ---
            None, None, None,
            None, None,
            None, None,
            None, None,
            None, None,
            None, None,
            None, None,
            None, None,
            None, None,
        )

    @staticmethod
    def _get_bound(ctx, side, t):
        v = getattr(ctx, 'u_' + side)

        # scalar python float
        if isinstance(v, float):
            return v

        # torch scalar / 1-element tensor: treat as scalar bound
        if torch.is_tensor(v) and v.numel() == 1:
            # return v  # broadcastable
            return float(v.item())

        # otherwise assume time-varying bounds indexed by t
        return v[t]

    # @staticmethod
    # def _lqr_backward(ctx, C, c, F, f):
    #     n_batch = C.size(1)
    #     u = ctx.current_u
    #     Ks = []
    #     ks = []
    #     prev_kt = None
    #     n_total_qp_iter = 0
    #     Vtp1 = vtp1 = None

    #     T = ctx.T
    #     n_state = ctx.n_state

    #     for t in range(T - 1, -1, -1):
    #         if t == T - 1:
    #             Qt = C[t]
    #             qt = c[t]
    #         else:
    #             Ft = F[t]
    #             Ft_T = Ft.transpose(1, 2)
    #             Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
    #             if f is None or f.nelement() == 0:
    #                 qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
    #             else:
    #                 ft = f[t]
    #                 qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + \
    #                      Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)

    #         Qt_xx = Qt[:, :n_state, :n_state]
    #         Qt_xu = Qt[:, :n_state, n_state:]
    #         Qt_ux = Qt[:, n_state:, :n_state]
    #         Qt_uu = Qt[:, n_state:, n_state:]
    #         qt_x = qt[:, :n_state]
    #         qt_u = qt[:, n_state:]

    #         if ctx.u_lower is None:
    #             if ctx.n_ctrl == 1 and ctx.u_zero_I is None:
    #                 Kt = -(1. / Qt_uu) * Qt_ux
    #                 kt = -(1. / Qt_uu.squeeze(2)) * qt_u
    #             else:
    #                 if ctx.u_zero_I is None:
    #                     Qt_uu_inv = [torch.pinverse(Qt_uu[i]) for i in range(Qt_uu.shape[0])]
    #                     Qt_uu_inv = torch.stack(Qt_uu_inv)
    #                     Kt = -Qt_uu_inv.bmm(Qt_ux)
    #                     kt = util.bmv(-Qt_uu_inv, qt_u)
    #                 else:
    #                     I = ctx.u_zero_I[t]
    #                     notI = 1 - I

    #                     qt_u_ = qt_u.clone()
    #                     qt_u_[I] = 0

    #                     Qt_uu_ = Qt_uu.clone()

    #                     if I.is_cuda:
    #                         notI_ = notI.float()
    #                         Qt_uu_I = (1 - util.bger(notI_, notI_)).type_as(I)
    #                     else:
    #                         Qt_uu_I = 1 - util.bger(notI, notI)

    #                     Qt_uu_[Qt_uu_I] = 0.
    #                     Qt_uu_[util.bdiag(I)] += 1e-8

    #                     Qt_ux_ = Qt_ux.clone()
    #                     Qt_ux_[I.unsqueeze(2).repeat(1, 1, Qt_ux.size(2))] = 0.

    #                     if ctx.n_ctrl == 1:
    #                         Kt = -(1. / Qt_uu_) * Qt_ux_
    #                         kt = -(1. / Qt_uu.squeeze(2)) * qt_u_
    #                     else:
    #                         # Qt_uu_LU_ = Qt_uu_.btrifact()
    #                         # Kt = -Qt_ux_.btrisolve(*Qt_uu_LU_)
    #                         # kt = -qt_u_.btrisolve(*Qt_uu_LU_)
    #                         Kt = -torch.linalg.solve(Qt_uu_, Qt_ux_)
    #                         kt = -torch.linalg.solve(Qt_uu_, qt_u_.unsqueeze(-1)).squeeze(-1)
    #         else:
    #             assert ctx.delta_space
    #             lb = _LQRStepFn._get_bound(ctx, 'lower', t) - u[t]
    #             ub = _LQRStepFn._get_bound(ctx, 'upper', t) - u[t]
    #             if ctx.delta_u is not None:
    #                 lb[lb < -ctx.delta_u] = -ctx.delta_u
    #                 ub[ub > ctx.delta_u] = ctx.delta_u

    #             kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(
    #                 Qt_uu, qt_u, lb, ub,
    #                 x_init=prev_kt, n_iter=20
    #             )
    #             if ctx.verbose > 1:
    #                 print('  + n_qp_iter: ', n_qp_iter + 1)

    #             n_total_qp_iter += 1 + n_qp_iter
    #             prev_kt = kt

    #             Qt_ux_ = Qt_ux.clone()
    #             Qt_ux_[(1 - If).unsqueeze(2).repeat(1, 1, Qt_ux.size(2))] = 0
    #             if ctx.n_ctrl == 1:
    #                 Kt = -((1. / Qt_uu_free_LU) * Qt_ux_)
    #             else:
    #                 # Kt = -Qt_ux_.btrisolve(*Qt_uu_free_LU)
    #                 Kt = -torch.linalg.solve(Qt_uu_free_LU, Qt_ux_)

    #         Kt_T = Kt.transpose(1, 2)
    #         Ks.append(Kt)
    #         ks.append(kt)

    #         Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
    #         vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
    #                Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
    #                Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

    #     return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)

    @staticmethod
    def _lqr_backward(ctx, C, c, F, f):
        n_batch = C.size(1)
        u = ctx.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None

        T = ctx.T
        n_state = ctx.n_state

        for t in range(T - 1, -1, -1):

            if t == T - 1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1, 2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)

                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = (
                        c[t]
                        + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2)
                        + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                    )

            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            # ================================
            # NO BOX CONSTRAINTS
            # ================================
            if ctx.u_lower is None:

                if ctx.n_ctrl == 1 and ctx.u_zero_I is None:
                    Kt = -(1.0 / Qt_uu) * Qt_ux
                    kt = -(1.0 / Qt_uu.squeeze(2)) * qt_u
                else:

                    if ctx.u_zero_I is None:
                        Qt_uu_inv = torch.linalg.pinv(Qt_uu)
                        Kt = -Qt_uu_inv.bmm(Qt_ux)
                        kt = util.bmv(-Qt_uu_inv, qt_u)

                    else:
                        I = ctx.u_zero_I[t]        # bool mask
                        notI = ~I                  # SAFE boolean inversion

                        qt_u_ = qt_u.clone()
                        qt_u_[I] = 0

                        Qt_uu_ = Qt_uu.clone()

                        # Build mask matrix safely
                        notI_f = notI.float()
                        Qt_uu_I = (1.0 - util.bger(notI_f, notI_f)).bool()

                        Qt_uu_[Qt_uu_I] = 0.0
                        Qt_uu_[util.bdiag(I)] += 1e-8

                        Qt_ux_ = Qt_ux.clone()
                        Qt_ux_[I.unsqueeze(2).repeat(1, 1, Qt_ux.size(2))] = 0.0

                        if ctx.n_ctrl == 1:
                            Kt = -(1.0 / Qt_uu_) * Qt_ux_
                            kt = -(1.0 / Qt_uu_.squeeze(2)) * qt_u_
                        else:
                            Kt = -torch.linalg.solve(Qt_uu_, Qt_ux_)
                            kt = -torch.linalg.solve(
                                Qt_uu_, qt_u_.unsqueeze(-1)
                            ).squeeze(-1)

            # ================================
            # BOX CONSTRAINTS (PNQP)
            # ================================
            else:
                assert ctx.delta_space

                lb = _LQRStepFn._get_bound(ctx, 'lower', t) - u[t]
                ub = _LQRStepFn._get_bound(ctx, 'upper', t) - u[t]

                if ctx.delta_u is not None:
                    lb = torch.maximum(lb, torch.full_like(lb, -ctx.delta_u))
                    ub = torch.minimum(ub, torch.full_like(ub, ctx.delta_u))

                kt, Qt_uu_free, If, n_qp_iter = pnqp(
                    Qt_uu, qt_u, lb, ub,
                    x_init=prev_kt, n_iter=20
                )

                n_total_qp_iter += 1 + n_qp_iter
                prev_kt = kt

                Qt_ux_ = Qt_ux.clone()

                # SAFE boolean masking
                Qt_ux_[(~If).unsqueeze(2).repeat(1, 1, Qt_ux.size(2))] = 0.0

                if ctx.n_ctrl == 1:
                    Kt = -((1.0 / Qt_uu_free) * Qt_ux_)
                else:
                    Kt = -torch.linalg.solve(Qt_uu_free, Qt_ux_)

            Kt_T = Kt.transpose(1, 2)

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = (
                Qt_xx
                + Qt_xu.bmm(Kt)
                + Kt_T.bmm(Qt_ux)
                + Kt_T.bmm(Qt_uu).bmm(Kt)
            )

            vtp1 = (
                qt_x
                + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2)
                + Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2)
                + Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)
            )

        return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)

    @staticmethod
    def _lqr_forward(ctx, x_init, C, c, F, f, Ks, ks):
        x = ctx.current_x
        u = ctx.current_u
        n_batch = C.size(1)

        old_cost = util.get_cost(ctx.T, u, ctx.true_cost, ctx.true_dynamics, x=x)

        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)
        full_du_norm = None

        i = 0
        while (current_cost is None or
               (old_cost is not None and torch.any((current_cost > old_cost)).cpu().item() == 1)) and \
              i < ctx.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(ctx.T):
                t_rev = ctx.T - 1 - t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = util.bmv(Kt, dxt) + ut + torch.diag(alphas).mm(kt)

                assert not ((ctx.delta_u is not None) and (ctx.u_lower is None))

                if ctx.u_zero_I is not None:
                    new_ut[ctx.u_zero_I[t]] = 0.

                if ctx.u_lower is not None:
                    lb = _LQRStepFn._get_bound(ctx, 'lower', t)
                    ub = _LQRStepFn._get_bound(ctx, 'upper', t)

                    if ctx.delta_u is not None:
                        lb_limit, ub_limit = lb, ub
                        lb = u[t] - ctx.delta_u
                        ub = u[t] + ctx.delta_u
                        I = lb < lb_limit
                        lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                        I = ub > ub_limit
                        ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                    new_ut = util.eclamp(new_ut, lb, ub)

                new_u.append(new_ut)

                new_xut = torch.cat((new_xt, new_ut), dim=1)
                if t < ctx.T - 1:
                    if isinstance(ctx.true_dynamics, mpc.LinDx):
                        Fdyn, fdyn = ctx.true_dynamics.F, ctx.true_dynamics.f
                        new_xtp1 = util.bmv(Fdyn[t], new_xut)
                        if fdyn is not None and fdyn.nelement() > 0:
                            new_xtp1 = new_xtp1 + fdyn[t]
                    else:
                        new_xtp1 = ctx.true_dynamics(Variable(new_xt), Variable(new_ut)).data

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t + 1])

                if isinstance(ctx.true_cost, mpc.QuadCost):
                    Cc, cc = ctx.true_cost.C, ctx.true_cost.c
                    obj = 0.5 * util.bquad(new_xut, Cc[t]) + util.bdot(new_xut, cc[t])
                else:
                    obj = ctx.true_cost(new_xut)

                objs.append(obj)

            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)

            new_u_t = torch.stack(new_u)
            new_x_t = torch.stack(new_x)
            if full_du_norm is None:
                full_du_norm = (u - new_u_t).transpose(1, 2).contiguous().view(n_batch, -1).norm(2, 1)

            alphas[current_cost > old_cost] *= ctx.linesearch_decay
            i += 1

            new_u, new_x = new_u_t, new_x_t

        alphas[current_cost > old_cost] /= ctx.linesearch_decay
        alpha_du_norm = (u - new_u).transpose(1, 2).contiguous().view(n_batch, -1).norm(2, 1)

        return new_x, new_u, LqrForOut(
            objs, full_du_norm,
            alpha_du_norm,
            torch.mean(alphas),
            current_cost
        )


class LQRStep(Module):
    """
    Drop-in replacement with the SAME external API as the old legacy Function class.
    Stores config in __init__, runs differentiable solve via _LQRStepFn.apply.
    """

    def __init__(
        self,
        n_state,
        n_ctrl,
        T,
        u_lower=None,
        u_upper=None,
        u_zero_I=None,
        delta_u=None,
        linesearch_decay=0.2,
        max_linesearch_iter=10,
        true_cost=None,
        true_dynamics=None,
        delta_space=True,
        current_x=None,
        current_u=None,
        verbose=0,
        back_eps=1e-3,
        no_op_forward=False,
    ):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T

        self.u_lower = util.get_data_maybe(u_lower)
        self.u_upper = util.get_data_maybe(u_upper)

        # --- PyTorch-2.x safe: scalar tensor bounds must become floats ---
        if torch.is_tensor(self.u_lower) and self.u_lower.numel() == 1:
            self.u_lower = float(self.u_lower.item())
        if torch.is_tensor(self.u_upper) and self.u_upper.numel() == 1:
            self.u_upper = float(self.u_upper.item())

        if isinstance(self.u_lower, int):
            self.u_lower = float(self.u_lower)
        if isinstance(self.u_upper, int):
            self.u_upper = float(self.u_upper)
        if isinstance(self.u_lower, np.float32):
            self.u_lower = u_lower.item()
        if isinstance(self.u_upper, np.float32):
            self.u_upper = u_upper.item()

        self.u_zero_I = u_zero_I
        self.delta_u = delta_u
        self.linesearch_decay = linesearch_decay
        self.max_linesearch_iter = max_linesearch_iter
        self.true_cost = true_cost
        self.true_dynamics = true_dynamics
        self.delta_space = delta_space
        self.current_x = util.get_data_maybe(current_x)
        self.current_u = util.get_data_maybe(current_u)
        self.verbose = verbose
        self.back_eps = back_eps
        self.no_op_forward = no_op_forward

    def forward(self, x_init, C, c, F, f=None):
        # Run differentiable step
        x, u = _LQRStepFn.apply(
            x_init, C, c, F, f,
            self.current_x, self.current_u,
            self.n_state, self.n_ctrl, self.T,
            self.u_lower, self.u_upper,
            self.u_zero_I, self.delta_u,
            self.linesearch_decay, self.max_linesearch_iter,
            self.true_cost, self.true_dynamics,
            self.delta_space, self.verbose,
            self.back_eps, self.no_op_forward
        )

        # Expose metrics in the same place the original code expects
        # (mpc.py reads _lqr.back_out and _lqr.for_out)
        self.back_out = getattr(_LQRStepFn, "last_back_out", None)
        self.for_out  = getattr(_LQRStepFn, "last_for_out", None)

        return x, u