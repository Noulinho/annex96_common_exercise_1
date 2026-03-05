import torch
from . import util

def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    """
    Projected Newton QP (box constraints) solver used by the diff_mpc iLQR step.

    Solves:
        minimize  0.5 x^T H x + q^T x
        s.t.      lower <= x <= upper

    Inputs:
        H: (B, n, n)
        q: (B, n)
        lower, upper: (B, n)
        x_init: (B, n) or None
        n_iter: int

    Returns:
        x: (B, n)           solution
        Hff: (B, n, n) or (B, n, n) "free-free" Hessian (masked)
        If: (B, n) bool     free set mask
        it: int             number of iterations used (0-indexed like original)
    """

    B, n, _ = H.shape
    device = H.device
    dtype = H.dtype

    # Initialize x
    if x_init is None:
        # Original code used solving H x = -q for init when possible
        # but to stay robust we clamp a zero init into bounds.
        x = torch.zeros(B, n, device=device, dtype=dtype)
        x = util.eclamp(x, lower, upper)
    else:
        x = util.eclamp(x_init, lower, upper)

    # Iterate
    for it in range(n_iter):
        # Gradient g = Hx + q
        g = util.bmv(H, x) + q

        # Clamped set Ic: at lower with positive grad OR at upper with negative grad
        Ic = ((x == lower) & (g > 0)) | ((x == upper) & (g < 0))

        # Free set If is complement (PyTorch-2 safe)
        If = ~Ic

        # If nothing is free, we are done
        if torch.all(~If).item():
            break

        # Build masked Hff and g_f
        If_f = If.float()

        # Mask matrix for free-free block
        # Hff_I is 1 where both indices are free
        Hff_I = util.bger(If_f, If_f).bool()

        Hff = H.clone()
        Hff[~Hff_I] = 0.0

        # Add small diagonal regularization on the free variables
        # (same intention as original adding eps on diag)
        diag_mask = util.bdiag(If)
        Hff[diag_mask] = Hff[diag_mask] + 1e-8

        g_f = g.clone()
        g_f[Ic] = 0.0

        # Solve Hff dx = -g_f  (batched, PyTorch-2 safe)
        dx = -torch.linalg.solve(Hff, g_f.unsqueeze(-1)).squeeze(-1)

        # Line-search / step (original often used full step)
        x_new = util.eclamp(x + dx, lower, upper)

        # Convergence check: if no movement, stop
        if torch.max(torch.abs(x_new - x)).item() < 1e-10:
            x = x_new
            break

        x = x_new

    # Recompute free set for output consistency
    g = util.bmv(H, x) + q
    Ic = ((x == lower) & (g > 0)) | ((x == upper) & (g < 0))
    If = ~Ic

    # Build final masked Hessian returned as "Hff"
    If_f = If.float()
    Hff_I = util.bger(If_f, If_f).bool()
    Hff = H.clone()
    Hff[~Hff_I] = 0.0
    diag_mask = util.bdiag(If)
    Hff[diag_mask] = Hff[diag_mask] + 1e-8

    return x, Hff, If, it