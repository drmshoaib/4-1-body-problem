"""Finite-difference validation of the analytic gradient and Hessian.

Central differences of Omega validate the gradient; central differences of the
analytic gradient validate the Hessian. Test points are drawn away from the
primaries (where the potential is singular and finite differences are
meaningless).
"""

from __future__ import annotations

import numpy as np

from .geometry import primary_positions
from .potential import omega, grad, hessian


def gradient_fd(x, y, a, b, c1, c2, m, h=1e-6):
    """Central-difference gradient of Omega."""
    ox = (omega(x + h, y, a, b, c1, c2, m) - omega(x - h, y, a, b, c1, c2, m)) / (2 * h)
    oy = (omega(x, y + h, a, b, c1, c2, m) - omega(x, y - h, a, b, c1, c2, m)) / (2 * h)
    return np.array([ox, oy])


def hessian_fd(x, y, a, b, c1, c2, m, h=1e-6):
    """Central differences of the analytic gradient -> (Oxx, Oxy_from_x, Oxy_from_y, Oyy)."""
    gxp = grad(x + h, y, a, b, c1, c2, m)
    gxm = grad(x - h, y, a, b, c1, c2, m)
    gyp = grad(x, y + h, a, b, c1, c2, m)
    gym = grad(x, y - h, a, b, c1, c2, m)
    Oxx = (gxp[0] - gxm[0]) / (2 * h)
    Oxy_from_x = (gxp[1] - gxm[1]) / (2 * h)   # d(Omega_y)/dx
    Oxy_from_y = (gyp[0] - gym[0]) / (2 * h)   # d(Omega_x)/dy
    Oyy = (gyp[1] - gym[1]) / (2 * h)
    return Oxx, Oxy_from_x, Oxy_from_y, Oyy


def random_nonsingular_points(a, b, c1, c2, n=20, box=4.0, min_dist=0.75, seed=12345):
    """n points in [-box, box]^2 whose distance to every primary exceeds min_dist."""
    rng = np.random.default_rng(seed)
    P = primary_positions(a, b, c1, c2)
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(-box, box, 2)
        if np.all(np.hypot(x - P[:, 0], y - P[:, 1]) > min_dist):
            pts.append((float(x), float(y)))
    return pts


def validate_derivatives(a, b, c1, c2, m, n=20, h=1e-6, seed=12345):
    """Compare analytic vs finite-difference gradient and Hessian at n random
    nonsingular points. Returns a dict of maximum absolute errors."""
    pts = random_nonsingular_points(a, b, c1, c2, n=n, seed=seed)
    grad_err = 0.0
    hess_err = 0.0
    sym_err = 0.0
    for (x, y) in pts:
        g_an = grad(x, y, a, b, c1, c2, m)
        g_fd = gradient_fd(x, y, a, b, c1, c2, m, h)
        grad_err = max(grad_err, float(np.max(np.abs(g_an - g_fd))))

        Oxx, Oxy, Oyy = hessian(x, y, a, b, c1, c2, m)
        hxx, hxy_x, hxy_y, hyy = hessian_fd(x, y, a, b, c1, c2, m, h)
        hess_err = max(
            hess_err,
            abs(Oxx - hxx), abs(Oyy - hyy),
            abs(Oxy - hxy_x), abs(Oxy - hxy_y),
        )
        # symmetry of the mixed partial from the two finite-difference routes
        sym_err = max(sym_err, abs(hxy_x - hxy_y))

    return {
        "n_points": n,
        "h": h,
        "max_grad_error": grad_err,
        "max_hessian_error": hess_err,
        "max_mixed_partial_asymmetry": sym_err,
    }
