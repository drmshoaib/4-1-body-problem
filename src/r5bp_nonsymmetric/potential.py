"""Rotating-frame effective potential and its analytic derivatives.

Effective potential:

    Omega(x, y) = (x^2 + y^2)/2
                + m0 / r40 + m1 / r41 + m2 / r42 + m3 / r43

with the four distances r4i to the primaries (see geometry.py). Every term
carries a plus sign.

Gradient:

    Omega_x = x - sum_i m_i (x - x_i) / r_i^3
    Omega_y = y - sum_i m_i (y - y_i) / r_i^3

Hessian (derived analytically in this package):

    Omega_xx = 1 - sum_i m_i ( 1/r_i^3 - 3 (x-x_i)^2 / r_i^5 )
    Omega_yy = 1 - sum_i m_i ( 1/r_i^3 - 3 (y-y_i)^2 / r_i^5 )
    Omega_xy = 3 sum_i m_i (x-x_i)(y-y_i) / r_i^5

Masses are passed as an array m = [m0, m1, m2, m3]; with the
normalization m3 = 1 and m0,m1,m2 = rho0,rho1,rho2.
"""

from __future__ import annotations

import numpy as np

from .geometry import primary_positions


def _deltas(x, y, a, b, c1, c2):
    """Return (dx, dy, r) arrays of length 4 for the four primaries."""
    P = primary_positions(a, b, c1, c2)
    dx = np.asarray(x, dtype=float) - P[:, 0]
    dy = np.asarray(y, dtype=float) - P[:, 1]
    r = np.sqrt(dx * dx + dy * dy)
    return dx, dy, r


def omega(x, y, a, b, c1, c2, m) -> float:
    """Effective potential Omega(x, y)."""
    m = np.asarray(m, dtype=float)
    dx, dy, r = _deltas(x, y, a, b, c1, c2)
    return 0.5 * (x * x + y * y) + float(np.sum(m / r))


def grad(x, y, a, b, c1, c2, m) -> np.ndarray:
    """Analytic gradient [Omega_x, Omega_y]."""
    m = np.asarray(m, dtype=float)
    dx, dy, r = _deltas(x, y, a, b, c1, c2)
    r3 = r ** 3
    Ox = x - float(np.sum(m * dx / r3))
    Oy = y - float(np.sum(m * dy / r3))
    return np.array([Ox, Oy], dtype=float)


def hessian(x, y, a, b, c1, c2, m):
    """Analytic Hessian entries (Omega_xx, Omega_xy, Omega_yy)."""
    m = np.asarray(m, dtype=float)
    dx, dy, r = _deltas(x, y, a, b, c1, c2)
    r3 = r ** 3
    r5 = r ** 5
    Oxx = 1.0 - float(np.sum(m * (1.0 / r3 - 3.0 * dx * dx / r5)))
    Oyy = 1.0 - float(np.sum(m * (1.0 / r3 - 3.0 * dy * dy / r5)))
    Oxy = 3.0 * float(np.sum(m * dx * dy / r5))
    return Oxx, Oxy, Oyy


def hessian_matrix(x, y, a, b, c1, c2, m) -> np.ndarray:
    """Hessian as a 2x2 symmetric matrix. This is exactly the Jacobian of the
    map F = (Omega_x, Omega_y), so it doubles as the Newton Jacobian for the
    equilibrium solver."""
    Oxx, Oxy, Oyy = hessian(x, y, a, b, c1, c2, m)
    return np.array([[Oxx, Oxy], [Oxy, Oyy]], dtype=float)


def hessian_determinant(x, y, a, b, c1, c2, m) -> float:
    """det(Hessian) = Omega_xx Omega_yy - Omega_xy^2."""
    Oxx, Oxy, Oyy = hessian(x, y, a, b, c1, c2, m)
    return Oxx * Oyy - Oxy * Oxy


def linearization_matrix(x, y, a, b, c1, c2, m) -> np.ndarray:
    """4x4 rotating-frame linearization matrix A.

    State vector is the perturbation (X, Y, Xdot, Ydot); the dynamics are

        d/dt (X, Y, Xdot, Ydot) = A (X, Y, Xdot, Ydot),

        A = [[ 0,    0,    1,  0],
             [ 0,    0,    0,  1],
             [Oxx,  Oxy,   0,  2],
             [Oxy,  Oyy,  -2,  0]].
    """
    Oxx, Oxy, Oyy = hessian(x, y, a, b, c1, c2, m)
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [Oxx, Oxy, 0.0, 2.0],
            [Oxy, Oyy, -2.0, 0.0],
        ],
        dtype=float,
    )


def stability_eigenvalues(x, y, a, b, c1, c2, m) -> np.ndarray:
    """The four eigenvalues of A."""
    return np.linalg.eigvals(linearization_matrix(x, y, a, b, c1, c2, m))


def is_linearly_stable(x, y, a, b, c1, c2, m, tol: float = 1e-8) -> bool:
    """Linearly stable iff every eigenvalue has non-positive real part
    (to tolerance). A single positive real part => unstable."""
    ev = stability_eigenvalues(x, y, a, b, c1, c2, m)
    return bool(np.max(ev.real) <= tol)


# ---------------------------------------------------------------------------
# Jacobi integral.
#
# From the equations of motion  xdd - 2 ydd = Omega_x,  ydd + 2 xdd = Omega_y,
# multiplying by (xdot, ydot) and adding gives  d/dt[ (1/2)(xdot^2+ydot^2) - Omega ] = 0,
# so  (1/2) v^2 - Omega = const.  We adopt the STANDARD restricted-problem
# normalization
#
#     C_J = 2 Omega - v^2          (v^2 = xdot^2 + ydot^2),
#
# for which C_J is conserved and, at an equilibrium (v = 0), C_J = 2 Omega(L).
# It derives cleanly from the governing equations (no factor-of-1/2 ambiguity).
#
# An alternative constant  C_alt = (1/2)v^2 - Omega  is sometimes used,
# giving C_alt = -Omega at equilibria and C_alt = -C_J/2.  We expose it
# separately so those zero-velocity-curve C-values can be cross-checked.
# ---------------------------------------------------------------------------

def jacobi_constant(x, y, vx, vy, a, b, c1, c2, m) -> float:
    """Adopted Jacobi constant  C_J = 2 Omega - (vx^2 + vy^2)."""
    return 2.0 * omega(x, y, a, b, c1, c2, m) - (vx * vx + vy * vy)


def jacobi_at_equilibrium(x, y, a, b, c1, c2, m) -> float:
    """Adopted Jacobi value at an equilibrium (v = 0):  C_J = 2 Omega(L)."""
    return 2.0 * omega(x, y, a, b, c1, c2, m)


def jacobi_alt(x, y, vx, vy, a, b, c1, c2, m) -> float:
    """Alternative constant  C_alt = (1/2)(vx^2+vy^2) - Omega
    (= -Omega at an equilibrium; = -C_J/2). Provided for cross-reference only."""
    return 0.5 * (vx * vx + vy * vy) - omega(x, y, a, b, c1, c2, m)
