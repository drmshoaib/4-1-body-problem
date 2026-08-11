"""Independent multi-start root search for the equilibria Omega_x = Omega_y = 0.

Deliberately independent of any previously reported equilibria: starts come
from a deterministic dense grid plus a fixed-seed random cloud, NOT from known
Lagrange points. Each start is driven by a damped Newton iteration using the
analytic Hessian as Jacobian, refined to ||grad Omega|| < 1e-12, then duplicate
roots are clustered and points sitting on a primary are rejected.

The Newton hot path uses plain-float arithmetic (see ``_grad_hess_scalar``) for
speed; it is identical in form to the vectorised functions in ``potential`` and
is cross-checked against them in the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .geometry import primary_positions
from .potential import (
    hessian_determinant,
    stability_eigenvalues,
    jacobi_at_equilibrium,
    omega,
    grad,
)


@dataclass
class Equilibrium:
    x: float
    y: float
    residual: float          # ||grad Omega||
    hess_det: float          # Omega_xx Omega_yy - Omega_xy^2
    eigenvalues: np.ndarray  # 4 eigenvalues of A
    max_real_part: float
    stable: bool
    jacobi: float            # adopted C_J = 2 Omega
    omega: float


def _grad_hess_scalar(x, y, prim, r_min):
    """Fast scalar (Omega_x, Omega_y, Omega_xx, Omega_xy, Omega_yy).

    ``prim`` is a tuple of (xi, yi, mi) triples. Returns None if the point is
    within r_min of any primary (singular)."""
    Ox = x
    Oy = y
    Oxx = 1.0
    Oyy = 1.0
    Oxy = 0.0
    for (xi, yi, mi) in prim:
        dx = x - xi
        dy = y - yi
        r2 = dx * dx + dy * dy
        if r2 < r_min * r_min:
            return None
        r = sqrt(r2)
        r3 = r2 * r
        r5 = r3 * r2
        Ox -= mi * dx / r3
        Oy -= mi * dy / r3
        Oxx -= mi * (1.0 / r3 - 3.0 * dx * dx / r5)
        Oyy -= mi * (1.0 / r3 - 3.0 * dy * dy / r5)
        Oxy += 3.0 * mi * dx * dy / r5
    return Ox, Oy, Oxx, Oxy, Oyy


def _newton_scalar(x0, y0, prim, tol, max_iter, r_min):
    """Damped Newton on F = grad Omega using scalar arithmetic."""
    x, y = x0, y0
    gh = _grad_hess_scalar(x, y, prim, r_min)
    if gh is None:
        return None
    Ox, Oy, Oxx, Oxy, Oyy = gh
    res = sqrt(Ox * Ox + Oy * Oy)
    for _ in range(max_iter):
        if res < tol:
            return x, y, res
        det = Oxx * Oyy - Oxy * Oxy
        if det == 0.0 or not np.isfinite(det):
            return None
        sx = (Oyy * Ox - Oxy * Oy) / det
        sy = (-Oxy * Ox + Oxx * Oy) / det
        alpha = 1.0
        improved = False
        for _ls in range(25):
            xn, yn = x - alpha * sx, y - alpha * sy
            gh = _grad_hess_scalar(xn, yn, prim, r_min)
            if gh is not None:
                Oxn, Oyn = gh[0], gh[1]
                rn = sqrt(Oxn * Oxn + Oyn * Oyn)
                if rn < res:
                    x, y, res = xn, yn, rn
                    Ox, Oy, Oxx, Oxy, Oyy = gh
                    improved = True
                    break
            alpha *= 0.5
        if not improved:
            return (x, y, res) if res < tol else None
    return (x, y, res) if res < tol else None


def find_equilibria(
    a,
    b,
    c1,
    c2,
    m,
    domain=(-5.0, 5.0, -5.0, 5.0),
    grid_n=81,
    n_random=2000,
    tol=1e-12,
    max_iter=80,
    r_min=1e-3,
    cluster_tol=1e-6,
    seed=20260808,
):
    """Return a list[Equilibrium], sorted, independently found and verified."""
    xlo, xhi, ylo, yhi = domain
    P = primary_positions(a, b, c1, c2)
    m = np.asarray(m, dtype=float)
    prim = tuple(
        (float(P[i, 0]), float(P[i, 1]), float(m[i])) for i in range(4)
    )

    # start points: deterministic grid + fixed-seed random cloud
    gx = np.linspace(xlo, xhi, grid_n)
    gy = np.linspace(ylo, yhi, grid_n)
    rng = np.random.default_rng(seed)
    rx = rng.uniform(xlo, xhi, n_random)
    ry = rng.uniform(ylo, yhi, n_random)

    found = []

    def try_start(x0, y0):
        # skip starts that begin on a primary
        for (xi, yi, _mi) in prim:
            if (x0 - xi) ** 2 + (y0 - yi) ** 2 < r_min * r_min:
                return
        sol = _newton_scalar(x0, y0, prim, tol, max_iter, r_min)
        if sol is None:
            return
        x, y, res = sol
        # reject converged points essentially on a primary
        for (xi, yi, _mi) in prim:
            if (x - xi) ** 2 + (y - yi) ** 2 < (10 * r_min) ** 2:
                return
        found.append((x, y, res))

    for xi in gx:
        for yi in gy:
            try_start(float(xi), float(yi))
    for xi, yi in zip(rx.tolist(), ry.tolist()):
        try_start(xi, yi)

    # cluster duplicates
    clusters = []
    for (x, y, res) in found:
        placed = False
        for cl in clusters:
            if (x - cl["x"]) ** 2 + (y - cl["y"]) ** 2 < cluster_tol * cluster_tol:
                if res < cl["res"]:
                    cl.update(x=x, y=y, res=res)
                placed = True
                break
        if not placed:
            clusters.append({"x": x, "y": y, "res": res})

    equilibria = []
    for cl in clusters:
        x, y, res = cl["x"], cl["y"], cl["res"]
        # recompute residual through the vetted numpy gradient for reporting
        g = grad(x, y, a, b, c1, c2, m)
        res_np = float(np.hypot(g[0], g[1]))
        det = hessian_determinant(x, y, a, b, c1, c2, m)
        ev = stability_eigenvalues(x, y, a, b, c1, c2, m)
        max_re = float(np.max(ev.real))
        equilibria.append(
            Equilibrium(
                x=x,
                y=y,
                residual=res_np,
                hess_det=det,
                eigenvalues=ev,
                max_real_part=max_re,
                stable=bool(max_re <= 1e-8),
                jacobi=jacobi_at_equilibrium(x, y, a, b, c1, c2, m),
                omega=omega(x, y, a, b, c1, c2, m),
            )
        )

    equilibria.sort(key=lambda e: (round(e.x, 6), round(e.y, 6)))
    return equilibria


def count_equilibria(equilibria) -> int:
    return len(equilibria)
