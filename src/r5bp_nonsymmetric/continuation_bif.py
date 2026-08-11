"""Admissible-arc tracing, augmented bifurcation solving, and fold
classification on the four-body CC manifold. Builds on continuation.py and the
verified model; modifies nothing in the core modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .central_config import psi, mass_ratios, masses_from_ratios, check_admissibility
from .potential import grad, hessian, hessian_determinant, hessian_matrix
from .continuation import cc_tangent, _corrector, PathPoint


# ---------------------------------------------------------------------------
# Trace the maximal admissible arc of psi=0 through a start point
# ---------------------------------------------------------------------------

def _trace_direction(a, b, start, sign, ds0=0.01, max_arclen=8.0):
    """Trace psi=0 from start in one direction until admissibility is lost.
    Returns (admissible_points, boundary_info). Points are (arclen, c1, c2)
    with arclen increasing from 0 at start."""
    t = cc_tangent(a, b, start[0], start[1])
    # orient by sign of the c1-component so +/- are reproducible
    if t[0] * sign < 0:
        t = -t
    q = start.astype(float).copy()
    arclen = 0.0
    pts = []
    ds = ds0
    prev_adm = True
    boundary = None
    steps = 0
    while arclen < max_arclen and steps < 6000:
        steps += 1
        qn = _corrector(a, b, q + ds * t, t, q, ds)
        if qn is None:
            ds *= 0.5
            if ds < 1e-7:
                break
            continue
        arclen += float(np.linalg.norm(qn - q))
        adm = check_admissibility(a, b, qn[0], qn[1])
        if adm.admissible:
            pts.append((arclen, float(qn[0]), float(qn[1])))
            t = cc_tangent(a, b, qn[0], qn[1], prev=t)
            q = qn
            ds = min(ds * 1.15, ds0)
        else:
            # refine the boundary by bisection in arclength between q and qn
            lo, hi = q.copy(), qn.copy()
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                # project mid back onto psi=0 along local tangent
                proj = _corrector(a, b, mid, t, q, float(np.dot(t, mid - q)))
                if proj is not None:
                    mid = proj
                if check_admissibility(a, b, mid[0], mid[1]).admissible:
                    lo = mid
                else:
                    hi = mid
            # which rho vanishes at the boundary
            r = mass_ratios(a, b, hi[0], hi[1])
            which = int(np.argmin(np.abs(r)))
            boundary = {"c1": float(hi[0]), "c2": float(hi[1]),
                        "vanishing_rho": which, "rho": tuple(float(x) for x in r)}
            break
    return pts, boundary


def trace_admissible_arc(a, b, start, ds0=0.01):
    """Assemble the maximal admissible arc through ``start`` as an ordered list
    of PathPoint with s in [0,1]. Returns (points, boundary_minus, boundary_plus)."""
    plus, bplus = _trace_direction(a, b, np.asarray(start, float), +1, ds0)
    minus, bminus = _trace_direction(a, b, np.asarray(start, float), -1, ds0)

    Lminus = minus[-1][0] if minus else 0.0

    ordered = []
    # minus branch reversed: farthest first (s=0 side)
    for (al, c1, c2) in reversed(minus):
        ordered.append((Lminus - al, c1, c2))
    ordered.append((Lminus, float(start[0]), float(start[1])))  # start
    for (al, c1, c2) in plus:
        ordered.append((Lminus + al, c1, c2))

    total = ordered[-1][0] - ordered[0][0]
    pts = []
    for (arclen, c1, c2) in ordered:
        adm = check_admissibility(a, b, c1, c2)
        s = (arclen - ordered[0][0]) / total if total > 0 else 0.0
        pts.append(PathPoint(s=s, arclen=arclen, c1=c1, c2=c2,
                             rho0=adm.rho[0], rho1=adm.rho[1], rho2=adm.rho[2],
                             psi_residual=psi(a, b, c1, c2), admissible=adm.admissible))
    return pts, bminus, bplus


# ---------------------------------------------------------------------------
# Augmented bifurcation system  G(x,y,c1,c2) = [Ox, Oy, detHO, psi] = 0
# ---------------------------------------------------------------------------

def _G(z, a, b):
    x, y, c1, c2 = z
    m = masses_from_ratios(*mass_ratios(a, b, c1, c2))
    g = grad(x, y, a, b, c1, c2, m)
    d = hessian_determinant(x, y, a, b, c1, c2, m)
    return np.array([g[0], g[1], d, psi(a, b, c1, c2)])


def _jac_fd(z, a, b, h=1e-6):
    n = len(z)
    J = np.zeros((n, n))
    for j in range(n):
        zp = z.copy(); zm = z.copy()
        zp[j] += h; zm[j] -= h
        J[:, j] = (_G(zp, a, b) - _G(zm, a, b)) / (2 * h)
    return J


@dataclass
class Critical:
    x: float
    y: float
    c1: float
    c2: float
    residual: float
    rho: tuple
    converged: bool


def solve_augmented(guess, a, b, tol=1e-11, max_iter=100):
    """Newton solve of G=0 for (x,y,c1,c2). Returns Critical."""
    z = np.asarray(guess, float).copy()
    for _ in range(max_iter):
        G = _G(z, a, b)
        res = float(np.linalg.norm(G))
        if res < tol:
            break
        J = _jac_fd(z, a, b)
        try:
            step = np.linalg.solve(J, G)
        except np.linalg.LinAlgError:
            return Critical(*z, res, tuple(mass_ratios(a, b, z[2], z[3])), False)
        # damped
        alpha = 1.0
        for _ls in range(30):
            zn = z - alpha * step
            if np.linalg.norm(_G(zn, a, b)) < res:
                break
            alpha *= 0.5
        z = zn
    G = _G(z, a, b)
    res = float(np.linalg.norm(G))
    r = mass_ratios(a, b, z[2], z[3])
    return Critical(float(z[0]), float(z[1]), float(z[2]), float(z[3]), res,
                    tuple(float(x) for x in r), res < 1e-8)


# ---------------------------------------------------------------------------
# Fold classification at a critical point
# ---------------------------------------------------------------------------

@dataclass
class FoldData:
    mu_small: float          # near-zero Hessian eigenvalue
    mu_large: float          # other eigenvalue
    null_vec: tuple          # null eigenvector (x,y)
    alpha: float             # transversality coefficient  v . dF/d(arclength)
    beta: float              # quadratic nondegeneracy  0.5 d3Omega/dv3
    is_generic_fold: bool


def classify_fold(crit: Critical, a, b, h=1e-5):
    x, y, c1, c2 = crit.x, crit.y, crit.c1, crit.c2
    m = masses_from_ratios(*mass_ratios(a, b, c1, c2))
    H = hessian_matrix(x, y, a, b, c1, c2, m)
    w, V = np.linalg.eigh(H)
    idx = int(np.argmin(np.abs(w)))
    mu_small = float(w[idx])
    mu_large = float(w[1 - idx])
    v = V[:, idx]
    v = v / np.linalg.norm(v)

    # beta: 0.5 * d^3 Omega / dv^3  via FD of  g(t)=v^T H(z+t v) v
    def gH(t):
        Ht = hessian_matrix(x + t * v[0], y + t * v[1], a, b, c1, c2, m)
        return float(v @ Ht @ v)
    beta = 0.5 * (gH(h) - gH(-h)) / (2 * h)

    # alpha: transversality along the CC-path tangent (masses follow rho(c1,c2))
    t_path = cc_tangent(a, b, c1, c2)
    eps = 1e-6

    def F_at(dc1, dc2):
        mm = masses_from_ratios(*mass_ratios(a, b, c1 + dc1, c2 + dc2))
        return grad(x, y, a, b, c1 + dc1, c2 + dc2, mm)
    dF = (F_at(eps * t_path[0], eps * t_path[1]) -
          F_at(-eps * t_path[0], -eps * t_path[1])) / (2 * eps)
    alpha = float(v @ dF)

    # generic fold: one ~zero eigenvalue, other nonzero, alpha and beta nonzero
    scale = max(abs(mu_large), 1.0)
    is_fold = (abs(mu_small) < 1e-5 * scale and abs(mu_large) > 1e-3 * scale
               and abs(alpha) > 1e-6 and abs(beta) > 1e-6)
    return FoldData(mu_small, mu_large, (float(v[0]), float(v[1])),
                    alpha, beta, bool(is_fold))
