"""Pseudo-arclength continuation on the four-body central-configuration
manifold psi(a,b,c1,c2)=0, and helpers for continuing the restricted
fifth-body equilibria along the resulting path.

This module ADDS to the verified model; it does not modify geometry.py,
central_config.py, potential.py, or equilibria.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .central_config import psi, mass_ratios, check_admissibility


# ---------------------------------------------------------------------------
# CC-manifold gradient and tangent
# ---------------------------------------------------------------------------

def psi_grad(a, b, c1, c2, h=1e-7):
    """Central finite-difference gradient (dpsi/dc1, dpsi/dc2)."""
    gc1 = (psi(a, b, c1 + h, c2) - psi(a, b, c1 - h, c2)) / (2 * h)
    gc2 = (psi(a, b, c1, c2 + h) - psi(a, b, c1, c2 - h)) / (2 * h)
    return np.array([gc1, gc2])


def cc_tangent(a, b, c1, c2, prev=None):
    """Unit tangent to psi=0 at (c1,c2); in 2-D it is perpendicular to grad psi.
    If ``prev`` is given, the sign is chosen so the tangent is consistent
    (dot(prev, t) > 0)."""
    g = psi_grad(a, b, c1, c2)
    t = np.array([-g[1], g[0]])
    n = np.linalg.norm(t)
    if n == 0:
        raise RuntimeError("Singular CC point: grad psi = 0.")
    t = t / n
    if prev is not None and np.dot(prev, t) < 0:
        t = -t
    return t


@dataclass
class PathPoint:
    s: float          # normalized arclength in [0,1] (filled after tracing)
    arclen: float     # cumulative raw arclength
    c1: float
    c2: float
    rho0: float
    rho1: float
    rho2: float
    psi_residual: float
    admissible: bool


def _corrector(a, b, q_pred, t_k, q_k, ds, tol=1e-13, max_iter=60):
    """Newton corrector solving  psi(q)=0,  t_k . (q - q_k) - ds = 0."""
    q = q_pred.astype(float).copy()
    for _ in range(max_iter):
        f1 = psi(a, b, q[0], q[1])
        f2 = float(np.dot(t_k, q - q_k) - ds)
        if abs(f1) < tol and abs(f2) < 1e-14:
            return q
        g = psi_grad(a, b, q[0], q[1])
        J = np.array([[g[0], g[1]], [t_k[0], t_k[1]]])
        try:
            step = np.linalg.solve(J, np.array([f1, f2]))
        except np.linalg.LinAlgError:
            return None
        q = q - step
    if abs(psi(a, b, q[0], q[1])) < 1e-11:
        return q
    return None


def trace_cc_path(a, b, start, target, ds0=0.01, max_arclen=12.0,
                  reach_tol=2e-3, direction_sign=None, stop_on_inadmissible=False):
    """Pseudo-arclength trace of psi=0 from ``start`` toward ``target``.

    Returns (points, reached, info). ``reached`` is True if the trace arrives
    within ``reach_tol`` of ``target``. Admissibility (rho>0, denominators,
    no collision) is recorded at every accepted point; if
    ``stop_on_inadmissible`` the trace halts at the first inadmissible point.
    """
    start = np.asarray(start, float)
    target = np.asarray(target, float)

    t = cc_tangent(a, b, start[0], start[1])
    if direction_sign is not None:
        # force initial direction sign relative to (target - start)
        if np.dot(t, target - start) * direction_sign < 0:
            t = -t
    else:
        if np.dot(t, target - start) < 0:
            t = -t

    def record(c1, c2, arclen):
        adm = check_admissibility(a, b, c1, c2)
        return PathPoint(
            s=np.nan, arclen=arclen, c1=float(c1), c2=float(c2),
            rho0=float(adm.rho[0]), rho1=float(adm.rho[1]), rho2=float(adm.rho[2]),
            psi_residual=float(psi(a, b, c1, c2)), admissible=adm.admissible,
        )

    q_k = start.copy()
    arclen = 0.0
    pts = [record(q_k[0], q_k[1], arclen)]
    ds = ds0
    min_dist = np.linalg.norm(q_k - target)
    reached = False
    info = {"folds": 0, "inadmissible_first_arclen": None}

    steps = 0
    while arclen < max_arclen and steps < 20000:
        steps += 1
        q_pred = q_k + ds * t
        q_new = _corrector(a, b, q_pred, t, q_k, ds)
        if q_new is None:
            ds *= 0.5
            if ds < 1e-6:
                break
            continue
        # accept
        arclen += np.linalg.norm(q_new - q_k)
        t_new = cc_tangent(a, b, q_new[0], q_new[1], prev=t)
        if np.dot(t_new, t) < 0.2:
            info["folds"] += 1  # sharp turn (near a fold)
        pt = record(q_new[0], q_new[1], arclen)
        pts.append(pt)
        if (not pt.admissible) and info["inadmissible_first_arclen"] is None:
            info["inadmissible_first_arclen"] = arclen
            if stop_on_inadmissible:
                break

        d = np.linalg.norm(q_new - target)
        if d < min_dist:
            min_dist = d
        # reached target?
        if d < reach_tol:
            reached = True
            # snap exactly to target
            pts.append(record(target[0], target[1],
                              arclen + np.linalg.norm(target - q_new)))
            break
        # if we are close and moving away, try a smaller step to home in
        if d < 3 * ds0 and d > min_dist + 1e-9:
            ds = max(ds * 0.5, 1e-4)

        q_k = q_new
        t = t_new
        # gentle step growth
        ds = min(ds * 1.15, ds0)

    info["min_dist_to_target"] = float(min_dist)
    info["total_arclen"] = float(arclen)
    return pts, reached, info


def normalize_s(points):
    """Fill the normalized arclength s in [0,1] over a list of PathPoint."""
    if not points:
        return points
    total = points[-1].arclen
    if total <= 0:
        for p in points:
            p.s = 0.0
        return points
    for p in points:
        p.s = p.arclen / total
    return points
