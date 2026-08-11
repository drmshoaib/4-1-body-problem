"""Central-configuration layer: auxiliary lengths, constraint psi, mass ratios,
positivity/admissibility, and numerical refinement of the free geometry so that
|psi| < 1e-12.

All formulae are the standard four-body central-configuration expressions;
they reproduce every reference mass ratio to full precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .geometry import inter_primary_distances


def cc_aux(a: float, b: float, c1: float, c2: float):
    """Auxiliary quantities (sigma, zeta, tau, G, T, M) of the CC formulation.

    sigma = (1 + a^2)^{3/2}
    zeta  = (1 + b^2)^{3/2}    # also written 'varsigma'
    tau   = (a - b)^3
    G = (c1^2 + (c2 - a)^2)^{3/2}
    T = (c1^2 + (c2 - b)^2)^{3/2}
    M = (c2^2 + (c1 + 1)^2)^{3/2}
    """
    sigma = (1.0 + a * a) ** 1.5
    zeta = (1.0 + b * b) ** 1.5
    tau = (a - b) ** 3
    G = (c1 * c1 + (c2 - a) ** 2) ** 1.5
    T = (c1 * c1 + (c2 - b) ** 2) ** 1.5
    M = (c2 * c2 + (c1 + 1.0) ** 2) ** 1.5
    return sigma, zeta, tau, G, T, M


def psi(a: float, b: float, c1: float, c2: float) -> float:
    """Central-configuration constraint psi for the planar four-body problem.

    psi = tau*(sigma - zeta)*M
        + T*[(tau - sigma)*M + sigma*(zeta - tau)]
        + G*[(sigma - zeta)*T + (zeta - tau)*M + zeta*(tau - sigma)]

    A valid non-collinear planar four-body CC requires psi = 0 together with
    positive mass ratios.
    """
    s, z, t, G, T, M = cc_aux(a, b, c1, c2)
    return (
        t * (s - z) * M
        + T * ((t - s) * M + s * (z - t))
        + G * ((s - z) * T + (z - t) * M + z * (t - s))
    )


def mass_ratios(a: float, b: float, c1: float, c2: float):
    """Mass ratios (rho0, rho1, rho2) = (m0/m3, m1/m3, m2/m3).

        rho0 = (c2 - a - a*c1)(M - G) zeta tau / [ (a - b)(zeta - tau) M G ]
        rho1 =  c1 (G - T) sigma zeta         / [ (sigma - zeta) G T ]
        rho2 = (c2 - b - b*c1)(M - T) sigma tau / [ (a - b)(tau - sigma) M T ]
    """
    s, z, t, G, T, M = cc_aux(a, b, c1, c2)
    rho0 = (c2 - a - a * c1) * (M - G) * z * t / ((a - b) * (z - t) * M * G)
    rho1 = c1 * (G - T) * s * z / ((s - z) * G * T)
    rho2 = (c2 - b - b * c1) * (M - T) * s * t / ((a - b) * (t - s) * M * T)
    return rho0, rho1, rho2


def masses_from_ratios(rho0: float, rho1: float, rho2: float) -> np.ndarray:
    """Physical primary masses with the normalization m3 = 1.

    Returns array [m0, m1, m2, m3] = [rho0, rho1, rho2, 1].
    """
    return np.array([rho0, rho1, rho2, 1.0], dtype=float)


@dataclass
class Admissibility:
    positive_masses: bool
    denominators_ok: bool
    no_collision: bool
    rho: tuple

    @property
    def admissible(self) -> bool:
        return self.positive_masses and self.denominators_ok and self.no_collision


def check_admissibility(a, b, c1, c2, collision_tol: float = 1e-9) -> Admissibility:
    """Admissibility: psi may be handled separately; here we
    check positive mass ratios, non-vanishing denominators of the mass-ratio
    formulae, and the
    absence of primary collisions.
    """
    s, z, t, G, T, M = cc_aux(a, b, c1, c2)
    denoms = [
        (a - b) * (z - t) * M * G,
        (s - z) * G * T,
        (a - b) * (t - s) * M * T,
    ]
    denominators_ok = all(abs(d) > 0.0 and np.isfinite(d) for d in denoms)

    # collisions: any inter-primary distance ~ 0
    dists = inter_primary_distances(a, b, c1, c2)
    no_collision = all(v > collision_tol for v in dists.values())

    if denominators_ok:
        rho = mass_ratios(a, b, c1, c2)
        positive = all(r > 0.0 and np.isfinite(r) for r in rho)
    else:
        rho = (np.nan, np.nan, np.nan)
        positive = False

    return Admissibility(positive, denominators_ok, no_collision, rho)


@dataclass
class Refinement:
    a: float
    b: float
    c1: float
    c2: float
    refined_var: str          # which coordinate was moved ("c1" or "c2")
    reference_value: float       # its reference (input) value
    refined_value: float      # its refined value
    delta: float              # refined - reference (signed)
    psi_before: float
    psi_after: float
    # transparency: displacements of the alternatives NOT taken
    delta_c1_option: float    # signed delta if c1 alone were refined (nan if none)
    delta_c2_option: float    # signed delta if c2 alone were refined (nan if none)
    min_norm_distance: float  # Euclidean distance of the true nearest CC (both moved)


def _refine_single(a, b, c1, c2, var, tol, windows=(0.15, 0.4, 1.0)):
    """Solve psi = 0 for a single coordinate (`var`) nearest its reference value.
    Holds a, b, and the other c-coordinate fixed. Returns signed delta or None."""
    if var == "c2":
        f = lambda v: psi(a, b, c1, v)
        guess = c2
    else:
        f = lambda v: psi(a, b, v, c2)
        guess = c1
    for w in windows:
        vs = np.linspace(guess - w, guess + w, 801)
        fv = np.array([f(v) for v in vs])
        sc = np.where(np.sign(fv[:-1]) * np.sign(fv[1:]) < 0)[0]
        if len(sc):
            centers = 0.5 * (vs[sc] + vs[sc + 1])
            k = sc[int(np.argmin(np.abs(centers - guess)))]
            root = brentq(f, vs[k], vs[k + 1], xtol=tol, rtol=8.9e-16, maxiter=200)
            return root - guess
    return None


def _min_norm_projection(a, b, c1, c2, tol=1e-14, max_iter=200):
    """Nearest point on the psi=0 curve (moving both c1, c2) via a gradient
    projection. Used only to REPORT how close the single-variable refinement is
    to the theoretical minimum; not used as the refinement itself."""
    p = np.array([c1, c2], dtype=float)
    h = 1e-6
    for _ in range(max_iter):
        val = psi(a, b, p[0], p[1])
        if abs(val) < tol:
            break
        gx = (psi(a, b, p[0] + h, p[1]) - psi(a, b, p[0] - h, p[1])) / (2 * h)
        gy = (psi(a, b, p[0], p[1] + h) - psi(a, b, p[0], p[1] - h)) / (2 * h)
        gn = gx * gx + gy * gy
        if gn == 0.0:
            break
        p = p - val * np.array([gx, gy]) / gn
    return float(np.hypot(p[0] - c1, p[1] - c2))


def refine_geometry(a: float, b: float, c1: float, c2: float,
                    tol: float = 1e-14) -> Refinement:
    """Refine ONE free geometric coordinate so that |psi| < 1e-12, staying as
    close as possible to the reference values.

    Policy (explicit, not silent):
      * ``a`` and ``b`` are FIXED parameters and are never changed;
      * we compute the c1-only and c2-only refinements (each solving psi = 0 for
        that single coordinate while holding the other fixed) and adopt the one
        with the SMALLER absolute displacement -- i.e. the refined configuration
        closest to the reference values;
      * the returned object records the chosen variable, its signed delta, the
        delta of the alternative, and the true minimal-norm distance (both
        coordinates moved) so the choice is fully auditable.

    The 1-D solve is a bracketed Brent root find, so |psi| is at machine
    precision (well below 1e-12).
    """
    psi_before = psi(a, b, c1, c2)
    dc1 = _refine_single(a, b, c1, c2, "c1", tol)
    dc2 = _refine_single(a, b, c1, c2, "c2", tol)
    min_norm = _min_norm_projection(a, b, c1, c2)

    candidates = []
    if dc1 is not None:
        candidates.append(("c1", dc1))
    if dc2 is not None:
        candidates.append(("c2", dc2))
    if not candidates:
        raise RuntimeError(
            f"Could not refine geometry to |psi|<1e-12 for "
            f"(a,b,c1,c2)=({a},{b},{c1},{c2})."
        )

    var, delta = min(candidates, key=lambda t: abs(t[1]))
    if var == "c1":
        c1_new, c2_new = c1 + delta, c2
        reference_value, refined_value = c1, c1_new
    else:
        c1_new, c2_new = c1, c2 + delta
        reference_value, refined_value = c2, c2_new

    p_after = psi(a, b, c1_new, c2_new)
    if abs(p_after) >= 1e-12:
        raise RuntimeError(f"Refinement failed |psi_after|={p_after:.3e}")

    return Refinement(
        a=a, b=b, c1=c1_new, c2=c2_new,
        refined_var=var, reference_value=reference_value, refined_value=refined_value,
        delta=delta, psi_before=psi_before, psi_after=p_after,
        delta_c1_option=(dc1 if dc1 is not None else float("nan")),
        delta_c2_option=(dc2 if dc2 is not None else float("nan")),
        min_norm_distance=min_norm,
    )
