"""Periodic-orbit computation in the rotating frame: seeding from centre
eigenmodes, single-shooting differential correction with an amplitude/phase
condition, natural-parameter continuation in amplitude, and Floquet analysis
from the integrated monodromy matrix.

Rotating-frame first-order system  u=(x,y,vx,vy):
    xdot  = vx
    ydot  = vy
    vxdot = 2 vy + Omega_x
    vydot = -2 vx + Omega_y
The state Jacobian is exactly the 4x4 matrix A of potential.linearization_matrix,
so the state-transition matrix (STM) obeys  Phidot = A(u) Phi,  Phi(0)=I.

Builds on the verified model; changes nothing in the core modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .central_config import mass_ratios, masses_from_ratios
from .potential import grad, linearization_matrix, omega
from .geometry import primary_positions

# Documented integrator tolerances (see docs/reproducibility_protocol.md).
RTOL = 1e-12
ATOL = 1e-12
METHOD = "DOP853"


@dataclass
class Params:
    a: float
    b: float
    c1: float
    c2: float
    m: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.m is None:
            self.m = masses_from_ratios(*mass_ratios(self.a, self.b, self.c1, self.c2))


def f_state(u, P: Params):
    x, y, vx, vy = u
    g = grad(x, y, P.a, P.b, P.c1, P.c2, P.m)
    return np.array([vx, vy, 2.0 * vy + g[0], -2.0 * vx + g[1]])


def _rhs_full(t, U, P: Params):
    u = U[:4]
    Phi = U[4:].reshape(4, 4)
    x, y = u[0], u[1]
    g = grad(x, y, P.a, P.b, P.c1, P.c2, P.m)
    du = np.array([u[2], u[3], 2.0 * u[3] + g[0], -2.0 * u[2] + g[1]])
    A = linearization_matrix(x, y, P.a, P.b, P.c1, P.c2, P.m)
    dPhi = A @ Phi
    return np.concatenate([du, dPhi.ravel()])


class IntegrationError(RuntimeError):
    """Raised when the ODE integration fails (e.g. a near-collision)."""


def integrate(u0, T, P: Params, n_eval=0):
    """Integrate state + STM over [0, T]. Returns (uT, M[4x4], traj or None).
    Raises IntegrationError on solver failure."""
    U0 = np.concatenate([u0, np.eye(4).ravel()])
    t_eval = np.linspace(0.0, T, n_eval) if n_eval else None
    sol = solve_ivp(lambda t, U: _rhs_full(t, U, P), (0.0, T), U0,
                    method=METHOD, rtol=RTOL, atol=ATOL,
                    t_eval=t_eval, dense_output=False)
    if not sol.success:
        raise IntegrationError(sol.message)
    UT = sol.y[:, -1]
    uT = UT[:4]
    M = UT[4:].reshape(4, 4)
    traj = sol.y[:4, :].T if n_eval else None
    return uT, M, traj


def centre_modes(x_eq, y_eq, P: Params, tol_re=1e-6, im_min=1e-6):
    """Return list of (omega, ehat, w) for eigenvalues +i*omega at the
    equilibrium, sorted by descending omega. ehat = normalized Re(w)."""
    A = linearization_matrix(x_eq, y_eq, P.a, P.b, P.c1, P.c2, P.m)
    w, V = np.linalg.eig(A)
    modes = []
    for k in range(4):
        if abs(w[k].real) < tol_re and w[k].imag > im_min:
            vec = V[:, k]
            re = vec.real
            ehat = re / np.linalg.norm(re)
            modes.append((float(w[k].imag), ehat, vec))
    modes.sort(key=lambda t: -t[0])
    return modes


@dataclass
class Orbit:
    u0: np.ndarray
    T: float
    closure: float
    M: np.ndarray
    amplitude: float          # ehat-projection amplitude used as continuation par
    geom_amplitude: float     # max |(x,y)-(x_eq,y_eq)| over the orbit
    jacobi: float
    min_dist: tuple           # to (m0,m1,m2,m3)
    converged: bool


def _jacobi(u, P: Params):
    x, y, vx, vy = u
    return 2.0 * omega(x, y, P.a, P.b, P.c1, P.c2, P.m) - (vx * vx + vy * vy)


def correct_orbit(u_seed, T0, ehat, amp, z_eq, P: Params,
                  tol=1e-10, max_iter=40):
    """Single-shooting differential correction of a periodic orbit.

    Unknowns X=(u0[4], T) in R^5. For an autonomous, energy-preserving flow the
    periodicity map has TWO neutral directions (time-translation and the
    one-parameter family); both must be removed. We impose two scalar
    conditions, giving a full-column-rank 6x5 system solved by Gauss-Newton:

        R[0:4] = phi_T(u0) - u0                      (periodicity)
        R[4]   = ehat . f(u0)                        (phase: ehat-projection
                                                      stationary at t=0)
        R[5]   = ehat . (u0 - z_eq) - amp            (amplitude / family member)

    Jacobian (6x5), with M=Phi(T) the monodromy and A0=Df(u0) the state
    Jacobian at u0 (= linearization_matrix):

        J[0:4,0:4] = M - I ,   J[0:4,4] = f(phi_T(u0))
        J[4,  0:4] = ehat^T A0 ,   J[4,4] = 0
        J[5,  0:4] = ehat^T   ,     J[5,4] = 0

    dX = argmin ||J dX + R|| (least squares; J has full column rank 5). An orbit
    is accepted when the closure ||phi_T(u0)-u0|| < tol.
    """
    u0 = np.array(u_seed, float)
    T = float(T0)
    closure = np.inf
    M = np.eye(4)
    for _ in range(max_iter):
        try:
            uT, M, _ = integrate(u0, T, P)
        except IntegrationError:
            # a Newton step pushed the trajectory into a near-collision
            return Orbit(u0=u0, T=T, closure=np.inf, M=np.eye(4), amplitude=amp,
                         geom_amplitude=np.nan, jacobi=np.nan,
                         min_dist=(np.nan,) * 4, converged=False)
        per = uT - u0
        closure = float(np.linalg.norm(per))
        f0 = f_state(u0, P)
        r_phase = float(ehat @ f0)
        r_amp = float(ehat @ (u0 - z_eq) - amp)
        R = np.concatenate([per, [r_phase, r_amp]])
        if closure < tol and abs(r_phase) < 1e-10 and abs(r_amp) < 1e-12:
            break
        fT = f_state(uT, P)
        A0 = linearization_matrix(u0[0], u0[1], P.a, P.b, P.c1, P.c2, P.m)
        J = np.zeros((6, 5))
        J[:4, :4] = M - np.eye(4)
        J[:4, 4] = fT
        J[4, :4] = ehat @ A0
        J[5, :4] = ehat
        dX = np.linalg.lstsq(J, -R, rcond=None)[0]
        # cap the step so Newton cannot overshoot into a collision
        max_state = 0.5
        if np.linalg.norm(dX[:4]) > max_state:
            dX = dX * (max_state / np.linalg.norm(dX[:4]))
        if abs(dX[4]) > 0.5 * T:
            dX = dX * (0.5 * T / abs(dX[4]))
        u0 = u0 + dX[:4]
        T = T + dX[4]
        if T <= 0:
            T = abs(T) + 1e-3
    # final metrics on a dense trajectory
    try:
        uT, M, traj = integrate(u0, T, P, n_eval=600)
    except IntegrationError:
        return Orbit(u0=u0, T=T, closure=np.inf, M=M, amplitude=amp,
                     geom_amplitude=np.nan, jacobi=np.nan,
                     min_dist=(np.nan,) * 4, converged=False)
    closure = float(np.linalg.norm(uT - u0))
    P4 = primary_positions(P.a, P.b, P.c1, P.c2)
    md = tuple(float(np.min(np.hypot(traj[:, 0] - P4[i, 0], traj[:, 1] - P4[i, 1])))
               for i in range(4))
    geom = float(np.max(np.hypot(traj[:, 0] - z_eq[0], traj[:, 1] - z_eq[1])))
    return Orbit(u0=u0, T=T, closure=closure, M=M, amplitude=amp,
                 geom_amplitude=geom, jacobi=_jacobi(u0, P), min_dist=md,
                 converged=closure < tol)


# ---------------------------------------------------------------------------
# Floquet analysis
# ---------------------------------------------------------------------------

@dataclass
class Floquet:
    multipliers: np.ndarray   # 4 complex
    trivial_err: float        # how close the trivial pair is to +1
    nontrivial: tuple         # the two nontrivial multipliers
    stability_index: float    # nu = lam + 1/lam (real part)
    classification: str


def floquet(M, unit_tol=1e-3):
    ev = np.linalg.eigvals(M)
    # identify the two closest to +1 (trivial pair of an autonomous Hamiltonian)
    order = np.argsort(np.abs(ev - 1.0))
    trivial = ev[order[:2]]
    nontriv = ev[order[2:]]
    trivial_err = float(np.max(np.abs(trivial - 1.0)))
    lam = nontriv[0]
    nu = float((lam + 1.0 / lam).real)
    # classify the nontrivial reciprocal pair
    mags = np.abs(nontriv)
    if np.all(np.abs(mags - 1.0) < unit_tol):
        cls = "spectrally stable"
    elif np.all(np.abs(nontriv.imag) < unit_tol * np.maximum(1.0, mags)):
        cls = "real unstable"
    else:
        cls = "complex unstable"
    if abs(abs(nu) - 2.0) < 1e-2:
        cls += " (near transition)"
    return Floquet(multipliers=ev, trivial_err=trivial_err,
                   nontrivial=tuple(nontriv), stability_index=nu,
                   classification=cls)
