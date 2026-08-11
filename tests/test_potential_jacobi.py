"""Effective potential, linearization matrix, and Jacobi-convention tests."""

import numpy as np
import pytest

from r5bp_nonsymmetric.central_config import (
    refine_geometry, mass_ratios, masses_from_ratios,
)
from r5bp_nonsymmetric.potential import (
    omega, grad, hessian, linearization_matrix,
    jacobi_constant, jacobi_alt, jacobi_at_equilibrium,
)
from r5bp_nonsymmetric.cases import REFERENCE_CASES


def _masses(case):
    ref = refine_geometry(case.a, case.b, case.c1, case.c2)
    return ref, masses_from_ratios(*mass_ratios(ref.a, ref.b, ref.c1, ref.c2))


def test_linearization_matrix_structure():
    ref, m = _masses(REFERENCE_CASES[0])
    x, y = 1.5, 1.5
    Oxx, Oxy, Oyy = hessian(x, y, ref.a, ref.b, ref.c1, ref.c2, m)
    A = linearization_matrix(x, y, ref.a, ref.b, ref.c1, ref.c2, m)
    expected = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [Oxx, Oxy, 0, 2],
        [Oxy, Oyy, -2, 0],
    ], dtype=float)
    assert np.allclose(A, expected)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_jacobi_conventions_are_consistent(case):
    """C_alt = -C_J / 2 identically; at equilibrium C_J = 2 Omega."""
    ref, m = _masses(case)
    x, y, vx, vy = 1.2, -0.6, 0.3, -0.4
    CJ = jacobi_constant(x, y, vx, vy, ref.a, ref.b, ref.c1, ref.c2, m)
    Ct = jacobi_alt(x, y, vx, vy, ref.a, ref.b, ref.c1, ref.c2, m)
    assert Ct == pytest.approx(-CJ / 2.0, rel=1e-12)
    # at v = 0
    CJ0 = jacobi_constant(x, y, 0.0, 0.0, ref.a, ref.b, ref.c1, ref.c2, m)
    assert CJ0 == pytest.approx(
        2.0 * omega(x, y, ref.a, ref.b, ref.c1, ref.c2, m), rel=1e-12)
    assert jacobi_at_equilibrium(x, y, ref.a, ref.b, ref.c1, ref.c2, m) == \
        pytest.approx(CJ0, rel=1e-12)


def test_jacobi_is_conserved_along_a_trajectory():
    """Numerically integrate the equations of motion and check C_J drift.

    xddot = 2 yddot + Omega_x  is implicit; use the standard form
      xdd = 2*yd + Omega_x ,  ydd = -2*xd + Omega_y .
    """
    ref, m = _masses(REFERENCE_CASES[0])
    a, b, c1, c2 = ref.a, ref.b, ref.c1, ref.c2

    def deriv(state):
        x, y, vx, vy = state
        g = grad(x, y, a, b, c1, c2, m)
        ax = 2.0 * vy + g[0]
        ay = -2.0 * vx + g[1]
        return np.array([vx, vy, ax, ay])

    s = np.array([2.5, 2.4, 0.05, -0.03])   # start away from primaries
    C0 = jacobi_constant(*s[:2], *s[2:], a, b, c1, c2, m)
    dt = 1e-4
    for _ in range(2000):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    C1 = jacobi_constant(*s[:2], *s[2:], a, b, c1, c2, m)
    assert abs(C1 - C0) < 1e-6
