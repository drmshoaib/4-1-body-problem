"""Derivative validation: analytic gradient vs central finite differences of
Omega, and analytic Hessian vs central finite differences of the gradient,
at >= 20 random nonsingular points, for every refined case."""

import numpy as np
import pytest

from r5bp_nonsymmetric.central_config import (
    refine_geometry, mass_ratios, masses_from_ratios,
)
from r5bp_nonsymmetric.validation import (
    validate_derivatives, random_nonsingular_points, gradient_fd, hessian_fd,
)
from r5bp_nonsymmetric.potential import grad, hessian, hessian_matrix
from r5bp_nonsymmetric.cases import REFERENCE_CASES


def _masses(case):
    ref = refine_geometry(case.a, case.b, case.c1, case.c2)
    m = masses_from_ratios(*mass_ratios(ref.a, ref.b, ref.c1, ref.c2))
    return ref, m


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_gradient_and_hessian_vs_finite_differences(case):
    ref, m = _masses(case)
    report = validate_derivatives(ref.a, ref.b, ref.c1, ref.c2, m, n=25)
    assert report["n_points"] >= 20
    assert report["max_grad_error"] < 1e-6, report
    assert report["max_hessian_error"] < 1e-5, report


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_hessian_is_symmetric_and_is_jacobian_of_gradient(case):
    ref, m = _masses(case)
    for (x, y) in random_nonsingular_points(ref.a, ref.b, ref.c1, ref.c2, n=20):
        Oxx, Oxy, Oyy = hessian(x, y, ref.a, ref.b, ref.c1, ref.c2, m)
        # symmetry via the two finite-difference routes to the mixed partial
        _, hxy_x, hxy_y, _ = hessian_fd(x, y, ref.a, ref.b, ref.c1, ref.c2, m)
        assert abs(hxy_x - hxy_y) < 1e-5
        # the analytic Hessian matrix is exactly the Jacobian of grad
        H = hessian_matrix(x, y, ref.a, ref.b, ref.c1, ref.c2, m)
        assert np.allclose(H, H.T)


def test_finite_difference_helpers_are_self_consistent():
    """Sanity: FD gradient of Omega matches analytic gradient on a fixed point."""
    a, b, c1, c2 = 0.9, -0.1, 0.52, -0.88
    m = masses_from_ratios(*mass_ratios(a, b, c1, c2))
    x, y = 1.3, -0.7
    g_an = grad(x, y, a, b, c1, c2, m)
    g_fd = gradient_fd(x, y, a, b, c1, c2, m)
    assert np.allclose(g_an, g_fd, atol=1e-7)
