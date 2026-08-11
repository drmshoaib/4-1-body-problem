"""Independent equilibrium recomputation: verified counts and residuals.

We do NOT assume any previously reported counts are right; we assert what the independent
multi-start search actually returns. (It happens to reproduce 5, 9, 5, 5.)"""

import numpy as np
import pytest

from r5bp_nonsymmetric.central_config import (
    refine_geometry, mass_ratios, masses_from_ratios,
)
from r5bp_nonsymmetric.equilibria import find_equilibria
from r5bp_nonsymmetric.potential import grad
from r5bp_nonsymmetric.cases import REFERENCE_CASES

# Recomputed counts observed by the independent solver (see reproduction doc).
RECOMPUTED_COUNTS = {"Case 1": 5, "Case 2": 9, "Case 3": 5, "Case 4": 5}


@pytest.fixture(scope="module")
def solved():
    out = {}
    for case in REFERENCE_CASES:
        ref = refine_geometry(case.a, case.b, case.c1, case.c2)
        m = masses_from_ratios(*mass_ratios(ref.a, ref.b, ref.c1, ref.c2))
        eqs = find_equilibria(ref.a, ref.b, ref.c1, ref.c2, m)
        out[case.name] = (ref, m, eqs)
    return out


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_recomputed_count_is_reproducible(case, solved):
    _, _, eqs = solved[case.name]
    assert len(eqs) == RECOMPUTED_COUNTS[case.name]


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_every_equilibrium_residual_below_1e12(case, solved):
    ref, m, eqs = solved[case.name]
    for e in eqs:
        g = grad(e.x, e.y, ref.a, ref.b, ref.c1, ref.c2, m)
        assert np.hypot(g[0], g[1]) < 1e-12


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_equilibria_are_distinct(case, solved):
    _, _, eqs = solved[case.name]
    pts = np.array([[e.x, e.y] for e in eqs])
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            assert np.hypot(*(pts[i] - pts[j])) > 1e-4


def test_case1_has_one_spectrally_stable_outer_point(solved):
    """Independent finding contradicting a previously reported 'all unstable' label:
    the outer Case-1 point (~2.67, 2.62) has an all-imaginary spectrum."""
    _, _, eqs = solved["Case 1"]
    stable = [e for e in eqs if e.max_real_part <= 1e-8]
    assert len(stable) == 1
    e = stable[0]
    assert e.x > 2.0 and e.y > 2.0
    assert np.all(np.abs(e.eigenvalues.real) < 1e-6)
