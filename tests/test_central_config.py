"""Central-configuration layer: auxiliary lengths, mass-ratio reproduction,
and psi refinement."""

import numpy as np
import pytest

from r5bp_nonsymmetric.geometry import inter_primary_distances
from r5bp_nonsymmetric.central_config import (
    cc_aux, psi, mass_ratios, refine_geometry, check_admissibility,
)
from r5bp_nonsymmetric.cases import REFERENCE_CASES


def test_aux_lengths_match_interprimary_cubes():
    """sigma, zeta, tau, G, T, M must equal the cubes of the physical
    inter-primary distances (geometry <-> closed form cross-check)."""
    a, b, c1, c2 = 0.9, -0.1, 0.52, -0.88
    s, z, t, G, T, M = cc_aux(a, b, c1, c2)
    d = inter_primary_distances(a, b, c1, c2)
    assert np.isclose(z, d["r01"] ** 3, rtol=1e-12)
    assert np.isclose(t, d["r02"] ** 3, rtol=1e-12)
    assert np.isclose(s, d["r12"] ** 3, rtol=1e-12)
    assert np.isclose(T, d["r03"] ** 3, rtol=1e-12)
    assert np.isclose(M, d["r13"] ** 3, rtol=1e-12)
    assert np.isclose(G, d["r23"] ** 3, rtol=1e-12)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_mass_ratios_reproduce_reported_at_reference_coords(case):
    """At the (unrefined) reference coordinates, the mass-ratio formulae must
    reproduce the reported mass ratios -- this is the exact-reproduction check and
    the decisive test that fixed the Case-2 signs."""
    rho = mass_ratios(case.a, case.b, case.c1, case.c2)
    assert np.allclose(rho, case.reported_rho, rtol=2e-4), (
        f"{case.name}: computed {rho} vs reported {case.reported_rho}"
    )


def test_case2_wrong_signs_do_not_reproduce():
    """The rejected caption signs (-0.52, +0.88) must NOT reproduce the Case-2
    ratios."""
    rho = mass_ratios(0.9, -0.1, -0.52, 0.88)
    assert not np.allclose(rho, (4.21137, 0.956716, 1.27145), rtol=1e-2)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_refinement_drives_psi_below_1e12(case):
    ref = refine_geometry(case.a, case.b, case.c1, case.c2)
    assert abs(ref.psi_after) < 1e-12
    assert ref.a == case.a and ref.b == case.b          # a, b never touched
    assert ref.refined_var in ("c1", "c2")
    # exactly one coordinate moved
    if ref.refined_var == "c1":
        assert ref.c2 == case.c2
    else:
        assert ref.c1 == case.c1
    # chosen displacement is the smaller single-variable option
    opts = [abs(v) for v in (ref.delta_c1_option, ref.delta_c2_option)
            if np.isfinite(v)]
    assert abs(ref.delta) == pytest.approx(min(opts), rel=1e-9)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c.name)
def test_refined_configs_are_admissible(case):
    ref = refine_geometry(case.a, case.b, case.c1, case.c2)
    adm = check_admissibility(ref.a, ref.b, ref.c1, ref.c2)
    assert adm.admissible
    assert all(r > 0 for r in adm.rho)
