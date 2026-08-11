"""Driver: refine each reference case to |psi|<1e-12, recompute mass ratios,
independently find and verify equilibria, compute Hessian determinants, the
four stability eigenvalues, and the adopted Jacobi value; run the derivative
validation; and write results/*.csv.

Run:  python scripts/run_reference_cases.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from r5bp_nonsymmetric.cases import REFERENCE_CASES
from r5bp_nonsymmetric.central_config import (
    refine_geometry, mass_ratios, masses_from_ratios, psi, check_admissibility,
)
from r5bp_nonsymmetric.equilibria import find_equilibria
from r5bp_nonsymmetric.validation import validate_derivatives

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    case_rows = []
    eq_rows = []

    for case in REFERENCE_CASES:
        print(f"\n=== {case.name} ===")
        ref = refine_geometry(case.a, case.b, case.c1, case.c2)
        a, b, c1, c2 = ref.a, ref.b, ref.c1, ref.c2
        print(f"  refined {ref.refined_var}: {ref.reference_value} -> {ref.refined_value:.12f} "
              f"(delta={ref.delta:+.3e})")
        print(f"  psi: {ref.psi_before:+.3e} -> {ref.psi_after:+.3e}")

        rho0, rho1, rho2 = mass_ratios(a, b, c1, c2)
        m = masses_from_ratios(rho0, rho1, rho2)
        adm = check_admissibility(a, b, c1, c2)
        print(f"  rho = ({rho0:.6f}, {rho1:.6f}, {rho2:.6f})  admissible={adm.admissible}")

        dv = validate_derivatives(a, b, c1, c2, m, n=25)
        print(f"  derivative check: max_grad_err={dv['max_grad_error']:.2e}  "
              f"max_hess_err={dv['max_hessian_error']:.2e}")

        eqs = find_equilibria(a, b, c1, c2, m)
        print(f"  reported count={case.reported_count}  recomputed count={len(eqs)}")

        max_res = max((e.residual for e in eqs), default=float("nan"))
        case_rows.append({
            "case": case.name,
            "a": a, "b": b,
            "c1_reference": case.c1, "c2_reference": case.c2,
            "c1_refined": c1, "c2_refined": c2,
            "refined_var": ref.refined_var,
            "refined_delta": ref.delta,
            "delta_c1_option": ref.delta_c1_option,
            "delta_c2_option": ref.delta_c2_option,
            "min_norm_distance": ref.min_norm_distance,
            "psi_reference": ref.psi_before,
            "psi_refined": ref.psi_after,
            "rho0": rho0, "rho1": rho1, "rho2": rho2,
            "rho0_reported": case.reported_rho[0],
            "rho1_reported": case.reported_rho[1],
            "rho2_reported": case.reported_rho[2],
            "admissible": adm.admissible,
            "reported_count": case.reported_count,
            "recomputed_count": len(eqs),
            "max_equilibrium_residual": max_res,
            "max_grad_error": dv["max_grad_error"],
            "max_hessian_error": dv["max_hessian_error"],
        })

        for i, e in enumerate(eqs, 1):
            ev = e.eigenvalues
            eq_rows.append({
                "case": case.name,
                "index": i,
                "x": e.x, "y": e.y,
                "grad_residual": e.residual,
                "hessian_det": e.hess_det,
                "eig1": ev[0], "eig2": ev[1], "eig3": ev[2], "eig4": ev[3],
                "max_real_part": e.max_real_part,
                "stable": e.stable,
                "omega": e.omega,
                "jacobi_CJ_2omega": e.jacobi,
                "jacobi_alt_negomega": -e.omega,
            })

    df_cases = pd.DataFrame(case_rows)
    df_eqs = pd.DataFrame(eq_rows)
    df_cases.to_csv(os.path.join(RESULTS, "refined_reference_cases.csv"), index=False)
    # complex eigenvalues -> stringified for CSV portability
    for col in ["eig1", "eig2", "eig3", "eig4"]:
        df_eqs[col] = df_eqs[col].apply(lambda z: f"{z.real:.10g}{z.imag:+.10g}j")
    df_eqs.to_csv(os.path.join(RESULTS, "refined_equilibria.csv"), index=False)

    print("\nWrote:")
    print("  results/refined_reference_cases.csv")
    print("  results/refined_equilibria.csv")
    return df_cases, df_eqs


if __name__ == "__main__":
    main()
