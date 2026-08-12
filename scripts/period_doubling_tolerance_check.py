"""Integration-tolerance check for the family-10 near-period-doubling minimum.

Recomputes the family-10 ultra-fine amplitude sweep at the double-precision
integration limit (rtol = atol = 1e-14) and compares the closest approach
min|nu+2| and the turn-back with the standard rtol = atol = 1e-12 result, to
show the near-degeneracy is not an integration artefact.

Reads:  results/period_doubling_scan.csv, results/periodic_orbits.csv
Writes: results/period_doubling_tolerance.csv

Run:  python scripts/period_doubling_tolerance_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
RESULTS = os.path.join(ROOT, "results")

import r5bp_nonsymmetric.periodic as pmod  # noqa: E402


def summarise(nu, amp):
    s = np.asarray(nu) + 2.0
    i = int(np.argmin(np.abs(s)))
    return float(np.min(np.abs(s))), float(np.asarray(amp)[i]), float(nu[i]), \
        bool(np.any(s[:-1] * s[1:] < 0))


def main():
    scan = pd.read_csv(os.path.join(RESULTS, "period_doubling_scan.csv"))
    ultra = scan[scan.family.str.contains("ultra")]
    m12, a12, nu12, x12 = summarise(ultra["nu"].values, ultra["amplitude"].values)
    print(f"[rtol=1e-12] n={len(ultra)} min|nu+2|={m12:.3e} at amp={a12:.4f} "
          f"nu={nu12:.9f} crosses(-2)={x12}")

    # recompute the same window at the double-precision limit
    pmod.RTOL = pmod.ATOL = 1e-14
    import refine_period_doubling as rpd  # noqa: E402
    o = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    r0 = o[o.family_id == 10].iloc[0]
    rows, _ = rpd.fine_sweep("fam10_ultra_1e14", r0.c1, r0.c2, r0.x_eq, r0.y_eq,
                             int(r0["mode"]), 0.258, 0.298, 0.0008)
    nu = [r["nu"] for r in rows]; amp = [r["amplitude"] for r in rows]
    m14, a14, nu14, x14 = summarise(nu, amp)
    print(f"[rtol=1e-14] n={len(rows)} min|nu+2|={m14:.3e} at amp={a14:.4f} "
          f"nu={nu14:.9f} crosses(-2)={x14}")
    print(f"\ndifference in min|nu+2|: {abs(m14 - m12):.1e}   "
          f"turn-back reproduced: {not x14}")

    pd.DataFrame([
        dict(rtol=1e-12, n=len(ultra), min_abs_nu_plus2=m12, amp_at_min=a12,
             nu_at_min=nu12, crosses_minus2=x12),
        dict(rtol=1e-14, n=len(rows), min_abs_nu_plus2=m14, amp_at_min=a14,
             nu_at_min=nu14, crosses_minus2=x14),
    ]).to_csv(os.path.join(RESULTS, "period_doubling_tolerance.csv"), index=False)
    print("wrote", os.path.join(RESULTS, "period_doubling_tolerance.csv"))


if __name__ == "__main__":
    main()
