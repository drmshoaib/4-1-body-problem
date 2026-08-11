"""Targeted period-doubling refinement of families 8 and 10.

Fine amplitude sweep near the minimum of |nu+2| to decide whether the
nontrivial Floquet multiplier actually crosses -1 (nu crosses -2), which is the
necessary condition for a period-doubling bifurcation. If it crosses, locate the
critical orbit and attempt to seed and correct a period-doubled (~2T) family.

Outputs:
    results/period_doubling_scan.csv
    results/period_doubled_family.csv   (only if a genuine crossing + 2T family)

Run:  python scripts/refine_period_doubling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from r5bp_nonsymmetric.periodic import (
    Params, centre_modes, correct_orbit, floquet, integrate,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
A, B = 0.9, -0.1
CLOSURE_TOL = 1e-10


def fine_sweep(fam_label, c1, c2, x_eq, y_eq, mode_idx, amp_lo, amp_hi, step):
    """Continue the family in amplitude with small steps; record Floquet nu."""
    P = Params(A, B, c1, c2)
    z = np.array([x_eq, y_eq, 0.0, 0.0])
    modes = centre_modes(x_eq, y_eq, P)
    om, ehat, w = modes[mode_idx - 1]
    T0 = 2 * np.pi / om
    rows = []
    prev_u0 = None
    prev_T = T0
    amps = np.arange(amp_lo, amp_hi + 0.5 * step, step)
    for amp in amps:
        seed = (prev_u0 + (amp - prev_amp) * ehat) if prev_u0 is not None \
            else z + amp * ehat
        orb = correct_orbit(seed, prev_T, ehat, amp, z, P, tol=CLOSURE_TOL)
        if not orb.converged:
            continue
        fl = floquet(orb.M)
        ev = fl.multipliers
        row = {"family": fam_label, "amplitude": amp,
               "geom_amplitude": orb.geom_amplitude, "T": orb.T,
               "jacobi_CJ": orb.jacobi, "closure": orb.closure,
               "nu": fl.stability_index, "abs_nu_plus2": abs(fl.stability_index + 2),
               "nontriv1_re": fl.nontrivial[0].real, "nontriv1_im": fl.nontrivial[0].imag,
               "nontriv1_abs": abs(fl.nontrivial[0]),
               "classification": fl.classification}
        for i in range(4):
            row[f"mult{i+1}_re"] = ev[i].real
            row[f"mult{i+1}_im"] = ev[i].imag
        rows.append(row)
        prev_u0, prev_amp, prev_T = orb.u0, amp, orb.T
    return rows, (om, ehat, z, P)


def detect_crossing(rows):
    """Return (crosses, amp_min, nu_min) for nu+2 along the sweep."""
    nu = np.array([r["nu"] for r in rows])
    amp = np.array([r["amplitude"] for r in rows])
    s = nu + 2.0
    crosses = bool(np.any(s[:-1] * s[1:] < 0))
    imin = int(np.argmin(np.abs(s)))
    return crosses, float(amp[imin]), float(nu[imin])


def attempt_period_doubled(crit_orb_row, parent):
    """Given a critical orbit (nu approx -2), seed a ~2T orbit along the
    eigenvector of the -1 multiplier and correct it. Returns dict or None."""
    om, ehat, z, P = parent
    u0 = np.array([crit_orb_row["x0"], crit_orb_row["y0"],
                   crit_orb_row["vx0"], crit_orb_row["vy0"]])
    T = crit_orb_row["T"]
    _, M, _ = integrate(u0, T, P)
    w, V = np.linalg.eig(M)
    # eigenvector for the multiplier closest to -1
    k = int(np.argmin(np.abs(w + 1.0)))
    v = np.real(V[:, k]); v = v / np.linalg.norm(v)
    for delta in [1e-3, 3e-3, 1e-2, 3e-2]:
        seed = u0 + delta * v
        # phase/amplitude anchor along v, target 2T period
        orb = correct_orbit(seed, 2.0 * T, v, float(v @ (seed - z)), z, P,
                            tol=CLOSURE_TOL)
        if orb.converged and orb.T > 1.4 * T:   # genuine ~2T, not the T orbit
            fl = floquet(orb.M)
            return {"delta": delta, "T": orb.T, "T_ratio": orb.T / T,
                    "closure": orb.closure, "jacobi_CJ": orb.jacobi,
                    "geom_amplitude": orb.geom_amplitude,
                    "nu": fl.stability_index, "classification": fl.classification,
                    "x0": orb.u0[0], "y0": orb.u0[1], "vx0": orb.u0[2], "vy0": orb.u0[3]}
    return None


def main():
    o = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    all_rows = []
    pd_family_rows = []
    for fid, lo, hi, st in [(8, 0.001, 0.02, 0.001), (10, 0.15, 0.34, 0.004)]:
        r0 = o[o.family_id == fid].iloc[0]
        label = f"fam{fid}_{r0.label}"
        rows, parent = fine_sweep(label, r0.c1, r0.c2, r0.x_eq, r0.y_eq,
                                  int(r0["mode"]), lo, hi, st)
        all_rows += rows
        crosses, amp_min, nu_min = detect_crossing(rows)
        print(f"{label}: {len(rows)} fine pts, min|nu+2| at amp={amp_min:.4f} "
              f"nu_min={nu_min:.6f}, crosses(-2)={crosses}")
        if fid == 10 and not crosses:
            # refine even finer around the minimum to be sure
            rows2, parent = fine_sweep(label + "_ultra", r0.c1, r0.c2, r0.x_eq,
                                       r0.y_eq, int(r0["mode"]),
                                       amp_min - 0.02, amp_min + 0.02, 0.0008)
            all_rows += rows2
            crosses2, amp_min2, nu_min2 = detect_crossing(rows2)
            print(f"  ultra-fine: min|nu+2| at amp={amp_min2:.4f} "
                  f"nu_min={nu_min2:.7f}, crosses(-2)={crosses2}")
            crosses = crosses2
        if crosses:
            # locate critical orbit (nearest nu to -2) and attempt 2T seed
            cand = min([r for r in rows], key=lambda r: abs(r["nu"] + 2))
            # need x0.. -> re-correct at that amplitude to get full state
            om, ehat, z, P = parent
            orb = correct_orbit(z + cand["amplitude"] * ehat, cand["T"], ehat,
                                cand["amplitude"], z, P, tol=CLOSURE_TOL)
            crit = {"x0": orb.u0[0], "y0": orb.u0[1], "vx0": orb.u0[2],
                    "vy0": orb.u0[3], "T": orb.T}
            pd2 = attempt_period_doubled(crit, parent)
            if pd2:
                pd2["parent_family"] = label
                pd_family_rows.append(pd2)
                print(f"  --> period-doubled orbit corrected: T_ratio={pd2['T_ratio']:.3f} "
                      f"closure={pd2['closure']:.1e} {pd2['classification']}")
            else:
                print("  --> no distinct 2T orbit could be corrected")

    pd.DataFrame(all_rows).to_csv(os.path.join(RESULTS, "period_doubling_scan.csv"), index=False)
    if pd_family_rows:
        pd.DataFrame(pd_family_rows).to_csv(os.path.join(RESULTS, "period_doubled_family.csv"), index=False)
    print(f"\nWrote period_doubling_scan.csv ({len(all_rows)} rows); "
          f"period-doubled family rows: {len(pd_family_rows)}")


if __name__ == "__main__":
    main()
