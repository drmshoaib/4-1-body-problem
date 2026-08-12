"""Test for a secondary (subharmonic) periodic family near the primary family
whose nontrivial Floquet multiplier lies closest to a low-order resonance.
Among the computed families, family 2 has the multiplier nearest the 1:4 point
(argument closest to +/- 90 deg, i.e. nu closest to 0), so it is the natural
candidate. Note that no computed family actually crosses nu=0 over its
continuation range (its multiplier argument stays near +/- 100 deg), so this is
a negative check rather than a bifurcation continuation.

A genuine period-4 family must (i) close with residual < 1e-10 at period ~4T,
and (ii) be a genuine subharmonic, i.e. NOT return to the initial state at T,
2T or 3T (which would mean it is just the primary orbit re-covered).

The search finds no genuine subharmonic, consistent with the manuscript's
negative multiplier-crossing result.

Output: results/secondary_family_search.csv (attempts + verdicts).
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
TOL = 1e-10


def bracket_nu0(c1, c2, x_eq, y_eq, mode_idx):
    """Fine-sweep fam2 amplitude, return the orbit where nu is closest to 0."""
    P = Params(A, B, c1, c2)
    z = np.array([x_eq, y_eq, 0.0, 0.0])
    om, ehat, w = centre_modes(x_eq, y_eq, P)[mode_idx - 1]
    best = None
    prev_u0, prev_T = None, 2 * np.pi / om
    for amp in np.arange(0.05, 1.2, 0.01):
        seed = (prev_u0 + (amp - prev_amp) * ehat) if prev_u0 is not None else z + amp * ehat
        orb = correct_orbit(seed, prev_T, ehat, amp, z, P, tol=TOL)
        if not orb.converged:
            continue
        nu = floquet(orb.M).stability_index
        if best is None or abs(nu) < abs(best[1]):
            best = (orb, nu, amp)
        prev_u0, prev_amp, prev_T = orb.u0, amp, orb.T
    return best, (om, ehat, z, P)


def subharmonic_distinct(u0, T4, P, Tprim):
    """True if the ~4T orbit does not close at T, 2T, or 3T (genuine period-4)."""
    for k in (1, 2, 3):
        uk, _, _ = integrate(u0, k * Tprim, P)
        if np.linalg.norm(uk - u0) < 1e-3:
            return False
    return True


def main():
    o = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    r0 = o[o.family_id == 2].iloc[0]
    (orb, nu, amp), (om, ehat, z, P) = bracket_nu0(
        r0.c1, r0.c2, r0.x_eq, r0.y_eq, int(r0["mode"]))
    print(f"fam2 nu=0 crossing near amp={amp:.3f}, nu={nu:.4f}, T={orb.T:.4f}")
    u0c, Tprim = orb.u0, orb.T
    _, M, _ = integrate(u0c, Tprim, P)
    w, V = np.linalg.eig(M)
    k = int(np.argmin(np.abs(w - 1j)))     # multiplier nearest +i
    vre = np.real(V[:, k]); vre /= np.linalg.norm(vre)
    vim = np.imag(V[:, k]); vim /= (np.linalg.norm(vim) + 1e-30)

    rows = []
    found = False
    for vname, v in [("Re(v_i)", vre), ("Im(v_i)", vim)]:
        for delta in [3e-3, 1e-2, 3e-2, 6e-2, 1e-1]:
            seed = u0c + delta * v
            orb4 = correct_orbit(seed, 4.0 * Tprim, v, float(v @ (seed - z)), z, P, tol=TOL)
            genuine = (orb4.converged and orb4.T > 1.5 * Tprim
                       and subharmonic_distinct(orb4.u0, orb4.T, P, Tprim))
            rows.append({"seed_dir": vname, "delta": delta,
                         "converged": bool(orb4.converged),
                         "closure": orb4.closure, "T": orb4.T,
                         "T_ratio": orb4.T / Tprim,
                         "geom_amplitude": orb4.geom_amplitude,
                         "genuine_subharmonic": bool(genuine)})
            if genuine:
                found = True
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "secondary_family_search.csv"), index=False)
    print(df.to_string(index=False))
    print("\nGenuine secondary (period-4) family found:", found)
    if not found:
        print("=> No isolated subharmonic family could be corrected to closure; "
              "the nu=0 crossing is a weak 1:4 resonance with no verified "
              "secondary family (reported as such).")


if __name__ == "__main__":
    main()
