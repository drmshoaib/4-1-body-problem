"""Compute, continue, and classify periodic-orbit families around centre-mode
and spectrally stable equilibria of the restricted five-body problem.

Genuine differential-corrected orbits only (closure < 1e-10); Floquet stability
from the integrated monodromy. Saves:
    results/periodic_orbits.csv
    results/periodic_orbit_floquet.csv
    results/periodic_orbit_seeds.csv   (every attempted seed + outcome)

Run:  python scripts/run_periodic_orbits.py
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
from r5bp_nonsymmetric.geometry import primary_positions

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
A, B = 0.9, -0.1

# ---- continuation / acceptance settings (see reproducibility protocol) ----
CLOSURE_TOL = 1e-10
AMP0 = 2e-3           # initial ehat-projection amplitude
AMP_GROWTH = 1.35
MAX_MEMBERS = 40
COLLISION_R = 0.05    # stop if orbit comes within this of a primary
GEOM_CAP = 2.5        # stop beyond this geometric amplitude
RES_TOL = 0.04


def resolve_equilibrium(spec, component, s_target, min_centre=1,
                        near_xy=None, prefer=None):
    d = spec[(spec.component == component) & spec.reliable
             & (spec.n_centre_pairs >= min_centre)].copy()
    d = d[(d.s - s_target).abs() < 0.06]
    if len(d) == 0:
        return None
    if near_xy is not None:
        d["key"] = np.hypot(d.x - near_xy[0], d.y - near_xy[1])
        return d.sort_values("key").iloc[0]
    if prefer == "max_omega":
        return d.sort_values("omega1", ascending=False).iloc[0]
    d["key"] = (d.n_centre_pairs.max() - d.n_centre_pairs) + (d.s - s_target).abs()
    return d.sort_values("key").iloc[0]


def winding(traj, px, py):
    ang = np.arctan2(traj[:, 1] - py, traj[:, 0] - px)
    return float(np.sum(np.diff(np.unwrap(ang))) / (2 * np.pi))


def geometry_flags(u0, T, z_eq, P):
    from r5bp_nonsymmetric.periodic import IntegrationError
    try:
        _, _, traj = integrate(u0, T, P, n_eval=800)
    except IntegrationError:
        return [0, 0, 0, 0], 0, 0, ""
    P4 = primary_positions(P.a, P.b, P.c1, P.c2)
    winds = [round(winding(traj, P4[i, 0], P4[i, 1])) for i in range(4)]
    cx, cy = np.mean(P4[:, 0]), np.mean(P4[:, 1])
    wcen = round(winding(traj, cx, cy))
    r = np.hypot(traj[:, 0] - z_eq[0], traj[:, 1] - z_eq[1])
    # count interior local maxima of the radius (lobe heuristic)
    peaks = int(np.sum((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:])))
    flags = []
    enc = [i for i, w in enumerate(winds) if w != 0]
    if len(enc) == 1:
        flags.append(f"encircles m{enc[0]}")
    elif len(enc) >= 2:
        flags.append("encircles m" + "".join(str(i) for i in enc))
    if abs(wcen) >= 1 and len(enc) == 4:
        flags.append("encircles all primaries")
    if peaks > 2:
        flags.append(f"multi-lobe({peaks})")
    return winds, wcen, peaks, ";".join(flags)


def run_family(fam_id, label, component, s, c1, c2, x_eq, y_eq, mode_idx,
               orbit_rows, floq_rows, seed_rows):
    P = Params(A, B, c1, c2)
    z = np.array([x_eq, y_eq, 0.0, 0.0])
    modes = centre_modes(x_eq, y_eq, P)
    if mode_idx > len(modes):
        return
    om, ehat, w = modes[mode_idx - 1]
    T0 = 2 * np.pi / om
    amp = AMP0
    prev_u0 = None
    prev_geom = None
    prev_class = None
    member = 0
    while member < MAX_MEMBERS:
        u_seed = (prev_u0 + (amp - prev_amp) * ehat) if prev_u0 is not None \
            else z + amp * ehat
        T_seed = T0 if prev_u0 is None else prev_T
        orb = correct_orbit(u_seed, T_seed, ehat, amp, z, P, tol=CLOSURE_TOL)
        fl = floquet(orb.M)
        # reject spurious/degenerate solutions:
        #  - T collapsed toward 0 (u(0)=u0 closes trivially)
        #  - Newton jumped to a qualitatively different (much larger) orbit
        reject = ""
        if orb.converged and orb.T < 0.4 * T0:
            reject = "period-collapse"
        elif orb.converged and prev_u0 is not None and \
                orb.geom_amplitude > 2.0 * prev_geom:
            reject = "orbit-jump"
        seed_rows.append({"family_id": fam_id, "label": label, "mode": mode_idx,
                          "amp": amp, "converged": orb.converged,
                          "closure": orb.closure, "geom_amp": orb.geom_amplitude,
                          "reject": reject})
        if not orb.converged or reject:
            break
        winds, wcen, peaks, flags = geometry_flags(orb.u0, orb.T, z[:2], P)
        # stopping conditions
        stop = ""
        if min(orb.min_dist) < COLLISION_R:
            stop = "near-primary"
        elif orb.geom_amplitude > GEOM_CAP:
            stop = "amplitude-cap"
        if prev_class is not None and fl.classification.split(" (")[0] != prev_class:
            flags = (flags + ";" if flags else "") + "stability-change"
        prev_class = fl.classification.split(" (")[0]

        member += 1
        orbit_rows.append({
            "family_id": fam_id, "member": member, "label": label,
            "component": component, "s": s, "c1": c1, "c2": c2,
            "x_eq": x_eq, "y_eq": y_eq, "mode": mode_idx, "omega_parent": om,
            "amplitude": amp, "geom_amplitude": orb.geom_amplitude,
            "x0": orb.u0[0], "y0": orb.u0[1], "vx0": orb.u0[2], "vy0": orb.u0[3],
            "T": orb.T, "closure": orb.closure, "jacobi_CJ": orb.jacobi,
            "min_d0": orb.min_dist[0], "min_d1": orb.min_dist[1],
            "min_d2": orb.min_dist[2], "min_d3": orb.min_dist[3],
            "wind_m0": winds[0], "wind_m1": winds[1], "wind_m2": winds[2],
            "wind_m3": winds[3], "wind_centroid": wcen, "n_lobes": peaks,
            "flags": flags, "stop": stop,
        })
        ev = fl.multipliers
        frow = {"family_id": fam_id, "member": member,
                "trivial_err": fl.trivial_err,
                "stability_index": fl.stability_index,
                "classification": fl.classification}
        for i in range(4):
            frow[f"mult{i+1}_re"] = ev[i].real
            frow[f"mult{i+1}_im"] = ev[i].imag
            frow[f"mult{i+1}_abs"] = abs(ev[i])
        nt = fl.nontrivial
        frow["nontriv_arg"] = float(np.angle(nt[0]))
        floq_rows.append(frow)

        if stop:
            break
        prev_u0, prev_amp, prev_T, prev_geom = orb.u0, amp, orb.T, orb.geom_amplitude
        amp *= AMP_GROWTH
    print(f"  family {fam_id} [{label} mode{mode_idx} om={om:.3f}]: "
          f"{member} members, last stop='{orbit_rows[-1]['stop'] if member else 'seed-fail'}'")


def main():
    spec = pd.read_csv(os.path.join(RESULTS, "branch_spectra.csv"))
    targets = [
        ("case1_stable", "Case1-arc", 0.75, dict(min_centre=2), "both"),
        ("case2_stable", "Case2-loop", 0.15, dict(min_centre=2), "both"),
        ("case1_saddlecentre", "Case1-arc", 0.30, dict(min_centre=1, prefer="max_omega"), "first"),
        ("case2_saddlecentre", "Case2-loop", 0.55, dict(min_centre=1, prefer="max_omega"), "first"),
        ("case1_res_3to2", "Case1-arc", 0.68, dict(min_centre=2), "both"),
        ("case2_res_2to1", "Case2-loop", 0.196, dict(min_centre=2), "both"),
    ]
    orbit_rows, floq_rows, seed_rows = [], [], []
    fam_id = 0
    for (label, comp, s_t, kw, which) in targets:
        row = resolve_equilibrium(spec, comp, s_t, **kw)
        if row is None:
            print(f"  [skip] {label}: no matching equilibrium")
            continue
        print(f"* {label}: s={row.s:.3f} (c1,c2)=({row.c1:.4f},{row.c2:.4f}) "
              f"eq=({row.x:.4f},{row.y:.4f}) ncentre={int(row.n_centre_pairs)}")
        n_modes = 2 if which == "both" else 1
        P = Params(A, B, row.c1, row.c2)
        avail = len(centre_modes(row.x, row.y, P))
        for mi in range(1, min(n_modes, avail) + 1):
            fam_id += 1
            run_family(fam_id, label, comp, row.s, row.c1, row.c2,
                       row.x, row.y, mi, orbit_rows, floq_rows, seed_rows)

    pd.DataFrame(orbit_rows).to_csv(os.path.join(RESULTS, "periodic_orbits.csv"), index=False)
    pd.DataFrame(floq_rows).to_csv(os.path.join(RESULTS, "periodic_orbit_floquet.csv"), index=False)
    pd.DataFrame(seed_rows).to_csv(os.path.join(RESULTS, "periodic_orbit_seeds.csv"), index=False)
    print(f"\nWrote periodic_orbits.csv ({len(orbit_rows)}), "
          f"periodic_orbit_floquet.csv ({len(floq_rows)}), "
          f"periodic_orbit_seeds.csv ({len(seed_rows)})")


if __name__ == "__main__":
    main()
