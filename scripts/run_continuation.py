"""Continuation and bifurcation driver.

Traces the admissible arcs of the four-body CC manifold psi=0 at (a,b)=(0.9,-0.1)
through the verified Case-1 and Case-2 configurations, continues the restricted
fifth-body equilibria along each arc with an INDEPENDENT global root search at
every sample, tracks branches, locates and classifies count-changing
bifurcations via the augmented system Ox=Oy=detH=0 (with psi=0), and writes all
result CSVs.

KEY EMPIRICAL FINDING (see docs/continuation_results.md): the refined Case-1 and
Case-2 configurations lie on DIFFERENT connected components of psi=0 at fixed
(a,b)=(0.9,-0.1); there is no (c1,c2)-continuation path connecting them. The
5/7/9 equilibrium-count structure occurs WITHIN each component's admissible arc.

Run:  python scripts/run_continuation.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from r5bp_nonsymmetric.central_config import mass_ratios, masses_from_ratios, check_admissibility
from r5bp_nonsymmetric.equilibria import find_equilibria
from r5bp_nonsymmetric.geometry import primary_positions
from r5bp_nonsymmetric.continuation import cc_tangent
from r5bp_nonsymmetric.continuation_bif import (
    trace_admissible_arc, solve_augmented, classify_fold,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

A, B = 0.9, -0.1
DOMAIN = (-8.0, 8.0, -8.0, 8.0)
GRID_N = 81
N_RANDOM = 1000
SAMPLE_DS = 0.015         # arclength stride for sampling along an arc
MATCH_TOL = 0.25          # branch nearest-neighbour matching tolerance


def endpoints():
    df = pd.read_csv(os.path.join(RESULTS, "refined_reference_cases.csv"))
    p1 = df[df.case == "Case 1"].iloc[0]
    p2 = df[df.case == "Case 2"].iloc[0]
    return ((float(p1.c1_refined), float(p1.c2_refined)),
            (float(p2.c1_refined), float(p2.c2_refined)))


def global_search(c1, c2):
    """Independent global search, robust near small mass ratios: a global grid
    plus dense local grids around every primary (where light-mass equilibria
    cluster and a uniform grid can miss them). Results are merged and
    de-duplicated, keeping the lowest-residual representative."""
    m = masses_from_ratios(*mass_ratios(A, B, c1, c2))
    cand = list(find_equilibria(A, B, c1, c2, m, domain=DOMAIN,
                                grid_n=GRID_N, n_random=N_RANDOM))
    P = primary_positions(A, B, c1, c2)
    for (px, py) in P:
        cand += list(find_equilibria(A, B, c1, c2, m,
                                     domain=(px - 0.8, px + 0.8, py - 0.8, py + 0.8),
                                     grid_n=35, n_random=0))
    uniq = []
    for e in cand:
        placed = False
        for i, u in enumerate(uniq):
            if np.hypot(e.x - u.x, e.y - u.y) < 1e-5:
                if e.residual < u.residual:
                    uniq[i] = e
                placed = True
                break
        if not placed:
            uniq.append(e)
    uniq.sort(key=lambda e: (round(e.x, 6), round(e.y, 6)))
    return uniq


def subsample(points, ds=SAMPLE_DS):
    out = [points[0]]
    last = points[0].arclen
    for p in points[1:-1]:
        if p.arclen - last >= ds:
            out.append(p); last = p.arclen
    out.append(points[-1])
    return out


def match_branches(prev, cur, next_id, tol=MATCH_TOL):
    """Assign persistent IDs to cur (list of Equilibrium) from prev
    (list of (id,x,y)). Returns (assignments list of ids, updated next_id)."""
    ids = [None] * len(cur)
    used = set()
    for i, e in enumerate(cur):
        best, bd = None, tol
        for (pid, px, py) in prev:
            if pid in used:
                continue
            d = np.hypot(e.x - px, e.y - py)
            if d < bd:
                bd, best = d, pid
        if best is not None:
            ids[i] = best; used.add(best)
    for i in range(len(cur)):
        if ids[i] is None:
            ids[i] = next_id; next_id += 1
    return ids, next_id


def analyze_arc(name, start):
    print(f"\n=== {name}: tracing admissible arc through {start} ===")
    pts, bminus, bplus = trace_admissible_arc(A, B, start, ds0=0.01)
    print(f"  admissible arc: {len(pts)} pts, s in [0,1], "
          f"arclen {pts[0].arclen:.3f}..{pts[-1].arclen:.3f}")
    print(f"  boundary- rho->0: {bminus}")
    print(f"  boundary+ rho->0: {bplus}")

    samples = subsample(pts)
    print(f"  sampling {len(samples)} points for global search ...")

    path_rows, count_rows, branch_rows = [], [], []
    per_sample = []   # (s, c1, c2, list[Equilibrium], ids)
    per_sample_arclen = []
    prev = []
    next_id = 1
    for k, p in enumerate(samples):
        eqs = global_search(p.c1, p.c2)
        ids, next_id = match_branches(prev, eqs, next_id)
        prev = [(ids[i], eqs[i].x, eqs[i].y) for i in range(len(eqs))]
        per_sample.append((p.s, p.c1, p.c2, eqs, ids))
        per_sample_arclen.append(p.arclen)
        count_rows.append({"component": name, "s": p.s, "c1": p.c1, "c2": p.c2,
                           "N_eq": len(eqs)})
        for e, bid in zip(eqs, ids):
            branch_rows.append({
                "component": name, "s": p.s, "branch_id": bid,
                "x": e.x, "y": e.y, "residual": e.residual,
                "det_hessian": e.hess_det, "jacobi": e.jacobi,
                "max_real_eigenvalue": e.max_real_part})
        if k % 10 == 0:
            print(f"    s={p.s:.3f} N_eq={len(eqs)}")

    for p in pts:
        path_rows.append({"component": name, "s": p.s, "c1": p.c1, "c2": p.c2,
                          "rho0": p.rho0, "rho1": p.rho1, "rho2": p.rho2,
                          "psi_residual": p.psi_residual})

    # ------- count changes + bisection bracketing of each transition -------
    fine_al = np.array([p.arclen for p in pts])
    total_al = pts[-1].arclen - pts[0].arclen

    def cc_at(al):
        j = int(np.argmin(np.abs(fine_al - al)))
        return pts[j].c1, pts[j].c2, pts[j].s

    def count_at(al):
        c1, c2, s = cc_at(al)
        r = global_search(c1, c2)
        return len(r), r, c1, c2, s

    def closest_pair_mid(roots):
        """Midpoint of the closest pair of roots. On a TIGHT bracket this is the
        pair about to merge at the fold -- the correct augmented-solve guess."""
        if len(roots) < 2:
            return None
        xy = np.array([[e.x, e.y] for e in roots])
        best, pair = 1e9, None
        for ii in range(len(xy)):
            for jj in range(ii + 1, len(xy)):
                dd = np.hypot(*(xy[ii] - xy[jj]))
                if dd < best:
                    best, pair = dd, (ii, jj)
        return (float(0.5 * (xy[pair[0]][0] + xy[pair[1]][0])),
                float(0.5 * (xy[pair[0]][1] + xy[pair[1]][1])))

    transitions = []
    for i in range(len(per_sample) - 1):
        na = len(per_sample[i][3]); nb = len(per_sample[i + 1][3])
        if na != nb:
            transitions.append((i, na, nb))
    print(f"  count changes at {len(transitions)} sample gaps: "
          + ", ".join(f"{per_sample[i][0]:.3f}:{na}->{nb}" for (i, na, nb) in transitions))

    bif_rows = []
    for (i, na, nb) in transitions:
        al_a = per_sample_arclen[i]; al_b = per_sample_arclen[i + 1]
        n_lo, r_lo, c1_lo, c2_lo, _ = count_at(al_a)
        n_hi, r_hi, c1_hi, c2_hi, _ = count_at(al_b)
        # Identify THIS transition's pair at the (wide) sample gap, where the
        # appearing/disappearing roots are well separated from other roots.
        hi0 = r_hi if n_hi > n_lo else r_lo
        lo0 = r_lo if n_hi > n_lo else r_hi
        pair_pos = [(e.x, e.y) for e in hi0
                    if min((np.hypot(e.x - f.x, e.y - f.y) for f in lo0), default=9) > MATCH_TOL]
        # bisection: narrow to a tight bracket around the FIRST change from al_a
        for _ in range(7):
            mid = 0.5 * (al_a + al_b)
            nm, rm, c1m, c2m, _ = count_at(mid)
            if nm == n_lo:
                al_a, r_lo, c1_lo, c2_lo = mid, rm, c1m, c2m
            else:
                al_b, n_hi, r_hi, c1_hi, c2_hi = mid, nm, rm, c1m, c2m
        dN = n_hi - n_lo
        near_boundary = (i == 0) or (i + 1 == len(per_sample) - 1)
        s_c = cc_at(0.5 * (al_a + al_b))[2]
        gc1, gc2 = 0.5 * (c1_lo + c1_hi), 0.5 * (c2_lo + c2_hi)

        # Guess: track THIS transition's pair into the tight bracket -- the two
        # roots on the higher-count tight side nearest the identified pair.
        def guess_from_pair(tight_roots):
            if len(pair_pos) == 2 and len(tight_roots) >= 2:
                chosen = []
                for (px, py) in pair_pos:
                    e = min(tight_roots, key=lambda r: np.hypot(r.x - px, r.y - py))
                    chosen.append((e.x, e.y))
                return (0.5 * (chosen[0][0] + chosen[1][0]),
                        0.5 * (chosen[0][1] + chosen[1][1]))
            return closest_pair_mid(tight_roots)

        classification = "boundary/mass-vanishing" if abs(dN) % 2 == 1 else "candidate saddle-node"
        crit = None; fold = None
        if abs(dN) == 2:
            g = guess_from_pair(r_hi if n_hi > n_lo else r_lo)
            if g is not None:
                crit = solve_augmented([g[0], g[1], gc1, gc2], A, B)
                if crit.converged and all(r > 0 for r in crit.rho):
                    fold = classify_fold(crit, A, B)
                    classification = ("saddle-node (fold)" if fold.is_generic_fold
                                      else "degenerate/near-fold (nondegeneracy not confirmed)")
                else:
                    classification = "augmented solve did not converge to interior CC point"
        sa = cc_at(al_a)[2]; sb = cc_at(al_b)[2]
        row = {"component": name, "s_a": sa, "s_b": sb, "count_change": f"{na}->{nb}",
               "dN": dN, "near_arc_boundary": near_boundary,
               "s_c": s_c, "classification": classification}
        if crit is not None:
            row.update({"x_c": crit.x, "y_c": crit.y, "c1_c": crit.c1, "c2_c": crit.c2,
                        "aug_residual": crit.residual,
                        "rho0_c": crit.rho[0], "rho1_c": crit.rho[1], "rho2_c": crit.rho[2]})
        if fold is not None:
            row.update({"mu_small": fold.mu_small, "mu_large": fold.mu_large,
                        "alpha": fold.alpha, "beta": fold.beta,
                        "is_generic_fold": fold.is_generic_fold})
        bif_rows.append(row)
        print(f"    transition {na}->{nb} at s~{s_c:.3f}: {classification}"
              + (f"  crit=({crit.x:.4f},{crit.y:.4f}) c=({crit.c1:.4f},{crit.c2:.4f}) res={crit.residual:.1e}"
                 if crit else ""))

    return path_rows, count_rows, branch_rows, bif_rows, (bminus, bplus)


def main():
    (P1, P2) = endpoints()
    all_path, all_count, all_branch, all_bif = [], [], [], []
    boundaries = {}
    for name, start in [("Case1-arc", P1), ("Case2-loop", P2)]:
        pr, cr, br, bf, bnd = analyze_arc(name, start)
        all_path += pr; all_count += cr; all_branch += br; all_bif += bf
        boundaries[name] = bnd

    pd.DataFrame(all_path).to_csv(os.path.join(RESULTS, "cc_continuation_path.csv"), index=False)
    pd.DataFrame(all_count).to_csv(os.path.join(RESULTS, "equilibrium_count_vs_s.csv"), index=False)
    pd.DataFrame(all_branch).to_csv(os.path.join(RESULTS, "equilibrium_branches.csv"), index=False)
    pd.DataFrame(all_bif).to_csv(os.path.join(RESULTS, "bifurcation_points.csv"), index=False)
    print("\nWrote results/cc_continuation_path.csv, equilibrium_count_vs_s.csv,")
    print("      equilibrium_branches.csv, bifurcation_points.csv")


if __name__ == "__main__":
    main()
