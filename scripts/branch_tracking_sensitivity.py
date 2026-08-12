"""Branch-tracking threshold sensitivity study (referee-response evidence).

Re-stitches the Case-2 equilibrium branches with tighter and looser match /
reliability thresholds and checks that the branch-exchange identification
(pair B_a created at F4 and persisting to F6; pair B_b destroyed at F5) is
invariant. Also reports the predictor-to-match distance / acceptance-gate
ratios through the fold sequence.

Match gate      : d <= max(g0, kg*step)
Reliability     : d <= max(r0, kr*step)
Default (paper) : (g0,kg,r0,kr) = (0.18, 3, 0.25, 2)

Reads:  results/equilibrium_branches.csv
Writes: results/branch_tracking_sensitivity.csv

Run:  python scripts/branch_tracking_sensitivity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")

# Case-2 fold s_c locations (from bifurcation_points.csv)
F3F6 = (0.3660, 0.8393)   # full seven-fold span
N11 = (0.7679, 0.8036)    # eleven-equilibrium interval (F4..F5)
F4F6 = (0.7679, 0.8393)   # B_a persistence window (created F4, persists to F6)

VARIANTS = {
    "default(0.18,3,0.25,2)": (0.18, 3.0, 0.25, 2.0),
    "tight(0.12,2,0.15,1.5)": (0.12, 2.0, 0.15, 1.5),
    "loose(0.30,4,0.40,3)": (0.30, 4.0, 0.40, 3.0),
    "gate-tight(0.12,2,0.25,2)": (0.12, 2.0, 0.25, 2.0),
}


def stitch(df_comp, g0, kg, r0, kr):
    svals = sorted(df_comp["s"].unique())
    active = []; next_id = 1; out = []; diag = []
    for s in svals:
        pts = df_comp[df_comp["s"] == s][["x", "y"]].values.tolist()
        preds = []
        for a in active:
            h = a["hist"]
            if len(h) >= 2:
                pr = (2 * h[-1][0] - h[-2][0], 2 * h[-1][1] - h[-2][1])
                step = np.hypot(h[-1][0] - h[-2][0], h[-1][1] - h[-2][1])
            else:
                pr = h[-1]; step = 0.1
            preds.append((pr, step))
        cand = []
        for ai, a in enumerate(active):
            (pr, step) = preds[ai]
            for pi, p in enumerate(pts):
                cand.append((np.hypot(p[0] - pr[0], p[1] - pr[1]), ai, pi, step))
        cand.sort()
        ma, mp = set(), set(); rows = [None] * len(pts)
        for dd, ai, pi, step in cand:
            if ai in ma or pi in mp:
                continue
            gate = max(g0, kg * step)
            if dd > gate:
                continue
            ma.add(ai); mp.add(pi)
            reli = max(r0, kr * step)
            active[ai]["hist"].append((pts[pi][0], pts[pi][1]))
            rows[pi] = (active[ai]["id"], dd <= reli)
            diag.append((s, dd, gate, reli, dd / gate))
        for pi, p in enumerate(pts):
            if rows[pi] is None:
                active.append({"id": next_id, "hist": [(p[0], p[1])]})
                rows[pi] = (next_id, True); next_id += 1
        for pi, p in enumerate(pts):
            out.append((round(s, 6), round(p[0], 6), round(p[1], 6), rows[pi][0], rows[pi][1]))
        touched = {r[0] for r in rows}
        active = [a for a in active if a["id"] in touched]
    return out, diag, next_id - 1


def partition(out, window):
    return {(s, x, y): bid for (s, x, y, bid, rel) in out if window[0] <= s <= window[1]}


def same_partition(pA, pB):
    keys = sorted(set(pA) & set(pB))
    def groups(p):
        inv = {}
        for k in keys:
            inv.setdefault(p[k], set()).add(k)
        return {frozenset(v) for v in inv.values()}
    return groups(pA) == groups(pB), len(keys)


def main():
    br = pd.read_csv(os.path.join(RESULTS, "equilibrium_branches.csv"))
    d2 = br[br.component == "Case2-loop"].copy()

    res = {}
    for name, (g0, kg, r0, kr) in VARIANTS.items():
        out, diag, nbr = stitch(d2, g0, kg, r0, kr)
        dg = pd.DataFrame(diag, columns=["s", "d", "gate", "reli", "ratio"])
        inf = dg[(dg.s >= F3F6[0]) & (dg.s <= F3F6[1])]
        inn = dg[(dg.s >= N11[0]) & (dg.s <= N11[1])]
        res[name] = dict(out=out, nbr=nbr,
                         max_ratio_all=float(dg.ratio.max()),
                         max_ratio_F3F6=float(inf.ratio.max()),
                         max_ratio_N11=float(inn.ratio.max()),
                         n_unreliable_F3F6=int((inf.d > inf.reli).sum()))

    pdef = res["default(0.18,3,0.25,2)"]
    rows = []
    for name, r in res.items():
        inv_n11, _ = same_partition(partition(pdef["out"], N11), partition(r["out"], N11))
        inv_bax, _ = same_partition(partition(pdef["out"], F4F6), partition(r["out"], F4F6))
        rows.append(dict(variant=name, n_branches=r["nbr"],
                         max_ratio_all=r["max_ratio_all"],
                         max_ratio_F3_F6=r["max_ratio_F3F6"],
                         max_ratio_N11=r["max_ratio_N11"],
                         n_unreliable_F3_F6=r["n_unreliable_F3F6"],
                         partition_invariant_N11=inv_n11,
                         partition_invariant_F4_F6=inv_bax))
        print(f"{name:26s} nbr={r['nbr']:2d} maxD/gate(all)={r['max_ratio_all']:.3f} "
              f"F3-F6={r['max_ratio_F3F6']:.3f} N11={r['max_ratio_N11']:.3f} "
              f"unrel@F3-F6={r['n_unreliable_F3F6']} "
              f"inv(N11)={inv_n11} inv(F4-F6)={inv_bax}")

    out = os.path.join(RESULTS, "branch_tracking_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
