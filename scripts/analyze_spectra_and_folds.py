"""Post-processing analysis on the verified continuation results.

Does NOT change any established bifurcation conclusion. It (re)derives spectral
diagnostics from the branch geometry and produces the data behind the analysis
figures:

  results/branch_spectra.csv          re-stitched branches + full spectra
  results/periodic_orbit_candidates.csv  ranked centre-mode candidates (+eigvecs)
  results/snapshot_equilibria.csv     equilibria at representative Case-2 s
  results/fold_local_sweeps.csv       fine fold sweeps + pair separation
  results/fold_exponents.csv          fitted exponent gamma in d ~ |s-sc|^gamma

Run:  python scripts/analyze_spectra_and_folds.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from r5bp_nonsymmetric.central_config import mass_ratios, masses_from_ratios
from r5bp_nonsymmetric.potential import (
    linearization_matrix, omega, hessian_determinant, grad,
)
from r5bp_nonsymmetric.equilibria import find_equilibria
from r5bp_nonsymmetric.continuation import cc_tangent, _corrector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
A, B = 0.9, -0.1
TOL_RE = 1e-6        # |Re lambda| below this -> treated as pure imaginary
IM_MIN = 1e-6

# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

def spectrum(x, y, c1, c2):
    m = masses_from_ratios(*mass_ratios(A, B, c1, c2))
    Amat = linearization_matrix(x, y, A, B, c1, c2, m)
    w, V = np.linalg.eig(Amat)
    max_re = float(np.max(w.real))
    centres = []   # (omega, eigenvector) for eigenvalues +i*omega
    for k in range(4):
        if abs(w[k].real) < TOL_RE and w[k].imag > IM_MIN:
            centres.append((float(w[k].imag), V[:, k]))
    centres.sort(key=lambda t: -t[0])   # omega1 >= omega2
    om1 = centres[0][0] if len(centres) >= 1 else np.nan
    om2 = centres[1][0] if len(centres) >= 2 else np.nan
    vecs = [c[1] for c in centres]
    return max_re, len(centres), om1, om2, vecs, float(omega(x, y, A, B, c1, c2, m))


# ---------------------------------------------------------------------------
# robust branch re-stitching (predictor-based nearest neighbour)
# ---------------------------------------------------------------------------

def stitch(df_comp):
    svals = sorted(df_comp["s"].unique())
    active = []       # {id, hist:[(x,y)]}
    next_id = 1
    out = []          # (s, x, y, bid, reliable)
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
                d = np.hypot(p[0] - pr[0], p[1] - pr[1])
                cand.append((d, ai, pi, step))
        cand.sort()
        ma, mp = set(), set()
        rows = [None] * len(pts)
        for d, ai, pi, step in cand:
            if ai in ma or pi in mp:
                continue
            tol = max(0.18, 3.0 * step)
            if d > tol:
                continue
            ma.add(ai); mp.add(pi)
            reliable = d <= max(0.25, 2.0 * step)
            active[ai]["hist"].append((pts[pi][0], pts[pi][1]))
            rows[pi] = (active[ai]["id"], reliable)
        for pi, p in enumerate(pts):
            if rows[pi] is None:
                active.append({"id": next_id, "hist": [(p[0], p[1])]})
                rows[pi] = (next_id, True); next_id += 1
        for pi, p in enumerate(pts):
            out.append((s, p[0], p[1], rows[pi][0], rows[pi][1]))
        active = [active[ai] for ai in range(len(active)) if ai in ma] + \
                 [a for a in active if a["hist"][-1] in [(p[0], p[1]) for pi, p in enumerate(pts) if pi not in mp]]
        # keep only branches touched at this step (matched or newly created)
        touched_ids = {r[0] for r in rows}
        active = [a for a in active if a["id"] in touched_ids]
    return out


def build_branch_spectra():
    br = pd.read_csv(os.path.join(RESULTS, "equilibrium_branches.csv"))
    ct = pd.read_csv(os.path.join(RESULTS, "equilibrium_count_vs_s.csv"))
    br = br.merge(ct[["component", "s", "c1", "c2"]], on=["component", "s"], how="left")
    rows = []
    for comp in ["Case1-arc", "Case2-loop"]:
        d = br[br.component == comp]
        stitched = stitch(d)
        lut = {(round(s, 9), round(x, 9), round(y, 9)): (bid, rel)
               for (s, x, y, bid, rel) in stitched}
        for _, r in d.iterrows():
            bid, rel = lut[(round(r.s, 9), round(r.x, 9), round(r.y, 9))]
            max_re, ncen, om1, om2, vecs, Om = spectrum(r.x, r.y, r.c1, r.c2)
            ratio = (om1 / om2) if (ncen >= 2 and om2 > 0) else np.nan
            # escape/unreliable region: Case-1 arc, large radius near boundary
            rr = np.hypot(r.x, r.y)
            escape = (comp == "Case1-arc" and r.s > 0.85 and rr > 3.0)
            rows.append({
                "component": comp, "s": r.s, "branch_id": bid,
                "reliable": bool(rel and not escape),
                "x": r.x, "y": r.y, "c1": r.c1, "c2": r.c2, "r": rr,
                "jacobi_CJ": 2.0 * Om, "max_real": max_re,
                "n_centre_pairs": ncen, "omega1": om1, "omega2": om2, "ratio": ratio,
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "branch_spectra.csv"), index=False)
    n1 = out[out.component == "Case1-arc"].branch_id.nunique()
    n2 = out[out.component == "Case2-loop"].branch_id.nunique()
    print(f"branch_spectra.csv: {len(out)} rows; stitched branches "
          f"Case1={n1}, Case2={n2}")
    return out


# ---------------------------------------------------------------------------
# periodic-orbit candidate ranking
# ---------------------------------------------------------------------------

LOW_ORDER = [(1, 1), (2, 1), (3, 1), (4, 1), (3, 2), (4, 3), (5, 2), (5, 3)]


def resonance_flag(ratio):
    if not np.isfinite(ratio):
        return ""
    for (p, q) in LOW_ORDER:
        if abs(ratio - p / q) < 0.04:
            return f"{p}:{q}"
    return ""


def rank_candidates(spec):
    cands = []
    for comp in ["Case1-arc", "Case2-loop"]:
        d = spec[(spec.component == comp) & spec.reliable].sort_values("s")
        for bid, g in d.groupby("branch_id"):
            g = g.sort_values("s")
            # contiguous runs with a centre mode (n_centre>=1)
            has_c = g.n_centre_pairs.values >= 1
            svals = g.s.values
            i = 0
            while i < len(g):
                if not has_c[i]:
                    i += 1; continue
                j = i
                while j + 1 < len(g) and has_c[j + 1]:
                    j += 1
                sub = g.iloc[i:j + 1]
                width = float(sub.s.max() - sub.s.min())
                mid = sub.iloc[len(sub) // 2]
                full = int((sub.n_centre_pairs >= 2).any())
                cands.append({
                    "component": comp, "branch_id": int(bid),
                    "s_lo": float(sub.s.min()), "s_hi": float(sub.s.max()),
                    "width_s": width,
                    "type": "centre2 (spectrally stable)" if full else "saddle x centre",
                    "s_rep": float(mid.s), "c1": float(mid.c1), "c2": float(mid.c2),
                    "x": float(mid.x), "y": float(mid.y),
                    "omega1": float(mid.omega1), "omega2": float(mid.omega2),
                    "ratio": float(mid.ratio) if np.isfinite(mid.ratio) else np.nan,
                    "resonance": resonance_flag(mid.ratio),
                    "jacobi_CJ": float(mid.jacobi_CJ),
                })
                i = j + 1
    cd = pd.DataFrame(cands)
    if len(cd):
        # score: prefer fully-imaginary (centre2) and wide intervals
        cd["score"] = cd["width_s"] * np.where(
            cd["type"].str.startswith("centre2"), 2.0, 1.0)
        cd = cd.sort_values("score", ascending=False).reset_index(drop=True)
        cd.insert(0, "rank", cd.index + 1)
        # eigenvectors for the representative point of each candidate
        vec_rows = []
        for _, r in cd.iterrows():
            _, ncen, om1, om2, vecs, _ = spectrum(r.x, r.y, r.c1, r.c2)
            rec = {"rank": int(r["rank"])}
            for mi, v in enumerate(vecs[:2], start=1):
                for ci in range(4):
                    rec[f"v{mi}_re_{ci}"] = float(v[ci].real)
                    rec[f"v{mi}_im_{ci}"] = float(v[ci].imag)
            vec_rows.append(rec)
        cd = cd.merge(pd.DataFrame(vec_rows), on="rank", how="left")
    cd.to_csv(os.path.join(RESULTS, "periodic_orbit_candidates.csv"), index=False)
    print(f"periodic_orbit_candidates.csv: {len(cd)} centre-mode intervals")
    return cd


# ---------------------------------------------------------------------------
# geometry snapshots through the Case-2 cascade 7->9->11->9->7->5
# ---------------------------------------------------------------------------

def snapshots():
    path = pd.read_csv(os.path.join(RESULTS, "cc_continuation_path.csv"))
    d = path[path.component == "Case2-loop"].sort_values("s")
    targets = [(0.15, "7a"), (0.55, "9a"), (0.785, "11"),
               (0.82, "9b"), (0.88, "7b"), (0.97, "5")]
    rows = []
    for s_t, lbl in targets:
        i = (d.s - s_t).abs().idxmin()
        c1, c2, s = d.loc[i, "c1"], d.loc[i, "c2"], d.loc[i, "s"]
        m = masses_from_ratios(*mass_ratios(A, B, c1, c2))
        eqs = find_equilibria(A, B, c1, c2, m, domain=(-8, 8, -8, 8),
                              grid_n=81, n_random=1500)
        # add per-primary local search for completeness near small masses
        from r5bp_nonsymmetric.geometry import primary_positions
        for (px, py) in primary_positions(A, B, c1, c2):
            eqs += [e for e in find_equilibria(A, B, c1, c2, m,
                    domain=(px - 0.8, px + 0.8, py - 0.8, py + 0.8),
                    grid_n=35, n_random=0)]
        uniq = []
        for e in eqs:
            if all(np.hypot(e.x - u.x, e.y - u.y) > 1e-5 for u in uniq):
                uniq.append(e)
        for e in uniq:
            rows.append({"label": lbl, "s": s, "c1": c1, "c2": c2,
                         "N_eq": len(uniq), "x": e.x, "y": e.y,
                         "stable": bool(e.max_real_part <= 1e-8)})
        # primaries
        for name, (px, py) in zip(["m0", "m1", "m2", "m3"],
                                  primary_positions(A, B, c1, c2)):
            rows.append({"label": lbl, "s": s, "c1": c1, "c2": c2,
                         "N_eq": len(uniq), "x": px, "y": py,
                         "stable": "primary", "which": name})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "snapshot_equilibria.csv"), index=False)
    print("snapshot_equilibria.csv written")


# ---------------------------------------------------------------------------
# fine fold sweeps + pair-separation exponent
# ---------------------------------------------------------------------------

def fold_sweeps():
    bif = pd.read_csv(os.path.join(RESULTS, "bifurcation_points.csv"))
    folds = bif[bif.classification.str.contains("saddle-node", na=False)]
    sweep_rows, exp_rows = [], []
    for _, f in folds.iterrows():
        comp = f["component"]; c1c, c2c = f["c1_c"], f["c2_c"]
        xc, yc = f["x_c"], f["y_c"]; sc = f["s_c"]; cc_label = f["count_change"]
        t = cc_tangent(A, B, c1c, c2c)
        for sign in (+1, -1):
            q = np.array([c1c, c2c]); tt = t * sign
            prev = [(xc, yc), (xc, yc)]
            dl = 0.0
            for k in range(30):
                ds = 0.003
                qn = _corrector(A, B, q + ds * tt, tt, q, ds)
                if qn is None:
                    break
                dl += float(np.linalg.norm(qn - q))
                tt = cc_tangent(A, B, qn[0], qn[1], prev=tt); q = qn
                m = masses_from_ratios(*mass_ratios(A, B, q[0], q[1]))
                loc = find_equilibria(A, B, q[0], q[1], m,
                                      domain=(xc - 1.4, xc + 1.4, yc - 1.4, yc + 1.4),
                                      grid_n=55, n_random=0)
                near = sorted(loc, key=lambda e: np.hypot(e.x - xc, e.y - yc))
                # the merging pair = two roots nearest previous pair, within radius
                near = [e for e in near if np.hypot(e.x - xc, e.y - yc) < 1.2]
                if len(near) >= 2:
                    p1, p2 = near[0], near[1]
                    dsep = float(np.hypot(p1.x - p2.x, p1.y - p2.y))
                    sweep_rows.append({"component": comp, "count_change": cc_label,
                                       "s_c": sc, "sign": sign, "dl": sign * dl,
                                       "c1": q[0], "c2": q[1],
                                       "x1": p1.x, "y1": p1.y, "x2": p2.x, "y2": p2.y,
                                       "d": dsep})
                    prev = [(p1.x, p1.y), (p2.x, p2.y)]
                else:
                    break
        # Fit the exponent on the EXISTENCE side: the side where the merging
        # pair separation -> 0 at the fold (smallest separation at the innermost
        # point). The other side is the annihilation side (pair gone; the two
        # nearest roots are unrelated and their distance does not vanish).
        sub = pd.DataFrame([r for r in sweep_rows
                            if r["count_change"] == cc_label and r["component"] == comp])
        best = None
        for sgn in (+1, -1):
            s2 = sub[sub.sign == sgn].copy()
            if len(s2) >= 5:
                s2 = s2.reindex(s2.dl.abs().sort_values().index)
                d_inner = float(s2.d.values[0])
                xlg = np.log(np.abs(s2.dl.values))
                ylg = np.log(s2.d.values)
                gamma, _ = np.polyfit(xlg, ylg, 1)
                if best is None or d_inner < best[3]:
                    best = (gamma, sgn, len(s2), d_inner)
        if best:
            exp_rows.append({"component": comp, "count_change": cc_label, "s_c": sc,
                             "exist_side_sign": best[1], "n_pts": best[2],
                             "d_innermost": best[3], "gamma_fit": best[0]})
    pd.DataFrame(sweep_rows).to_csv(os.path.join(RESULTS, "fold_local_sweeps.csv"), index=False)
    pd.DataFrame(exp_rows).to_csv(os.path.join(RESULTS, "fold_exponents.csv"), index=False)
    print("fold_local_sweeps.csv, fold_exponents.csv written")
    if exp_rows:
        for r in exp_rows:
            print(f"  {r['component']} {r['count_change']}: gamma~{r['gamma_fit']:.3f} "
                  f"(n={r['n_pts']})")


def main():
    spec = build_branch_spectra()
    rank_candidates(spec)
    snapshots()
    fold_sweeps()


if __name__ == "__main__":
    main()
