"""Analytic outer-radius bound for equilibria of the restricted problem.

At an equilibrium of Omega = (x^2+y^2)/2 + sum_i m_i/r_i,

    r_vec = sum_i m_i (r_vec - r_i) / |r_vec - r_i|^3 ,

so, taking magnitudes and using |r_vec - r_i| >= r - R_p for r > R_p
(R_p = max_i |r_i|),

    r <= sum_i m_i / (r - R_p)^2 = M / (r - R_p)^2 ,   M = sum_i m_i .

Hence every equilibrium with r > R_p satisfies  r (r - R_p)^2 <= M.
The largest positive root r_max of r (r - R_p)^2 = M is an outer radius beyond
which no equilibrium can lie. This is NOT a bound on the number of equilibria;
it only excludes equilibria at arbitrarily large radius.

For every accepted continuation sample this script computes R_p, M, r_max, and
the largest actually detected equilibrium radius, and reports the maximum r_max
over the analysed arcs against the numerically searched domain.

The square search box [-8,8]^2 contains exactly the disk of radius 8 (its
corners reach 8*sqrt(2) ~ 11.31, but we do NOT rely on the corners). The disk
of radius r_max is wholly contained in the box iff r_max <= 8.

Reads:  results/equilibrium_count_vs_s.csv, results/equilibrium_branches.csv
Writes: results/equilibrium_outer_radius_bound.csv

Run:  python scripts/equilibrium_outer_radius_bound.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS = os.path.join(ROOT, "results")

from r5bp_nonsymmetric.central_config import mass_ratios, masses_from_ratios
from r5bp_nonsymmetric.geometry import primary_positions

A, B = 0.9, -0.1
BOX_RADIUS = 8.0                 # guaranteed radius wholly inside [-8,8]^2


def r_max_bound(Rp, M):
    """Largest positive root of r (r - Rp)^2 = M (unique for r > Rp)."""
    f = lambda r: r * (r - Rp) ** 2 - M
    hi = Rp + M ** (1.0 / 3.0) + 10.0     # f(Rp) = -M < 0, f(hi) > 0
    return brentq(f, Rp, hi, xtol=1e-12, rtol=8.9e-16, maxiter=200)


def main():
    ct = pd.read_csv(os.path.join(RESULTS, "equilibrium_count_vs_s.csv"))
    br = pd.read_csv(os.path.join(RESULTS, "equilibrium_branches.csv"))
    br["r"] = np.hypot(br["x"], br["y"])
    det_r = br.groupby(["component", "s"])["r"].max().to_dict()

    rows = []
    for _, row in ct.iterrows():
        comp, s, c1, c2 = row["component"], float(row["s"]), float(row["c1"]), float(row["c2"])
        P = primary_positions(A, B, c1, c2)
        Rp = float(np.max(np.hypot(P[:, 0], P[:, 1])))
        m = masses_from_ratios(*mass_ratios(A, B, c1, c2))
        M = float(np.sum(m))
        rmax = r_max_bound(Rp, M)
        detr = det_r.get((comp, s), float("nan"))
        rows.append(dict(arc=comp, s=s, R_p=Rp, M=M, r_max_bound=rmax,
                         max_detected_r=detr,
                         detected_within_bound=(detr <= rmax + 1e-9),
                         bound_in_box=(rmax <= BOX_RADIUS)))

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "equilibrium_outer_radius_bound.csv")
    df.to_csv(out, index=False)
    print("wrote", out, f"({len(df)} samples)")

    print("\n=== summary ===")
    for arc, g in df.groupby("arc"):
        print(f"  {arc}: max r_max_bound = {g.r_max_bound.max():.4f}  "
              f"(R_p<= {g.R_p.max():.3f}, M<= {g.M.max():.2f})  "
              f"max detected r = {g.max_detected_r.max():.4f}")
    gmax = df.r_max_bound.max()
    print(f"\n  GLOBAL max analytic r_max over all samples = {gmax:.4f}")
    print(f"  guaranteed radius inside [-8,8]^2 (not corners) = {BOX_RADIUS}")
    print(f"  disk of radius r_max wholly inside the search box: {gmax <= BOX_RADIUS}")
    print(f"  every detected equilibrium radius <= its analytic bound: "
          f"{bool(df.detected_within_bound.all())}")
    bad = df[~df.bound_in_box]
    if len(bad):
        print(f"\n  !! {len(bad)} samples have r_max > {BOX_RADIUS} -- bound NOT closed by the box")
        print(bad[["arc", "s", "R_p", "M", "r_max_bound"]].to_string(index=False))


if __name__ == "__main__":
    main()
