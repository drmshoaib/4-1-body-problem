"""Equilibrium-search convergence and exclusion-radius sensitivity study.

Referee-response evidence for the completeness of the independent multi-start
equilibrium search. For representative configurations (the N_eq = 11 Case-2
regime and the four reference cases) it re-runs find_equilibria while varying,
one factor at a time:

  * grid density        grid_n in {61, 81, 101, 121}
  * random-start count  n_random in {1000, 2000, 4000}
  * random seed         several seeds
  * exclusion radius    r_min in {1e-3, 5e-4, 1e-4}  (converged-root rejection
                        radius = 10*r_min = 1e-2, 5e-3, 1e-3)

For each run it records the detected count and the minimum distance from any
detected equilibrium to the nearest primary, so we can show (a) the count
saturates, and (b) lowering the rejection radius does not reveal genuine
near-primary equilibria that were previously discarded.

Writes: results/equilibrium_search_convergence.csv

Run:  python scripts/equilibrium_search_convergence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from r5bp_nonsymmetric.equilibria import find_equilibria
from r5bp_nonsymmetric.central_config import mass_ratios, masses_from_ratios
from r5bp_nonsymmetric.geometry import primary_positions
from r5bp_nonsymmetric.cases import REFERENCE_CASES

RESULTS = os.path.join(ROOT, "results")
A, B = 0.9, -0.1
SEEDS = [20260808, 12345, 99991]


def min_primary_distance(eqs, a, b, c1, c2):
    if not eqs:
        return float("nan")
    P = primary_positions(a, b, c1, c2)
    d = []
    for e in eqs:
        d.append(min(np.hypot(e.x - P[i, 0], e.y - P[i, 1]) for i in range(4)))
    return float(min(d))


def target_configs():
    cfgs = []
    cnt = pd.read_csv(os.path.join(RESULTS, "equilibrium_count_vs_s.csv"))
    n11 = cnt[cnt["N_eq"] == 11]
    if len(n11):
        r = n11.iloc[0]
        cfgs.append(("N=11 (Case-2)", A, B, float(r["c1"]), float(r["c2"])))
    for c in REFERENCE_CASES:
        cfgs.append((c.name, c.a, c.b, c.c1, c.c2))
    return cfgs


def run(a, b, c1, c2, grid_n, n_random, seed, r_min):
    m = masses_from_ratios(*mass_ratios(a, b, c1, c2))
    eqs = find_equilibria(a, b, c1, c2, m, grid_n=grid_n, n_random=n_random,
                          seed=seed, r_min=r_min)
    return len(eqs), min_primary_distance(eqs, a, b, c1, c2)


def main():
    rows = []
    for name, a, b, c1, c2 in target_configs():
        full = (name.startswith("N=11"))
        print(f"\n=== {name}  (c1,c2)=({c1:.4f},{c2:.4f}) ===")

        # grid-density sweep
        for g in ([61, 81, 101, 121] if full else [81, 121]):
            n, md = run(a, b, c1, c2, g, 2000, SEEDS[0], 1e-3)
            rows.append(dict(config=name, study="grid", grid_n=g, n_random=2000,
                             seed=SEEDS[0], r_min=1e-3, exclusion=1e-2, count=n,
                             min_prim_dist=md))
            print(f"  grid_n={g:4d}: count={n}  min_prim_dist={md:.4f}")

        if full:
            # random-start sweep
            for nr in [1000, 2000, 4000]:
                n, md = run(a, b, c1, c2, 81, nr, SEEDS[0], 1e-3)
                rows.append(dict(config=name, study="random", grid_n=81, n_random=nr,
                                 seed=SEEDS[0], r_min=1e-3, exclusion=1e-2, count=n,
                                 min_prim_dist=md))
                print(f"  n_random={nr:5d}: count={n}  min_prim_dist={md:.4f}")
            # seed sweep
            for sd in SEEDS:
                n, md = run(a, b, c1, c2, 81, 2000, sd, 1e-3)
                rows.append(dict(config=name, study="seed", grid_n=81, n_random=2000,
                                 seed=sd, r_min=1e-3, exclusion=1e-2, count=n,
                                 min_prim_dist=md))
                print(f"  seed={sd:9d}: count={n}  min_prim_dist={md:.4f}")

        # exclusion-radius sweep (r_min; rejection = 10*r_min)
        for rm in [1e-3, 5e-4, 1e-4]:
            n, md = run(a, b, c1, c2, (101 if full else 81), (4000 if full else 2000),
                        SEEDS[0], rm)
            rows.append(dict(config=name, study="exclusion",
                             grid_n=(101 if full else 81),
                             n_random=(4000 if full else 2000), seed=SEEDS[0],
                             r_min=rm, exclusion=10 * rm, count=n, min_prim_dist=md))
            print(f"  exclusion={10*rm:.0e}: count={n}  min_prim_dist={md:.4f}")

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "equilibrium_search_convergence.csv")
    df.to_csv(out, index=False)
    print("\nwrote", out)
    # concise verdict
    print("\n=== saturation check (count by config/study) ===")
    for name in df["config"].unique():
        s = df[df.config == name]
        print(f"  {name}: counts seen = {sorted(s['count'].unique())}, "
              f"min prim-dist over all runs = {s['min_prim_dist'].min():.4f}")


if __name__ == "__main__":
    main()
