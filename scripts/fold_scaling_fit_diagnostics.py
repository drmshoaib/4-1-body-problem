"""Fold square-root scaling: fitting-window sensitivity diagnostics.

The pair separation near a saddle-node fold scales as d ~ |l - l_c|^gamma with
gamma = 1/2 asymptotically. The fold classification does NOT rest on this fit
(it uses the augmented system Omega_x=Omega_y=det H_Omega=psi=0, a simple
Hessian null direction, mu_perp != 0, and alpha,beta != 0); gamma ~ 1/2 is an
independent corroboration. This script makes the corroboration auditable by
refitting log d vs log|l-l_c| over nested windows (innermost points -> full
sweep) and recording gamma, prefactor, and log-space fit quality for each fold.

These are deterministic numerical data; R^2 / RMS describe fit quality, not
statistical uncertainty.

Reads:  results/fold_local_sweeps.csv
Writes: results/fold_scaling_fit_diagnostics.csv

Run:  python scripts/fold_scaling_fit_diagnostics.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")

WINDOWS = [("narrow", 10), ("default", 20), ("wide", 30)]  # innermost N points


def loglog_fit(dl, d):
    x = np.log(dl); y = np.log(d)
    gamma, logC = np.polyfit(x, y, 1)
    yhat = gamma * x + logC
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rms = float(np.sqrt(ss_res / len(y)))
    return float(gamma), float(np.exp(logC)), r2, rms


def main():
    sw = pd.read_csv(os.path.join(RESULTS, "fold_local_sweeps.csv"))
    exp = pd.read_csv(os.path.join(RESULTS, "fold_exponents.csv"))
    # canonical F1..F7 order = row order of fold_exponents.csv; carry the
    # existence-side sign (the side on which the merging pair exists)
    rows = []
    for i, er in exp.reset_index(drop=True).iterrows():
        Fid = f"F{i+1}"
        comp, cc, sc, side = er["component"], er["count_change"], er["s_c"], int(er["exist_side_sign"])
        g = sw[(sw.component == comp) & (sw.count_change == cc) & (sw["sign"] == side)].copy()
        g["adl"] = g["dl"].abs()
        g = g[(g["adl"] > 0) & (g["d"] > 0) & np.isfinite(g["d"])].sort_values("adl")
        for wname, n in WINDOWS:                       # innermost N points (smallest |dl|)
            sub = g.head(n)
            if len(sub) < 2:
                continue
            gamma, C, r2, rms = loglog_fit(sub["adl"].values, sub["d"].values)
            rows.append(dict(fold=Fid, component=comp, count_change=cc, s_c=sc,
                             exist_side_sign=side, window=wname, n_pts=len(sub),
                             abs_dl_min=float(sub["adl"].min()), abs_dl_max=float(sub["adl"].max()),
                             gamma=gamma, prefactor=C, r2_logspace=r2, rms_logspace=rms))
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "fold_scaling_fit_diagnostics.csv")
    df.to_csv(out, index=False)
    print("wrote", out)

    print(f"\n{'fold':>4} {'change':>8} {'narrow_g':>9} {'default_g':>10} {'wide_g':>8} "
          f"{'g_range':>9} {'max|g-0.5|':>10}")
    for Fid, g in df.groupby("fold", sort=False):
        gv = {r.window: r.gamma for r in g.itertuples()}
        gam = [gv["narrow"], gv["default"], gv["wide"]]
        cc = g.iloc[0]["count_change"]
        print(f"{Fid:>4} {cc:>8} {gam[0]:>9.3f} {gam[1]:>10.3f} {gam[2]:>8.3f} "
              f"{max(gam)-min(gam):>9.3f} {max(abs(x-0.5) for x in gam):>10.3f}")
    worst = df.assign(dev=(df.gamma - 0.5).abs()).sort_values("dev").iloc[-1]
    print(f"\n  largest deviation from 1/2: fold {worst.fold} ({worst.window} window) "
          f"gamma={worst.gamma:.3f}")
    # narrow-window (most asymptotic) deviation is the physically relevant test
    nb = df[df.window == "narrow"].assign(dev=(df[df.window=='narrow'].gamma-0.5).abs())
    print(f"  narrow-window (innermost, most asymptotic) max |gamma-1/2| = {nb.dev.max():.3f}")


if __name__ == "__main__":
    main()
