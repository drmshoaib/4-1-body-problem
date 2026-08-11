"""Hill-region (zero-velocity) analysis and figures.

Tracks the critical Jacobi values C_{J,i}(s)=2 Omega(L_i(s)) along the reliable
equilibrium branches, determines where the topology of the allowed region
H(C_J)={(x,y): 2 Omega >= C_J} changes (connected-component count on a grid),
and draws zero-velocity curves below/above critical values, through the N_eq=11
regime, and around the stable-equilibrium regimes with periodic orbits overlaid.

Outputs (paper/figures/):
    critical_jacobi_vs_s.pdf
    hill_topology_with_orbits.pdf
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r5bp_nonsymmetric.central_config import mass_ratios, masses_from_ratios  # noqa
from r5bp_nonsymmetric.equilibria import find_equilibria                      # noqa
from r5bp_nonsymmetric.geometry import primary_positions                      # noqa
from r5bp_nonsymmetric.periodic import Params, integrate                      # noqa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
A, B = 0.9, -0.1


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def two_omega(X, Y, c1, c2, m):
    m0, m1, m2, m3 = m
    return (X**2 + Y**2
            + 2*m0/np.sqrt(X**2 + (Y-B)**2)
            + 2*m1/np.sqrt((X+1)**2 + Y**2)
            + 2*m2/np.sqrt(X**2 + (Y-A)**2)
            + 2*m3/np.sqrt((X-c1)**2 + (Y-c2)**2))


def equilibria_at(c1, c2):
    m = masses_from_ratios(*mass_ratios(A, B, c1, c2))
    eqs = find_equilibria(A, B, c1, c2, m, domain=(-8, 8, -8, 8),
                          grid_n=71, n_random=800)
    for (px, py) in primary_positions(A, B, c1, c2):
        eqs += [e for e in find_equilibria(A, B, c1, c2, m,
                domain=(px-0.8, px+0.8, py-0.8, py+0.8), grid_n=35, n_random=0)]
    uniq = []
    for e in eqs:
        if all(np.hypot(e.x-u.x, e.y-u.y) > 1e-5 for u in uniq):
            uniq.append(e)
    return m, uniq


def allowed_components(c1, c2, m, CJ, box=6.0, n=400):
    xs = np.linspace(-box, box, n); ys = np.linspace(-box, box, n)
    X, Y = np.meshgrid(xs, ys)
    W = two_omega(X, Y, c1, c2, m)
    allowed = W >= CJ
    lab, ncomp = ndimage.label(allowed)
    return ncomp


def fig_critical_jacobi(spec):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.0))
    bif = pd.read_csv(os.path.join(RESULTS, "bifurcation_points.csv"))
    DISP={"Case1-arc":"Case-1 arc","Case2-loop":"Case-2 arc"}
    for ax, comp in zip(axes, ["Case1-arc", "Case2-loop"]):
        d = spec[(spec.component == comp) & spec.reliable]
        for bid, g in d.groupby("branch_id"):
            g = g.sort_values("s")
            ax.plot(g.s, g.jacobi_CJ, "-o", ms=2, lw=1)
        for _, r in bif[(bif.component == comp) &
                        bif.classification.str.contains("saddle-node", na=False)].iterrows():
            ax.axvline(r.s_c, color="0.6", ls="--", lw=0.9)
        ax.set_ylabel(r"$C_{J,i}=2\Omega(L_i)$"); ax.set_title(DISP[comp], fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("arc parameter $s$")
    fig.suptitle("Critical Jacobi values along the equilibrium branches "
                 "(dashed: saddle-node folds)", fontsize=11)
    fig.tight_layout()
    _save(fig, "critical_jacobi_vs_s.pdf")


def draw_hill(ax, c1, c2, m, CJ, eqs=None, orbit=None, box=4.5, n=500):
    xs = np.linspace(-box, box, n); ys = np.linspace(-box, box, n)
    X, Y = np.meshgrid(xs, ys)
    W = two_omega(X, Y, c1, c2, m)
    ax.contourf(X, Y, (W >= CJ).astype(float), levels=[0.5, 1.5],
                colors=["#cfe3f2"], alpha=0.8)          # allowed region
    ax.contour(X, Y, W, levels=[CJ], colors="#1f4e79", linewidths=1.1)  # ZVC
    P4 = primary_positions(A, B, c1, c2)
    ax.scatter(P4[:, 0], P4[:, 1], marker="*", s=120, color="k",
               edgecolors="white", linewidths=0.5, zorder=6)
    if eqs is not None:
        ex = [e.x for e in eqs]; ey = [e.y for e in eqs]
        ax.scatter(ex, ey, s=16, color="#c1121f", zorder=6)
    if orbit is not None:
        ax.plot(orbit[:, 0], orbit[:, 1], color="#7a0177", lw=1.8, zorder=7)
    ax.set_aspect("equal"); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")


def fig_hill_with_orbits(orb):
    path = pd.read_csv(os.path.join(RESULTS, "cc_continuation_path.csv"))

    def cc_of(comp, s_t):
        d = path[path.component == comp]
        i = (d.s - s_t).abs().idxmin()
        return float(d.loc[i, "c1"]), float(d.loc[i, "c2"])

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 10.0))

    # --- N=11 regime: below and above an interior critical Jacobi value ---
    c1, c2 = cc_of("Case2-loop", 0.786)
    m, eqs = equilibria_at(c1, c2)
    cjs = sorted(2 * e.omega for e in eqs)          # e.jacobi already = 2 Omega
    cj_vals = sorted(e.jacobi for e in eqs)
    lo = cj_vals[len(cj_vals)//2] - 0.6
    hi = cj_vals[len(cj_vals)//2] + 0.6
    n_lo = allowed_components(c1, c2, m, lo); n_hi = allowed_components(c1, c2, m, hi)
    # persist the grid-resolution robustness of the 1->2 component change
    res_rows = []
    for gn in (300, 400, 500):
        res_rows.append({"grid_n": gn, "CJ": round(lo, 3),
                         "n_components": allowed_components(c1, c2, m, lo, n=gn)})
        res_rows.append({"grid_n": gn, "CJ": round(hi, 3),
                         "n_components": allowed_components(c1, c2, m, hi, n=gn)})
    pd.DataFrame(res_rows).to_csv(
        os.path.join(RESULTS, "hill_connectivity_resolution.csv"), index=False)
    margins = []
    draw_hill(axes[0, 0], c1, c2, m, lo, eqs=eqs)
    axes[0, 0].set_title(f"$N_{{eq}}=11$ regime, $C_J={lo:.2f}$ "
                         f"(allowed comps={n_lo})", fontsize=9)
    draw_hill(axes[0, 1], c1, c2, m, hi, eqs=eqs)
    axes[0, 1].set_title(f"$N_{{eq}}=11$ regime, $C_J={hi:.2f}$ "
                         f"(allowed comps={n_hi})", fontsize=9)

    # --- Case-2 stable regime with a stable orbit overlaid ---
    g3 = orb[orb.family_id == 3].sort_values("geom_amplitude")
    row = g3.iloc[-1]
    P = Params(A, B, row.c1, row.c2)
    _, _, traj = integrate(np.array([row.x0, row.y0, row.vx0, row.vy0]), row["T"], P, n_eval=600)
    m2s = masses_from_ratios(*mass_ratios(A, B, row.c1, row.c2))
    _, eqs2 = equilibria_at(row.c1, row.c2)
    draw_hill(axes[1, 0], row.c1, row.c2, m2s, row.jacobi_CJ, eqs=eqs2, orbit=traj)
    axes[1, 0].set_title(f"Case-2 stable regime, orbit fam3 "
                         f"($C_J={row.jacobi_CJ:.2f}$)", fontsize=9)
    margins.append({"family_id": 3, "CJ": round(float(row.jacobi_CJ), 4),
                    "min_hill_margin": float(np.min(
                        two_omega(traj[:, 0], traj[:, 1], row.c1, row.c2, m2s)
                        - row.jacobi_CJ))})

    # --- Case-1 stable regime with a stable orbit overlaid ---
    g1 = orb[orb.family_id == 2].sort_values("geom_amplitude")
    row = g1.iloc[-1]
    P = Params(A, B, row.c1, row.c2)
    _, _, traj = integrate(np.array([row.x0, row.y0, row.vx0, row.vy0]), row["T"], P, n_eval=600)
    m1s = masses_from_ratios(*mass_ratios(A, B, row.c1, row.c2))
    _, eqs1 = equilibria_at(row.c1, row.c2)
    draw_hill(axes[1, 1], row.c1, row.c2, m1s, row.jacobi_CJ, eqs=eqs1, orbit=traj)
    axes[1, 1].set_title(f"Case-1 stable regime, orbit fam2 "
                         f"($C_J={row.jacobi_CJ:.2f}$)", fontsize=9)
    margins.append({"family_id": 2, "CJ": round(float(row.jacobi_CJ), 4),
                    "min_hill_margin": float(np.min(
                        two_omega(traj[:, 0], traj[:, 1], row.c1, row.c2, m1s)
                        - row.jacobi_CJ))})
    pd.DataFrame(margins).to_csv(
        os.path.join(RESULTS, "hill_orbit_margin.csv"), index=False)

    fig.suptitle("Hill regions (shaded = allowed, $2\\Omega\\geq C_J$; blue = "
                 "zero-velocity curve) with periodic orbits (purple)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "hill_topology_with_orbits.pdf")
    print(f"  N=11 regime allowed-component count: {n_lo} (C_J={lo:.2f}) -> "
          f"{n_hi} (C_J={hi:.2f})")
    for mrow in margins:
        print(f"  Hill margin fam{mrow['family_id']} (C_J={mrow['CJ']}): "
              f"min(2Omega-C_J)={mrow['min_hill_margin']:.4e}")


def main():
    spec = pd.read_csv(os.path.join(RESULTS, "branch_spectra.csv"))
    orb = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    fig_critical_jacobi(spec)
    fig_hill_with_orbits(orb)


if __name__ == "__main__":
    main()
