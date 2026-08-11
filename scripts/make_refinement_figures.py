"""Refinement-stage figures:
    period_doubling_scan.pdf     nu vs amplitude near the nu=-2 threshold (fam8/10)
    family_evolution.pdf         T, amplitude, nu vs C_J for the principal families
    hill_stable_vs_unstable.pdf  a stable and an unstable orbit on their Hill geometry
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r5bp_nonsymmetric.periodic import Params, integrate           # noqa
from r5bp_nonsymmetric.central_config import mass_ratios, masses_from_ratios  # noqa
from r5bp_nonsymmetric.geometry import primary_positions           # noqa

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
    return (X**2 + Y**2 + 2*m0/np.sqrt(X**2+(Y-B)**2) + 2*m1/np.sqrt((X+1)**2+Y**2)
            + 2*m2/np.sqrt(X**2+(Y-A)**2) + 2*m3/np.sqrt((X-c1)**2+(Y-c2)**2))


def fig_pd_scan():
    d = pd.read_csv(os.path.join(RESULTS, "period_doubling_scan.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    disp = {"fam10_case2_res_2to1": "fam10 Case-2 5:2",
            "fam10_case2_res_2to1_ultra": "fam10 Case-2 5:2 ultra",
            "fam8_case1_res_3to2": "fam8 Case-1 3:2"}
    for fam, g in d.groupby("family"):
        g = g.sort_values("geom_amplitude")
        ax.plot(g.geom_amplitude, g.nu, "-o", ms=3, label=disp.get(fam, fam))
    ax.axhline(-2, color="crimson", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], -2, " $\\nu=-2$ (period-doubling)", va="bottom",
            ha="right", fontsize=8, color="crimson")
    # annotate closest approach
    g10 = d[d.family.str.startswith("fam10")]
    if len(g10):
        r = g10.iloc[g10.nu.add(2).abs().argmin()]
        ax.annotate(f"closest: $\\nu={r.nu:.7f}$\n($|\\nu+2|\\approx${abs(r.nu+2):.1e}, no crossing)",
                    xy=(r.geom_amplitude, r.nu), xytext=(r.geom_amplitude*0.5, -1.985),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.set_xlabel("geometric amplitude $A$")
    ax.set_ylabel(r"Floquet stability index $\nu$")
    ax.set_title(r"Approach to the period-doubling threshold $\nu=-2$ "
                 r"(fine scan; the multiplier stays on the unit circle)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, "period_doubling_scan.pdf")


def fig_family_evolution():
    o = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    f = pd.read_csv(os.path.join(RESULTS, "periodic_orbit_floquet.csv"))
    m = o.merge(f[["family_id", "member", "stability_index"]], on=["family_id", "member"])
    fams = [1, 2, 3, 4, 7, 8, 9, 10, 5, 6]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for fid in fams:
        g = m[m.family_id == fid].sort_values("jacobi_CJ")
        lab = f"fam{fid}"
        axes[0].plot(g.jacobi_CJ, g["T"], "-o", ms=2.5, lw=1, label=lab)
        axes[1].plot(g.jacobi_CJ, g.geom_amplitude, "-o", ms=2.5, lw=1, label=lab)
        axes[2].plot(g.jacobi_CJ, g.stability_index, "-o", ms=2.5, lw=1, label=lab)
    axes[0].set_ylabel("period $T$"); axes[0].set_title("$T$ vs $C_J$", fontsize=10)
    axes[1].set_ylabel("geometric amplitude $A$"); axes[1].set_title("$A$ vs $C_J$", fontsize=10)
    axes[2].set_yscale("symlog", linthresh=2.0)
    axes[2].axhline(2, color="0.5", ls="--", lw=0.9); axes[2].axhline(-2, color="0.5", ls="--", lw=0.9)
    axes[2].set_ylabel(r"$\nu$ (symlog)"); axes[2].set_title(r"$\nu$ vs $C_J$ ($|\nu|<2$ stable)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("$C_J$"); ax.grid(alpha=0.3)
    axes[2].legend(fontsize=6.5, ncol=2, loc="best")
    fig.suptitle("Principal periodic-orbit family evolution", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "family_evolution.pdf")


def _hill_panel(ax, row, box, title):
    P = Params(A, B, row.c1, row.c2)
    m = masses_from_ratios(*mass_ratios(A, B, row.c1, row.c2))
    _, _, traj = integrate(np.array([row.x0, row.y0, row.vx0, row.vy0]), row["T"], P, n_eval=700)
    xs = np.linspace(-box, box, 500); ys = np.linspace(-box, box, 500)
    X, Y = np.meshgrid(xs, ys); W = two_omega(X, Y, row.c1, row.c2, m)
    ax.contourf(X, Y, (W >= row.jacobi_CJ).astype(float), levels=[0.5, 1.5],
                colors=["#cfe3f2"], alpha=0.8)
    ax.contour(X, Y, W, levels=[row.jacobi_CJ], colors="#1f4e79", linewidths=1.0)
    P4 = primary_positions(A, B, row.c1, row.c2)
    ax.scatter(P4[:, 0], P4[:, 1], marker="*", s=120, color="k",
               edgecolors="white", linewidths=0.5, zorder=6)
    ax.scatter([row.x_eq], [row.y_eq], marker="P", s=60, color="#2a9d8f", zorder=6)
    ax.plot(traj[:, 0], traj[:, 1], color="#7a0177", lw=1.8, zorder=7)
    ax.set_aspect("equal"); ax.set_xlim(-box, box); ax.set_ylim(-box, box)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_title(title, fontsize=9)


def fig_hill_stable_vs_unstable():
    o = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    f = pd.read_csv(os.path.join(RESULTS, "periodic_orbit_floquet.csv"))
    m = o.merge(f[["family_id", "member", "classification"]], on=["family_id", "member"])
    stable = m[m.family_id == 3].sort_values("geom_amplitude").iloc[-1]
    unstable = m[m.family_id == 5].sort_values("geom_amplitude").iloc[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    _hill_panel(axes[0], stable, 4.5,
                f"Stable orbit (fam3), $C_J={stable.jacobi_CJ:.2f}$\n"
                f"librates inside one Hill component")
    _hill_panel(axes[1], unstable, 3.0,
                f"Real-unstable orbit (fam5), $C_J={unstable.jacobi_CJ:.2f}$\n"
                f"small libration near the inner (primary) region")
    fig.suptitle("Stable vs. unstable periodic orbit on the Hill geometry "
                 "(shaded = allowed; purple = orbit)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "hill_stable_vs_unstable.pdf")


def main():
    fig_pd_scan()
    fig_family_evolution()
    fig_hill_stable_vs_unstable()


if __name__ == "__main__":
    main()
