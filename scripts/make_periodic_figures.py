"""Publication figures for the periodic-orbit families. Every orbit is
re-integrated from its saved row in results/periodic_orbits.csv (reproducible).

Outputs (paper/figures/):
    periodic_family_case1_mode1.pdf   periodic_family_case1_mode2.pdf
    periodic_family_case2_mode1.pdf   periodic_family_case2_mode2.pdf
    period_vs_jacobi.pdf
    floquet_vs_family_parameter.pdf
    selected_resonant_orbits.pdf
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
from r5bp_nonsymmetric.periodic import Params, integrate  # noqa: E402
from r5bp_nonsymmetric.geometry import primary_positions   # noqa: E402

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


def orbit_traj(row, n=800):
    P = Params(A, B, row.c1, row.c2)
    u0 = np.array([row.x0, row.y0, row.vx0, row.vy0])
    _, _, traj = integrate(u0, row["T"], P, n_eval=n)
    return traj, P


def _draw_primaries_eq(ax, row, P):
    P4 = primary_positions(A, B, row.c1, row.c2)
    ax.scatter(P4[:, 0], P4[:, 1], marker="*", s=150, color="k",
               edgecolors="white", linewidths=0.6, zorder=6)
    ax.scatter([row.x_eq], [row.y_eq], marker="P", s=70, color="#555",
               zorder=6, label="parent equilibrium")


def family_figure(orb, fam_id, title, fname):
    g = orb[orb.family_id == fam_id].sort_values("geom_amplitude")
    if len(g) == 0:
        print("  (no data for family", fam_id, ")"); return
    idx = [0, len(g) // 2, len(g) - 1]
    picks = g.iloc[idx]
    labs = ["small (linear-mode seed)", "moderate", "large (nonlinear)"]
    col = "#c1121f"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, lab, (_, row) in zip(axes, labs, picks.iterrows()):
        traj, P = orbit_traj(row)
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.7, zorder=5)
        ax.plot(traj[0, 0], traj[0, 1], "o", color="k", ms=5, zorder=7)
        ax.annotate("", xy=(traj[6, 0], traj[6, 1]), xytext=(traj[0, 0], traj[0, 1]),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.4), zorder=7)
        ax.scatter([row.x_eq], [row.y_eq], marker="P", s=70, color="#2a9d8f", zorder=6)
        # zoom to the orbit with margin
        xr = traj[:, 0]; yr = traj[:, 1]
        mx = 0.18 * (xr.max() - xr.min() + 1e-6) + 0.02
        my = 0.18 * (yr.max() - yr.min() + 1e-6) + 0.02
        ax.set_xlim(min(xr.min(), row.x_eq) - mx, max(xr.max(), row.x_eq) + mx)
        ax.set_ylim(min(yr.min(), row.y_eq) - my, max(yr.max(), row.y_eq) + my)
        # primaries only if inside view
        P4 = primary_positions(A, B, row.c1, row.c2)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        vis = [(px, py) for (px, py) in P4 if xl[0] <= px <= xl[1] and yl[0] <= py <= yl[1]]
        if vis:
            vis = np.array(vis)
            ax.scatter(vis[:, 0], vis[:, 1], marker="*", s=150, color="k",
                       edgecolors="white", linewidths=0.6, zorder=6)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        ax.set_title(f"{lab}\n$A$={row.geom_amplitude:.3f}, $T$={row['T']:.2f}",
                     fontsize=9)
    fig.suptitle(title + "  (green $+$: parent equilibrium; black dot: "
                 "initial point; arrow: direction; stars: primaries in view)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, fname)


def fig_period_jacobi(orb):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for fid, g in orb.groupby("family_id"):
        g = g.sort_values("jacobi_CJ")
        lab = f"fam{fid}: {g.label.iloc[0]} m{g['mode'].iloc[0]}"
        ax.plot(g.jacobi_CJ, g["T"], "-o", ms=3, lw=1, label=lab)
    ax.set_xlabel(r"Jacobi constant $C_J$"); ax.set_ylabel(r"period $T$")
    ax.set_title("Period vs Jacobi constant along each family")
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5, ncol=2, loc="best")
    _save(fig, "period_vs_jacobi.pdf")


def fig_floquet(orb, flo):
    m = orb.merge(flo[["family_id", "member", "stability_index", "classification"]],
                  on=["family_id", "member"])
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for fid, g in m.groupby("family_id"):
        g = g.sort_values("geom_amplitude")
        lab = f"fam{fid}: {g.label.iloc[0]} m{g['mode'].iloc[0]}"
        ax.plot(g.geom_amplitude, g.stability_index, "-o", ms=3, lw=1, label=lab)
    ax.axhline(2, color="0.5", ls="--", lw=1); ax.axhline(-2, color="0.5", ls="--", lw=1)
    ax.text(ax.get_xlim()[1], 2, " +2 (tangent)", fontsize=7, va="center", color="0.4")
    ax.text(ax.get_xlim()[1], -2, " -2 (period-doubling)", fontsize=7, va="center", color="0.4")
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"geometric amplitude $A$")
    ax.set_ylabel(r"Floquet stability index $\nu=\lambda+1/\lambda$")
    ax.set_title(r"Floquet stability index vs family amplitude "
                 r"($|\nu|<2$: Floquet stable)")
    ax.grid(alpha=0.3); ax.legend(fontsize=6.5, ncol=2, loc="best")
    _save(fig, "floquet_vs_family_parameter.pdf")


def fig_resonant(orb, flo):
    m = orb.merge(flo[["family_id", "member", "stability_index", "classification"]],
                  on=["family_id", "member"])
    # panels: real-unstable (fam5/6), near period-doubling (fam8, fam10 last), multi-lobe (fam2 last)
    panels = []
    for fid, want in [(5, "last"), (10, "extreme"), (8, "last"), (2, "last")]:
        g = m[m.family_id == fid]
        if len(g) == 0:
            continue
        if want == "extreme":
            row = g.iloc[g.stability_index.abs().argmax()]
        else:
            row = g.sort_values("geom_amplitude").iloc[-1]
        panels.append(row)
    disp = {"case1_res_3to2": "Case-1 3:2 reg.", "case2_res_2to1": "Case-2 5:2 reg.",
            "case1_stable": "Case-1 stable", "case2_stable": "Case-2 stable",
            "case1_saddlecentre": "Case-1 sdl-ctr", "case2_saddlecentre": "Case-2 sdl-ctr"}
    n = len(panels); fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, panels):
        traj, P = orbit_traj(row)
        ax.plot(traj[:, 0], traj[:, 1], color="#c1121f", lw=1.7, zorder=5)
        ax.plot(traj[0, 0], traj[0, 1], "ko", ms=4, zorder=7)
        ax.scatter([row.x_eq], [row.y_eq], marker="P", s=70, color="#2a9d8f", zorder=6)
        xr, yr = traj[:, 0], traj[:, 1]
        mx = 0.2 * (xr.max() - xr.min() + 1e-6) + 0.02
        my = 0.2 * (yr.max() - yr.min() + 1e-6) + 0.02
        ax.set_xlim(min(xr.min(), row.x_eq) - mx, max(xr.max(), row.x_eq) + mx)
        ax.set_ylim(min(yr.min(), row.y_eq) - my, max(yr.max(), row.y_eq) + my)
        P4 = primary_positions(A, B, row.c1, row.c2)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        vis = np.array([(px, py) for (px, py) in P4
                        if xl[0] <= px <= xl[1] and yl[0] <= py <= yl[1]])
        if len(vis):
            ax.scatter(vis[:, 0], vis[:, 1], marker="*", s=140, color="k",
                       edgecolors="white", linewidths=0.6, zorder=6)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        cls_disp = str(row.classification).replace("spectrally stable", "Floquet stable")
        ax.set_title(f"fam{int(row.family_id)} {disp.get(row.label, row.label)}\n"
                     f"$A$={row.geom_amplitude:.3f}, $\\nu$={row.stability_index:.3f}\n"
                     f"{cls_disp}", fontsize=8)
    fig.suptitle("Selected unstable / near-transition / multi-lobe orbits", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "selected_resonant_orbits.pdf")


def main():
    orb = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    flo = pd.read_csv(os.path.join(RESULTS, "periodic_orbit_floquet.csv"))
    family_figure(orb, 1, "Case-1 stable equilibrium, mode 1 ($\\omega_1$) family",
                  "periodic_family_case1_mode1.pdf")
    family_figure(orb, 2, "Case-1 stable equilibrium, mode 2 ($\\omega_2$) family",
                  "periodic_family_case1_mode2.pdf")
    family_figure(orb, 3, "Case-2 stable equilibrium, mode 1 ($\\omega_1$) family",
                  "periodic_family_case2_mode1.pdf")
    family_figure(orb, 4, "Case-2 stable equilibrium, mode 2 ($\\omega_2$) family",
                  "periodic_family_case2_mode2.pdf")
    fig_period_jacobi(orb)
    fig_floquet(orb, flo)
    fig_resonant(orb, flo)


if __name__ == "__main__":
    main()
