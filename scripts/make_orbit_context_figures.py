"""Explanatory context figures for the periodic-orbit families.

Produces two dark-theme figures for one representative family:

  orbit_rotating_vs_inertial.{png,pdf}
      The same periodic orbit shown in the co-rotating frame (a closed libration
      loop about the equilibrium) and in the inertial frame (a rosette confined
      to an annulus near the equilibrium's radius, while the primaries move on
      circles). Makes clear that the body circulates the whole system in inertial
      space rather than orbiting an individual primary.

  orbit_potential_context.{png,pdf}
      The rotating-frame effective-potential landscape 2*Omega. The four
      primaries are deep singular wells; the libration point (equilibrium) is a
      local trough in which the family nests. Level curves are zero-velocity
      curves.

Reads:  results/periodic_orbits.csv
Writes: paper/figures/orbit_rotating_vs_inertial.{png,pdf}
        paper/figures/orbit_potential_context.{png,pdf}

Run:  python scripts/make_orbit_context_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from r5bp_nonsymmetric.periodic import Params, integrate
from r5bp_nonsymmetric.geometry import primary_positions

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")

A, B = 0.9, -0.1
FID = 3                           # Case-2 doubly-elliptic stable family
BG = "#0e1117"; FG = "#d6dde6"; GRID = "#262d38"
PRIM = ["#ff5c7a", "#5ab0ff", "#5ce1b6", "#ffb703"]
PMARK = ["o", "s", "^", "D"]


def rotating_vs_inertial(g, P, P4, m, msz, xeq, yeq):
    k = int(np.abs(g["amplitude"].values - 0.7).argmin())
    r = g.iloc[k]
    u0 = np.array([r["x0"], r["y0"], r["vx0"], r["vy0"]], float)
    T = float(r["T"])
    _, _, traj = integrate(u0, T, P, n_eval=2000)
    xr, yr = traj[:, 0], traj[:, 1]

    fig, (axr, axi) = plt.subplots(1, 2, figsize=(15, 7.4))
    fig.patch.set_facecolor(BG)

    axr.set_facecolor(BG)
    axr.plot(xr, yr, color="#ffd166", lw=2.0, zorder=4)
    axr.plot(xeq, yeq, marker="*", ms=16, mfc="#fff", mec=BG, mew=0.8, zorder=6)
    for i in range(4):
        axr.scatter(P4[i, 0], P4[i, 1], s=msz[i], marker=PMARK[i], c=PRIM[i],
                    edgecolors=BG, linewidths=0.8, zorder=7)
        axr.annotate(f"$m_{i}$", (P4[i, 0], P4[i, 1]), xytext=(9, -3),
                     textcoords="offset points", color=PRIM[i], fontsize=10)
    axr.scatter([0], [0], marker="+", c="#8b97a7", s=90, zorder=5)
    axr.set_aspect("equal"); axr.grid(True, color=GRID, lw=0.6)
    for sp in axr.spines.values():
        sp.set_color(GRID)
    axr.tick_params(colors=FG)
    axr.set_title("Rotating frame (co-rotating with the primaries)", color=FG,
                  fontsize=12, pad=8)
    axr.set_xlabel("$x$", color=FG); axr.set_ylabel("$y$", color=FG)

    axi.set_facecolor(BG)
    REV = 4.0
    nper = int(np.ceil(REV * 2 * np.pi / T))
    s = np.linspace(0, T, len(xr))
    tt, X, Y = [], [], []
    for n in range(nper):
        t = s + n * T
        tt.append(t)
        X.append(xr * np.cos(t) - yr * np.sin(t))
        Y.append(xr * np.sin(t) + yr * np.cos(t))
    tt = np.concatenate(tt); X = np.concatenate(X); Y = np.concatenate(Y)
    pts = np.array([X, Y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=tt[:-1], lw=1.1, alpha=0.9)
    axi.add_collection(lc)
    th = np.linspace(0, 2 * np.pi, 400)
    req = np.hypot(xeq, yeq)
    axi.plot(req * np.cos(th), req * np.sin(th), ls="--", color="#ffffff",
             lw=1.0, alpha=0.5)
    for i in range(4):
        ri = np.hypot(P4[i, 0], P4[i, 1])
        axi.plot(ri * np.cos(th), ri * np.sin(th), ls=":", color=PRIM[i],
                 lw=1.0, alpha=0.7)
        axi.scatter(P4[i, 0], P4[i, 1], s=msz[i], marker=PMARK[i], c=PRIM[i],
                    edgecolors=BG, linewidths=0.8, zorder=7)
    axi.scatter([0], [0], marker="+", c="#c9d1d9", s=110, zorder=6)
    axi.set_aspect("equal"); axi.grid(True, color=GRID, lw=0.6)
    for sp in axi.spines.values():
        sp.set_color(GRID)
    axi.tick_params(colors=FG)
    axi.set_title(f"Inertial frame ({nper} periods, ~{REV:.0f} revolutions)",
                  color=FG, fontsize=12, pad=8)
    axi.set_xlabel("$X$", color=FG); axi.set_ylabel("$Y$", color=FG)
    cb = fig.colorbar(lc, ax=axi, fraction=0.046, pad=0.02)
    cb.set_label("time", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.get_yticklabels(), color=FG)
    cb.outline.set_edgecolor(GRID)

    fig.suptitle(f"Family {FID} periodic orbit: same motion in two frames "
                 f"(amplitude A={r['amplitude']:.3f})", color="#fff",
                 fontsize=14, y=0.98)
    fig.text(0.5, 0.925, "Left: a closed libration loop about the equilibrium. "
             "Right: in inertial space the body traces a rosette confined to an "
             "annulus near the equilibrium's radius (dashed); primaries move on "
             "circles (dotted).", color="#9aa6b6", fontsize=9.5, ha="center")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.09, wspace=0.18)
    for ext in ("png", "pdf"):
        p = os.path.join(FIGDIR, f"orbit_rotating_vs_inertial.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, facecolor=BG)
        print("wrote", p)
    plt.close(fig)


def potential_context(g, P, P4, m, msz, xeq, yeq):
    fig, ax = plt.subplots(figsize=(9.6, 8.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    xlo, xhi, ylo, yhi = -3.2, 4.4, -3.2, 4.4
    gx = np.linspace(xlo, xhi, 700); gy = np.linspace(ylo, yhi, 700)
    GX, GY = np.meshgrid(gx, gy)
    acc = np.zeros_like(GX)
    for i in range(4):
        acc += m[i] / np.hypot(GX - P4[i, 0], GY - P4[i, 1])
    twoOm = GX * GX + GY * GY + 2 * acc

    vmin = float(twoOm.min()); vmax = float(np.percentile(twoOm, 80))
    cf = ax.contourf(GX, GY, twoOm, levels=np.linspace(vmin, vmax, 32),
                     cmap="cividis", extend="max", zorder=0)
    ax.contour(GX, GY, twoOm, levels=[vmin + 0.5, vmin + 1.5, vmin + 3.5, vmin + 7.0],
               colors="#cdd6e0", linewidths=0.7, alpha=0.7, zorder=1)

    amps = g["amplitude"].values
    idx = sorted({int(np.abs(amps - t).argmin())
                  for t in np.linspace(amps.min(), amps.max(), 6)})
    famcm = plt.get_cmap("cool")
    for j, kk in enumerate(idx):
        rr = g.iloc[kk]
        _, _, tj = integrate(np.array([rr["x0"], rr["y0"], rr["vx0"], rr["vy0"]], float),
                             float(rr["T"]), P, n_eval=600)
        ax.plot(tj[:, 0], tj[:, 1], color=famcm(j / max(1, len(idx) - 1)),
                lw=1.7, zorder=5)

    ax.plot(xeq, yeq, marker="*", ms=16, mfc="#fff", mec=BG, mew=0.8, zorder=8)
    for i in range(4):
        ax.scatter(P4[i, 0], P4[i, 1], s=msz[i], marker=PMARK[i], c=PRIM[i],
                   edgecolors="#000", linewidths=1.0, zorder=9)
        ax.annotate(f"$m_{i}$", (P4[i, 0], P4[i, 1]), xytext=(9, -3),
                    textcoords="offset points", color="#ffffff", fontsize=10, zorder=9)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=FG)
    ax.set_xlabel("$x$", color=FG); ax.set_ylabel("$y$", color=FG)
    ax.set_title(f"Family {FID} in context: effective-potential landscape",
                 color="#fff", fontsize=13, pad=10)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(r"effective potential $2\Omega$ (capped)", color=FG)
    cb.ax.yaxis.set_tick_params(color=FG)
    plt.setp(cb.ax.get_yticklabels(), color=FG)
    cb.outline.set_edgecolor(GRID)

    leg = [Line2D([0], [0], marker="*", color="none", mfc="#fff", mec=BG, ms=13,
                  label="libration point (equilibrium)"),
           Line2D([0], [0], color=famcm(0.5), lw=2, label="orbit family"),
           Line2D([0], [0], color="#cdd6e0", lw=1.2, label="zero-velocity curves")]
    ax.legend(handles=leg, loc="lower left", frameon=True, facecolor="#161b22",
              edgecolor=GRID, labelcolor=FG, fontsize=9)
    fig.text(0.5, 0.028, "The body librates in a local trough of the effective "
             "potential; the four primaries are deep singular wells.\nLevel curves "
             "are zero-velocity curves. Slice $(a,b)=(0.9,-0.1)$.",
             color="#9aa6b6", fontsize=9, ha="center")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.115)
    for ext in ("png", "pdf"):
        p = os.path.join(FIGDIR, f"orbit_potential_context.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, facecolor=BG)
        print("wrote", p)
    plt.close(fig)


def main():
    po = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    g = po[po.family_id == FID].sort_values("amplitude").reset_index(drop=True)
    c1, c2 = g["c1"].iloc[0], g["c2"].iloc[0]
    P = Params(A, B, c1, c2)
    m = P.m
    P4 = primary_positions(A, B, c1, c2)
    msz = 70 + 200 * (m / m.max()) ** 0.5
    xeq, yeq = g["x_eq"].iloc[0], g["y_eq"].iloc[0]

    os.makedirs(FIGDIR, exist_ok=True)
    rotating_vs_inertial(g, P, P4, m, msz, xeq, yeq)
    potential_context(g, P, P4, m, msz, xeq, yeq)


if __name__ == "__main__":
    main()
