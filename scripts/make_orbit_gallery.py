"""Dark-theme gallery of periodic-orbit families in the rotating frame.

For each of a set of representative families, integrates several members spanning
the family's amplitude range and plots them as nested closed curves, with the
parent equilibrium and the four primaries marked (primaries that fall outside a
panel are drawn as faded edge markers pointing toward their true location).

Reads:
    results/periodic_orbits.csv
    results/periodic_orbit_floquet.csv
Writes:
    paper/figures/periodic_orbit_gallery.png
    paper/figures/periodic_orbit_gallery.pdf

Run:  python scripts/make_orbit_gallery.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from r5bp_nonsymmetric.periodic import Params, integrate
from r5bp_nonsymmetric.geometry import primary_positions

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")

A, B = 0.9, -0.1                  # fixed geometric slice of the study

# dark theme
BG = "#0e1117"; FG = "#d6dde6"; GRID = "#262d38"
PRIM = ["#ff5c7a", "#5ab0ff", "#5ce1b6", "#ffb703"]     # m0 m1 m2 m3
PMARK = ["o", "s", "^", "D"]
CMAP = plt.get_cmap("plasma")

SHOW = [1, 3, 8, 9, 5, 10]        # representative families (2 x 3 grid)
NMEM = 9                          # members drawn per family
NEVAL = 700                       # samples per integrated orbit


def kind_of(label):
    l = str(label)
    if "saddlecentre" in l:
        return "saddle x centre (unstable)"
    if "res_3to2" in l:
        return "3:2 near-commensurability"
    if "res_2to1" in l:
        return "5:2 near-commensurability"   # corrected parent ratio (see paper)
    if "stable" in l:
        return "doubly-elliptic (stable)"
    return l


def floq_of(fid, cls):
    return {"spectrally stable": "Floquet stable",
            "real unstable": "real unstable",
            "spectrally stable (near transition)": "Floquet stable, near transition"
            }.get(cls.get(fid, ""), cls.get(fid, ""))


def comp_of(c):
    return "Case-1 arc" if "Case1" in c else "Case-2 loop"


def main():
    po = pd.read_csv(os.path.join(RESULTS, "periodic_orbits.csv"))
    fl = pd.read_csv(os.path.join(RESULTS, "periodic_orbit_floquet.csv"))
    cls = fl.groupby("family_id")["classification"].agg(
        lambda s: s.value_counts().index[0])

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.4))
    fig.patch.set_facecolor(BG)

    for ax, fid in zip(axes.ravel(), SHOW):
        ax.set_facecolor(BG)
        g = po[po.family_id == fid].sort_values("amplitude").reset_index(drop=True)
        c1, c2 = g["c1"].iloc[0], g["c2"].iloc[0]
        P = Params(A, B, c1, c2)
        m = P.m

        amps = g["amplitude"].values
        targets = np.linspace(amps.min(), amps.max(), NMEM)
        idx = sorted({int(np.abs(amps - t).argmin()) for t in targets})
        norm = colors.Normalize(vmin=amps[idx].min(), vmax=amps[idx].max() + 1e-12)

        allx, ally = [], []
        for k in idx:
            r = g.iloc[k]
            u0 = np.array([r["x0"], r["y0"], r["vx0"], r["vy0"]], float)
            try:
                _, _, traj = integrate(u0, float(r["T"]), P, n_eval=NEVAL)
            except Exception as e:
                print("skip family", fid, "member", int(r["member"]), e)
                continue
            col = CMAP(0.12 + 0.83 * norm(r["amplitude"]))
            ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.4, alpha=0.95, zorder=3)
            allx += [traj[:, 0].min(), traj[:, 0].max()]
            ally += [traj[:, 1].min(), traj[:, 1].max()]

        # zoom to the orbit family (+ its parent equilibrium)
        xeq, yeq = g["x_eq"].iloc[0], g["y_eq"].iloc[0]
        allx += [xeq]; ally += [yeq]
        xmin, xmax = min(allx), max(allx); ymin, ymax = min(ally), max(ally)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half = 0.62 * max(xmax - xmin, ymax - ymin) + 0.06
        xlim = (cx - half, cx + half); ylim = (cy - half, cy + half)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")

        ax.plot(xeq, yeq, marker="*", ms=15, mfc="#ffffff", mec=BG, mew=0.8, zorder=6)

        P4 = primary_positions(A, B, c1, c2)
        msz = 60 + 190 * (m / m.max()) ** 0.5
        for i in range(4):
            px, py = P4[i, 0], P4[i, 1]
            if xlim[0] <= px <= xlim[1] and ylim[0] <= py <= ylim[1]:
                ax.scatter(px, py, s=msz[i], marker=PMARK[i], c=PRIM[i],
                           edgecolors=BG, linewidths=0.8, zorder=7)
            else:
                dx, dy = px - cx, py - cy
                ts = []
                if dx > 0: ts.append((xlim[1] - cx) / dx)
                elif dx < 0: ts.append((xlim[0] - cx) / dx)
                if dy > 0: ts.append((ylim[1] - cy) / dy)
                elif dy < 0: ts.append((ylim[0] - cy) / dy)
                t = min([tt for tt in ts if tt > 0]) if ts else 1.0
                ex, ey = cx + dx * t * 0.90, cy + dy * t * 0.90
                ax.scatter(ex, ey, s=0.7 * msz[i], marker=PMARK[i], c=PRIM[i],
                           alpha=0.5, edgecolors=BG, linewidths=0.6, zorder=7)
                ax.annotate(f"$m_{i}$", (ex, ey),
                            xytext=(-11 * np.sign(dx), -11 * np.sign(dy)),
                            textcoords="offset points", color=PRIM[i],
                            fontsize=8.5, ha="center", va="center", zorder=8)

        ax.grid(True, color=GRID, lw=0.6)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=FG, labelsize=8)
        ax.set_title(f"Family {fid}   {comp_of(g['component'].iloc[0])}\n"
                     f"{kind_of(g['label'].iloc[0])}  |  {floq_of(fid, cls)}",
                     color=FG, fontsize=10.5, pad=8)
        ax.text(0.03, 0.03, f"{len(idx)} of {len(g)} members\n"
                f"A = {amps[idx].min():.3f} to {amps[idx].max():.3f}",
                transform=ax.transAxes, color="#8b97a7", fontsize=7.5,
                va="bottom", ha="left")

    leg = [Line2D([0], [0], marker="*", color="none", mfc="#fff", mec=BG,
                  ms=13, label="parent equilibrium")]
    leg += [Line2D([0], [0], marker=PMARK[i], color="none", mfc=PRIM[i],
                   mec=BG, ms=10, label=f"primary $m_{i}$") for i in range(4)]
    leg += [Line2D([0], [0], color=CMAP(0.15), lw=2, label="small amplitude"),
            Line2D([0], [0], color=CMAP(0.95), lw=2, label="large amplitude")]
    fig.legend(handles=leg, loc="lower center", ncol=7, frameon=False,
               labelcolor=FG, fontsize=9.5, bbox_to_anchor=(0.5, 0.005))

    fig.suptitle("Periodic-orbit families of the restricted five-body problem "
                 "(rotating frame)", color="#ffffff", fontsize=15.5, y=0.985)
    fig.text(0.5, 0.945, "Each curve is one differential-corrected periodic orbit; "
             "colour tracks amplitude within the family. Panels are zoomed on each "
             "family; faded edge markers point to off-frame primaries. Slice "
             "$(a,b)=(0.9,-0.1)$.", color="#9aa6b6", fontsize=10, ha="center")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.075,
                        wspace=0.16, hspace=0.28)

    os.makedirs(FIGDIR, exist_ok=True)
    png = os.path.join(FIGDIR, "periodic_orbit_gallery.png")
    pdf = os.path.join(FIGDIR, "periodic_orbit_gallery.pdf")
    fig.savefig(png, dpi=200, facecolor=BG)
    fig.savefig(pdf, facecolor=BG)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
