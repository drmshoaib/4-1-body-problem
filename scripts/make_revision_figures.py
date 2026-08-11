"""Figures added in the manuscript revision:
    model_schematic.pdf     the four-primary geometry with unequal masses
    workflow_schematic.pdf  the numerical-method pipeline
    fold_zoom_main.pdf       three representative fold panels (main paper)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from r5bp_nonsymmetric.central_config import mass_ratios  # noqa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def model_schematic():
    """Clean conceptual schematic of the (4+1) geometry: four primaries at their
    symbolic normalized positions, the massless test particle, and the rotating
    frame. Deliberately not a data plot -- no axes box, no grid; marker area
    conveys the (unequal) masses only qualitatively."""
    # symbolic normalized positions; a representative admissible geometry is used
    # only to place r_3 = (c_1,c_2) sensibly, and mass markers are scaled softly
    # so that no single numerical case "styles" the figure.
    a, b, c1, c2 = 0.9, -0.1, -0.61, 0.715655
    rho0, rho1, rho2 = mass_ratios(a, b, c1, c2)
    prim = [("m_0", "(0,\\,b)", 0.0, b, rho0),
            ("m_1", "(-1,\\,0)", -1.0, 0.0, rho1),
            ("m_2", "(0,\\,a)", 0.0, a, rho2),
            ("m_3", "(c_1,\\,c_2)", c1, c2, 1.0)]
    # test-particle position: a generic point in the field (not an equilibrium)
    px, py = -0.55, 0.15

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.axis("off")
    ax.set_aspect("equal")

    # faint edges of the four-body configuration (convey "central configuration")
    xs = [p[2] for p in prim]; ys = [p[3] for p in prim]
    order = [1, 0, 3, 2, 1]  # a simple closed quadrilateral through the primaries
    ax.plot([xs[i] for i in order], [ys[i] for i in order],
            color="0.82", lw=1.0, ls="-", zorder=1)

    # light reference axes with arrowheads (rotating-frame axes), unobtrusive
    ax.annotate("", xy=(1.05, 0), xytext=(-1.75, 0),
                arrowprops=dict(arrowstyle="->", color="0.7", lw=0.9), zorder=0)
    ax.annotate("", xy=(0, 1.35), xytext=(0, -0.55),
                arrowprops=dict(arrowstyle="->", color="0.7", lw=0.9), zorder=0)
    ax.text(1.06, -0.02, "$x$", color="0.55", fontsize=11, va="top")
    ax.text(0.03, 1.36, "$y$", color="0.55", fontsize=11)

    # primaries: marker area softly increasing with mass, symbolic labels
    mmax = max(p[4] for p in prim)
    for name, coord, x, y, mval in prim:
        size = 260 * (mval / mmax) ** (1 / 3.0) + 90
        ax.scatter(x, y, s=size, color="#2b3a55", edgecolors="white",
                   linewidths=1.2, zorder=5)
        dx = 0.10 if x > -0.9 else 0.12
        ax.annotate(f"${name}$\n${coord}$", (x, y), textcoords="offset points",
                    xytext=(12, 8), fontsize=10.5, zorder=6)

    # massless test particle
    ax.scatter([px], [py], s=26, facecolor="white", edgecolors="#c1121f",
               linewidths=1.4, zorder=6)
    ax.annotate("$P$ (massless)", (px, py), textcoords="offset points",
                xytext=(8, -14), fontsize=9.5, color="#c1121f", zorder=6)

    # rotating-frame indicator: a small curved arrow near the origin
    ax.annotate("", xy=(0.34, 0.30), xytext=(0.30, 0.05),
                arrowprops=dict(arrowstyle="->", color="#457b9d", lw=1.4,
                                connectionstyle="arc3,rad=0.5"), zorder=4)
    ax.text(0.40, 0.24, "$\\omega$", color="#457b9d", fontsize=11, zorder=4)

    ax.set_xlim(-1.85, 1.15); ax.set_ylim(-0.7, 1.5)
    _save(fig, "model_schematic.pdf")


def workflow_schematic():
    steps = [
        "Four-body CC\nconstraint $\\psi=0$",
        "Admissible pseudo-\narclength continuation",
        "Global equilibrium\nsearch $\\nabla\\Omega=0$",
        "Fold detection\n$\\det\\mathrm{H}\\Omega=0$",
        "Linear stability\n(variational spectrum)",
        "Periodic-orbit\ncontinuation + Floquet",
        "Jacobi levels &\nHill topology",
    ]
    fig, ax = plt.subplots(figsize=(13.5, 2.5))
    ax.axis("off")
    n = len(steps)
    w, h, gap = 1.55, 1.0, 0.35
    x = 0.0
    centers = []
    for i, s in enumerate(steps):
        box = FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#22303f", facecolor="#e8eef4")
        ax.add_patch(box)
        ax.text(x + w / 2, h / 2, s, ha="center", va="center", fontsize=8.5)
        centers.append((x + w, h / 2))
        x += w + gap
    for i in range(n - 1):
        cx, cy = centers[i]
        arr = FancyArrowPatch((cx, cy), (cx + gap, cy), arrowstyle="-|>",
                              mutation_scale=14, linewidth=1.2, color="#22303f")
        ax.add_patch(arr)
    ax.set_xlim(-0.2, x); ax.set_ylim(-0.2, h + 0.2)
    ax.set_aspect("equal")
    _save(fig, "workflow_schematic.pdf")


def fold_zoom_main():
    sw = pd.read_csv(os.path.join(RESULTS, "fold_local_sweeps.csv"))
    picks = [("Case1-arc", "5->7", "Case-1 fold ($5\\to7$)"),
             ("Case2-loop", "7->9", "Case-2 fold ($7\\to9$)"),
             ("Case2-loop", "9->11", "$N=11$-bounding fold ($9\\to11$)")]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    for ax, (comp, cc, title) in zip(axes, picks):
        d = sw[(sw.component == comp) & (sw.count_change == cc)].sort_values("dl")
        ax.plot(d.dl, d.x1, "-o", ms=2.5, color="#1f77b4", label="$x_1$")
        ax.plot(d.dl, d.x2, "-o", ms=2.5, color="#d62728", label="$x_2$")
        ax.axvline(0, color="0.5", ls="--", lw=1)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"$\ell-\ell_c$"); ax.set_ylabel("$x$ of merging pair")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Representative saddle-node folds: the merging pair meets at "
                 "$\\ell=\\ell_c$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fold_zoom_main.pdf")


def main():
    model_schematic()
    workflow_schematic()
    fold_zoom_main()


if __name__ == "__main__":
    main()
