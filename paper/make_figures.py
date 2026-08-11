"""Generate the publication figure of the four VERIFIED reference cases directly
from the result CSVs. No continuation / bifurcation / Hill-region content.

Sources (verified outputs only):
    results/refined_reference_cases.csv   -> refined primary geometry per case
    results/refined_equilibria.csv     -> recomputed equilibria + stability

Outputs:
    paper/figures/verified_cases_equilibria.pdf
    paper/figures/verified_cases_equilibria.png

Run:  python paper/make_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

PRIMARY_COLOR = "black"
UNSTABLE_COLOR = "#c1121f"
STABLE_COLOR = "#2a9d8f"


def primaries(row):
    """Refined primary positions (m0, m1, m2, m3) for a case row."""
    a, b = row["a"], row["b"]
    c1, c2 = row["c1_refined"], row["c2_refined"]
    return [
        ("$m_0$", 0.0, b),
        ("$m_1$", -1.0, 0.0),
        ("$m_2$", 0.0, a),
        ("$m_3$", c1, c2),
    ]


def main():
    dfc = pd.read_csv(os.path.join(RESULTS, "refined_reference_cases.csv"))
    dfe = pd.read_csv(os.path.join(RESULTS, "refined_equilibria.csv"))

    cases = list(dfc["case"])
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.8))
    axes = axes.ravel()

    for ax, case in zip(axes, cases):
        crow = dfc[dfc["case"] == case].iloc[0]
        eqs = dfe[dfe["case"] == case]

        xs, ys = [], []

        # primaries
        for lbl, px, py in primaries(crow):
            ax.scatter(px, py, marker="*", s=200, color=PRIMARY_COLOR,
                       zorder=5, edgecolors="white", linewidths=0.6)
            ax.annotate(lbl, (px, py), textcoords="offset points",
                        xytext=(-17, 4), fontsize=8, color="0.30")
            xs.append(px); ys.append(py)

        # equilibria (branch-neutral labels E1..EN, indexed as in Table 3)
        for _, e in eqs.iterrows():
            stable = bool(e["stable"])
            ax.scatter(
                e["x"], e["y"],
                marker=("D" if stable else "o"),
                s=(55 if stable else 34),
                facecolor=(STABLE_COLOR if stable else UNSTABLE_COLOR),
                edgecolors="black", linewidths=(1.0 if stable else 0.4),
                zorder=6,
            )
            ax.annotate(f"$E_{{{int(e['index'])}}}$", (e["x"], e["y"]),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=7.5, color="black")
            xs.append(e["x"]); ys.append(e["y"])

        # equal aspect, padded limits
        pad_x = 0.12 * (max(xs) - min(xs) + 1e-9) + 0.35
        pad_y = 0.12 * (max(ys) - min(ys) + 1e-9) + 0.35
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="0.85", lw=0.6, zorder=0)
        ax.axvline(0, color="0.85", lw=0.6, zorder=0)
        ax.grid(True, color="0.92", lw=0.5, zorder=0)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        n = int(crow["recomputed_count"])
        ax.set_title(f"{case}: {n} equilibria", fontsize=11)

    # single shared legend
    handles = [
        Line2D([0], [0], marker="*", color="white", markerfacecolor=PRIMARY_COLOR,
               markeredgecolor="white", markersize=13, label="Primary $m_i$"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=UNSTABLE_COLOR,
               markeredgecolor="black", markersize=8, label="Equilibrium (unstable)"),
        Line2D([0], [0], marker="D", color="white", markerfacecolor=STABLE_COLOR,
               markeredgecolor="black", markersize=8,
               label="Equilibrium (spectrally stable)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005), fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    pdf = os.path.join(FIGDIR, "verified_cases_equilibria.pdf")
    png = os.path.join(FIGDIR, "verified_cases_equilibria.png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(pdf, ROOT))
    print("wrote", os.path.relpath(png, ROOT))


if __name__ == "__main__":
    main()
