"""Figures for the continuation/bifurcation stage, built from the result CSVs.

Outputs (into paper/figures/):
    cc_continuation_path.pdf
    equilibrium_count_vs_s.pdf
    equilibrium_branches_x_vs_s.pdf
    equilibrium_branches_y_vs_s.pdf
    equilibrium_branches_r_vs_s.pdf

Verified bifurcation points (results/bifurcation_points.csv) are marked
explicitly. No Hill-region or periodic-orbit figures are produced.

The equilibrium-branch figures take their branch identities from the adaptive
predictor-based tracker (results/branch_spectra.csv), the same scheme used for
the stability/frequency/Jacobi/periodic analyses; this requires
analyze_spectra_and_folds.py to have been run first.

Run:  python scripts/make_continuation_figures.py
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
from r5bp_nonsymmetric.central_config import psi, mass_ratios  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
A, B = 0.9, -0.1
COMPS = ["Case1-arc", "Case2-loop"]
DISP = {"Case1-arc": "Case-1 arc", "Case2-loop": "Case-2 arc"}
CCOL = {"Case1-arc": "#1f77b4", "Case2-loop": "#c1121f"}


def _load():
    # Branch identities for the branch-diagram figures come from the adaptive
    # predictor-based tracker (results/branch_spectra.csv, produced by
    # analyze_spectra_and_folds.py::stitch) -- the SAME tracker used for the
    # stability, centre-frequency, Jacobi, and periodic-orbit-parent analyses,
    # so every branch-dependent figure derives from one authoritative
    # branch-identification scheme. (Its (s,x,y) rows are identical to the raw
    # equilibrium_branches.csv; only the branch_id labelling differs -- the
    # adaptive tracker follows the escaping Case-1 branches without fragmenting
    # them. The stable-point highlighting and fold markers are data-driven and
    # tracker-independent.)
    branch = pd.read_csv(os.path.join(RESULTS, "branch_spectra.csv"))
    branch = branch.rename(columns={"max_real": "max_real_eigenvalue"})
    return (pd.read_csv(os.path.join(RESULTS, "cc_continuation_path.csv")),
            pd.read_csv(os.path.join(RESULTS, "equilibrium_count_vs_s.csv")),
            branch,
            pd.read_csv(os.path.join(RESULTS, "bifurcation_points.csv")))


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def fig_cc_path(path, bif):
    # psi=0 contour background
    n = 500
    c1 = np.linspace(-1.4, 1.4, n); c2 = np.linspace(-1.4, 1.4, n)
    C1, C2 = np.meshgrid(c1, c2)
    PS = np.empty_like(C1)
    for i in range(n):
        for j in range(n):
            PS[i, j] = psi(A, B, C1[i, j], C2[i, j])
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.contour(C1, C2, PS, levels=[0], colors="0.75", linewidths=0.8)
    for comp in COMPS:
        d = path[path.component == comp]
        ax.plot(d.c1, d.c2, color=CCOL[comp], lw=2.6, label=f"{DISP[comp]} (admissible)")
    # endpoints
    ep = pd.read_csv(os.path.join(RESULTS, "refined_reference_cases.csv"))
    p1 = ep[ep.case == "Case 1"].iloc[0]; p2 = ep[ep.case == "Case 2"].iloc[0]
    ax.plot(p1.c1_refined, p1.c2_refined, "k*", ms=16, label="Case 1")
    ax.plot(p2.c1_refined, p2.c2_refined, "kD", ms=10, label="Case 2")
    # bifurcation points, numbered F1..F7 in the same order as Table 2
    # (Case-1 folds by s, then Case-2 folds by s; the boundary row drops out
    #  under dropna because it has no critical (c1_c,c2_c)).
    if "c1_c" in bif.columns:
        bb = bif.dropna(subset=["c1_c"])
        fold = bb[bb.classification.str.contains("saddle-node")].reset_index(drop=True)
        if len(fold):
            ax.scatter(fold.c1_c, fold.c2_c, s=90, facecolor="gold",
                       edgecolor="black", zorder=6, label="fold bifurcation")
            # small per-point label offsets so the F-ids do not overlap markers
            offs = [(0.06, 0.05), (0.06, 0.05), (-0.14, 0.04), (0.07, 0.05),
                    (0.07, -0.11), (-0.15, 0.04), (0.06, 0.05)]
            for k, r in fold.iterrows():
                dx, dy = offs[k] if k < len(offs) else (0.06, 0.05)
                ax.annotate(f"$F_{{{k+1}}}$", (r.c1_c, r.c2_c),
                            xytext=(r.c1_c + dx, r.c2_c + dy), fontsize=10,
                            fontweight="bold", zorder=7)

            # inset: magnify the crowded Case-2 fold cluster (F4..F7)
            cluster = fold.iloc[3:7]
            if len(cluster) == 4:
                x0, x1 = cluster.c1_c.min() - 0.05, cluster.c1_c.max() + 0.05
                y0, y1 = cluster.c2_c.min() - 0.05, cluster.c2_c.max() + 0.05
                axin = ax.inset_axes([0.63, 0.60, 0.35, 0.35])
                d2 = path[path.component == "Case2-loop"]
                axin.plot(d2.c1, d2.c2, color=CCOL["Case2-loop"], lw=2.2)
                axin.scatter(cluster.c1_c, cluster.c2_c, s=70, facecolor="gold",
                             edgecolor="black", zorder=6)
                for k, r in cluster.iterrows():
                    axin.annotate(f"$F_{{{k+1}}}$", (r.c1_c, r.c2_c),
                                  xytext=(r.c1_c + 0.012, r.c2_c + 0.012),
                                  fontsize=9, fontweight="bold", zorder=7)
                axin.set_xlim(x0, x1); axin.set_ylim(y0, y1)
                axin.set_aspect("equal")
                axin.set_xticks([]); axin.set_yticks([])
                axin.set_title("$F_4$--$F_7$ (zoom)", fontsize=8)
                ax.indicate_inset_zoom(axin, edgecolor="0.4")
    ax.set_xlabel("$c_1$"); ax.set_ylabel("$c_2$"); ax.set_aspect("equal")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Admissible CC arcs (grey: full $\\psi=0$ level set)")
    _save(fig, "cc_continuation_path.pdf")


def fig_count(count, bif):
    fig, axes = plt.subplots(len(COMPS), 1, figsize=(7.5, 6.5), sharex=True)
    for ax, comp in zip(axes, COMPS):
        d = count[count.component == comp].sort_values("s")
        nmin, nmax = int(d.N_eq.min()), int(d.N_eq.max())
        ax.set_ylim(nmin - 0.6, nmax + 1.1)
        ax.step(d.s, d.N_eq, where="mid", color=CCOL[comp], lw=2)
        ax.scatter(d.s, d.N_eq, s=14, color=CCOL[comp])
        for _, r in bif[bif.component == comp].iterrows():
            ax.axvline(r["s_c"], color="0.5", ls="--", lw=1)
            lbl = "fold" if "saddle-node" in str(r["classification"]) else "boundary"
            ax.text(r["s_c"], nmax + 1.0, lbl, rotation=90,
                    va="top", ha="right", fontsize=7, color="0.35")
        ax.set_ylabel("$N_{eq}$"); ax.set_title(DISP[comp], fontsize=10)
        ax.set_yticks(list(range(nmin, nmax + 1))); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("arc parameter $s$")
    fig.tight_layout()
    _save(fig, "equilibrium_count_vs_s.pdf")


def _fold_ids(bif):
    """Map each saddle-node fold to its global F-index (F1..F7), ordered exactly
    as in Table 2 (Case-1 folds then Case-2 folds, in CSV order)."""
    ids = {}
    k = 0
    for _, r in bif.iterrows():
        if "saddle-node" in str(r["classification"]):
            k += 1
            ids[(r["component"], round(float(r["s_c"]), 6))] = k
    return ids


def _branch_plot(branch, bif, ycol, ylabel, fname, transform=None):
    """Branch diagram with an explicit taxonomy so the figure reads at a glance:
    ordinary branches muted grey; spectrally stable points highlighted; the
    created pair B_a and the pre-existing pair B_b of the N=11 window picked out;
    folds marked and labelled F_i (consistent with Table 2 and Figure 3)."""
    fold_ids = _fold_ids(bif)
    fig, axes = plt.subplots(len(COMPS), 1, figsize=(7.5, 8.4), sharex=True)
    for ax, comp in zip(axes, COMPS):
        d = branch[branch.component == comp]
        # identify the N=11 pairs on the Case-2 arc (robust structural rules)
        ba, bb = set(), set()
        if comp == "Case2-loop":
            for bid, g in d.groupby("branch_id"):
                if abs(g.s.min() - 0.768) < 0.02:          # born at F4 (9->11)
                    ba.add(bid)
                elif 0.75 <= g.s.max() <= 0.83 and g.y.mean() < -0.5:
                    bb.add(bid)                            # lower pair, gone by F5
        seen = set()

        def _lab(key, text):
            if key in seen:
                return None
            seen.add(key)
            return text

        for bid, g in d.groupby("branch_id"):
            g = g.sort_values("s")
            yy = transform(g) if transform else g[ycol]
            ax.plot(g.s, yy, color="0.80", lw=0.8, zorder=1)     # muted base
            if bid in ba:
                ax.plot(g.s, yy, color="#e07a00", lw=2.6, zorder=5,
                        label=_lab("ba", "$B_a$ (created at $F_4$)"))
            if bid in bb:
                ax.plot(g.s, yy, color="#7b2cbf", lw=2.6, zorder=5,
                        label=_lab("bb", "$B_b$ (pre-existing pair, destroyed at $F_5$)"))
        # spectrally stable points (data-driven; tracker splits them across ids)
        st = d[d.max_real_eigenvalue.abs() < 1e-6]
        if len(st):
            yst = transform(st) if transform else st[ycol]
            ax.scatter(st.s, yst, s=22, facecolor="#2a9d8f", edgecolors="k",
                       linewidths=0.3, zorder=6,
                       label="spectrally stable equilibria")
        # folds: dashed verticals + F-labels at the top axis edge
        for _, r in bif[bif.component == comp].iterrows():
            ax.axvline(r["s_c"], color="0.55", ls="--", lw=1, zorder=2)
            fid = fold_ids.get((comp, round(float(r["s_c"]), 6)))
            if fid is not None:
                ax.text(r["s_c"], 0.98, f"$F_{{{fid}}}$",
                        transform=ax.get_xaxis_transform(),
                        va="top", ha="center", fontsize=8, color="0.3")
        ax.set_ylabel(ylabel); ax.set_title(DISP[comp], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("arc parameter $s$")
    fig.tight_layout()
    _save(fig, fname)


def main():
    path, count, branch, bif = _load()
    fig_cc_path(path, bif)
    fig_count(count, bif)
    _branch_plot(branch, bif, "x", "$x_i(s)$", "equilibrium_branches_x_vs_s.pdf")
    _branch_plot(branch, bif, "y", "$y_i(s)$", "equilibrium_branches_y_vs_s.pdf")
    _branch_plot(branch, bif, "r", "$r_i(s)=\\sqrt{x^2+y^2}$",
                 "equilibrium_branches_r_vs_s.pdf",
                 transform=lambda g: np.hypot(g.x, g.y))


if __name__ == "__main__":
    main()
