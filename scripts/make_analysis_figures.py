"""Publication figures for the spectral / local-fold analysis, from the CSVs
written by analyze_spectra_and_folds.py. No bifurcation conclusion is changed.

Outputs (paper/figures/):
    sigma_max_re_vs_s.pdf
    centre_frequencies_vs_s.pdf
    frequency_ratio_vs_s.pdf
    cascade_geometry_snapshots.pdf
    fold_zoom_branches.pdf
    fold_separation_exponent.pdf
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
COMPS = ["Case1-arc", "Case2-loop"]
DISP = {"Case1-arc": "Case-1 arc", "Case2-loop": "Case-2 arc"}
LOW_ORDER = [(1, 1), (3, 2), (2, 1), (5, 2), (3, 1), (4, 1)]


def _save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def _folds(bif, comp):
    d = bif[(bif.component == comp) & bif.classification.str.contains("saddle-node", na=False)]
    return d["s_c"].tolist()


def fig_sigma(spec, bif):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2))
    for ax, comp in zip(axes, COMPS):
        d = spec[(spec.component == comp) & spec.reliable]
        for bid, g in d.groupby("branch_id"):
            g = g.sort_values("s")
            if len(g) < 2:            # drop isolated single-sample edge artefacts
                continue
            ax.plot(g.s, g.max_real, lw=1.2, marker="o", ms=2)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        for sc in _folds(bif, comp):
            ax.axvline(sc, color="0.6", ls="--", lw=0.9)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylabel(r"$\sigma_i=\max\operatorname{Re}\lambda_i$")
        ax.set_title(DISP[comp], fontsize=10); ax.grid(alpha=0.3)
    axes[-1].set_xlabel("arc parameter $s$")
    fig.suptitle("Leading eigenvalue real part along reliable branches "
                 "(symlog; $\\sigma=0$ dotted)", fontsize=11)
    fig.tight_layout()
    _save(fig, "sigma_max_re_vs_s.pdf")


def fig_frequencies(spec, bif):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2))
    for ax, comp in zip(axes, COMPS):
        d = spec[(spec.component == comp) & spec.reliable & (spec.n_centre_pairs >= 2)]
        for bid, g in d.groupby("branch_id"):
            g = g.sort_values("s")
            if len(g) < 2:            # drop isolated single-sample edge artefacts
                continue
            ax.plot(g.s, g.omega1, ".", ms=4, color="#1f77b4")
            ax.plot(g.s, g.omega2, ".", ms=4, color="#d62728")
        ax.plot([], [], ".", color="#1f77b4", label=r"$\omega_1$")
        ax.plot([], [], ".", color="#d62728", label=r"$\omega_2$")
        for sc in _folds(bif, comp):
            ax.axvline(sc, color="0.6", ls="--", lw=0.9)
        ax.set_ylabel(r"centre frequency $\omega$")
        ax.set_title(f"{DISP[comp]}: purely-imaginary (spectrally stable) intervals",
                     fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("arc parameter $s$")
    fig.tight_layout()
    _save(fig, "centre_frequencies_vs_s.pdf")


def fig_ratio(spec):
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 7.2))
    for ax, comp in zip(axes, COMPS):
        d = spec[(spec.component == comp) & spec.reliable
                 & (spec.n_centre_pairs >= 2) & np.isfinite(spec.ratio)]
        d = d.groupby("branch_id").filter(lambda g: len(g) >= 2)  # drop edge artefacts
        for bid, g in d.groupby("branch_id"):
            g = g.sort_values("s")
            ax.plot(g.s, g.ratio, "-o", ms=3, color="#2a9d8f")
        if len(d):
            ylo, yhi = d.ratio.min() - 0.3, d.ratio.max() + 0.3
            for (p, q) in LOW_ORDER:
                yv = p / q
                if ylo <= yv <= yhi:
                    ax.axhline(yv, color="0.7", ls=":", lw=1)
                    ax.text(0.01, yv, f"{p}:{q}", fontsize=7, va="bottom",
                            color="0.4", transform=ax.get_yaxis_transform())
        # flag near-resonant points
        for _, r in d.iterrows():
            for (p, q) in LOW_ORDER:
                if abs(r.ratio - p / q) < 0.04:
                    ax.plot(r.s, r.ratio, "*", ms=11, color="crimson")
        ax.set_ylabel(r"$\omega_1/\omega_2$")
        ax.set_title(f"{DISP[comp]}: centre-frequency ratio "
                     "(red star = near-commensurability indicator)",
                     fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("arc parameter $s$")
    fig.tight_layout()
    _save(fig, "frequency_ratio_vs_s.pdf")


def fig_snapshots(snap):
    order = ["7a", "9a", "11", "9b", "7b", "5"]
    titles = {"7a": "N=7", "9a": "N=9", "11": "N=11",
              "9b": "N=9", "7b": "N=7", "5": "N=5"}
    # Branch identities of the two pairs involved in the N=11 window
    # (created pair B_a born at F4, pre-existing pair B_b destroyed at F5).
    # Anchors are the pair centroids; each pair is annotated only on the panels
    # where it is present, so the "created != destroyed" result is visible.
    BA = (-0.37, 0.30)   # created pair, present at N=11 and the following N=9
    BB = (0.50, -1.69)   # pre-existing pair, present at N=9 (first) and N=11
    pair_labels = {"9a": [("$B_b$", BB)],
                   "11": [("$B_a$", BA), ("$B_b$", BB)],
                   "9b": [("$B_a$", BA)]}
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6))
    for ax, lbl in zip(axes.ravel(), order):
        d = snap[snap.label == lbl]
        prim = d[d.stable == "primary"]
        eqs = d[d.stable != "primary"]
        ax.scatter(prim.x, prim.y, marker="*", s=170, color="k",
                   edgecolors="white", linewidths=0.6, zorder=5)
        un = eqs[eqs.stable.astype(str) == "False"]
        st = eqs[eqs.stable.astype(str) == "True"]
        ax.scatter(un.x, un.y, s=34, facecolor="#c1121f", edgecolors="k",
                   linewidths=0.4, zorder=6)
        if len(st):
            ax.scatter(st.x, st.y, s=55, marker="D", facecolor="#2a9d8f",
                       edgecolors="k", zorder=6)
        # branch-identity annotations with a light circle around the pair
        for txt, (ax0, ay0) in pair_labels.get(lbl, []):
            ax.annotate(txt, (ax0, ay0), xytext=(ax0 + 0.28, ay0 + 0.20),
                        fontsize=11, fontweight="bold", color="#1d3557",
                        zorder=8,
                        arrowprops=dict(arrowstyle="-", color="#1d3557", lw=0.8))
            ax.scatter([ax0], [ay0], s=430, facecolor="none",
                       edgecolors="#1d3557", linewidths=1.1, zorder=7)
        s = d.s.iloc[0]
        ax.set_title(f"{titles[lbl]}  ($s={s:.3f}$)", fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    fig.suptitle("Case-2 equilibrium cascade", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _save(fig, "cascade_geometry_snapshots.pdf")


def fig_fold_zoom(sw):
    keys = list(dict.fromkeys(zip(sw.component, sw.count_change)))
    n = len(keys)
    ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (comp, cc) in zip(axes, keys):
        d = sw[(sw.component == comp) & (sw.count_change == cc)].sort_values("dl")
        ax.plot(d.dl, d.x1, "-o", ms=2.5, color="#1f77b4", label="$x_1$")
        ax.plot(d.dl, d.x2, "-o", ms=2.5, color="#d62728", label="$x_2$")
        ax.axvline(0, color="0.5", ls="--", lw=1)
        ax.set_title(f"{DISP[comp]} {cc}", fontsize=9)
        ax.set_xlabel(r"$\ell-\ell_c$ (arclength)"); ax.set_ylabel("$x$ of pair")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle("Local branch structure at each saddle-node "
                 "(merging pair coordinates; fold at $\\ell=\\ell_c$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fold_zoom_branches.pdf")


def fig_separation(sw):
    """Left: one representative fold (log-log, with fit). Right: the fitted
    exponents of all seven folds against the theoretical line gamma=1/2."""
    keys = list(dict.fromkeys(zip(sw.component, sw.count_change)))

    def existence_side(comp, cc):
        d = sw[(sw.component == comp) & (sw.count_change == cc)]
        best = None
        for sgn in (+1, -1):
            s2 = d[d.sign == sgn].copy()
            if len(s2) >= 5:
                s2 = s2.reindex(s2.dl.abs().sort_values().index)
                if best is None or s2.d.values[0] < best[1]:
                    best = (s2, s2.d.values[0])
        return best[0] if best else None

    def fit_gamma(s2):
        inner = s2[s2.dl.abs() <= 0.04]
        if len(inner) < 4:
            inner = s2.head(8)
        g, b0 = np.polyfit(np.log(np.abs(inner.dl)), np.log(inner.d), 1)
        return g, b0, inner

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # left: representative fold (Case-2 7->9)
    rep = ("Case2-loop", "7->9")
    s2 = existence_side(*rep); g, b0, inner = fit_gamma(s2)
    axL.plot(np.abs(s2.dl), s2.d, "o", ms=4, color="#1f77b4", label="data")
    xx = np.linspace(np.abs(s2.dl).min(), np.abs(s2.dl).max(), 60)
    axL.plot(xx, np.exp(b0) * xx ** g, "-", color="#c1121f", lw=1.4,
             label=f"fit $\\gamma={g:.2f}$")
    axL.plot(xx, np.exp(b0) * xx ** 0.5, "--", color="0.5", lw=1.2,
             label=r"slope $\frac{1}{2}$")
    axL.set_xscale("log"); axL.set_yscale("log")
    axL.set_xlabel(r"$|\ell-\ell_c|$"); axL.set_ylabel(r"pair separation $d$")
    axL.set_title(r"Representative fold (Case-2 $7\to9$)", fontsize=10)
    axL.grid(alpha=0.3, which="both"); axL.legend(fontsize=8, loc="best")

    # right: fitted gamma for all folds
    labels, gammas = [], []
    for (comp, cc) in keys:
        s2 = existence_side(comp, cc)
        if s2 is None:
            continue
        g, _, _ = fit_gamma(s2)
        labels.append(f"{comp.replace('-arc','').replace('-loop','')}\n{cc}")
        gammas.append(g)
    xpos = np.arange(len(gammas))
    axR.axhline(0.5, color="0.5", ls="--", lw=1.3, label=r"$\gamma=\frac{1}{2}$")
    axR.axhspan(0.45, 0.55, color="0.85", alpha=0.5, zorder=0)
    axR.plot(xpos, gammas, "o", ms=8, color="#2a9d8f")
    axR.set_xticks(xpos); axR.set_xticklabels(labels, fontsize=7)
    axR.set_ylim(0.3, 0.7); axR.set_ylabel(r"fitted exponent $\gamma$")
    axR.set_title(r"Fitted exponents vs. $\gamma=\frac{1}{2}$ (all 7 folds)", fontsize=10)
    axR.grid(alpha=0.3, axis="y"); axR.legend(fontsize=8, loc="best")

    fig.suptitle(r"Saddle-node pair separation $d\propto|s-s_c|^{\gamma}$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fold_separation_exponent.pdf")


def main():
    spec = pd.read_csv(os.path.join(RESULTS, "branch_spectra.csv"))
    bif = pd.read_csv(os.path.join(RESULTS, "bifurcation_points.csv"))
    snap = pd.read_csv(os.path.join(RESULTS, "snapshot_equilibria.csv"))
    sw = pd.read_csv(os.path.join(RESULTS, "fold_local_sweeps.csv"))
    fig_sigma(spec, bif)
    fig_frequencies(spec, bif)
    fig_ratio(spec)
    fig_snapshots(snap)
    fig_fold_zoom(sw)
    fig_separation(sw)


if __name__ == "__main__":
    main()
