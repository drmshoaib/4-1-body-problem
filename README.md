# Restricted Five-Body Problem with Four Unequal Primaries

Equilibrium bifurcations and periodic dynamics of an infinitesimal body moving
in the field of four unequal primaries that form a non-symmetric coplanar
central configuration.

This repository contains the LaTeX manuscript and the complete computation code
behind every numerical result, table, and figure in the paper:

> **Equilibrium Bifurcations and Periodic Dynamics in a Planar Restricted
> Five-Body Problem with Four Unequal Primaries.**
> Muhammad Shoaib and A. R. Kashif.

The compiled manuscript is included as [`paper/main.pdf`](paper/main.pdf). This
repository, together with an archived DOI-tagged release, is the reproducibility
archive for the paper: it holds the source code, the verified result CSVs, the
driver scripts, the tests, and all manuscript figures and tables. There is no
separate supplementary PDF.

---

## The model

Four primaries with masses `m0, m1, m2, m3` are placed in a normalized rotating
frame so that they form a relative equilibrium (a central configuration). An
infinitesimal fifth body moves in the resulting uniformly rotating gravitational
field. Its motion is governed by the effective potential

```
Omega(x, y) = (x^2 + y^2)/2 + sum_i m_i / r_i,
```

where `r_i` is the distance from the test body to primary `i`. Equilibria of the
restricted problem are the critical points `Omega_x = Omega_y = 0`; their linear
character follows from the 4x4 rotating-frame variational matrix, and the Jacobi
integral

```
C_J = 2 Omega - (xdot^2 + ydot^2)
```

is conserved along trajectories and defines the accessible Hill regions
`{ 2 Omega >= C_J }`.

The admissible primary geometries are not arbitrary: the four masses must
satisfy a scalar central-configuration (CC) constraint `psi(a, b, c1, c2) = 0`
together with positivity of the mass ratios. The study fixes the slice
`(a, b) = (0.9, -0.1)` and follows the restricted dynamics as the remaining
geometry is deformed along the admissible CC set.

## Key results

- On the slice `(a, b) = (0.9, -0.1)` the admissible CC set `psi = 0` separates
  numerically into **two disjoint connected components** (a Case-1 arc and a
  Case-2 arc), so configurations with different equilibrium counts need not lie
  on one continuous family.
- Equilibrium multiplicity changes through **seven generic saddle-node folds**,
  giving the interior cascades `5 -> 7 -> 5` on the Case-1 component and
  `7 -> 9 -> 11 -> 9 -> 7 -> 5` on the Case-2 component.
- The eleven-equilibrium regime contains a **branch exchange**: the pair created
  at the `9 -> 11` fold is not the pair destroyed at the following `11 -> 9`
  fold.
- Both components contain intervals of **spectrally stable equilibria**. From
  selected centre modes, **ten Lyapunov periodic-orbit families** (198 corrected
  orbits, closure residuals below `1e-10`) are continued and classified by their
  Floquet multipliers.
- Families near low-order centre-frequency commensurabilities approach the
  period-doubling threshold; the closest (family 10) reaches
  `|nu + 2| ~ 1e-7` but turns back without crossing.
- Critical Jacobi levels are used to demonstrate a resolution-robust change in
  **Hill-region connectivity** and to place periodic families within their
  accessible regions.

All connectivity and multiplicity statements are numerical and specific to the
investigated parameter slice.

## Repository layout

```
paper/                     LaTeX manuscript and compiled PDFs
  main.tex, sections/      the paper body (one file per section)
  tables/                  auto-generated LaTeX tables
  figures/                 all manuscript figures (PDF + PNG)
  references.bib           bibliography
  main.pdf                 compiled manuscript
  make_tables.py           regenerate tables/ from result CSVs
  make_figures.py          regenerate the configuration figure
src/r5bp_nonsymmetric/     the model package
  geometry.py              primary positions and inter-body distances
  central_config.py        CC constraint psi, mass ratios, geometry refinement
  potential.py             effective potential, gradient, Hessian, variational matrix, Jacobi integral
  equilibria.py            independent multi-start equilibrium search
  continuation.py          pseudo-arclength continuation of the CC arcs
  continuation_bif.py      fold detection and classification
  periodic.py              centre modes, single-shooting correction, Floquet analysis
  validation.py            finite-difference checks of the analytic derivatives
  cases.py                 reference configurations used for validation
scripts/                   drivers that produce the result CSVs and figures
results/                   verified result CSVs used to produce the paper
tests/                     unit tests (pytest)
conftest.py                puts src/ on the import path for the tests
```

The `results/` directory contains the **verified result CSVs used to produce the
paper**, so the values behind the tables and figures can be inspected directly
without rerunning the pipeline. The scripts below regenerate them
deterministically.

## Requirements

- **Python 3.14** or later.
- **NumPy**, **SciPy**, **pandas**, **Matplotlib**.

The results in the paper were produced with Python 3.14.5, NumPy 2.5.0,
SciPy 1.17.1, Matplotlib 3.10.9, and pandas 3.0.3.

Building the manuscript additionally requires a LaTeX distribution (for example
MiKTeX or TeX Live) providing `pdflatex` and `bibtex`.

## Setup

No installation step is required. The package uses a `src/` layout and is made
importable either by the test harness (`conftest.py`) or by each script, which
prepends `src/` to `sys.path`. From the repository root:

```bash
python -m pip install numpy scipy pandas matplotlib
```

## Running the tests

```bash
python -m pytest -q
```

This runs the unit tests that check the central-configuration formulae, the
analytic gradient and Hessian, the equilibrium search, and the Jacobi-integral
conventions against independent finite-difference and geometric references.

## Reproducing the results

Run the drivers from the repository root, in order. Each writes its CSV outputs
into `results/` (created automatically).

1. **Reference cases and equilibria**
   ```bash
   python scripts/run_reference_cases.py
   ```
   Refines each reference configuration to `|psi| < 1e-12`, recomputes the mass
   ratios, and finds the equilibria independently.
   Outputs: `refined_reference_cases.csv`, `refined_equilibria.csv`.

2. **Continuation and bifurcations**
   ```bash
   python scripts/run_continuation.py
   ```
   Traces the admissible Case-1 and Case-2 arcs of `psi = 0`, continues the
   equilibria with an independent global search at every sample, tracks branches,
   and locates and classifies the count-changing folds.

3. **Spectral and fold post-processing**
   ```bash
   python scripts/analyze_spectra_and_folds.py
   ```
   Re-stitches the branches with full spectra, ranks centre-mode candidates, and
   produces the fine fold sweeps and fitted scaling exponents.

4. **Periodic-orbit families**
   ```bash
   python scripts/run_periodic_orbits.py
   ```
   Computes, continues, and classifies differential-corrected periodic orbits
   (closure below `1e-10`) with Floquet stability from the monodromy matrix.

5. **Targeted refinements** (optional)
   ```bash
   python scripts/refine_period_doubling.py
   python scripts/search_secondary_families.py
   ```
   Resolve the near-period-doubling turn-back (`refine_period_doubling.py`) and
   test for a genuine subharmonic near the family whose nontrivial multiplier is
   closest to a low-order resonance (`search_secondary_families.py`). Both are
   negative checks: no computed family crosses the selected multiplier conditions
   (`nu = 2, 0, -1, -2`), and no genuine subharmonic is found.

### Regenerating figures and tables

With the result CSVs in place:

```bash
python scripts/make_continuation_figures.py
python scripts/make_analysis_figures.py
python scripts/make_periodic_figures.py
python scripts/make_refinement_figures.py
python scripts/make_hill_figures.py
python paper/make_tables.py
python paper/make_figures.py
```

These write the PDF and PNG figures into `paper/figures/` and the LaTeX tables
into `paper/tables/`.

## Building the paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Reproducibility notes

All numerical integrations use `scipy.integrate.solve_ivp` with the `DOP853`
method at `rtol = atol = 1e-12`. Random seeds in the multi-start searches are
fixed, so the pipeline is designed to be deterministically reproducible within a
pinned software environment.

## Reproducibility archive

This repository is the code/data reproducibility archive for the paper. At
submission the exact repository state is tagged as a release (for example
`v1.0-paper-submission`) and deposited in a permanent archive (e.g. Zenodo,
figshare, or ResearchGate) to mint a DOI. The paper's data-and-code-availability
statement then
cites the repository URL, the immutable commit/tag, and the archive DOI, so a
reader obtains the exact code and numerical outputs behind every result.

## Citation

If you use this code or the results, please cite the accompanying manuscript
(Shoaib and Kashif). Full bibliographic details will be added here once the
paper is published.

## License

No license has been specified yet. Until one is added, all rights are reserved;
please contact the authors before reusing the code or data.
