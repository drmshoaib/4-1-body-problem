"""r5bp_nonsymmetric
=================

Mathematical model of the *planar restricted five-body problem with four
unequal primaries*: four primaries in a coplanar central configuration and an
infinitesimal fifth body moving in the associated uniformly rotating field.

The package implements, from first principles:

* four-primary normalized coordinate system;
* inter-primary distances and the auxiliary lengths sigma, zeta, tau, G, T, M;
* mass-ratio formulae rho0, rho1, rho2;
* central-configuration constraint psi;
* positivity / admissibility checks;
* rotating-frame effective potential Omega;
* analytic gradient Omega_x, Omega_y;
* analytic Hessian Omega_xx, Omega_xy, Omega_yy;
* 4x4 rotating-frame linearization matrix A;
* the Jacobi integral.

No continuation / bifurcation logic lives in this package; that is handled by
the continuation module and the driver scripts.
"""

from . import geometry, central_config, potential, equilibria, cases  # noqa: F401

__all__ = ["geometry", "central_config", "potential", "equilibria", "cases"]
