"""Four reference configurations used to validate the model implementation.

`reported_rho` and `reported_count` are the previously reported values for
each configuration, kept ONLY for comparison in the validation report; they
are never used as inputs to the independent recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReferenceCase:
    name: str
    a: float
    b: float
    c1: float
    c2: float
    reported_rho: tuple
    reported_count: int
    reported_equilibria: tuple = field(default=())


REFERENCE_CASES = [
    ReferenceCase(
        name="Case 1",
        a=0.9, b=-0.1, c1=-0.61, c2=0.71,
        reported_rho=(47.3288, 3.05273, 1.17191),
        reported_count=5,
        reported_equilibria=(
            (-0.5313, 0.6081), (-0.7944, -0.01641), (-0.002206, 0.7642),
            (2.68, 2.536), (-3.422, -1.679),
        ),
    ),
    ReferenceCase(
        name="Case 2",
        a=0.9, b=-0.1, c1=0.52, c2=-0.88,   # resolved sign convention
        reported_rho=(4.21137, 0.956716, 1.27145),
        reported_count=9,
        reported_equilibria=(
            (-1.348, 1.473), (-2.096, 0.03881), (-0.2338, 2.006),
            (-1.035, -1.647), (1.773, 0.5767), (0.961, -1.875),
            (0.3455, -0.6233), (-0.6786, -0.01292), (-0.01654, 0.5457),
        ),
    ),
    ReferenceCase(
        name="Case 3",
        a=1.5, b=0.4, c1=-0.96, c2=0.99999,  # resolved value
        reported_rho=(3.63932, 0.151366, 0.475099),
        reported_count=5,
        reported_equilibria=(
            (0.26, -1.3), (-0.85, 0.12), (-1.31, 1.87),
            (-0.601, 0.8147), (-0.05, 1.19),
        ),
    ),
    ReferenceCase(
        name="Case 4",
        a=1.5, b=0.4, c1=1.05, c2=0.13,      # resolved value
        reported_rho=(4.31185, 0.98422, 0.723114),
        reported_count=5,
        reported_equilibria=(
            (0.01602, 2.376), (-0.004887, 1.183), (0.7131, 0.2378),
            (-0.681, 0.1432), (0.05785, -1.595),
        ),
    ),
]
