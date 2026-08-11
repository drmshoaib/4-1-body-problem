"""Four-primary coordinate system and all inter-body distances.

Normalized frame fixed as in the central-configuration construction and
used unchanged by the dynamics:

    r0 = (0,  b)      # mass m0, on the +/-y axis
    r1 = (-1, 0)      # mass m1, on the -x axis at unit distance
    r2 = (0,  a)      # mass m2, on the y axis
    r3 = (c1, c2)     # mass m3, the free primary

with a > 0, b < a, b != +/- a.

NOTE: some sources print ``r1 = (1, 0)``, which contradicts the distance
convention ``r41 = sqrt((x+1)^2 + y^2)`` (that places m1 at x = -1). We use the
operative, dynamically-used convention ``r1 = (-1, 0)``.
"""

from __future__ import annotations

import numpy as np

# Order of the primaries everywhere in this package: (m0, m1, m2, m3).
PRIMARY_LABELS = ("m0", "m1", "m2", "m3")


def primary_positions(a: float, b: float, c1: float, c2: float) -> np.ndarray:
    """Return the 4x2 array of primary positions [m0, m1, m2, m3]."""
    return np.array(
        [
            [0.0, b],    # m0
            [-1.0, 0.0], # m1
            [0.0, a],    # m2
            [c1, c2],    # m3
        ],
        dtype=float,
    )


def distances_to_primaries(x, y, a: float, b: float, c1: float, c2: float) -> np.ndarray:
    """Distances r40, r41, r42, r43 from the fifth body (x, y) to each primary.

    Distances from the fifth body to the four primaries:
        r40 = sqrt(x^2 + (y-b)^2)
        r41 = sqrt((x+1)^2 + y^2)
        r42 = sqrt(x^2 + (y-a)^2)
        r43 = sqrt((x-c1)^2 + (y-c2)^2)
    """
    P = primary_positions(a, b, c1, c2)
    dx = x - P[:, 0]
    dy = y - P[:, 1]
    return np.sqrt(dx * dx + dy * dy)


def inter_primary_distances(a: float, b: float, c1: float, c2: float) -> dict:
    """The six inter-primary distances r_ij = |r_i - r_j|.

    These are the physical lengths whose cubes are the auxiliary quantities
    used in the CC equations, i.e. (S_ij = r_ij^{-3}):

        zeta  = r01^3 = (1 + b^2)^{3/2}
        tau   = r02^3 = (a - b)^3
        sigma = r12^3 = (1 + a^2)^{3/2}
        T     = r03^3 = (c1^2 + (c2 - b)^2)^{3/2}
        M     = r13^3 = ((c1 + 1)^2 + c2^2)^{3/2}
        G     = r23^3 = (c1^2 + (c2 - a)^2)^{3/2}

    Providing them explicitly lets a unit test cross-check the closed-form
    sigma/zeta/tau/G/T/M against sqrt-of-sum-of-squares geometry.
    """
    P = primary_positions(a, b, c1, c2)

    def d(i, j):
        return float(np.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1]))

    return {
        "r01": d(0, 1),
        "r02": d(0, 2),
        "r03": d(0, 3),
        "r12": d(1, 2),
        "r13": d(1, 3),
        "r23": d(2, 3),
    }
