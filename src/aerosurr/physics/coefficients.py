"""Pressure and skin-friction coefficients (SPEC §5.1).

DrivAerML stores kinematic quantities (p/rho, tau/rho), so q_inf here is kinematic too:
q_inf = 0.5 * U_inf**2 (`Freestream.q_inf`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]


def pressure_coefficient(p: ArrayLike, q_inf: float, p_inf: float = 0.0) -> F64:
    """Cp = (p - p_inf) / q_inf."""
    if q_inf <= 0:
        raise ValueError("q_inf must be positive")
    return (np.asarray(p, dtype=np.float64) - p_inf) / q_inf


def skin_friction_coefficient(tau: ArrayLike, q_inf: float) -> F64:
    """Cf = tau / q_inf, componentwise for a (n, 3) wall-shear-stress field."""
    if q_inf <= 0:
        raise ValueError("q_inf must be positive")
    return np.asarray(tau, dtype=np.float64) / q_inf
