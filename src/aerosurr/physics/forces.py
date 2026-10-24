"""Integrated force and moment coefficients from surface fields (SPEC §5.1).

Conventions, established against DrivAerML's `force_mom_i.csv` by tests/test_physics.py
(they are not assumptions):

- `area_vectors` are the Newell polygon area vectors in native connectivity order. On
  DrivAerML they point *into* the car, OpenFOAM's out-of-the-fluid convention. The
  pressure force on the car is therefore +sum(Cp * a) * q.
- `wallShearStressMeanTrim` is the stress the wall exerts on the fluid (OpenFOAM's
  `wallShearStress`), so the friction force on the car is -sum(Cf * |a|) * q.
- Axes: x streamwise (drag), y lateral (side force, Cs = +Fy / (q A)), z up (lift).
  Moments are about `forcesCoR`, and Cm = M / (q A l).
- Axle lift: Clf = Cl/2 + Cm_pitch and Clr = Cl/2 - Cm_pitch, with Cm_pitch = Cm_y.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

F64 = NDArray[np.float64]

#: Sign of the stored shear stress in the force on the body (see module docstring).
SHEAR_SIGN = -1.0


@dataclass(frozen=True)
class Reference:
    a_ref: float  # m^2
    l_ref: float  # m
    cor: tuple[float, float, float]  # moment centre, m


@dataclass(frozen=True)
class ForceCoefficients:
    cd: float
    cl: float
    cs: float
    clf: float
    clr: float
    cm_roll: float
    cm_pitch: float
    cm_yaw: float
    cd_pressure: float
    cd_friction: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def cell_forces(cp: ArrayLike, cf: ArrayLike, area_vectors: ArrayLike) -> F64:
    """Force on the body per cell, divided by q_inf: (n, 3), units of m^2."""
    a = np.asarray(area_vectors, dtype=np.float64)
    cp64 = np.asarray(cp, dtype=np.float64).reshape(-1)
    cf64 = np.asarray(cf, dtype=np.float64).reshape(-1, 3)
    if not (a.shape == cf64.shape and a.shape[0] == cp64.shape[0]):
        raise ValueError(f"shape mismatch: cp {cp64.shape}, cf {cf64.shape}, a {a.shape}")
    mag = np.linalg.norm(a, axis=1)
    out: F64 = cp64[:, None] * a + SHEAR_SIGN * cf64 * mag[:, None]
    return out


def integrate(
    cp: ArrayLike,
    cf: ArrayLike,
    area_vectors: ArrayLike,
    centres: ArrayLike,
    ref: Reference,
) -> ForceCoefficients:
    """Cd, Cl, Cs, axle lifts and moment coefficients of one surface."""
    a = np.asarray(area_vectors, dtype=np.float64)
    df = cell_forces(cp, cf, a)
    f = df.sum(axis=0) / ref.a_ref
    r = np.asarray(centres, dtype=np.float64) - np.asarray(ref.cor, dtype=np.float64)
    cm = np.cross(r, df).sum(axis=0) / (ref.a_ref * ref.l_ref)
    cp64 = np.asarray(cp, dtype=np.float64).reshape(-1)
    cd_p = float((cp64 * a[:, 0]).sum() / ref.a_ref)
    cl = float(f[2])
    return ForceCoefficients(
        cd=float(f[0]),
        cl=cl,
        cs=float(f[1]),
        clf=cl / 2 + float(cm[1]),
        clr=cl / 2 - float(cm[1]),
        cm_roll=float(cm[0]),
        cm_pitch=float(cm[1]),
        cm_yaw=float(cm[2]),
        cd_pressure=cd_p,
        cd_friction=float(f[0]) - cd_p,
    )
