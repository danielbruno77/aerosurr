"""Map low-fidelity RANS surface fields onto the HF native cells (SPEC §5.3).

The RANS car surface (`postProcessing/carSurface/<it>/car.vtp`) carries face data
`pMean` and `wallShearStressMean` (averaged like the forces); both are kinematic and use
the same OpenFOAM conventions as DrivAerML, so Cp = p / q and Cf = tau / q directly.
Values reach HF cell centres by inverse-distance weighting of the k nearest LF faces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aerosurr.data.geometry import polygon_centres_and_area_vectors
from aerosurr.data.vtp import read_vtp

LF_DIR = "lf"  # <store>/lf/<run>.npz


def read_lf_surface(
    vtp: Path, q_inf: float, p_inf: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Face centres (m, 3) and LF fields (m, 4) = (Cp, Cf_x, Cf_y, Cf_z)."""
    s = read_vtp(vtp)
    centres, _ = polygon_centres_and_area_vectors(s.points, s.offsets, s.connectivity)
    cd = s.cell_data
    p = cd["pMean"] if "pMean" in cd else cd["p"]
    tau = cd["wallShearStressMean"] if "wallShearStressMean" in cd else cd["wallShearStress"]
    fields = np.column_stack([(np.asarray(p, np.float64).ravel() - p_inf) / q_inf,
                              np.asarray(tau, np.float64).reshape(-1, 3) / q_inf])  # fmt: skip
    return centres, fields


def idw(
    src: NDArray[Any], values: NDArray[Any], dst: NDArray[Any], k: int = 4, workers: int = 4
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Inverse-distance-weighted values at `dst` from the k nearest `src` points.

    Returns (values (n, f), distance to the nearest source point (n,)).
    """
    tree = cKDTree(np.asarray(src, np.float64))
    k = min(k, len(src))
    d, i = tree.query(np.asarray(dst, np.float64), k=k, workers=workers)
    if k == 1:
        d, i = d[:, None], i[:, None]
    w = 1.0 / np.maximum(d, 1e-9) ** 2
    w /= w.sum(axis=1, keepdims=True)
    out = np.einsum("nk,nkf->nf", w, np.asarray(values, np.float64)[i])
    return out.astype(np.float32), d[:, 0].astype(np.float32)


def map_run(
    lf_vtp: Path, store: Path, run: str, q_inf: float, p_inf: float = 0.0, k: int = 4
) -> dict[str, float]:
    """Write `<store>/lf/<run>.npz` with `pool_lf` (and `full_lf`) = LF fields at HF cells."""
    from aerosurr.data.store import load_group

    src, vals = read_lf_surface(lf_vtp, q_inf, p_inf)
    out: dict[str, NDArray[Any]] = {}
    stats: dict[str, float] = {"lf_faces": float(len(src))}
    for group in ("pool", "full"):
        try:
            g = load_group(store, run, group)
        except KeyError:
            continue
        mapped, dist = idw(src, vals, g["centres"], k)
        out[f"{group}_lf"] = mapped.astype(np.float16)
        out[f"{group}_lf_dist"] = dist
        stats[f"{group}_dist_p99_mm"] = float(np.percentile(dist, 99) * 1e3)
    (store / LF_DIR).mkdir(parents=True, exist_ok=True)
    tmp = store / LF_DIR / f".{run}.tmp.npz"
    np.savez(tmp, **out)  # type: ignore[arg-type]
    tmp.replace(store / LF_DIR / f"{run}.npz")
    return stats
