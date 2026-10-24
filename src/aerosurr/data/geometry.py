"""Per-polygon geometry of a native boundary mesh (float64 throughout)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def polygon_centres_and_area_vectors(
    points: NDArray[np.floating],
    offsets: NDArray[np.integer],
    connectivity: NDArray[np.integer],
) -> tuple[F64, F64]:
    """Vertex-average centre and Newell area vector of every polygon.

    Matches the DrivAerML `boundary_cell_area` definition
    (`surface_cell_areas/algorithm.json`): with cyclic vertices p_i and centre c,
    area_vector = 0.5 * sum_i cross(p_i - c, p_{i+1} - c). The scalar area is its norm.
    """
    off = offsets.astype(np.int64)
    conn = connectivity.astype(np.int64)
    counts = np.diff(off)
    if counts.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    if counts.min() < 3:
        raise ValueError("polygon with fewer than 3 vertices")
    starts = off[:-1]
    p = points[conn].astype(np.float64)
    centres = np.add.reduceat(p, starts, axis=0) / counts[:, None]
    p -= np.repeat(centres, counts, axis=0)
    nxt = np.arange(1, conn.size + 1, dtype=np.int64)
    nxt[off[1:] - 1] = starts  # last vertex of each polygon wraps to its first
    area_vectors = 0.5 * np.add.reduceat(np.cross(p, p[nxt]), starts, axis=0)
    return centres, area_vectors


def polygon_arity_histogram(offsets: NDArray[np.integer]) -> dict[int, int]:
    counts = np.diff(offsets.astype(np.int64))
    values, n = np.unique(counts, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, n, strict=True)}


VTK_CELL_TYPE = {3: 5, 4: 9}  # triangle -> VTK_TRIANGLE, quad -> VTK_QUAD; others VTK_POLYGON (7)


def vtk_cell_type_histogram(arity: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, n in arity.items():
        t = VTK_CELL_TYPE.get(k, 7)
        out[t] = out.get(t, 0) + n
    return dict(sorted(out.items()))
