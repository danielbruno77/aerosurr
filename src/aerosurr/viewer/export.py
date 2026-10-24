"""Export decimated cars with per-vertex fields for the three.js viewer (SPEC §6).

Each car is decimated with `vtkQuadricClustering` (vertex clustering on a regular grid;
fast and robust on 9 M-cell surfaces). Every decimated vertex takes the values of its
nearest native cell, so the fields shown are the native data, not re-interpolated.

Binary layout of `<run>.bin` (little endian):

    u32  header length H, then H bytes of UTF-8 JSON (counts, field names, ranges)
    f32  positions (n_vertices, 3)
    u32  triangle indices (n_triangles, 3)
    u16  one array per field (n_vertices,), value = min + q / 65535 * (max - min)

`<run>.b64.txt` holds the same bytes base64-encoded: claude.ai artifacts serve text but
not raw binary, so the published viewer fetches the text form.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkQuadricClustering


def decimate(
    points: NDArray[Any], offsets: NDArray[Any], connectivity: NDArray[Any], cell_size: float
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """Vertex-clustered triangle mesh of a polygonal surface with ~`cell_size` spacing."""
    poly = vtkPolyData()
    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    poly.SetPoints(pts)
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(offsets, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(connectivity, dtype=np.int64), deep=True),
    )
    poly.SetPolys(cells)
    lo, hi = points.min(axis=0), points.max(axis=0)
    div = np.maximum(1, np.ceil((hi - lo) / cell_size)).astype(int)
    qc = vtkQuadricClustering()
    qc.SetInputData(poly)
    qc.SetNumberOfDivisions(int(div[0]), int(div[1]), int(div[2]))
    qc.UseInputPointsOn()  # output vertices are native points: no drift off the surface
    qc.Update()
    out = qc.GetOutput()
    verts = np.asarray(vtk_to_numpy(out.GetPoints().GetData()), dtype=np.float32)
    polys = out.GetPolys()
    conn = np.asarray(vtk_to_numpy(polys.GetConnectivityArray()))
    offs = np.asarray(vtk_to_numpy(polys.GetOffsetsArray()))
    if not np.all(np.diff(offs) == 3):
        raise ValueError("quadric clustering produced non-triangles")
    return verts, conn.reshape(-1, 3).astype(np.uint32)


def nearest_cell_values(
    cell_centres: NDArray[Any], fields: dict[str, NDArray[Any]], vertices: NDArray[Any]
) -> dict[str, NDArray[np.float32]]:
    _, idx = cKDTree(np.asarray(cell_centres, np.float64)).query(
        np.asarray(vertices, np.float64), workers=4
    )
    return {k: np.asarray(v, np.float32)[idx] for k, v in fields.items()}


def write_bin(
    path: Path,
    vertices: NDArray[np.float32],
    triangles: NDArray[np.uint32],
    fields: dict[str, NDArray[np.float32]],
    meta: dict[str, Any],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Write the viewer binary; returns its header. `ranges` fixes the quantisation range."""
    ranges = ranges or {}
    header: dict[str, Any] = {
        **meta,
        "n_vertices": len(vertices),
        "n_triangles": len(triangles),
        "fields": [],
    }
    payload = [np.ascontiguousarray(vertices, "<f4").tobytes(),
               np.ascontiguousarray(triangles, "<u4").tobytes()]  # fmt: skip
    for name, v in fields.items():
        lo, hi = ranges.get(name, (float(np.nanmin(v)), float(np.nanmax(v))))
        scale = (hi - lo) or 1.0
        q = np.clip(np.round((v - lo) / scale * 65535), 0, 65535).astype("<u2")
        header["fields"].append({"name": name, "min": lo, "max": hi})
        payload.append(q.tobytes())
    h = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", len(h)))
        f.write(h)
        pad = (4 - (4 + len(h)) % 4) % 4  # keep the float32 block 4-byte aligned
        f.write(b" " * pad)
        for block in payload:
            f.write(block)
    import base64

    path.with_suffix(".b64.txt").write_bytes(base64.b64encode(path.read_bytes()))
    header["header_bytes"] = 4 + len(h) + pad
    return header


def read_bin(
    path: Path,
) -> tuple[dict[str, Any], NDArray[np.float32], NDArray[np.uint32], dict[str, NDArray[np.float32]]]:
    """Inverse of `write_bin` (for tests and checks)."""
    raw = path.read_bytes()
    (hl,) = struct.unpack_from("<I", raw, 0)
    header = json.loads(raw[4 : 4 + hl])
    o = 4 + hl + (4 - (4 + hl) % 4) % 4
    nv, nt = header["n_vertices"], header["n_triangles"]
    verts = np.frombuffer(raw, "<f4", nv * 3, o).reshape(nv, 3)
    o += nv * 12
    tris = np.frombuffer(raw, "<u4", nt * 3, o).reshape(nt, 3)
    o += nt * 12
    fields = {}
    for f in header["fields"]:
        q = np.frombuffer(raw, "<u2", nv, o).astype(np.float32)
        fields[f["name"]] = f["min"] + q / 65535 * (f["max"] - f["min"])
        o += nv * 2
    return header, verts, tris, fields
