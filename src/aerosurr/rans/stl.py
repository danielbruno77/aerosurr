"""STL I/O for RANS meshing: read DrivAerML's multi-solid ASCII STL, write one binary solid."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_VERTEX = re.compile(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)")
_SOLID = re.compile(rb"^\s*solid\s+(.*?)\s*$", re.MULTILINE)
_FACET = re.compile(rb"^\s*facet\b", re.MULTILINE)


@dataclass(frozen=True)
class Stl:
    triangles: NDArray[np.float32]  # (n, 3, 3)
    solids: list[tuple[str, int]]  # (name, number of triangles), in file order

    @property
    def bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        v = self.triangles.reshape(-1, 3)
        return v.min(axis=0), v.max(axis=0)


def _is_binary(raw: bytes) -> bool:
    if len(raw) < 84:
        return False
    (n,) = struct.unpack_from("<I", raw, 80)
    return 84 + 50 * n == len(raw)


def read_stl(path: Path) -> Stl:
    raw = path.read_bytes()
    if _is_binary(raw):
        (n,) = struct.unpack_from("<I", raw, 80)
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        tris = np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"].astype(np.float32)
        return Stl(tris, [(raw[:80].rstrip(b"\0 ").decode(errors="replace"), n)])
    verts = np.array(_VERTEX.findall(raw), dtype=np.float32)
    if verts.size == 0 or verts.shape[0] % 3:
        raise ValueError(f"{path}: not a triangle STL")
    starts = [m.start() for m in _SOLID.finditer(raw)]
    names = [m.group(1).decode(errors="replace") for m in _SOLID.finditer(raw)]
    facets = np.array([m.start() for m in _FACET.finditer(raw)])
    counts = np.diff(np.searchsorted(facets, [*starts, len(raw)]))
    return Stl(verts.reshape(-1, 3, 3), list(zip(names, (int(c) for c in counts), strict=True)))


def write_binary_stl(path: Path, triangles: NDArray[np.floating], name: str = "car") -> None:
    """One binary solid; facet normals from the vertex winding."""
    t = np.asarray(triangles, dtype=np.float32)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]).astype(np.float64)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, t
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        # A binary header must not start with "solid": readers (OpenFOAM too) take it as ASCII.
        f.write(f"binary STL {name} (aerosurr)".encode().ljust(80, b" "))
        f.write(struct.pack("<I", len(t)))
        f.write(rec.tobytes())
