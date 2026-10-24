"""Read DrivAerML boundary `.vtp` files into numpy arrays (VTK XML PolyData)."""

from __future__ import annotations

import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkDataArray
from vtkmodules.vtkCommonDataModel import vtkFieldData, vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader


@dataclass(frozen=True)
class Surface:
    """A polygonal surface mesh with its attached arrays, as stored on disk."""

    points: NDArray[np.floating]  # (n_points, 3)
    offsets: NDArray[np.integer]  # (n_cells + 1,), polygon i uses connectivity[o[i]:o[i+1]]
    connectivity: NDArray[np.integer]  # (n_edges,)
    point_data: dict[str, NDArray[np.generic]] = field(default_factory=dict)
    cell_data: dict[str, NDArray[np.generic]] = field(default_factory=dict)
    field_data: dict[str, NDArray[np.generic]] = field(default_factory=dict)
    n_verts: int = 0
    n_lines: int = 0
    n_strips: int = 0

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.offsets.shape[0] - 1)


def _np(arr: Any) -> NDArray[Any]:
    return np.asarray(vtk_to_numpy(arr))  # type: ignore[no-untyped-call]


def _arrays(fd: vtkFieldData) -> dict[str, NDArray[np.generic]]:
    out: dict[str, NDArray[np.generic]] = {}
    for i in range(fd.GetNumberOfArrays()):
        arr = fd.GetAbstractArray(i)
        if not isinstance(arr, vtkDataArray):  # string arrays etc.: not used by DrivAerML
            continue
        out[arr.GetName()] = _np(arr)
    return out


def read_vtp(path: Path) -> Surface:
    """Read a `.vtp` file. Raises if it is missing or VTK reports a read error."""
    if not path.is_file():
        raise FileNotFoundError(path)
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    if reader.GetErrorCode() != 0:
        raise OSError(f"VTK failed to read {path} (error code {reader.GetErrorCode()})")
    poly: vtkPolyData = reader.GetOutput()
    polys = poly.GetPolys()
    return Surface(
        points=_np(poly.GetPoints().GetData()),
        offsets=_np(polys.GetOffsetsArray()),
        connectivity=_np(polys.GetConnectivityArray()),
        point_data=_arrays(poly.GetPointData()),
        cell_data=_arrays(poly.GetCellData()),
        field_data=_arrays(poly.GetFieldData()),
        n_verts=int(poly.GetNumberOfVerts()),
        n_lines=int(poly.GetNumberOfLines()),
        n_strips=int(poly.GetNumberOfStrips()),
    )


_ATTR_RE = re.compile(rb"""(\w+)=['"]([^'"]*)['"]""")


@dataclass(frozen=True)
class XmlTag:
    tag: str
    attrs: dict[str, str]
    section: str  # enclosing PointData/CellData/FieldData/Points/Polys, or "" at top level


def scan_xml_tags(path: Path) -> list[XmlTag]:
    """Return the VTK XML structural tags in file order, as written on disk.

    Inline base64 payloads are skipped by the regex, so this reads only what the file
    declares (array types, names, encodings), independent of how VTK interprets it.
    """
    tags: list[XmlTag] = []
    section = ""
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for m in re.finditer(
            rb"<(/?)(VTKFile|Piece|DataArray|PointData|CellData|FieldData|Points|Polys)\b([^>]*)>",
            mm,
        ):
            closing, name, body = m.group(1), m.group(2).decode(), m.group(3)
            if name in {"PointData", "CellData", "FieldData", "Points", "Polys"}:
                section = "" if closing else name
                continue
            if closing:
                continue
            attrs = {k.decode(): v.decode() for k, v in _ATTR_RE.findall(body)}
            tags.append(XmlTag(tag=name, attrs=attrs, section=section))
    return tags
