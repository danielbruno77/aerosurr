"""Shared fixtures: a tiny synthetic DrivAerML-like VTP, and real-data / GPU gating."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
Q_INF = 756.17716  # used by the synthetic file only; mirrors 0.5 * 38.889^2

# Unit cube [0,1]^3, faces wound so Newell normals point outward.
CUBE_POINTS = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=np.float32,
)
CUBE_FACES = [
    [0, 3, 2, 1],  # z = 0, normal -z
    [4, 5, 6, 7],  # z = 1, +z
    [0, 1, 5, 4],  # y = 0, -y
    [3, 7, 6, 2],  # y = 1, +y
    [0, 4, 7, 3],  # x = 0, -x
    [1, 2, 6],  # x = 1, +x, split into two triangles
    [1, 6, 5],
]


def write_vtp(
    path: Path,
    points: np.ndarray,
    faces: list[list[int]],
    cell_arrays: dict[str, np.ndarray],
    field_arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

    pts = vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points), deep=True))
    cells = vtkCellArray()
    for f in faces:
        cells.InsertNextCell(len(f), f)
    poly = vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)
    for name, a in cell_arrays.items():
        arr = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
        arr.SetName(name)
        poly.GetCellData().AddArray(arr)
    for name, a in (field_arrays or {}).items():
        arr = numpy_to_vtk(np.ascontiguousarray(a), deep=True)
        arr.SetName(name)
        poly.GetFieldData().AddArray(arr)
    w = vtkXMLPolyDataWriter()
    w.SetFileName(str(path))
    w.SetInputData(poly)
    w.SetDataModeToBinary()  # inline base64, like DrivAerML
    w.SetCompressorTypeToNone()
    assert w.Write() == 1
    return path


@pytest.fixture
def cube_vtp(tmp_path: Path) -> Path:
    """`run_7/boundary_7.vtp` (a unit cube with DrivAerML array names) + its area file."""
    rng = np.random.default_rng(0)
    n = len(CUBE_FACES)
    cp = rng.uniform(-1, 1, n).astype(np.float32)
    run = tmp_path / "run_7"
    run.mkdir()
    path = write_vtp(
        run / "boundary_7.vtp",
        CUBE_POINTS,
        CUBE_FACES,
        {
            "CpMeanTrim": cp,
            "pMeanTrim": (Q_INF * cp.astype(np.float64)).astype(np.float32),
            "pPrime2MeanTrim": rng.uniform(1, 2, n).astype(np.float32),
            "wallShearStressMeanTrim": rng.normal(size=(n, 3)).astype(np.float32),
        },
        {"TimeValue": np.array([3.6], dtype=np.float32)},
    )
    areas = np.array([1, 1, 1, 1, 1, 0.5, 0.5], dtype="<f4")
    np.save(run / "boundary_cell_area_7.npy", areas)
    return path


@pytest.fixture
def data_root() -> Path:
    """Real DrivAerML directory from AEROSURR_DATA; skips the test if absent."""
    env = os.environ.get("AEROSURR_DATA")
    if not env or not (Path(env) / "run_1" / "boundary_1.vtp").is_file():
        pytest.skip("set AEROSURR_DATA to a DrivAerML dir containing run_1/boundary_1.vtp")
    return Path(env)


@pytest.fixture
def config_dir() -> Path:
    return REPO / "configs"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="needs CUDA")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
