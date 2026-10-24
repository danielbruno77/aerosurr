from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerosurr.data.geometry import (
    polygon_arity_histogram,
    polygon_centres_and_area_vectors,
    vtk_cell_type_histogram,
)
from aerosurr.data.inspect import default_areas_path, inspect_vtp, render_markdown
from aerosurr.data.vtp import read_vtp, scan_xml_tags

from .conftest import Q_INF


def test_read_vtp(cube_vtp: Path) -> None:
    s = read_vtp(cube_vtp)
    assert (s.n_points, s.n_cells) == (8, 7)
    assert set(s.cell_data) == {
        "CpMeanTrim",
        "pMeanTrim",
        "pPrime2MeanTrim",
        "wallShearStressMeanTrim",
    }
    assert s.cell_data["wallShearStressMeanTrim"].shape == (7, 3)
    assert s.point_data == {}
    assert s.field_data["TimeValue"][0] == pytest.approx(3.6)


def test_geometry_of_cube(cube_vtp: Path) -> None:
    s = read_vtp(cube_vtp)
    c, av = polygon_centres_and_area_vectors(s.points, s.offsets, s.connectivity)
    np.testing.assert_allclose(np.linalg.norm(av, axis=1), [1, 1, 1, 1, 1, 0.5, 0.5])
    np.testing.assert_allclose(av.sum(axis=0), 0, atol=1e-12)  # closed surface
    np.testing.assert_allclose(c[0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(av[1] / np.linalg.norm(av[1]), [0, 0, 1])  # outward
    assert np.einsum("ij,ij->", c, av) / 3 == pytest.approx(1.0)  # divergence theorem


def test_area_vector_is_translation_invariant() -> None:
    pts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0]], dtype=np.float64)
    off, conn = np.array([0, 5]), np.arange(5)
    _, a0 = polygon_centres_and_area_vectors(pts, off, conn)
    _, a1 = polygon_centres_and_area_vectors(pts + 1e3, off, conn)
    np.testing.assert_allclose(a0, [[0, 0, 3.0]])  # shoelace
    np.testing.assert_allclose(a1, a0, rtol=1e-12)


def test_degenerate_polygon_rejected() -> None:
    with pytest.raises(ValueError, match="fewer than 3"):
        polygon_centres_and_area_vectors(np.zeros((2, 3)), np.array([0, 2]), np.array([0, 1]))


def test_cell_type_histograms() -> None:
    arity = polygon_arity_histogram(np.array([0, 3, 7, 11, 16, 22]))
    assert arity == {3: 1, 4: 2, 5: 1, 6: 1}
    assert vtk_cell_type_histogram(arity) == {5: 1, 7: 2, 9: 2}


def test_inspect_report(cube_vtp: Path) -> None:
    rep = inspect_vtp(cube_vtp, default_areas_path(cube_vtp), ("pMeanTrim", "CpMeanTrim"))
    assert rep.passed
    assert rep.areas is not None
    assert rep.areas.recomputed_max_rel_err < 1e-6
    assert rep.areas.signed_volume == pytest.approx(1.0)
    assert rep.pressure_fit is not None
    assert rep.pressure_fit.q_inf == pytest.approx(Q_INF, rel=1e-6)
    assert rep.pressure_fit.p_inf == pytest.approx(0.0, abs=1e-3)
    assert rep.pressure_fit.u_inf_if_kinematic == pytest.approx(38.889, rel=1e-6)
    names = {(a.association, a.name): a for a in rep.arrays}
    assert names["cell", "wallShearStressMeanTrim"].shape == (7, 3)
    assert names["cell", "wallShearStressMeanTrim"].magnitude is not None
    assert names["field", "TimeValue"].dtype == "<f4"
    md = render_markdown(rep)
    for token in ("CpMeanTrim", "**PASS**", "Int32", "outward"):
        assert token in md


def test_inspect_area_mismatch(cube_vtp: Path) -> None:
    bad = cube_vtp.with_name("boundary_cell_area_7.npy")
    np.save(bad, np.ones(6, dtype="<f4"))
    rep = inspect_vtp(cube_vtp, bad)
    assert not rep.passed
    assert "**FAIL**" in render_markdown(rep)


def test_xml_scan(cube_vtp: Path) -> None:
    tags = scan_xml_tags(cube_vtp)
    by_name = {t.attrs.get("Name"): t for t in tags if t.tag == "DataArray"}
    assert by_name["pMeanTrim"].section == "CellData"
    assert by_name["Points"].section == "Points"
    assert by_name["TimeValue"].section == "FieldData"
    assert tags[0].tag == "VTKFile"


def test_default_areas_path() -> None:
    assert default_areas_path(Path("run_12/boundary_12.vtp")) == Path(
        "run_12/boundary_cell_area_12.npy"
    )
    with pytest.raises(ValueError):
        default_areas_path(Path("foo.vtp"))
