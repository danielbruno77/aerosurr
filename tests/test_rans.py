"""RANS case generation and result parsing (no OpenFOAM needed)."""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np
import pytest

from aerosurr.config import load_data, load_rans
from aerosurr.data.validate import read_geo_refs
from aerosurr.rans.case import CAR, block_grid, block_mesh_dict, write_case
from aerosurr.rans.run import (
    mpirun_cmd,
    parse_cells,
    parse_cpus,
    parse_s_per_iter,
    read_coefficients,
    read_force_mom,
    summarise_coefficients,
    write_rankfile,
)
from aerosurr.rans.stl import read_stl, write_binary_stl

GEO_REF = (
    "lRef,aRef,forcesCoR,lRefRef,aRefRef,forcesCoRRef\n"
    "2.972,2.223,(1.4929999999999999 0 -0.3176),2.78618,2.17,(1.40009 0 -0.3176)\n"
)


def ascii_stl(path: Path, solids: dict[str, np.ndarray]) -> Path:
    lines = []
    for name, tris in solids.items():
        lines.append(f"solid {name}")
        for t in tris:
            lines += ["  facet normal 0 0 1", "    outer loop"]
            lines += [f"      vertex {x:.6e} {y:.6e} {z:.6e}" for x, y, z in t]
            lines += ["    endloop", "  endfacet"]
        lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def box_stl(tmp_path: Path) -> Path:
    """A 2-solid 'car' (two triangles each) that dips below the ground like DrivAerML tyres."""
    a = np.array([[[0, -0.5, -0.32], [1, -0.5, -0.32], [1, 0.5, 0.5]],
                  [[0, -0.5, -0.32], [1, 0.5, 0.5], [0, 0.5, 0.5]]])  # fmt: skip
    return ascii_stl(
        tmp_path / "drivaer_9.stl", {"BodyA": a, "Tiresfront": a + np.array([2.0, 0.0, 0.0])}
    )


def test_read_ascii_stl_solids(box_stl: Path) -> None:
    s = read_stl(box_stl)
    assert s.triangles.shape == (4, 3, 3)
    assert s.solids == [("BodyA", 2), ("Tiresfront", 2)]
    lo, hi = s.bounds
    np.testing.assert_allclose(lo, [0, -0.5, -0.32], atol=1e-6)
    np.testing.assert_allclose(hi, [3, 0.5, 0.5], atol=1e-6)


def test_binary_stl_roundtrip_and_header(box_stl: Path, tmp_path: Path) -> None:
    s = read_stl(box_stl)
    out = tmp_path / "car.stl"
    write_binary_stl(out, s.triangles)
    raw = out.read_bytes()
    assert not raw.startswith(b"solid")  # OpenFOAM would parse it as ASCII
    assert struct.unpack_from("<I", raw, 80)[0] == 4 and len(raw) == 84 + 50 * 4
    back = read_stl(out)
    np.testing.assert_array_equal(back.triangles, s.triangles)


def test_geo_refs(tmp_path: Path) -> None:
    p = tmp_path / "geo_ref_1.csv"
    p.write_text(GEO_REF)
    g = read_geo_refs(p)
    assert (g.varying.l_ref, g.varying.a_ref) == (2.972, 2.223)
    assert g.varying.cor == pytest.approx((1.493, 0.0, -0.3176))  # stored as 1.4929999999999999
    assert (g.constant.l_ref, g.constant.a_ref, g.constant.cor) == (
        2.78618,
        2.17,
        (1.40009, 0.0, -0.3176),
    )


def test_block_grid_is_cubic_and_split_at_x_bl(config_dir: Path) -> None:
    cfg = load_rans("coarse", config_dir)
    for mesh in cfg.mesh_presets.values():
        g = block_grid(cfg, mesh)
        x0, xbl, x2 = g.x
        na, nb, ny, nz = g.n
        assert xbl == cfg.domain.x_bl and g.z[0] == cfg.domain.ground_z
        assert x0 <= cfg.domain.x_min and x2 >= cfg.domain.x_max
        for length, n in (
            (xbl - x0, na),
            (x2 - xbl, nb),
            (g.y[1] - g.y[0], ny),
            (g.z[1] - g.z[0], nz),
        ):
            assert length / n == pytest.approx(mesh.base_cell)


def test_block_mesh_dict_patches(config_dir: Path) -> None:
    cfg = load_rans("coarse", config_dir)
    text = block_mesh_dict(block_grid(cfg, cfg.mesh_presets["coarse"]))
    expected = {
        "inlet": "patch",
        "outlet": "patch",
        "sides": "symmetry",
        "top": "symmetry",
        "groundSlip": "symmetryPlane",
        "ground": "wall",
    }
    for name, kind in expected.items():
        assert re.search(rf"{name}\s*\{{\s*type {kind};", text), name
    assert text.count("hex (") == 2
    n_vertices = len(re.findall(r"^\s+\([-\d.e ]+\)$", text.split("blocks")[0], re.MULTILINE))
    assert n_vertices == 12


def test_write_case(box_stl: Path, tmp_path: Path, config_dir: Path) -> None:
    cfg = load_rans("coarse", config_dir)
    fs = load_data(config_dir).freestream
    ref_path = tmp_path / "geo_ref_9.csv"
    ref_path.write_text(GEO_REF)
    case = tmp_path / "case"
    write_case(case, cfg, "medium", box_stl, fs, read_geo_refs(ref_path).varying, n_ranks=24)
    for rel in ["system/blockMeshDict", "system/snappyHexMeshDict", "system/controlDict",
                "system/fvSchemes", "system/fvSolution", "system/decomposeParDict",
                "system/surfaceFeatureExtractDict", "constant/transportProperties",
                "constant/turbulenceProperties", f"constant/triSurface/{CAR}.stl",
                "0.orig/U", "0.orig/p", "0.orig/k", "0.orig/omega", "0.orig/nut"]:  # fmt: skip
        assert (case / rel).is_file(), rel
    assert read_stl(case / "constant/triSurface/car.stl").triangles.shape == (4, 3, 3)
    snappy = (case / "system/snappyHexMeshDict").read_text()
    assert "level (5 5);" in snappy and "close" in snappy and "addLayers       false" in snappy
    ctrl = (case / "system/controlDict").read_text()
    assert "magUInf         38.889;" in ctrl and "Aref            2.223;" in ctrl
    assert "CofR            (1.493 0 -0.3176);" in ctrl
    assert "numberOfSubdomains 24;" in (case / "system/decomposeParDict").read_text()
    u = (case / "0.orig/U").read_text()
    assert "uniform (38.889 0 0)" in u and re.search(rf"{CAR}\s*\{{\s*type noSlip;", u)
    omega = (case / "0.orig/omega").read_text()
    k_in = 1.5 * (cfg.solver.turbulence_intensity * fs.u_inf) ** 2
    assert f"uniform {k_in / (fs.nu * cfg.solver.viscosity_ratio):.6g}" in omega
    with pytest.raises(KeyError):
        write_case(tmp_path / "c2", cfg, "nope", box_stl, fs, read_geo_refs(ref_path).varying, 4)


def test_write_case_rejects_floating_car(tmp_path: Path, config_dir: Path) -> None:
    tri = np.array([[[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.2]]])
    stl = ascii_stl(tmp_path / "s.stl", {"x": tri})
    ref = tmp_path / "g.csv"
    ref.write_text(GEO_REF)
    with pytest.raises(ValueError, match="above the ground"):
        write_case(tmp_path / "c", load_rans("coarse", config_dir), "coarse", stl,
                   load_data(config_dir).freestream, read_geo_refs(ref).varying, 4)  # fmt: skip


def test_parse_cpus() -> None:
    assert parse_cpus("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    with pytest.raises(ValueError):
        parse_cpus("3-1")
    with pytest.raises(ValueError):
        parse_cpus("0-2,1")


def test_rankfile_and_mpirun(tmp_path: Path) -> None:
    rf = write_rankfile(tmp_path / "rankfile", [24, 25, 56])
    assert rf.read_text().splitlines() == [
        "rank 0=localhost slot=24",
        "rank 1=localhost slot=25",
        "rank 2=localhost slot=56",
    ]
    cmd = mpirun_cmd(["simpleFoam"], 3, rf)
    assert cmd[:3] == ["mpirun", "-np", "3"] and "--rankfile" in cmd and cmd[-1] == "-parallel"
    assert "--map-by" not in cmd and "--cpu-set" not in cmd  # OpenMPI 4.1 rejects the pair


def test_log_parsers(tmp_path: Path) -> None:
    assert parse_cells("Mesh stats\n    points:  10\n    cells:            1990594\n") == 1990594
    log = "".join(f"Time = {i}\nExecutionTime = {2.0 + 0.5 * i:.2f} s  ClockTime = {i} s\n"
                  for i in range(1, 11))  # fmt: skip
    n, dt = parse_s_per_iter(log)
    assert n == 10 and dt == pytest.approx(0.5)
    coef = tmp_path / "coefficient.dat"
    coef.write_text("# Force coefficients\n# Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) CmPitch Cs\n"
                    + "".join(f"{i} {0.3 + 0.01 * (i % 2)} 0 0 0.1 0.05 0.05 0.01 0.02\n"
                              for i in range(1, 101)))  # fmt: skip
    mean, std = summarise_coefficients(read_coefficients(coef), last=50)
    assert mean["Cd"] == pytest.approx(0.305) and std["Cd"] == pytest.approx(0.005)
    assert mean["Cl(f)"] == pytest.approx(0.05)
    fm = tmp_path / "force_mom_1.csv"
    fm.write_text(
        "Cd,Cl,Clf,Clr,Cs\n3.035117e-01,6.772802e-02,-3.728616e-02,1.050142e-01,4.766758e-02\n"
    )
    assert read_force_mom(fm)["Cd"] == pytest.approx(0.3035117)


def test_idw_and_lf_surface(tmp_path: Path) -> None:
    from aerosurr.rans.mapping import idw, read_lf_surface

    from .conftest import CUBE_FACES, CUBE_POINTS, write_vtp

    src = np.random.default_rng(0).normal(size=(50, 3))
    vals = np.column_stack([src[:, 0], src**2])
    out, dist = idw(src, vals, src, k=4)  # at the source points: exact
    np.testing.assert_allclose(out, vals, rtol=1e-5, atol=1e-6)
    assert dist.max() < 1e-6
    mid, _ = idw(np.array([[0.0, 0, 0], [1.0, 0, 0]]), np.array([[0.0], [2.0]]),
                 np.array([[0.5, 0, 0]]), k=2)  # fmt: skip
    assert mid[0, 0] == pytest.approx(1.0)

    n = len(CUBE_FACES)
    q = 756.17716
    vtp = write_vtp(tmp_path / "car.vtp", CUBE_POINTS, CUBE_FACES, {
        "p": np.zeros(n, np.float32), "pMean": np.full(n, q, np.float32),
        "wallShearStressMean": np.tile(np.array([q, 0, 0], np.float32), (n, 1)),
    })  # fmt: skip
    centres, fields = read_lf_surface(vtp, q)
    assert centres.shape == (n, 3) and fields.shape == (n, 4)
    np.testing.assert_allclose(fields, np.tile([1.0, 1.0, 0, 0], (n, 1)), rtol=1e-6)
