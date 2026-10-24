"""Checks against real DrivAerML files (`-m data`, needs AEROSURR_DATA). Run in CI's physics job."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerosurr.config import load_data
from aerosurr.data.inspect import default_areas_path, inspect_vtp
from aerosurr.data.splits import all_runs, load_manifest, priority_runs

pytestmark = pytest.mark.data


def test_run1_inspection_gate(data_root: Path, config_dir: Path) -> None:
    cfg = load_data(config_dir)
    f = cfg.fields
    vtp = data_root / "run_1" / "boundary_1.vtp"
    rep = inspect_vtp(vtp, default_areas_path(vtp), (f.pressure, f.pressure_coefficient),
                      compute_sha256=False)  # fmt: skip
    assert rep.passed
    assert rep.n_cells == 8_828_095 and rep.n_points == 8_846_052
    names = {a.name for a in rep.arrays if a.association == f.association}
    assert {
        f.pressure,
        f.pressure_coefficient,
        f.pressure_fluctuation,
        f.wall_shear_stress,
    } <= names
    assert not [a for a in rep.arrays if a.association == "point"]
    assert rep.areas is not None and rep.areas.recomputed_max_rel_err < 1e-6
    # Cp = (p - p_inf) / q_inf with the configured freestream, on every cell.
    fit = rep.pressure_fit
    assert fit is not None
    assert fit.q_inf == pytest.approx(cfg.freestream.q_inf, rel=1e-6)
    assert fit.p_inf == pytest.approx(cfg.freestream.p_inf, abs=1e-3)
    assert fit.u_inf_if_kinematic == pytest.approx(cfg.freestream.u_inf, rel=1e-6)


def test_area_file_matches_contract(data_root: Path) -> None:
    a = np.load(data_root / "run_1" / "boundary_cell_area_1.npy", mmap_mode="r")
    assert a.dtype == np.dtype("<f4") and a.ndim == 1
    assert float(a.min()) > 0


def test_split_priority_covers_all_runs(data_root: Path, config_dir: Path) -> None:
    m = load_manifest(data_root / "splits" / "manifest.json")
    prio = priority_runs(m, load_data(config_dir).download.priority)
    assert len(prio) == len(set(prio)) == len(all_runs(m)) == load_data(config_dir).n_runs == 484
    assert prio[:50] == m["full_test"]
