"""Physics: analytic checks of the integration, and the DrivAerML reference-force gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerosurr.config import load_data
from aerosurr.data.geometry import polygon_centres_and_area_vectors
from aerosurr.data.validate import CSV_KEYS, validate_run, within_tolerance
from aerosurr.physics import forces
from aerosurr.physics.coefficients import pressure_coefficient, skin_friction_coefficient
from aerosurr.physics.forces import Reference, integrate

from .conftest import CUBE_FACES, CUBE_POINTS

REF = Reference(a_ref=2.0, l_ref=4.0, cor=(0.5, 0.5, 0.5))


def cube() -> tuple[np.ndarray, np.ndarray]:
    """Centres and area vectors pointing INTO the unit cube (the DrivAerML orientation)."""
    off = np.cumsum([0] + [len(f) for f in CUBE_FACES])
    conn = np.concatenate(CUBE_FACES)
    c, a_out = polygon_centres_and_area_vectors(CUBE_POINTS, off, conn)
    return c, -a_out


def test_coefficients() -> None:
    np.testing.assert_allclose(pressure_coefficient([756.177, 0.0], 756.177, 0.0), [1.0, 0.0])
    np.testing.assert_allclose(pressure_coefficient([10.0], 5.0, p_inf=5.0), [1.0])
    np.testing.assert_allclose(skin_friction_coefficient([[2.0, 0, -1]], 4.0), [[0.5, 0, -0.25]])
    with pytest.raises(ValueError):
        pressure_coefficient([1.0], 0.0)


def test_constant_pressure_on_closed_surface_gives_no_force() -> None:
    c, a = cube()
    r = integrate(np.full(len(a), 3.7), np.zeros((len(a), 3)), a, c, REF)
    for v in (r.cd, r.cl, r.cs, r.cm_roll, r.cm_pitch, r.cm_yaw):
        assert v == pytest.approx(0.0, abs=1e-12)


def test_pressure_on_floor_lifts_the_body() -> None:
    c, a = cube()
    cp = np.zeros(len(a))
    cp[0] = 1.0  # face z = 0 (area 1): over-pressure underneath pushes the body up
    r = integrate(cp, np.zeros((len(a), 3)), a, c, REF)
    assert r.cl == pytest.approx(1.0 / REF.a_ref)
    assert r.cd == pytest.approx(0.0, abs=1e-12) and r.cs == pytest.approx(0.0, abs=1e-12)
    assert r.clf + r.clr == pytest.approx(r.cl)


def test_known_shear_gives_known_drag() -> None:
    """Uniform stored shear tau_x = -c (stress on the fluid) is drag +c * area on the body."""
    c, a = cube()
    cf = np.zeros((len(a), 3))
    cf[:, 0] = -0.004
    r = integrate(np.zeros(len(a)), cf, a, c, REF)
    total_area = np.linalg.norm(a, axis=1).sum()
    assert total_area == pytest.approx(6.0)
    assert forces.SHEAR_SIGN == -1.0
    assert r.cd == pytest.approx(0.004 * 6.0 / REF.a_ref)
    assert r.cd_friction == pytest.approx(r.cd) and r.cd_pressure == pytest.approx(0.0, abs=1e-15)


def test_pitch_moment_and_axle_split() -> None:
    """Lift on the face x = 1 (behind the CoR) pitches the nose down: rear axle loaded."""
    c, a = cube()
    cf = np.zeros((len(a), 3))
    cf[5:, 2] = -0.01  # the two triangles of x = 1: upward force on the body
    r = integrate(np.zeros(len(a)), cf, a, c, REF)
    lift = 0.01 * 1.0 / REF.a_ref
    assert r.cl == pytest.approx(lift)
    # r x F about the CoR: (0.5, *, *) x (0, 0, F) -> M_y = -0.5 F
    assert r.cm_pitch == pytest.approx(-0.5 * 0.01 / (REF.a_ref * REF.l_ref), rel=1e-6)
    assert r.clr > r.clf and r.clf + r.clr == pytest.approx(r.cl)


def test_tolerance_rule() -> None:
    assert within_tolerance("cd", 0.3035 * 1.004, 0.3035)
    assert not within_tolerance("cd", 0.3035 * 1.006, 0.3035)
    assert within_tolerance("cl", 0.00005, 0.0)  # absolute floor for near-zero lift
    assert within_tolerance("cs", 0.0219, 0.02) and not within_tolerance("clf", 0.0221, 0.02)


# --------------------------------------------------------------------------- real data

GATE_RUNS = [
    "run_1",
    "run_11",
    "run_12",
    "run_19",
    "run_4",
    "run_22",
    "run_56",
    "run_2",
    "run_3",
    "run_5",
]


@pytest.mark.data
@pytest.mark.parametrize("run", GATE_RUNS)
def test_reference_forces(data_root: Path, config_dir: Path, run: str) -> None:
    """SPEC §5.1 gate: integrated raw fields match force_mom*_i.csv on run_1 and 9 others."""
    if not (data_root / run / f"boundary_{run.split('_')[1]}.vtp").is_file():
        pytest.skip(f"{run} not downloaded (CI fetches run_1 only)")
    cfg = load_data(config_dir)
    res = validate_run(data_root, run, cfg.fields, cfg.freestream)
    for which in ("varying", "constant"):
        for k in CSV_KEYS:
            assert within_tolerance(k, res.ours[which][k], res.ref[which][k]), (which, k, res)
    assert abs(res.ours["varying"]["cd"] - res.ref["varying"]["cd"]) < 1e-6  # in fact exact


@pytest.mark.data
def test_wrong_shear_sign_fails_the_gate(
    data_root: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sign is established by the data: flipping it breaks Cd by ~19 %."""
    cfg = load_data(config_dir)
    monkeypatch.setattr(forces, "SHEAR_SIGN", +1.0)
    res = validate_run(data_root, "run_1", cfg.fields, cfg.freestream)
    assert not within_tolerance("cd", res.ours["varying"]["cd"], res.ref["varying"]["cd"])


def test_integrate_torch_matches_numpy() -> None:
    import torch

    from aerosurr.physics.forces import integrate_torch

    c, a = cube()
    rng = np.random.default_rng(3)
    cp, cf = rng.normal(size=len(a)), rng.normal(size=(len(a), 3)) * 0.01
    ref = integrate(cp, cf, a, c, REF).as_dict()
    got = integrate_torch(torch.from_numpy(cp), torch.from_numpy(cf), torch.from_numpy(a),
                          torch.from_numpy(c), REF).as_dict()  # fmt: skip
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-15), k
