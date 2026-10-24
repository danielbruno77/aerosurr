"""CLI smoke tests: every command group parses, and the data commands run on tiny inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from aerosurr import __version__
from aerosurr.cli import BEGIN, END, app, write_generated

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AEROSURR_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("AEROSURR_DATA", str(tmp_path / "ws" / "raw"))
    monkeypatch.setenv("AEROSURR_BIND_INTERFACE", "")


@pytest.mark.parametrize("args", [[], ["env"], ["data"], ["config"]])
def test_help(args: list[str]) -> None:
    res = runner.invoke(app, [*args, "--help"])
    assert res.exit_code == 0, res.output
    assert "Usage" in res.output


def test_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0 and res.output.strip() == __version__


def test_config_show(tmp_path: Path) -> None:
    res = runner.invoke(app, ["config", "show"])
    assert res.exit_code == 0, res.output
    cfg = json.loads(res.output)
    assert cfg["paths"]["workspace"] == str(tmp_path / "ws")
    assert cfg["data"]["fields"]["pressure"] == "pMeanTrim"


def test_bad_config_dir(tmp_path: Path) -> None:
    res = runner.invoke(app, ["--config-dir", str(tmp_path), "config", "show"])
    assert res.exit_code != 0


def test_inspect_writes_doc(cube_vtp: Path, tmp_path: Path) -> None:
    out = tmp_path / "docs" / "inspection.md"
    res = runner.invoke(app, ["data", "inspect", str(cube_vtp), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "wallShearStressMeanTrim" in res.output and "U∞ = sqrt(2 q_inf) = 38.889" in res.output
    text = out.read_text()
    assert text.startswith("# DrivAerML data inspection") and BEGIN in text and END in text


def test_inspect_gate_fails_on_area_mismatch(cube_vtp: Path) -> None:
    np.save(cube_vtp.with_name("boundary_cell_area_7.npy"), np.ones(3, dtype="<f4"))
    res = runner.invoke(app, ["data", "inspect", str(cube_vtp)])
    assert res.exit_code == 1


def test_inspect_missing_area_file(cube_vtp: Path) -> None:
    cube_vtp.with_name("boundary_cell_area_7.npy").unlink()
    assert runner.invoke(app, ["data", "inspect", str(cube_vtp)]).exit_code == 2
    assert runner.invoke(app, ["data", "inspect", str(cube_vtp), "--no-areas"]).exit_code == 0


def test_download_rejects_unknown_tier() -> None:
    res = runner.invoke(app, ["data", "download", "--tier", "t9"])
    assert res.exit_code != 0


def test_status_without_plans() -> None:
    res = runner.invoke(app, ["data", "status"])
    assert res.exit_code == 1 and "no download plans" in res.output


def test_write_generated_keeps_hand_written_text(tmp_path: Path) -> None:
    doc = tmp_path / "d.md"
    write_generated(doc, "v1", "Title")
    doc.write_text(doc.read_text() + "\n## Notes\nkeep me\n")
    write_generated(doc, "v2", "Title")
    text = doc.read_text()
    assert "v2" in text and "v1" not in text and "keep me" in text


def test_download_min_free_gb_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """--min-free-gb reaches download() through the validated config."""
    import aerosurr.data.download as dl

    seen: dict[str, float] = {}

    def fake_download(hub: object, cfg: dl.DataConfig, *args: object, **kw: object) -> list[str]:
        seen["min_free_gb"] = cfg.download.min_free_gb
        return []

    monkeypatch.setattr(dl, "download", fake_download)
    res = runner.invoke(app, ["data", "download", "-t", "t2", "--dry-run", "--min-free-gb", "5"])
    assert res.exit_code == 0, res.output
    assert seen == {"min_free_gb": 5.0}
    assert runner.invoke(app, ["data", "download", "--min-free-gb", "-1"]).exit_code != 0


def test_rans_setup_and_summary(tmp_path: Path) -> None:
    """`rans setup` builds a case from run files under AEROSURR_DATA; `summary` tabulates."""
    from .test_rans import GEO_REF, ascii_stl

    run = tmp_path / "ws" / "raw" / "run_9"
    run.mkdir(parents=True)
    tri = np.array([[[0, 0, -0.33], [1, 0, -0.33], [0, 1, 0.5]]])
    ascii_stl(run / "drivaer_9.stl", {"BodyA": tri})
    (run / "geo_ref_9.csv").write_text(GEO_REF)
    case = tmp_path / "case"
    args = ["rans", "setup", "run_9", "--preset", "coarse", "--ranks", "4"]
    res = runner.invoke(app, [*args, "--iterations", "20", "--out", str(case)])
    assert res.exit_code == 0, res.output
    meta = json.loads((case / "aerosurr_case.json").read_text())
    assert meta["ranks"] == 4 and meta["iterations"] == 20
    assert "endTime         20;" in (case / "system/controlDict").read_text()
    assert runner.invoke(app, [*args, "--out", str(case)]).exit_code == 2  # refuses to overwrite
    assert runner.invoke(app, ["rans", "run", str(case), "--cpus", "0-7"]).exit_code != 0
    (case / "aerosurr_run.json").write_text(json.dumps({
        "case": "case", "n_ranks": 4, "cpus": "0-3", "status": "done", "cells": 1000,
        "steps": {"snappyHexMesh": 60.0, "simpleFoam": 120.0}, "total_s": 180.0,
        "s_per_iter": 0.5, "coeffs": {"Cd": 0.31, "Cl": 0.05},
        "reference": {"Cd": 0.3035, "Cl": 0.0677},
    }))  # fmt: skip
    out = runner.invoke(app, ["rans", "summary", str(case)])
    assert out.exit_code == 0 and "| +7 |" in out.output and "| 3.0 |" in out.output
