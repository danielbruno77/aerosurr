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
