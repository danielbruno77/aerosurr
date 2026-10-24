from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from aerosurr.config import DataConfig, load_data, load_paths


def test_repo_configs_load(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AEROSURR_WORKSPACE", "AEROSURR_DATA", "AEROSURR_BIND_INTERFACE"):
        monkeypatch.delenv(var, raising=False)
    paths = load_paths(config_dir)
    assert paths.workspace == Path("~/aero").expanduser()
    assert paths.data_raw == paths.workspace / "data/raw/drivaerml"
    assert paths.net.bind_interface == "enp97s0"
    data = load_data(config_dir)
    assert data.source.revision == "5d448b209bf654503c64ce7261c34fa125f46392"
    assert data.fields.association == "cell"
    assert data.download.priority[0] == "full_test"


def test_env_overrides(config_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AEROSURR_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("AEROSURR_DATA", str(tmp_path / "dml"))
    monkeypatch.setenv("AEROSURR_BIND_INTERFACE", "")
    paths = load_paths(config_dir)
    assert paths.workspace == tmp_path
    assert paths.logs == tmp_path / "logs"
    assert paths.data_raw == tmp_path / "dml"
    assert paths.net.bind_interface == ""


def test_q_inf_kinematic(config_dir: Path) -> None:
    fs = load_data(config_dir).freestream
    assert fs.pressure_is_kinematic
    assert fs.q_inf == pytest.approx(0.5 * 38.889**2)


def _data_dict(config_dir: Path) -> dict:  # type: ignore[type-arg]
    return yaml.safe_load((config_dir / "data.yaml").read_text())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "revision"), "main"),  # must be a pinned 40-hex sha
        (("download", "priority"), ["full_test", "full_test"]),
        (("download", "workers"), 0),
        (("freestream", "u_inf"), -1.0),
        (("fields", "association"), "vertex"),
    ],
)
def test_invalid_data_config(config_dir: Path, path: tuple[str, str], value: object) -> None:
    d = _data_dict(config_dir)
    d[path[0]][path[1]] = value
    with pytest.raises(ValidationError):
        DataConfig.model_validate(d)


def test_unknown_key_rejected(config_dir: Path) -> None:
    d = _data_dict(config_dir)
    d["fields"]["pressure_typo"] = "x"
    with pytest.raises(ValidationError):
        DataConfig.model_validate(d)
