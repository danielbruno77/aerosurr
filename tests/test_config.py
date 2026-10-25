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


def test_step_budget_schedule() -> None:
    from aerosurr.config import TrainConfig, resolve_schedule

    tc = TrainConfig(epochs=1, steps=20_000, eval_every_steps=2_000, full_eval_every_steps=10_000)
    s11 = resolve_schedule(tc, 11)
    assert s11.epochs == 1819 and s11.eval_every == 182 and s11.full_eval_every == 909
    assert s11.warmup_epochs == 91
    s400 = resolve_schedule(tc, 400)
    assert s400.epochs == 50 and s400.eval_every == 5 and s400.full_eval_every == 25
    assert resolve_schedule(TrainConfig(epochs=3), 7).epochs == 3  # no budget: unchanged


def test_data_variant_overlay(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEROSURR_DATA_VARIANT", raising=False)
    base, geo = load_data(config_dir), load_data(config_dir, "geometry")
    assert base.store.version == "v1" and base.store.stats_split == "full_train"
    assert geo.store.version == "v1_geometry" and geo.store.stats_split == "geometry_train"
    assert geo.store.full_splits == ["geometry_val", "geometry_test"]
    assert geo.store.pool_size == base.store.pool_size and geo.store.seed == base.store.seed
    assert geo.fields == base.fields  # everything outside `store` is inherited
    monkeypatch.setenv("AEROSURR_DATA_VARIANT", "geometry")
    assert load_data(config_dir).store.version == "v1_geometry"
