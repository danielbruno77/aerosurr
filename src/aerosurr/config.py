"""Pydantic schemas for `configs/*.yaml` and the loader that validates them.

Paths in `paths.yaml` may be relative; they resolve against `workspace`. Environment
variables override the yaml (SPEC §8):

- `AEROSURR_CONFIG_DIR`: directory holding the yaml files.
- `AEROSURR_WORKSPACE`: workspace root (default `~/aero`).
- `AEROSURR_DATA`: raw DrivAerML directory (the one containing `run_1/`).
- `AEROSURR_BIND_INTERFACE`: network interface for bulk downloads; empty = default route.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- paths.yaml


class NetConfig(_Strict):
    bind_interface: str = Field(
        default="", description="Interface for SO_BINDTODEVICE; empty uses the default route."
    )


class PathsConfig(_Strict):
    workspace: Path
    data_raw: Path
    data_store: Path
    runs: Path
    rans: Path
    logs: Path
    net: NetConfig = NetConfig()

    def resolved(self) -> PathsConfig:
        """Return a copy with `~` expanded and relative paths anchored at `workspace`."""
        ws = self.workspace.expanduser()

        def anchor(p: Path) -> Path:
            p = p.expanduser()
            return p if p.is_absolute() else ws / p

        return self.model_copy(
            update={
                "workspace": ws,
                "data_raw": anchor(self.data_raw),
                "data_store": anchor(self.data_store),
                "runs": anchor(self.runs),
                "rans": anchor(self.rans),
                "logs": anchor(self.logs),
            }
        )


# --------------------------------------------------------------------------- data.yaml


class DatasetSource(_Strict):
    repo_id: str
    repo_type: Literal["dataset"] = "dataset"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    endpoint: str = "https://huggingface.co"
    license: str


class SurfaceFields(_Strict):
    """Names of the arrays in `boundary_i.vtp`, copied from `docs/data_inspection.md`."""

    association: Literal["cell", "point"]
    pressure: str
    pressure_coefficient: str
    pressure_fluctuation: str
    wall_shear_stress: str


class Freestream(_Strict):
    """Flow conditions; see `docs/data_inspection.md` for where each value comes from."""

    u_inf: float = Field(gt=0, description="m/s")
    rho_inf: float = Field(gt=0, description="kg/m^3")
    nu: float = Field(gt=0, description="m^2/s")
    p_inf: float = Field(description="kinematic, m^2/s^2")
    pressure_is_kinematic: bool

    @property
    def q_inf(self) -> float:
        """Dynamic pressure in the units of the stored pressure field."""
        rho = 1.0 if self.pressure_is_kinematic else self.rho_inf
        return 0.5 * rho * self.u_inf**2


class DownloadConfig(_Strict):
    priority: list[str] = Field(min_length=1, description="Split names, downloaded in order.")
    workers: int = Field(default=4, ge=1, le=32)
    chunk_bytes: int = Field(default=8 << 20, ge=1 << 16)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=8, ge=0)
    min_free_gb: float = Field(default=100.0, ge=0)
    paths_info_batch: int = Field(default=200, ge=1, le=1000)


class DataConfig(_Strict):
    source: DatasetSource
    n_runs: int = Field(gt=0)
    fields: SurfaceFields
    freestream: Freestream
    download: DownloadConfig

    @field_validator("download")
    @classmethod
    def _unique_priority(cls, v: DownloadConfig) -> DownloadConfig:
        if len(set(v.priority)) != len(v.priority):
            raise ValueError("download.priority has duplicate split names")
        return v


# --------------------------------------------------------------------------- loader


def config_dir(override: Path | None = None) -> Path:
    if override is not None:
        return override
    env = os.environ.get("AEROSURR_CONFIG_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_paths(cfg_dir: Path | None = None) -> PathsConfig:
    raw = _read_yaml(config_dir(cfg_dir) / "paths.yaml")
    if ws := os.environ.get("AEROSURR_WORKSPACE"):
        raw["workspace"] = ws
    if data := os.environ.get("AEROSURR_DATA"):
        raw["data_raw"] = data
    if "AEROSURR_BIND_INTERFACE" in os.environ:
        raw.setdefault("net", {})["bind_interface"] = os.environ["AEROSURR_BIND_INTERFACE"]
    return PathsConfig.model_validate(raw).resolved()


def load_data(cfg_dir: Path | None = None) -> DataConfig:
    return DataConfig.model_validate(_read_yaml(config_dir(cfg_dir) / "data.yaml"))
