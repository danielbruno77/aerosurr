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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class StoreConfig(_Strict):
    version: str = Field(default="v1", pattern=r"^v\d+$")
    pool_size: int = Field(default=500_000, gt=0, description="training cells kept per run")
    seed: int = 0
    full_splits: list[str] = Field(
        default_factory=lambda: ["full_val", "full_test"],
        description="runs of these splits also keep every native cell (evaluation)",
    )
    stats_split: str = Field(default="full_train", description="normalisation statistics")


class DataConfig(_Strict):
    source: DatasetSource
    n_runs: int = Field(gt=0)
    fields: SurfaceFields
    freestream: Freestream
    download: DownloadConfig
    store: StoreConfig = StoreConfig()

    @field_validator("download")
    @classmethod
    def _unique_priority(cls, v: DownloadConfig) -> DownloadConfig:
        if len(set(v.priority)) != len(v.priority):
            raise ValueError("download.priority has duplicate split names")
        return v


# --------------------------------------------------------------------------- rans/*.yaml

Vec3 = tuple[float, float, float]


class RansDomain(_Strict):
    """Virtual wind tunnel (SPEC §5.3). x streamwise, z up, ground at `ground_z`."""

    x_min: float = Field(lt=0)
    x_max: float = Field(gt=0)
    y_half: float = Field(gt=0)
    height: float = Field(gt=0, description="top boundary height above the ground")
    ground_z: float = Field(description="ground plane z (HF VTP z_min = forcesCoR z)")
    x_bl: float = Field(description="ground is slip upstream of this x, a no-slip wall downstream")
    location_in_mesh: Vec3


class RansBox(_Strict):
    name: str = Field(pattern=r"^[A-Za-z]\w*$")
    min: Vec3
    max: Vec3
    level: int = Field(ge=0, le=10)


class RansMesh(_Strict):
    base_cell: float = Field(gt=0, description="blockMesh cell size (m)")
    surface_levels: tuple[int, int]
    feature_level: int = Field(ge=0, le=10)
    feature_angle: float = Field(default=150.0, gt=0, le=180)
    resolve_feature_angle: float = Field(default=30.0, gt=0, le=180)
    n_cells_between_levels: int = Field(default=3, ge=1)
    max_local_cells: int = Field(default=5_000_000, gt=0)
    max_global_cells: int = Field(default=100_000_000, gt=0)
    boxes: list[RansBox] = Field(default_factory=list)


class RansSolver(_Strict):
    iterations: int = Field(gt=0)
    average_last: int = Field(gt=0, description="iterations averaged for force coefficients")
    turbulence_intensity: float = Field(gt=0, lt=1)
    viscosity_ratio: float = Field(gt=0, description="nut/nu at the inlet")
    potential_init: bool = True


class RansOutput(_Strict):
    keep_volume: bool = Field(
        default=False,
        description="reconstruct and keep the volume mesh + final fields (~1 GB/case); "
        "otherwise keep only the car surface, forces and logs",
    )


class RansConfig(_Strict):
    domain: RansDomain
    mesh_presets: dict[str, RansMesh] = Field(min_length=1)
    solver: RansSolver
    output: RansOutput = RansOutput()


# --------------------------------------------------------------------------- model/, train/


class ModelConfig(_Strict):
    """Surface model (SPEC §5.2). Transolver variants come from NVIDIA PhysicsNeMo."""

    kind: Literal["transolver", "geotransolver", "pointnet"]
    n_layers: int = Field(default=8, ge=1)
    n_hidden: int = Field(default=256, ge=8)
    n_head: int = Field(default=8, ge=1)
    slice_num: int = Field(default=64, ge=1)
    mlp_ratio: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    heteroscedastic: bool = Field(default=False, description="predict mean + log-variance")
    lf_inputs: bool = Field(default=False, description="append mapped LF RANS (Cp, Cf) features")
    predict_residual: bool = Field(default=False, description="with lf_inputs: learn HF - LF")
    pn_local: list[int] = Field(default_factory=lambda: [128, 256])
    pn_global: int = 512
    pn_head: list[int] = Field(default_factory=lambda: [256, 128])

    @property
    def feature_dim(self) -> int:
        """Per-point features besides position: the normal, plus LF fields if used."""
        return 3 + (4 if self.lf_inputs else 0)

    @model_validator(mode="after")
    def _residual_needs_lf(self) -> ModelConfig:
        if self.predict_residual and not self.lf_inputs:
            raise ValueError("predict_residual needs lf_inputs")
        return self


class TrainConfig(_Strict):
    """Training schedule. With `steps`, epochs and eval cadences follow from the number of
    training runs, so different training-set sizes get the same budget."""

    split: str = Field(default="full", description="uses <split>_train and <split>_val")
    epochs: int = Field(gt=0)
    points: int = Field(default=200_000, gt=0, description="cells sampled per car per step")
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    warmup_epochs: int = Field(default=2, ge=0)
    grad_clip: float = Field(default=1.0, gt=0)
    seed: int = 0
    compile: bool = True
    bf16: bool = True
    data_on_gpu: bool = True
    eval_every: int = Field(default=5, gt=0, description="epochs between pool validations")
    full_eval_every: int = Field(default=25, gt=0, description="epochs between full-mesh evals")
    steps: int | None = Field(
        default=None, gt=0, description="step budget; overrides epochs as ceil(steps / runs)"
    )
    eval_every_steps: int | None = Field(default=None, gt=0, description="with steps")
    full_eval_every_steps: int | None = Field(default=None, gt=0, description="with steps")
    max_train_runs: int | None = Field(default=None, gt=0)
    train_runs: list[str] | None = Field(
        default=None, description="explicit training runs (overrides split/max_train_runs)"
    )
    max_hours: float | None = Field(default=None, gt=0)
    select_on_val: bool = Field(
        default=True,
        description="keep the best checkpoint on the val split; False keeps the final one "
        "(MF study: val cars have no LF RANS, so no model there validates)",
    )


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


def load_rans(name: str = "coarse", cfg_dir: Path | None = None) -> RansConfig:
    return RansConfig.model_validate(_read_yaml(config_dir(cfg_dir) / "rans" / f"{name}.yaml"))


def load_model(name: str, cfg_dir: Path | None = None) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml(config_dir(cfg_dir) / "model" / f"{name}.yaml"))


def load_train(name: str, cfg_dir: Path | None = None) -> TrainConfig:
    return TrainConfig.model_validate(_read_yaml(config_dir(cfg_dir) / "train" / f"{name}.yaml"))


def resolve_schedule(tc: TrainConfig, n_runs: int) -> TrainConfig:
    """Turn a step budget into epochs, eval cadences and a 5 % warm-up for `n_runs` runs."""
    if tc.steps is None:
        return tc
    import math

    def every(steps: int | None, default: int) -> int:
        return max(1, round(steps / n_runs)) if steps else default

    epochs = math.ceil(tc.steps / n_runs)
    return tc.model_copy(update={
        "epochs": epochs,
        "eval_every": every(tc.eval_every_steps, tc.eval_every),
        "full_eval_every": min(epochs, every(tc.full_eval_every_steps, tc.full_eval_every)),
        "warmup_epochs": max(1, math.ceil(0.05 * epochs)),  # 5 % of the budget
    })  # fmt: skip
