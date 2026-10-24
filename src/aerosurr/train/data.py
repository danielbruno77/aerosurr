"""Training data: normalisation and the in-memory pools of the compact store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from aerosurr.data.store import load_group


@dataclass(frozen=True)
class Normaliser:
    """Positions scaled isotropically into ~[-1, 1]; each target field standardised."""

    pos_centre: tuple[float, float, float]
    pos_scale: float
    y_mean: tuple[float, float, float, float]
    y_std: tuple[float, float, float, float]

    @classmethod
    def from_stats(cls, s: dict[str, Any]) -> Normaliser:
        c = s["pos_centre"]
        return cls(
            pos_centre=(float(c[0]), float(c[1]), float(c[2])),
            pos_scale=float(s["pos_scale"]),
            y_mean=(float(s["cp_mean"]), *(float(v) for v in s["cf_mean"])),  # type: ignore[arg-type]
            y_std=(float(s["cp_std"]), *(float(v) for v in s["cf_std"])),  # type: ignore[arg-type]
        )

    def pos(self, x: NDArray[Any]) -> NDArray[np.float32]:
        return ((x - np.asarray(self.pos_centre)) / self.pos_scale).astype(np.float32)

    def y(self, cp: NDArray[Any], cf: NDArray[Any]) -> NDArray[np.float32]:
        raw = np.concatenate([cp.reshape(-1, 1), cf.reshape(-1, 3)], axis=1).astype(np.float32)
        out = (raw - np.asarray(self.y_mean, np.float32)) / np.asarray(self.y_std, np.float32)
        return out.astype(np.float32)

    def pos_torch(self, x: torch.Tensor) -> torch.Tensor:
        c = torch.as_tensor(self.pos_centre, dtype=torch.float32, device=x.device)
        return (x.float() - c) / self.pos_scale

    def y_torch(self, y: torch.Tensor) -> torch.Tensor:
        """Physical (n, 4) (Cp, Cf) -> standardised."""
        m = torch.as_tensor(self.y_mean, dtype=torch.float32, device=y.device)
        s = torch.as_tensor(self.y_std, dtype=torch.float32, device=y.device)
        return (y.float() - m) / s

    def y_torch_inv(self, y: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.y_mean, dtype=y.dtype, device=y.device)
        s = torch.as_tensor(self.y_std, dtype=y.dtype, device=y.device)
        return y * s + m


class Batch(NamedTuple):
    pos: torch.Tensor  # (1, n, 3) normalised positions
    feat: torch.Tensor  # (1, n, 3 or 7) normals [+ normalised LF fields]
    y: torch.Tensor  # (1, n, 4) normalised HF targets
    area: torch.Tensor  # (1, n)
    lf: torch.Tensor | None  # (1, n, 4) normalised LF fields, if loaded


def load_lf(store: Path, run: str, group: str) -> NDArray[np.float16]:
    """Mapped LF fields (Cp, Cf) at the HF cells of a run's `pool` or `full` group."""
    from aerosurr.rans.mapping import LF_DIR

    with np.load(store / LF_DIR / f"{run}.npz") as z:
        out: NDArray[np.float16] = z[f"{group}_lf"]
    return out


class PoolSet:
    """The pools of several runs as stacked tensors: pos (R, N, 3), nrm (R, N, 3), y (R, N, 4),
    areas (R, N) and, with `lf`, the mapped LF fields (R, N, 4). All pools have the same
    size N (N <= every run's cell count)."""

    def __init__(
        self,
        store: Path,
        runs: list[str],
        norm: Normaliser,
        device: torch.device,
        lf: bool = False,
    ) -> None:
        pos, nrm, y, area, lfs = [], [], [], [], []
        for r in runs:
            g = load_group(store, r, "pool")
            pos.append(norm.pos(g["centres"]))
            nrm.append(g["normals"].astype(np.float32))
            y.append(norm.y(g["cp"], g["cf"]).astype(np.float16))
            area.append(g["areas"].astype(np.float32))
            if lf:
                f = load_lf(store, r, "pool").astype(np.float32)
                lfs.append(norm.y(f[:, 0], f[:, 1:]).astype(np.float16))
        self.runs = runs
        self.pos = torch.from_numpy(np.stack(pos)).to(device)
        self.nrm = torch.from_numpy(np.stack(nrm)).to(device)
        self.y = torch.from_numpy(np.stack(y)).to(device)
        self.area = torch.from_numpy(np.stack(area)).to(device)
        self.lf = torch.from_numpy(np.stack(lfs)).to(device) if lf else None

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def pool_size(self) -> int:
        return int(self.pos.shape[1])

    def sample(self, i: int, n: int, gen: torch.Generator, device: torch.device) -> Batch:
        """n random cells of run i, as (1, n, .) tensors on `device`."""
        idx = torch.randint(0, self.pool_size, (n,), generator=gen, device=gen.device)
        idx = idx.to(self.pos.device)

        def take(t: torch.Tensor) -> torch.Tensor:
            return t[i, idx].to(device, non_blocking=True)[None]

        lf = take(self.lf).float() if self.lf is not None else None
        nrm = take(self.nrm)
        feat = torch.cat([nrm, lf], dim=-1) if lf is not None else nrm
        return Batch(take(self.pos), feat, take(self.y).float(), take(self.area), lf)
