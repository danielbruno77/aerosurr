"""Surface-field models: (cell centres, unit normals) -> (Cp, Cf_x, Cf_y, Cf_z), normalised.

The main models wrap NVIDIA PhysicsNeMo networks (SPEC §5.2). The PointNet baseline is
in-house and gives the floor. With `heteroscedastic`, a model outputs 4 means and 4
log-variances (deep-ensemble members, SPEC §5.4).
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from aerosurr.config import ModelConfig

N_FIELDS = 4  # Cp, Cf_x, Cf_y, Cf_z


def _mlp(dims: list[int], last_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(itertools.pairwise(dims)):
        layers.append(nn.Linear(a, b))
        if last_act or i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class PointNetBaseline(nn.Module):
    """Per-point MLP, max-pooled into a global shape code, concatenated back per point."""

    def __init__(self, cfg: ModelConfig, out_dim: int) -> None:
        super().__init__()
        self.local = _mlp([6, *cfg.pn_local])
        self.glob = _mlp([cfg.pn_local[-1], cfg.pn_global])
        self.head = _mlp([cfg.pn_local[-1] + cfg.pn_global, *cfg.pn_head, out_dim], last_act=False)

    def forward(self, pos: torch.Tensor, nrm: torch.Tensor) -> torch.Tensor:
        h = self.local(torch.cat([pos, nrm], dim=-1))  # (B, N, C)
        # Global max-pool in fp32 via max(dim): under torch.compile + bf16, amax's backward
        # (which splits the gradient among tied maxima) gave NaN gradients.
        g = self.glob(h).float().max(dim=1, keepdim=True).values  # (B, 1, G)
        g = g.to(h.dtype).expand(-1, h.shape[1], -1)
        out: torch.Tensor = self.head(torch.cat([h, g], dim=-1))
        return out


class PhysicsNeMoTransolver(nn.Module):
    """`physicsnemo.models.transolver.Transolver`: normals as features, positions as embedding."""

    def __init__(self, cfg: ModelConfig, out_dim: int) -> None:
        super().__init__()
        from physicsnemo.models.transolver import Transolver

        self.net = Transolver(
            functional_dim=3,
            embedding_dim=3,
            out_dim=out_dim,
            n_layers=cfg.n_layers,
            n_hidden=cfg.n_hidden,
            n_head=cfg.n_head,
            slice_num=cfg.slice_num,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_te=False,
        )

    def forward(self, pos: torch.Tensor, nrm: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(nrm, embedding=pos)
        return out


class PhysicsNeMoGeoTransolver(nn.Module):
    """`physicsnemo.models.geotransolver.GeoTransolver` with the positions as geometry."""

    def __init__(self, cfg: ModelConfig, out_dim: int) -> None:
        super().__init__()
        from physicsnemo.models.geotransolver import GeoTransolver

        self.net = GeoTransolver(
            functional_dim=6,
            out_dim=out_dim,
            geometry_dim=3,
            n_layers=cfg.n_layers,
            n_hidden=cfg.n_hidden,
            n_head=cfg.n_head,
            slice_num=cfg.slice_num,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_te=False,
        )

    def forward(self, pos: torch.Tensor, nrm: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(torch.cat([pos, nrm], dim=-1), geometry=pos)
        return out


def build_model(cfg: ModelConfig) -> nn.Module:
    out_dim = 2 * N_FIELDS if cfg.heteroscedastic else N_FIELDS
    kinds: dict[str, type[nn.Module]] = {
        "pointnet": PointNetBaseline,
        "transolver": PhysicsNeMoTransolver,
        "geotransolver": PhysicsNeMoGeoTransolver,
    }
    model: nn.Module = kinds[cfg.kind](cfg, out_dim)
    return model


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
