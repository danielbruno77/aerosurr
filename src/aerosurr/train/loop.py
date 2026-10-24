"""Training loop and evaluation for the surface models (SPEC §5.2, §5.5)."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from aerosurr.config import ModelConfig, TrainConfig
from aerosurr.data.splits import run_number
from aerosurr.data.store import load_group, load_index
from aerosurr.data.validate import read_geo_refs
from aerosurr.eval.metrics import FIELD_NAMES, relative_l2
from aerosurr.models.surface import N_FIELDS, build_model, n_params
from aerosurr.physics.forces import integrate_torch
from aerosurr.train.data import Normaliser, PoolSet, load_lf

log = logging.getLogger("aerosurr.train")

#: An nn.Module or its torch.compile wrapper: (pos, nrm) -> outputs.
ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def split_outputs(
    out: torch.Tensor, heteroscedastic: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if heteroscedastic:
        return out[..., :N_FIELDS], out[..., N_FIELDS:]
    return out, None


def loss_fn(out: torch.Tensor, y: torch.Tensor, heteroscedastic: bool) -> torch.Tensor:
    """MSE on standardised targets, or the Gaussian NLL with predicted log-variance."""
    mu, logvar = split_outputs(out.float(), heteroscedastic)
    if logvar is None:
        return ((mu - y) ** 2).mean()
    logvar = logvar.clamp(-12.0, 8.0)
    return (0.5 * (logvar + (mu - y) ** 2 * torch.exp(-logvar))).mean()


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def predict_tensor(
    model: ModelFn,
    norm: Normaliser,
    centres: torch.Tensor,
    normals: torch.Tensor,
    chunk: int,
    bf16: bool,
    heteroscedastic: bool = False,
    seed: int = 0,
    lf: torch.Tensor | None = None,
    residual: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Physical (Cp, Cf) (n, 4) for every cell, on the tensors' device, in random chunks of
    exactly `chunk` cells. Chunks are a random partition (the last one is padded by
    wrapping), so every chunk looks like a training sample and the compiled graph keeps
    one static shape. `lf` are the mapped LF fields (n, 4, physical) for MF models; with
    `residual` the model output is HF - LF (standardised)."""
    dev = centres.device
    n = centres.shape[0]
    gen = torch.Generator(device=dev).manual_seed(seed)
    perm = torch.randperm(n, generator=gen, device=dev)
    n_chunks = max(1, math.ceil(n / chunk))
    padded = torch.cat([perm, perm[: n_chunks * chunk - n]])
    pos = norm.pos_torch(centres)
    feat = normals.float()
    lf_n = None
    if lf is not None:
        lf_n = norm.y_torch(lf)
        feat = torch.cat([feat, lf_n], dim=-1)
    mean = torch.empty((n, N_FIELDS), dtype=torch.float32, device=dev)
    var = torch.empty((n, N_FIELDS), dtype=torch.float32, device=dev) if heteroscedastic else None
    for k in range(n_chunks):
        idx = padded[k * chunk : (k + 1) * chunk]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and dev.type == "cuda"):
            out = model(pos[idx][None], feat[idx][None])[0].float()
        mu, logvar = split_outputs(out, heteroscedastic)
        if residual and lf_n is not None:
            mu = mu + lf_n[idx]
        mean[idx] = norm.y_torch_inv(mu)
        if var is not None and logvar is not None:
            s = torch.as_tensor(norm.y_std, device=dev)
            var[idx] = torch.exp(logvar.clamp(-12.0, 8.0)) * s**2
    return mean, var


def predict(
    model: ModelFn,
    norm: Normaliser,
    centres: NDArray[Any],
    normals: NDArray[Any],
    chunk: int,
    device: torch.device,
    bf16: bool,
    heteroscedastic: bool = False,
    seed: int = 0,
    lf: NDArray[Any] | None = None,
    residual: bool = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """`predict_tensor` from and to numpy."""

    def t(x: NDArray[Any]) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)

    mean, var = predict_tensor(model, norm, t(centres), t(normals), chunk, bf16, heteroscedastic,
                               seed, None if lf is None else t(lf), residual)  # fmt: skip
    return mean.cpu().numpy(), None if var is None else var.cpu().numpy()


def _rel_l2_torch(pred: torch.Tensor, true: torch.Tensor, w: torch.Tensor) -> list[float]:
    num = (w[:, None].double() * (pred.double() - true.double()) ** 2).sum(dim=0)
    den = (w[:, None].double() * true.double() ** 2).sum(dim=0).clamp_min(1e-300)
    return [float(v) for v in torch.sqrt(num / den)]


def evaluate_full(
    model: ModelFn,
    norm: Normaliser,
    store: Path,
    raw: Path,
    runs: list[str],
    chunk: int,
    device: torch.device,
    bf16: bool,
    lf_inputs: bool = False,
    residual: bool = False,
) -> dict[str, Any]:
    """Field errors on the native mesh and integrated-force errors, per run and averaged.

    Runs on `device`. The reference ("true") coefficients come from the store index,
    integrated from the raw float64 fields at build time.
    """
    index = load_index(store)
    per_run: dict[str, Any] = {}
    for r in runs:
        g = {k: torch.from_numpy(v).to(device) for k, v in load_group(store, r, "full").items()}
        lf = torch.from_numpy(load_lf(store, r, "full")).to(device) if lf_inputs else None
        pred, _ = predict_tensor(model, norm, g["centres"], g["normals"], chunk, bf16, lf=lf,
                                 residual=residual)  # fmt: skip
        true = torch.cat([g["cp"].float().reshape(-1, 1), g["cf"].float().reshape(-1, 3)], 1)
        a_in = -g["normals"].double() * g["areas"].double()[:, None]
        ref = read_geo_refs(raw / r / f"geo_ref_{run_number(r)}.csv").varying
        fp = integrate_torch(pred[:, 0], pred[:, 1:], a_in, g["centres"], ref)
        if r in index:
            ft = index[r].coeffs["varying"]
        else:
            ft = integrate_torch(true[:, 0], true[:, 1:], a_in, g["centres"], ref).as_dict()
        rel = _rel_l2_torch(pred, true, g["areas"])
        per_run[r] = {"rel_l2": dict(zip(FIELD_NAMES, rel, strict=True)), "pred": fp.as_dict(),
                      "true": ft}  # fmt: skip
    rel_mean = {f: float(np.mean([v["rel_l2"][f] for v in per_run.values()])) for f in FIELD_NAMES}
    dcd = np.array([v["pred"]["cd"] - v["true"]["cd"] for v in per_run.values()])
    dcl = np.array([v["pred"]["cl"] - v["true"]["cl"] for v in per_run.values()])
    return {
        "runs": len(runs),
        "rel_l2": rel_mean,
        "cd_mae_counts": float(np.abs(dcd).mean() * 1000),
        "cd_max_counts": float(np.abs(dcd).max() * 1000),
        "cl_mae": float(np.abs(dcl).mean()),
        "per_run": per_run,
    }


@torch.no_grad()
def evaluate_pools(
    model: ModelFn, pools: PoolSet, n: int, device: torch.device, bf16: bool, hetero: bool,
    norm: Normaliser, residual: bool = False,
) -> dict[str, float]:  # fmt: skip
    """Quick validation: one fixed random subset per run, area-weighted relative L2."""
    gen = torch.Generator(device=pools.pos.device).manual_seed(1234)
    rels = []
    for i in range(len(pools)):
        b = pools.sample(i, n, gen, device)
        y, area = b.y, b.area
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            out = model(b.pos, b.feat)
        mu, _ = split_outputs(out.float(), hetero)
        if residual and b.lf is not None:
            mu = mu + b.lf
        p = norm.y_torch_inv(mu)[0].cpu().numpy()
        t = norm.y_torch_inv(y)[0].cpu().numpy()
        rels.append(relative_l2(p, t, area[0].cpu().numpy()))
    m = np.mean(rels, axis=0)
    return dict(zip(FIELD_NAMES, (float(v) for v in m), strict=True))


@dataclass
class TrainSummary:
    name: str
    epochs_done: int
    steps: int
    hours: float
    best_epoch: int
    best_val: dict[str, float]
    final_full_val: dict[str, Any] | None


def train(
    name: str,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    manifest: dict[str, list[str]],
    store: Path,
    raw: Path,
    stats: dict[str, Any],
    out_dir: Path,
    device: torch.device,
) -> TrainSummary:
    tc, hetero = train_cfg, model_cfg.heteroscedastic
    torch.manual_seed(tc.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = {"model": model_cfg.model_dump(), "train": tc.model_dump(), "stats": stats}
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))
    norm = Normaliser.from_stats(stats)
    train_runs = tc.train_runs or manifest[f"{tc.split}_train"][: tc.max_train_runs]
    val_runs = manifest[f"{tc.split}_val"]
    lf, res = model_cfg.lf_inputs, model_cfg.predict_residual
    data_dev = device if tc.data_on_gpu else torch.device("cpu")
    t_load = time.time()
    tr = PoolSet(store, train_runs, norm, data_dev, lf=lf)
    va = PoolSet(store, val_runs, norm, data_dev, lf=lf)
    log.info("%s: %d train / %d val runs, pool %d, loaded in %.0f s (data on %s)", name,
             len(tr), len(va), tr.pool_size, time.time() - t_load, data_dev)  # fmt: skip

    model = build_model(model_cfg).to(device)
    log.info("%s: %s, %.2f M params", name, model_cfg.kind, n_params(model) / 1e6)
    fwd = torch.compile(model) if tc.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    total = tc.epochs * len(tr)
    warm = tc.warmup_epochs * len(tr)
    gen = torch.Generator(device=tr.pos.device).manual_seed(tc.seed)
    t0 = time.time()
    step, best, best_epoch, epoch = 0, float("inf"), -1, 0
    best_val: dict[str, float] = {}
    metrics = (out_dir / "metrics.jsonl").open("a")
    for epoch in range(1, tc.epochs + 1):
        model.train()
        order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(tc.seed + epoch))
        ep_loss = 0.0
        for i in order.tolist():
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step, total, warm, tc.lr)
            b = tr.sample(i, tc.points, gen, device)
            target = b.y - b.lf if res and b.lf is not None else b.y
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=tc.bf16 and device.type == "cuda"
            ):
                out = fwd(b.pos, b.feat)
            loss = loss_fn(out, target, hetero)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            ep_loss += float(loss.detach())
            step += 1
        hours = (time.time() - t0) / 3600
        rec: dict[str, Any] = {"epoch": epoch, "step": step, "loss": ep_loss / len(tr),
                               "lr": opt.param_groups[0]["lr"], "hours": hours}  # fmt: skip
        if epoch % tc.eval_every == 0 or epoch == tc.epochs:
            model.eval()
            val = evaluate_pools(fwd, va, tc.points, device, tc.bf16, hetero, norm, res)
            rec["val_rel_l2"] = val
            score = float(np.mean(list(val.values())))
            if score < best:
                best, best_epoch, best_val = score, epoch, val
                torch.save(model.state_dict(), out_dir / "best.pt")
        if epoch % tc.full_eval_every == 0:
            model.eval()
            full = evaluate_full(
                fwd, norm, store, raw, val_runs, tc.points, device, tc.bf16, lf, res
            )
            rec["full_val"] = {k: v for k, v in full.items() if k != "per_run"}
        torch.save(model.state_dict(), out_dir / "last.pt")
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        log.info("%s: epoch %d loss %.4f lr %.2e %s%s", name, epoch, rec["loss"], rec["lr"],
                 f"val {rec['val_rel_l2']} " if "val_rel_l2" in rec else "",
                 f"full {rec['full_val']}" if "full_val" in rec else "")  # fmt: skip
        if tc.max_hours and hours > tc.max_hours:
            log.warning("%s: time limit %.1f h reached after epoch %d", name, tc.max_hours, epoch)
            break
    metrics.close()
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()
    full = evaluate_full(fwd, norm, store, raw, val_runs, tc.points, device, tc.bf16, lf, res)
    (out_dir / "full_val.json").write_text(json.dumps(full, indent=1))
    summary = TrainSummary(name, epoch, step, (time.time() - t0) / 3600, best_epoch, best_val,
                           {k: v for k, v in full.items() if k != "per_run"})  # fmt: skip
    (out_dir / "summary.json").write_text(json.dumps(summary.__dict__, indent=2))
    log.info("%s: done; best epoch %d, full-mesh val %s", name, best_epoch, summary.final_full_val)
    return summary
