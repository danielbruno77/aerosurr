"""Deep-ensemble uncertainty (SPEC §5.4).

Each member predicts a Gaussian per cell and field (mean, variance). The ensemble mean is
the average of member means; the total variance is the mean aleatoric variance plus the
variance of the member means (epistemic). A single temperature per field, fitted on the
calibration split, rescales sigma. Integrated forces get their spread from integrating
every member's mean field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from aerosurr.config import ModelConfig
from aerosurr.data.splits import run_number
from aerosurr.data.store import load_group, load_index
from aerosurr.data.validate import read_geo_refs
from aerosurr.eval.metrics import FIELD_NAMES
from aerosurr.physics.forces import integrate_torch
from aerosurr.train.data import Normaliser, load_lf
from aerosurr.train.loop import ModelFn, predict_tensor

Z = {0.5: 0.6744897501960817, 0.9: 1.6448536269514722, 0.95: 1.959963984540054}


@dataclass(frozen=True)
class Combined:
    mean: torch.Tensor  # (n, f)
    var_aleatoric: torch.Tensor  # (n, f)
    var_epistemic: torch.Tensor  # (n, f)

    @property
    def var(self) -> torch.Tensor:
        return self.var_aleatoric + self.var_epistemic


def combine(means: list[torch.Tensor], variances: list[torch.Tensor | None]) -> Combined:
    """Members' (n, f) means and variances (None = no aleatoric head) -> ensemble moments."""
    m = torch.stack([x.double() for x in means])
    mu = m.mean(dim=0)
    epi = m.var(dim=0, unbiased=False) if len(means) > 1 else torch.zeros_like(mu)
    al = [v.double() for v in variances if v is not None]
    ale = torch.stack(al).mean(dim=0) if al else torch.zeros_like(mu)
    return Combined(mu, ale, epi)


def fit_temperature(err: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Per-field T minimising the Gaussian NLL of err ~ N(0, T^2 var): T^2 = mean(err^2/var)."""
    return torch.sqrt(((err.double() ** 2) / var.double().clamp_min(1e-30)).mean(dim=0))


def coverage(err: torch.Tensor, sigma: torch.Tensor, q: float) -> NDArray[np.float64]:
    """Fraction of |err| <= z_q sigma, per field (central interval of probability q)."""
    inside = err.abs() <= Z[q] * sigma
    out: NDArray[np.float64] = inside.double().mean(dim=0).cpu().numpy()
    return out


def gaussian_nll(err: torch.Tensor, var: torch.Tensor) -> NDArray[np.float64]:
    v = var.double().clamp_min(1e-30)
    nll = 0.5 * (torch.log(2 * math.pi * v) + err.double() ** 2 / v)
    out: NDArray[np.float64] = nll.mean(dim=0).cpu().numpy()
    return out


def spearman(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Spearman rank correlation (no ties handling needed for continuous data)."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


@dataclass
class _Acc:
    """Streaming accumulators over runs (per field unless noted)."""

    ratio: NDArray[np.float64]  # sum err^2 / var
    n: int
    cover: dict[float, NDArray[np.float64]]  # sum of inside-interval counts
    nll: NDArray[np.float64]  # sum of NLL
    sig_sample: list[NDArray[np.float64]]
    err_sample: list[NDArray[np.float64]]
    ale_frac: list[float]
    cd_mean: list[float]
    cd_std: list[float]
    cd_true: list[float]

    @classmethod
    def empty(cls) -> _Acc:
        return cls(np.zeros(4), 0, {q: np.zeros(4) for q in Z}, np.zeros(4), [], [], [], [], [], [])


def evaluate_ensemble(
    members: list[tuple[ModelFn, ModelConfig]],
    norm: Normaliser,
    store: Path,
    raw: Path,
    calib_runs: list[str],
    test_runs: list[str],
    chunk: int,
    bf16: bool,
    device: torch.device,
    sample_per_run: int = 20_000,
) -> dict[str, Any]:
    """Calibrate on `calib_runs` (a temperature per field and one for Cd), report on
    `test_runs`. Heteroscedastic members give the aleatoric part. Streams over runs: only
    one run's full mesh is on the GPU at a time."""
    index = load_index(store)
    gen = torch.Generator(device="cpu").manual_seed(0)

    def run_pass(runs: list[str], t_field: torch.Tensor | None) -> _Acc:
        acc = _Acc.empty()
        for r in runs:
            g = {k: torch.from_numpy(v).to(device) for k, v in load_group(store, r, "full").items()}
            true = torch.cat([g["cp"].float().reshape(-1, 1), g["cf"].float().reshape(-1, 3)], 1)
            a_in = -g["normals"].double() * g["areas"].double()[:, None]
            ref = read_geo_refs(raw / r / f"geo_ref_{run_number(r)}.csv").varying
            means, variances, cds = [], [], []
            for model, cfg in members:
                lf = None
                if cfg.lf_inputs:
                    lf = torch.from_numpy(load_lf(store, r, "full")).to(device)
                mu, var = predict_tensor(
                    model,
                    norm,
                    g["centres"],
                    g["normals"],
                    chunk,
                    bf16,
                    cfg.heteroscedastic,
                    lf=lf,
                    residual=cfg.predict_residual,
                )
                means.append(mu)
                variances.append(var)
                cds.append(integrate_torch(mu[:, 0], mu[:, 1:], a_in, g["centres"], ref).cd)
            c = combine(means, variances)
            del means, variances
            err = c.mean - true.double()
            var = c.var.clamp_min(1e-30)
            acc.ratio += (err**2 / var).sum(dim=0).cpu().numpy()
            acc.n += err.shape[0]
            acc.ale_frac.append(float(c.var_aleatoric.sum() / var.sum()))
            acc.cd_mean.append(float(np.mean(cds)))
            acc.cd_std.append(float(np.std(cds)))
            acc.cd_true.append(float(index[r].coeffs["varying"]["cd"]))
            if t_field is not None:
                sig = t_field * var.sqrt()
                for q in Z:
                    acc.cover[q] += coverage(err, sig, q) * err.shape[0]
                acc.nll += gaussian_nll(err, sig**2) * err.shape[0]
                idx = torch.randperm(err.shape[0], generator=gen)[:sample_per_run].to(device)
                acc.sig_sample.append(sig[idx].cpu().numpy())
                acc.err_sample.append(err[idx].abs().cpu().numpy())
            del g, c, err, var
        return acc

    def cd_arrays(acc: _Acc) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.array(acc.cd_mean) - np.array(acc.cd_true), np.array(acc.cd_std)

    cal = run_pass(calib_runs, None)
    t_field = np.sqrt(cal.ratio / cal.n)
    cd_err_c, cd_std_c = cd_arrays(cal)
    t_cd = float(np.sqrt(np.mean(cd_err_c**2 / np.maximum(cd_std_c, 1e-12) ** 2)))
    test = run_pass(test_runs, torch.as_tensor(t_field, device=device))
    cd_err, cd_std = cd_arrays(test)
    sig_cd = t_cd * cd_std
    ss, ee = np.concatenate(test.sig_sample), np.concatenate(test.err_sample)
    fields = FIELD_NAMES
    return {
        "members": len(members),
        "calib_runs": len(calib_runs),
        "test_runs": len(test_runs),
        "temperature": {"fields": dict(zip(fields, t_field.tolist(), strict=True)), "cd": t_cd},
        "aleatoric_fraction": float(np.mean(test.ale_frac)),
        "coverage": {
            str(q): dict(zip(fields, (test.cover[q] / test.n).tolist(), strict=True)) for q in Z
        },
        "nll": dict(zip(fields, (test.nll / test.n).tolist(), strict=True)),
        "spearman_point": {f: spearman(ss[:, i], ee[:, i]) for i, f in enumerate(fields)},
        "cd_mae_counts": float(np.abs(cd_err).mean() * 1000),
        "cd_sigma_mean_counts": float(sig_cd.mean() * 1000),
        "cd_spearman_car": spearman(sig_cd, np.abs(cd_err)),
        "cd_coverage": {str(q): float(np.mean(np.abs(cd_err) <= Z[q] * sig_cd)) for q in Z},
        "per_run": {
            r: {"cd_mean": m, "cd_std_cal": t_cd * sd, "cd_true": t}
            for r, m, sd, t in zip(test_runs, test.cd_mean, test.cd_std, test.cd_true, strict=True)
        },
    }


# --------------------------------------------------------------------------- per-car Cd check


def member_forces(
    members: list[tuple[ModelFn, ModelConfig]],
    norm: Normaliser,
    store: Path,
    raw: Path,
    runs: list[str],
    chunk: int,
    bf16: bool,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Each member's integrated Cd and Cl per run (from its mean field), plus the truth."""
    index = load_index(store)
    out: dict[str, dict[str, Any]] = {}
    for r in runs:
        g = {k: torch.from_numpy(v).to(device) for k, v in load_group(store, r, "full").items()}
        a_in = -g["normals"].double() * g["areas"].double()[:, None]
        ref = read_geo_refs(raw / r / f"geo_ref_{run_number(r)}.csv").varying
        cd, cl = [], []
        for model, cfg in members:
            lf = torch.from_numpy(load_lf(store, r, "full")).to(device) if cfg.lf_inputs else None
            mu, _ = predict_tensor(
                model,
                norm,
                g["centres"],
                g["normals"],
                chunk,
                bf16,
                cfg.heteroscedastic,
                lf=lf,
                residual=cfg.predict_residual,
            )
            f = integrate_torch(mu[:, 0], mu[:, 1:], a_in, g["centres"], ref)
            cd.append(f.cd)
            cl.append(f.cl)
        out[r] = {
            "cd_members": cd,
            "cl_members": cl,
            "cd_true": float(index[r].coeffs["varying"]["cd"]),
            "cl_true": float(index[r].coeffs["varying"]["cl"]),
        }
        del g
    return out


def _spearman_ci(
    sigma: NDArray[np.float64], abs_err: NDArray[np.float64], n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Spearman(sigma, |err|) and a percentile bootstrap 95 % CI over cars."""
    rho = spearman(sigma, abs_err)
    rng = np.random.default_rng(seed)
    n = len(sigma)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(sigma[i]) > 0 and np.ptp(abs_err[i]) > 0:
            boots.append(spearman(sigma[i], abs_err[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


def sigma_row(
    sigma: NDArray[np.float64], err: NDArray[np.float64], n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Coverage, mean sigma (counts) and Spearman with a bootstrap CI, for per-car Cd."""
    rho, lo, hi = _spearman_ci(sigma, np.abs(err), n_boot, seed)
    return {
        "coverage": {str(q): float(np.mean(np.abs(err) <= Z[q] * sigma)) for q in Z},
        "sigma_mean_counts": float(sigma.mean() * 1000),
        "spearman": rho,
        "spearman_ci95": [lo, hi],
        "n": len(sigma),
    }


def per_car_cd_check(
    val: dict[str, dict[str, Any]],
    test: dict[str, dict[str, Any]],
    ddof: int = 0,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """sigma_epi(Cd) = std over members of each member's integrated Cd (ddof as given),
    raw and with one temperature fitted on `val` (T^2 = mean(err^2 / sigma^2)), scored on
    `test`. The error is the ensemble-mean Cd minus the HRLES Cd."""

    def arrays(t: dict[str, dict[str, Any]]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        m = np.array([v["cd_members"] for v in t.values()], dtype=np.float64)
        truth = np.array([v["cd_true"] for v in t.values()], dtype=np.float64)
        return m.std(axis=1, ddof=ddof), m.mean(axis=1) - truth

    s_val, e_val = arrays(val)
    s_test, e_test = arrays(test)
    temp = float(np.sqrt(np.mean(e_val**2 / np.maximum(s_val, 1e-12) ** 2)))
    return {
        "ddof": ddof,
        "members": len(next(iter(test.values()))["cd_members"]),
        "temperature": temp,
        "cd_mae_counts": float(np.abs(e_test).mean() * 1000),
        "raw": sigma_row(s_test, e_test, n_boot, seed),
        "calibrated": sigma_row(temp * s_test, e_test, n_boot, seed),
        "val_calibrated": sigma_row(temp * s_val, e_val, n_boot, seed),
    }
