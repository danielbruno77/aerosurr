"""Out-of-distribution check (SPEC §5.4): does the ensemble's sigma rise on unfamiliar
geometries?

Input is an `evaluate_ensemble` result calibrated on the in-distribution side (the OOD
split's val cars, drawn from the training-side population) and reported on its OOD side
(the test cars: the most isolated STL surfaces). Both sides carry per-car statistics:
cell-mean uncalibrated sigma and |err| per field, and each member's integrated Cd.

Sigma is compared after calibration, so a ratio above 1 means sigma itself grew, not the
temperature. Geometric novelty is the dataset's own `ood_score` (mean symmetric Chamfer
RMS distance to the 10 nearest public geometries, `splits/chamfer_metrics.csv`).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosurr.eval.ensemble import _spearman_ci, sigma_row
from aerosurr.eval.metrics import FIELD_NAMES


def read_novelty(path: Path, column: str = "ood_score") -> dict[str, float]:
    """`run_N` -> the geometric-novelty score of `chamfer_metrics.csv`."""
    with path.open() as f:
        return {f"run_{int(row['run'])}": float(row[column]) for row in csv.DictReader(f)}


def auroc(neg: NDArray[np.float64], pos: NDArray[np.float64]) -> float:
    """P(score of a positive > score of a negative), ties counting half (Mann-Whitney U)."""
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def auroc_ci(
    neg: NDArray[np.float64], pos: NDArray[np.float64], n_boot: int, seed: int
) -> tuple[float, float, float]:
    """AUROC and a percentile bootstrap 95 % CI, resampling each group separately."""
    rng = np.random.default_rng(seed)
    boots = [
        auroc(neg[rng.integers(0, len(neg), len(neg))], pos[rng.integers(0, len(pos), len(pos))])
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auroc(neg, pos), float(lo), float(hi)


def _side(pr: dict[str, dict[str, Any]], t_field: dict[str, float]) -> dict[str, Any]:
    """Per-car arrays of one side: calibrated cell-mean sigma and |err|, sigma(Cd), Cd error."""
    runs = list(pr)
    sig = {
        f: np.array([t_field[f] * pr[r]["sigma_cell_mean_raw"][f] for r in runs])
        for f in FIELD_NAMES
    }
    err = {f: np.array([pr[r]["abs_err_cell_mean"][f] for r in runs]) for f in FIELD_NAMES}
    sig_cd = np.array([pr[r]["cd_std_cal"] for r in runs])
    cd_err = np.array([pr[r]["cd_mean"] - pr[r]["cd_true"] for r in runs])
    return {"runs": runs, "sig": sig, "err": err, "sig_cd": sig_cd, "cd_err": cd_err}


def _summary(s: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    return {
        "n": len(s["runs"]),
        "sigma_cell_mean": {f: float(v.mean()) for f, v in s["sig"].items()},
        "abs_err_cell_mean": {f: float(v.mean()) for f, v in s["err"].items()},
        "cd_sigma_mean_counts": float(s["sig_cd"].mean() * 1000),
        "cd_mae_counts": float(np.abs(s["cd_err"]).mean() * 1000),
        "cd": sigma_row(s["sig_cd"], s["cd_err"], n_boot, seed),
    }


def _rho(a: NDArray[np.float64], b: NDArray[np.float64], n_boot: int, seed: int) -> dict[str, Any]:
    rho, lo, hi = _spearman_ci(a, b, n_boot, seed)
    return {"spearman": rho, "ci95": [lo, hi], "n": len(a)}


def ood_check(
    res: dict[str, Any], novelty: dict[str, float], n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """ID (calibration side) vs OOD (report side): sigma and error levels and their ratios,
    AUROC of per-car sigma(Cd) for telling OOD cars from ID cars, and Spearman of per-car
    sigma(Cd) (and, for reference, of |Cd error|) against geometric novelty."""
    t_field = res["temperature"]["fields"]
    sid, sood = _side(res["per_run_calib"], t_field), _side(res["per_run"], t_field)
    id_sum, ood_sum = _summary(sid, n_boot, seed), _summary(sood, n_boot, seed)
    ratio = {
        "sigma_cell": {
            f: ood_sum["sigma_cell_mean"][f] / id_sum["sigma_cell_mean"][f] for f in FIELD_NAMES
        },
        "abs_err_cell": {
            f: ood_sum["abs_err_cell_mean"][f] / id_sum["abs_err_cell_mean"][f] for f in FIELD_NAMES
        },
        "cd_sigma": ood_sum["cd_sigma_mean_counts"] / id_sum["cd_sigma_mean_counts"],
        "cd_mae": ood_sum["cd_mae_counts"] / id_sum["cd_mae_counts"],
    }
    au: dict[str, Any] = {}
    for key, a, b in (
        ("cd_sigma", sid["sig_cd"], sood["sig_cd"]),
        ("cp_sigma_cell", sid["sig"]["cp"], sood["sig"]["cp"]),
        ("cd_abs_err", np.abs(sid["cd_err"]), np.abs(sood["cd_err"])),  # what sigma should track
    ):
        v, lo, hi = auroc_ci(a, b, n_boot, seed)
        au[key] = {"auroc": v, "ci95": [lo, hi]}
    nov_ood = np.array([novelty[r] for r in sood["runs"]])
    nov_all = np.concatenate([np.array([novelty[r] for r in sid["runs"]]), nov_ood])
    sig_all = np.concatenate([sid["sig_cd"], sood["sig_cd"]])
    err_all = np.abs(np.concatenate([sid["cd_err"], sood["cd_err"]]))
    nov = {
        "cd_sigma_vs_novelty_ood": _rho(sood["sig_cd"], nov_ood, n_boot, seed),
        "cd_sigma_vs_novelty_all": _rho(sig_all, nov_all, n_boot, seed),
        "cd_abs_err_vs_novelty_ood": _rho(np.abs(sood["cd_err"]), nov_ood, n_boot, seed),
        "cd_abs_err_vs_novelty_all": _rho(err_all, nov_all, n_boot, seed),
        "cp_sigma_vs_novelty_all": _rho(
            np.concatenate([sid["sig"]["cp"], sood["sig"]["cp"]]), nov_all, n_boot, seed
        ),
    }
    return {
        "id": id_sum,
        "ood": ood_sum,
        "ratio_ood_over_id": ratio,
        "auroc_ood_vs_id": au,
        "novelty": nov,
        "novelty_range": {
            "id": [
                float(nov_all[: len(sid["runs"])].min()),
                float(nov_all[: len(sid["runs"])].max()),
            ],
            "ood": [float(nov_ood.min()), float(nov_ood.max())],
        },
        "per_cell_ood": {
            k: res[k]
            for k in ("coverage", "nll", "spearman_point", "spearman_point_ci95", "temperature")
            if k in res
        },
        "n_boot": n_boot,
    }
