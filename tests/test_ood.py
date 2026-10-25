"""OOD check: AUROC, ID/OOD ratios and novelty correlation on synthetic per-car tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aerosurr.eval.metrics import FIELD_NAMES
from aerosurr.eval.ood import auroc, auroc_ci, ood_check, read_novelty


def test_auroc_known_values() -> None:
    assert auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 0.0
    assert auroc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5  # ties count half
    rng = np.random.default_rng(0)
    v, lo, hi = auroc_ci(rng.normal(0, 1, 300), rng.normal(0, 1, 300), 500, 0)
    assert lo < 0.5 < hi and abs(v - 0.5) < 0.07


def _table(
    rng: np.random.Generator, runs: list[str], scale: float, nov: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Cars whose spread and error both grow with `scale` and with their novelty."""
    out = {}
    for r in runs:
        s = scale * (1 + 50 * (nov[r] - 0.008))
        members = (0.3 + 0.001 * s * rng.normal(size=5)).tolist()
        out[r] = {
            "cd_mean": float(np.mean(members)),
            "cd_std_cal": 2.0 * float(np.std(members)),
            "cd_true": float(np.mean(members) + 0.002 * s * rng.normal()),
            "cd_members": members,
            "sigma_cell_mean_raw": {f: 0.01 * s for f in FIELD_NAMES},
            "abs_err_cell_mean": {f: 0.02 * s * (1 + 0.1 * rng.random()) for f in FIELD_NAMES},
        }
    return out


def test_ood_check_detects_rising_sigma() -> None:
    rng = np.random.default_rng(3)
    ids, oods = [f"run_{i}" for i in range(1, 49)], [f"run_{i}" for i in range(100, 197)]
    nov = {r: 0.0075 + 0.0008 * rng.random() for r in ids}
    nov |= {r: 0.0083 + 0.003 * rng.random() for r in oods}
    res = {
        "temperature": {"fields": dict.fromkeys(FIELD_NAMES, 2.0), "cd": 2.0},
        "per_run_calib": _table(rng, ids, 1.0, nov),
        "per_run": _table(rng, oods, 1.5, nov),
        "coverage": {"0.9": dict.fromkeys(FIELD_NAMES, 0.8)},
    }
    chk = ood_check(res, nov, n_boot=300)
    assert chk["id"]["n"] == 48 and chk["ood"]["n"] == 97
    r = chk["ratio_ood_over_id"]
    assert r["sigma_cell"]["cp"] > 1.5  # the scale and the novelty both grow on the OOD side
    assert r["abs_err_cell"]["cp"] == pytest.approx(r["sigma_cell"]["cp"], rel=0.1)
    assert chk["id"]["sigma_cell_mean"]["cp"] == pytest.approx(
        2.0 * np.mean([0.01 * (1 + 50 * (nov[x] - 0.008)) for x in ids])
    )  # calibrated: temperature times the raw cell mean
    au = chk["auroc_ood_vs_id"]["cp_sigma_cell"]
    assert au["auroc"] > 0.95 and au["ci95"][0] > 0.9
    assert chk["auroc_ood_vs_id"]["cd_sigma"]["auroc"] > 0.6
    assert chk["novelty"]["cp_sigma_vs_novelty_all"]["spearman"] > 0.9
    assert chk["per_cell_ood"]["coverage"]["0.9"]["cp"] == 0.8

    from aerosurr.eval.report import _ood_section

    res["coverage"] = {q: dict.fromkeys(FIELD_NAMES, 0.8) for q in ("0.5", "0.9", "0.95")}
    res["nll"] = dict.fromkeys(FIELD_NAMES, -3.0)
    res["spearman_point"] = dict.fromkeys(FIELD_NAMES, 0.6)
    res["spearman_point_ci95"] = {f: [0.5, 0.7] for f in FIELD_NAMES}
    chk = ood_check(res, nov, n_boot=300) | {"calib": "geometry_val", "split": "geometry_test"}
    lines: list[str] = []
    _ood_section(chk, lines.append)
    text = "\n".join(lines)
    assert "`geometry_test` (OOD, 97 cars) vs `geometry_val` (ID, 48 cars)" in text
    assert "σ rises in step with the error" in text
    assert "| mean per-cell σ(Cp) per car | 1.00 [1.00, 1.00] | better than chance |" in text
    assert "| cp (per cell) | 0.80 | 0.80 | 0.80 | -3.00 | 0.60 [0.50, 0.70] |" in text


def test_read_novelty(tmp_path: Path) -> None:
    f = tmp_path / "chamfer_metrics.csv"
    f.write_text("run,mean_10_nn_chamfer,ood_score\n7,0.008,0.008\n12,0.01,0.01\n")
    assert read_novelty(f) == {"run_7": 0.008, "run_12": 0.01}
