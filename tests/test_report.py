"""The results report renders every section from synthetic evaluation files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aerosurr.data.store import StoreEntry, save_index
from aerosurr.eval.report import build_report, scalar_correction
from aerosurr.rans.mapping import LF_DIR


def fake_eval(rng: np.random.Generator, runs: list[str], noise: float) -> dict:  # type: ignore[type-arg]
    per = {}
    for r in runs:
        cd, cl = 0.25 + 0.08 * rng.random(), 0.05 + 0.1 * rng.random()
        per[r] = {
            "rel_l2": {"cp": 0.1, "cf_x": 0.12, "cf_y": 0.2, "cf_z": 0.2},
            "true": {"cd": cd, "cl": cl},
            "pred": {"cd": cd + noise * rng.normal(), "cl": cl + noise * rng.normal()},
        }
    return {
        "runs": len(runs),
        "rel_l2": {"cp": 0.1, "cf_x": 0.12, "cf_y": 0.2, "cf_z": 0.2},
        "per_run": per,
    }


def test_build_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    runs_dir, store = tmp_path / "runs", tmp_path / "store"
    test = [f"run_{i}" for i in range(10, 20)]
    train = [f"run_{i}" for i in range(1, 9)]

    def put(name: str, payload: dict, file: str = "full_test.json") -> None:  # type: ignore[type-arg]
        (runs_dir / name).mkdir(parents=True, exist_ok=True)
        (runs_dir / name / file).write_text(json.dumps(payload))

    put("pointnet_baseline_v1", fake_eval(rng, test, 0.02))
    put("transolver_main_v1", fake_eval(rng, test, 0.005))
    put("transolver_main_v1", {"final_full_val": {"cd_mae_counts": 3.0}}, "summary.json")
    for split in ("super_scarce", "scarce", "medium", "full"):
        put(f"transolver_ds_{split}", fake_eval(rng, test, 0.01))
    for n in (11, 25, 50):
        put(f"mf_n{n}", fake_eval(rng, test, 0.004))
        put(f"hf_n{n}", fake_eval(rng, test, 0.008))
        put(f"mf_n{n}", {"train_runs": train}, "config.json")
    cov = {q: {f: float(q) for f in ("cp", "cf_x", "cf_y", "cf_z")} for q in ("0.5", "0.9", "0.95")}
    cd_cov = {"0.5": 0.5, "0.9": 0.88, "0.95": 0.94}
    put(
        "ensemble_v1",
        {
            "members": 5,
            "coverage": cov,
            "cd_coverage": cd_cov,
            "nll": {f: -1.0 for f in ("cp", "cf_x", "cf_y", "cf_z")},
            "spearman_point": {f: 0.4 for f in ("cp", "cf_x", "cf_y", "cf_z")},
            "cd_spearman_car": 0.5,
            "cd_mae_counts": 4.0,
            "cd_sigma_mean_counts": 4.5,
            "aleatoric_fraction": 0.6,
            "per_run": {r: {"cd_std_cal": 0.004} for r in test},
        },
        "ensemble_full_test.json",
    )
    coeffs = {"varying": {"cd": 0.3}, "constant": {"cd": 0.3}}
    store.mkdir(parents=True)
    save_index(
        store,
        {
            r: StoreEntry(r, 1, 1, False, f"{r}.npz", 0.0, coeffs, coeffs, True)
            for r in train + test
        },
        {},
    )
    (store / LF_DIR).mkdir()
    (store / LF_DIR / "index.json").write_text(
        json.dumps({r: {"coeffs": {"Cd": 0.28}} for r in train + test})
    )
    (store / LF_DIR / "eval_full_test.json").write_text(
        json.dumps({"runs": 10, "rel_l2": {"cp": 0.3}, "cd_mae_counts": 20.0})
    )

    out = tmp_path / "docs" / "results.md"
    done = build_report(runs_dir, store, out, tmp_path / "docs" / "figures")
    assert done == ["models", "data-efficiency", "multi-fidelity", "ensemble"]
    text = out.read_text()
    for token in ("PhysicsNeMo Transolver", "| 400 |", "LF alone", "Scalar Cd_HF", "Cd (per car)"):
        assert token in text, token
    assert "selected on `full_val`" in text
    for png in ("parity_main", "data_efficiency_cd", "mf_cd", "calibration"):
        assert (tmp_path / "docs" / "figures" / f"{png}.png").stat().st_size > 10_000


def test_scalar_correction_recovers_a_linear_map() -> None:
    lf = {f"r{i}": {"coeffs": {"Cd": 0.2 + 0.01 * i}} for i in range(10)}

    def entry(k: str, cd_lf: float) -> StoreEntry:
        return StoreEntry(
            k, 1, 1, False, "", 0.0, {"varying": {"cd": 0.02 + 1.1 * cd_lf}}, {}, True
        )

    hf = {k: entry(k, v["coeffs"]["Cd"]) for k, v in lf.items()}
    res = scalar_correction(lf, hf, [f"r{i}" for i in range(6)], [f"r{i}" for i in range(6, 10)])
    assert abs(res["b"] - 1.1) < 1e-9 and res["cd_mae_counts"] < 1e-6 < res["cd_raw_lf_mae_counts"]
