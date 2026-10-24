"""Training loop smoke tests on a tiny synthetic store (CPU, no compile)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aerosurr.config import ModelConfig, TrainConfig
from aerosurr.data.store import compute_stats, load_group, pool_indices
from aerosurr.models.surface import build_model
from aerosurr.train.data import Normaliser, PoolSet
from aerosurr.train.loop import loss_fn, lr_at, predict, train

from .test_rans import GEO_REF

N_CELLS, POOL = 300, 200


def fake_run(store: Path, raw: Path, run: str, seed: int, full: bool) -> None:
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(N_CELLS, 3)).astype(np.float32)
    nrm = c / np.linalg.norm(c, axis=1, keepdims=True)
    cp = (0.5 * c[:, 0] - 0.2).astype(np.float16)  # a learnable function of position
    cf = (0.003 * nrm).astype(np.float16)
    idx = pool_indices(N_CELLS, POOL, seed)
    arrays = {"pool_idx": idx}
    groups = {"pool": idx, "full": slice(None)} if full else {"pool": idx}
    for g, sel in groups.items():
        arrays |= {f"{g}_centres": c[sel], f"{g}_normals": nrm[sel],
                   f"{g}_areas": np.full(N_CELLS, 0.01, np.float32)[sel],
                   f"{g}_cp": cp[sel], f"{g}_cf": cf[sel]}  # fmt: skip
    store.mkdir(parents=True, exist_ok=True)
    np.savez(store / f"{run}.npz", **arrays)
    (raw / run).mkdir(parents=True, exist_ok=True)
    (raw / run / f"geo_ref_{run.split('_')[1]}.csv").write_text(GEO_REF)


@pytest.fixture
def tiny(tmp_path: Path) -> tuple[Path, Path, dict[str, list[str]]]:
    store, raw = tmp_path / "store", tmp_path / "raw"
    manifest = {"full_train": ["run_1", "run_2", "run_3"], "full_val": ["run_4"]}
    for i, r in enumerate(manifest["full_train"] + manifest["full_val"]):
        fake_run(store, raw, r, i, full=r in manifest["full_val"])
    return store, raw, manifest


def test_pool_indices_deterministic() -> None:
    a, b = pool_indices(1000, 10, 7), pool_indices(1000, 10, 7)
    assert np.array_equal(a, b) and len(np.unique(a)) == 10 and a.dtype == np.int32
    assert len(pool_indices(5, 10, 0)) == 5


def test_stats_and_normaliser(tiny: tuple[Path, Path, dict[str, list[str]]]) -> None:
    store, _, m = tiny
    s = compute_stats(store, m["full_train"])
    assert s["points"] == 3 * POOL and s["pos_scale"] > 0
    norm = Normaliser.from_stats(s)
    g = load_group(store, "run_1", "pool")
    y = norm.y(g["cp"], g["cf"])
    assert y.shape == (POOL, 4)
    back = norm.y_torch_inv(torch.from_numpy(y)).numpy()
    np.testing.assert_allclose(back[:, 0], g["cp"].astype(np.float32), atol=1e-5)
    pools = PoolSet(store, m["full_train"], norm, torch.device("cpu"))
    b = pools.sample(0, 50, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert b.pos.shape == (1, 50, 3) and b.y.shape == (1, 50, 4) and b.area.shape == (1, 50)
    assert b.feat.shape == (1, 50, 3) and b.lf is None


@pytest.mark.parametrize("kind", ["pointnet", "transolver"])
def test_models_forward(kind: str) -> None:
    cfg = ModelConfig(kind=kind, n_layers=2, n_hidden=32, n_head=2, slice_num=4)  # type: ignore[arg-type]
    m = build_model(cfg)
    out = m(torch.randn(2, 64, 3), torch.randn(2, 64, 3))
    assert out.shape == (2, 64, 4)
    het = build_model(cfg.model_copy(update={"heteroscedastic": True}))
    assert het(torch.randn(1, 16, 3), torch.randn(1, 16, 3)).shape == (1, 16, 8)


def test_loss_and_schedule() -> None:
    y = torch.zeros(1, 10, 4)
    assert float(loss_fn(torch.ones(1, 10, 4), y, False)) == pytest.approx(1.0)
    nll = loss_fn(torch.zeros(1, 10, 8), y, True)  # mu = 0, logvar = 0
    assert float(nll) == pytest.approx(0.0)
    assert lr_at(0, 100, 10, 1.0) == pytest.approx(0.1)
    assert lr_at(10, 100, 10, 1.0) == pytest.approx(1.0)
    assert lr_at(100, 100, 10, 1.0) == pytest.approx(0.01)


def test_predict_covers_every_cell() -> None:
    norm = Normaliser((0, 0, 0), 1.0, (0, 0, 0, 0), (1, 1, 1, 1))

    class Echo(torch.nn.Module):
        def forward(self, pos: torch.Tensor, nrm: torch.Tensor) -> torch.Tensor:
            return torch.cat([pos[..., :1], nrm], dim=-1)

    c = np.arange(30, dtype=np.float32).reshape(10, 3)
    pred, var = predict(Echo(), norm, c, c, chunk=4, device=torch.device("cpu"), bf16=False)
    np.testing.assert_allclose(pred[:, 0], c[:, 0])  # every cell predicted at its own index
    assert var is None


@pytest.mark.parametrize("kind", ["pointnet", "transolver"])
def test_train_end_to_end(tiny: tuple[Path, Path, dict[str, list[str]]], tmp_path: Path,
                          kind: str) -> None:  # fmt: skip
    store, raw, m = tiny
    stats = compute_stats(store, m["full_train"])
    mc = ModelConfig(kind=kind, n_layers=1, n_hidden=16, n_head=2, slice_num=4)  # type: ignore[arg-type]
    tc = TrainConfig(epochs=2, points=64, compile=False, bf16=False, data_on_gpu=False,
                     eval_every=1, full_eval_every=2, warmup_epochs=1)  # fmt: skip
    out = tmp_path / "run"
    s = train("t", mc, tc, m, store, raw, stats, out, torch.device("cpu"))
    assert s.epochs_done == 2 and s.steps == 6 and s.best_epoch in (1, 2)
    assert (out / "best.pt").is_file() and (out / "last.pt").is_file()
    lines = (out / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2 and "full_val" in json.loads(lines[1])
    full = json.loads((out / "full_val.json").read_text())
    assert full["runs"] == 1 and set(full["rel_l2"]) == {"cp", "cf_x", "cf_y", "cf_z"}
    assert np.isfinite(full["cd_mae_counts"])


def add_fake_lf(store: Path, run: str) -> None:
    """LF = HF + a smooth bias, as `rans map` would write it."""
    from aerosurr.rans.mapping import LF_DIR

    out = {}
    with np.load(store / f"{run}.npz") as z:
        for g in ("pool", "full"):
            if f"{g}_cp" in z.files:
                hf = np.column_stack([z[f"{g}_cp"], z[f"{g}_cf"]]).astype(np.float32)
                out[f"{g}_lf"] = (hf + 0.1).astype(np.float16)
    (store / LF_DIR).mkdir(exist_ok=True)
    np.savez(store / LF_DIR / f"{run}.npz", **out)


def test_mf_residual_training(
    tiny: tuple[Path, Path, dict[str, list[str]]], tmp_path: Path
) -> None:
    store, raw, m = tiny
    for r in m["full_train"] + m["full_val"]:
        add_fake_lf(store, r)
    stats = compute_stats(store, m["full_train"])
    norm = Normaliser.from_stats(stats)
    pools = PoolSet(store, m["full_train"], norm, torch.device("cpu"), lf=True)
    b = pools.sample(0, 32, torch.Generator().manual_seed(0), torch.device("cpu"))
    assert b.feat.shape == (1, 32, 7) and b.lf is not None and b.lf.shape == (1, 32, 4)
    mc = ModelConfig(kind="transolver", n_layers=1, n_hidden=16, n_head=2, slice_num=4,  # type: ignore[arg-type]
                     lf_inputs=True, predict_residual=True)  # fmt: skip
    assert mc.feature_dim == 7
    tc = TrainConfig(epochs=1, points=64, compile=False, bf16=False, data_on_gpu=False,
                     eval_every=1, full_eval_every=1, warmup_epochs=0,
                     train_runs=["run_1", "run_2"])  # fmt: skip
    s = train("mf", mc, tc, m, store, raw, stats, tmp_path / "mf", torch.device("cpu"))
    assert s.steps == 2  # train_runs overrides the split
    full = json.loads((tmp_path / "mf" / "full_val.json").read_text())
    assert np.isfinite(full["rel_l2"]["cp"])
    with pytest.raises(ValueError, match="needs lf_inputs"):
        ModelConfig(kind="pointnet", predict_residual=True)  # type: ignore[arg-type]


def test_ensemble_end_to_end(tiny: tuple[Path, Path, dict[str, list[str]]], tmp_path: Path) -> None:
    from aerosurr.data.store import StoreEntry, save_index
    from aerosurr.eval.ensemble import evaluate_ensemble
    from aerosurr.models.surface import build_model

    store, raw, m = tiny
    coeffs = {"varying": {"cd": 0.3, "cl": 0.1}, "constant": {"cd": 0.3, "cl": 0.1}}
    save_index(store, {"run_4": StoreEntry("run_4", N_CELLS, POOL, True, "run_4.npz", 0.0,
                                           coeffs, coeffs, True)}, {})  # fmt: skip
    stats = compute_stats(store, m["full_train"])
    mc = ModelConfig(kind="pointnet", heteroscedastic=True)  # type: ignore[arg-type]
    tc = TrainConfig(
        epochs=1,
        points=64,
        compile=False,
        bf16=False,
        data_on_gpu=False,
        eval_every=1,
        full_eval_every=1,
        warmup_epochs=0,
    )  # periodic full eval too
    members = []
    for seed in (0, 1):
        out = tmp_path / f"m{seed}"
        train(f"m{seed}", mc, tc.model_copy(update={"seed": seed}), m, store, raw, stats, out,
              torch.device("cpu"))  # fmt: skip
        model = build_model(mc)
        model.load_state_dict(torch.load(out / "best.pt"))
        members.append((model.eval(), mc))
    res = evaluate_ensemble(
        members,
        Normaliser.from_stats(stats),
        store,
        raw,
        ["run_4"],
        ["run_4"],
        64,
        False,
        torch.device("cpu"),
        sample_per_run=50,
    )
    assert res["members"] == 2 and 0.0 < res["aleatoric_fraction"] <= 1.0
    for q in ("0.5", "0.9", "0.95"):
        assert 0.0 <= res["coverage"][q]["cp"] <= 1.0
    assert res["temperature"]["fields"]["cp"] > 0 and np.isfinite(res["cd_mae_counts"])


def test_mf_without_val_lf(tiny: tuple[Path, Path, dict[str, list[str]]], tmp_path: Path) -> None:
    """MF study: no LF on the val cars, so no selection; the final weights are kept."""
    store, raw, m = tiny
    for r in m["full_train"]:
        add_fake_lf(store, r)  # val (run_4) deliberately has none
    stats = compute_stats(store, m["full_train"])
    mc = ModelConfig(kind="pointnet", lf_inputs=True, predict_residual=True)  # type: ignore[arg-type]
    tc = TrainConfig(epochs=2, points=64, compile=False, bf16=False, data_on_gpu=False,
                     select_on_val=False)  # fmt: skip
    s = train("mf", mc, tc, m, store, raw, stats, tmp_path / "mf", torch.device("cpu"))
    assert s.best_epoch == 2 and s.final_full_val is None
    assert (tmp_path / "mf" / "best.pt").is_file()
