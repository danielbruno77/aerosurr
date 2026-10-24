from __future__ import annotations

import numpy as np
import pytest
import torch

from aerosurr.eval.ensemble import combine, coverage, fit_temperature, gaussian_nll, spearman


def test_combine_moments() -> None:
    m1, m2 = torch.zeros(4, 2), torch.ones(4, 2) * 2
    v = torch.full((4, 2), 0.5)
    c = combine([m1, m2], [v, v])
    assert torch.allclose(c.mean, torch.ones(4, 2, dtype=torch.float64))
    assert torch.allclose(c.var_epistemic, torch.ones(4, 2, dtype=torch.float64))  # var of {0,2}
    assert torch.allclose(c.var, torch.full((4, 2), 1.5, dtype=torch.float64))
    single = combine([m1], [None])
    assert float(single.var.abs().max()) == 0.0


def test_temperature_and_coverage_on_gaussian_errors() -> None:
    g = torch.Generator().manual_seed(0)
    sigma_true = torch.tensor([0.5, 2.0])
    err = torch.randn(200_000, 2, generator=g) * sigma_true
    var_pred = torch.ones(200_000, 2)  # wrong scale on purpose
    t = fit_temperature(err, var_pred)
    np.testing.assert_allclose(t.numpy(), sigma_true.numpy(), rtol=0.01)
    sigma = t.float() * var_pred.sqrt()
    for q in (0.5, 0.9, 0.95):
        np.testing.assert_allclose(coverage(err, sigma, q), [q, q], atol=0.005)
    nll_good = gaussian_nll(err, sigma**2)
    nll_bad = gaussian_nll(err, var_pred)
    assert (nll_good < nll_bad).all()


def test_spearman() -> None:
    a = np.arange(100.0)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    r = spearman(
        np.random.default_rng(0).normal(size=5000), np.random.default_rng(1).normal(size=5000)
    )
    assert abs(r) < 0.05
