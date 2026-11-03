"""Design-ranking metrics on small synthetic inputs (CPU only)."""

from __future__ import annotations

import math

import pytest

from aerosurr.eval.ranking import (
    band_order,
    delta_error,
    lowest_drag_pick,
    ranks,
    render,
    score,
    share_ordered,
    spearman,
)

TRUE = [0.300, 0.301, 0.304, 0.310, 0.330]  # pairwise gaps 1 to 30 counts


def test_spearman() -> None:
    assert ranks([0.3, 0.1, 0.2]) == [2, 0, 1]
    assert spearman(TRUE, TRUE) == 1.0
    assert spearman(TRUE, TRUE[::-1]) == -1.0
    # one adjacent swap among 5: sum d^2 = 2, rho = 1 - 12 / 120
    assert spearman(TRUE, [0.301, 0.300, 0.304, 0.310, 0.330]) == pytest.approx(0.9)


def test_pair_order_per_band() -> None:
    pred = [0.301, 0.300, 0.304, 0.310, 0.330]  # swaps the only pair 1 count apart
    bands = band_order(TRUE, pred)
    # gaps (counts): 1 | 3, 4 | 6, 9 | 10 | 20, 26, 29, 30 -> 10 pairs in all
    assert [b["pairs"] for b in bands] == [1, 2, 2, 1, 4]
    assert bands[0]["share"] == 0.0 and all(b["share"] == 1.0 for b in bands[1:])
    assert math.isinf(bands[4]["hi"])
    assert band_order(TRUE, pred, [(50, 60)])[0] == {"lo": 50, "hi": 60, "pairs": 0, "share": None}
    assert share_ordered(TRUE, pred, 5.0) == 1.0
    assert share_ordered(TRUE, TRUE[::-1], 5.0) == 0.0


def test_delta_error() -> None:
    assert delta_error(TRUE, TRUE) == {"mean_counts": 0.0, "p90_counts": 0.0}
    shifted = [t + 0.002 for t in TRUE]  # a constant bias cancels in every difference
    assert delta_error(TRUE, shifted)["mean_counts"] == pytest.approx(0.0, abs=1e-9)
    one = [*TRUE[:4], TRUE[4] + 0.004]  # car 4 off by 4 counts: 4 of 10 pairs see it
    de = delta_error(TRUE, one)
    assert de["mean_counts"] == pytest.approx(1.6) and de["p90_counts"] == pytest.approx(4.0)


def test_lowest_drag_pick() -> None:
    runs = [f"run_{i}" for i in range(1, 6)]
    right = lowest_drag_pick(runs, TRUE, TRUE)
    assert right == {
        "run": "run_1",
        "true_best": "run_1",
        "true_rank": 1,
        "counts_off": 0.0,
        "top5_overlap": 5,
    }
    wrong = lowest_drag_pick(runs, TRUE, [0.305, 0.299, 0.304, 0.310, 0.330])
    assert wrong["run"] == "run_2" and wrong["true_rank"] == 2
    assert wrong["counts_off"] == pytest.approx(1.0)


def test_score_and_render() -> None:
    runs = [f"run_{i}" for i in range(1, 6)]
    s = score(runs, TRUE, [t + 0.001 for t in TRUE])
    assert s["cd_bias_counts"] == pytest.approx(1.0) and s["spearman"] == 1.0
    res = {
        "n_cars": 5,
        "n_pairs": 10,
        "true_cd_range_counts": [300.0, 330.0],
        "predictors": {"Model": s},
        "coefficients": {"Model, other coefficients": {"cl": {"mae_counts": 1.0, "spearman": 0.9}}},
        "rans_cost": {
            "cases": 1,
            "median_min_per_case": 12.0,
            "median_cells_m": 2.0,
            "ranks": 24,
            "median_cd_std_counts": 5.0,
        },
    }
    text = render(res)
    assert "== Model: Cd MAE 1.00 counts, bias +1.00, Spearman 1.000" in text
    assert "   band 20-inf counts: 4 pairs, 100.0% right" in text
