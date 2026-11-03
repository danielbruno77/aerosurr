"""Design ranking (SPEC §5.5): does a predictor put cars in the right drag order?

For design work the absolute Cd matters less than the order of candidates and the size
of a drag difference. On the test cars this module scores a predictor (GeoTransolver,
the ensemble mean, coarse RANS) against the HRLES Cd:

- Spearman rank correlation of predicted against true Cd;
- the share of car pairs whose predicted order matches the true order, overall for pairs
  at least 5 counts apart and per band of true |dCd|;
- the error on a drag difference, |(p_i - p_j) - (t_i - t_j)| over all pairs;
- the lowest-drag pick: the true rank of the predicted best car, how far it is from the
  true best, and the overlap of the predicted and true top 5.

One drag count is 0.001 in Cd. Ties in the true or predicted Cd do not occur in this data;
a tied pair counts as wrongly ordered.
"""

from __future__ import annotations

import itertools
import math
import statistics as st
from collections.abc import Sequence
from typing import Any

COUNTS = 1000.0
BANDS: tuple[tuple[float, float], ...] = ((0, 2), (2, 5), (5, 10), (10, 20), (20, math.inf))


def ranks(v: Sequence[float]) -> list[int]:
    """0-based rank of each value (ties broken by position)."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0] * len(v)
    for k, i in enumerate(order):
        out[i] = k
    return out


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman's rho, 1 - 6 sum d^2 / (n (n^2 - 1)) (exact without ties)."""
    ra, rb, n = ranks(a), ranks(b), len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb, strict=True)) / (n * (n * n - 1))


def _pairs(n: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(n), 2))


def _right(true: Sequence[float], pred: Sequence[float], i: int, j: int) -> bool:
    return (pred[i] - pred[j]) * (true[i] - true[j]) > 0


def share_ordered(true: Sequence[float], pred: Sequence[float], min_counts: float) -> float:
    """Share of pairs at least `min_counts` apart (true) that the prediction orders right."""
    sel = [(i, j) for i, j in _pairs(len(true)) if abs(true[i] - true[j]) * COUNTS >= min_counts]
    return sum(_right(true, pred, i, j) for i, j in sel) / len(sel)


def band_order(
    true: Sequence[float], pred: Sequence[float], bands: Sequence[tuple[float, float]] = BANDS
) -> list[dict[str, Any]]:
    """Per band lo <= |dCd_true| < hi (counts): number of pairs and the share ordered right."""
    out = []
    for lo, hi in bands:
        sel = [(i, j) for i, j in _pairs(len(true)) if lo <= abs(true[i] - true[j]) * COUNTS < hi]
        ok = sum(_right(true, pred, i, j) for i, j in sel)
        out.append({"lo": lo, "hi": hi, "pairs": len(sel), "share": ok / len(sel) if sel else None})
    return out


def delta_error(true: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    """Error on a drag difference over all pairs (counts): mean and the 90th percentile
    (the value at index floor(0.9 n) of the sorted errors)."""
    d = sorted(
        abs((pred[i] - pred[j]) - (true[i] - true[j])) * COUNTS for i, j in _pairs(len(true))
    )
    return {"mean_counts": st.mean(d), "p90_counts": d[int(0.9 * len(d))]}


def lowest_drag_pick(
    runs: Sequence[str], true: Sequence[float], pred: Sequence[float]
) -> dict[str, Any]:
    """The predicted lowest-drag car: its true rank (1 = best), its distance from the true
    best (counts), and how many of the predicted top 5 are in the true top 5."""
    by_true = sorted(range(len(runs)), key=lambda i: true[i])
    by_pred = sorted(range(len(runs)), key=lambda i: pred[i])
    pick = min(range(len(runs)), key=lambda i: pred[i])
    return {
        "run": runs[pick],
        "true_best": runs[by_true[0]],
        "true_rank": by_true.index(pick) + 1,
        "counts_off": (true[pick] - true[by_true[0]]) * COUNTS,
        "top5_overlap": len(set(by_pred[:5]) & set(by_true[:5])),
    }


def coeff_row(true: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    """MAE (counts) and Spearman of one coefficient."""
    return {
        "mae_counts": st.mean(abs(a - b) * COUNTS for a, b in zip(pred, true, strict=True)),
        "spearman": spearman(pred, true),
    }


def score(runs: Sequence[str], true: Sequence[float], pred: Sequence[float]) -> dict[str, Any]:
    """Every ranking metric of one Cd predictor."""
    err = [COUNTS * (a - b) for a, b in zip(pred, true, strict=True)]
    return {
        "cd_mae_counts": st.mean(map(abs, err)),
        "cd_bias_counts": st.mean(err),
        "spearman": spearman(pred, true),
        "pairs_ge5_ordered": share_ordered(true, pred, 5.0),
        "bands": band_order(true, pred),
        "delta_error": delta_error(true, pred),
        "pick": lowest_drag_pick(runs, true, pred),
    }


def render(res: dict[str, Any]) -> str:
    """The plain-text summary (one block per predictor, then the other coefficients)."""
    lines = [
        f"{res['n_cars']} test cars, {res['n_pairs']} pairs; true Cd "
        f"{res['true_cd_range_counts'][0]:.1f}-{res['true_cd_range_counts'][1]:.1f} counts"
    ]
    for name, s in res["predictors"].items():
        lines += [
            "",
            f"== {name}: Cd MAE {s['cd_mae_counts']:.2f} counts, bias "
            f"{s['cd_bias_counts']:+.2f}, Spearman {s['spearman']:.3f}",
            f"   pairs >= 5 counts in the right order: {100 * s['pairs_ge5_ordered']:.1f}%",
        ]
        for b in s["bands"]:
            hi = "inf" if math.isinf(b["hi"]) else f"{b['hi']:g}"
            lines.append(
                f"   band {b['lo']:g}-{hi} counts: {b['pairs']} pairs, "
                f"{100 * b['share']:.1f}% right"
            )
        de, pk = s["delta_error"], s["pick"]
        lines += [
            f"   error on a drag difference: mean {de['mean_counts']:.2f}, p90 "
            f"{de['p90_counts']:.2f} counts",
            f"   lowest-drag pick {pk['run']}: true rank {pk['true_rank']}, "
            f"{pk['counts_off']:.2f} counts off; top-5 overlap {pk['top5_overlap']}/5",
        ]
    for title, rows in res["coefficients"].items():
        lines += ["", f"== {title} (MAE in counts, Spearman)"]
        for q, r in rows.items():
            lines.append(f"   {q:12s} {r['mae_counts']:6.2f}  {r['spearman']:.3f}")
    c = res["rans_cost"]
    lines += [
        "",
        f"== RANS cost over {c['cases']} cases: median {c['median_min_per_case']:.1f} min per "
        f"case, {c['median_cells_m']:.2f} M cells, {c['ranks']} ranks; median Cd std over the "
        f"averaging window {c['median_cd_std_counts']:.1f} counts",
    ]
    return "\n".join(lines) + "\n"
