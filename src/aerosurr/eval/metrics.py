"""Surface and force metrics (SPEC §5.5). Fields are area-weighted on the native mesh."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FIELD_NAMES = ("cp", "cf_x", "cf_y", "cf_z")


def relative_l2(
    pred: ArrayLike, true: ArrayLike, weights: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Per-field sqrt(sum w (p - t)^2 / sum w t^2) for (n, k) arrays."""
    p = np.asarray(pred, dtype=np.float64).reshape(len(np.asarray(pred)), -1)
    t = np.asarray(true, dtype=np.float64).reshape(p.shape)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    num = (w[:, None] * (p - t) ** 2).sum(axis=0)
    den = (w[:, None] * t**2).sum(axis=0)
    out: NDArray[np.float64] = np.sqrt(num / np.maximum(den, 1e-300))
    return out


def mae(pred: ArrayLike, true: ArrayLike, weights: ArrayLike | None = None) -> NDArray[np.float64]:
    """Per-field area-weighted mean absolute error."""
    p = np.asarray(pred, dtype=np.float64).reshape(len(np.asarray(pred)), -1)
    t = np.asarray(true, dtype=np.float64).reshape(p.shape)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    out: NDArray[np.float64] = (w[:, None] * np.abs(p - t)).sum(axis=0) / w.sum()
    return out


def r2(pred: ArrayLike, true: ArrayLike) -> float:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
