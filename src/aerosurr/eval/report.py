"""Results report (SPEC §6): tables and figures from whatever evaluations exist.

Every section reads JSON written by `aerosurr eval run / eval ensemble / eval lf` and the
store index, and is skipped when its inputs are not there yet, so the report can be
regenerated at any point of the project.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Chart tokens: one light theme (the PNGs are embedded in markdown), the reference
# categorical slots 1-3 (validated all-pairs), recessive grid, text in text ink.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3de"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
MUTED = "#8d8c86"


@dataclass(frozen=True)
class ModelResult:
    name: str
    label: str
    eval: dict[str, Any]


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _cd_arrays(ev: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    runs = ev["per_run"]
    true = np.array([v["true"]["cd"] for v in runs.values()])
    pred = np.array([v["pred"]["cd"] for v in runs.values()])
    return true, pred


def summary_row(label: str, ev: dict[str, Any]) -> str:
    t, p = _cd_arrays(ev)
    cl_t = np.array([v["true"]["cl"] for v in ev["per_run"].values()])
    cl_p = np.array([v["pred"]["cl"] for v in ev["per_run"].values()])
    r2 = 1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    rl = ev["rel_l2"]
    return (
        f"| {label} | {ev['runs']} | {rl['cp']:.3f} | {rl['cf_x']:.3f} | {rl['cf_z']:.3f} | "
        f"{np.abs(p - t).mean() * 1000:.1f} | {np.abs(p - t).max() * 1000:.1f} | {r2:.3f} | "
        f"{np.abs(cl_p - cl_t).mean():.4f} |"
    )


TABLE_HEAD = (
    "| Model | Test cars | Cp rel L2 | Cf_x rel L2 | Cf_z rel L2 | Cd MAE (counts) | "
    "Cd max (counts) | Cd R² | Cl MAE |\n|---|---|---|---|---|---|---|---|---|"
)


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)
    ax.title.set_color(TEXT)


def fig_parity(ev: dict[str, Any], sigma: dict[str, float] | None, out: Path, title: str) -> Path:
    """Cd and Cl parity on the test cars, as two charts (one axis each)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), facecolor=SURFACE)
    runs = list(ev["per_run"])
    for ax, key, unit in ((axes[0], "cd", "Cd"), (axes[1], "cl", "Cl")):
        t = np.array([ev["per_run"][r]["true"][key] for r in runs])
        p = np.array([ev["per_run"][r]["pred"][key] for r in runs])
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1.2, zorder=1)
        if key == "cd":
            ax.fill_between(
                [lo - pad, hi + pad],
                [lo - pad - 0.005, hi + pad - 0.005],
                [lo - pad + 0.005, hi + pad + 0.005],
                color=GRID,
                zorder=0,
                label="±5 counts",
            )
        yerr = None
        if key == "cd" and sigma:
            yerr = np.array([sigma.get(r, 0.0) for r in runs])
        ax.errorbar(
            t,
            p,
            yerr=yerr,
            fmt="o",
            ms=5.5,
            color=SERIES[0],
            ecolor=SERIES[0],
            elinewidth=1.2,
            capsize=0,
            mec=SURFACE,
            mew=1.0,
            zorder=3,
        )
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{unit} HRLES")
        ax.set_ylabel(f"{unit} predicted")
        mae = np.abs(p - t).mean()
        label = (
            f"{unit}: MAE {mae * 1000:.1f} counts (band ±5)"
            if key == "cd"
            else f"{unit}: MAE {mae:.4f}"
        )
        ax.set_title(label, fontsize=11, loc="left")
        _style(ax)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_lines(
    series: dict[str, list[tuple[int, float]]],
    out: Path,
    title: str,
    ylabel: str,
    reference: tuple[str, float] | None = None,
    bars: dict[str, list[tuple[int, float, float]]] | None = None,
) -> Path:
    """Error versus number of HF training runs (log x); up to 3 series, direct-labelled.
    `bars`: per series, (x, low, high) drawn as vertical range bars (e.g. min-max over seeds)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, NullLocator

    fig, ax = plt.subplots(figsize=(7.0, 4.3), facecolor=SURFACE)
    xs = sorted({p[0] for pts in series.values() for p in pts})
    ax.set_xscale("log")
    ax.set_xlim(xs[0] / 1.25, xs[-1] * 2.6)  # room on the right for the direct labels
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(xs)
    ax.set_xticklabels([str(v) for v in xs])
    bars = bars or {}
    highs = [hi for b in bars.values() for _, _, hi in b]
    ymax = max(
        [v for pts in series.values() for _, v in pts] + highs + [reference[1] if reference else 0]
    )
    ax.set_ylim(0, ymax * 1.15)
    for k, ((name, pts), color) in enumerate(zip(series.items(), SERIES, strict=False)):
        pts = sorted(pts)
        shift = 1.0 + 0.035 * (k - (len(series) - 1) / 2) if bars else 1.0  # keep bars apart
        x, y = [p[0] * shift for p in pts], [p[1] for p in pts]
        if name in bars:
            for bx, lo, hi in bars[name]:
                ax.plot([bx * shift] * 2, [lo, hi], color=color, linewidth=1.5, alpha=0.6)
                ax.plot([bx * shift] * 2, [lo, hi], "_", color=color, ms=8, alpha=0.6)
        ax.plot(x, y, "-o", color=color, linewidth=2, ms=7, mec=SURFACE, mew=1.5, label=name)
        ax.annotate(
            name.split(" (")[0],
            (x[-1], y[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_2,
        )
    if reference:
        ax.axhline(reference[1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.annotate(
            f"{reference[0]}: {reference[1]:.1f}",
            (xs[0] / 1.2, reference[1]),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_2,
        )
    ax.set_xlabel("HF training runs")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    if len(series) >= 2:
        ax.legend(
            frameon=False,
            fontsize=9,
            labelcolor=TEXT_2,
            loc="upper center",
            bbox_to_anchor=(0.45, -0.17),
            ncol=len(series),
        )
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_calibration(ens: dict[str, Any], out: Path) -> Path:
    """Observed against nominal coverage for Cp, Cf_x and Cd (diagonal = calibrated)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = [0.5, 0.9, 0.95]
    fig, ax = plt.subplots(figsize=(4.6, 4.4), facecolor=SURFACE)
    ax.plot([0.4, 1.0], [0.4, 1.0], color=MUTED, linewidth=1.2)
    curves = {
        "Cp (per cell)": [ens["coverage"][str(v)]["cp"] for v in q],
        "Cf_x (per cell)": [ens["coverage"][str(v)]["cf_x"] for v in q],
        "Cd (per car)": [ens["cd_coverage"][str(v)] for v in q],
    }
    for (name, ys), color in zip(curves.items(), SERIES, strict=True):
        ax.plot(q, ys, "-o", color=color, linewidth=2, ms=7, mec=SURFACE, mew=1.5, label=name)
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("observed coverage (test)")
    ax.set_title("Ensemble calibration", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def scalar_correction(
    lf_index: dict[str, Any], store_index: dict[str, Any], train: list[str], test: list[str]
) -> dict[str, float]:
    """Classical baseline (SPEC §5.3, model 4): Cd_HF = a + b Cd_LF fitted on `train`."""

    def pairs(runs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        lf = np.array([lf_index[r]["coeffs"]["Cd"] for r in runs])
        hf = np.array([store_index[r].coeffs["varying"]["cd"] for r in runs])
        return lf, hf

    x, y = pairs(train)
    if len(train) > 1 and np.ptp(x) > 1e-9:
        b, a = np.polyfit(x, y, 1)
    else:  # one run, or no spread in LF Cd: an offset correction only
        b, a = 1.0, float(np.mean(y - x))
    xt, yt = pairs(test)
    err = a + b * xt - yt
    return {
        "a": float(a),
        "b": float(b),
        "n_train": len(train),
        "n_test": len(test),
        "cd_mae_counts": float(np.abs(err).mean() * 1000),
        "cd_raw_lf_mae_counts": float(np.abs(xt - yt).mean() * 1000),
    }


MODELS = (
    ("pointnet_baseline_v1", "PointNet baseline (in-house)"),
    ("transolver_main_v1", "PhysicsNeMo Transolver"),
    ("geotransolver_main_v1", "PhysicsNeMo GeoTransolver"),
)
DS = (("super_scarce", 11), ("scarce", 67), ("medium", 133), ("full", 400))
MF_N = (11, 25, 50)
MF_SEEDS = ("", "_s1", "_s2")  # run-name suffix per seed; seed 0 is the original mf_n<N>
MF_QUANTITIES = ("cd", "cp", "cf_x", "cf_y", "cf_z")


def mf_metric(ev: dict[str, Any], q: str) -> float:
    """Cd MAE in counts, or the area-weighted relative L2 of a field."""
    if q == "cd":
        t, p = _cd_arrays(ev)
        return float(np.abs(p - t).mean() * 1000)
    return float(ev["rel_l2"][q])


def seed_stats(vals: list[float]) -> dict[str, float]:
    """Mean, std (ddof 1; 0 for one seed), min and max over seeds."""
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": len(a),
    }


def paired_verdict(mf: list[float | None], hf: list[float | None]) -> dict[str, Any]:
    """MF - HF-only per seed (lower is better). A winner only if the two models' seed
    ranges do not overlap and the paired difference has the same sign in every seed."""
    pairs = [(m, h) for m, h in zip(mf, hf, strict=True) if m is not None and h is not None]
    d = [m - h for m, h in pairs]
    out: dict[str, Any] = {
        "diff": [None if m is None or h is None else m - h for m, h in zip(mf, hf, strict=True)]
    }
    if len(d) < 2:
        return out | {"mean": d[0] if d else None, "std": None, "verdict": "single seed"}
    ms, hs = [m for m, _ in pairs], [h for _, h in pairs]
    overlap = max(min(ms), min(hs)) <= min(max(ms), max(hs))
    same_sign = all(x < 0 for x in d) or all(x > 0 for x in d)
    if not overlap and same_sign:
        verdict = "MF better" if d[0] < 0 else "HF-only better"
    else:
        verdict = "within seed spread"
    return out | {"mean": float(np.mean(d)), "std": float(np.std(d, ddof=1)), "verdict": verdict}


#: Reported ensemble first; earlier ones stay in the tables as baselines.
ENSEMBLES = (
    ("ensemble_v2", "5 × PhysicsNeMo GeoTransolver, main-model budget"),
    ("ensemble_v1", "5 × PhysicsNeMo Transolver, 20k steps"),
)


OOD_ENSEMBLE = "ensemble_ood_v1"


def _ci_verdict(lo: float, hi: float, pos: str, neg: str, none: str) -> str:
    """Sign of a correlation from its CI; a bound within 0.05 of zero reads as weak."""
    if lo > 0:
        return pos if lo >= 0.05 else f"weakly {pos} (CI lower bound ≈ 0)"
    if hi < 0:
        return neg if hi <= -0.05 else f"weakly {neg} (CI upper bound ≈ 0)"
    return none


def _ratio_verdict(rs: float, re_: float) -> str:
    if rs < 1.05:
        return "σ does not rise"
    if rs / re_ < 0.8:
        return "σ rises, but less than the error (OOD intervals too narrow)"
    if rs / re_ > 1.25:
        return "σ rises more than the error (OOD intervals conservative)"
    return "σ rises in step with the error"


def _ood_answer(o: dict[str, Any]) -> str:
    """The headline, in words, from the numbers."""
    i, d, r = o["id"], o["ood"], o["ratio_ood_over_id"]
    au, nov = o["auroc_ood_vs_id"], o["novelty"]
    pairs = [(r["sigma_cell"]["cp"], r["abs_err_cell"]["cp"]), (r["cd_sigma"], r["cd_mae"])]
    if all(rs < 1.05 for rs, _ in pairs):
        head = "no, σ does not rise out of distribution"
    elif all(rs >= 1.05 and 0.8 <= rs / re_ <= 1.25 for rs, re_ in pairs):
        head = "yes, σ rises out of distribution in step with the error"
    elif all(rs / re_ < 0.8 for rs, re_ in pairs):
        head = "partly, σ rises but less than the error"
    else:
        head = "mixed, σ rises but not in step with the error everywhere"
    shift = max(re_ for _, re_ in pairs)
    size = "a mild shift" if shift < 1.5 else "a substantial shift"
    cov = [o["per_cell_ood"]["coverage"]["0.9"][f] for f in ("cp", "cf_x", "cf_y", "cf_z")]

    def ci(x: dict[str, Any], key: str = "auroc") -> str:
        lo, hi = x["ci95"]
        return f"{x[key]:.2f} [{lo:.2f}, {hi:.2f}]"

    sp_ood = d["cd"]
    lo, hi = sp_ood["spearman_ci95"]
    return (
        f"**Result: {head}, on {size}.** On the {d['n']} most isolated geometries the errors "
        f"grow by ×{r['abs_err_cell']['cp']:.2f} per cell (Cp) and ×{r['cd_mae']:.2f} in Cd "
        f"({i['cd_mae_counts']:.1f} → {d['cd_mae_counts']:.1f} counts); calibrated σ grows "
        f"by ×{r['sigma_cell']['cp']:.2f} and ×{r['cd_sigma']:.2f}, and per-cell 90 % "
        f"coverage on the OOD side is {min(cov):.2f}–{max(cov):.2f}. Per car, mean σ(Cp) "
        f"tells OOD from ID cars with AUROC {ci(au['cp_sigma_cell'])} and σ(Cd) with "
        f"{ci(au['cd_sigma'])}, while the Cd error itself only reaches "
        f"{ci(au['cd_abs_err'])}. σ follows geometric novelty (Spearman with `ood_score`: "
        f"{ci(nov['cp_sigma_vs_novelty_all'], 'spearman')} for σ(Cp), "
        f"{ci(nov['cd_sigma_vs_novelty_all'], 'spearman')} for σ(Cd), ID + OOD) more than the "
        f"Cd error does ({ci(nov['cd_abs_err_vs_novelty_all'], 'spearman')}). So σ works as "
        "a novelty flag; it still does not rank individual cars' Cd errors (Spearman "
        f"{sp_ood['spearman']:+.2f} [{lo:+.2f}, {hi:+.2f}] on the OOD cars). The shift stays "
        "inside the DrivAer notchback morph family; it does not validate σ for other "
        "vehicle types.\n"
    )


def _ood_section(o: dict[str, Any], w: Callable[[str], None]) -> None:
    """Out-of-distribution check: ID = the OOD split's val cars, OOD = its test cars."""
    i, d, r = o["id"], o["ood"], o["ratio_ood_over_id"]
    pc = o["per_cell_ood"]
    w(
        f"## Out-of-distribution check: `{o['split']}` (OOD, {d['n']} cars) vs "
        f"`{o['calib']}` (ID, {i['n']} cars)\n"
    )
    w(
        f"`{OOD_ENSEMBLE}`: 5 heteroscedastic GeoTransolver members with ensemble_v2's config "
        "and budget, trained on the official `geometry` split (339 training cars; "
        "normalisation statistics from those cars only) and kept apart from every `full`-split "
        "result. Test cars are the 20 % most isolated STL surfaces (mean Chamfer RMS distance "
        "to the 10 nearest public geometries, the dataset's `ood_score`); val cars come from "
        "the training-side population. Temperatures are fitted on the val cars, so the ID "
        "side is calibrated by construction and the OOD side shows what survives the shift. "
        "σ below is calibrated; per-cell means are cell means per car, averaged over cars.\n"
    )
    w(_ood_answer(o) + "\n")
    w("### Does σ rise out of distribution?\n")
    w(
        "| Quantity | ID (val) | OOD (test) | OOD / ID "
        "| Error ID | Error OOD | Error OOD / ID | Verdict |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for f in ("cp", "cf_x", "cf_y", "cf_z"):
        rs, re_ = r["sigma_cell"][f], r["abs_err_cell"][f]
        w(
            f"| σ({f}) per cell | {i['sigma_cell_mean'][f]:.4g} | {d['sigma_cell_mean'][f]:.4g} | "
            f"{rs:.2f} | {i['abs_err_cell_mean'][f]:.4g} | {d['abs_err_cell_mean'][f]:.4g} | "
            f"{re_:.2f} | {_ratio_verdict(rs, re_)} |"
        )
    w(
        f"| σ(Cd) per car (counts) | {i['cd_sigma_mean_counts']:.2f} | "
        f"{d['cd_sigma_mean_counts']:.2f} | {r['cd_sigma']:.2f} | {i['cd_mae_counts']:.2f} | "
        f"{d['cd_mae_counts']:.2f} | {r['cd_mae']:.2f} | "
        f"{_ratio_verdict(r['cd_sigma'], r['cd_mae'])} |"
    )
    w("\nError columns: mean |err| per cell for fields, Cd MAE for σ(Cd).\n")
    w("### Does per-car σ flag the OOD cars?\n")
    w("| Score | AUROC (OOD test vs ID val) [95 % CI] | Reading |")
    w("|---|---|---|")
    labels = {
        "cd_sigma": "σ(Cd) per car",
        "cp_sigma_cell": "mean per-cell σ(Cp) per car",
        "cd_abs_err": "\\|Cd error\\| per car (reference: how separable the errors are)",
    }
    for k, lab in labels.items():
        a = o["auroc_ood_vs_id"][k]
        lo, hi = a["ci95"]
        reading = (
            "better than chance"
            if lo > 0.5
            else "worse than chance"
            if hi < 0.5
            else "chance level"
        )
        w(f"| {lab} | {a['auroc']:.2f} [{lo:.2f}, {hi:.2f}] | {reading} |")
    w("")
    w("| Spearman with geometric novelty (`ood_score`) | Cars | ρ [95 % CI] | Reading |")
    w("|---|---|---|---|")
    nl = {
        "cd_sigma_vs_novelty_ood": "σ(Cd), OOD test cars",
        "cd_sigma_vs_novelty_all": "σ(Cd), ID + OOD",
        "cp_sigma_vs_novelty_all": "mean per-cell σ(Cp), ID + OOD",
        "cd_abs_err_vs_novelty_ood": "\\|Cd error\\|, OOD test cars (reference)",
        "cd_abs_err_vs_novelty_all": "\\|Cd error\\|, ID + OOD (reference)",
    }
    for k, lab in nl.items():
        x = o["novelty"][k]
        lo, hi = x["ci95"]
        reading = _ci_verdict(lo, hi, "positive", "negative", "indistinguishable from zero")
        w(f"| {lab} | {x['n']} | {x['spearman']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {reading} |")
    w("")
    w("### Calibration on the OOD side (temperatures from ID val)\n")
    w("| Quantity | 50 % | 90 % | 95 % | NLL | Spearman(σ, \\|err\\|) [95 % CI] |")
    w("|---|---|---|---|---|---|")
    for f in ("cp", "cf_x", "cf_y", "cf_z"):
        c = [pc["coverage"][q][f] for q in ("0.5", "0.9", "0.95")]
        lo, hi = pc["spearman_point_ci95"][f]
        w(
            f"| {f} (per cell) | {c[0]:.2f} | {c[1]:.2f} | {c[2]:.2f} | {pc['nll'][f]:.2f} | "
            f"{pc['spearman_point'][f]:.2f} [{lo:.2f}, {hi:.2f}] |"
        )
    for side, x in (("ID val", i), ("OOD test", d)):
        c, (lo, hi) = x["cd"]["coverage"], x["cd"]["spearman_ci95"]
        w(
            f"| Cd (per car, {side}) | {c['0.5']:.2f} | {c['0.9']:.2f} | {c['0.95']:.2f} | – | "
            f"{x['cd']['spearman']:+.2f} [{lo:+.2f}, {hi:+.2f}] |"
        )
    w(
        "\nPer-cell Spearman CIs bootstrap whole cars (2,000 cells each, 500 resamples); "
        f"per-car CIs bootstrap cars ({o['n_boot']:,} resamples). ID per-cell coverage is "
        "not listed: the temperatures are fitted there.\n"
    )


RANKING_MODEL = "geotransolver_main_v1"


def _band_label(b: dict[str, Any]) -> str:
    return f"≥ {b['lo']:g}" if math.isinf(b["hi"]) else f"{b['lo']:g}–{b['hi']:g}"


def fig_ranking(res: dict[str, Any], out: Path) -> Path:
    """Share of car pairs put in the right drag order, per band of true |dCd|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    preds = res["predictors"]
    names = list(preds)
    # surrogate blue, ensemble aqua, RANS orange: fixed by role, not by position
    color = {
        n: SERIES[1] if n == "coarse RANS" else SERIES[2] if "mean" in n else SERIES[0]
        for n in names
    }
    bands = preds[names[0]]["bands"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3), facecolor=SURFACE)
    width = 0.8 / len(names)
    for k, n in enumerate(names):
        x = [i + (k - (len(names) - 1) / 2) * width for i in range(len(bands))]
        y = [100 * b["share"] for b in preds[n]["bands"]]
        ax.bar(x, y, width * 0.92, color=color[n], label=n, edgecolor=SURFACE, linewidth=1)
    ax.axhline(50, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="50 % (coin flip)")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{_band_label(b)}\n({b['pairs']} pairs)" for b in bands])
    ax.set_ylim(0, 105)
    ax.set_xlabel("True |ΔCd| between two cars (counts)")
    ax.set_ylabel("Pairs in the right drag order (%)")
    ax.set_title(
        f"Design ranking on {res['split']} ({res['n_cars']} cars)", fontsize=11, loc="left"
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(names) + 1,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def _ranking_section(
    res: dict[str, Any], fig: Path, rel: Callable[[Path], str], w: Callable[[str], None]
) -> None:
    preds = res["predictors"]
    w(f"## Design ranking on `{res['split']}` ({res['n_cars']} cars, {res['n_pairs']} pairs)\n")
    w(
        "For design work the order of candidates and the size of a drag difference matter "
        "more than absolute Cd. A pair is in the right order when the predicted and true Cd "
        "differences have the same sign. The error on a drag difference is "
        "|(p_i − p_j) − (t_i − t_j)| over all pairs; a constant bias cancels in it. The "
        "lowest-drag pick is the car with the lowest predicted Cd: its true rank, its "
        "distance from the true best and the overlap of the predicted and true top 5. "
        "Coarse RANS is the in-house OpenFOAM case of each car (SPEC §5.3). "
        f"From `aerosurr eval ranking` (`{res['model']}/ranking_{res['split']}.json`).\n"
    )
    w(
        "| Predictor | Cd MAE (counts) | Bias | Spearman | Pairs ≥ 5 counts apart in order "
        "| ΔCd error mean / p90 (counts) | Lowest-drag pick | Top-5 overlap |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for n, s in preds.items():
        de, pk = s["delta_error"], s["pick"]
        w(
            f"| {n} | {s['cd_mae_counts']:.2f} | {s['cd_bias_counts']:+.2f} | "
            f"{s['spearman']:.3f} | {100 * s['pairs_ge5_ordered']:.1f} % | "
            f"{de['mean_counts']:.2f} / {de['p90_counts']:.2f} | `{pk['run']}`: true rank "
            f"{pk['true_rank']}, {pk['counts_off']:.2f} counts off | {pk['top5_overlap']}/5 |"
        )
    bands = preds[next(iter(preds))]["bands"]
    w("\nPairs in the right drag order, by band of true |ΔCd|:\n")
    w("| True \\|ΔCd\\| (counts) | Pairs | " + " | ".join(preds) + " |")
    w("|---|---|" + "---|" * len(preds))
    for i, b in enumerate(bands):
        cells = " | ".join(f"{100 * preds[n]['bands'][i]['share']:.1f} %" for n in preds)
        w(f"| {_band_label(b)} | {b['pairs']} | {cells} |")
    w("\nOther coefficients (MAE in counts, Spearman):\n")
    w("| Predictor | Coefficient | MAE (counts) | Spearman |\n|---|---|---|---|")
    for title, rows in res["coefficients"].items():
        who = title.split(",")[0]
        for q, r in rows.items():
            w(f"| {who} | {q} | {r['mae_counts']:.2f} | {r['spearman']:.3f} |")
    w(f"\n![Design ranking per band]({rel(fig)})\n")


# Compute groups, in table order: (label, test on the run name). The first match wins.
COMPUTE_GROUPS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("Main models", lambda n: n.endswith("_main_v1")),
    ("Ensembles (v1, v2)", lambda n: n.startswith(("ens_m", "ens2_m"))),
    ("OOD ensemble", lambda n: n.startswith("ood_m")),
    ("Data efficiency", lambda n: n.startswith("transolver_ds_")),
    ("Multi-fidelity study (MF + HF-only, 3 seeds)", lambda n: n.startswith(("mf_n", "hf_n"))),
    ("Baseline (PointNet)", lambda n: n.startswith("pointnet_")),
    ("Smoke test", lambda n: n.startswith("smoke_")),
)
RANS_GLOB = "batch/run_*_coarse_np24/aerosurr_run.json"


def compute_totals(runs: Path, rans: Path | None) -> dict[str, Any]:
    """GPU-hours per group from each run's `summary.json` (`hours`, training wall time on
    one GPU), and coarse-RANS hours from each finished case's `total_s`."""
    groups: dict[str, dict[str, Any]] = {
        lab: {"runs": 0, "hours": 0.0} for lab, _ in COMPUTE_GROUPS
    }
    other: list[str] = []
    for f in sorted(runs.glob("*/summary.json")):
        name, hours = f.parent.name, (_load(f) or {}).get("hours")
        if hours is None:
            continue
        lab = next((lab for lab, match in COMPUTE_GROUPS if match(name)), None)
        if lab is None:
            other.append(name)
            lab = "Other"
            groups.setdefault(lab, {"runs": 0, "hours": 0.0})
        groups[lab]["runs"] += 1
        groups[lab]["hours"] += hours
    secs = []
    for f in sorted(rans.glob(RANS_GLOB)) if rans else []:
        d = _load(f) or {}
        if d.get("status") == "done":
            secs.append(d["total_s"])
    return {
        "groups": {k: v for k, v in groups.items() if v["runs"]},
        "other": other,
        "gpu_runs": sum(v["runs"] for v in groups.values()),
        "gpu_hours": sum(v["hours"] for v in groups.values()),
        "rans_cases": len(secs),
        "rans_hours": sum(secs) / 3600,
        "rans_median_min": float(np.median(secs)) / 60 if secs else None,
    }


def _compute_section(c: dict[str, Any], w: Callable[[str], None]) -> None:
    w("## Compute\n")
    w("Hardware: AMD Threadripper 9980X (64 cores) and 2 × RTX 5090, one workstation.")
    w("GPU-hours are training wall time on one GPU, from each run's `summary.json`;")
    w("evaluation and the store build are not included.\n")
    w("| Work | Runs | GPU-hours |\n|---|---|---|")
    for lab, g in c["groups"].items():
        w(f"| {lab} | {g['runs']} | {g['hours']:.1f} |")
    w(f"| **Total** | **{c['gpu_runs']}** | **{c['gpu_hours']:.1f}** |\n")
    if c["rans_cases"]:
        w(
            f"Coarse RANS (OpenFOAM, one case at a time on 24 cores): {c['rans_cases']} cases, "
            f"median {c['rans_median_min']:.1f} min per case, {c['rans_hours']:.1f} h in all "
            "(`total_s` of each case).\n"
        )


def build_report(
    runs: Path, store: Path, out_md: Path, fig_dir: Path, rans: Path | None = None
) -> list[str]:
    """Write `out_md` (+ figures). Returns the sections that had data."""
    from aerosurr.data.store import load_index
    from aerosurr.rans.mapping import LF_DIR

    fig_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda p: p.relative_to(out_md.parent).as_posix()  # noqa: E731
    done: list[str] = []
    md: list[str] = []
    w = md.append

    loaded = [(label, _load(runs / name / "full_test.json")) for name, label in MODELS]
    rows = [(lab, ev) for lab, ev in loaded if ev is not None]
    if rows:
        done.append("models")
        w("## Surrogates on `full_test` (50 cars, native mesh)\n")
        w("Field errors are area-weighted relative L2. Forces come from integrating the")
        w("predicted fields over the native cells (1 count = 0.001 Cd).\n")
        w(TABLE_HEAD)
        for lab, ev in rows:
            w(summary_row(lab, ev))
        w("")

    # The main model is selected on full_val (lowest full-mesh Cd MAE), never on test.
    main_name, main_label, best_val = None, "", float("inf")
    for name, label in MODELS[1:]:
        summ = _load(runs / name / "summary.json")
        fv = (summ or {}).get("final_full_val")
        if fv and (runs / name / "full_test.json").is_file() and fv["cd_mae_counts"] < best_val:
            main_name, main_label, best_val = name, label, fv["cd_mae_counts"]
    main = _load(runs / main_name / "full_test.json") if main_name else None
    ens_by = {k: _load(runs / k / "ensemble_full_test.json") for k, _ in ENSEMBLES}
    ens_avail = [(k, lab, e) for (k, lab), e in zip(ENSEMBLES, ens_by.values(), strict=True) if e]
    ens = ens_avail[0][2] if ens_avail else None  # the first available is the reported one
    if main:
        w(f"Main model, selected on `full_val` Cd MAE ({best_val:.2f} counts): **{main_label}**.\n")
        sigma = None
        if ens:
            sigma = {r: v["cd_std_cal"] for r, v in ens["per_run"].items()}
        title = f"{main_label} on full_test" + (" (bars: ensemble σ)" if sigma else "")
        f = fig_parity(main, sigma, fig_dir / "parity_main.png", title)
        w(f"![Cd and Cl parity]({rel(f)})\n")

    rank = _load(runs / RANKING_MODEL / "ranking_full_test.json")
    if rank:
        done.append("ranking")
        _ranking_section(rank, fig_ranking(rank, fig_dir / "ranking_bands.png"), rel, w)

    ds_all = {n: _load(runs / f"transolver_ds_{s}" / "full_test.json") for s, n in DS}
    ds = {n: ev for n, ev in ds_all.items() if ev is not None}
    if len(ds) >= 2:
        done.append("data-efficiency")

        def cd_mae(ev: dict[str, Any]) -> float:
            t, p = _cd_arrays(ev)
            return float(np.abs(p - t).mean() * 1000)

        cd = {"20k steps (all sizes)": [(n, cd_mae(ev)) for n, ev in ds.items()]}
        if main:
            cd["60k steps (main run)"] = [(400, cd_mae(main))]
        f = fig_lines(
            cd,
            fig_dir / "data_efficiency_cd.png",
            "Data efficiency: Cd error vs training-set size",
            "Cd MAE on full_test (counts)",
        )
        w("## Data efficiency (nested official splits)\n")
        w("HF-only Transolver trained on `super_scarce` ⊂ `scarce` ⊂ `medium` ⊂ `full` with")
        w("the same 20k-step budget each. At that budget the curve flattens beyond ~70 cars:")
        w("400 cars get only 50 epochs. The main run (400 cars, 60k steps) shows the gain from")
        w("training longer.\n")
        w("| Training runs | Steps | Cp rel L2 | Cd MAE (counts) |\n|---|---|---|---|")
        for n, ev in sorted(ds.items()):
            w(f"| {n} | 20k | {ev['rel_l2']['cp']:.3f} | {cd_mae(ev):.1f} |")
        if main:
            w(f"| 400 | 60k (main) | {main['rel_l2']['cp']:.3f} | {cd_mae(main):.1f} |")
        w(f"\n![Data efficiency]({rel(f)})\n")

    lf_eval = _load(store / LF_DIR / "eval_full_test.json")
    lf_index = _load(store / LF_DIR / "index.json")
    # seeds[m][n]: the full_test evaluations of model m at N cars, one per seed (or None)
    seeds = {
        m: {
            n: [_load(runs / f"{m}_n{n}{sfx}" / "full_test.json") for sfx in MF_SEEDS] for n in MF_N
        }
        for m in ("mf", "hf")
    }
    mf = {n: seeds["mf"][n][0] for n in MF_N}
    if any(mf.values()) or lf_eval:
        done.append("multi-fidelity")
        w("## Multi-fidelity correction (coarse RANS → HRLES)\n")
        if lf_eval:
            w(
                f"- LF alone (coarse RANS mapped onto the HF mesh, {lf_eval['runs']} test cars): "
                f"Cp rel L2 {lf_eval['rel_l2']['cp']:.3f}, "
                f"Cd MAE {lf_eval['cd_mae_counts']:.1f} counts."
            )

        def vals(m: str, n: int, q: str) -> list[float | None]:
            return [mf_metric(ev, q) if ev else None for ev in seeds[m][n]]

        def fmt(v: list[float | None], digits: int) -> str:
            got = [x for x in v if x is not None]
            if not got:
                return "–"
            st = seed_stats(got)
            if st["n"] == 1:
                return f"{st['mean']:.{digits}f}"
            return (
                f"{st['mean']:.{digits}f} ± {st['std']:.{digits}f} "
                f"[{st['min']:.{digits}f}, {st['max']:.{digits}f}]"
            )

        def fmt_diff(pv: dict[str, Any], digits: int) -> str:
            return " / ".join("–" if d is None else f"{d:+.{digits}f}" for d in pv["diff"])

        sidx = load_index(store)
        scalar: dict[int, float] = {}
        for n in MF_N:
            cfg = _load(runs / f"mf_n{n}" / "config.json")
            mf_n = mf[n]
            if cfg and lf_index and mf_n:
                sc = scalar_correction(lf_index, sidx, cfg["train_runs"], list(mf_n["per_run"]))
                scalar[n] = sc["cd_mae_counts"]
        present = [ev for m in seeds for n in MF_N for ev in seeds[m][n] if ev]
        if present:
            n_cars = len(present[0]["per_run"])
            n_seeds = min(sum(ev is not None for ev in seeds[m][n]) for m in seeds for n in MF_N)
            verdicts: dict[str, dict[int, str]] = {q: {} for q in MF_QUANTITIES}
            for q in MF_QUANTITIES:
                for n in MF_N:
                    verdicts[q][n] = paired_verdict(vals("mf", n, q), vals("hf", n, q))["verdict"]
            w(
                f"\nEvery model is evaluated on the same {n_cars} LF-mapped `full_test` cars. "
                f"Values are mean ± std (ddof 1) [min, max] over {n_seeds} seed(s); a seed sets "
                "the initialisation and the data order. Every seed uses the same nested "
                "training cars, so the variation across training subsets is not covered. "
                "MF − HF-only is paired by seed (seeds 0 / 1 / 2; negative = MF better). The "
                "verdict names a winner only when the two models' seed ranges do not overlap "
                "and the paired difference has the same sign in every seed; otherwise the "
                "difference is *within seed spread*. LF alone and the scalar correction are "
                "deterministic for a given set of training cars, so they are single values.\n"
            )

            def grouped(v: dict[int, str]) -> str:
                by: dict[str, list[int]] = {}
                for k, verdict in v.items():
                    by.setdefault(verdict, []).append(k)
                parts = []
                for verdict, ns in by.items():
                    lst = ", ".join(map(str, ns[:-1])) + " and " if len(ns) > 1 else ""
                    parts.append(f"{verdict} at {lst}{ns[-1]} cars")
                return ", ".join(parts)

            labels = {"cd": "Cd MAE", "cp": "Cp", "cf_x": "Cf_x", "cf_y": "Cf_y", "cf_z": "Cf_z"}
            w(
                "**Result:** "
                + "; ".join(f"{labels[q]}: {grouped(verdicts[q])}" for q in MF_QUANTITIES)
                + ".\n"
            )
            w("Cd MAE (counts):\n")
            w(
                "| Training cars | HF only | MF (LF inputs, residual) | MF − HF-only per seed "
                "| Mean difference | Verdict | Scalar Cd_HF = a + b·Cd_LF |"
            )
            w("|---|---|---|---|---|---|---|")
            for n in MF_N:
                pv = paired_verdict(vals("mf", n, "cd"), vals("hf", n, "cd"))
                mdiff = "–" if pv["mean"] is None else f"{pv['mean']:+.1f}"
                sc_v = f"{scalar[n]:.1f}" if n in scalar else "–"
                w(
                    f"| {n} | {fmt(vals('hf', n, 'cd'), 1)} | {fmt(vals('mf', n, 'cd'), 1)} | "
                    f"{fmt_diff(pv, 1)} | {mdiff} | {pv['verdict']} | {sc_v} |"
                )
            w("\nField errors (area-weighted relative L2):\n")
            w(
                "| Training cars | Field | HF only | MF | MF − HF-only per seed | Mean difference "
                "| Verdict |"
            )
            w("|---|---|---|---|---|---|---|")
            for n in MF_N:
                for q in MF_QUANTITIES[1:]:
                    pv = paired_verdict(vals("mf", n, q), vals("hf", n, q))
                    mdiff = "–" if pv["mean"] is None else f"{pv['mean']:+.3f}"
                    w(
                        f"| {n} | {q} | {fmt(vals('hf', n, q), 3)} | {fmt(vals('mf', n, q), 3)} | "
                        f"{fmt_diff(pv, 3)} | {mdiff} | {pv['verdict']} |"
                    )
            series: dict[str, list[tuple[int, float]]] = {}
            bars: dict[str, list[tuple[int, float, float]]] = {}
            for m, lab in (("hf", "HF only"), ("mf", "MF (LF inputs, residual)")):
                for n in MF_N:
                    got = [x for x in vals(m, n, "cd") if x is not None]
                    if got:
                        st = seed_stats(got)
                        series.setdefault(lab, []).append((n, st["mean"]))
                        if st["n"] > 1:
                            bars.setdefault(lab, []).append((n, st["min"], st["max"]))
            if scalar:
                series["Scalar Cd_HF = a + b·Cd_LF"] = sorted(scalar.items())
            ref = ("LF alone", lf_eval["cd_mae_counts"]) if lf_eval else None
            f = fig_lines(
                series,
                fig_dir / "mf_cd.png",
                f"Cd error vs HF training runs (mean; bars: min–max over {n_seeds} seeds)"
                if bars
                else "Cd error vs HF training runs",
                "Cd MAE on LF-mapped test cars (counts)",
                ref,
                bars,
            )
            w(f"\n![Multi-fidelity]({rel(f)})\n")

    if ens_avail:
        done.append("ensemble")
        key, label, e0 = ens_avail[0]
        f = fig_calibration(e0, fig_dir / "calibration.png")
        w("## Deep ensemble, calibrated on `full_val`, reported on `full_test`\n")
        w(f"Reported ensemble: **{label}** (`{key}`, {e0['members']} members).\n")
        w("| Ensemble | Quantity | 50 % | 90 % | 95 % | NLL | Spearman(σ, \\|err\\|) |")
        w("|---|---|---|---|---|---|---|")
        for k, _lab, e in ens_avail:
            for fname in ("cp", "cf_x", "cf_y", "cf_z"):
                c = [e["coverage"][q][fname] for q in ("0.5", "0.9", "0.95")]
                w(
                    f"| {k} | {fname} (per cell) | {c[0]:.2f} | {c[1]:.2f} | {c[2]:.2f} | "
                    f"{e['nll'][fname]:.2f} | {e['spearman_point'][fname]:.2f} |"
                )
            c = [e["cd_coverage"][q] for q in ("0.5", "0.9", "0.95")]
            sp = e["cd_spearman_car"]
            w(f"| {k} | Cd (per car) | {c[0]:.2f} | {c[1]:.2f} | {c[2]:.2f} | – | {sp:.2f} |")
        w(
            "\n| Ensemble | Members | Ensemble-mean Cd MAE (counts) "
            "| Mean calibrated σ(Cd) (counts) | Cd temperature | Aleatoric share |"
        )
        w("|---|---|---|---|---|---|")
        for k, lab, e in ens_avail:
            w(
                f"| {k} ({lab}) | {e['members']} | {e['cd_mae_counts']:.1f} | "
                f"{e['cd_sigma_mean_counts']:.1f} | {e['temperature']['cd']:.2f} | "
                f"{e['aleatoric_fraction']:.0%} |"
            )
        pr = e0["per_run"]
        cm = np.array([v["cd_mean"] for v in pr.values()])
        ct = np.array([v["cd_true"] for v in pr.values()])
        r2e = 1 - ((cm - ct) ** 2).sum() / ((ct - ct.mean()) ** 2).sum()
        n_cars = len(pr)
        se = 1 / np.sqrt(max(n_cars - 1, 1))
        sp0 = e0["cd_spearman_car"]
        err = np.abs(cm - ct) * 1000
        w(
            f"\n**Ensemble-mean forces ({key}):** Cd MAE {err.mean():.1f} counts "
            f"(max {err.max():.1f}), Cd R² {r2e:.3f}, over {n_cars} test cars."
        )
        verdict = (
            "is indistinguishable from zero"
            if abs(sp0) < 2 * se
            else ("is positive" if sp0 > 0 else "is negative")
        )
        w(
            f"Per-car σ(Cd) is calibrated in coverage, but its rank correlation with the Cd "
            f"error ({sp0:+.2f}) {verdict} at n = {n_cars} (standard error ≈ {se:.2f}): it says "
            "how large drag errors are on average, not which cars are worst.\n"
        )
        checks: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for k, _, e in ens_avail:
            if chk := _load(runs / k / "per_car_sigma.json"):
                checks.append((k, chk, e))
        if checks:
            same = all(
                "current" in c
                and c["current"]["coverage"] == c["calibrated"]["coverage"]
                and np.isclose(
                    c["current"]["sigma_mean_counts"], c["calibrated"]["sigma_mean_counts"]
                )
                for _, c, _ in checks
            )
            agree = "do" if same else "**do not** (investigate before quoting either)"
            w("### Per-car Cd uncertainty, like for like (`full_test`, 50 cars)\n")
            w(
                "σ_epi(Cd) is the std over members of each member's integrated Cd "
                f"(ddof = {checks[0][1]['ddof']}); the error is the ensemble-mean Cd minus HRLES. "
                "Calibrated rows use one temperature fitted on `full_val`. In this pipeline "
                "the reported per-car σ(Cd) already is that member spread (the aleatoric share "
                "above is per-cell field variance), so the *current* and *epistemic, "
                "calibrated* rows must coincide; the second recomputes it from saved "
                f"per-member Cd, and here they {agree}. Spearman CIs are percentile bootstraps "
                "over cars (10,000).\n"
            )
            w(
                "| Ensemble | σ(Cd) | Temperature | Mean σ (counts) | 50 % | 90 % | 95 % "
                "| Spearman(σ, \\|err\\|) [95 % CI] |"
            )
            w("|---|---|---|---|---|---|---|---|")

            def crow(k: str, label: str, t: str, r: dict[str, Any]) -> None:
                cov = r["coverage"]
                lo, hi = r["spearman_ci95"]
                w(
                    f"| {k} | {label} | {t} | {r['sigma_mean_counts']:.1f} | {cov['0.5']:.2f} | "
                    f"{cov['0.9']:.2f} | {cov['0.95']:.2f} | "
                    f"{r['spearman']:+.2f} [{lo:+.2f}, {hi:+.2f}] |"
                )

            for k, chk, e in checks:
                if "current" in chk:
                    t_cd = f"{e['temperature']['cd']:.2f}"
                    crow(k, "current (reported above)", t_cd, chk["current"])
                crow(k, "epistemic, raw", "1", chk["raw"])
                crow(k, "epistemic, calibrated", f"{chk['temperature']:.2f}", chk["calibrated"])
            w("")

    ood = _load(runs / OOD_ENSEMBLE / "ood.json")
    if ood:
        done.append("ood")
        _ood_section(ood, w)

    comp = compute_totals(runs, rans)
    if comp["gpu_runs"] or comp["rans_cases"]:
        done.append("compute")
        _compute_section(comp, w)

    header = (
        "# Results\n\nGenerated by `aerosurr report` from the evaluation JSON under "
        "`~/aero/runs` and the store; sections appear once their runs finish.\n\n"
    )
    out_md.write_text(header + "\n".join(md).rstrip() + "\n")
    return done
