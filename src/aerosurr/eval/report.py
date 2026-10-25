"""Results report (SPEC §6): tables and figures from whatever evaluations exist.

Every section reads JSON written by `aerosurr eval run / eval ensemble / eval lf` and the
store index, and is skipped when its inputs are not there yet, so the report can be
regenerated at any point of the project.
"""

from __future__ import annotations

import json
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
) -> Path:
    """Error versus number of HF training runs (log x); up to 3 series, direct-labelled."""
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
    ymax = max([v for pts in series.values() for _, v in pts] + [reference[1] if reference else 0])
    ax.set_ylim(0, ymax * 1.15)
    for (name, pts), color in zip(series.items(), SERIES, strict=False):
        pts = sorted(pts)
        x, y = [p[0] for p in pts], [p[1] for p in pts]
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
#: Reported ensemble first; earlier ones stay in the tables as baselines.
ENSEMBLES = (
    ("ensemble_v2", "5 × PhysicsNeMo GeoTransolver, main-model budget"),
    ("ensemble_v1", "5 × PhysicsNeMo Transolver, 20k steps"),
)


def build_report(runs: Path, store: Path, out_md: Path, fig_dir: Path) -> list[str]:
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
    mf = {n: _load(runs / f"mf_n{n}" / "full_test.json") for n in MF_N}
    hf = {n: _load(runs / f"hf_n{n}" / "full_test.json") for n in MF_N}
    if any(mf.values()) or lf_eval:
        done.append("multi-fidelity")
        w("## Multi-fidelity correction (coarse RANS → HRLES)\n")
        if lf_eval:
            w(
                f"- LF alone (coarse RANS mapped onto the HF mesh, {lf_eval['runs']} test cars): "
                f"Cp rel L2 {lf_eval['rel_l2']['cp']:.3f}, "
                f"Cd MAE {lf_eval['cd_mae_counts']:.1f} counts."
            )
        series: dict[str, list[tuple[int, float]]] = {
            "HF only": [],
            "MF (LF inputs, residual)": [],
            "Scalar Cd_HF = a + b·Cd_LF": [],
        }
        sidx = load_index(store)
        for n in MF_N:
            for key, res in (("HF only", hf[n]), ("MF (LF inputs, residual)", mf[n])):
                if res:
                    t, p = _cd_arrays(res)
                    series[key].append((n, float(np.abs(p - t).mean() * 1000)))
            cfg = _load(runs / f"mf_n{n}" / "config.json")
            mf_n = mf[n]
            if cfg and lf_index and mf_n:
                sc = scalar_correction(lf_index, sidx, cfg["train_runs"], list(mf_n["per_run"]))
                series["Scalar Cd_HF = a + b·Cd_LF"].append((n, sc["cd_mae_counts"]))
        series = {k: v for k, v in series.items() if v}
        if series:
            w("\n| Training runs | " + " | ".join(series) + " |\n|---|" + "---|" * len(series))
            for n in MF_N:
                vals = [
                    next((f"{v:.1f}" for m, v in pts if m == n), "–") for pts in series.values()
                ]
                w(f"| {n} | " + " | ".join(vals) + " |")
            ref = ("LF alone", lf_eval["cd_mae_counts"]) if lf_eval else None
            f = fig_lines(
                series,
                fig_dir / "mf_cd.png",
                "Cd error vs HF training runs",
                "Cd MAE on LF-mapped test cars (counts)",
                ref,
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

    header = (
        "# Results\n\nGenerated by `aerosurr report` from the evaluation JSON under "
        "`~/aero/runs` and the store; sections appear once their runs finish.\n\n"
    )
    out_md.write_text(header + "\n".join(md).rstrip() + "\n")
    return done
