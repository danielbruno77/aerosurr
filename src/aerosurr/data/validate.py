"""Reference-force gate (SPEC §5.1): integrate the raw surface fields of a run and compare
with DrivAerML's `force_mom_i.csv` (per-car reference) and `force_mom_constref_i.csv`."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerosurr.config import Freestream, SurfaceFields
from aerosurr.data.geometry import polygon_centres_and_area_vectors
from aerosurr.data.splits import run_number
from aerosurr.data.vtp import read_vtp
from aerosurr.physics.coefficients import pressure_coefficient, skin_friction_coefficient
from aerosurr.physics.forces import ForceCoefficients, Reference, integrate

#: SPEC §5.1 tolerances. Cd and Cl are relative (0.5 %) with an absolute floor of 1e-4
#: (a tenth of a drag count) so that near-zero lift is not held to a relative bound.
REL_TOL = 0.005
REL_FLOOR = 1e-4
ABS_TOL = 0.002
CSV_KEYS = {"cd": "Cd", "cl": "Cl", "clf": "Clf", "clr": "Clr", "cs": "Cs"}


@dataclass(frozen=True)
class GeoRefs:
    varying: Reference
    constant: Reference


def read_geo_refs(path: Path) -> GeoRefs:
    header, row = path.read_text().strip().splitlines()[:2]
    if header.replace(" ", "") != "lRef,aRef,forcesCoR,lRefRef,aRefRef,forcesCoRRef":
        raise ValueError(f"{path}: unexpected header {header!r}")
    m = re.fullmatch(r"([^,]+),([^,]+),\(([^)]*)\),([^,]+),([^,]+),\(([^)]*)\)", row.strip())
    if not m:
        raise ValueError(f"{path}: cannot parse {row!r}")
    g = m.groups()

    def vec(s: str) -> tuple[float, float, float]:
        x, y, z = (float(t) for t in s.split())
        return (x, y, z)

    return GeoRefs(
        varying=Reference(a_ref=float(g[1]), l_ref=float(g[0]), cor=vec(g[2])),
        constant=Reference(a_ref=float(g[4]), l_ref=float(g[3]), cor=vec(g[5])),
    )


def read_force_csv(path: Path) -> dict[str, float]:
    header, row = path.read_text().strip().splitlines()[:2]
    vals = {k.strip(): float(v) for k, v in zip(header.split(","), row.split(","), strict=True)}
    return {ours: vals[theirs] for ours, theirs in CSV_KEYS.items()}


def within_tolerance(name: str, ours: float, ref: float) -> bool:
    err = abs(ours - ref)
    if name in ("cd", "cl"):
        return err <= max(REL_TOL * abs(ref), REL_FLOOR)
    return err <= ABS_TOL


@dataclass
class RunValidation:
    run: str
    n_cells: int
    ours: dict[str, dict[str, float]] = field(default_factory=dict)  # "varying"/"constant"
    ref: dict[str, dict[str, float]] = field(default_factory=dict)

    def errors(self, which: str) -> dict[str, float]:
        return {k: self.ours[which][k] - self.ref[which][k] for k in CSV_KEYS}

    @property
    def passed(self) -> bool:
        return all(
            within_tolerance(k, self.ours[w][k], self.ref[w][k]) for w in self.ref for k in CSV_KEYS
        )


def coefficients_of_run(
    raw: Path, run: str, fields: SurfaceFields, fs: Freestream
) -> tuple[int, ForceCoefficients, ForceCoefficients]:
    n = run_number(run)
    surf = read_vtp(raw / run / f"boundary_{n}.vtp")
    centres, a = polygon_centres_and_area_vectors(surf.points, surf.offsets, surf.connectivity)
    cp = pressure_coefficient(surf.cell_data[fields.pressure], fs.q_inf, fs.p_inf)
    cf = skin_friction_coefficient(surf.cell_data[fields.wall_shear_stress], fs.q_inf)
    refs = read_geo_refs(raw / run / f"geo_ref_{n}.csv")
    return (
        surf.n_cells,
        integrate(cp, cf, a, centres, refs.varying),
        integrate(cp, cf, a, centres, refs.constant),
    )


def validate_run(raw: Path, run: str, fields: SurfaceFields, fs: Freestream) -> RunValidation:
    n = run_number(run)
    n_cells, var, const = coefficients_of_run(raw, run, fields, fs)
    res = RunValidation(run=run, n_cells=n_cells)
    res.ours = {
        "varying": {k: getattr(var, k) for k in CSV_KEYS},
        "constant": {k: getattr(const, k) for k in CSV_KEYS},
    }
    res.ref = {
        "varying": read_force_csv(raw / run / f"force_mom_{n}.csv"),
        "constant": read_force_csv(raw / run / f"force_mom_constref_{n}.csv"),
    }
    return res


def render_markdown(results: list[RunValidation]) -> str:
    out: list[str] = []
    w = out.append
    n_pass = sum(r.passed for r in results)
    w(f"- Runs: {len(results)}; passed: **{n_pass}**; failed: {len(results) - n_pass}")
    w(f"- Tolerances: Cd, Cl within {REL_TOL:.1%} (floor {REL_FLOOR:g}); Clf, Clr, Cs within "
      f"{ABS_TOL:g} absolute")  # fmt: skip
    w("")
    labels = {
        "varying": "`force_mom_i.csv` (per-car reference)",
        "constant": "`force_mom_constref_i.csv` (constant reference)",
    }
    for which, label in labels.items():
        w(f"#### Against {label}\n")
        w("| Coefficient | max abs error | mean abs error | max rel error | worst run |")
        w("|---|---|---|---|---|")
        for k, name in CSV_KEYS.items():
            errs = np.array([abs(r.errors(which)[k]) for r in results])
            rels = np.array([abs(r.errors(which)[k]) / max(abs(r.ref[which][k]), 1e-12)
                             for r in results])  # fmt: skip
            worst = results[int(errs.argmax())].run
            w(f"| {name} | {errs.max():.2e} | {errs.mean():.2e} | {rels.max():.2e} | {worst} |")
        w("")
    failed = [r for r in results if not r.passed]
    if failed:
        w("#### Failed runs\n")
        w("| Run | which | errors |")
        w("|---|---|---|")
        for r in failed:
            for which in r.ref:
                detail = ", ".join(f"{CSV_KEYS[k]} {v:+.2e}" for k, v in r.errors(which).items())
                w(f"| {r.run} | {which} | {detail} |")
        w("")
    return "\n".join(out)
