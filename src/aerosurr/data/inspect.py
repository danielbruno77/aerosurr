"""Inspection gate (SPEC §4.5): describe one real boundary VTP exactly as stored."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aerosurr.data.geometry import (
    polygon_arity_histogram,
    polygon_centres_and_area_vectors,
    vtk_cell_type_histogram,
)
from aerosurr.data.vtp import Surface, XmlTag, read_vtp, scan_xml_tags


@dataclass(frozen=True)
class ArrayStats:
    association: str  # "point", "cell" or "field"
    name: str
    dtype: str
    shape: tuple[int, ...]
    minimum: list[float]
    maximum: list[float]
    mean: list[float]
    std: list[float]
    n_nonfinite: int
    magnitude: tuple[float, float, float] | None = None  # (min, max, mean) of |v| for 3-vectors


@dataclass(frozen=True)
class AreaCheck:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    matches_n_cells: bool
    total_area: float
    min_area: float
    max_area: float
    recomputed_max_rel_err: float
    closure_rel: float  # |sum(A n)| / sum(A); ~0 for a closed surface
    signed_volume: float  # (1/3) sum(c . A n); > 0 means outward-pointing normals


@dataclass(frozen=True)
class PressureFit:
    """Least-squares fit p = p_inf + q_inf * Cp over all cells."""

    pressure: str
    coefficient: str
    q_inf: float
    p_inf: float
    r2: float
    max_abs_residual: float
    u_inf_if_kinematic: float  # sqrt(2 q_inf): valid when p is p/rho


@dataclass
class InspectionReport:
    path: Path
    size_bytes: int
    sha256: str | None
    xml_tags: list[XmlTag]
    header_comment: str | None
    n_points: int
    n_cells: int
    n_verts: int
    n_lines: int
    n_strips: int
    arity: dict[int, int]
    cell_types: dict[int, int]
    bounds_min: list[float]
    bounds_max: list[float]
    arrays: list[ArrayStats] = field(default_factory=list)
    areas: AreaCheck | None = None
    pressure_fit: PressureFit | None = None

    @property
    def passed(self) -> bool:
        return self.areas is not None and self.areas.matches_n_cells


def array_stats(association: str, name: str, a: NDArray[np.generic]) -> ArrayStats:
    x = np.asarray(a)
    x2 = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(-1, 1)
    xf = x2.astype(np.float64)
    finite = np.isfinite(xf)
    n_bad = int((~finite).sum())
    xm = np.where(finite, xf, np.nan)
    mag = None
    if x2.shape[1] == 3:
        m = np.linalg.norm(xm, axis=1)
        mag = (float(np.nanmin(m)), float(np.nanmax(m)), float(np.nanmean(m)))
    return ArrayStats(
        association=association,
        name=name,
        dtype=x.dtype.str,
        shape=tuple(int(s) for s in x.shape),
        minimum=[float(v) for v in np.nanmin(xm, axis=0)],
        maximum=[float(v) for v in np.nanmax(xm, axis=0)],
        mean=[float(v) for v in np.nanmean(xm, axis=0)],
        std=[float(v) for v in np.nanstd(xm, axis=0)],
        n_nonfinite=n_bad,
        magnitude=mag,
    )


def default_areas_path(vtp: Path) -> Path:
    """`run_N/boundary_N.vtp` -> `run_N/boundary_cell_area_N.npy`."""
    m = re.fullmatch(r"boundary_(\d+)\.vtp", vtp.name)
    if not m:
        raise ValueError(f"cannot derive area file from {vtp.name}")
    return vtp.with_name(f"boundary_cell_area_{m.group(1)}.npy")


def check_areas(surf: Surface, areas_path: Path) -> AreaCheck:
    areas = np.load(areas_path, mmap_mode="r", allow_pickle=False)
    ok = areas.ndim == 1 and areas.shape[0] == surf.n_cells
    centres, av = polygon_centres_and_area_vectors(surf.points, surf.offsets, surf.connectivity)
    a_calc = np.linalg.norm(av, axis=1)
    a64 = np.asarray(areas, dtype=np.float64)
    rel = float(np.max(np.abs(a64 - a_calc) / a_calc)) if ok else float("nan")
    total = float(a_calc.sum())
    return AreaCheck(
        path=areas_path,
        dtype=areas.dtype.str,
        shape=tuple(int(s) for s in areas.shape),
        matches_n_cells=bool(ok),
        total_area=float(a64.sum()),
        min_area=float(a64.min()),
        max_area=float(a64.max()),
        recomputed_max_rel_err=rel,
        closure_rel=float(np.linalg.norm(av.sum(axis=0)) / total),
        signed_volume=float(np.einsum("ij,ij->", centres, av) / 3.0),
    )


def fit_pressure(surf: Surface, pressure: str, coefficient: str) -> PressureFit:
    p = np.asarray(surf.cell_data[pressure], dtype=np.float64).ravel()
    cp = np.asarray(surf.cell_data[coefficient], dtype=np.float64).ravel()
    design = np.column_stack([np.ones_like(cp), cp])
    (p0, q), *_ = np.linalg.lstsq(design, p, rcond=None)
    resid = p - (p0 + q * cp)
    r2 = 1.0 - float(resid @ resid) / float(((p - p.mean()) ** 2).sum())
    return PressureFit(
        pressure=pressure,
        coefficient=coefficient,
        q_inf=float(q),
        p_inf=float(p0),
        r2=r2,
        max_abs_residual=float(np.abs(resid).max()),
        u_inf_if_kinematic=float(np.sqrt(2.0 * q)) if q > 0 else float("nan"),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def _header_comment(path: Path) -> str | None:
    with path.open("rb") as f:
        head = f.read(512).decode("ascii", errors="replace")
    m = re.search(r"<!--(.*?)-->", head)
    return m.group(1).strip() if m else None


def inspect_vtp(
    path: Path,
    areas_path: Path | None = None,
    pressure_fit: tuple[str, str] | None = None,
    compute_sha256: bool = True,
) -> InspectionReport:
    surf = read_vtp(path)
    arity = polygon_arity_histogram(surf.offsets)
    rep = InspectionReport(
        path=path,
        size_bytes=path.stat().st_size,
        sha256=_sha256(path) if compute_sha256 else None,
        xml_tags=scan_xml_tags(path),
        header_comment=_header_comment(path),
        n_points=surf.n_points,
        n_cells=surf.n_cells,
        n_verts=surf.n_verts,
        n_lines=surf.n_lines,
        n_strips=surf.n_strips,
        arity=arity,
        cell_types=vtk_cell_type_histogram(arity),
        bounds_min=[float(v) for v in surf.points.min(axis=0)],
        bounds_max=[float(v) for v in surf.points.max(axis=0)],
    )
    for assoc, arrays in (
        ("point", surf.point_data),
        ("cell", surf.cell_data),
        ("field", surf.field_data),
    ):
        for name, a in arrays.items():
            rep.arrays.append(array_stats(assoc, name, a))
    if areas_path is not None:
        rep.areas = check_areas(surf, areas_path)
    if pressure_fit is not None:
        rep.pressure_fit = fit_pressure(surf, *pressure_fit)
    return rep


# --------------------------------------------------------------------------- markdown


def _g(v: float) -> str:
    return f"{v:.6g}"


def _vec(vs: list[float]) -> str:
    return ", ".join(_g(v) for v in vs)


def render_markdown(rep: InspectionReport) -> str:
    out: list[str] = []
    w = out.append
    w(f"### `{rep.path.name}`\n")
    w(f"- Path: `{rep.path}`")
    w(f"- Size: {rep.size_bytes:,} bytes")
    if rep.sha256:
        w(f"- sha256: `{rep.sha256}`")
    if rep.header_comment:
        w(f"- XML header comment: `{rep.header_comment}`")
    w("")
    w("#### XML structure (as declared on disk)\n")
    w("| Tag | Section | Attributes |")
    w("|---|---|---|")
    for t in rep.xml_tags:
        attrs = " ".join(f"{k}=`{v}`" for k, v in t.attrs.items())
        w(f"| {t.tag} | {t.section or '-'} | {attrs} |")
    w("")
    w("#### Topology\n")
    w(f"- Points: **{rep.n_points:,}**")
    w(
        f"- Cells: **{rep.n_cells:,}** (polygons; verts {rep.n_verts}, lines {rep.n_lines}, "
        f"strips {rep.n_strips})"
    )
    w("- Polygon arity: " + ", ".join(f"{k}: {v:,}" for k, v in rep.arity.items()))
    w(
        "- VTK cell types: "
        + ", ".join(f"{k}: {v:,}" for k, v in rep.cell_types.items())
        + " (5 triangle, 9 quad, 7 polygon)"
    )
    w(f"- Bounding box min (m): ({_vec(rep.bounds_min)})")
    w(f"- Bounding box max (m): ({_vec(rep.bounds_max)})")
    w("")
    w("#### Arrays\n")
    w("| Assoc. | Name | dtype | Shape | Min | Max | Mean | Std | Non-finite |")
    w("|---|---|---|---|---|---|---|---|---|")
    for a in rep.arrays:
        w(
            f"| {a.association} | `{a.name}` | `{a.dtype}` | {a.shape} | {_vec(a.minimum)} | "
            f"{_vec(a.maximum)} | {_vec(a.mean)} | {_vec(a.std)} | {a.n_nonfinite} |"
        )
    mags = [a for a in rep.arrays if a.magnitude is not None]
    if mags:
        w("")
        w("| Vector array | \\|v\\| min | \\|v\\| max | \\|v\\| mean |")
        w("|---|---|---|---|")
        for a in mags:
            assert a.magnitude is not None
            lo, hi, mu = a.magnitude
            w(f"| `{a.name}` | {_g(lo)} | {_g(hi)} | {_g(mu)} |")
    w("")
    if rep.areas is not None:
        c = rep.areas
        verdict = "PASS" if c.matches_n_cells else "FAIL"
        w("#### Cell-area check\n")
        w(f"- File: `{c.path.name}`, dtype `{c.dtype}`, shape {c.shape}")
        w(f"- `len(areas) == n_cells` ({c.shape[0]:,} vs {rep.n_cells:,}): **{verdict}**")
        w(f"- Total area {_g(c.total_area)} m², min {_g(c.min_area)} m², max {_g(c.max_area)} m²")
        w(
            f"- Max relative error vs Newell area recomputed here in float64: "
            f"{_g(c.recomputed_max_rel_err)}"
        )
        w(f"- Area-vector closure |Σ A·n| / Σ A: {_g(c.closure_rel)}")
        w(
            f"- Signed volume (1/3) Σ c·n A: {_g(c.signed_volume)} m³ "
            f"({'outward' if c.signed_volume > 0 else 'inward'}-pointing polygon normals)"
        )
        w("")
    if rep.pressure_fit is not None:
        f = rep.pressure_fit
        w("#### Pressure / Cp relation\n")
        w(f"Least-squares fit `{f.pressure} = p_inf + q_inf * {f.coefficient}` over all cells:\n")
        w(
            f"- q_inf = {_g(f.q_inf)}, p_inf = {_g(f.p_inf)}, R² = {f.r2:.12f}, "
            f"max |residual| = {_g(f.max_abs_residual)}"
        )
        w(
            f"- If `{f.pressure}` is kinematic (p/ρ, m²/s²): U∞ = sqrt(2 q_inf) = "
            f"{_g(f.u_inf_if_kinematic)} m/s"
        )
        w("")
    return "\n".join(out)
