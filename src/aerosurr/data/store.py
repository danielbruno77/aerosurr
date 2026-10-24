"""Compact training store (SPEC §5.1): one `.npz` per run plus an `index.json`.

Every run gets a fixed random **pool** of N cells (training samples from it). Runs listed
as `full` (val and test: evaluation on the native mesh) also keep **all** cells. The raw
VTPs stay the source of truth, so a store can always be rebuilt.

Arrays per group (`pool_*`, `full_*`): `centres` f4 (n, 3) m, `normals` f4 (n, 3) unit,
pointing out of the car into the fluid, `areas` f4 (n,) m^2, `cp` f2 (n,), `cf` f2 (n, 3)
(the stored shear sign: stress on the fluid; see physics.forces). `pool_idx` i4 (N,)
indexes the native cells.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aerosurr.config import Freestream, SurfaceFields
from aerosurr.data.geometry import polygon_centres_and_area_vectors
from aerosurr.data.splits import run_number
from aerosurr.data.validate import CSV_KEYS, read_force_csv, read_geo_refs, within_tolerance
from aerosurr.data.vtp import read_vtp
from aerosurr.physics.coefficients import pressure_coefficient, skin_friction_coefficient
from aerosurr.physics.forces import integrate

INDEX = "index.json"
# numpy's stubs match **arrays against `allow_pickle: bool`; the call itself is fine.
_savez: Any = np.savez
STORE_VERSION = 1


@dataclass(frozen=True)
class StoreEntry:
    run: str
    n_cells: int
    pool: int
    full: bool
    path: str
    seconds: float
    coeffs: dict[str, dict[str, float]]  # "varying"/"constant" -> integrated from raw fields
    ref: dict[str, dict[str, float]]  # force_mom*_i.csv
    passed: bool


def pool_indices(n_cells: int, size: int, seed: int) -> NDArray[np.int32]:
    """A fixed, sorted random subset of cell ids (deterministic per run)."""
    rng = np.random.default_rng(seed)
    k = min(size, n_cells)
    return np.sort(rng.choice(n_cells, size=k, replace=False)).astype(np.int32)


def build_entry(
    raw: Path,
    store: Path,
    run: str,
    fields: SurfaceFields,
    fs: Freestream,
    pool_size: int,
    full: bool,
    seed: int = 0,
) -> StoreEntry:
    t0 = time.perf_counter()
    n = run_number(run)
    surf = read_vtp(raw / run / f"boundary_{n}.vtp")
    centres, a_in = polygon_centres_and_area_vectors(surf.points, surf.offsets, surf.connectivity)
    areas = np.linalg.norm(a_in, axis=1)
    normals = -a_in / areas[:, None]  # out of the car
    cp = pressure_coefficient(surf.cell_data[fields.pressure], fs.q_inf, fs.p_inf)
    cf = skin_friction_coefficient(surf.cell_data[fields.wall_shear_stress], fs.q_inf)

    refs = read_geo_refs(raw / run / f"geo_ref_{n}.csv")
    coeffs = {
        "varying": integrate(cp, cf, a_in, centres, refs.varying).as_dict(),
        "constant": integrate(cp, cf, a_in, centres, refs.constant).as_dict(),
    }
    ref = {
        "varying": read_force_csv(raw / run / f"force_mom_{n}.csv"),
        "constant": read_force_csv(raw / run / f"force_mom_constref_{n}.csv"),
    }
    passed = all(within_tolerance(k, coeffs[w][k], ref[w][k]) for w in ref for k in CSV_KEYS)

    idx = pool_indices(surf.n_cells, pool_size, seed + n)
    arrays: dict[str, NDArray[Any]] = {"pool_idx": idx}

    def put(prefix: str, sel: NDArray[np.int32] | slice) -> None:
        arrays[f"{prefix}_centres"] = centres[sel].astype(np.float32)
        arrays[f"{prefix}_normals"] = normals[sel].astype(np.float32)
        arrays[f"{prefix}_areas"] = areas[sel].astype(np.float32)
        arrays[f"{prefix}_cp"] = cp[sel].astype(np.float16)
        arrays[f"{prefix}_cf"] = cf[sel].astype(np.float16)

    put("pool", idx)
    if full:
        put("full", slice(None))
    store.mkdir(parents=True, exist_ok=True)
    out = store / f"{run}.npz"
    tmp = store / f".{run}.tmp.npz"
    _savez(tmp, **arrays)  # uncompressed: loads at disk speed
    tmp.replace(out)
    return StoreEntry(run, surf.n_cells, len(idx), full, out.name, time.perf_counter() - t0,
                      coeffs, ref, passed)  # fmt: skip


def load_index(store: Path) -> dict[str, StoreEntry]:
    f = store / INDEX
    if not f.is_file():
        return {}
    d = json.loads(f.read_text())
    return {k: StoreEntry(**v) for k, v in d["runs"].items()}


def save_index(store: Path, entries: dict[str, StoreEntry], meta: dict[str, Any]) -> None:
    payload = {"version": STORE_VERSION, **meta,
               "runs": {k: asdict(v) for k, v in sorted(entries.items())}}  # fmt: skip
    tmp = store / f".{INDEX}.tmp"
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(store / INDEX)


def load_group(store: Path, run: str, group: str = "pool") -> dict[str, NDArray[Any]]:
    """`{"centres", "normals", "areas", "cp", "cf"}` of one run's `pool` or `full` group."""
    with np.load(store / f"{run}.npz") as z:
        keys = [k for k in z.files if k.startswith(f"{group}_")]
        if not keys:
            raise KeyError(f"{run}: no '{group}' group in the store")
        return {k.removeprefix(f"{group}_"): z[k] for k in keys}


STATS = "stats.json"


def compute_stats(store: Path, runs: list[str]) -> dict[str, Any]:
    """Normalisation statistics from the pools of `runs` (SPEC: `full_train` only)."""
    cp_sum = cp_sq = 0.0
    cf_sum = np.zeros(3)
    cf_sq = np.zeros(3)
    n = 0
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for r in runs:
        g = load_group(store, r, "pool")
        cp = g["cp"].astype(np.float64)
        cf = g["cf"].astype(np.float64)
        cp_sum += cp.sum()
        cp_sq += (cp**2).sum()
        cf_sum += cf.sum(axis=0)
        cf_sq += (cf**2).sum(axis=0)
        n += cp.size
        lo = np.minimum(lo, g["centres"].min(axis=0))
        hi = np.maximum(hi, g["centres"].max(axis=0))
    cp_mean = cp_sum / n
    cf_mean = cf_sum / n
    return {
        "runs": len(runs),
        "points": n,
        "cp_mean": float(cp_mean),
        "cp_std": float(np.sqrt(cp_sq / n - cp_mean**2)),
        "cf_mean": cf_mean.tolist(),
        "cf_std": np.sqrt(cf_sq / n - cf_mean**2).tolist(),
        "pos_centre": ((lo + hi) / 2).tolist(),
        "pos_scale": float((hi - lo).max() / 2),  # isotropic: keeps the shape undistorted
    }
