"""Run an OpenFOAM case step by step, pinned to a CPU set, and record timings and forces."""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("aerosurr.rans")

OPENFOAM_BASHRC = Path("/usr/lib/openfoam/openfoam2506/etc/bashrc")
RESULT_NAME = "aerosurr_run.json"


def parse_cpus(spec: str) -> list[int]:
    """`"0-3,8,10-11"` -> [0, 1, 2, 3, 8, 10, 11]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        a, b = int(lo), int(hi or lo)
        if b < a:
            raise ValueError(f"bad CPU range {part!r}")
        out.extend(range(a, b + 1))
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate CPUs in {spec!r}")
    return out


def openfoam_env(bashrc: Path = OPENFOAM_BASHRC) -> dict[str, str]:
    """The environment after `source <bashrc>`, captured once."""
    raw = subprocess.run(
        ["bash", "-c", f"source {bashrc} >/dev/null 2>&1 && env -0"],
        check=True,
        capture_output=True,
    ).stdout
    env = dict(item.split("=", 1) for item in raw.decode().split("\0") if "=" in item)
    env["OMP_NUM_THREADS"] = "1"
    return env


def write_rankfile(path: Path, cpus: Sequence[int]) -> Path:
    """OpenMPI rankfile: rank i on physical core cpus[i] (and its SMT sibling)."""
    path.write_text("".join(f"rank {i}=localhost slot={c}\n" for i, c in enumerate(cpus)))
    return path


def mpirun_cmd(app: Sequence[str], n_ranks: int, rankfile: Path) -> list[str]:
    """`mpirun` with one rank per rankfile line. (OpenMPI 4.1 rejects --cpu-set with
    --map-by, and --cpu-set alone confines ranks to the set without binding each one.)"""
    return [
        "mpirun", "-np", str(n_ranks), "--rankfile", str(rankfile), "--report-bindings",
        *app, "-parallel",
    ]  # fmt: skip


@dataclass
class RunResult:
    case: str
    preset: str
    n_ranks: int
    cpus: str
    status: str = "running"
    failed_step: str | None = None
    steps: dict[str, float] = field(default_factory=dict)  # wall seconds per step
    cells: int | None = None
    iterations: int | None = None
    s_per_iter: float | None = None
    coeffs: dict[str, float] = field(default_factory=dict)  # mean over the last N iterations
    coeffs_std: dict[str, float] = field(default_factory=dict)
    reference: dict[str, float] = field(default_factory=dict)  # HF force_mom_i.csv
    surface: str | None = None  # car surface (p, wallShearStress, yPlus), relative to the case
    disk_mb: float | None = None
    started: str = ""
    finished: str = ""

    @property
    def total_s(self) -> float:
        return sum(self.steps.values())

    def save(self, case: Path) -> None:
        d = asdict(self) | {"total_s": self.total_s}
        (case / RESULT_NAME).write_text(json.dumps(d, indent=2))


_CELLS = re.compile(r"^\s+cells:\s+(\d+)", re.MULTILINE)
_EXEC = re.compile(r"^ExecutionTime = ([\d.eE+-]+) s", re.MULTILINE)


def parse_cells(check_mesh_log: str) -> int | None:
    m = _CELLS.findall(check_mesh_log)
    return int(m[-1]) if m else None


def parse_s_per_iter(solver_log: str, last: int = 200) -> tuple[int, float | None]:
    """Iterations done and the median wall time per iteration over the last `last`."""
    t = [float(x) for x in _EXEC.findall(solver_log)]
    if len(t) < 3:
        return len(t), None
    dt = [b - a for a, b in itertools.pairwise(t[1:])][-last:]  # skip the first (setup)
    return len(t), statistics.median(dt)


def read_coefficients(path: Path) -> dict[str, list[float]]:
    """`postProcessing/forceCoeffs1/0/coefficient.dat` -> column name -> values."""
    cols: list[str] = []
    data: list[list[float]] = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            names = line.lstrip("#").split()
            if names and names[0] == "Time":
                cols = names
            continue
        if line.strip():
            data.append([float(x) for x in line.split()])
    if not cols:
        raise ValueError(f"{path}: no '# Time ...' header")
    return {c: [row[i] for row in data] for i, c in enumerate(cols)}


def summarise_coefficients(
    series: dict[str, list[float]], last: int
) -> tuple[dict[str, float], dict[str, float]]:
    mean, std = {}, {}
    for name in ("Cd", "Cl", "Cs", "Cl(f)", "Cl(r)", "CmPitch"):
        if series.get(name):
            v = series[name][-last:]
            mean[name] = statistics.fmean(v)
            std[name] = statistics.pstdev(v)
    return mean, std


def read_force_mom(path: Path) -> dict[str, float]:
    header, row = path.read_text().strip().splitlines()[:2]
    return {k.strip(): float(v) for k, v in zip(header.split(","), row.split(","), strict=True)}


def _restore0(case: Path) -> None:
    """Copy 0.orig into every processor directory (the parallel `restore0Dir -processor`)."""
    procs = sorted(case.glob("processor*"))
    if not procs:
        raise RuntimeError(f"{case}: no processor directories")
    for p in procs:
        shutil.rmtree(p / "0", ignore_errors=True)
        shutil.copytree(case / "0.orig", p / "0")


def run_case(
    case: Path,
    cpus: Sequence[int],
    preset: str,
    potential_init: bool = True,
    average_last: int = 300,
    reference: dict[str, float] | None = None,
    env: dict[str, str] | None = None,
    keep_volume: bool = False,
) -> RunResult:
    """Mesh and solve `case` on `cpus` (one MPI rank per CPU). Writes `aerosurr_run.json`."""
    env = env or openfoam_env()
    cpu_list = ",".join(map(str, cpus))
    res = RunResult(case.name, preset, len(cpus), cpu_list, reference=reference or {})
    res.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    serial = ["taskset", "-c", cpu_list]
    rf = write_rankfile(case / "rankfile", cpus)
    n = len(cpus)
    steps: list[tuple[str, list[str] | None]] = [
        ("surfaceFeatureExtract", [*serial, "surfaceFeatureExtract"]),
        ("blockMesh", [*serial, "blockMesh"]),
        ("decomposePar", [*serial, "decomposePar", "-force"]),
        ("snappyHexMesh", mpirun_cmd(["snappyHexMesh", "-overwrite"], n, rf)),
        ("restore0", None),
        ("checkMesh", mpirun_cmd(["checkMesh", "-constant"], n, rf)),
    ]
    if potential_init:
        steps.append(("potentialFoam", mpirun_cmd(["potentialFoam", "-writephi"], n, rf)))
    steps.append(("simpleFoam", mpirun_cmd(["simpleFoam"], n, rf)))

    for name, cmd in steps:
        log.info("%s: %s started", case.name, name)
        t0 = time.monotonic()
        try:
            if cmd is None:
                _restore0(case)
            else:
                with (case / f"log.{name}").open("w") as out:
                    subprocess.run(cmd, cwd=case, env=env | {"PWD": str(case)}, stdout=out,
                                   stderr=subprocess.STDOUT,
                                   check=True)  # fmt: skip
        except (subprocess.CalledProcessError, OSError, RuntimeError) as e:
            res.steps[name] = time.monotonic() - t0
            res.status, res.failed_step = "failed", name
            res.finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            res.save(case)
            log.error("%s: %s failed: %s (see %s)", case.name, name, e, case / f"log.{name}")
            return res
        res.steps[name] = time.monotonic() - t0
        log.info("%s: %s done in %.1f s", case.name, name, res.steps[name])
        if name == "checkMesh":
            res.cells = parse_cells((case / "log.checkMesh").read_text())
            log.info("%s: %s cells", case.name, f"{res.cells:,}" if res.cells else "?")
        res.save(case)

    surfaces = sorted(case.glob("postProcessing/carSurface/*/car.vtp"))
    res.surface = str(surfaces[-1].relative_to(case)) if surfaces else None
    if res.surface is None:
        log.warning("%s: no car surface written (postProcessing/carSurface)", case.name)
    t0 = time.monotonic()
    if keep_volume:
        for name, cmd in (("reconstructParMesh", ["reconstructParMesh", "-constant"]),
                          ("reconstructPar", ["reconstructPar", "-latestTime"])):  # fmt: skip
            with (case / f"log.{name}").open("w") as out:
                subprocess.run([*serial, *cmd], cwd=case, env=env | {"PWD": str(case)},
                               stdout=out, stderr=subprocess.STDOUT, check=True)  # fmt: skip
    if res.surface is not None or keep_volume:
        for proc in case.glob("processor*"):
            shutil.rmtree(proc)
    res.steps["cleanup"] = time.monotonic() - t0
    res.disk_mb = sum(f.stat().st_size for f in case.rglob("*") if f.is_file()) / 1e6
    res.iterations, res.s_per_iter = parse_s_per_iter((case / "log.simpleFoam").read_text())
    coeff_file = case / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
    if coeff_file.is_file():
        res.coeffs, res.coeffs_std = summarise_coefficients(
            read_coefficients(coeff_file), average_last
        )
    res.status = "done"
    res.finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    res.save(case)
    log.info("%s: done in %.1f min; %s", case.name, res.total_s / 60, res.coeffs)
    return res


def cpu_count_ok(cpus: Sequence[int]) -> None:
    online = os.sched_getaffinity(0)
    missing = sorted(set(cpus) - online)
    if missing:
        raise ValueError(f"CPUs not available to this process: {missing}")
