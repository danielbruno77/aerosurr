"""Side-load probe: the CPU and GPU work that runs next to OpenFOAM, measured on its own cores.

- CPU: worker processes read boundary VTPs and compute polygon centres/areas, the
  compact-store preprocessing workload (SPEC §5.1).
- GPU: PhysicsNeMo Transolver training steps (bf16, torch.compile) fed from host memory:
  each step samples points on the CPU and copies them to the GPU, as the loader will.

Run once on an idle machine and once while OpenFOAM runs; compare the numbers.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _read_one(path: str) -> tuple[str, float]:
    from aerosurr.data.geometry import polygon_centres_and_area_vectors
    from aerosurr.data.vtp import read_vtp

    t0 = time.perf_counter()
    s = read_vtp(Path(path))
    polygon_centres_and_area_vectors(s.points, s.offsets, s.connectivity)
    return path, time.perf_counter() - t0


@dataclass
class ProbeResult:
    cpus: list[int]
    seconds: float
    cpu_files: int = 0
    cpu_s_per_file: list[float] = field(default_factory=list)
    gpu_steps: int = 0
    gpu_step_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def med(x: list[float]) -> float | None:
            return statistics.median(x) if x else None

        return {
            "cpus": self.cpus,
            "seconds": self.seconds,
            "cpu_files": self.cpu_files,
            "cpu_median_s_per_file": med(self.cpu_s_per_file),
            "gpu_steps": self.gpu_steps,
            "gpu_median_step_ms": med(self.gpu_step_ms),
        }

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"summary": self.summary()}


def gpu_steps(device: int, n_points: int, deadline: float) -> list[float]:
    """Compiled Transolver fwd+bwd steps until `deadline`; returns per-step ms (after warm-up)."""
    import torch
    from physicsnemo.models.transolver import Transolver

    dev = torch.device("cuda", device)
    torch.manual_seed(0)
    n_car = 2_000_000  # a host-resident pool, like one compact-store entry
    pos = torch.randn(n_car, 3).pin_memory()
    nrm = torch.nn.functional.normalize(torch.randn(n_car, 3), dim=-1).pin_memory()
    target = torch.randn(n_car, 4).pin_memory()
    model = Transolver(functional_dim=3, embedding_dim=3, out_dim=4, n_layers=8, n_hidden=256,
                       n_head=8, slice_num=64, use_te=False).to(dev)  # fmt: skip
    step_fn = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times: list[float] = []
    warm = 3
    while time.monotonic() < deadline or warm > 0:
        t0 = time.perf_counter()
        idx = torch.randint(0, n_car, (n_points,))
        p = pos[idx].to(dev, non_blocking=True)[None]
        n = nrm[idx].to(dev, non_blocking=True)[None]
        y = target[idx].to(dev, non_blocking=True)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = step_fn(n, embedding=p)
        loss = (out.float() - y).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(dev)
        if warm > 0:
            warm -= 1
        else:
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def run_probe(
    cpus: list[int],
    vtps: list[Path],
    seconds: float,
    cpu_workers: int = 4,
    gpu: int | None = 0,
    n_points: int = 200_000,
) -> ProbeResult:
    """Pin this process (and its children) to `cpus`; run the CPU and GPU loads together."""
    os.sched_setaffinity(0, cpus)
    res = ProbeResult(cpus=cpus, seconds=seconds)
    deadline = time.monotonic() + seconds
    queue = [str(v) for v in vtps]

    def cpu_loop(pool: ProcessPoolExecutor) -> None:
        """Keep `cpu_workers` files in flight until the deadline (runs beside the GPU loop)."""
        futs = {pool.submit(_read_one, queue[i % len(queue)]) for i in range(cpu_workers)}
        submitted = len(futs)
        while futs:
            done, futs = wait(futs, return_when=FIRST_COMPLETED)
            for f in done:
                res.cpu_s_per_file.append(f.result()[1])
                if time.monotonic() < deadline:
                    futs.add(pool.submit(_read_one, queue[submitted % len(queue)]))
                    submitted += 1

    with ProcessPoolExecutor(max_workers=cpu_workers) as pool:
        t = threading.Thread(target=cpu_loop, args=(pool,), daemon=True)
        t.start()
        if gpu is not None:
            res.gpu_step_ms = gpu_steps(gpu, n_points, deadline)
            res.gpu_steps = len(res.gpu_step_ms)
        t.join()
    res.cpu_files = len(res.cpu_s_per_file)
    return res
