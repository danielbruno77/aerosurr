"""GPU stack check (SPEC §9 Day 0): idle state, torch/CUDA build, bf16 GEMM and torch.compile."""

from __future__ import annotations

import csv
import io
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

SMI_FIELDS = (
    "index",
    "pci.bus_id",
    "name",
    "pstate",
    "power.draw",
    "power.limit",
    "clocks.max.sm",
    "clocks.max.mem",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "driver_version",
)


@dataclass(frozen=True)
class SmiGpu:
    index: int
    pci_bus_id: str
    name: str
    pstate: str
    power_w: float
    power_limit_w: float
    clock_sm_max_mhz: int
    clock_mem_max_mhz: int
    util_pct: int
    mem_used_mib: int
    mem_total_mib: int
    temp_c: int
    driver: str


def _num(s: str) -> float:
    return float(s.strip().split()[0])


def query_smi() -> tuple[list[SmiGpu], list[str]]:
    """GPUs and compute processes as nvidia-smi sees them (no CUDA context is created)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise RuntimeError("nvidia-smi not found")
    q = subprocess.run(
        [smi, f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    gpus = []
    for row in csv.reader(io.StringIO(q)):
        r = [c.strip() for c in row]
        gpus.append(
            SmiGpu(
                index=int(r[0]),
                pci_bus_id=r[1],
                name=r[2],
                pstate=r[3],
                power_w=_num(r[4]),
                power_limit_w=_num(r[5]),
                clock_sm_max_mhz=int(_num(r[6])),
                clock_mem_max_mhz=int(_num(r[7])),
                util_pct=int(_num(r[8])),
                mem_used_mib=int(_num(r[9])),
                mem_total_mib=int(_num(r[10])),
                temp_c=int(_num(r[11])),
                driver=r[12],
            )
        )
    apps = subprocess.run(
        [
            smi,
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return gpus, [line.strip() for line in apps.splitlines() if line.strip()]


@dataclass
class DeviceResult:
    index: int
    name: str
    pci_bus_id: str
    capability: str
    total_mem_gib: float
    sm_count: int
    gemm_tflops: dict[int, float] = field(default_factory=dict)  # n -> bf16 TFLOPS
    compile_ok: bool = False
    compile_first_call_s: float = 0.0
    compile_max_abs_err: float = float("nan")
    compile_rel_err: float = float("nan")  # ||compiled - eager|| / ||eager||
    eager_ms: float = float("nan")
    compiled_ms: float = float("nan")
    error: str = ""


@dataclass
class EnvReport:
    timestamp: str
    host: str
    python: str
    torch: str
    torch_cuda: str
    cudnn: str
    arch_list: list[str]
    triton: str
    smi_before: list[SmiGpu]
    apps_before: list[str]
    idle_ok: bool
    idle_notes: list[str]
    expected_devices: int
    device_count: int
    devices: list[DeviceResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.idle_ok
            and self.device_count == self.expected_devices
            and "sm_120" in self.arch_list
            and len(self.devices) == self.expected_devices
            and all(d.compile_ok and d.gemm_tflops and not d.error for d in self.devices)
        )

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d


def idle_check(gpus: list[SmiGpu], apps: list[str]) -> tuple[bool, list[str]]:
    notes = []
    if apps:
        notes.append(f"{len(apps)} compute process(es) running: {apps}")
    for g in gpus:
        if g.pstate != "P8":
            notes.append(f"GPU {g.index} in {g.pstate}, not P8")
        if g.util_pct > 5:
            notes.append(f"GPU {g.index} utilisation {g.util_pct} %")
    return not notes, notes


MAX_PLAUSIBLE_TFLOPS = 2000.0  # sanity bound: dense bf16 on any current GPU is far below


def bench_gemm(device: int, n: int, iters: int) -> float:
    """Dense bf16 GEMM throughput (TFLOPS, fp32 accumulate) on one device."""
    import torch

    dev = torch.device("cuda", device)
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize(dev)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        c = a @ b
    end.record()
    torch.cuda.synchronize(dev)
    del a, b, c
    torch.cuda.empty_cache()
    return 2.0 * n**3 * iters / (start.elapsed_time(end) * 1e-3) / 1e12


def compile_smoke(device: int, iters: int = 20) -> tuple[float, float, float, float, float]:
    """Compile a small attention + MLP block (fwd+bwd, bf16) and compare with eager.

    Returns (first-call seconds incl. compilation, max |compiled - eager|,
    ||compiled - eager|| / ||eager||, eager ms, compiled ms).
    """
    import torch
    import torch.nn.functional as F

    torch._dynamo.reset()
    dev = torch.device("cuda", device)
    torch.manual_seed(0)
    d, h = 256, 8
    w_qkv = torch.randn(d, 3 * d, device=dev, dtype=torch.bfloat16) * d**-0.5
    w1 = torch.randn(d, 4 * d, device=dev, dtype=torch.bfloat16) * d**-0.5
    w2 = torch.randn(4 * d, d, device=dev, dtype=torch.bfloat16) * (4 * d) ** -0.5

    def block(x: torch.Tensor) -> torch.Tensor:
        bsz, n, _ = x.shape
        q, k, v = (x @ w_qkv).view(bsz, n, 3, h, d // h).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, n, d)
        y = F.layer_norm(x + att, (d,))
        return y + F.gelu(y @ w1) @ w2

    x = torch.randn(4, 2048, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(block)

    def step(fn: Any) -> torch.Tensor:
        out: torch.Tensor = fn(x)
        out.float().square().mean().backward()
        return out

    t0 = time.perf_counter()
    out_c = step(compiled)
    torch.cuda.synchronize(dev)
    first = time.perf_counter() - t0
    out_e = step(block)
    diff = out_c.detach().float() - out_e.detach().float()
    err = float(diff.abs().max())
    rel = float(diff.norm() / out_e.detach().float().norm())

    def timed(fn: Any) -> float:
        for _ in range(3):
            step(fn)
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        for _ in range(iters):
            step(fn)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t) / iters * 1e3

    return first, err, rel, timed(block), timed(compiled)


def run_check(
    devices: list[int] | None = None,
    sizes: tuple[int, ...] = (8192, 16384),
    iters: int = 50,
    expected_devices: int = 2,
    max_compile_rel_err: float = 1e-2,
) -> EnvReport:
    smi, apps = query_smi()  # before torch touches the GPUs
    idle_ok, notes = idle_check(smi, apps)

    import torch

    try:
        import triton

        triton_v = str(triton.__version__)
    except ImportError:
        triton_v = "not installed"
    rep = EnvReport(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        host=platform.node(),
        python=platform.python_version(),
        torch=torch.__version__,
        torch_cuda=str(torch.version.cuda),
        cudnn=str(torch.backends.cudnn.version()),
        arch_list=list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else [],
        triton=triton_v,
        smi_before=smi,
        apps_before=apps,
        idle_ok=idle_ok,
        idle_notes=notes,
        expected_devices=expected_devices,
        device_count=torch.cuda.device_count(),
    )
    for i in devices if devices is not None else range(rep.device_count):
        p = torch.cuda.get_device_properties(i)
        dr = DeviceResult(
            index=i,
            name=p.name,
            pci_bus_id="{:08X}:{:02X}:{:02X}.0".format(
                *(getattr(p, k) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
            ),
            capability=f"{p.major}.{p.minor}",
            total_mem_gib=p.total_memory / 2**30,
            sm_count=p.multi_processor_count,
        )
        try:
            # Events, streams and compiled kernels bind to the *current* device.
            with torch.cuda.device(i):
                for n in sizes:
                    dr.gemm_tflops[n] = bench_gemm(i, n, iters)
                first, err, rel, eager, comp = compile_smoke(i)
            dr.compile_first_call_s, dr.compile_max_abs_err, dr.compile_rel_err = first, err, rel
            dr.eager_ms, dr.compiled_ms = eager, comp
            dr.compile_ok = rel <= max_compile_rel_err  # False for NaN too
            if any(not 1.0 < t < MAX_PLAUSIBLE_TFLOPS for t in dr.gemm_tflops.values()):
                dr.error = f"implausible GEMM timing {dr.gemm_tflops}"
        except Exception as e:  # report, don't crash: the report is the deliverable
            dr.error = f"{type(e).__name__}: {e}"
        rep.devices.append(dr)
    return rep


def render_markdown(rep: EnvReport, command: str) -> str:
    w: list[str] = []
    a = w.append
    a(f"- When: {rep.timestamp} on `{rep.host}`")
    a(f"- Command: `{command}`")
    a(f"- Result: **{'PASS' if rep.passed else 'FAIL'}**")
    a("")
    a("### Software\n")
    a("| Component | Version |")
    a("|---|---|")
    drv = rep.smi_before[0].driver if rep.smi_before else "?"
    for k, v in [
        ("NVIDIA driver", drv),
        ("Python", rep.python),
        ("torch", rep.torch),
        ("torch CUDA", rep.torch_cuda),
        ("cuDNN", rep.cudnn),
        ("Triton", rep.triton),
        ("torch arch list", " ".join(rep.arch_list)),
    ]:
        a(f"| {k} | {v} |")
    a("")
    a("### Idle state before the benchmark (nvidia-smi, before CUDA init)\n")
    a(
        "| GPU | Bus ID | P-state | Power (W) | Limit (W) | Max SM / mem clock (MHz) | Util | "
        "Mem used (MiB) | Temp (°C) |"
    )
    a("|---|---|---|---|---|---|---|---|---|")
    for g in rep.smi_before:
        a(
            f"| {g.index} | {g.pci_bus_id} | {g.pstate} | {g.power_w:.1f} | "
            f"{g.power_limit_w:.0f} | {g.clock_sm_max_mhz} / {g.clock_mem_max_mhz} | "
            f"{g.util_pct} % | {g.mem_used_mib} | "
            f"{g.temp_c} |"
        )
    a("")
    a(
        f"- Compute processes: {len(rep.apps_before)}"
        + (f" ({rep.apps_before})" if rep.apps_before else "")
    )
    a(
        f"- Idle check: **{'PASS' if rep.idle_ok else 'FAIL'}**"
        + (f" ({'; '.join(rep.idle_notes)})" if rep.idle_notes else "")
    )
    a(
        f"- torch device count: {rep.device_count} (expected {rep.expected_devices}); "
        f"`sm_120` in arch list: {'yes' if 'sm_120' in rep.arch_list else 'NO'}"
    )
    a("")
    a("### Per device\n")
    sizes = sorted({n for d in rep.devices for n in d.gemm_tflops})
    head = " | ".join(f"bf16 GEMM n={n} (TFLOPS)" for n in sizes)
    a(
        f"| torch idx | Name | Bus ID | CC | Mem (GiB) | SMs | {head} | compile first call (s) | "
        "compile max abs / rel L2 err | eager fwd+bwd (ms) | compiled fwd+bwd (ms) | OK |"
    )
    a("|" + "---|" * (10 + len(sizes)))
    for d in rep.devices:
        t = " | ".join(f"{d.gemm_tflops.get(n, float('nan')):.1f}" for n in sizes)
        ok = "yes" if d.compile_ok and not d.error else f"NO {d.error}"
        a(
            f"| {d.index} | {d.name} | {d.pci_bus_id} | {d.capability} | {d.total_mem_gib:.1f} | "
            f"{d.sm_count} | {t} | {d.compile_first_call_s:.1f} | "
            f"{d.compile_max_abs_err:.3g} / {d.compile_rel_err:.2g} | "
            f"{d.eager_ms:.2f} | {d.compiled_ms:.2f} | {ok} |"
        )
    a("")
    a(
        "GEMM: square bf16 `a @ b` (fp32 accumulate), timed with CUDA events after warm-up. "
        "Compile smoke test: `torch.compile` (inductor, default mode) of an attention + MLP block "
        "(B=4, N=2048, d=256, 8 heads), forward + backward, compared with eager; it passes when "
        "the relative L2 difference of the output is below 1e-2 (bf16 has a 2^-8 ulp). "
        "First-call time includes compilation and depends on whether the inductor/Triton "
        "caches under `~/aero/cache` are warm."
    )
    return "\n".join(w) + "\n"
