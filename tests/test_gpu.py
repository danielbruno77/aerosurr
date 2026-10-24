"""GPU stack smoke test (`-m gpu`); the full benchmark is `aerosurr env check`."""

from __future__ import annotations

import pytest

from aerosurr.envcheck import SmiGpu, compile_smoke, idle_check

pytestmark = pytest.mark.gpu


def test_compile_smoke_gpu0() -> None:
    import torch

    assert "sm_120" in torch.cuda.get_arch_list()
    with torch.cuda.device(0):
        _, _, rel, _, _ = compile_smoke(0, iters=2)
    assert rel < 1e-2


def test_idle_check_flags_busy_gpu() -> None:
    g = SmiGpu(0, "00000000:01:00.0", "x", "P2", 300, 600, 3150, 14001, 90, 1000, 32000, 60, "d")
    ok, notes = idle_check([g], ["pid 1"])
    assert not ok and len(notes) == 3


def test_pointnet_compiled_bf16_gradients_finite() -> None:
    """Regression: compiled bf16 amax over many points gave NaN gradients."""
    import torch

    from aerosurr.config import ModelConfig
    from aerosurr.models.surface import build_model

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    m = build_model(ModelConfig(kind="pointnet")).to(dev)
    f = torch.compile(m)
    pos = torch.rand(1, 50_000, 3, device=dev)  # bf16 rounding makes many maxima tie
    nrm = torch.nn.functional.normalize(torch.randn(1, 50_000, 3, device=dev), dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = f(pos, nrm)
    out.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    torch._dynamo.reset()
