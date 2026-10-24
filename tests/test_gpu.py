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
