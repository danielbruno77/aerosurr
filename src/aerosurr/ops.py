"""Process-level setup shared by every CLI command: logging, cache dirs, device order."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def configure_runtime(workspace: Path) -> None:
    """Keep compiler caches inside the workspace and make CUDA indices match nvidia-smi.

    Only sets variables the caller has not set already.
    """
    cache = workspace / "cache"
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))


def setup_logging(log_dir: Path | None, tag: str, level: int = logging.INFO) -> Path | None:
    """Log to stdout and, if `log_dir` is given, to `<log_dir>/<timestamp>_<tag>.log`.

    `<log_dir>/latest.log` is re-pointed at the new file (SPEC §8).
    """
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger("aerosurr")
    root.setLevel(level)
    root.handlers.clear()
    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    root.addHandler(out)
    if log_dir is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{datetime.now():%Y%m%dT%H%M%S}_{tag}.log"
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    latest = log_dir / "latest.log"
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)
    return path
