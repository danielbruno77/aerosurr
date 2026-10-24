"""Official DrivAerML splits (`splits/manifest.json`)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

MANIFEST_PATH = "splits/manifest.json"
_RUN_RE = re.compile(r"run_(\d+)")


def load_manifest(path: Path) -> dict[str, list[str]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(
        isinstance(v, list) and all(isinstance(r, str) for r in v) for v in data.values()
    ):
        raise ValueError(f"{path}: expected a mapping of split name -> list of run ids")
    return {str(k): [str(r) for r in v] for k, v in data.items()}


def run_number(run_id: str) -> int:
    m = _RUN_RE.fullmatch(run_id)
    if not m:
        raise ValueError(f"not a run id: {run_id!r}")
    return int(m.group(1))


def priority_runs(manifest: dict[str, list[str]], priority: Iterable[str]) -> list[str]:
    """Runs in the order of the named splits, each run once (first occurrence wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for name in priority:
        if name not in manifest:
            raise KeyError(f"split {name!r} not in manifest; have {sorted(manifest)}")
        for run in manifest[name]:
            if run not in seen:
                seen.add(run)
                out.append(run)
    return out


def all_runs(manifest: dict[str, list[str]]) -> list[str]:
    """Every public run (the union of the `full` partition), sorted by run number."""
    runs = {r for k in ("full_train", "full_val", "full_test") for r in manifest[k]}
    return sorted(runs, key=run_number)
