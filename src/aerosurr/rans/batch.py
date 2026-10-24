"""The low-fidelity RANS batch (SPEC §5.3): which runs, in what order, on which CPU slots."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("aerosurr.rans")


def rans_subset(manifest: dict[str, list[str]], n_test: int = 50, n_train: int = 50) -> list[str]:
    """Interleaved queue: test and train alternate, so any finished prefix is usable.

    Test runs are `full_test` in manifest order. Train runs are `super_scarce_train` first,
    then the rest of `scarce_train`, then `medium_train`, so the nested data-efficiency
    sets get low-fidelity coverage first.
    """
    test = manifest["full_test"][:n_test]
    train: list[str] = []
    for split in ("super_scarce_train", "scarce_train", "medium_train", "full_train"):
        for r in manifest[split]:
            if r not in train and len(train) < n_train:
                train.append(r)
    out: list[str] = []
    for i in range(max(len(test), len(train))):
        out += [x[i] for x in (test, train) if i < len(x)]
    return out


@dataclass(frozen=True)
class Slot:
    name: str
    cpus: list[int]


def case_dir(root: Path, run: str, preset: str, n_ranks: int) -> Path:
    return root / f"{run}_{preset}_np{n_ranks}"


def is_done(case: Path) -> bool:
    f = case / "aerosurr_run.json"
    return f.is_file() and json.loads(f.read_text()).get("status") == "done"


def run_queue(
    runs: Sequence[str],
    slots: Sequence[Slot],
    work: Callable[[str, Slot], bool],
    stop: threading.Event | None = None,
) -> dict[str, bool]:
    """Feed `runs` in order to the slots; each slot runs one case at a time.

    `work(run, slot)` sets up (if needed) and runs one case and returns success. Runs
    already done are the caller's to skip. Returns run -> success.
    """
    stop = stop or threading.Event()
    lock = threading.Lock()
    queue = list(runs)
    results: dict[str, bool] = {}

    def worker(slot: Slot) -> None:
        while not stop.is_set():
            with lock:
                if not queue:
                    return
                run = queue.pop(0)
            log.info("slot %s: %s", slot.name, run)
            ok = work(run, slot)
            with lock:
                results[run] = ok
            log.info("slot %s: %s %s (%d/%d done)", slot.name, run, "ok" if ok else "FAILED",
                     len(results), len(runs))  # fmt: skip

    threads = [threading.Thread(target=worker, args=(s,), name=s.name) for s in slots]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
