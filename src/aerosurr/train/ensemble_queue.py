"""Two-GPU work queue for deep-ensemble members (SPEC §5.4, ensemble_v2).

Members are claimed atomically (`mkdir` of a claim directory), so each member runs once
and on one GPU. Two kinds of worker take them:

- `primary` (GPU 1, the ensemble GPU): claims members until none are left, waits for any
  member still running elsewhere, then calibrates the ensemble and writes the report.
- `filler` (GPU 0, the main-model GPU): runs a member only while the multi-fidelity queue
  (`scripts/aero-mf-queue.sh`) is waiting for RANS data, and only when the predicted wait
  is longer than a member; or, with `after_mf_done`, only once the MF queue is done.
  As a hard guarantee it pre-empts its member within seconds of
  any MF/HF train or eval process appearing; the member is discarded and re-queued, to
  start again from scratch (a member never resumes on the other GPU).
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("aerosurr.ens_queue")

MF_STAGES = (11, 25, 50)
_MF_NAME = re.compile(r"^(mf|hf)_n\d+$")


# --------------------------------------------------------------------------- claims


def member_name(prefix: str, k: int) -> str:
    return f"{prefix}_m{k}"


def claim(claims: Path, k: int) -> bool:
    claims.mkdir(parents=True, exist_ok=True)
    try:
        (claims / f"m{k}").mkdir()
        return True
    except FileExistsError:
        return False


def release(claims: Path, k: int) -> None:
    shutil.rmtree(claims / f"m{k}", ignore_errors=True)


def is_done(runs: Path, name: str) -> bool:
    return (runs / name / "summary.json").is_file()


# --------------------------------------------------------------------------- MF queue state


def _argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def is_mf_gpu_argv(argv: list[str]) -> bool:
    """True for an `aerosurr train run --name mf_n*/hf_n*` or `aerosurr eval run .../mf_n*`."""
    if "aerosurr" not in " ".join(argv[:3]) and not any(a.endswith("/aerosurr") for a in argv):
        return False
    for i in range(len(argv) - 1):
        if argv[i] == "train" and argv[i + 1] == "run" and "--name" in argv:
            j = argv.index("--name")
            return j + 1 < len(argv) and bool(_MF_NAME.match(argv[j + 1]))
        if argv[i] == "eval" and argv[i + 1] == "run" and i + 2 < len(argv):
            return bool(_MF_NAME.match(Path(argv[i + 2].rstrip("/")).name))
    return False


def mf_processes(exclude: set[int] | None = None) -> list[int]:
    """PIDs of running MF/HF train or eval processes (exact argv match, never a shell)."""
    exclude = exclude or set()
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) not in exclude and is_mf_gpu_argv(_argv(int(d.name))):
            out.append(int(d.name))
    return out


@dataclass(frozen=True)
class MfState:
    stage: str  # "waiting", "training", "final", "done"
    needed_cases: int | None  # RANS cases that must be finished before MF needs GPU 0


def parse_mf_log(text: str, total_cases: int = 100) -> MfState:
    """Where `aero-mf-queue.sh` is, from its log."""
    if "MF queue done" in text:
        return MfState("done", None)
    if "waiting for the RANS batch to finish" in text or "final evaluation" in text:
        return MfState("final", total_cases)
    for n in MF_STAGES:
        started = re.search(rf"\] N={n}: run_", text)
        if not started:
            waiting = re.search(rf"\] N={n}: waiting", text)
            return MfState("waiting" if waiting else "training", 2 * n)
    return MfState("training", None)


def rans_progress(batch_log: str) -> tuple[int, float | None]:
    """(cases finished, mean minutes per case over the last 10) from the RANS batch log."""
    stamps = re.findall(
        r"^\S+ (\d\d:\d\d:\d\d) INFO\s+slot \w+: \S+ (?:ok|FAILED)", batch_log, re.MULTILINE
    )
    if len(stamps) < 2:
        return len(stamps), None
    t = [datetime.strptime(s, "%H:%M:%S") for s in stamps[-11:]]
    mins = [((b - a) % timedelta(days=1)).total_seconds() / 60 for a, b in itertools.pairwise(t)]
    return len(stamps), sum(mins) / len(mins)


def gpu0_free_for(mf_log: str, batch_log: str, mf_running: bool) -> timedelta:
    """How long GPU 0 is predicted to stay free of MF work (0 if MF is using it now)."""
    if mf_running:
        return timedelta(0)
    st = parse_mf_log(mf_log)
    if st.stage == "done":
        return timedelta(days=365)
    if st.needed_cases is None or st.stage == "training":
        return timedelta(0)
    done, per_case = rans_progress(batch_log)
    if per_case is None:
        return timedelta(0)
    return timedelta(minutes=max(0, st.needed_cases - done) * per_case)


# --------------------------------------------------------------------------- workers


@dataclass(frozen=True)
class Job:
    prefix: str  # member run names: <prefix>_m<k>
    members: int
    model: str
    train: str
    seed0: int
    runs: Path
    claims: Path


def train_cmd(exe: str, job: Job, k: int, cpus: str, session: str) -> list[str]:
    return [
        exe,
        "train",
        "run",
        "--model",
        job.model,
        "--train",
        job.train,
        "--seed",
        str(job.seed0 + k),
        "--name",
        member_name(job.prefix, k),
        "--cpus",
        cpus,
        "--session",
        session,
    ]


def _start(cmd: list[str], gpu: int) -> subprocess.Popen[bytes]:
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(
        cmd, env=env, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _kill(p: subprocess.Popen[bytes], grace_s: float = 15.0) -> None:
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(grace_s)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    except ProcessLookupError:
        pass


def next_member(job: Job) -> int | None:
    """Claim the lowest member not done and not claimed; None when none is left."""
    for k in range(job.members):
        if not is_done(job.runs, member_name(job.prefix, k)) and claim(job.claims, k):
            return k
    return None


def all_done(job: Job) -> bool:
    return all(is_done(job.runs, member_name(job.prefix, k)) for k in range(job.members))


def run_primary(
    job: Job,
    exe: str,
    gpu: int,
    cpus: str,
    session: str,
    wait_for: Path | None,
    finish: Callable[[], None],
    poll_s: float = 60.0,
) -> None:
    """Claim and run members on `gpu`; when every member is done, call `finish`."""
    if wait_for is not None:
        log.info("primary: waiting for %s", wait_for)
        while not wait_for.is_file():
            time.sleep(poll_s)
    while (k := next_member(job)) is not None:
        name = member_name(job.prefix, k)
        log.info("primary: %s on GPU %d", name, gpu)
        rc = _start(train_cmd(exe, job, k, cpus, session), gpu).wait()
        if rc != 0 or not is_done(job.runs, name):
            log.error("primary: %s failed (exit %d); released for retry", name, rc)
            shutil.rmtree(job.runs / name, ignore_errors=True)
            release(job.claims, k)
            time.sleep(poll_s)
    log.info("primary: no unclaimed members; waiting for the rest to finish")
    while not all_done(job):
        time.sleep(poll_s)
    finish()


def run_filler(
    job: Job,
    exe: str,
    gpu: int,
    cpus: str,
    session: str,
    mf_log: Path,
    batch_log: Path,
    member_h: float,
    margin_h: float,
    poll_s: float = 120.0,
    watch_s: float = 3.0,
    after_mf_done: bool = False,
) -> None:
    """Run members on `gpu` only in predicted MF gaps (or, with `after_mf_done`, only once
    the MF queue has logged that it is done); pre-empt the moment MF needs it."""
    me = os.getpid()
    need = timedelta(hours=member_h + margin_h)
    last = ""
    while not all_done(job):
        busy = bool(mf_processes({me}))
        mf_text = _read(mf_log)
        free = gpu0_free_for(mf_text, _read(batch_log), busy)
        if after_mf_done and parse_mf_log(mf_text).stage != "done":
            free = timedelta(0)
        if free < need:
            why = "MF is using GPU 0" if busy else f"MF stage {parse_mf_log(mf_text).stage}"
            msg = f"filler: waiting ({why}; predicted gap {free} < {need})"
            if msg.split(";")[0] != last.split(";")[0]:  # log state changes, not every poll
                log.info(msg)
            last = msg
            time.sleep(poll_s)
            continue
        last = ""
        k = next_member(job)
        if k is None:  # everything claimed: the primary finishes the rest
            log.info("filler: nothing left to claim")
            return
        name = member_name(job.prefix, k)
        log.info("filler: %s on GPU %d (MF predicted to need it in %s)", name, gpu, free)
        p = _start(train_cmd(exe, job, k, cpus, session), gpu)
        pre_empted = False
        while p.poll() is None:
            if mf_processes({me, p.pid}):
                log.warning(
                    "filler: MF needs GPU %d; pre-empting %s (restarts from scratch)", gpu, name
                )
                _kill(p)
                pre_empted = True
                break
            time.sleep(watch_s)
        if pre_empted or p.returncode != 0 or not is_done(job.runs, name):
            shutil.rmtree(job.runs / name, ignore_errors=True)
            release(job.claims, k)
            if not pre_empted:
                log.error("filler: %s failed (exit %s); released", name, p.returncode)
        else:
            log.info("filler: %s done", name)


def _read(p: Path) -> str:
    try:
        return p.read_text(errors="replace")
    except OSError:
        return ""
