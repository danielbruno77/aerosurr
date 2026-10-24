"""Ensemble work queue: claims, MF-gap prediction and pre-emption (with fake processes)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from aerosurr.train import ensemble_queue as eq

MF_LOG_WAITING_25 = """[12:48:57] waiting for the GeoTransolver run on GPU 0
[12:48:57] N=11: waiting for 11 training cars with LF fields
[12:48:58] N=11: run_39,run_49
2026-10-24 13:51:08 INFO    hf_n11: epoch 636 loss 0.0393
[14:22:01] N=25: waiting for 25 training cars with LF fields
"""


def batch_log(n: int, start_min: int = 0, step: float = 12.8) -> str:
    lines = []
    for i in range(n):
        m = start_min + i * step
        hh, mm, ss = int(m // 60) % 24, int(m % 60), int((m * 60) % 60)
        lines.append(
            f"2026-10-24 {hh:02d}:{mm:02d}:{ss:02d} INFO    slot A: run_{i} ok ({i + 1}/100 done)"
        )
    return "\n".join(lines) + "\n"


def test_argv_matching() -> None:
    exe = "/home/daniel/aero-surrogate-demo/.venv/bin/aerosurr"
    py = "/home/daniel/.venv/bin/python"
    assert eq.is_mf_gpu_argv(
        [py, exe, "train", "run", "--model", "transolver_mf", "--name", "mf_n25"]
    )
    assert eq.is_mf_gpu_argv(
        [py, exe, "eval", "run", "/home/daniel/aero/runs/hf_n11", "--split", "x"]
    )
    assert not eq.is_mf_gpu_argv([py, exe, "train", "run", "--name", "ens2_m0"])
    assert not eq.is_mf_gpu_argv([py, exe, "eval", "lf", "--split", "full_test"])
    # A shell whose command line merely contains the words must never match.
    assert not eq.is_mf_gpu_argv(["bash", "-c", f"{exe} train run --name mf_n25"])


def test_parse_mf_log_stages() -> None:
    assert eq.parse_mf_log(MF_LOG_WAITING_25) == eq.MfState("waiting", 50)
    training_11 = MF_LOG_WAITING_25.split("[14:22:01]")[0]
    assert eq.parse_mf_log(training_11).stage == "training"
    final = (
        MF_LOG_WAITING_25 + "[20:00:00] N=25: run_1\n[22:00:00] N=50: waiting\n"
        "[05:00:00] N=50: run_1\n[06:00:00] waiting for the RANS batch to finish\n"
    )
    assert eq.parse_mf_log(final) == eq.MfState("final", 100)
    assert eq.parse_mf_log(final + "[07:00:00] MF queue done\n").stage == "done"


def test_rans_progress_wraps_midnight() -> None:
    done, per = eq.rans_progress(batch_log(31, start_min=23 * 60))  # crosses 00:00
    assert done == 31 and per is not None and abs(per - 12.8) < 0.05


def test_gpu0_free_for() -> None:
    free = eq.gpu0_free_for(MF_LOG_WAITING_25, batch_log(31), mf_running=False)
    assert abs(free - timedelta(minutes=19 * 12.8)) < timedelta(minutes=1)
    assert eq.gpu0_free_for(MF_LOG_WAITING_25, batch_log(31), mf_running=True) == timedelta(0)
    training = MF_LOG_WAITING_25.split("[14:22:01]")[0]
    assert eq.gpu0_free_for(training, batch_log(31), mf_running=False) == timedelta(0)


def test_claims(tmp_path: Path) -> None:
    job = eq.Job("e", 3, "m", "t", 0, tmp_path / "runs", tmp_path / "claims")
    (tmp_path / "runs" / "e_m0").mkdir(parents=True)
    (tmp_path / "runs" / "e_m0" / "summary.json").write_text("{}")
    assert eq.next_member(job) == 1  # m0 is done
    assert eq.next_member(job) == 2  # m1 is claimed
    assert eq.next_member(job) is None
    eq.release(job.claims, 1)
    assert eq.next_member(job) == 1


FAKE_EXE = """\
import sys, time, pathlib, json
args = sys.argv[1:]
name = args[args.index("--name") + 1]
time.sleep(float(sys.argv[0].split("_")[-1]) if False else 1.5)
runs = pathlib.Path(__file__).parent / "runs" / name
runs.mkdir(parents=True, exist_ok=True)
(runs / "summary.json").write_text(json.dumps({"name": name}))
"""


def test_filler_pre_empts_for_mf_and_requeues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "aerosurr"
    real_scan = eq.mf_processes

    def only_ours(exclude: set[int] | None = None) -> list[int]:
        """The real machine may be running MF jobs: look only at this test's processes."""
        return [p for p in real_scan(exclude) if str(exe) in " ".join(eq._argv(p))]

    monkeypatch.setattr(eq, "mf_processes", only_ours)
    exe.write_text(f"#!{sys.executable}\n" + FAKE_EXE)
    exe.chmod(0o755)
    mf_log, rans_log = tmp_path / "mf.log", tmp_path / "rans.log"
    mf_log.write_text(MF_LOG_WAITING_25)
    rans_log.write_text(batch_log(31))
    job = eq.Job("ens2", 1, "m", "t", 100, tmp_path / "runs", tmp_path / "claims")
    t = threading.Thread(
        target=eq.run_filler,
        args=(job, str(exe), 0, "0", "s", mf_log, rans_log, 0.0001, 0.0, 0.2, 0.05),
        daemon=True,
    )
    t.start()
    deadline = time.time() + 10
    while not (job.claims / "m0").exists() and time.time() < deadline:  # member started
        time.sleep(0.05)
    assert (job.claims / "m0").exists()
    mf = subprocess.Popen([sys.executable, str(exe), "train", "run", "--name", "mf_n25"])
    time.sleep(0.6)  # the filler polls every 0.05 s
    assert not (job.runs / "ens2_m0" / "summary.json").exists()  # killed before finishing
    assert not (job.claims / "m0").exists()  # and re-queued
    mf.wait()  # MF "finishes" (writes runs/mf_n25)
    t.join(timeout=15)
    assert not t.is_alive()
    assert (job.runs / "ens2_m0" / "summary.json").is_file()  # re-run to completion afterwards
