"""Resumable, verified, pinned-revision DrivAerML downloader (SPEC §4.3).

- Every file is fetched from `resolve/<revision>/...`, streamed into `<file>.part` and
  resumed with an HTTP Range request after any interruption.
- Every file is verified before it is renamed into place: LFS files against the LFS oid
  (sha256 of the content), plain git files against the git blob sha1. Size is always checked.
- Completed files are appended to `MANIFEST.jsonl` (path, size, sha256, git oid, how verified),
  and `SHA256SUMS` is rewritten at the end, so `sha256sum -c SHA256SUMS` re-checks everything.
- Files start in queue order, so any prefix of the T2 queue is a usable dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from aerosurr.config import DataConfig
from aerosurr.data.hub import Hub, RemoteFile
from aerosurr.data.splits import MANIFEST_PATH, all_runs, load_manifest, priority_runs, run_number

log = logging.getLogger("aerosurr.download")

Tier = Literal["t0", "t1", "t2", "t3"]
TIERS: tuple[Tier, ...] = ("t0", "t1", "t2", "t3")

MANIFEST_NAME = "MANIFEST.jsonl"
SUMS_NAME = "SHA256SUMS"
STATE_DIR = ".aerosurr"
PART_SUFFIX = ".part"

ROOT_FILES = (
    "README.md",
    "LICENSE.txt",
    "force_mom_all.csv",
    "force_mom_constref_all.csv",
    "geo_parameters_all.csv",
)
TREE_PREFIXES = ("splits", "surface_cell_areas")
RUN_CSVS = (
    "force_mom_{n}.csv",
    "force_mom_constref_{n}.csv",
    "geo_ref_{n}.csv",
    "geo_parameters_{n}.csv",
)
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class DownloadError(RuntimeError):
    pass


class VerifyError(DownloadError):
    pass


class _Retryable(Exception):
    def __init__(self, msg: str, retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class _Stopped(Exception):
    pass


# --------------------------------------------------------------------------- planning


def run_file(run: str, kind: Literal["boundary", "areas", "stl"]) -> str:
    n = run_number(run)
    return {
        "boundary": f"{run}/boundary_{n}.vtp",
        "areas": f"{run}/boundary_cell_area_{n}.npy",
        "stl": f"{run}/drivaer_{n}.stl",
    }[kind]


def tier_paths(
    tier: Tier,
    manifest: dict[str, list[str]],
    priority: Sequence[str],
    runs: Sequence[str] | None = None,
) -> list[str]:
    """Repo paths of a tier, in download order (T0 excludes the tree-listed directories).

    `runs` restricts the per-run files of T0 and T2 (e.g. CI fetches only run_1).
    """
    if tier == "t0":
        csv_runs = list(runs) if runs else all_runs(manifest)
        csvs = [f"{r}/{t.format(n=run_number(r))}" for r in csv_runs for t in RUN_CSVS]
        return [*ROOT_FILES, *csvs]
    if tier == "t1":
        return [run_file("run_1", "boundary"), run_file("run_1", "areas"), run_file("run_1", "stl")]
    if tier == "t2":
        order = list(runs) if runs else priority_runs(manifest, priority)
        return [run_file(r, k) for r in order for k in ("boundary", "areas")]
    if tier == "t3":
        if not runs:
            raise ValueError("tier t3 needs an explicit run list (the RANS subset, SPEC §5.3)")
        return [run_file(r, "stl") for r in runs]
    raise ValueError(f"unknown tier {tier!r}")


def plan_tier(
    hub: Hub,
    tier: Tier,
    manifest: dict[str, list[str]],
    priority: Sequence[str],
    runs: Sequence[str] | None = None,
    batch: int = 200,
) -> list[RemoteFile]:
    paths = tier_paths(tier, manifest, priority, runs)
    info = hub.paths_info(paths, batch=batch)
    missing = [p for p in paths if p not in info]
    if missing:
        raise DownloadError(f"{len(missing)} paths not at this revision, e.g. {missing[:5]}")
    files = [info[p] for p in paths]
    if tier == "t0":
        tree = [f for prefix in TREE_PREFIXES for f in hub.list_tree(prefix)]
        files = files[: len(ROOT_FILES)] + tree + files[len(ROOT_FILES) :]
    return files


# --------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class ManifestRecord:
    path: str
    size: int
    sha256: str
    git_oid: str
    verified: Literal["lfs-sha256", "git-sha1"]
    revision: str
    completed_utc: str

    def matches(self, rf: RemoteFile) -> bool:
        return (
            self.size == rf.size
            and self.git_oid == rf.git_oid
            and (rf.sha256 is None or self.sha256 == rf.sha256)
        )


class Manifest:
    """Append-only JSONL of verified files; the last record for a path wins."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.records: dict[str, ManifestRecord] = {}
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = ManifestRecord(**json.loads(line))
                    self.records[rec.path] = rec

    def get(self, path: str) -> ManifestRecord | None:
        return self.records.get(path)

    def append(self, rec: ManifestRecord) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.records[rec.path] = rec

    def write_sha256sums(self, path: Path) -> None:
        lines = [
            f"{r.sha256}  {r.path}\n" for r in sorted(self.records.values(), key=lambda r: r.path)
        ]
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(lines))
        tmp.replace(path)


# --------------------------------------------------------------------------- hashing


class _Digests:
    def __init__(self, expected_size: int) -> None:
        self.sha256 = hashlib.sha256()
        self.git = hashlib.sha1(f"blob {expected_size}\0".encode(), usedforsecurity=False)
        self.n = 0

    def update(self, chunk: bytes) -> None:
        self.sha256.update(chunk)
        self.git.update(chunk)
        self.n += len(chunk)

    def check(self, rf: RemoteFile) -> Literal["lfs-sha256", "git-sha1"]:
        if self.n != rf.size:
            raise VerifyError(f"{rf.path}: size {self.n} != {rf.size}")
        if rf.sha256 is not None:
            if self.sha256.hexdigest() != rf.sha256:
                raise VerifyError(
                    f"{rf.path}: sha256 {self.sha256.hexdigest()} != LFS oid {rf.sha256}"
                )
            return "lfs-sha256"
        if self.git.hexdigest() != rf.git_oid:
            raise VerifyError(f"{rf.path}: git sha1 {self.git.hexdigest()} != oid {rf.git_oid}")
        return "git-sha1"


def _hash_prefix(path: Path, n: int, expected_size: int, chunk: int = 16 << 20) -> _Digests:
    d = _Digests(expected_size)
    with path.open("rb") as f:
        while d.n < n:
            buf = f.read(min(chunk, n - d.n))
            if not buf:
                break
            d.update(buf)
    return d


# --------------------------------------------------------------------------- progress


class Progress:
    def __init__(self, total_files: int, total_bytes: int) -> None:
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.done_files = 0
        self.done_bytes = 0  # sizes of completed (or already present) files
        self.received = 0  # bytes off the network this session
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._last = (self.t0, 0)

    def add_received(self, n: int) -> None:
        with self._lock:
            self.received += n

    def file_done(self, size: int) -> None:
        with self._lock:
            self.done_files += 1
            self.done_bytes += size

    def line(self) -> str:
        with self._lock:
            now = time.monotonic()
            t_last, r_last = self._last
            rate = (self.received - r_last) / max(now - t_last, 1e-9)
            self._last = (now, self.received)
            avg = self.received / max(now - self.t0, 1e-9)
            left = self.total_bytes - self.done_bytes
            eta = left / avg if avg > 0 else float("inf")
            pct = 100.0 * self.done_bytes / self.total_bytes if self.total_bytes else 100.0
            eta_s = time.strftime("%H:%M:%S", time.gmtime(eta)) if eta != float("inf") else "--"
            return (
                f"progress {self.done_files}/{self.total_files} files, "
                f"{self.done_bytes / 1e9:.2f}/{self.total_bytes / 1e9:.2f} GB ({pct:.1f}%), "
                f"{rate / 1e6:.1f} MB/s now, {avg / 1e6:.1f} MB/s avg, ETA {eta_s}"
            )


# --------------------------------------------------------------------------- downloader


@dataclass(frozen=True)
class FetchResult:
    path: str
    status: Literal["cached", "adopted", "downloaded"]
    size: int
    seconds: float


class Downloader:
    def __init__(
        self,
        hub: Hub,
        root: Path,
        chunk_bytes: int = 8 << 20,
        max_retries: int = 8,
        rehash: bool = False,
        sleep: Callable[[float], object] | None = None,
    ) -> None:
        self.hub = hub
        self.root = root
        self.chunk_bytes = chunk_bytes
        self.max_retries = max_retries
        self.rehash = rehash
        self.manifest = Manifest(root / MANIFEST_NAME)
        self.stop = threading.Event()
        self.sleep = sleep or self.stop.wait  # back-off that Ctrl-C can cut short
        self.progress: Progress | None = None

    # ---- one file

    def _record(self, rf: RemoteFile, d: _Digests) -> None:
        how = d.check(rf)
        self.manifest.append(
            ManifestRecord(
                path=rf.path,
                size=rf.size,
                sha256=d.sha256.hexdigest(),
                git_oid=rf.git_oid,
                verified=how,
                revision=self.hub.source.revision,
                completed_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    def fetch(self, rf: RemoteFile) -> FetchResult:
        t_start = time.monotonic()
        final = self.root / rf.path
        part = final.with_name(final.name + PART_SUFFIX)
        final.parent.mkdir(parents=True, exist_ok=True)

        if final.is_file():
            rec = self.manifest.get(rf.path)
            if final.stat().st_size == rf.size and rec and rec.matches(rf) and not self.rehash:
                return FetchResult(rf.path, "cached", rf.size, 0.0)
            if final.stat().st_size == rf.size:
                d = _hash_prefix(final, rf.size, rf.size)
                try:
                    self._record(rf, d)
                    return FetchResult(rf.path, "adopted", rf.size, time.monotonic() - t_start)
                except VerifyError as e:
                    log.warning("existing file fails verification, re-downloading: %s", e)
                    final.replace(part)
                    part.write_bytes(b"")
            else:
                log.warning(
                    "existing %s has size %d != %d; resuming it as a partial",
                    rf.path,
                    final.stat().st_size,
                    rf.size,
                )
                final.replace(part)

        attempt = 0
        while True:
            if self.stop.is_set():
                raise _Stopped
            offset = part.stat().st_size if part.exists() else 0
            if offset > rf.size:
                part.write_bytes(b"")
                offset = 0
            d = _hash_prefix(part, offset, rf.size) if offset else _Digests(rf.size)
            if offset == rf.size:
                break
            try:
                d = self._stream(rf, part, offset, d)
                if d.n == rf.size:
                    break
                raise _Retryable(f"short read: {d.n}/{rf.size} bytes")
            except (httpx.TransportError, _Retryable) as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise DownloadError(
                        f"{rf.path}: giving up after {attempt - 1} retries: {e}"
                    ) from e
                wait = getattr(e, "retry_after", None) or min(60.0, 2.0**attempt)
                log.warning(
                    "%s: %s; retry %d/%d in %.0f s (resume at %d bytes)",
                    rf.path,
                    e,
                    attempt,
                    self.max_retries,
                    wait,
                    part.stat().st_size if part.exists() else 0,
                )
                self.sleep(wait)

        try:
            self._record(rf, d)
        except VerifyError:
            part.unlink(missing_ok=True)  # corrupt bytes must not be resumed from
            raise
        part.replace(final)
        return FetchResult(rf.path, "downloaded", rf.size, time.monotonic() - t_start)

    def _stream(self, rf: RemoteFile, part: Path, offset: int, d: _Digests) -> _Digests:
        """Append the bytes from `offset` to `part`; return the digests of the whole file."""
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with self.hub.client.stream("GET", self.hub.resolve_url(rf.path), headers=headers) as r:
            if r.status_code in RETRY_STATUS:
                ra = r.headers.get("retry-after")
                raise _Retryable(
                    f"HTTP {r.status_code}", float(ra) if ra and ra.isdigit() else None
                )
            r.raise_for_status()
            if offset and r.status_code == 200:  # server ignored Range: start over
                log.warning("%s: server ignored Range, restarting from 0", rf.path)
                offset = 0
                d = _Digests(rf.size)
            elif offset:
                cr = r.headers.get("content-range", "")
                if not cr.startswith(f"bytes {offset}-"):
                    raise _Retryable(f"unexpected Content-Range {cr!r} for offset {offset}")
            mode = "r+b" if part.exists() else "wb"
            with part.open(mode) as f:
                f.seek(offset)
                f.truncate()
                for chunk in r.iter_bytes(self.chunk_bytes):
                    if self.stop.is_set():
                        f.flush()
                        raise _Stopped
                    f.write(chunk)
                    d.update(chunk)
                    if self.progress:
                        self.progress.add_received(len(chunk))
                f.flush()
                os.fsync(f.fileno())
        return d

    # ---- a queue of files

    def run(
        self, files: Sequence[RemoteFile], workers: int = 4, report_s: float = 30.0
    ) -> list[str]:
        """Fetch `files` with `workers` threads, in queue order. Returns the failed paths."""
        self.progress = prog = Progress(len(files), sum(f.size for f in files))
        failed: list[str] = []
        done_evt = threading.Event()

        def reporter() -> None:
            while not done_evt.wait(report_s):
                log.info(prog.line())

        def one(i: int, rf: RemoteFile) -> None:
            try:
                res = self.fetch(rf)
            except _Stopped:
                return
            except Exception as e:  # report and keep going with the rest of the queue
                log.error("[%d/%d] FAIL %s: %s", i, len(files), rf.path, e)
                failed.append(rf.path)
                if isinstance(e, OSError) and e.errno == 28:  # ENOSPC
                    self.stop.set()
                return
            prog.file_done(res.size)
            if res.status != "cached":
                rate = res.size / res.seconds / 1e6 if res.seconds > 0 else float("nan")
                log.info(
                    "[%d/%d] %s %s %.1f MB in %.1f s (%.1f MB/s)",
                    i,
                    len(files),
                    res.status.upper(),
                    rf.path,
                    res.size / 1e6,
                    res.seconds,
                    rate,
                )

        rep = threading.Thread(target=reporter, daemon=True)
        rep.start()
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl")
        futures: list[Future[None]] = []
        try:
            futures = [pool.submit(one, i, rf) for i, rf in enumerate(files, 1)]
            for fut in futures:
                fut.result()
        except KeyboardInterrupt:
            log.warning("interrupted: finishing writes; .part files are kept for resume")
            self.stop.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            done_evt.set()
            self.manifest.write_sha256sums(self.root / SUMS_NAME)
            log.info(prog.line())
        return failed


# --------------------------------------------------------------------------- orchestration


def check_revision(root: Path, revision: str) -> None:
    """Refuse to mix revisions in one data directory."""
    rev_file = root / "REVISION"
    if rev_file.is_file():
        have = rev_file.read_text().strip()
        if have != revision:
            raise DownloadError(f"{root} holds revision {have}, config pins {revision}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        rev_file.write_text(revision + "\n")


def remaining_bytes(root: Path, files: Sequence[RemoteFile], manifest: Manifest) -> int:
    left = 0
    for rf in files:
        final = root / rf.path
        if final.is_file() and final.stat().st_size == rf.size:
            continue  # verified already, or will be hashed and adopted
        part = final.with_name(final.name + PART_SUFFIX)
        left += rf.size - (part.stat().st_size if part.is_file() else 0)
    return max(left, 0)


def save_plan(root: Path, tier: Tier, revision: str, files: Sequence[RemoteFile]) -> Path:
    path = root / STATE_DIR / f"plan-{tier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "tier": tier,
        "revision": revision,
        "files": [f.to_json() for f in files],
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_plan(root: Path, tier: Tier) -> list[RemoteFile] | None:
    path = root / STATE_DIR / f"plan-{tier}.json"
    if not path.is_file():
        return None
    return [RemoteFile.from_json(d) for d in json.loads(path.read_text())["files"]]


def download(
    hub: Hub,
    cfg: DataConfig,
    root: Path,
    tiers: Sequence[Tier],
    runs: Sequence[str] | None = None,
    workers: int | None = None,
    force: bool = False,
    rehash: bool = False,
    dry_run: bool = False,
    report_s: float = 30.0,
) -> list[str]:
    """Plan, check disk, and fetch the given tiers in order. Returns failed paths."""
    dl = cfg.download
    check_revision(root, cfg.source.revision)
    downloader = Downloader(hub, root, dl.chunk_bytes, dl.max_retries, rehash=rehash)

    # The split manifest drives every per-run plan, so it is fetched (and verified) first.
    split_info = hub.paths_info([MANIFEST_PATH])
    if MANIFEST_PATH not in split_info:
        raise DownloadError(f"{MANIFEST_PATH} not found at {cfg.source.revision}")
    downloader.fetch(split_info[MANIFEST_PATH])
    manifest = load_manifest(root / MANIFEST_PATH)

    queue: list[RemoteFile] = []
    seen: set[str] = set()
    for tier in tiers:
        files = plan_tier(hub, tier, manifest, dl.priority, runs, dl.paths_info_batch)
        save_plan(root, tier, cfg.source.revision, files)
        log.info("plan %s: %d files, %.2f GB", tier, len(files), sum(f.size for f in files) / 1e9)
        for f in files:
            if f.path not in seen:
                seen.add(f.path)
                queue.append(f)

    left = remaining_bytes(root, queue, downloader.manifest)
    free = shutil.disk_usage(root).free
    log.info(
        "queue: %d files, %.2f GB total, %.2f GB still to fetch; %.1f GB free on disk",
        len(queue),
        sum(f.size for f in queue) / 1e9,
        left / 1e9,
        free / 1e9,
    )
    if free - left < dl.min_free_gb * 1e9 and not force:
        raise DownloadError(
            f"would leave {(free - left) / 1e9:.1f} GB free (< min_free_gb={dl.min_free_gb}); "
            "use --force to override"
        )
    if dry_run:
        return []
    failed = downloader.run(queue, workers=workers or dl.workers, report_s=report_s)
    if failed:
        log.error("%d files failed: %s", len(failed), failed[:10])
    else:
        log.info("all %d files present and verified", len(queue))
    return failed


@dataclass(frozen=True)
class TierStatus:
    tier: Tier
    files_total: int
    files_done: int
    bytes_total: int
    bytes_done: int
    bytes_partial: int


def status(root: Path) -> list[TierStatus]:
    """Progress per tier from the saved plans and the manifest (no network)."""
    man = Manifest(root / MANIFEST_NAME)
    out: list[TierStatus] = []
    for tier in TIERS:
        files = load_plan(root, tier)
        if files is None:
            continue
        done = [f for f in files if (r := man.get(f.path)) is not None and r.matches(f)]
        partial = 0
        for f in files:
            part = (root / f.path).with_name(Path(f.path).name + PART_SUFFIX)
            if part.is_file():
                partial += part.stat().st_size
        out.append(
            TierStatus(
                tier,
                len(files),
                len(done),
                sum(f.size for f in files),
                sum(f.size for f in done),
                partial,
            )
        )
    return out
