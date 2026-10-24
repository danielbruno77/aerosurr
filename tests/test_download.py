"""Downloader logic against an in-memory fake of the Hugging Face Hub (httpx.MockTransport)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from aerosurr.config import DataConfig, load_data
from aerosurr.data import download as dl
from aerosurr.data.hub import Hub, RemoteFile, make_client

REV = "5d448b209bf654503c64ce7261c34fa125f46392"
CDN = "https://cdn.example"


def git_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FailingStream(httpx.SyncByteStream):
    def __init__(self, data: bytes, fail_after: int) -> None:
        self.data, self.fail_after = data, fail_after

    def __iter__(self) -> Iterator[bytes]:
        yield self.data[: self.fail_after]
        raise httpx.ReadError("connection reset (simulated)")


class FakeHub:
    """Serves paths-info, paginated tree listings, resolve (+302 for LFS) and Range."""

    def __init__(self, files: dict[str, bytes], lfs: set[str]) -> None:
        self.files, self.lfs = files, lfs
        self.requests: list[httpx.Request] = []
        self.fail_after: dict[str, int] = {}  # path -> break the first stream after n bytes
        self.status_once: dict[str, int] = {}  # path -> return this status once
        self.ignore_range: set[str] = set()
        self.corrupt: set[str] = set()

    def info(self, path: str) -> dict[str, Any]:
        data = self.files[path]
        d: dict[str, Any] = {"type": "file", "path": path, "size": len(data), "oid": git_oid(data)}
        if path in self.lfs:
            d["oid"] = "0" * 40  # pointer-file oid; not the content hash
            d["lfs"] = {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)}
        return d

    def serve(self, path: str, request: httpx.Request) -> httpx.Response:
        if code := self.status_once.pop(path, None):
            return httpx.Response(code, headers={"retry-after": "1"})
        data = self.files[path]
        if path in self.corrupt:
            data = b"X" + data[1:]
        start = 0
        rng = request.headers.get("range")
        if rng and path not in self.ignore_range:
            start = int(re.fullmatch(r"bytes=(\d+)-", rng).group(1))  # type: ignore[union-attr]
            headers = {"content-range": f"bytes {start}-{len(data) - 1}/{len(data)}"}
            status = 206
        else:
            headers, status = {}, 200
        body = data[start:]
        if (n := self.fail_after.pop(path, None)) is not None:
            return httpx.Response(status, headers=headers, stream=FailingStream(body, n))
        return httpx.Response(status, headers=headers, content=body)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        api = "https://huggingface.co/api/datasets/o/r"
        if url == f"{api}/paths-info/{REV}":
            paths = json.loads(request.content)["paths"]
            return httpx.Response(200, json=[self.info(p) for p in paths if p in self.files])
        if url.startswith(f"{api}/tree/{REV}/"):
            prefix = request.url.path.split(f"/tree/{REV}/", 1)[1]
            names = sorted(p for p in self.files if p.startswith(prefix + "/"))
            page = int(request.url.params.get("cursor", "0"))
            headers = {}
            if page + 2 < len(names):
                headers["link"] = f'<{api}/tree/{REV}/{prefix}?cursor={page + 2}>; rel="next"'
            return httpx.Response(
                200, json=[self.info(p) for p in names[page : page + 2]], headers=headers
            )
        m = re.fullmatch(rf"https://huggingface.co/datasets/o/r/resolve/{REV}/(.+)", url)
        if m:
            path = m.group(1)
            if path in self.lfs:
                sha = hashlib.sha256(self.files[path]).hexdigest()
                return httpx.Response(302, headers={"location": f"{CDN}/{sha}?sig=1"})
            return self.serve(path, request)
        if url.startswith(CDN):
            sha = request.url.path.lstrip("/")
            path = next(p for p in self.lfs if hashlib.sha256(self.files[p]).hexdigest() == sha)
            return self.serve(path, request)
        return httpx.Response(404)


MANIFEST = {
    "full_train": ["run_1", "run_2", "run_3"],
    "full_val": ["run_4"],
    "full_test": ["run_5"],
    "super_scarce_train": ["run_3"],
    "scarce_train": ["run_3", "run_1"],
    "medium_train": ["run_3", "run_1", "run_2"],
}


def make_files() -> tuple[dict[str, bytes], set[str]]:
    files: dict[str, bytes] = {
        "splits/manifest.json": json.dumps(MANIFEST).encode(),
        "splits/README.md": b"# splits\n",
        "splits/fig.png": b"\x89PNG" + bytes(range(256)) * 4,
        "surface_cell_areas/README.md": b"areas\n",
        "surface_cell_areas/cases/run_1.json": b"{}",
    }
    lfs = {"splits/fig.png"}
    for name in dl.ROOT_FILES:
        files[name] = f"root {name}\n".encode()
    for i in range(1, 6):
        for t in dl.RUN_CSVS:
            files[f"run_{i}/{t.format(n=i)}"] = f"Cd\n0.{i}\n".encode()
        vtp, npy = f"run_{i}/boundary_{i}.vtp", f"run_{i}/boundary_cell_area_{i}.npy"
        files[vtp] = bytes([i]) * (50_000 + i)
        files[npy] = bytes([i + 100]) * 4_000
        lfs |= {vtp, npy}
    return files, lfs


@pytest.fixture
def cfg(config_dir: Path) -> DataConfig:
    base = load_data(config_dir).model_dump()
    base["source"]["repo_id"] = "o/r"
    base["download"]["min_free_gb"] = 0
    return DataConfig.model_validate(base)


@pytest.fixture
def fake() -> FakeHub:
    return FakeHub(*make_files())


@pytest.fixture
def hub(fake: FakeHub, cfg: DataConfig) -> Iterator[Hub]:
    with make_client(transport=httpx.MockTransport(fake)) as client:
        yield Hub(cfg.source, client)


def downloader(hub: Hub, root: Path, **kw: Any) -> dl.Downloader:
    return dl.Downloader(hub, root, chunk_bytes=4096, max_retries=3, sleep=lambda s: None, **kw)


def remote(hub: Hub, path: str) -> RemoteFile:
    return hub.paths_info([path])[path]


# --------------------------------------------------------------------------- planning


def test_tier_paths_priority_order() -> None:
    prio = [
        "full_test",
        "full_val",
        "super_scarce_train",
        "scarce_train",
        "medium_train",
        "full_train",
    ]
    t2 = dl.tier_paths("t2", MANIFEST, prio)
    runs = [p.split("/")[0] for p in t2[::2]]
    assert runs == ["run_5", "run_4", "run_3", "run_1", "run_2"]  # each run once, first wins
    assert t2[:2] == ["run_5/boundary_5.vtp", "run_5/boundary_cell_area_5.npy"]
    t0 = dl.tier_paths("t0", MANIFEST, prio)
    assert t0[: len(dl.ROOT_FILES)] == list(dl.ROOT_FILES)
    assert len(t0) == len(dl.ROOT_FILES) + 5 * len(dl.RUN_CSVS)
    assert dl.tier_paths("t0", MANIFEST, prio, runs=["run_1"])[-1] == "run_1/geo_parameters_1.csv"
    with pytest.raises(ValueError, match="explicit run list"):
        dl.tier_paths("t3", MANIFEST, prio)
    with pytest.raises(KeyError):
        dl.tier_paths("t2", MANIFEST, ["nope"])


def test_plan_t0_includes_paginated_trees(hub: Hub, fake: FakeHub) -> None:
    files = dl.plan_tier(hub, "t0", MANIFEST, ["full_test"])
    paths = [f.path for f in files]
    assert "splits/fig.png" in paths and "surface_cell_areas/cases/run_1.json" in paths
    assert paths.index("splits/README.md") < paths.index("run_1/force_mom_1.csv")
    assert sum("cursor=" in str(r.url) for r in fake.requests) >= 1  # followed rel="next"
    fig = next(f for f in files if f.path == "splits/fig.png")
    assert fig.is_lfs and fig.sha256 == hashlib.sha256(fake.files["splits/fig.png"]).hexdigest()


def test_plan_missing_path_raises(hub: Hub, fake: FakeHub) -> None:
    del fake.files["run_4/boundary_4.vtp"]
    with pytest.raises(dl.DownloadError, match="not at this revision"):
        dl.plan_tier(hub, "t2", MANIFEST, ["full_val"])


# --------------------------------------------------------------------------- one file


def test_fetch_lfs_via_redirect(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_2/boundary_2.vtp")
    res = downloader(hub, tmp_path).fetch(rf)
    assert res.status == "downloaded"
    assert (tmp_path / rf.path).read_bytes() == fake.files[rf.path]
    assert not (tmp_path / (rf.path + ".part")).exists()
    rec = json.loads((tmp_path / dl.MANIFEST_NAME).read_text().splitlines()[-1])
    assert rec["verified"] == "lfs-sha256" and rec["sha256"] == rf.sha256 and rec["revision"] == REV
    assert any(str(r.url).startswith(CDN) for r in fake.requests)


def test_fetch_git_file_verified_by_blob_sha1(hub: Hub, tmp_path: Path) -> None:
    rf = remote(hub, "run_1/force_mom_1.csv")
    assert not rf.is_lfs
    downloader(hub, tmp_path).fetch(rf)
    rec = dl.Manifest(tmp_path / dl.MANIFEST_NAME).get(rf.path)
    assert rec is not None and rec.verified == "git-sha1"


def test_resume_partial_with_range(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_3/boundary_3.vtp")
    part = tmp_path / (rf.path + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(fake.files[rf.path][:12_345])
    downloader(hub, tmp_path).fetch(rf)
    assert (tmp_path / rf.path).read_bytes() == fake.files[rf.path]
    assert [r.headers.get("range") for r in fake.requests if str(r.url).startswith(CDN)] == [
        "bytes=12345-"
    ]


def test_resume_after_connection_drop(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_3/boundary_3.vtp")
    fake.fail_after[rf.path] = 20_000
    downloader(hub, tmp_path).fetch(rf)
    assert (tmp_path / rf.path).read_bytes() == fake.files[rf.path]
    ranges = [r.headers.get("range") for r in fake.requests if str(r.url).startswith(CDN)]
    # Only whole chunks reach the .part file before the drop; the resume starts right there.
    assert ranges[0] is None and len(ranges) == 2
    resumed_at = int(ranges[1].removeprefix("bytes=").rstrip("-"))
    assert 0 < resumed_at <= 20_000 and resumed_at % 4096 == 0


def test_server_ignoring_range_restarts(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_3/boundary_3.vtp")
    fake.ignore_range.add(rf.path)
    part = tmp_path / (rf.path + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"garbage" * 100)
    downloader(hub, tmp_path).fetch(rf)
    assert (tmp_path / rf.path).read_bytes() == fake.files[rf.path]


def test_retry_on_429(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_1/boundary_1.vtp")
    fake.status_once[rf.path] = 429
    waits: list[float] = []
    d = dl.Downloader(hub, tmp_path, chunk_bytes=4096, max_retries=3, sleep=waits.append)
    assert d.fetch(rf).status == "downloaded"
    assert waits == [1.0]  # honoured Retry-After


def test_gives_up_after_max_retries(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_1/boundary_1.vtp")
    fake.files["run_1/boundary_1.vtp"] = fake.files[rf.path]  # unchanged content
    fake.status_once[rf.path] = 503
    d = dl.Downloader(hub, tmp_path, chunk_bytes=4096, max_retries=0, sleep=lambda s: None)
    with pytest.raises(dl.DownloadError, match="giving up"):
        d.fetch(rf)


def test_corrupt_content_rejected(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_2/boundary_2.vtp")
    fake.corrupt.add(rf.path)
    with pytest.raises(dl.VerifyError, match="sha256"):
        downloader(hub, tmp_path).fetch(rf)
    assert not (tmp_path / rf.path).exists()
    assert not (tmp_path / (rf.path + ".part")).exists()  # never resume from bad bytes


def test_cached_and_adopted(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_4/boundary_4.vtp")
    final = tmp_path / rf.path
    final.parent.mkdir(parents=True)
    final.write_bytes(fake.files[rf.path])  # e.g. from the bootstrap script
    assert downloader(hub, tmp_path).fetch(rf).status == "adopted"
    n = len(fake.requests)
    assert downloader(hub, tmp_path).fetch(rf).status == "cached"
    assert len(fake.requests) == n  # no network for verified files


def test_existing_bad_file_is_replaced(hub: Hub, fake: FakeHub, tmp_path: Path) -> None:
    rf = remote(hub, "run_4/boundary_4.vtp")
    final = tmp_path / rf.path
    final.parent.mkdir(parents=True)
    final.write_bytes(b"\0" * rf.size)  # right size, wrong bytes
    assert downloader(hub, tmp_path).fetch(rf).status == "downloaded"
    assert final.read_bytes() == fake.files[rf.path]


# --------------------------------------------------------------------------- whole runs


def test_download_t0_t2_end_to_end(hub: Hub, cfg: DataConfig, tmp_path: Path) -> None:
    failed = dl.download(hub, cfg, tmp_path, ["t0", "t2"], workers=3, report_s=3600)
    assert failed == []
    sums = (tmp_path / dl.SUMS_NAME).read_text().splitlines()
    assert any(line.endswith("  run_5/boundary_5.vtp") for line in sums)
    st = {s.tier: s for s in dl.status(tmp_path)}
    assert st["t2"].files_done == st["t2"].files_total == 10
    assert st["t0"].files_done == st["t0"].files_total
    assert (tmp_path / "REVISION").read_text().strip() == REV


def test_download_runs_subset(hub: Hub, cfg: DataConfig, tmp_path: Path) -> None:
    assert dl.download(hub, cfg, tmp_path, ["t2"], runs=["run_1"], report_s=3600) == []
    assert (tmp_path / "run_1/boundary_1.vtp").is_file()
    assert not (tmp_path / "run_2").exists()


def test_failed_file_does_not_stop_queue(
    hub: Hub, fake: FakeHub, cfg: DataConfig, tmp_path: Path
) -> None:
    fake.corrupt.add("run_5/boundary_5.vtp")
    failed = dl.download(hub, cfg, tmp_path, ["t2"], workers=2, report_s=3600)
    assert failed == ["run_5/boundary_5.vtp"]
    assert (tmp_path / "run_2/boundary_2.vtp").is_file()


def test_revision_mismatch(hub: Hub, cfg: DataConfig, tmp_path: Path) -> None:
    (tmp_path / "REVISION").write_text("f" * 40 + "\n")
    with pytest.raises(dl.DownloadError, match="holds revision"):
        dl.download(hub, cfg, tmp_path, ["t2"])


def test_disk_guard(
    hub: Hub, cfg: DataConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strict = DataConfig.model_validate(
        {**cfg.model_dump(), "download": {**cfg.download.model_dump(), "min_free_gb": 1.0}}
    )
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(dl.shutil, "disk_usage", lambda p: usage._replace(free=10**8))
    with pytest.raises(dl.DownloadError, match="min_free_gb"):
        dl.download(hub, strict, tmp_path, ["t2"])
    assert dl.download(hub, strict, tmp_path, ["t2"], force=True, report_s=3600) == []


def test_dry_run_fetches_only_split_manifest(hub: Hub, cfg: DataConfig, tmp_path: Path) -> None:
    assert dl.download(hub, cfg, tmp_path, ["t0", "t2"], dry_run=True) == []
    assert (tmp_path / "splits/manifest.json").is_file()
    assert not (tmp_path / "run_5").exists()
    assert {s.tier for s in dl.status(tmp_path)} == {"t0", "t2"}


# --------------------------------------------------------------------------- socket binding


class _Hello(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"hi")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def local_server() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Hello)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def _can_bind_device() -> bool:
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, b"lo")
        return True
    except OSError:
        return False
    finally:
        s.close()


def _non_loopback_interface() -> str | None:
    names = [n for _, n in socket.if_nameindex() if n != "lo"]
    return names[0] if names else None


@pytest.mark.skipif(not _can_bind_device(), reason="unprivileged SO_BINDTODEVICE unavailable")
def test_bound_client_binds_before_connect(local_server: str) -> None:
    with make_client("lo", timeout_s=1) as c:
        assert c.get(local_server).text == "hi"
    # 127.0.0.1 is reachable only through `lo`. A socket bound to another device fails in
    # connect(); with httpx's own socket_options the option is applied only after the
    # connection is up, which is the bug BoundSyncBackend exists to avoid.
    other = _non_loopback_interface()
    if other is None:
        pytest.skip("no non-loopback interface")
    with (
        make_client(other, timeout_s=1) as c,
        pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)),
    ):
        c.get(local_server)
