"""Minimal Hugging Face Hub client: pinned-revision file metadata and download URLs."""

from __future__ import annotations

import socket
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

import httpcore
import httpx
from httpcore._backends.sync import SyncStream

from aerosurr import __version__
from aerosurr.config import DatasetSource


@dataclass(frozen=True)
class RemoteFile:
    """One file at the pinned revision, as reported by the Hub API."""

    path: str
    size: int
    git_oid: str  # git blob sha1 (for LFS files: of the pointer, not the content)
    sha256: str | None  # LFS oid = sha256 of the content; None for plain git files

    @property
    def is_lfs(self) -> bool:
        return self.sha256 is not None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> RemoteFile:
        return cls(
            path=str(d["path"]),
            size=int(d["size"]),
            git_oid=str(d["git_oid"]),
            sha256=None if d.get("sha256") is None else str(d["sha256"]),
        )

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> RemoteFile:
        lfs = d.get("lfs")
        return cls(
            path=str(d["path"]),
            size=int(d["size"]),
            git_oid=str(d["oid"]),
            sha256=str(lfs["oid"]) if lfs else None,
        )


class BoundSyncBackend(httpcore.SyncBackend):
    """httpcore backend that sets SO_BINDTODEVICE on each socket *before* `connect()`.

    httpx's `socket_options` are applied after `socket.create_connection()`, when the TCP
    handshake has already left through the default route (a VPN tunnel on workstation-III).
    Re-binding that socket strands the connection and TLS times out, so it can't be used.
    Unprivileged SO_BINDTODEVICE works on Linux >= 5.7 for a socket that is not yet bound.
    """

    def __init__(self, interface: str) -> None:
        self.interface = interface.encode()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.NetworkStream:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as e:
            raise httpcore.ConnectError(f"{host}: {e}") from e
        err: OSError | None = None
        for family, kind, proto, _, addr in infos:
            sock = socket.socket(family, kind, proto)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, self.interface)
                for opt in socket_options or ():
                    sock.setsockopt(*opt)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if local_address:
                    sock.bind((local_address, 0))
                sock.settimeout(timeout)
                sock.connect(addr)
                return SyncStream(sock)
            except OSError as e:
                sock.close()
                err = e
        if isinstance(err, TimeoutError):
            raise httpcore.ConnectTimeout(f"{host}:{port}: {err}") from err
        raise httpcore.ConnectError(f"{host}:{port} via {self.interface.decode()}: {err}") from err


def make_client(
    bind_interface: str = "",
    timeout_s: float = 60.0,
    max_connections: int = 8,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """An httpx client whose sockets are bound to `bind_interface` (SO_BINDTODEVICE).

    An empty interface uses the default route. `transport` overrides everything (tests).
    """
    if transport is None:
        keepalive: list[httpcore.SOCKET_OPTION] = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        limits = httpx.Limits(max_connections=max_connections)
        http = httpx.HTTPTransport(limits=limits, retries=2, socket_options=keepalive)
        if bind_interface:
            # HTTPTransport has no network-backend argument: swap in an equivalent pool.
            http._pool = httpcore.ConnectionPool(
                ssl_context=httpx.create_ssl_context(),
                max_connections=max_connections,
                retries=2,  # connection-level only; the downloader retries reads itself
                socket_options=keepalive,
                network_backend=BoundSyncBackend(bind_interface),
            )
        transport = http
    return httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout_s, connect=min(timeout_s, 20.0)),
        follow_redirects=True,
        headers={"user-agent": f"aerosurr/{__version__}"},
    )


class Hub:
    def __init__(self, source: DatasetSource, client: httpx.Client) -> None:
        self.source = source
        self.client = client

    @property
    def _api(self) -> str:
        return f"{self.source.endpoint}/api/{self.source.repo_type}s/{self.source.repo_id}"

    def resolve_url(self, path: str) -> str:
        return (
            f"{self.source.endpoint}/{self.source.repo_type}s/{self.source.repo_id}"
            f"/resolve/{self.source.revision}/{quote(path)}"
        )

    def paths_info(self, paths: Iterable[str], batch: int = 200) -> dict[str, RemoteFile]:
        """Metadata for `paths` at the pinned revision. Missing paths are absent from the result."""
        todo = list(paths)
        out: dict[str, RemoteFile] = {}
        for i in range(0, len(todo), batch):
            r = self.client.post(
                f"{self._api}/paths-info/{self.source.revision}",
                json={"paths": todo[i : i + batch]},
            )
            r.raise_for_status()
            for d in r.json():
                if d.get("type") == "file":
                    f = RemoteFile.from_api(d)
                    out[f.path] = f
        return out

    def list_tree(self, prefix: str, recursive: bool = True) -> list[RemoteFile]:
        """All files under `prefix` at the pinned revision (follows `Link: rel="next"`)."""
        return list(self._iter_tree(prefix, recursive))

    def _iter_tree(self, prefix: str, recursive: bool) -> Iterator[RemoteFile]:
        url: str | None = f"{self._api}/tree/{self.source.revision}/{quote(prefix)}"
        params: dict[str, str] | None = {"recursive": str(recursive).lower()}
        while url:
            r = self.client.get(url, params=params)
            r.raise_for_status()
            for d in r.json():
                if d.get("type") == "file":
                    yield RemoteFile.from_api(d)
            nxt = r.links.get("next")
            url, params = (nxt["url"], None) if nxt else (None, None)
