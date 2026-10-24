"""`aerosurr <group> <command>`: every capability of the package is a CLI command."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Annotated, cast

import typer

from aerosurr import __version__
from aerosurr.config import DataConfig, PathsConfig, load_data, load_paths
from aerosurr.ops import configure_runtime, setup_logging

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
env_app = typer.Typer(no_args_is_help=True, help="Environment and GPU checks.")
data_app = typer.Typer(no_args_is_help=True, help="DrivAerML download and inspection.")
config_app = typer.Typer(no_args_is_help=True, help="Show validated configuration.")
app.add_typer(env_app, name="env")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")

BEGIN, END = "<!-- BEGIN GENERATED", "<!-- END GENERATED -->"


class _State:
    config_dir: Path | None = None

    def paths(self) -> PathsConfig:
        return load_paths(self.config_dir)

    def data(self) -> DataConfig:
        return load_data(self.config_dir)


state = _State()


def _command_line() -> str:
    return "aerosurr " + shlex.join(sys.argv[1:])


def write_generated(path: Path, body: str, title: str) -> None:
    """Write `body` between the GENERATED markers of `path`, keeping hand-written text around it.

    A new file gets a title and the block only.
    """
    block = (
        f"{BEGIN} by `{_command_line()}` (aerosurr {__version__}); do not edit -->\n{body}\n{END}"
    )
    if path.is_file():
        text = path.read_text()
        i, j = text.find(BEGIN), text.find(END)
        if i >= 0 and j > i:
            path.write_text(text[:i] + block + text[j + len(END) :])
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{block}\n")


@app.callback()
def main(
    config_dir: Annotated[
        Path | None,
        typer.Option(
            "--config-dir", envvar="AEROSURR_CONFIG_DIR", help="Directory of yaml configs."
        ),
    ] = None,
) -> None:
    state.config_dir = config_dir


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


# --------------------------------------------------------------------------- config


@config_app.command("show")
def config_show() -> None:
    """Print paths.yaml and data.yaml after validation and env overrides, as JSON."""
    out = {
        "paths": state.paths().model_dump(mode="json"),
        "data": state.data().model_dump(mode="json"),
    }
    typer.echo(json.dumps(out, indent=2))


# --------------------------------------------------------------------------- env


@env_app.command("check")
def env_check(
    devices: Annotated[
        str | None, typer.Option(help="Comma-separated torch device indices.")
    ] = None,
    sizes: Annotated[
        str, typer.Option(help="GEMM sizes n (n x n x n), comma-separated.")
    ] = "8192,16384",
    iters: Annotated[int, typer.Option(help="Timed GEMM iterations per size.")] = 50,
    expect: Annotated[int, typer.Option(help="Expected CUDA device count.")] = 2,
    md: Annotated[Path | None, typer.Option(help="Write a markdown report here.")] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write a JSON report here.")
    ] = None,
) -> None:
    """Idle check, torch/CUDA build check, bf16 GEMM TFLOPS and torch.compile smoke per GPU."""
    configure_runtime(state.paths().workspace)
    from aerosurr.envcheck import render_markdown, run_check

    dev_list = [int(d) for d in devices.split(",")] if devices else None
    rep = run_check(dev_list, tuple(int(s) for s in sizes.split(",")), iters, expect)
    text = render_markdown(rep, _command_line())
    typer.echo(text)
    if md:
        write_generated(md, text, "GPU stack check")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rep.to_json(), indent=2, default=str))
    raise typer.Exit(0 if rep.passed else 1)


# --------------------------------------------------------------------------- data


@data_app.command("inspect")
def data_inspect(
    file: Annotated[
        Path, typer.Argument(help="A boundary_N.vtp file.", exists=True, dir_okay=False)
    ],
    areas: Annotated[
        Path | None,
        typer.Option(help="Cell-area .npy (default: boundary_cell_area_N.npy beside it)."),
    ] = None,
    no_areas: Annotated[bool, typer.Option("--no-areas", help="Skip the cell-area check.")] = False,
    sha256: Annotated[bool, typer.Option(help="Hash the file.")] = True,
    out: Annotated[
        Path | None, typer.Option(help="Write the report into this markdown file.")
    ] = None,
) -> None:
    """Print every array of a VTP with dtype, shape and stats; check n_cells == len(areas)."""
    from aerosurr.data.inspect import default_areas_path, inspect_vtp, render_markdown

    area_path = None if no_areas else (areas or default_areas_path(file))
    if area_path is not None and not area_path.is_file():
        typer.echo(f"area file not found: {area_path}", err=True)
        raise typer.Exit(2)
    fit: tuple[str, str] | None = None
    try:
        f = state.data().fields
        fit = (f.pressure, f.pressure_coefficient)
    except FileNotFoundError:
        pass
    rep = inspect_vtp(file, area_path, pressure_fit=fit, compute_sha256=sha256)
    text = render_markdown(rep)
    typer.echo(text)
    if out:
        write_generated(out, text, "DrivAerML data inspection")
    if area_path is not None and not rep.passed:
        typer.echo("GATE FAIL: len(areas) != n_cells", err=True)
        raise typer.Exit(1)


@data_app.command("download")
def data_download(
    tier: Annotated[
        list[str] | None,
        typer.Option("--tier", "-t", help="t0, t1, t2 or t3; repeatable, in order."),
    ] = None,
    runs: Annotated[
        str | None, typer.Option(help="Comma-separated run ids (t2 subset, t3).")
    ] = None,
    workers: Annotated[
        int | None, typer.Option(help="Parallel files (default: data.yaml).")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan and check disk only.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Ignore the min_free_gb guard.")] = False,
    min_free_gb: Annotated[
        float | None,
        typer.Option(min=0, help="Free-space floor in GB (default: data.yaml), e.g. on CI."),
    ] = None,
    rehash: Annotated[
        bool, typer.Option("--rehash", help="Re-verify files already in MANIFEST.")
    ] = False,
    log_session: Annotated[str, typer.Option(help="Log dir name under paths.logs.")] = "aero-dl",
) -> None:
    """Resumable, pinned, sha256-verified download of DrivAerML tiers (SPEC §4.3)."""
    import logging

    from aerosurr.data.download import TIERS, DownloadError, Tier, download
    from aerosurr.data.hub import Hub, make_client

    paths, cfg = state.paths(), state.data()
    if min_free_gb is not None:
        cfg = cfg.model_copy(
            update={"download": cfg.download.model_copy(update={"min_free_gb": min_free_gb})}
        )
    bad = [t for t in tier or [] if t not in TIERS]
    if bad:
        raise typer.BadParameter(f"unknown tier(s) {bad}; choose from {TIERS}")
    tiers = cast(list[Tier], tier or ["t0", "t2"])
    log_file = setup_logging(paths.logs / log_session, "download")
    log = logging.getLogger("aerosurr.download")
    iface = paths.net.bind_interface or "(default route)"
    log.info("aerosurr %s: %s", __version__, _command_line())
    log.info(
        "repo %s @ %s -> %s; interface %s; log %s",
        cfg.source.repo_id,
        cfg.source.revision,
        paths.data_raw,
        iface,
        log_file,
    )
    run_list = [r.strip() for r in runs.split(",")] if runs else None
    with make_client(
        paths.net.bind_interface,
        cfg.download.timeout_s,
        max_connections=(workers or cfg.download.workers) + 2,
    ) as client:
        try:
            failed = download(
                Hub(cfg.source, client),
                cfg,
                paths.data_raw,
                tiers,
                run_list,
                workers,
                force,
                rehash,
                dry_run,
            )
        except DownloadError as e:
            log.error("%s", e)
            raise typer.Exit(1) from e
    raise typer.Exit(1 if failed else 0)


@data_app.command("status")
def data_status() -> None:
    """Download progress per tier, from the saved plans and MANIFEST.jsonl (offline)."""
    from aerosurr.data.download import status

    rows = status(state.paths().data_raw)
    if not rows:
        typer.echo("no download plans yet; run `aerosurr data download` first")
        raise typer.Exit(1)
    for s in rows:
        pct = 100.0 * s.bytes_done / s.bytes_total if s.bytes_total else 100.0
        typer.echo(
            f"{s.tier}: {s.files_done}/{s.files_total} files, "
            f"{s.bytes_done / 1e9:.2f}/{s.bytes_total / 1e9:.2f} GB ({pct:.1f}%)"
            + (f", {s.bytes_partial / 1e9:.2f} GB in .part files" if s.bytes_partial else "")
        )


if __name__ == "__main__":
    app()
