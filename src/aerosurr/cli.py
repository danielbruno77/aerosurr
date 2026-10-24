"""`aerosurr <group> <command>`: every capability of the package is a CLI command."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer

from aerosurr import __version__
from aerosurr.config import DataConfig, PathsConfig, load_data, load_paths
from aerosurr.ops import configure_runtime, setup_logging

if TYPE_CHECKING:  # torch is imported lazily: `aerosurr --help` stays fast
    import torch

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
env_app = typer.Typer(no_args_is_help=True, help="Environment and GPU checks.")
data_app = typer.Typer(no_args_is_help=True, help="DrivAerML download and inspection.")
config_app = typer.Typer(no_args_is_help=True, help="Show validated configuration.")
app.add_typer(env_app, name="env")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
rans_app = typer.Typer(no_args_is_help=True, help="Coarse OpenFOAM RANS cases (SPEC §5.3).")
app.add_typer(rans_app, name="rans")
train_app = typer.Typer(no_args_is_help=True, help="Train surface models (SPEC §5.2).")
app.add_typer(train_app, name="train")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate trained models (SPEC §5.5).")
app.add_typer(eval_app, name="eval")
viewer_app = typer.Typer(no_args_is_help=True, help="Interactive 3D viewer data (SPEC §6).")
app.add_typer(viewer_app, name="viewer")

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


@env_app.command("probe")
def env_probe(
    cpus: Annotated[str, typer.Option(help="CPU list to pin the probe to, e.g. 24-31,56-63.")],
    seconds: Annotated[float, typer.Option(help="How long to run.")] = 240.0,
    gpu: Annotated[int | None, typer.Option(help="CUDA device for the training load.")] = 0,
    workers: Annotated[int, typer.Option(help="VTP-reading worker processes.")] = 4,
    runs: Annotated[str, typer.Option(help="Runs whose VTPs the CPU load reads.")] = (
        "run_2,run_3,run_5,run_6,run_7,run_8"
    ),
    json_out: Annotated[Path | None, typer.Option("--json", help="Write results here.")] = None,
) -> None:
    """Measure the side load (VTP preprocessing + GPU training steps) on its own cores."""
    paths = state.paths()
    configure_runtime(paths.workspace)
    from aerosurr.data.splits import run_number
    from aerosurr.probe import run_probe
    from aerosurr.rans.run import parse_cpus

    vtps = [paths.data_raw / r / f"boundary_{run_number(r)}.vtp" for r in runs.split(",")]
    missing = [v for v in vtps if not v.is_file()]
    if missing:
        raise typer.BadParameter(f"missing VTPs: {missing}")
    res = run_probe(parse_cpus(cpus), vtps, seconds, workers, gpu)
    typer.echo(json.dumps(res.summary(), indent=2))
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(res.to_json(), indent=2))


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


def _validate_one(args: tuple[str, str, str, str]) -> dict[str, object]:
    """Worker for `data validate` (module-level so it pickles)."""
    from dataclasses import asdict

    from aerosurr.config import Freestream, SurfaceFields
    from aerosurr.data.validate import validate_run

    raw, run, fields_json, fs_json = args
    res = validate_run(Path(raw), run, SurfaceFields.model_validate_json(fields_json),
                       Freestream.model_validate_json(fs_json))  # fmt: skip
    return asdict(res)


@data_app.command("validate")
def data_validate(
    runs: Annotated[
        str | None, typer.Option(help="Comma-separated run ids (default: all).")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Parallel processes.")] = 8,
    cpus: Annotated[str | None, typer.Option(help="Pin the workers to these CPUs.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Write the report into this markdown file.")
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Per-run results.")] = None,
) -> None:
    """Reference-force gate: integrated coefficients vs force_mom*_i.csv (SPEC §5.1)."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    from aerosurr.data.splits import all_runs, load_manifest
    from aerosurr.data.validate import RunValidation, render_markdown
    from aerosurr.rans.run import parse_cpus

    paths, data = state.paths(), state.data()
    raw = paths.data_raw
    run_list = runs.split(",") if runs else all_runs(load_manifest(raw / "splits/manifest.json"))
    if cpus:
        os.sched_setaffinity(0, parse_cpus(cpus))
    args = [(str(raw), r, data.fields.model_dump_json(), data.freestream.model_dump_json())
            for r in run_list]  # fmt: skip
    results: list[RunValidation] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, d in enumerate(pool.map(_validate_one, args), 1):
            rv = RunValidation(**d)  # type: ignore[arg-type]
            results.append(rv)
            if i % 25 == 0 or i == len(args):
                typer.echo(f"[{i}/{len(args)}] validated; failures so far: "
                           f"{sum(not r.passed for r in results)}", err=True)  # fmt: skip
    text = render_markdown(results)
    typer.echo(text)
    if out:
        write_generated(out, text, "Reference-force validation")
    if json_out:
        from dataclasses import asdict

        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([asdict(r) for r in results], indent=1))
    raise typer.Exit(0 if all(r.passed for r in results) else 1)


def _store_one(args: tuple[str, str, str, str, str, int, bool, int]) -> dict[str, object]:
    """Worker for `data build-store` (module-level so it pickles)."""
    from dataclasses import asdict

    from aerosurr.config import Freestream, SurfaceFields
    from aerosurr.data.store import build_entry

    raw, store, run, fields_json, fs_json, pool, full, seed = args
    e = build_entry(Path(raw), Path(store), run, SurfaceFields.model_validate_json(fields_json),
                    Freestream.model_validate_json(fs_json), pool, full, seed)  # fmt: skip
    return asdict(e)


@data_app.command("build-store")
def data_build_store(
    runs: Annotated[
        str | None, typer.Option(help="Comma-separated run ids (default: all).")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, help="Parallel processes.")] = 8,
    cpus: Annotated[str | None, typer.Option(help="Pin the workers to these CPUs.")] = None,
    report: Annotated[Path | None, typer.Option(help="Validation report (markdown).")] = None,
) -> None:
    """Build the compact store and validate every run's integrated forces on the way."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from aerosurr.data.splits import all_runs, load_manifest
    from aerosurr.data.store import StoreEntry, load_index, save_index
    from aerosurr.data.validate import RunValidation, render_markdown
    from aerosurr.rans.run import parse_cpus

    paths, data = state.paths(), state.data()
    sc = data.store
    raw, store = paths.data_raw, paths.data_store / sc.version
    manifest = load_manifest(raw / "splits/manifest.json")
    full = {r for s in sc.full_splits for r in manifest[s]}
    run_list = runs.split(",") if runs else all_runs(manifest)
    entries = load_index(store)
    todo = [r for r in run_list if r not in entries or not (store / entries[r].path).is_file()]
    if cpus:
        os.sched_setaffinity(0, parse_cpus(cpus))
    log_file = setup_logging(paths.logs / "aero-store", "build")
    import logging

    log = logging.getLogger("aerosurr.store")
    log.info("store %s: %d runs, %d to build, %d workers; log %s", store, len(run_list),
             len(todo), workers, log_file)  # fmt: skip
    meta = {"pool_size": sc.pool_size, "seed": sc.seed, "full_splits": sc.full_splits,
            "revision": data.source.revision}  # fmt: skip
    fj, fsj = data.fields.model_dump_json(), data.freestream.model_dump_json()
    args = [(str(raw), str(store), r, fj, fsj, sc.pool_size, r in full, sc.seed) for r in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_store_one, a) for a in args]
        for i, fut in enumerate(as_completed(futs), 1):
            e = StoreEntry(**fut.result())  # type: ignore[arg-type]
            entries[e.run] = e
            save_index(store, entries, meta)
            log.info("[%d/%d] %s %s cells, %s, %.1f s, gate %s", i, len(todo), e.run,
                     f"{e.n_cells:,}", "full+pool" if e.full else "pool", e.seconds,
                     "ok" if e.passed else "FAIL")  # fmt: skip
    keys = ("cd", "cl", "clf", "clr", "cs")
    results = [RunValidation(run=e.run, n_cells=e.n_cells,
                             ours={w: {k: e.coeffs[w][k] for k in keys} for w in e.coeffs},
                             ref=e.ref) for e in entries.values() if e.run in run_list]  # fmt: skip
    text = render_markdown(results)
    typer.echo(text)
    if report:
        write_generated(report, text, "Reference-force validation")
    raise typer.Exit(0 if all(r.passed for r in results) else 1)


@data_app.command("stats")
def data_stats() -> None:
    """Normalisation statistics from the training split's pools -> <store>/stats.json."""
    from aerosurr.data.splits import load_manifest
    from aerosurr.data.store import STATS, compute_stats

    paths, data = state.paths(), state.data()
    store = paths.data_store / data.store.version
    runs = load_manifest(paths.data_raw / "splits/manifest.json")[data.store.stats_split]
    stats = compute_stats(store, runs) | {"split": data.store.stats_split}
    (store / STATS).write_text(json.dumps(stats, indent=2))
    typer.echo(json.dumps(stats, indent=2))


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


# --------------------------------------------------------------------------- rans

CASE_META = "aerosurr_case.json"


@rans_app.command("setup")
def rans_setup(
    run: Annotated[str, typer.Argument(help="DrivAerML run id, e.g. run_1.")],
    preset: Annotated[str, typer.Option(help="Mesh preset in configs/rans/<config>.yaml.")],
    ranks: Annotated[int, typer.Option(min=1, max=30, help="MPI ranks (SPEC cap: 30).")],
    config: Annotated[str, typer.Option(help="configs/rans/<config>.yaml")] = "coarse",
    iterations: Annotated[int | None, typer.Option(help="Override solver.iterations.")] = None,
    out: Annotated[Path | None, typer.Option(help="Case dir (default under paths.rans).")] = None,
) -> None:
    """Write an OpenFOAM case for one geometry from its STL and geo_ref CSV."""
    from aerosurr.config import load_rans
    from aerosurr.data.splits import run_number
    from aerosurr.data.validate import read_geo_refs
    from aerosurr.rans.case import write_case

    paths, data = state.paths(), state.data()
    cfg = load_rans(config, state.config_dir)
    if iterations is not None:
        cfg = cfg.model_copy(
            update={"solver": cfg.solver.model_copy(update={"iterations": iterations})}
        )
    n = run_number(run)
    src = paths.data_raw / run
    case = out or paths.rans / "cases" / f"{run}_{preset}_np{ranks}"
    if case.exists():
        typer.echo(f"{case} exists; remove it or pass --out", err=True)
        raise typer.Exit(2)
    ref = read_geo_refs(src / f"geo_ref_{n}.csv").varying
    grid = write_case(case, cfg, preset, src / f"drivaer_{n}.stl", data.freestream, ref, ranks)
    meta = {"run": run, "preset": preset, "config": config, "ranks": ranks,
            "iterations": cfg.solver.iterations, "average_last": cfg.solver.average_last,
            "potential_init": cfg.solver.potential_init,
            "keep_volume": cfg.output.keep_volume,
            "background_cells": grid.n_cells}  # fmt: skip
    (case / CASE_META).write_text(json.dumps(meta, indent=2))
    typer.echo(f"{case}  ({grid.n_cells:,} background cells of {grid.h} m)")


@rans_app.command("run")
def rans_run(
    case: Annotated[Path, typer.Argument(help="Case dir from `aerosurr rans setup`.")],
    cpus: Annotated[str, typer.Option(help="CPU list, one rank per CPU, e.g. 0-23.")],
) -> None:
    """Mesh and solve a case pinned to `--cpus`; writes aerosurr_run.json."""
    from aerosurr.rans.run import cpu_count_ok, parse_cpus, read_force_mom, run_case

    meta = json.loads((case / CASE_META).read_text())
    cpu_list = parse_cpus(cpus)
    if len(cpu_list) != meta["ranks"]:
        raise typer.BadParameter(
            f"{len(cpu_list)} CPUs given, case is decomposed for {meta['ranks']}"
        )
    cpu_count_ok(cpu_list)
    paths = state.paths()
    # Not "aero-rans": that is the batch's log dir, and its latest.log must stay the batch's.
    setup_logging(paths.logs / "aero-rans-run", case.name)
    from aerosurr.data.splits import run_number

    fm = paths.data_raw / meta["run"] / f"force_mom_{run_number(meta['run'])}.csv"
    res = run_case(case.resolve(), cpu_list, meta["preset"], meta["potential_init"],
                   meta["average_last"], read_force_mom(fm) if fm.is_file() else None,
                   keep_volume=meta.get("keep_volume", False))  # fmt: skip
    raise typer.Exit(0 if res.status == "done" else 1)


@rans_app.command("batch")
def rans_batch(
    runs_file: Annotated[Path, typer.Argument(help="Comma- or newline-separated run ids.")],
    preset: Annotated[str, typer.Option(help="Mesh preset.")],
    slots: Annotated[str, typer.Option(help='CPU slots, e.g. "A=0-23;B=32-55".')],
    config: Annotated[str, typer.Option(help="configs/rans/<config>.yaml")] = "coarse",
    iterations: Annotated[int | None, typer.Option(help="Override solver.iterations.")] = None,
    out: Annotated[Path | None, typer.Option(help="Cases root (default paths.rans/batch).")] = None,
) -> None:
    """Run the LF RANS queue over CPU slots, one case per slot at a time; resumable."""
    import logging
    import re

    from aerosurr.config import load_rans
    from aerosurr.data.splits import run_number
    from aerosurr.data.validate import read_geo_refs
    from aerosurr.rans.batch import Slot, case_dir, is_done, run_queue
    from aerosurr.rans.case import write_case
    from aerosurr.rans.run import cpu_count_ok, openfoam_env, parse_cpus, read_force_mom, run_case

    paths, data = state.paths(), state.data()
    cfg = load_rans(config, state.config_dir)
    if iterations is not None:
        cfg = cfg.model_copy(
            update={"solver": cfg.solver.model_copy(update={"iterations": iterations})}
        )
    root = out or paths.rans / "batch"
    slot_list = []
    for part in slots.split(";"):
        name, _, cpus = part.partition("=")
        slot_list.append(Slot(name.strip(), parse_cpus(cpus)))
        cpu_count_ok(slot_list[-1].cpus)
    run_list = [r for r in re.split(r"[,\s]+", runs_file.read_text()) if r]
    setup_logging(paths.logs / "aero-rans", "batch")
    log = logging.getLogger("aerosurr.rans")
    env = openfoam_env()
    todo = [r for r in run_list if not any(is_done(case_dir(root, r, preset, len(s.cpus)))
                                           for s in slot_list)]  # fmt: skip
    log.info("batch: %d runs, %d done, %d to go; preset %s, %d iterations; slots %s",
             len(run_list), len(run_list) - len(todo), len(todo), preset, cfg.solver.iterations,
             {s.name: f"{len(s.cpus)} ranks" for s in slot_list})  # fmt: skip

    def work(run: str, slot: Slot) -> bool:
        n = run_number(run)
        src = paths.data_raw / run
        case = case_dir(root, run, preset, len(slot.cpus))
        if case.exists() and not is_done(case):
            import shutil

            shutil.rmtree(case)  # a half-finished case: start it again cleanly
        ref = read_geo_refs(src / f"geo_ref_{n}.csv").varying
        write_case(case, cfg, preset, src / f"drivaer_{n}.stl", data.freestream, ref,
                   len(slot.cpus))  # fmt: skip
        fm = src / f"force_mom_{n}.csv"
        res = run_case(case, slot.cpus, preset, cfg.solver.potential_init,
                       cfg.solver.average_last, read_force_mom(fm), env,
                       keep_volume=cfg.output.keep_volume)  # fmt: skip
        return res.status == "done"

    results = run_queue(todo, slot_list, work)
    failed = [r for r, ok in results.items() if not ok]
    log.info("batch finished: %d ok, %d failed %s", len(results) - len(failed), len(failed), failed)
    raise typer.Exit(1 if failed else 0)


@rans_app.command("map")
def rans_map(
    root: Annotated[
        Path | None, typer.Option(help="Cases root (default paths.rans/batch).")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-map runs already mapped.")] = False,
    cpus: Annotated[str | None, typer.Option(help="Pin to these CPUs.")] = None,
) -> None:
    """Map every finished LF case's car surface onto its HF cells -> <store>/lf/<run>.npz."""
    import os

    from aerosurr.rans.mapping import LF_DIR, map_run
    from aerosurr.rans.run import RESULT_NAME, parse_cpus

    paths, data = state.paths(), state.data()
    if cpus:
        os.sched_setaffinity(0, parse_cpus(cpus))
    store = paths.data_store / data.store.version
    root = root or paths.rans / "batch"
    index_path = store / LF_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    for case in sorted(root.glob("*/")):
        r = json.loads((case / RESULT_NAME).read_text()) if (case / RESULT_NAME).is_file() else {}
        if r.get("status") != "done" or not r.get("surface"):
            continue
        run = case.name.split("_coarse")[0]
        if run in index and not force:
            continue
        st = map_run(case / r["surface"], store, run, data.freestream.q_inf, data.freestream.p_inf)
        index[run] = st | {"case": case.name, "coeffs": r["coeffs"], "reference": r["reference"]}
        index_path.write_text(json.dumps(index, indent=1))
        typer.echo(f"{run}: {st}")
    typer.echo(f"{len(index)} runs mapped -> {store / LF_DIR}")


@rans_app.command("lf-runs")
def rans_lf_runs(
    split: Annotated[str, typer.Option(help="Split to draw from.")] = "full_train",
    n: Annotated[int | None, typer.Option(help="First n, in RANS queue order.")] = None,
) -> None:
    """Runs of a split that have mapped LF fields, in RANS queue order (comma-separated)."""
    from aerosurr.data.splits import load_manifest
    from aerosurr.rans.batch import rans_subset
    from aerosurr.rans.mapping import LF_DIR

    paths, data = state.paths(), state.data()
    lf_dir = paths.data_store / data.store.version / LF_DIR
    manifest = load_manifest(paths.data_raw / "splits/manifest.json")
    members = set(manifest[split])
    runs = [r for r in rans_subset(manifest) if r in members and (lf_dir / f"{r}.npz").is_file()]
    if n is not None:
        if len(runs) < n:
            typer.echo(f"only {len(runs)} {split} runs have LF fields (wanted {n})", err=True)
            raise typer.Exit(1)
        runs = runs[:n]
    typer.echo(",".join(runs))


@rans_app.command("summary")
def rans_summary(
    cases: Annotated[list[Path], typer.Argument(help="Case dirs with aerosurr_run.json.")],
) -> None:
    """Markdown table of timings, cell counts and Cd/Cl against the HF reference."""
    from aerosurr.rans.run import RESULT_NAME

    typer.echo("| Case | Ranks | CPUs | Status | Cells | Mesh (min) | Solve (min) | s/iter | "
               "Total (min) | Cd | Cd HF | ΔCd (counts) | Cl | Cl HF |")  # fmt: skip
    typer.echo("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cases:
        r = json.loads((c / RESULT_NAME).read_text())
        st = r["steps"]
        mesh = sum(v for k, v in st.items() if k not in ("potentialFoam", "simpleFoam")) / 60
        solve = (st.get("potentialFoam", 0) + st.get("simpleFoam", 0)) / 60
        cd, cl = r["coeffs"].get("Cd"), r["coeffs"].get("Cl")
        cd_hf, cl_hf = r["reference"].get("Cd"), r["reference"].get("Cl")
        dcd = f"{(cd - cd_hf) * 1000:+.0f}" if cd is not None and cd_hf is not None else "-"

        def f(x: float | None, fmt: str) -> str:
            return format(x, fmt) if x is not None else "-"

        typer.echo(
            f"| {r['case']} | {r['n_ranks']} | {r['cpus'] if len(r['cpus']) < 20 else '…'} | "
            f"{r['status']} | {f(r['cells'], ',')} | {mesh:.1f} | {solve:.1f} | "
            f"{f(r['s_per_iter'], '.3f')} | {r['total_s'] / 60:.1f} | {f(cd, '.4f')} | "
            f"{f(cd_hf, '.4f')} | {dcd} | {f(cl, '.4f')} | {f(cl_hf, '.4f')} |"
        )


# --------------------------------------------------------------------------- train / eval


def _cuda_device(gpu: int) -> torch.device:
    import os

    import torch

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        typer.echo("set CUDA_VISIBLE_DEVICES explicitly (SPEC §2: GPU 0 main, GPU 1 ensembles)",
                   err=True)  # fmt: skip
        raise typer.Exit(2)
    if not torch.cuda.is_available():
        typer.echo("CUDA not available", err=True)
        raise typer.Exit(2)
    return torch.device("cuda", gpu)


@train_app.command("run")
def train_run(
    model: Annotated[str, typer.Option(help="configs/model/<model>.yaml")] = "transolver",
    train: Annotated[str, typer.Option(help="configs/train/<train>.yaml")] = "main",
    name: Annotated[str | None, typer.Option(help="Run name (dir under paths.runs).")] = None,
    gpu: Annotated[int, typer.Option(help="Index among CUDA_VISIBLE_DEVICES.")] = 0,
    cpus: Annotated[str | None, typer.Option(help="Pin this process to these CPUs.")] = None,
    session: Annotated[str, typer.Option(help="Log dir name under paths.logs.")] = "aero-train0",
    seed: Annotated[
        int | None, typer.Option(help="Override train.seed (ensemble members).")
    ] = None,
    train_runs: Annotated[
        str | None, typer.Option(help="Comma-separated training runs (overrides the split).")
    ] = None,
) -> None:
    """Train a surface model on the compact store; checkpoints and metrics in paths.runs/<name>."""
    import os
    from datetime import datetime

    from aerosurr.config import load_model, load_train
    from aerosurr.data.splits import load_manifest
    from aerosurr.data.store import STATS
    from aerosurr.rans.run import parse_cpus

    paths, data = state.paths(), state.data()
    configure_runtime(paths.workspace)
    if cpus:
        os.sched_setaffinity(0, parse_cpus(cpus))
    device = _cuda_device(gpu)
    from aerosurr.train.loop import train as train_loop

    mc, tc = load_model(model, state.config_dir), load_train(train, state.config_dir)
    if seed is not None:
        tc = tc.model_copy(update={"seed": seed})
    if train_runs:
        tc = tc.model_copy(update={"train_runs": [r.strip() for r in train_runs.split(",")]})
    name = name or f"{model}_{train}_{datetime.now():%Y%m%dT%H%M}"
    store = paths.data_store / data.store.version
    stats = json.loads((store / STATS).read_text())
    setup_logging(paths.logs / session, name)
    manifest = load_manifest(paths.data_raw / "splits/manifest.json")
    s = train_loop(name, mc, tc, manifest, store, paths.data_raw, stats, paths.runs / name, device)
    typer.echo(json.dumps(s.__dict__, indent=2, default=str))


@eval_app.command("lf")
def eval_lf(
    split: Annotated[str, typer.Option(help="Split whose mapped runs to evaluate.")] = "full_test",
) -> None:
    """'LF alone' baseline (SPEC §5.3, model 1): mapped RANS fields vs HF on the native mesh."""
    from aerosurr.data.splits import load_manifest, run_number
    from aerosurr.data.store import load_group
    from aerosurr.data.validate import read_geo_refs
    from aerosurr.eval.metrics import FIELD_NAMES, relative_l2
    from aerosurr.physics.forces import integrate
    from aerosurr.rans.mapping import LF_DIR
    from aerosurr.train.data import load_lf

    paths, data = state.paths(), state.data()
    store = paths.data_store / data.store.version
    runs = [r for r in load_manifest(paths.data_raw / "splits/manifest.json")[split]
            if (store / LF_DIR / f"{r}.npz").is_file()]  # fmt: skip
    import numpy as np

    rows = {}
    for r in runs:
        g = load_group(store, r, "full")
        lf = load_lf(store, r, "full").astype(np.float64)
        true = np.concatenate([g["cp"].reshape(-1, 1), g["cf"].reshape(-1, 3)], 1)
        a_in = -g["normals"].astype(np.float64) * g["areas"].astype(np.float64)[:, None]
        ref = read_geo_refs(paths.data_raw / r / f"geo_ref_{run_number(r)}.csv").varying
        rows[r] = {
            "rel_l2": dict(
                zip(FIELD_NAMES, relative_l2(lf, true, g["areas"]).tolist(), strict=True)
            ),
            "pred": integrate(lf[:, 0], lf[:, 1:], a_in, g["centres"], ref).as_dict(),
            "true": integrate(true[:, 0], true[:, 1:], a_in, g["centres"], ref).as_dict(),
        }
    if not rows:
        typer.echo(f"no mapped LF runs in {split}")
        raise typer.Exit(1)
    dcd = np.array([v["pred"]["cd"] - v["true"]["cd"] for v in rows.values()])
    rel = {f: float(np.mean([v["rel_l2"][f] for v in rows.values()])) for f in FIELD_NAMES}
    out = {"split": split, "runs": len(rows), "rel_l2": rel,
           "cd_mae_counts": float(np.abs(dcd).mean() * 1000), "per_run": rows}  # fmt: skip
    (store / LF_DIR / f"eval_{split}.json").write_text(json.dumps(out, indent=1))
    typer.echo(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=2))


@eval_app.command("ensemble")
def eval_ensemble(
    run_dirs: Annotated[list[Path], typer.Argument(help="Member run directories.")],
    name: Annotated[str, typer.Option(help="Output dir under paths.runs.")],
    calib: Annotated[str, typer.Option(help="Calibration split.")] = "full_val",
    split: Annotated[str, typer.Option(help="Report split.")] = "full_test",
    gpu: Annotated[int, typer.Option(help="Index among CUDA_VISIBLE_DEVICES.")] = 0,
) -> None:
    """Deep-ensemble uncertainty: calibrate on `calib`, coverage/NLL/Spearman on `split`."""
    import torch

    from aerosurr.config import ModelConfig, TrainConfig
    from aerosurr.data.splits import load_manifest
    from aerosurr.eval.ensemble import evaluate_ensemble
    from aerosurr.models.surface import build_model
    from aerosurr.train.data import Normaliser

    paths, data = state.paths(), state.data()
    configure_runtime(paths.workspace)
    device = _cuda_device(gpu)
    members = []
    stats = None
    tc = None
    for d in run_dirs:
        cfg = json.loads((d / "config.json").read_text())
        mc, tc = ModelConfig.model_validate(cfg["model"]), TrainConfig.model_validate(cfg["train"])
        m = build_model(mc).to(device)
        m.load_state_dict(torch.load(d / "best.pt", map_location=device))
        m.eval()
        members.append((torch.compile(m) if tc.compile else m, mc))
        stats = stats or cfg["stats"]
    assert tc is not None and stats is not None
    manifest = load_manifest(paths.data_raw / "splits/manifest.json")
    store = paths.data_store / data.store.version
    norm = Normaliser.from_stats(stats)
    res = evaluate_ensemble(
        members,
        norm,
        store,
        paths.data_raw,
        manifest[calib],
        manifest[split],
        tc.points,
        tc.bf16,
        device,
    )
    out = paths.runs / name
    out.mkdir(parents=True, exist_ok=True)
    res["member_dirs"] = [str(d) for d in run_dirs]
    (out / f"ensemble_{split}.json").write_text(json.dumps(res, indent=1))
    typer.echo(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=2))


@eval_app.command("run")
def eval_run(
    run_dir: Annotated[Path, typer.Argument(help="A training run directory.")],
    split: Annotated[str, typer.Option(help="full_test or full_val.")] = "full_test",
    checkpoint: Annotated[str, typer.Option(help="best.pt or last.pt")] = "best.pt",
    gpu: Annotated[int, typer.Option(help="Index among CUDA_VISIBLE_DEVICES.")] = 0,
) -> None:
    """Full-mesh field and force errors of a trained model on a split -> <run>/<split>.json."""
    import torch

    from aerosurr.config import ModelConfig, TrainConfig
    from aerosurr.data.splits import load_manifest
    from aerosurr.models.surface import build_model
    from aerosurr.train.data import Normaliser
    from aerosurr.train.loop import evaluate_full

    paths, data = state.paths(), state.data()
    configure_runtime(paths.workspace)
    device = _cuda_device(gpu)
    cfg = json.loads((run_dir / "config.json").read_text())
    mc, tc = ModelConfig.model_validate(cfg["model"]), TrainConfig.model_validate(cfg["train"])
    model = build_model(mc).to(device)
    model.load_state_dict(torch.load(run_dir / checkpoint, map_location=device))
    model.eval()
    fwd = torch.compile(model) if tc.compile else model
    runs = load_manifest(paths.data_raw / "splits/manifest.json")[split]
    store = paths.data_store / data.store.version
    res = evaluate_full(fwd, Normaliser.from_stats(cfg["stats"]), store, paths.data_raw, runs,
                        tc.points, device, tc.bf16, mc.lf_inputs, mc.predict_residual,
                        mc.heteroscedastic)  # fmt: skip
    (run_dir / f"{split}.json").write_text(json.dumps(res, indent=1))
    typer.echo(json.dumps({k: v for k, v in res.items() if k != "per_run"}, indent=2))


@app.command()
def report(
    out: Annotated[Path, typer.Option(help="Markdown report.")] = Path("docs/results.md"),
    figures: Annotated[Path, typer.Option(help="Figure directory.")] = Path("docs/figures"),
) -> None:
    """Results tables and figures from the evaluations that exist so far."""
    from aerosurr.eval.report import build_report

    paths, data = state.paths(), state.data()
    done = build_report(paths.runs, paths.data_store / data.store.version, out, figures)
    typer.echo(f"{out}: sections {done or 'none yet'}")


# --------------------------------------------------------------------------- viewer

#: Fixed colour-map ranges, so every car is shown on the same scale.
VIEWER_RANGES = {
    "cp_true": (-1.5, 1.0),
    "cp_pred": (-1.5, 1.0),
    "cp_err": (-0.3, 0.3),
    "cf_true": (0.0, 0.008),
    "cf_pred": (0.0, 0.008),
    "cp_sigma": (0.0, 0.15),
}


@viewer_app.command("export")
def viewer_export(
    runs: Annotated[str, typer.Option(help="Comma-separated runs (test cars).")],
    model_dir: Annotated[Path, typer.Option("--model", help="Main model run directory.")],
    members: Annotated[
        list[Path] | None, typer.Option("--member", help="Ensemble members (sigma).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output dir (default ~/aero/viewer/data).")
    ] = None,
    cell: Annotated[float, typer.Option(help="Decimation cell size (m).")] = 0.025,
    gpu: Annotated[int, typer.Option(help="Index among CUDA_VISIBLE_DEVICES.")] = 0,
) -> None:
    """Decimated cars with HF / predicted Cp, |Cf|, error and sigma for the three.js viewer."""
    import numpy as np
    import torch

    from aerosurr.config import ModelConfig, TrainConfig
    from aerosurr.data.splits import run_number
    from aerosurr.data.store import load_group, load_index
    from aerosurr.data.validate import read_geo_refs
    from aerosurr.data.vtp import read_vtp
    from aerosurr.eval.ensemble import combine
    from aerosurr.models.surface import build_model
    from aerosurr.physics.forces import integrate_torch
    from aerosurr.train.data import Normaliser
    from aerosurr.train.loop import predict_tensor
    from aerosurr.viewer.export import decimate, nearest_cell_values, write_bin

    paths, data = state.paths(), state.data()
    configure_runtime(paths.workspace)
    device = _cuda_device(gpu)
    store = paths.data_store / data.store.version
    out = out or paths.workspace / "viewer" / "data"
    index = load_index(store)

    def load(d: Path) -> tuple[torch.nn.Module, ModelConfig, TrainConfig, dict[str, object]]:
        cfg = json.loads((d / "config.json").read_text())
        mc, tc = ModelConfig.model_validate(cfg["model"]), TrainConfig.model_validate(cfg["train"])
        m = build_model(mc).to(device)
        m.load_state_dict(torch.load(d / "best.pt", map_location=device))
        return m.eval(), mc, tc, cfg["stats"]

    main, _mc, tc, stats = load(model_dir)
    norm = Normaliser.from_stats(stats)
    ens = [load(d) for d in members or []]
    cars = []
    for r in runs.split(","):
        n = run_number(r)
        g = {k: torch.from_numpy(v).to(device) for k, v in load_group(store, r, "full").items()}
        mu, _ = predict_tensor(main, norm, g["centres"], g["normals"], tc.points, tc.bf16)
        a_in = -g["normals"].double() * g["areas"].double()[:, None]
        ref = read_geo_refs(paths.data_raw / r / f"geo_ref_{n}.csv").varying
        fp = integrate_torch(mu[:, 0], mu[:, 1:], a_in, g["centres"], ref)
        fields = {
            "cp_true": g["cp"].float().cpu().numpy(),
            "cp_pred": mu[:, 0].cpu().numpy(),
            "cp_err": (mu[:, 0] - g["cp"].float()).cpu().numpy(),
            "cf_true": torch.linalg.vector_norm(g["cf"].float(), dim=1).cpu().numpy(),
            "cf_pred": torch.linalg.vector_norm(mu[:, 1:], dim=1).cpu().numpy(),
        }
        cd_sigma = None
        if ens:
            preds = [predict_tensor(m, norm, g["centres"], g["normals"], t.points, t.bf16,
                                    c.heteroscedastic) for m, c, t, _ in ens]  # fmt: skip
            comb = combine([p[0] for p in preds], [p[1] for p in preds])
            fields["cp_sigma"] = comb.var[:, 0].sqrt().float().cpu().numpy()
            cds = [
                integrate_torch(p[0][:, 0], p[0][:, 1:], a_in, g["centres"], ref).cd for p in preds
            ]
            cd_sigma = float(np.std(cds))
        surf = read_vtp(paths.data_raw / r / f"boundary_{n}.vtp")
        verts, tris = decimate(surf.points, surf.offsets, surf.connectivity, cell)
        vals = nearest_cell_values(g["centres"].cpu().numpy(), fields, verts)
        meta = {"run": r, "cd_true": index[r].coeffs["varying"]["cd"], "cd_pred": fp.cd,
                "cl_true": index[r].coeffs["varying"]["cl"], "cl_pred": fp.cl,
                "cd_sigma": cd_sigma, "model": model_dir.name}  # fmt: skip
        h = write_bin(out / f"{r}.bin", verts, tris, vals, meta, VIEWER_RANGES)
        cars.append({k: v for k, v in h.items() if k != "header_bytes"} | {"file": f"{r}.b64.txt"})
        typer.echo(f"{r}: {h['n_vertices']:,} vertices, {h['n_triangles']:,} triangles, "
                   f"Cd {meta['cd_true']:.4f} -> {fp.cd:.4f}")  # fmt: skip
        del g, mu
    (out / "index.json").write_text(json.dumps({"cars": cars, "ranges": VIEWER_RANGES}, indent=1))
    typer.echo(f"{len(cars)} cars -> {out}")


if __name__ == "__main__":
    app()
