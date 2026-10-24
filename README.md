# aerosurr

A multi-fidelity 3D surface-flow surrogate on [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml):
it predicts time-averaged surface Cp and wall shear stress on a full car from geometry,
corrects a coarse in-house OpenFOAM RANS towards the dataset's HRLES, and reports
deep-ensemble uncertainty. The design and the plan are in [SPEC.md](SPEC.md).

**Status: Day 0.** The environment, GPU check, data inspection gate and bulk download are
done; models come next.

## Quick start (workstation-III)

```bash
. ~/aero/env.sh                      # uv on PATH, caches under ~/aero
make sync                            # .venv from uv.lock, torch 2.14.1+cu132
make check                           # ruff + mypy + CPU tests
make gpu-check                       # docs/gpu_check.md
make inspect                         # docs/data_inspection.md (inspection gate, SPEC §4.5)
make download                        # T0 + T2 in tmux `aero-dl`; `make status` for progress
```

Every capability is a CLI command: `uv run aerosurr --help`.

| Command | What it does |
|---|---|
| `aerosurr env check` | GPU idle check, torch/CUDA build, bf16 GEMM TFLOPS, `torch.compile` smoke test per GPU |
| `aerosurr data inspect FILE` | every array of a boundary VTP with dtype/shape/stats; `n_cells == len(areas)` gate |
| `aerosurr data download -t t0 -t t2` | resumable, revision-pinned, sha256-verified download, bound to `net.bind_interface` |
| `aerosurr data status` | per-tier download progress (offline) |
| `aerosurr config show` | validated `configs/*.yaml` after env overrides |

## Layout

`src/aerosurr/` is the package, `configs/` the yaml (validated by `aerosurr.config`),
`tests/` the pytest suite (`-m data` needs `AEROSURR_DATA`, `-m gpu` needs CUDA) and
`scripts/` the thin tmux launchers. Data, runs and logs live in `~/aero`, never in the repo.

## Data licence

DrivAerML is © its authors, CC BY-SA 4.0 (Ashton et al., arXiv:2408.11969). Derived data
published from this project inherits ShareAlike.
