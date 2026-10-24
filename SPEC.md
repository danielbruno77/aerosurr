# SPEC: aerosurr, a multi-fidelity 3D surface-flow surrogate on DrivAerML

> **Status: DRAFT v0.2 (2026-10-24, after Day 0).** Written by Claude from Daniel's brief
> plus a read of the real DrivAerML repository (file list, sizes, CSV headers, split
> manifest). v0.2 records what Day 0 verified on real files and on the machine; those
> items are marked **[VERIFIED 2026-10-24]**. Anything marked **[PROPOSAL]** is a
> design choice awaiting Daniel's sign-off. Anything still marked **[VERIFY]** must be
> confirmed by opening a real file before code depends on it.

## 1. Goal

A 4-day portfolio project for the Aerodynamic AI Engineer role at Williams F1.
It should show, end to end and at production quality:

1. **Surface surrogate.** Predict time-averaged surface pressure (Cp) and wall
   shear stress (Cf vector) on a full 3D car from geometry alone, trained on
   DrivAerML scale-resolving (HRLES) results.
2. **Multi-fidelity correction.** Run my own coarse OpenFOAM RANS on a subset of
   the geometries, and learn the correction from coarse RANS to the dataset's
   HRLES. Show what one cheap RANS buys over a geometry-only model.
3. **Uncertainty.** A deep ensemble with calibrated per-point and integrated-force
   uncertainty, which grows where the model is wrong.
4. **Engineering.** A Python package, CLI for everything, yaml configs, tests,
   CI, a model card and an interactive 3D viewer.

**Non-goals:** volume-field prediction, unsteady or transient output, yaw, other
car families, beating SOTA leaderboards. Published baselines are context, not targets.

## 2. Compute: workstation-III

| Resource | Value | Use |
|---|---|---|
| CPU | Threadripper 9980X, 64 cores / 128 threads, 1 NUMA node | OpenFOAM, preprocessing |
| GPU 0 | RTX 5090 32 GB, sm_120 | Main model |
| GPU 1 | RTX 5090 32 GB, sm_120 | Ensembles, ablations |
| RAM | 250 GB | Training data held in RAM or on the GPU |
| Disk | 1 TB NVMe (one LV), **~705 GB free** on 2026-10-24 after cleanup | See §4.4 |
| Network | The default route is a slow VPN tunnel (~6 MB/s); direct LAN uplink `enp97s0` (1 Gb/s link) | Bulk downloads bind to `enp97s0` (`net.bind_interface`): measured **~90 MB/s** with 4 parallel files |
| Driver | 595.99.02, CUDA 13.2 | GPU check: `docs/gpu_check.md` (bf16 GEMM 229 / 231 TFLOPS on GPU 0 / 1, `torch.compile` OK on both) |
| Python | uv-managed CPython 3.12.15 in `~/aero/tools/python` | The system 3.12.3 has no `python3.12-dev`; Triton (`torch.compile`) needs `Python.h`. uv 0.13.5 from conda-forge in `~/aero/tools/uv`. `. ~/aero/env.sh` sets PATH and keeps every cache under `~/aero/cache` |
| OpenFOAM | ESI v2506 (`/usr/lib/openfoam/openfoam2506`) | Coarse RANS |

**Resource rules (from Daniel):**

- GPU 0 runs the main model; GPU 1 runs ensembles and ablations. Every job sets
  `CUDA_VISIBLE_DEVICES` explicitly.
- OpenFOAM runs **at most 2 concurrent cases × 30 MPI ranks**, pinned to disjoint
  physical cores. Training and data-loading processes are pinned to cores
  OpenFOAM never touches. **[VERIFIED 2026-10-24]** with `lscpu -p`: logical CPU
  *c* and *c* + 64 are SMT siblings of physical core *c*; there are 8 CCDs of 8 cores,
  each with its own 32 MB L3 (CCD *k* = cores 8k…8k+7, plus siblings 64+8k…64+8k+7).
  **[DECISION 2026-10-24, measured]** The benchmark in `docs/rans_benchmark.md` set the
  layout:
  - **One case at a time, 24 ranks spread 3 per CCD** (CPUs
    `0-2,8-10,16-18,24-26,32-34,40-42,48-50,56-58`): 0.38 s/iteration on 2 M cells.
  - simpleFoam is limited by each CCD's memory link. 8 ranks packed on one CCD are 2.7×
    slower than 8 spread over eight. Two concurrent cases slow each other 3–6×, so
    2 × 24 CCD-aligned slots (0.88 iterations/s for the machine) and 2 × 16 spread
    (1.1/s) both lose to a single spread case (2.6/s).
  - Training, loaders and preprocessing use CCD offsets 3–7: `3-7,11-15,…,59-63` and
    their siblings.
  - A memory-heavy job there (the store build, 8 workers) still slows OpenFOAM 1.8×
    while it runs, so such jobs are kept short and one-off.
- Training data is preprocessed into a compact store and held in RAM or on the
  GPU, so loaders need ≤ 4 workers per job. Data loading must never compete with
  OpenFOAM for cores.
- Every long job runs in a named tmux session and logs to disk (§8).
- The GPUs are dedicated to the project: no other GPU workloads run while it does. Both
  idle in P8 at 10–14 W with stock max clocks (3150 MHz core, 14001 MHz memory) and a
  600 W limit. Every benchmark re-checks this first. **[VERIFIED 2026-10-24 03:53]**:
  `aerosurr env check` found both GPUs in P8 with 0 compute processes before CUDA init.

## 3. Repository

```
aero-surrogate-demo/
  SPEC.md  README.md  MODEL_CARD.md  LICENSE  pyproject.toml  uv.lock  Makefile
  .pre-commit-config.yaml  .github/workflows/ci.yml
  configs/            # yaml only; validated by pydantic models in aerosurr.config
    paths.yaml  data.yaml  model/*.yaml  train/*.yaml  rans/*.yaml  ensemble.yaml
  src/aerosurr/
    cli.py            # typer app: `aerosurr <group> <command>`
    config.py         # pydantic schemas + yaml loader
    data/             # download.py, inspect.py, vtp.py (I/O), store.py (compact store), splits.py
    physics/          # coefficients.py (Cp, Cf), forces.py (integration, moments)
    models/           # baseline.py, transolver.py, heads.py
    train/            # loop.py, losses.py, ensemble.py
    rans/             # case.py (template), mesh.py, run.py, extract.py, mapping.py (LF→HF surface)
    mf/               # multi-fidelity correction models
    eval/             # metrics.py, calibration.py, plots.py
    viewer/           # export.py (decimate + binary per-vertex fields)
  viewer/             # static three.js page (GitHub Pages / claude.ai artifact)
  tests/              # unit tests; `-m data` tests need real DrivAerML files
  scripts/            # thin tmux launchers only; all logic lives in the package
```

- Python 3.12. uv manages the env and the lockfile. Use ruff (lint + format),
  mypy (strict on `physics/` and `data/`), pytest and pre-commit, the same stack
  as `groundeffect-ml`.
- On the server, torch 2.14.1 comes from the cu132 wheel index (`nvidia-smi` reports
  CUDA 13.2); its arch list includes sm_120. CI uses the CPU wheel index. One `uv.lock`
  covers both through the mutually exclusive extras `cu132` and `cpu`
  (`uv sync --extra cu132` here, `--extra cpu` in CI). Each extra also carries
  `nvidia-physicsnemo` 2.2.2 and a matching `torchvision` 0.29.1, so a plain `uv sync`
  without an extra can never pull PyPI's default torch build.
- Every capability is a CLI command. There are no notebooks on the critical path.

## 4. Data: DrivAerML

### 4.1 Source

- Hugging Face `neashton/drivaerml` (dataset), license **CC BY-SA 4.0**.
- **Pinned revision `5d448b209bf654503c64ce7261c34fa125f46392`** (last modified
  2026-09-13). Every download and cache key uses this hash.
- 484 public runs (`run_1`..`run_500`; 167, 211, 218, 221, 248, 282, 291, 295,
  316, 325, 329, 364, 370, 376, 403 and 473 are absent).

### 4.2 What we know from real files (read 2026-10-24)

| File | Size | Content |
|---|---|---|
| `run_i/boundary_i.vtp` | ~0.66 GB | Surface mesh + time-averaged fields. **[VERIFIED 2026-10-24]** (`docs/data_inspection.md`): VTK XML PolyData, inline base64, uncompressed; Float32 points, Int32 connectivity/offsets; run_1 has 8,846,052 points and 8,828,095 polygons (3–7 vertices, 95 % quads). **Cell data only**, no point data: `CpMeanTrim`, `pMeanTrim` (kinematic p/ρ in m²/s², gauge to p_ref = 0 at the outlet), `pPrime2MeanTrim`, `wallShearStressMeanTrim` (3-vector, kinematic m²/s²), all `<f4`. Field data holds only `TimeValue` = 3.6 |
| `run_i/boundary_cell_area_i.npy` | ~35 MB | float32 `<f4`, one area in m² per native VTP cell, original cell order (per dataset README). **[VERIFIED 2026-10-24]** on run_1: length = n_cells, and Newell areas recomputed from the polygons agree to 6e-8 |
| `run_i/force_mom_i.csv` | tiny | `Cd,Cl,Clf,Clr,Cs`, coefficients only, varying reference per car |
| `run_i/force_mom_constref_i.csv` | tiny | Same columns, constant reference |
| `run_i/geo_ref_i.csv` | tiny | `lRef,aRef,forcesCoR,lRefRef,aRefRef,forcesCoRRef`; run_1: 2.972 m, 2.223 m², CoR (1.493, 0, −0.3176) |
| `run_i/geo_parameters_i.csv` | tiny | 16 morph parameters (Vehicle_Length … Vehicle_Pitch) |
| `run_i/drivaer_i.stl` | ~0.14 GB | Geometry for my own RANS meshing |
| `splits/manifest.json` | tiny | Official splits: `full` 400/34/50, nested `medium`/`scarce`/`super_scarce`, OOD `geometry`/`high_drag`/`low_drag`/`rear_separation` |
| `force_mom_all.csv`, `force_mom_constref_all.csv`, `geo_parameters_all.csv` | tiny | All runs in one file |
| `run_i/volume_i.vtu.0{0,1}.part` | ~49 GB | **Not downloaded** |
| `run_i/slices/`, `images/`, `drivaer_i.step` | ~1.4 GB | **Not downloaded** (slices are an optional stretch goal) |

**Freestream [VERIFIED 2026-10-24].** Neither the VTP nor the README stores U∞, ρ or p∞.
They come from the paper (Ashton et al., arXiv 2408.11969), and the data confirms them:

- U∞ = 38.889 m/s (App. A). A least-squares fit `pMeanTrim = p∞ + q∞·CpMeanTrim` over all
  run_1 cells gives q∞ = 756.17716 m²/s² = ½·38.889² (R² = 1 − 2e-15) and p∞ = 4e-9.
- ρ∞ = 1 kg/m³ for the force files (Table 3), and the stored pressure is kinematic
  (Table 4, App. C). So Cp = pMeanTrim / 756.177 and Cf = τ / 756.177, with no ρ.
- p∞ = p_ref = 0 at the outlet, x = 80 m (App. B.2). ν = 1.507e-5 m²/s, Re_L = 7.19e6 on
  the wheelbase.

Values are in `configs/data.yaml: freestream`.

### 4.3 Download list [PROPOSAL]

All downloads are resumable and pinned to the revision above, and write a
manifest with sha256 and size per file. They run in tmux session `aero-dl`.
Order matters: every prefix of the download is a usable dataset. The
downloader binds its sockets to `net.bind_interface` from
`configs/paths.yaml` (`enp97s0`; SO_BINDTODEVICE, via curl `--interface` or
httpx `socket_options`). An empty value uses the default route.

T1 and the root/split files were fetched on 2026-10-24 by a bootstrap curl
script (`~/aero/bootstrap/fetch_t1.sh`). Every file passed sha256 against the HF
LFS oid. They are in `~/aero/data/raw/drivaerml`.

**Implementation [VERIFIED 2026-10-24]:** `aerosurr data download` (`src/aerosurr/data/`).
It plans each tier from the Hub `paths-info` and `tree` APIs at the pinned revision, so
plans hold sizes, LFS sha256 and git blob oids. It streams each file into `<file>.part`,
resumes with HTTP Range, and verifies before rename: LFS files by sha256, plain git files
by git blob sha1, always by size. Each verified file is appended to `MANIFEST.jsonl`, and
`SHA256SUMS` is rewritten for `sha256sum -c`. The bootstrap T1 files were adopted by
hashing. One trap: httpx applies `socket_options` only *after* `connect()`, so its
SO_BINDTODEVICE strands IPv4 connections that already left via the VPN default route,
and TLS times out.
`BoundSyncBackend` therefore sets SO_BINDTODEVICE before connect, as curl does.

| Tier | Files | Runs | Size |
|---|---|---|---|
| T0 | Root CSVs, `README.md`, `LICENSE.txt`, `splits/**`, `surface_cell_areas/**`, all per-run `*.csv` | all 484 | 2,447 files, 0.01 GB — **done 2026-10-24** |
| T1 | `boundary_1.vtp`, `boundary_cell_area_1.npy`, `drivaer_1.stl` | run_1 | 0.84 GB — **done** |
| — | **Gate:** inspect T1 (§4.5) before starting T2 — **passed 2026-10-24** | | |
| T2 | `boundary_i.vtp` + `boundary_cell_area_i.npy` in order `full_test`, `full_val`, `super_scarce_train`, `scarce_train`, `medium_train`, rest of `full_train` | 484 | 968 files, **327.4 GB** (measured). Started 2026-10-24 04:10 in tmux `aero-dl`, ~90 MB/s, ETA ~1 h |
| T3 | `drivaer_i.stl` for the RANS subset (§6) | ~100 | ~14 GB |

### 4.4 Disk budget

T0–T3 take ~342 GB of the ~705 GB free (749 GB in decimal units at the T2 start). That
leaves ~400 GB for the compact store, checkpoints and RANS cases. The downloader refuses
to start if it would leave less than `download.min_free_gb` (100 GB). Policy:

- All raw files are kept. Deleting train-run VTPs after compaction is the
  fallback if free space drops below 100 GB.
- RANS cases keep only the mesh, the final time step and the extracted surface;
  `processor*` directories are removed after reconstruction.
- Claude never deletes anything outside the project directories.

### 4.5 Inspection gate (Day 0)

`aerosurr data inspect <file>` prints, for one real VTP: number of points and
cells, cell types, every point-data and cell-data array with dtype, shape,
min/max/mean, and the bounding box. It also checks that
`len(boundary_cell_area) == n_cells`. Its output is committed to
`docs/data_inspection.md`, and field names in `configs/data.yaml` are copied
from it, never typed from memory. **Passed on run_1, 2026-10-24.** The command also
recomputes cell areas, the area-vector closure, the signed volume and the p–Cp fit
(§4.2). The surface is watertight except at the four tyre contact patches: 7,383
single-use edges, all at z = −0.3176 m, give Σ A·n = (0, 0, −0.0157 m²).

## 5. Methods

### 5.1 Physics and data pipeline

- `physics.coefficients`: Cp = (p − p∞)/q∞ and Cf = τ/q∞. Both p and τ are kinematic,
  q∞ = ½U∞² = 756.177 m²/s² and p∞ = 0 (§4.2).
- `physics.forces` **[VERIFIED 2026-10-24]**: the conventions were established against
  `force_mom_i.csv` on run_1 and then confirmed on 9 further runs
  (`tests/test_physics.py`):
  - The Newell area vectors **a** (native connectivity order) point *into* the car
    (OpenFOAM: out of the fluid), so the pressure force on the car is +q∞ Σ Cp **a**.
  - The stored `wallShearStressMeanTrim` is the stress on the *fluid*, so the friction
    force on the car is −q∞ Σ **Cf** |**a**| (`SHEAR_SIGN = −1`). With +1, Cd comes out
    19 % low.
  - Cd, Cs and Cl are Fx, +Fy and Fz over q∞·A_ref. Cm = M/(q∞·A_ref·l_ref) about
    `forcesCoR`, and **Clf = ½Cl + Cm_pitch, Clr = ½Cl − Cm_pitch**, where Cm_pitch = Cm_y.
    The paper's "/l_ref" in Eq. 33 is an extraction artefact.
  - The surface is open at the tyre contact patches (§4.5), so a pressure offset ΔCp would
    shift Cl by ≈ 0.007 ΔCp. With p∞ = 0 the fields are integrated as stored.
- **Gate, the physics test** (**passed 2026-10-24**): integrated Cd, Cl, Cs, Clf and Clr
  from the raw VTP match `force_mom_i.csv` to within **0.5 %** (Cd, Cl; absolute floor
  1e-4 for near-zero lift) or **0.002 absolute** (Clf, Clr, Cs). This holds on run_1 and
  9 other runs (unit test), and the same check runs against `force_mom_constref_i.csv`
  with the `*RefRef` values. Measured errors are ≤ 5e-8, the CSVs' 7-digit precision. All
  484 runs are checked by `aerosurr data build-store` (or `data validate`) and reported in
  `docs/physics_validation.md`.
- Compact store (`aerosurr data build-store`, `~/aero/data/store/v1/`):
  - Per run, an `.npz` with cell centres (f4), outward unit normals (f4), areas (f4), Cp
    (f2) and Cf (3 × f2), plus `index.json` with the integrated and reference coefficients.
  - Every run keeps a fixed random **pool of N = 500 k cells** (seeded by run number) for
    training. Only the `full_val` and `full_test` runs also keep **all** cells, for
    full-mesh evaluation.
  - At 36 B per cell (8.83 M cells on run_1) that is ~27 GB for the 84 full runs plus
    ~9 GB of pools, versus ~154 GB at full resolution for everything. The raw VTPs remain
    the source of truth.
  - Normalisation statistics come from the `full_train` pools only (`aerosurr data stats`).

### 5.2 Surrogate (GPU 0)

**[DECISION 2026-10-24, Daniel]** The model comes from **NVIDIA PhysicsNeMo**, which the
job requirements name. PhysicsNeMo is used as a model library inside `aerosurr`. Our own
data store, training loop, losses, MF inputs (§5.3) and heteroscedastic heads (§5.4) wrap
its networks, rather than adopting its Hydra recipes wholesale.

- Inputs: point positions and normals sampled from the car surface (geometry
  only, with no morph parameters, so the model applies to new STLs).
  Outputs: Cp and Cf (3) at query points.
- Baseline: a point-wise MLP plus global geometry encoding (PointNet-style). It
  is cheap and gives a floor.
- Main model: `physicsnemo.models.transolver.Transolver`, with normals as the features
  and positions as the embedding, in bf16 with `torch.compile`. `GeoTransolver` (and
  `domino`) are the comparison candidates if time allows. Training uses random point
  subsets; inference runs in chunks on the full native mesh.
- Feasibility **[VERIFIED 2026-10-24]**: PhysicsNeMo 2.2.2 (Apache-2.0) runs on torch
  2.14.1+cu132, sm_120 and Python 3.12.
  - Transolver (8 layers, width 256, 64 slices, 6.0 M params) on GPU 0 at 200 k points per
    step: 235 ms eager, **127 ms compiled** for fwd + bwd + AdamW, 17.3 GiB peak.
  - GeoTransolver: 133 ms compiled, 17.8 GiB.
  - Fed from host memory (sampling + H2D copy per step): 146 ms.
  - Transformer Engine is not installed (`use_te=False`).
- Loss: per-field relative L2, with an optional integrated-force consistency term.
- Splits: train on `full_train`, select on `full_val`, report on `full_test`.
  Data efficiency uses the nested `super_scarce` ⊂ `scarce` ⊂ `medium` ⊂ `full`
  sets.

### 5.3 Multi-fidelity correction (CPU for RANS, GPU 1 for models) [PROPOSAL]

- **Low fidelity:** OpenFOAM v2506, `snappyHexMesh` from `drivaer_i.stl`, steady
  `simpleFoam` with k-ω SST and wall functions. **[DECISION 2026-10-24, measured]**
  `coarse` preset: 1.99 M cells, 1000 iterations, forces and surface fields averaged over
  the last 300. That is ~8 min per case on the layout in §2, and 100 cases take ~13 h.
  On run_1: Cd 0.3013 against HRLES 0.3035 (−2 counts) and Cl 0.132 against 0.068.
  The domain is 45 × 20 × 10 m (blockage ~1 %). Boundary conditions match the
  DrivAerML setup, which the paper gives as follows (**[VERIFIED 2026-10-24]**,
  arXiv 2408.11969, Baseline section and App. C):
  - U∞ = 38.889 m/s at the inlet, and p = 0 at the outlet.
  - Symmetry on the lateral and top boundaries; blockage ~0.25 %.
  - The ground is slip upstream and a stationary no-slip wall from x = −2.339 m.
  - The wheels are static (not rotating).
  - The solver is OpenFOAM v2212, SA-σ-DDES.

  The domain extents are only in the paper's Fig. 11 **[VERIFY]**. The original ANSA
  meshes are in `openfoam_meshes/` (not downloaded).
- **Subset:** about 100 runs. All 50 `full_test` runs, plus ~50 runs drawn first
  from `scarce_train`. The queue interleaves train and test so that any finished
  prefix is usable. One case at a time (§2), in tmux session `aero-rans`, window `batch`
  (`scripts/aero-rans-batch.sh`). The queue is `~/aero/rans/subset_v1.txt`, from
  `aerosurr.rans.batch.rans_subset`. Each finished case keeps
  `postProcessing/carSurface/1000/car.vtp` (`pMean`, `wallShearStressMean`, `yPlus`), its
  forces and logs: ~90 MB. The `processor*` directories are deleted, and the volume is
  regenerable from the case dictionaries.
- **Mapping:** LF surface fields are interpolated onto the HF native cell centres
  (KD-tree, inverse-distance weighting) and saved next to the HF store entry.
- **Models, all compared on the same test runs:**
  1. LF alone, the coarse RANS as-is.
  2. HF-only surrogate (§5.2) trained on the same HF runs.
  3. **MF correction:** the surrogate gets LF Cp and Cf as extra input features
     and predicts the HF residual.
  4. Scalar force correction (Cd_HF ≈ a + b·Cd_LF) as the classical baseline.
- The headline plot is error versus number of HF training runs for 2, 3 and 4.

### 5.4 Uncertainty (GPU 1)

**[IMPLEMENTED 2026-10-24]** in `eval/ensemble.py` and `aerosurr eval ensemble`, queued by
`scripts/aero-ens-queue.sh` after the data-efficiency runs.

- Deep ensemble of M = 5 members: the §5.2 Transolver with a heteroscedastic head
  (`configs/model/transolver_het.yaml`), ~20k steps each (`configs/train/ens.yaml`).
  Members differ by seed, which changes both initialisation and data order. Each member
  outputs a mean and a variance (Gaussian NLL). Total σ² = mean aleatoric + ensemble
  spread.
- **ensemble_v2 [DECISION 2026-10-24, Daniel]:**
  - **Why:** the v1 members, at 20k steps, are ~2× worse than the main model (val Cp rel
    L2 0.125–0.136 against 0.064), so v1 σ describes a weaker model. v1 is kept as the
    baseline.
  - **Members:** 5 heteroscedastic GeoTransolver members (`geotransolver_het`, seeds
    100–104) at the main-model budget (`train/main.yaml`, ~2.6 h each).
  - **Scheduling:** a shared claim queue (`train/ensemble_queue.py`,
    `scripts/aero-ens2-queue.sh`). GPU 1 takes members after v1 is calibrated, then
    calibrates v2.
  - **GPU 0 filler:** runs a member only when the predicted wait of the MF queue (from
    RANS progress) exceeds a member plus 25 min. It pre-empts within seconds when any MF
    or HF train or eval process appears; the member restarts from scratch, and no member
    moves between GPUs.
- σ for integrated forces comes from integrating each member's field.
- Calibration is fitted on `full_val`, a single temperature per field, and
  reported on `full_test`: coverage at 50/90/95 %, NLL, and Spearman correlation
  between σ and |error|, per point and per car.
- OOD check (stretch): retrain the ensemble on the `geometry` split and show σ
  rising on its test set.

### 5.5 Metrics

- Surface: relative L2 and MAE for Cp and each Cf component on the native mesh,
  area-weighted.
- Forces: MAE in drag counts (1 count = 0.001 Cd), R² for Cd and Cl, and
  max error, all on `full_test`.
- Published baselines (AB-UPT, the PhysicsNeMo-CFD benchmark, Transolver or
  GeoTransolver on DrivAerML) are quoted with citation and the split they used.
  Numbers are never copied without the source.

## 6. Deliverables

- `README.md`: one-paragraph pitch, headline figure, results table,
  reproduce-in-N-commands.
- `MODEL_CARD.md`: data (CC BY-SA 4.0 attribution, ShareAlike on derived data),
  intended use, metrics, calibration, limitations (one car family, notchback,
  one speed, no yaw, time-averaged), compute and energy.
- Interactive 3D viewer: static three.js page. It shows decimated test cars with
  per-vertex Cp, |Cf|, error and σ, a case picker and a colour-map toggle.
  Data is exported by `aerosurr viewer export`:
  - decimation by quadric clustering to ~70k vertices per car;
  - u16-quantised fields with fixed ranges across cars;
  - a base64 text copy for hosting.

  **[IMPLEMENTED 2026-10-24]** as `viewer/index.html` (three.js r169), published
  privately as a claude.ai artifact: https://claude.ai/artifact/TUdgikAyaEBpoyzqejMkPn.
  The current version is a preview with a mid-training checkpoint; it is re-exported with
  the final model and ensemble σ. GitHub Pages needs a public repo, or a paid plan for a
  private one.
- Figures: parity (Cd and Cl with σ bars), error-versus-HF-count (MF), calibration
  curves, Cp on the surface (truth / prediction / error / σ).

## 7. Testing and CI

- Unit tests sit next to the code. Every physics function has a unit test against
  **DrivAerML reference forces** (§5.1 gate), plus an analytic test (for example,
  constant pressure on a closed surface gives zero force, and a known shear field
  gives a known drag).
- Markers: `-m data` needs real files (`AEROSURR_DATA` env var); `-m gpu` needs CUDA.
- GitHub Actions `ci.yml`:
  - `lint`: ruff check, ruff format --check, mypy.
  - `test`: pytest on CPU, without data or GPU markers, with coverage.
  - `physics`: `aerosurr data download -t t0 -t t2 --runs run_1` fetches the `run_1`
    boundary VTP, areas and CSVs plus `splits/` at the pinned revision (cached with
    `actions/cache`, re-verified from `MANIFEST.jsonl`), then runs `pytest -m data`.
    This makes the reference-force gate run in CI on every push. On Day 0 `-m data`
    covers the inspection gate and the Cp/freestream contract; Day 1 adds the
    reference forces.
- Local equivalents: `make check` (lint + test), `make test-data`, `make test-gpu`,
  `make pre-commit`. As of Day 0 there are 54 CPU tests, 3 data tests and 2 GPU tests,
  all green.
- **[DECISION]** A self-hosted GPU runner on workstation-III is optional. The repo is
  private as of 2026-10-24, which removes GitHub's main objection (fork PRs running code
  on the runner). Undecided; until then `-m gpu` and `aerosurr env check` run locally.

## 8. Operations

- Paths, set in `configs/paths.yaml` and overridable by env: repo
  `~/aero-surrogate-demo`, workspace `~/aero/` with `data/raw`, `data/store`,
  `runs/`, `rans/` and `logs/`.
- tmux sessions, each job logging to `~/aero/logs/<session>/<timestamp>.log` (with
  `latest.log`) and writing JSON metrics to its run directory:
  - `aero-dl`: downloads.
  - `aero-store`: store build.
  - `aero-rans`: windows `bench`, `scaling` and `batch`. Ad-hoc `rans run` logs to
    `aero-rans-run`.
  - `aero-train0` (GPU 0): main model, then the `geo` and `mf` queues.
  - `aero-ens1` (GPU 1): windows `ds` and `ens`.

  `aerosurr report` turns the evaluation JSON into `docs/results.md` and
  `docs/figures/`.
- Shell setup: `. ~/aero/env.sh` (uv on PATH; uv, pre-commit, inductor and Triton caches
  under `~/aero/cache`; `AEROSURR_DATA`). Launchers in `scripts/` call `.venv/bin/aerosurr`
  directly, so a running job never re-syncs the env.
- To monitor: `tmux attach -t <session>` (detach with Ctrl-b d),
  `tail -f ~/aero/logs/<session>/latest.log`, `nvidia-smi dmon -s pucm`, `btop`.
- Every task ends with a 3-line status: **done**, **numbers**, **next**.

## 9. Plan

| Day | Main thread | Background |
|---|---|---|
| **0** ✅ | Connection, GPU stack verified, env, repo skeleton + CI, T0/T1 download, inspection gate passed (all done 2026-10-24) | T2 download running in `aero-dl` (started 04:10) |
| **1** ✅ | Physics module + reference-force tests green (all 484 runs); compact store (34 GB); PointNet baseline; PhysicsNeMo Transolver training on GPU 0. RANS pipeline built and validated on run_1 (Cd −2 counts vs HF), coarse mesh and 1000 iterations fixed by benchmark | RANS batch running, 1 case × 24 spread ranks (§2); data-efficiency + ensemble queue on GPU 1; MF queue on GPU 0; viewer preview published |
| **2** | Main-model evaluation and figures; data-efficiency ablation; LF→HF mapping | Ensemble on GPU 1; RANS batch continues |
| **3** | MF correction and comparison; calibration; viewer; model card; README; final numbers | Remaining ensemble or OOD runs |

## 10. Open decisions

1. Download list and order in §4.3. T0, T1 and T2 are running as proposed; T3 waits
   for the RANS subset (5).
2. ~~Disk~~: resolved, 705 GB free. ~~Bandwidth~~: resolved, bind to `enp97s0`.
3. ~~GitHub repo~~: resolved, `danielbruno77/aerosurr`, **private** (2026-10-24). It was
   briefly public with machine details scrubbed first. Self-hosted runner: still open (§7).
4. ~~Model choice~~: resolved, NVIDIA PhysicsNeMo (§5.2).
5. ~~RANS subset and budget~~: resolved by measurement. 100 runs at ~8 min each (target
   was ≤ 30), ~13 h in all (`docs/rans_benchmark.md`). Running since 2026-10-24 06:38.
6. ~~Code licence~~: deferred while the repo is private; decide before making it public.
   The data is CC BY-SA 4.0 either way.
7. ~~CPU slot masks~~: resolved by measurement. One case at a time with 24 ranks spread
   3 per CCD; training on CCD offsets 3–7 (§2). Two-case layouts lose throughput.
8. Git identity: no global `user.name` or `user.email` is configured. The Day-0 commit
   was made with `-c` flags only.
