# RANS budget and core-layout benchmark

Measured on 2026-10-24 on workstation-III (Threadripper 9980X, 64 cores in 8 CCDs of 8,
SMT on) with OpenFOAM v2506, using `aerosurr rans` on run_1. The case is the `coarse`
preset in `configs/rans/coarse.yaml`: 1.99 M cells, snappyHexMesh, steady simpleFoam with
k-ω SST and wall functions. Every MPI rank is bound to its own physical core via an
OpenMPI rankfile. Raw results are in `~/aero/rans/bench/*/aerosurr_run.json`.
Launchers: `scripts/aero-rans-bench.sh` (P0/P1) and `scripts/aero-rans-scaling.sh` (S1–S10).

## Answers

1. **~100 cases at ≤ 30 min each fit easily.** With the chosen layout a case takes
   **~8 min** (1 min meshing + 1000 iterations × 0.38 s + clean-up). 100 cases run
   **one at a time** take **~13 h**.
2. **The recommended 2 × 24 CCD-aligned layout is not adequate. It is the
   worst layout measured.** The solver is limited by memory bandwidth per CCD. Ranks
   packed into whole CCDs saturate each CCD's link to memory, and two concurrent cases
   slow each other 3–6×. The best layout is **one case at a time, 24 ranks spread
   evenly over all 8 CCDs (3 per CCD)**: CPUs `0-2,8-10,16-18,24-26,32-34,40-42,48-50,56-58`.
   That leaves 5 cores per CCD (offsets 3–7, 40 cores + SMT siblings) for the store,
   loaders and training.
3. **Mesh preset: coarse.** At 1500 iterations it gives Cd = 0.3013 ± 0.0028 against
   the HRLES 0.3035 (−2 counts, −0.7 %). Cl is 0.132 against 0.068: RANS over-predicts
   lift, which is what the MF correction is for. The `medium` preset (4.36 M cells,
   6 min of snappyHexMesh) costs ~3× per iteration and does not fit 100 cases in the
   Day 1–3 window.
4. **Iterations: 1000**, with forces and surface fields averaged over the last 300.
   Cd is within ±1.5 counts of its final value from iteration ~300, and Cl settles by
   ~1000 (table below).

## Rank scaling and placement (one case alone unless noted, 200 iterations)

"pack" fills whole CCDs; "spread" puts the same number of ranks on every CCD.

| Run | Ranks | Placement | s / iteration | Iterations/s (machine) | Notes |
|---|---|---|---|---|---|
| S1 | 8 | pack (CCD 0) | 1.740 | 0.57 | one CCD link saturated |
| S2 | 8 | spread (1/CCD) | 0.640 | 1.56 | 2.7× faster than S1 with the same 8 cores |
| S3 | 16 | pack (CCD 0–1) | 1.035 | 0.97 | |
| S4 | 16 | spread (2/CCD) | 0.450 | 2.22 | |
| S5 | 24 | pack (CCD 0–2) | 0.760 | 1.32 | P1 alone: 0.69 |
| **S6** | **24** | **spread (3/CCD)** | **0.380** | **2.63** | **best** |
| S7 | 30 | spread (4/CCD on 6, 3 on 2) | 0.495 | 2.02 | uneven per-CCD load, slower than S6 |
| S8 | 2 × 16 | spread, interleaved | 1.81 / 1.80 | 1.11 | each case 4× slower than alone |
| S9 | 2 × 24 | pack (CCD 0–2 / 4–6) | 2.26 / 2.28 | 0.88 | the original 2 × slot plan |
| S10 | 2 × 12 | spread, interleaved | 1.55 / 1.55 | 1.29 | |

Meshing (snappyHexMesh) takes 1.0–1.4 min at 16–30 spread ranks, 2.0 min at 8 packed
and 1.6 min when two cases mesh at once.

Why: every case's iteration time follows the per-CCD load. 8 ranks on one CCD are 2.7×
slower than 8 ranks on eight CCDs. Two cases on disjoint CCDs still slow each other
3–6×, more than bandwidth sharing alone explains. So the winning strategy is the fewest
concurrent memory streams per CCD with the work spread over all eight. Whatever the
mechanism, the throughput numbers above are what the batch is planned on.

## Convergence of the coarse case (P1, 1500 iterations, run_1)

| Iterations averaged | Cd | Δ vs final (counts) | Cl |
|---|---|---|---|
| 100–300 | 0.3069 | +5.5 | 0.125 |
| 300–500 | 0.3008 | −0.5 | 0.120 |
| 500–700 | 0.2998 | −1.5 | 0.125 |
| 800–1000 | 0.3015 | +0.1 | 0.130 |
| 1200–1500 (final) | 0.3013 ± 0.0028 | 0 | 0.132 ± 0.009 |

HRLES reference (`force_mom_1.csv`): Cd 0.3035, Cl 0.0677, Clf −0.037, Clr 0.105. The
coarse RANS gives Clf −0.011 and Clr 0.143.

## Interference with the training side

| Load on the free cores (offsets 3–7 of each CCD) | RANS s / iteration | Side load |
|---|---|---|
| none | 0.38 | — |
| compact-store build, 8 workers (VTP parse + geometry) | 0.68 | 12–22 s per run instead of ~5 s alone |

The store build is a one-off (~20 min for 484 runs), so it costs the batch ~9 min. The
idle-machine probe (`aerosurr env probe`, P0) measured VTP preprocessing at 4.6 s per
file on 4 workers and a PhysicsNeMo Transolver training step fed from host memory at
146 ms on GPU 0. Training keeps its data on the GPU, so it puts little load on host
memory.

## Decision

- Batch: `scripts/aero-rans-batch.sh` with
  `SLOTS="A=0-2,8-10,16-18,24-26,32-34,40-42,48-50,56-58"`: one slot, 24 spread ranks,
  coarse preset, 1000 iterations, over the 100-run queue in `~/aero/rans/subset_v1.txt`
  (50 `full_test` interleaved with 50 train runs, `super_scarce` → `scarce` first).
  Started 2026-10-24 06:38. Expected to finish ~20:00 the same day.
- Each finished case keeps the car surface (`postProcessing/carSurface/1000/car.vtp`,
  with `pMean`, `wallShearStressMean` and `yPlus`), the forces and the logs: ~90 MB. The
  `processor*` directories are removed.
- Training, store and loader processes go on CPU offsets 3–7 of each CCD.
