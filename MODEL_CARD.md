# Model card: aerosurr surface surrogate

Status: draft, written while training runs. Every metric below links to
[docs/results.md](docs/results.md), which `aerosurr report` regenerates from the
evaluation files, so this card never carries numbers by hand.

## Model

| | |
|---|---|
| Task | Time-averaged surface pressure coefficient Cp and skin-friction vector Cf on a car's native surface mesh, from geometry alone |
| Main model | NVIDIA PhysicsNeMo `GeoTransolver`, selected on `full_val` Cd MAE (3.1 against 3.9 counts for `Transolver`): 8 layers × 256 wide, 8 heads, 64 slices, 6.0 M parameters, with the cell positions also passed as geometry |
| Inputs per cell | cell centre (normalised isotropically into ~[-1, 1]), outward unit normal |
| Outputs per cell | Cp, Cf_x, Cf_y, Cf_z, each standardised with `full_train` statistics |
| Multi-fidelity variant | `Transolver` plus the coarse-RANS Cp and Cf mapped onto each cell (7 input features); it learns the HF − LF residual. It is compared with an HF-only `Transolver` trained on the same cars, and both keep their final weights because the val cars have no RANS |
| Uncertainty | **ensemble_v2**, the reported ensemble: 5 heteroscedastic `GeoTransolver` members (mean + log-variance, Gaussian NLL) at the main-model budget, with seeds 100–104 that also change the data order. Temperatures are fitted on `full_val`, one per field and one for Cd. Its mean is also the most accurate Cd predictor here. **ensemble_v1** (5 `Transolver` members at 20k steps) is kept as a baseline |
| Baselines | In-house PointNet-style model (0.4 M parameters); LF RANS alone; scalar correction Cd_HF = a + b·Cd_LF |
| Code | `src/aerosurr` (CLI: `aerosurr train run`, `eval run`, `eval ensemble`, `report`) |

Training samples 200,000 cells per car per step from a fixed 500,000-cell pool per run.
It uses AdamW (lr 1e-3, cosine with warm-up, weight decay 1e-4), bf16 autocast and
`torch.compile`: 150 epochs over the 400 `full_train` cars, about 60k steps, on one
RTX 5090. Model selection is on `full_val`; every number reported is on `full_test`.
Inference runs on the full native mesh (7–10 M cells) in random chunks of 200,000 cells.

## Data

- **DrivAerML** (Ashton et al., 2024, [arXiv:2408.11969](https://arxiv.org/abs/2408.11969)),
  Hugging Face `neashton/drivaerml` at revision `5d448b209bf654503c64ce7261c34fa125f46392`.
  It has 484 parametric morphs of the DrivAer notchback, simulated with scale-resolving
  HRLES (SA-σ-DDES, OpenFOAM v2212) at U∞ = 38.889 m/s, 0° yaw, with static wheels and a
  stationary ground.
- **Licence:** CC BY-SA 4.0. Derived data (the compact store, mapped LF fields, viewer
  exports) is shared under the same licence, with attribution.
- **Splits:** the official `splits/manifest.json`, with `full` = 400 / 34 / 50 and the
  nested `super_scarce` ⊂ `scarce` ⊂ `medium` ⊂ `full` training sets.
- **Integrity:** every file is verified against its Hugging Face sha256 or git oid. All
  484 runs pass the reference-force gate: integrating the stored fields reproduces
  `force_mom*_i.csv` to ≤ 8e-7 ([docs/physics_validation.md](docs/physics_validation.md)).
- **Low fidelity:** in-house OpenFOAM v2506 steady k-ω SST RANS with wall functions, on a
  2.0 M-cell snappyHexMesh mesh, for 100 cars (all 50 test + 50 train). On run_1 it gives
  Cd 0.3013 against HRLES 0.3035 ([docs/rans_benchmark.md](docs/rans_benchmark.md)).

## Intended use

Fast early-stage estimates of surface pressure, wall shear and integrated forces (Cd, Cl,
front/rear lift) for **DrivAer-family notchback shapes inside the DrivAerML design
space**, at the dataset's single operating point. Also a demonstration of an
engineering-grade surrogate workflow: verified data, physics gate, multi-fidelity
correction and calibrated uncertainty.

## Out of scope

Other vehicle classes or topologies (fastback, estate, open wheel). Yaw, ride-height
sweeps beyond the morph range, rotating wheels or a moving ground. Other speeds or
Reynolds numbers. Unsteady loads, volume fields and absolute pressure. Any
safety-relevant or homologation decision.

## Metrics

See [docs/results.md](docs/results.md):
- field errors (area-weighted relative L2 of Cp and each Cf component, native mesh);
- force errors (Cd MAE and max in drag counts, Cd R², Cl MAE);
- data efficiency across the nested splits;
- the MF comparison against HF-only and scalar baselines;
- ensemble calibration (coverage at 50/90/95 %, NLL, Spearman of σ against |error|,
  per cell and per car).

## Limitations

- **One car family, one operating point:** DrivAer notchback morphs at 140 km/h, 0° yaw.
  Extrapolation is not supported. An OOD check on the official `geometry` split
  (`ensemble_ood_v1`) is running; its result will be added here.
- **Time-averaged only.** HRLES side force (Cs) comes largely from wake asymmetry, so it
  is noisy at the dataset's averaging window and hard to predict from geometry.
- **The LF model is coarse:** wall functions on 1–2 cm cells give poor skin friction (LF
  Cf relative L2 ≈ 0.7–0.9). The MF model therefore leans on LF pressure more than on
  LF shear.
- **The MF variant does not beat HF-only.** On 50 LF-mapped test cars its field errors
  are higher at 11, 25 and 50 training cars (Cp relative L2 0.174 / 0.137 / 0.114 against
  0.167 / 0.117 / 0.097). Its Cd is better only at 11 cars (8.5 against 15.1 counts), and
  without better fields behind it. Each model is a single seed. The cheapest use of
  the RANS is a scalar correction, Cd_HF = a + b·Cd_LF (10.9 / 8.3 / 8.3 counts); RANS
  alone gives 8.8. With 50 HF cars, HF-only is best (6.2 counts).
- **Surface subtleties:** the native surface is open at the tyre contact patches. A
  constant pressure offset would shift Cl slightly; p∞ = 0 by construction.
- **Uncertainty is calibrated** on `full_val` (34 cars) in-distribution only.
  Out-of-distribution σ is not validated.
- **Per-cell σ ranks errors well; per-car σ(Cd) does not.** Per-cell coverage is close
  to nominal, and σ correlates with the error at Spearman 0.60–0.75. Per-car σ(Cd) is
  the spread of the members' integrated Cd: epistemic only, since the per-cell
  aleatoric variance (91 % of the field variance) does not enter it. Raw, it is about
  2.5× too small (ensemble_v2: mean 1.3 counts, 90 % coverage 0.64 on 50 test cars).
  After one temperature fitted on `full_val` (2.53) it has the right average size
  (3.3 counts, coverage 0.64 / 0.88 / 0.92 at 50 / 90 / 95 %). Its rank correlation
  with the Cd error is still −0.11 (95 % bootstrap CI −0.42 to +0.20; ensemble_v1
  +0.19, CI −0.09 to +0.45), which is indistinguishable from zero. Use it as an error
  scale, not to flag individual cars.

## Compute and energy

Hardware: AMD Threadripper 9980X (64 cores) and 2 × RTX 5090, single workstation.

| Work | Approximate cost |
|---|---|
| Main Transolver (150 epochs) | ~2.5 GPU-h at ~0.5 kW |
| Data-efficiency runs, ensemble, MF study | ~12 GPU-h |
| Coarse RANS, 100 cases | ~22 h on 24 cores at ~0.25 kW package power |

Final totals go in [docs/results.md](docs/results.md) once the runs finish.

## Citation

Please cite the dataset: Ashton, N., Mockett, C., Fuchs, M., Fliessbach, L., Hetmann, H.,
Knacke, T., Schonwald, N., Skaperdas, V., Fotiadis, G., Walle, A., Hupertz, B. and Maddix,
D., *DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External
Aerodynamics*, arXiv:2408.11969 (2024).
