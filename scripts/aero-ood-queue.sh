#!/usr/bin/env bash
# ensemble_ood_v1 (SPEC §5.4, OOD check): 5 heteroscedastic GeoTransolver members with
# ensemble_v2's config and budget (configs/train/ood.yaml = main.yaml on the geometry split,
# seeds 100-104), on the `geometry` data variant (own store and train-only normalisation
# statistics, configs/data/geometry.yaml). Two workers share one claim queue:
#   primary  GPU 1, tmux aero-ood:primary  takes members now; when all 5 are done, calibrates
#                                          on geometry_val (ID), reports on geometry_test
#                                          (OOD), runs `eval ood` and regenerates
#                                          docs/results.md
#   filler   GPU 0, tmux aero-ood:filler   starts only after the MF queue logs "MF queue
#                                          done"; still pre-empts within seconds if an MF/HF
#                                          process appears
# CPU: CCD offsets 4 (GPU 1) and 3 (GPU 0), disjoint from the OpenFOAM slot (offsets 0-2)
# and the RANS mapping CPUs (offset 5).
#   start:    scripts/aero-ood-queue.sh
#   monitor:  tail -f ~/aero/logs/aero-ood/*_worker_*.log
#             tail -f "$(command ls -t ~/aero/logs/aero-ood/*_ood_m*.log | head -1)"  (member)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/.venv/bin/aerosurr
C1=4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124
C0=3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,123
STATS=$HOME/aero/data/store/v1_geometry/stats.json
grep -q '"split": "geometry_train"' "$STATS" || { echo "no geometry_train stats in $STATS" >&2; exit 1; }
COMMON=(--name ensemble_ood_v1 --prefix ood --train ood --seed0 100 --session aero-ood)
export AEROSURR_DATA_VARIANT=geometry
tmux new-window -t aero-ood -n primary -c "$REPO" -e AEROSURR_DATA_VARIANT=geometry \
  "$A ensemble worker --role primary --gpu 1 --cpus $C1 ${COMMON[*]} --calib geometry_val --split geometry_test --ood; echo '[ood primary done]'; exec \$SHELL"
tmux new-window -t aero-ood -n filler -c "$REPO" -e AEROSURR_DATA_VARIANT=geometry \
  "$A ensemble worker --role filler --gpu 0 --cpus $C0 ${COMMON[*]} --after-mf-done; echo '[ood filler done]'; exec \$SHELL"
echo "ensemble_ood_v1 workers started: aero-ood:primary (GPU 1) and aero-ood:filler (GPU 0, after MF)"
