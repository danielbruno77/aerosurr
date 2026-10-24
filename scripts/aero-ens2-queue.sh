#!/usr/bin/env bash
# ensemble_v2 (SPEC §5.4): 5 heteroscedastic GeoTransolver members at the main-model budget,
# from a shared queue served by two workers:
#   primary  GPU 1, tmux aero-ens1:ens2       starts after ensemble_v1 is calibrated, takes
#                                            members, then calibrates v2 and regenerates
#                                            docs/results.md
#   filler   GPU 0, tmux aero-train0:ens2fill only in predicted MF-queue gaps; pre-empts its
#                                            member within seconds when MF needs GPU 0
# CPU: training on CCD offsets 4 (GPU 1) and 3 (GPU 0), disjoint from the OpenFOAM slot.
#   start:    scripts/aero-ens2-queue.sh
#   monitor:  tail -f ~/aero/logs/aero-ens2/*_worker_*.log
#             ls -t ~/aero/logs/aero-ens2/*ens2_m*.log | head -2   (member training logs)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
A=$REPO/.venv/bin/aerosurr
C1=4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124
C0=3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,123
WAIT=$HOME/aero/runs/ensemble_v1/ensemble_full_test.json
tmux new-window -t aero-ens1 -n ens2 -c "$REPO" \
  "$A ensemble worker --role primary --gpu 1 --cpus $C1 --wait-for $WAIT; echo '[ens2 primary done]'; exec \$SHELL"
tmux new-window -t aero-train0 -n ens2fill -c "$REPO" \
  "$A ensemble worker --role filler --gpu 0 --cpus $C0; echo '[ens2 filler done]'; exec \$SHELL"
echo "ensemble_v2 workers started: aero-ens1:ens2 (GPU 1) and aero-train0:ens2fill (GPU 0)"
