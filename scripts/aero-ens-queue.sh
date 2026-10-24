#!/usr/bin/env bash
# Deep ensemble (SPEC §5.4) on GPU 1, tmux `aero-ens1`, window `ens`: M heteroscedastic
# Transolver members (seeds 0..M-1), then calibration on full_val and report on full_test.
#   start:    [M=5] scripts/aero-ens-queue.sh      (waits for the `ds` window to finish)
#   monitor:  tail -f ~/aero/logs/aero-ens1/latest.log
set -euo pipefail
SESSION=aero-ens1
REPO=$(cd "$(dirname "$0")/.." && pwd)
M=${M:-5}
CPUS=${CPUS:-4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124}
A="CUDA_VISIBLE_DEVICES=1 $REPO/.venv/bin/aerosurr"
CMD="while tmux list-windows -t $SESSION -F '#W' | grep -qx ds && ! tmux capture-pane -p -t $SESSION:ds | grep -q 'ds queue done'; do sleep 60; done; "
DIRS=""
for ((s = 0; s < M; s++)); do
  CMD+="$A train run --model transolver_het --train ens --seed $s --name ens_m$s --cpus $CPUS --session $SESSION; "
  DIRS+=" $HOME/aero/runs/ens_m$s"
done
CMD+="$A eval ensemble $DIRS --name ensemble_v1; "
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n ens -c "$REPO" "$CMD echo '[ens queue done]'; exec \$SHELL"
else
  tmux new-session -d -s "$SESSION" -n ens -c "$REPO" "$CMD echo '[ens queue done]'; exec \$SHELL"
fi
echo "ensemble queue ($M members) queued in tmux '$SESSION' (window 'ens')"
