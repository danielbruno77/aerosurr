#!/usr/bin/env bash
# Data-efficiency study (SPEC §5.2) on GPU 1, tmux `aero-ens1`, window `ds`: HF-only
# Transolver on the nested training sets, then a full-mesh evaluation on full_test.
#   start:    scripts/aero-ds-queue.sh
#   monitor:  tail -f ~/aero/logs/aero-ens1/latest.log
set -euo pipefail
SESSION=aero-ens1
REPO=$(cd "$(dirname "$0")/.." && pwd)
CPUS=${CPUS:-4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124}
A="CUDA_VISIBLE_DEVICES=1 $REPO/.venv/bin/aerosurr"
CMD=""
for t in ds_super_scarce ds_scarce ds_medium ds_full; do
  CMD+="$A train run --model transolver --train $t --name transolver_$t --cpus $CPUS --session $SESSION && "
  CMD+="$A eval run $HOME/aero/runs/transolver_$t --split full_test; "
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n ds -c "$REPO" "$CMD echo '[ds queue done]'; exec \$SHELL"
else
  tmux new-session -d -s "$SESSION" -n ds -c "$REPO" "$CMD echo '[ds queue done]'; exec \$SHELL"
fi
echo "data-efficiency queue started in tmux '$SESSION' (window 'ds')"
