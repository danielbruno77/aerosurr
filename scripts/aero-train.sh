#!/usr/bin/env bash
# Train a surface model in tmux: GPU 0 -> session aero-train0 (main model), GPU 1 ->
# aero-ens1 (ensembles, ablations). SPEC §2: CUDA_VISIBLE_DEVICES is always explicit.
#
#   scripts/aero-train.sh <gpu> <cpus> <model> <train> <name>
#   e.g. scripts/aero-train.sh 0 28-31,92-95 transolver main transolver_main_v1
#   monitor:  tail -f ~/aero/logs/aero-train0/latest.log ; nvidia-smi dmon -s pucm
set -euo pipefail
GPU=$1 CPUS=$2 MODEL=$3 TRAIN=$4 NAME=$5
SESSION=$([ "$GPU" = 0 ] && echo aero-train0 || echo aero-ens1)
REPO=$(cd "$(dirname "$0")/.." && pwd)
CMD="CUDA_VISIBLE_DEVICES=$GPU $REPO/.venv/bin/aerosurr train run --model $MODEL --train $TRAIN --name $NAME --cpus $CPUS --session $SESSION"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n "$NAME" -c "$REPO" "$CMD; echo \"[train exited \$?]\"; exec \$SHELL"
else
  tmux new-session -d -s "$SESSION" -n "$NAME" -c "$REPO" "$CMD; echo \"[train exited \$?]\"; exec \$SHELL"
fi
echo "training '$NAME' started in tmux '$SESSION': $CMD"
