#!/usr/bin/env bash
# Multi-fidelity study (SPEC §5.3) on GPU 0, tmux `aero-train0`, window `mf`.
#
# For N = 11, 25, 50: wait until N training cars have mapped LF fields (mapping finished
# RANS cases as they come), then train the MF model and the HF-only model on the same N
# cars with the same 20k-step budget. When the RANS batch has finished, every model is
# evaluated on the same final set of LF-mapped test cars, the LF-alone baseline is
# scored, and docs/results.md is regenerated.
#
#   start:    scripts/aero-mf-queue.sh        (waits for the GeoTransolver window first)
#   monitor:  tail -f ~/aero/logs/aero-train0/latest.log
set -euo pipefail
SESSION=aero-train0
REPO=$(cd "$(dirname "$0")/.." && pwd)
GPU_CPUS=${GPU_CPUS:-3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,123}
MAP_CPUS=${MAP_CPUS:-5,13,21,29,69,77,85,93}
if [ "${1:-}" != "--inside" ]; then
  tmux new-window -t "$SESSION" -n mf -c "$REPO" "$0 --inside 2>&1 | tee -a $HOME/aero/logs/$SESSION/mf_queue.log; exec \$SHELL"
  echo "MF queue started in tmux '$SESSION' (window 'mf'); log: ~/aero/logs/$SESSION/mf_queue.log"
  exit 0
fi

A=$REPO/.venv/bin/aerosurr
G="env CUDA_VISIBLE_DEVICES=0 $A"
R=$HOME/aero/runs
say() { echo "[$(date +%T)] $*"; }
map() { $A rans map --cpus "$MAP_CPUS" >/dev/null 2>&1 || say "map failed (will retry)"; }

say "waiting for the GeoTransolver run on GPU 0"
until tmux capture-pane -p -t $SESSION:geo 2>/dev/null | grep -q 'geo done'; do sleep 120; done

for N in 11 25 50; do
  say "N=$N: waiting for $N training cars with LF fields"
  until RUNS=$($A rans lf-runs --split full_train --n $N 2>/dev/null); do map; sleep 600; done
  say "N=$N: $RUNS"
  [ -f $R/mf_n$N/summary.json ] || $G train run --model transolver_mf --train mf --train-runs "$RUNS" \
    --name mf_n$N --cpus "$GPU_CPUS" --session $SESSION || say "FAILED: mf_n$N"
  [ -f $R/hf_n$N/summary.json ] || $G train run --model transolver --train mf --train-runs "$RUNS" \
    --name hf_n$N --cpus "$GPU_CPUS" --session $SESSION || say "FAILED: hf_n$N"
done

say "waiting for the RANS batch to finish"
until grep -q "batch finished" "$(readlink -f $HOME/aero/logs/aero-rans/latest.log)"; do map; sleep 900; done
map
$A eval lf --split full_test > $HOME/aero/logs/$SESSION/eval_lf.json || say "FAILED: eval lf"
TEST=$($A rans lf-runs --split full_test)
say "final evaluation on $(tr ',' '\n' <<<"$TEST" | wc -l) LF-mapped test cars"
for N in 11 25 50; do
  for m in mf hf; do
    [ -f $R/${m}_n$N/best.pt ] && $G eval run $R/${m}_n$N --split full_test --runs "$TEST" --tag full_test \
      >/dev/null || say "FAILED: eval ${m}_n$N"
  done
done
(cd "$REPO" && $A report) || say "FAILED: report"
say "MF queue done"
