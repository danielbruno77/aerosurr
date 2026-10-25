#!/usr/bin/env bash
# MF study, seeds 1 and 2 (SPEC §5.3): the 3-seed rerun of `aero-mf-queue.sh`.
# Seed 0 is the existing mf_n<N> / hf_n<N> (same config, verified). Runs <m>_n<N>_s<S>
# use the same nested training cars (read from the seed-0 config.json), the same model and
# train configs and the same 20k-step budget; only --seed (init and data order) changes.
# Each run is evaluated right after training with the final-eval protocol: the 50
# LF-mapped full_test cars, `eval run --tag full_test`.
#
# Two workers, one per GPU, take jobs from a shared claim directory (mkdir is atomic):
#   tmux aero-mfs:gpu0   GPU 0, CPUs on CCD offset 3
#   tmux aero-mfs:gpu1   GPU 1, CPUs on CCD offset 4
#   start:    scripts/aero-mf-seeds.sh
#   monitor:  tail -f ~/aero/logs/aero-mfs/queue_gpu*.log
#             tail -f "$(command ls -t ~/aero/logs/aero-mfs/*_?f_n*_s?.log | head -1)"
set -euo pipefail
SESSION=aero-mfs
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$HOME/aero/logs/$SESSION
if [ "${1:-}" != "--worker" ]; then
  mkdir -p "$LOGS"
  tmux new-session -d -s $SESSION -n gpu0 -c "$REPO" \
    "$0 --worker 0 3,11,19,27,35,43,51,59,67,75,83,91,99,107,115,123 2>&1 | tee -a $LOGS/queue_gpu0.log; exec \$SHELL"
  tmux new-window -t $SESSION -n gpu1 -c "$REPO" \
    "$0 --worker 1 4,12,20,28,36,44,52,60,68,76,84,92,100,108,116,124 2>&1 | tee -a $LOGS/queue_gpu1.log; exec \$SHELL"
  echo "MF seed rerun started in tmux '$SESSION' (windows gpu0, gpu1); logs: $LOGS"
  exit 0
fi

unset AEROSURR_DATA_VARIANT  # the default (full-split) store: LF fields live there
GPU=$2
CPUS=$3
A=$REPO/.venv/bin/aerosurr
PY=$REPO/.venv/bin/python
R=$HOME/aero/runs
CLAIMS=$R/mf_seeds_claims
G="env CUDA_VISIBLE_DEVICES=$GPU $A"
say() { echo "[$(date +%T)] gpu$GPU: $*"; }
mkdir -p "$CLAIMS"
TEST=$($A rans lf-runs --split full_test)

for S in 1 2; do
  for N in 11 25 50; do
    RUNS=$($PY -c "import json; print(','.join(json.load(open('$R/mf_n$N/config.json'))['train']['train_runs']))")
    for m in mf hf; do
      name=${m}_n${N}_s$S
      [ -f "$R/$name/full_test.json" ] && continue
      mkdir "$CLAIMS/$name" 2>/dev/null || continue
      model=$([ $m = mf ] && echo transolver_mf || echo transolver)
      say "$name: training ($model, seed $S, $N cars)"
      if [ -f "$R/$name/summary.json" ] || $G train run --model $model --train mf --train-runs "$RUNS" \
          --seed $S --name $name --cpus "$CPUS" --session $SESSION >/dev/null; then
        say "$name: evaluating on $(tr ',' '\n' <<<"$TEST" | wc -l) test cars"
        $G eval run "$R/$name" --split full_test --runs "$TEST" --tag full_test >/dev/null \
          && say "$name: done" || say "FAILED: eval $name"
      else
        say "FAILED: train $name"
        rmdir "$CLAIMS/$name"
      fi
    done
  done
done
say "no jobs left"
