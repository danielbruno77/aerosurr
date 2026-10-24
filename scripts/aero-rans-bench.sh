#!/usr/bin/env bash
# RANS budget and core-layout benchmark on run_1 (SPEC §2, §5.3). Runs in tmux `aero-rans`.
#
#   P0   side-load probe alone (idle machine): VTP preprocessing + GPU training steps
#   P1   2 x 24 ranks (CCDs 0-2 | 4-6): coarse + medium, full 1500 iterations
#   P2   2 x 30 ranks (cores 0-29 | 32-61): coarse + medium, 300 iterations
#   P3   1 x 24 ranks alone: coarse, 300 iterations
#   P4a  2 x 24 ranks: coarse + coarse, 300 iterations
#   P4b  as P4a, with the side-load probe on the free CCDs 3 + 7 and GPU 0
#
# Results: ~/aero/rans/bench/<phase>_*/aerosurr_run.json, probe JSON in ~/aero/rans/bench/,
# logs in ~/aero/logs/aero-rans/. Summarise with `aerosurr rans summary ~/aero/rans/bench/*/`.
#
#   start:    scripts/aero-rans-bench.sh            (or SESSION=... to rename)
#   monitor:  tmux attach -t aero-rans  |  tail -f ~/aero/logs/aero-rans/bench.log
set -uo pipefail
SESSION=${SESSION:-aero-rans}
REPO=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" != "--inside" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n bench -c "$REPO" "$0 --inside 2>&1 | tee -a $HOME/aero/logs/aero-rans/bench.log; exec \$SHELL"
  else
    tmux new-session -d -s "$SESSION" -n bench -c "$REPO" "$0 --inside 2>&1 | tee -a $HOME/aero/logs/aero-rans/bench.log; exec \$SHELL"
  fi
  echo "benchmark started in tmux '$SESSION' (window 'bench'); log: ~/aero/logs/aero-rans/bench.log"
  exit 0
fi

A=$REPO/.venv/bin/aerosurr
B=$HOME/aero/rans/bench
FREE=24-31,56-63,88-95,120-127          # CCDs 3 and 7 with their SMT siblings
mkdir -p "$B" "$HOME/aero/logs/aero-rans"
say() { echo "[$(date +%T)] $*"; }

case_() {  # case_ <name> <preset> <ranks> <iterations>
  [ -d "$B/$1" ] || "$A" rans setup run_1 --preset "$2" --ranks "$3" --iterations "$4" --out "$B/$1" >/dev/null
}
run_() {   # run_ <name> <cpus>
  "$A" rans run "$B/$1" --cpus "$2" >/dev/null 2>&1 || say "FAILED: $1 (see $B/$1/log.*)"
}
probe_() { # probe_ <tag> <seconds>
  CUDA_VISIBLE_DEVICES=0 "$A" env probe --cpus "$FREE" --seconds "$2" --gpu 0 --json "$B/probe_$1.json" \
    > "$B/probe_$1.log" 2>&1 || say "FAILED: probe $1"
}

say "P0 probe alone";        probe_ P0_idle 180
say "P1 2x24 coarse+medium"; case_ P1_coarse_np24_A coarse 24 1500; case_ P1_medium_np24_B medium 24 1500
run_ P1_coarse_np24_A 0-23 & run_ P1_medium_np24_B 32-55 & wait
say "P2 2x30 coarse+medium"; case_ P2_coarse_np30_A coarse 30 300; case_ P2_medium_np30_B medium 30 300
run_ P2_coarse_np30_A 0-29 & run_ P2_medium_np30_B 32-61 & wait
say "P3 1x24 coarse alone";  case_ P3_coarse_np24_alone coarse 24 300
run_ P3_coarse_np24_alone 0-23
say "P4a 2x24 coarse+coarse"; case_ P4a_coarse_np24_A coarse 24 300; case_ P4a_coarse_np24_B coarse 24 300
run_ P4a_coarse_np24_A 0-23 & run_ P4a_coarse_np24_B 32-55 & wait
say "P4b 2x24 coarse+coarse + probe"; case_ P4b_coarse_np24_A coarse 24 300; case_ P4b_coarse_np24_B coarse 24 300
run_ P4b_coarse_np24_A 0-23 & run_ P4b_coarse_np24_B 32-55 & sleep 80; probe_ P4b_loaded 240 & wait
say "done"; "$A" rans summary "$B"/P*/ | tee "$B/summary.md"
