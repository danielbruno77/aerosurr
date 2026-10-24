#!/usr/bin/env bash
# RANS rank-scaling and core-placement series on run_1, coarse mesh, 200 iterations each
# (SPEC §2, §5.3). Started after P1 showed that two concurrent 24-rank cases each run
# ~2.7x slower than one alone, i.e. the solver is memory-bandwidth bound.
#
#   pack   = ranks fill whole CCDs (8 cores, one L3 each)
#   spread = ranks spread evenly over all 8 CCDs
#
# Results: ~/aero/rans/bench/S*/aerosurr_run.json; log ~/aero/logs/aero-rans/scaling.log.
#   start:    scripts/aero-rans-scaling.sh
#   monitor:  tail -f ~/aero/logs/aero-rans/scaling.log
set -uo pipefail
SESSION=${SESSION:-aero-rans}
REPO=$(cd "$(dirname "$0")/.." && pwd)
LOG=$HOME/aero/logs/aero-rans/scaling.log
if [ "${1:-}" != "--inside" ]; then
  mkdir -p "$(dirname "$LOG")"
  tmux new-window -t "$SESSION" -n scaling -c "$REPO" "$0 --inside 2>&1 | tee -a $LOG; exec \$SHELL" 2>/dev/null \
    || tmux new-session -d -s "$SESSION" -n scaling -c "$REPO" "$0 --inside 2>&1 | tee -a $LOG; exec \$SHELL"
  echo "scaling series started in tmux '$SESSION' (window 'scaling'); log: $LOG"
  exit 0
fi

A=$REPO/.venv/bin/aerosurr
B=$HOME/aero/rans/bench
IT=${IT:-200}
say() { echo "[$(date +%T)] $*"; }
spread() {  # spread <per_ccd> [offset]: cores offset..offset+per_ccd-1 on each of the 8 CCDs
  local n=$1 off=${2:-0} out=() c k
  for c in 0 8 16 24 32 40 48 56; do for ((k = 0; k < n; k++)); do out+=($((c + off + k))); done; done
  local IFS=,; echo "${out[*]}"
}
one() {     # one <name> <cpus>: set up (ranks = number of CPUs) and run alone
  local n; n=$(tr ',' '\n' <<<"$(expand "$2")" | wc -l)
  [ -d "$B/$1" ] || "$A" rans setup run_1 --preset coarse --ranks "$n" --iterations "$IT" --out "$B/$1" >/dev/null
  "$A" rans run "$B/$1" --cpus "$2" >/dev/null 2>&1 || say "FAILED: $1"
}
expand() { python3 -I -c "import sys
out=[]
for p in sys.argv[1].split(','):
    a,_,b=p.partition('-'); out+=range(int(a),int(b or a)+1)
print(','.join(map(str,out)))" "$1"; }

# Wait for the P1 coarse case (full 1500 iterations, running alone) to finish first.
until python3 -I -c "import json,sys;sys.exit(json.load(open('$B/P1_coarse_np24_A/aerosurr_run.json'))['status']=='running')" 2>/dev/null; do sleep 20; done
say "P1 coarse finished; starting scaling series ($IT iterations each)"

say "S1  8 pack";    one S1_np8_pack 0-7
say "S2  8 spread";  one S2_np8_spread "$(spread 1)"
say "S3 16 pack";    one S3_np16_pack 0-15
say "S4 16 spread";  one S4_np16_spread "$(spread 2)"
say "S5 24 pack";    one S5_np24_pack 0-23
say "S6 24 spread";  one S6_np24_spread "$(spread 3)"
say "S7 30 spread";  one S7_np30_spread 0-3,8-11,16-19,24-27,32-35,40-43,48-50,56-58  # 4/CCD on 6, 3 on 2
say "S8 2x16 spread (interleaved)"
one S8_np16_spreadA "$(spread 2 0)" & one S8_np16_spreadB "$(spread 2 2)" & wait
say "S9 2x24 pack (CCD 0-2 | 4-6)"
one S9_np24_packA 0-23 & one S9_np24_packB 32-55 & wait
say "S10 2x12 spread (interleaved)"
one S10_np12_spreadA "$(spread 1 0),$(spread 1 1 | cut -d, -f1-4)" & \
  one S10_np12_spreadB "$(spread 1 2),$(spread 1 3 | cut -d, -f1-4)" & wait
say "done"
"$A" rans summary "$B"/P1_coarse_np24_A/ "$B"/S*/ | tee "$B/scaling_summary.md"
