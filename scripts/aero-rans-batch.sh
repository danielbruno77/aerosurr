#!/usr/bin/env bash
# The LF RANS batch (SPEC §5.3) in tmux `aero-rans`, window `batch`. Resumable: finished
# cases are skipped, half-finished ones restart. Arguments go to `aerosurr rans batch`;
# the defaults are the layout chosen in docs/rans_benchmark.md.
#
#   start:    scripts/aero-rans-batch.sh [--slots "A=...;B=..."] [--iterations N]
#   monitor:  tail -f ~/aero/logs/aero-rans/latest.log
#             .venv/bin/aerosurr rans summary ~/aero/rans/batch/*/ | less
#   stop:     tmux send-keys -t aero-rans:batch C-c   (re-run this script to resume)
set -euo pipefail
SESSION=${SESSION:-aero-rans}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-$HOME/aero/rans/subset_v1.txt}
SLOTS=${SLOTS:?set SLOTS, e.g. SLOTS="A=0-23;B=32-55" (see docs/rans_benchmark.md)}
CMD="$REPO/.venv/bin/aerosurr rans batch $RUNS --preset coarse --slots '$SLOTS' $*"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n batch -c "$REPO" "$CMD; echo \"[batch exited \$?]\"; exec \$SHELL"
else
  tmux new-session -d -s "$SESSION" -n batch -c "$REPO" "$CMD; echo \"[batch exited \$?]\"; exec \$SHELL"
fi
echo "RANS batch started in tmux '$SESSION' (window 'batch'): $CMD"
