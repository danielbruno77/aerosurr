#!/usr/bin/env bash
# Start the DrivAerML bulk download (T0 + T2, SPEC §4.3) in tmux session `aero-dl`.
# Extra arguments go to `aerosurr data download`. Re-running resumes: verified files are
# skipped and `.part` files continue with HTTP Range requests.
#
#   monitor:  tmux attach -t aero-dl          (detach: Ctrl-b d)
#             tail -f ~/aero/logs/aero-dl/latest.log
#             .venv/bin/aerosurr data status
#   stop:     tmux send-keys -t aero-dl C-c   (safe; resume by running this script again)
set -euo pipefail
SESSION=${SESSION:-aero-dl}
REPO=$(cd "$(dirname "$0")/.." && pwd)
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already running: tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "$REPO/.venv/bin/aerosurr data download --tier t0 --tier t2 $*; \
   echo \"[aerosurr data download exited with status \$?]\"; exec \$SHELL"
echo "started tmux session '$SESSION'; log: ~/aero/logs/$SESSION/latest.log"
