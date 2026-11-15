#!/bin/bash
# Restarts the job runner on a Nexus submission node. Safe to run at any time: jobs that
# are already running or queued are not touched. It notes what it found (so the cause of
# a stop can be seen afterwards), clears what could block git, brings the folder up to
# date, and starts the runner again.
DEST="$HOME/specdec-runner"
LOG="$HOME/specdec-runner.log"
NOTE="/tmp/specdec-restart-$USER.txt"
[ -d "$DEST/.git" ] || { echo "ERROR: $DEST is missing. Run the first-time setup line instead."; exit 1; }
# Two runners on two machines would get in each other's way: restart where the last one ran.
OLD=$(sed -n 2p "$DEST/nexus/status/heartbeat.txt" 2>/dev/null)
if [ -n "$OLD" ] && [ "$OLD" != "$(hostname)" ] && [ -z "$FORCE" ]; then
  echo "NOT RESTARTED: the runner last ran on $OLD, and this machine is $(hostname)."
  echo "Please log in to $OLD and paste the same line there."
  echo "(Only if $OLD cannot be reached: paste the line again with 'FORCE=1 bash' in place of 'bash' at its end.)"
  exit 1
fi
{
  echo "== restart at $(date -u +%FT%TZ) on $(hostname)"
  echo "== uptime"; uptime
  echo "== tmux sessions"; tmux ls 2>&1
  echo "== processes of the runner"; pgrep -u "$USER" -af "runner.sh|git-remote|git pull|git push|github-specdec" 2>&1 | head -20
  echo "== last lines of the runner's own log"; tail -n 30 "$LOG" 2>&1
  echo "== git folder"; ls -la "$DEST/.git" 2>&1 | grep -E "lock|rebase|MERGE|REBASE" ; git -C "$DEST" status -sb 2>&1 | head -15
  echo "== last commits"; git -C "$DEST" log --format='%ci %s' -3 2>&1
  echo "== jobs"; squeue -u "$USER" -o "%i %j %T %M %l %R" 2>&1
} > "$NOTE" 2>&1

# Stop the old runner and any git command of it that is still hanging.
tmux kill-session -t specdec 2>/dev/null
pkill -u "$USER" -f "bash $DEST/nexus/runner.sh" 2>/dev/null
for pid in $(pgrep -u "$USER" -f "git|github-specdec"); do
  [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$DEST" ] && kill "$pid" 2>/dev/null
done
sleep 2
cd "$DEST" || exit 1
for lock in .git/index.lock .git/shallow.lock; do
  [ -e "$lock" ] && mv "$lock" "$lock.stale.$(date +%s)"
done
git rebase --abort 2>/dev/null
export GIT_SSH_COMMAND="ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ConnectTimeout=20"
if timeout -k 10 120 git pull -q --rebase --autostash origin nexus-jobs; then
  echo "== pull at restart: ok" >> "$NOTE"
else
  git rebase --abort 2>/dev/null
  echo "== pull at restart: FAILED" >> "$NOTE"
fi
mkdir -p nexus/status && cp "$NOTE" nexus/status/restart-report.txt

if command -v tmux >/dev/null; then
  tmux new -d -s specdec "bash $DEST/nexus/runner.sh 2>&1 | tee -a $LOG"
else
  nohup setsid bash "$DEST/nexus/runner.sh" >> "$LOG" 2>&1 &
fi
echo "DONE: runner restarted on $(hostname). You can close this window."
