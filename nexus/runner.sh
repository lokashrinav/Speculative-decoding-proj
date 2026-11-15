#!/bin/bash
# Nexus job runner. Runs in tmux on a Nexus submission node.
# Once a minute it pulls the nexus-jobs branch, submits any new nexus/queue/*.sbatch
# file with sbatch, and pushes job status, log tails and small result files back.
# Stop it by pushing a file named nexus/STOP, or with: tmux kill-session -t specdec
#
# A job file may name jobs that must have ended before it starts, on a line such as
#   #DEPENDS: collect-0 collect-1
# The runner then waits until those are submitted and passes their job numbers to sbatch.
# When this script itself changes on the branch, the runner starts itself again with the new version.
BRANCH=nexus-jobs
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
git config user.name "nexus-runner"
git config user.email "nexus-runner@users.noreply.github.com"
# A connection to GitHub that goes quiet must not block the loop: ssh gives up after a
# minute without an answer, and no pull or push may take longer than five minutes.
export GIT_SSH_COMMAND="ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o ConnectTimeout=20"
NET="timeout -k 10 300"
LOGDIR="$HOME/specdec-slurm"
mkdir -p nexus/status nexus/logs nexus/results "$LOGDIR"
SELF="$(readlink -f "$0")"
SELF_SUM=$(md5sum "$SELF" | cut -d" " -f1)
tick=0
while true; do
  # A lock left behind by an interrupted git command would block every later one: set it aside.
  for lock in .git/index.lock .git/shallow.lock; do
    if [ -e "$lock" ] && [ -n "$(find "$lock" -mmin +5 2>/dev/null)" ]; then mv "$lock" "$lock.stale.$(date +%s)"; fi
  done
  $NET git pull -q --rebase --autostash origin "$BRANCH" || { git rebase --abort 2>/dev/null; }
  if [ -f nexus/STOP ]; then echo "STOP file found, exiting"; exit 0; fi
  if [ "$(md5sum "$SELF" | cut -d" " -f1)" != "$SELF_SUM" ]; then
    echo "$(date -u +%FT%TZ) the runner script changed: starting the new version"
    exec bash "$SELF"
  fi
  submitted=0
  for f in nexus/queue/*.sbatch; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .sbatch)
    [ -e "nexus/status/$name.jobid" ] && continue
    after=""
    waiting=0
    for needed in $(sed -n 's/^#DEPENDS:[[:space:]]*//p' "$f" | head -1); do
      id=$(head -1 "nexus/status/$needed.jobid" 2>/dev/null)
      id=${id%%;*}
      case "$id" in ""|*[!0-9]*) waiting=1 ;; *) after="$after:$id" ;; esac
    done
    [ "$waiting" = 1 ] && continue    # a job it depends on has no job number yet: look again next minute
    [ -n "$after" ] && after="--dependency=afterany$after"
    mkdir -p "nexus/results/$name"
    sbatch --parsable $after --job-name="$name" --output="$LOGDIR/$name.log" \
      --export=ALL,JOB_NAME="$name",RESULTS_DIR="$PWD/nexus/results/$name" "$f" \
      > "nexus/status/$name.jobid" 2>&1
    submitted=1
  done
  # Push status every 5 minutes, or right away after a new submission.
  if [ "$submitted" = 1 ] || [ $((tick % 5)) = 0 ]; then
    squeue -u "$USER" -o "%i %j %T %M %l %R" > nexus/status/squeue.txt 2>&1
    sacct -u "$USER" -S now-3days -X -n -o JobID,JobName%30,State,Elapsed,ExitCode > nexus/status/sacct.txt 2>&1
    date -u +%FT%TZ > nexus/status/heartbeat.txt
    hostname >> nexus/status/heartbeat.txt
    for l in "$LOGDIR"/*.log; do
      [ -e "$l" ] && tail -c 20000 "$l" > "nexus/logs/$(basename "$l")"
    done
    git add -A nexus/status nexus/logs
    # Result files are added in small groups: a file that a job replaces at that moment must not stop the others.
    find nexus/results -type f -size -1000k ! -name "*.tmp" -print0 | xargs -0 -r -n 20 git add 2>/dev/null
    git diff --cached --quiet || git commit -qm "runner: status update"
    $NET git push -q origin "HEAD:$BRANCH" || echo "$(date -u +%FT%TZ) push failed"
  fi
  tick=$((tick + 1))
  sleep "${RUNNER_SLEEP:-60}"
done
