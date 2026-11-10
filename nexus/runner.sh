#!/bin/bash
# Nexus job runner. Runs in tmux on a Nexus submission node.
# Once a minute it pulls the nexus-jobs branch, submits any new nexus/queue/*.sbatch
# file with sbatch, and pushes job status, log tails and small result files back.
# Stop it by pushing a file named nexus/STOP, or with: tmux kill-session -t specdec
BRANCH=nexus-jobs
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
git config user.name "nexus-runner"
git config user.email "nexus-runner@users.noreply.github.com"
LOGDIR="$HOME/specdec-slurm"
mkdir -p nexus/status nexus/logs nexus/results "$LOGDIR"
tick=0
while true; do
  git pull -q --rebase --autostash origin "$BRANCH" || { git rebase --abort 2>/dev/null; }
  if [ -f nexus/STOP ]; then echo "STOP file found, exiting"; exit 0; fi
  submitted=0
  for f in nexus/queue/*.sbatch; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .sbatch)
    [ -e "nexus/status/$name.jobid" ] && continue
    mkdir -p "nexus/results/$name"
    sbatch --parsable --job-name="$name" --output="$LOGDIR/$name.log" \
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
    find nexus/results -type f -size -1000k -print0 | xargs -0 -r git add
    git diff --cached --quiet || git commit -qm "runner: status update"
    git push -q origin "HEAD:$BRANCH"
  fi
  tick=$((tick + 1))
  sleep 60
done
