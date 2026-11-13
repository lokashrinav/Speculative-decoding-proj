#!/bin/bash
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:05:00
# Frees two GPUs for the second version of the editor (editor2-a, editor2-b).
# It stops two first-version runs whose answer is already clear, and whose saved
# editors are scored again by editor2-a:
#   8395651 editor-keepfirst   8395652 editor-main
# and the waiting baseline 8395655 retrained-drafter, which is queued again as
# retrained-drafter-b so that it runs after the new editors instead of before.
# The other two first-version runs (editor-tokens, editor-teacher) keep going.
{
  date -u
  scancel 8395651 8395652 8395655
  echo "scancel exit $?"
  sleep 20
  squeue -u "$USER" -o "%i %j %T %M %R"
} > "$RESULTS_DIR/report.txt" 2>&1
