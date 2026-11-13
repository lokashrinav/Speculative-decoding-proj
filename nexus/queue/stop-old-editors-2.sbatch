#!/bin/bash
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=1
#SBATCH --mem=1gb
#SBATCH --time=00:45:00
# Frees GPUs for the headroom check and the next editor version.
#   8395653 editor-teacher (first version, answer already clear: about 5% worse)
#   8396948 retrained-drafter-b (still waiting; queued again as retrained-drafter-c)
# and, once its training has finished, 8396947 editor2-b-inputs: its final numbers
# are kept, only its speed test is skipped because editor2-a already runs that test.
{
  date -u
  scancel 8395653 8396948
  echo "scancel exit $?"
  for i in $(seq 1 150); do
    [ -f "$RESULTS_DIR/../editor2-b-inputs/exit.txt" ] && break
    sleep 15
  done
  date -u
  if [ -f "$RESULTS_DIR/../editor2-b-inputs/exit.txt" ]; then scancel 8396947; echo "scancel editor2-b exit $?"; fi
  sleep 15
  squeue -u "$USER" -o "%i %j %T %M %R"
} > "$RESULTS_DIR/report.txt" 2>&1
