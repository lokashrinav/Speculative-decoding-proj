#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
#SBATCH --dependency=afterany:8419424:8419425:8419426:8419427
# Plan item E4, pulled forward to complete the evidence for Gate 1. Starts when the four replay
# jobs have ended. No language model runs here: small gates are fitted on the recorded rounds
# of 60% of the answers and scored on another 20% (test), with the margin chosen on the rest.
# Four sets of inputs are compared (drafter only as a control, token memory, hidden-state
# memory, both), on the drafter's own draft and on the left-to-right editor's draft.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/long1"
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/fit_gate.py --rounds-dir "$DATA/rounds1" --results-dir "$RESULTS_DIR" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
ls "$DATA/rounds1" | wc -l > "$RESULTS_DIR/disk.txt" 2>&1
du -sh "$DATA/rounds1" >> "$RESULTS_DIR/disk.txt" 2>&1
