#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
# The D2SD second drafter made here (drafter-opd-d2sd; training done in loop6-d2sd, 3.68 h, validation: right after the
# restart 79.3%, 80.9% after one revealed token) died at its probe: a saved drafter could not be loaded outside the benchmark
# script (fixed). This is the probe and the replay of loop6 again: the restart at the four likely breaks with the trained
# second drafter given nothing, predicted states and the target's real states (the hidden-states question for a trained
# pass), and the replay of plain, the same drafter again at four breaks, and the trained second drafter at four breaks.
ENV="$HOME/specdec-env"
CODE="/tmp/specdec-code-$USER/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
( mkdir -p "$RESULTS_DIR/probe" "$RESULTS_DIR/replay" \
  && python nexus/restart_probe.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/probe" --part validation --max-answers 30 --max-hours 1.2 \
    --predictor opd-m0 --second-draft drafter-opd-d2sd > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay" --part validation --max-answers 8 --max-hours 1.5 \
    --settings "plain,branches4,branches4@drafter-opd-d2sd" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

