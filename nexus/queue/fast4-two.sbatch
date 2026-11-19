#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:25:00
# Check 3, a first read before the maintenance: the training of opd-e1 with the drafter's last TWO layers run again
# over the branch's tokens, for an hour (opd-e1 after 1,000 answers was within a point of its final accuracy).
# Saved opd-e2q; the full run is diag3-two.
# (The jobs diag1 to diag5 wait for the maintenance window to pass: their time limits reach into it. This one is
# short enough to finish before it.)
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
python nexus/train_every.py --draft opd --data-dir "$HOME/specdec-data/long1" --more-data "$HOME/specdec-data/think1" --results-dir "$RESULTS_DIR" \
  --layers 2 --passes 3 --save opd-e2q --max-hours 0.95 --check-answers 12 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

