#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
# What branching inside the drafter would cost (nexus/branch_cost.py), before anything is trained: the first run; the
# first run with a readout and the choice of four breaks after layer 2 or 3; the method (k layers on the block, readout,
# four branches, the rest of the layers on five blocks); the first run with a whole second pass on the branches; and the
# target's check with 64, 76, 80 and 128 rows. At 1,024, 4,096 and 8,192 tokens of text, medians of 15 repeats on 3
# answers, with the drafter reading a 2,048-token window (and 512, in the second folder, to show what the window costs).
# The second of the two checks before the branching drafter is built; the first is probe1-readout.
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
( python nexus/branch_cost.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --contexts 1024,4096,8192 --splits 2,3 --window 2048 --repeats 15 --answers 3 \
    > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/window512" \
  && python nexus/branch_cost.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/window512" --contexts 1024,4096,8192 --splits 2,3 --window 512 --repeats 15 --answers 3 \
    > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

