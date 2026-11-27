#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
# Did the four-branch, three-token restriction hide a larger room? The oracle branches (C13) with 4 or 8 breaks and 3
# or 6 tokens of perfect continuation, inside the 64-row budget (the first tree shrunk to 63 - B(T+1) nodes: f47, f31,
# f35, f7) and with the rows added (b4t3 ... b8t6). DDTree and the 4 x 3 case of C13 on the same rounds, to reproduce.
# Judge against what a real continuation fills: B2 measured a second run inside the block at 0.64 tokens after the
# right token where the answer's own tokens give 2.03, about a third of this kind of room.
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
mkdir -p "$RESULTS_DIR/trace" && python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/trace" --part validation --max-answers 8 --max-hours 0.8 \
  --settings "tree63,otree63f47b4t3,otree63f31b8t3,otree63f35b4t6,otree63f7b8t6,otree63b4t3,otree63b8t3,otree63b4t6,otree63b8t6" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

