#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:25:00
# The trace of diag1-trace (nexus/trace_trees.py: where the answer's path leaves each tree and why, all trees on
# DDTree's rounds), with the one-pass trees that have rows for a larger first tree (f255) next to it.
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
python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --count-data "$HOME/specdec-data/think1" --results-dir "$RESULTS_DIR" --part validation --max-answers 8 --max-hours 1.0 \
  --settings "tree63,ctree63,ctree63m32,ptree63@opd-p1,cptree63@opd-p1c,stree63@opd-e1,stree63p2@opd-e1,stree63p16@opd-e1,stree63f255@opd-e1,cstree63@opd-e1c,cstree63p16@opd-e1c" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

