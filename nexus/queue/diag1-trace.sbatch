#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
# Where the answer's own path leaves each tree, and why (nexus/trace_trees.py): DDTree, the tree with counts, the
# pair scores and the second drafter's tree, all built in the SAME rounds (those of DDTree). The second drafter is
# right more often than the counts after a fork (49% against 39%) and its tree still holds fewer tokens (8.21 against
# 8.31): cost cannot explain that. The suspicion: its one pass corrects one step past the first tree, the counts
# correct every depth. p2, p16: the second drafter with more passes, up to a row for every node of its tree (a
# measurement of what its scores are worth, time ignored). No training; 8 validation answers.
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
python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --count-data "$HOME/specdec-data/think1" --results-dir "$RESULTS_DIR" \
  --part validation --max-answers 8 --max-hours 2.6 \
  --settings "tree63,ctree63,ctree63m32,ptree63@opd-p1,cptree63@opd-p1c,stree63@opd-e1,stree63p2@opd-e1,stree63p16@opd-e1,cstree63@opd-e1c,cstree63p16@opd-e1c" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
