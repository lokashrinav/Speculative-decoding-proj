#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
# The trace (where the answer's path leaves each tree, all on DDTree's rounds) of what the five jobs after the
# maintenance trained: the pair scores trained two hours as they were (opd-p2: 42.4% right after a fork) and told how
# sure the first run was (opd-p2s: 47.5%, and the chain of first choices no longer shortened); the second drafter
# shown only the last token (opd-e1m1: 46.2%); two layers (opd-e2: 55.6%); four layers (opd-e4: 57.3%).
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
python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --max-answers 8 --max-hours 1.2 \
  --settings "tree63,ptree63@opd-p1,ptree63@opd-p2,ptree63@opd-p2s,stree63@opd-e1m1,stree63@opd-e2,stree63@opd-e4,stree63p16@opd-e4" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

