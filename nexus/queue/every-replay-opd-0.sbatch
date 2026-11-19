#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# The tree whose edges the second drafter scores (dflash/redraft.py: SecondTree), in replay on the 8 validation
# answers the other trees were measured on (Draft-OPD). The second drafter is one layer trained at every place
# with the answer's own tokens before it (nexus/train_every.py): opd-e1c on top of the counts, opd-e1 without.
# Next to DDTree (tree63), the tree with counts (ctree63: what the second drafter has to add to), larger trees,
# and the tree whose second run reads the target's REAL states (the limit measured before: +17.5% over DDTree).
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation \
  --max-answers 8 --max-hours 2.7 --count-data "$HOME/specdec-data/think1" \
  --settings "plain,tree63,ctree63,cstree63@opd-e1c,stree63@opd-e1,stree63@opd-e1c,cstree63@opd-e1,cstree63f40@opd-e1c,cstree63t1@opd-e1c,ctree95,cstree95@opd-e1c,cbtree63f40t2b8r" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
