#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# DForkTree's scores in our replay, and the second run on top of them (nexus/redraft_long.py). DForkTree (arXiv
# 2611.12356) corrects DDTree's scores with counts of which token the target writes after which two tokens: no
# training, no pass added. It is the method the second run has to add to. Here: DDTree with 63 nodes, the tree with
# counts (dflash/redraft.py: CountTree; the paper's weights and four others), and the tree in which the second run
# writes after every node with a tree with counts as its first tree (cbtree...: one trained layer, the whole drafter
# with predicted states, and with real states as the limit). The counts are taken from long1's training answers and
# the new text (~/specdec-data/think1, as much as is written when the job starts). 8 validation answers, Draft-OPD.
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
  --max-answers 8 --max-hours 2.5 --count-data "$HOME/specdec-data/think1" \
  --settings "plain,tree63,ctree63,ctree63l20_40,ctree63l15_30,ctree63l30_30,ctree63l45_90,btree63f40t2n@~opd-r1,cbtree63f40t2n@~opd-r1,cbtree63f40t2b8p@opd-m0,btree63f40t2b8r,cbtree63f40t2b8r" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
