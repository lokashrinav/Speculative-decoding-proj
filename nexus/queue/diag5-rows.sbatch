#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
# If the second drafter's tree stops because the tokens its pass brings in have no row of their own (diag1-trace), one
# way to give them one WITHOUT a second pass is rows for a larger first tree: fF = rows for the F most probable nodes
# by DDTree's rule, the tree that is checked stays 63 nodes. A token the pass moves up is then often a node that
# already has a row. Trace on DDTree's rounds (folder trace), then replay, each tree on its own rounds: F = 127 and
# 255, with 2 places asked a row and with 1 (t1: half the places that run through the layer).
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
( mkdir -p "$RESULTS_DIR/trace" \
  && python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/trace" --part validation --max-answers 8 --max-hours 1.2 \
    --settings "tree63,stree63@opd-e1,stree63f127@opd-e1,stree63f255@opd-e1,stree63f255t1@opd-e1,stree63f127t1@opd-e1" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --max-answers 8 --max-hours 1.5 \
    --settings "stree63f127@opd-e1,stree63f255@opd-e1,stree63f255t1@opd-e1,stree63f127t1@opd-e1" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

