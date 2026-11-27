#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:30:00
# Staleness inputs (the review of November 27): the correction's strength gated by four numbers of the branch, places since
# its first fork, since its last fork, forks so far, and the summed first-run log-probability gap along it, which a flag
# about the token before cannot say and which DForkTree's distance split (at most 0.4%) only partly covers. Trained as
# opd-p2s (the three numbers too; 2 passes, 2 hours), then replayed on the 8 answers and traced on DDTree's rounds for
# the stop causes by distance from the fork. Controls: opd-p2s 8.667 / opd-p2s-s1 8.695 tokens a round.
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
( python nexus/train_pairs.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --more-data "$HOME/specdec-data/think1" --sure --branch --save opd-p2s-b --passes 2 --max-hours 2.0 --check-answers 12 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/replay" "$RESULTS_DIR/trace" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay" --part validation --max-answers 8 --max-hours 0.6 \
    --settings "ptree63@opd-p2s-b" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/trace" --part validation --max-answers 8 --max-hours 0.6 \
    --settings "tree63,ptree63@opd-p2s,ptree63@opd-p2s-b" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

