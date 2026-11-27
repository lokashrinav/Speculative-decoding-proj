#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
# Ordering against strength (dflash/redraft.py: SecondTree.mix). The four-layer second drafter correcting every node
# (8.98 tokens a round on the 8 answers) beats the pair scores told how sure the first run was (8.67) by 0.31 net: it
# gains 0.655 a round where it is further (29.6% of rounds) and loses 0.344 where it is less far (17.2%). Is what it
# knows the ORDER of the candidates or the STRENGTH of its scores? On the same nodes and candidates: mt = its order
# with the pair scores' values; mc = the pair scores' order with its values; mb = the pair scores alone in this
# harness (must equal ptree63); mx = the second drafter alone without chains (the harness with nothing swapped).
# Traced on DDTree's rounds (stop causes, further or less far than each other), then replayed on their own rounds.
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
    --settings "tree63,ptree63@opd-p2s,stree63p16@opd-e4,stree63p16mx@opd-e4~opd-p2s,stree63p16mt@opd-e4~opd-p2s,stree63p16mc@opd-e4~opd-p2s,stree63p16mb@opd-e4~opd-p2s" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --max-answers 8 --max-hours 1.5 \
    --settings "stree63p16mx@opd-e4~opd-p2s,stree63p16mt@opd-e4~opd-p2s,stree63p16mc@opd-e4~opd-p2s,stree63p16mb@opd-e4~opd-p2s" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

