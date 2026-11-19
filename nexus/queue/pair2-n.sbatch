#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:40:00
# The second drafter's knowledge without a pass (dflash/pair_scores.py, nexus/train_pairs.py): trained scores of pairs
# of neighbouring tokens, read from the first run's own states, searched at every depth of the tree. The second
# drafter's pass costs 8 ms of a round for 3% more tokens; these scores cost a few small products. One job: train
# for 0.6 hours (about 2,000 answers; the second drafter needed about 1,000), then tokens per round in replay
# (folder replay) and real speed on 3 answers (folder speed), all before the maintenance at 16:45 UTC.
# This one without counts (ptree63): what the scores are worth alone, against the tree with counts.
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
( python nexus/train_pairs.py --draft opd --data-dir "$HOME/specdec-data/long1" --more-data "$HOME/specdec-data/think1" --results-dir "$RESULTS_DIR" \
    --save opd-p1 --max-hours 0.6 --check-answers 12 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/replay" "$RESULTS_DIR/speed" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/replay" --part validation --max-answers 8 --max-hours 0.4 \
    --count-data "$HOME/specdec-data/think1" --settings "plain,tree63,ctree63,ptree63@opd-p1,stree63@opd-e1" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/speed" --part validation --rank 16 --min-tokens 2000 \
    --max-answers 3 --max-hours 0.3 --count-data "$HOME/specdec-data/think1" --settings "plain,plain+tree63,plain+ctree63,plain+ptree63@opd-p1" \
    > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
