#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=07:00:00
#DEPENDS: gate-score-0 gate-score-1 gate-score-2 gate-score-3
# Plan items E9 and E14: training the drafter itself while the answer is written, and which
# positions it should learn from. Small adapters on the drafter start at zero for every answer.
# "immediate" trains only on positions up to a round's first rejection (the on-path part of
# what the published Test-Time Speculation method uses); "hindsight" trains on every drafted
# position once the answer has passed it; "both" does the first and then the rest. Each
# validation answer is replayed once per setting, following the adapted drafter's own drafts.
# Starts when the fitted-gate jobs have ended. Four jobs each take a quarter of the answers.
SHARD=2
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/long1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/adapt_drafter.py --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part validation \
  --settings "plain,immediate:1e-4,immediate:3e-4,hindsight:1e-4,hindsight:3e-4,hindsight:1e-3,both:3e-4" \
  --rank 16 --shard "$SHARD" --num-shards 4 --max-hours 6 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
