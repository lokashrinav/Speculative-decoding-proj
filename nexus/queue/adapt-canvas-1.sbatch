#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=10:00:00
#DEPENDS: adapt-drafter-0 adapt-drafter-1 adapt-drafter-2 adapt-drafter-3
# Plan item E9: the published method's training signal as the baseline. After every round the
# drafter is trained on what the target predicted at every position of that round's draft
# ("canvas", as Test-Time Speculation does), with the same adapters and optimiser as our own
# variants in the adapt-drafter and adapt-kl jobs. "canvas+hindsight" then trains the positions
# after the first rejection once more with their exact tokens, when the answer has passed them.
# The target has to check every draft again here, so these jobs run at about decoding speed.
# The settings are read when the job starts, from nexus/settings/adapt-canvas.txt.
SHARD=1
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
SETTINGS="$(head -n 1 nexus/settings/adapt-canvas.txt)"
[ -n "$SETTINGS" ] || exit 1
echo "$SETTINGS" > "$RESULTS_DIR/settings.txt"
python nexus/adapt_drafter.py --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part validation \
  --settings "$SETTINGS" \
  --rank 16 --shard "$SHARD" --num-shards 4 --max-hours 9 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
