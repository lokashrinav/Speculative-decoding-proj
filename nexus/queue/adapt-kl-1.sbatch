#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=07:00:00
#DEPENDS: adapt-drafter-0 adapt-drafter-1 adapt-drafter-2 adapt-drafter-3
# Plan items E9 and E14, second part: the same comparison as the adapt-drafter jobs (training the
# drafter during the answer on "immediate" positions against "hindsight" positions), but the
# drafter is now compared with what the target itself predicted at each position (its most
# likely tokens with their probabilities) instead of the committed token alone. That is how the
# published Test-Time Speculation method trains. Starts when the adapt-drafter jobs have ended.
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
python nexus/adapt_drafter.py --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part validation \
  --settings "plain,immediate:1e-4:kl,immediate:3e-4:kl,hindsight:1e-4:kl,hindsight:3e-4:kl" \
  --rank 16 --shard "$SHARD" --num-shards 4 --max-hours 6 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
