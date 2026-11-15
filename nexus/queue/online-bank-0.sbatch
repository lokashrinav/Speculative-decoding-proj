#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
#SBATCH --dependency=afterany:8419424:8419425:8419426:8419427
# Plan item E14, first look: an editor that is trained while the answer is being written.
# The drafter stays frozen. A small low-rank correction starts at zero for every answer and
# is trained during the answer on the drafter's own drafts and the tokens the target
# committed. Sixteen settings (learning rate, steps per round, size, and whether labels for
# positions after a round's first rejection are used) are trained side by side while the
# saved validation answers are replayed, and each is scored against the text the target
# wrote. Starts when the replay jobs have ended. Four jobs each take a quarter of the answers.
SHARD=0
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/long1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/online_bank.py --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part validation \
  --shard "$SHARD" --num-shards 4 --max-hours 4.5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
