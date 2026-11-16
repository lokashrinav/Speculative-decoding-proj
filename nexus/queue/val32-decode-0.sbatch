#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
# A long validation set: the validation prompts whose answers ran into the old limit of 16,384 new
# tokens are decoded again with a limit of 32,768, with the plain public drafter. The benchmark answers
# stay for final comparisons; choices about very long answers (when to start learning, how often)
# are made on these. Four jobs, a quarter of the prompts each.
SHARD=0
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
DATA="$HOME/specdec-data/long2"
mkdir -p "$DATA"
python nexus/probe_memory.py --again-from "$HOME/specdec-data/long1" --again-part validation --again-min-tokens 16384 \
  --out-dir "$DATA" --results-dir "$RESULTS_DIR" \
  --max-new-tokens 32768 --shard "$SHARD" --num-shards 4 --max-hours 5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
ls -la "$DATA" > "$RESULTS_DIR/disk.txt" 2>&1
