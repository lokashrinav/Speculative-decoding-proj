#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#DEPENDS: bench-decode-0 bench-decode-1 bench-decode-2 bench-decode-3
# Plan item E9, the head-to-head on the published method's own math benchmarks (AIME 2024, AIME 2025,
# hardest MATH-500 problems; answers up to 32,768 tokens, twice as long as anything the frozen
# drafters learned from). Plain drafter, frozen drafters taught ahead of time on our training
# answers, and training during the answer with the published signal and with the simple one.
# Starts when the benchmark answers have been decoded. Its settings are read when it starts.
SHARD=3
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/bench1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
SETTINGS="$(head -n 1 nexus/settings/bench-eval.txt)"
[ -n "$SETTINGS" ] || exit 1
echo "$SETTINGS" > "$RESULTS_DIR/settings.txt"
python nexus/adapt_drafter.py --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part all \
  --settings "$SETTINGS" \
  --rank 16 --shard "$SHARD" --num-shards 4 --max-hours 11 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
