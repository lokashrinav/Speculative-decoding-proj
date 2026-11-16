#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=10:00:00
#DEPENDS: adapt-canvas-0 adapt-canvas-1 adapt-canvas-2 adapt-canvas-3
# Plan item E9, the answers for the head-to-head on long answers: decodes the math benchmarks the
# published method reports on (AIME 2024, AIME 2025, hardest MATH-500 problems) with up to 32,768
# new tokens, twice our earlier limit, and saves the answers. These answers are used for final
# comparisons only. Starts when the adapt-canvas jobs have ended. Four jobs, a quarter each.
SHARD=3
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
DATA="$HOME/specdec-data/bench1"
python nexus/probe_memory.py --prompts-file "$DATA/prompts.json" --out-dir "$DATA" --results-dir "$RESULTS_DIR" \
  --max-new-tokens 32768 --shard "$SHARD" --num-shards 4 --max-hours 9 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
ls -la "$DATA" > "$RESULTS_DIR/disk.txt" 2>&1
