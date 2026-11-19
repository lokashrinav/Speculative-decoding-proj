#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# More text from the target in thinking mode, about twenty answers at a time (nexus/gen_text.py): one list of 8,000
# prompts of long1's kinds (none of long1's own), a quarter of it per job; the first two quarters are written in the
# places of train-depth3-opd-0 and train-depth4-opd-0. Up to 3,072 tokens an answer. Saved in ~/specdec-data/think1
# (about 12 MB per job). This job writes quarter 0 (queued as gen-think-0 before, whose place went to the public drafter's replay).
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/gen_text.py --out-dir "$HOME/specdec-data/think1" --known "$HOME/specdec-data/long1" \
  --total 8000 --seed 7 --shard 0 --num-shards 4 --max-new-tokens 3072 --max-hours 3.55 --results-dir "$RESULTS_DIR" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
