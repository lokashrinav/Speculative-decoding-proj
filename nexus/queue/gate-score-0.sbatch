#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#DEPENDS: gate-fit
# The fitted gates, step 3 of 3: replays a quarter of the validation and test answers and
# scores every gate at every margin on them. Only small tables are written.
SHARD=0
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
CKPT="$HOME/specdec-ckpt"
DATA="$HOME/specdec-data/long1"
WORK="$HOME/specdec-data/gate1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" "$WORK" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/gate_pipeline.py --mode score --data-dir "$DATA" --work-dir "$WORK" --results-dir "$RESULTS_DIR" \
  --editor "$CKPT/pair-v2-bs-d7-slow.pt" --table "$CKPT/bigram-v2-pair.pt" --shard "$SHARD" --num-shards 4 --max-hours 3.5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
du -sh "$WORK" > "$RESULTS_DIR/disk.txt" 2>&1
