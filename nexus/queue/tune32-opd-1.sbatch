#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=07:00:00
# Choices about very long answers are made here, not on the benchmark: the 12 validation answers of up
# to 32,768 tokens (val32-decode) replayed with the better drafter. Plain, sharper attention, fixed weights
# taught ahead of time, learning on every round, learning only after 8,000 or 16,000 tokens (also on
# every 2nd round, with a smaller and a larger step), and fixed weights that start to learn at 16,000
# tokens: coasting on what is already known and learning only where that stops working.
SHARD=1
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/long2"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
for base in /scratch1 /scratch0 /tmp; do OPD="$base/$USER/opd-4b-thinking"; mkdir -p "$OPD" 2>/dev/null && break; done
python - "$OPD" > "$RESULTS_DIR/download.txt" 2>&1 <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("bingyang-lei/Qwen3-4B-Thinking-Draft-OPD", local_dir=sys.argv[1])
PY
SETTINGS="$(head -n 1 nexus/settings/tune32-opd.txt)"
[ -n "$SETTINGS" ] || exit 1
echo "$SETTINGS" > "$RESULTS_DIR/settings.txt"
python nexus/adapt_drafter.py --draft "$OPD" --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part all \
  --settings "$SETTINGS" \
  --rank 16 --shard "$SHARD" --num-shards 2 --max-hours 6.5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
