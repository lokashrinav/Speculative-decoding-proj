#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:30:00
# A control with a second, stronger drafter. Another group (Draft-OPD, arXiv 2605.29343) released a
# drafter for the same target that was trained further on its own verified drafts. Is the weakness
# on long answers special to the public drafter we use? This downloads their drafter to the node's
# local disk (not the home folder), prints what the files look like, and replays 12 validation
# answers with it: plain, and with training during the answer. The same 12 answers were replayed
# with the public drafter by adapt-drafter-0.
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
for base in /scratch1 /scratch0 /tmp; do OPD="$base/$USER/opd-4b-thinking"; mkdir -p "$OPD" 2>/dev/null && break; done
echo "$OPD" > "$RESULTS_DIR/where.txt"
python - "$OPD" > "$RESULTS_DIR/download.txt" 2>&1 <<'PY'
import os, sys
from huggingface_hub import snapshot_download
path = snapshot_download("bingyang-lei/Qwen3-4B-Thinking-Draft-OPD", local_dir=sys.argv[1])
for name in sorted(os.listdir(path)):
    full = os.path.join(path, name)
    print(name, os.path.getsize(full) if os.path.isfile(full) else "folder")
for name in ("config.json", "README.md"):
    full = os.path.join(path, name)
    if os.path.exists(full):
        print("-----", name)
        print(open(full).read()[:4000])
PY
python nexus/adapt_drafter.py --draft "$OPD" --data-dir "$DATA" --results-dir "$RESULTS_DIR" --part validation \
  --settings "plain,immediate:3e-4" --rank 16 --shard 0 --num-shards 4 --max-answers 12 --max-hours 1.5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
