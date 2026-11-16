#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=06:30:00
# The longer run of the carry-over test: 150 training answers of every kind, immediate labels at
# rate 1e-4. The adapters are also kept as they are after 10, 30 and 100 answers, to see later how
# many answers are needed. Saved to specdec-ckpt/lora-off-d.pt (and -10, -30, -100; 16 MB each).
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
mkdir -p "$HOME/specdec-ckpt"
python nexus/train_adapter.py --data-dir "$DATA" --results-dir "$RESULTS_DIR" --out "$HOME/specdec-ckpt/lora-off-d.pt" \
  --labels immediate --rate 1e-4 --rank 16 --max-answers 150 --max-hours 5.5 --snapshots 10,30,100 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
