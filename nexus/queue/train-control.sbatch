#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=16:00:00
#SBATCH --dependency=afterok:8384852:8384853:8384854
# Train the same drafter to take a second look at its own draft (second-pass-only
# add-on), then measure real generation speed with the trained second look.
VARIANT=control
MODE=keep0
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
CKPT="$HOME/specdec-ckpt/v1-$VARIANT"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" "$HOME/specdec-ckpt" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/train_redraft.py --data-dir "$HOME/specdec-data/v1" --variant "$VARIANT" --out "$CKPT" \
  --status "$RESULTS_DIR/status.json" --epochs 2 --max-hours 11 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.train.err"
echo "train exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback|time budget" -A12 "$HOME/specdec-slurm/$JOB_NAME.train.err" | tail -c 3000 > "$RESULTS_DIR/messages.txt"
[ -f "$CKPT/model.safetensors" ] || exit 1
VARIANTS="dflash,self-keep1,ckpt:$CKPT:$MODE"
for name in gsm8k mt-bench math500 humaneval; do
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants "$VARIANTS" \
    --analysis-samples 40 --out "$RESULTS_DIR/$name" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.$name.err"
  echo "bench $name exit $?" >> "$RESULTS_DIR/exit.txt"
  grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.$name.err" | tail -c 2000 >> "$RESULTS_DIR/messages.txt"
done
du -sh "$HOME/specdec-ckpt" "$HOME/hf-cache" "$ENV" > "$RESULTS_DIR/disk.txt" 2>&1
quota -s >> "$RESULTS_DIR/disk.txt" 2>&1
