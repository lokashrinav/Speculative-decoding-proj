#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# Baseline: how much faster is plain decoding when the drafter itself has simply
# been retrained on our 20,000 answers (no second look, no editor)?
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
RETRAINED="$HOME/specdec-ckpt/v1-control"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
: > "$RESULTS_DIR/exit.txt"
for name in gsm8k mt-bench math500 humaneval; do
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants dflash \
    --out "$RESULTS_DIR/$name.original" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.$name.a.err"
  echo "original $name exit $?" >> "$RESULTS_DIR/exit.txt"
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants dflash --draft "$RETRAINED" \
    --out "$RESULTS_DIR/$name.retrained" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.$name.b.err"
  echo "retrained $name exit $?" >> "$RESULTS_DIR/exit.txt"
  grep -h -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.$name.a.err" "$HOME/specdec-slurm/$JOB_NAME.$name.b.err" | tail -c 2000 >> "$RESULTS_DIR/messages.txt"
done
