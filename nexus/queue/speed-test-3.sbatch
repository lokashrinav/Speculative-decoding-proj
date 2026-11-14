#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# One speed test of the best left-to-right editors in the same job, so their numbers can be
# compared directly: trained on v1 + v2 with the full and the cheap confidence input, with and
# without the counted table, and the no-training editor with counts from v1 + v2.
# The edited draft replaces the first draft.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
CKPT="$HOME/specdec-ckpt"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
: > "$RESULTS_DIR/exit.txt"
VARIANTS="dflash,pair:$CKPT/pair-v2-bs-d7-slow.pt:$CKPT/bigram-v2-pair.pt,pair:$CKPT/pair-v2d-f-bs-d7-slow.pt:$CKPT/bigram-v2-pair-d.pt,pair:$CKPT/pair-v2d-f-d7-slow.pt,bigram:$CKPT/bigram-v2-pair.pt:16:0.25:10"
for name in gsm8k mt-bench math500 humaneval; do
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants "$VARIANTS" \
    --analysis-samples 40 --out "$RESULTS_DIR/$name" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.$name.err"
  echo "bench $name exit $?" >> "$RESULTS_DIR/exit.txt"
  grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.$name.err" | tail -c 2000 >> "$RESULTS_DIR/messages.txt"
done
