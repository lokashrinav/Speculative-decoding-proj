#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
# Two no-training diagnostics for "a drafter that learns from its own draft":
#  - where the time of a second look goes, and per-round data for a gate
#  - which part of the draft carries the useful information (real token,
#    runner-up token, random token, soft mix, or plain resampling)
DATASETS="gsm8k math500"
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
VARIANTS="dflash,self-keep1,runnerup,randomcond,softcond,resample"
for name in $DATASETS; do
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants "$VARIANTS" \
    --analysis-samples 80 --out "$RESULTS_DIR/$name" > /dev/null 2> "$RESULTS_DIR/$name.err"
  echo "exit $?" >> "$RESULTS_DIR/$name.txt"
  grep -E "Error|Traceback" -A12 "$RESULTS_DIR/$name.err" | tail -c 3000 > "$RESULTS_DIR/$name.errors.txt"
  mv "$RESULTS_DIR/$name.err" "$HOME/specdec-slurm/$JOB_NAME-$name.err"
done
