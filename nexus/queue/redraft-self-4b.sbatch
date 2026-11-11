#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
# Real speed of "improve the draft using the draft" with no second model:
# the same DFlash drafter re-reads the first part of its own draft, both drafts
# are verified in one target pass. Compared with plain DFlash on the same prompts.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
module load Python3/3.12.3 >/dev/null 2>&1 || true
rm -rf "$CODE" && mkdir -p "$(dirname "$CODE")" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
python -m pip install -q -e "$CODE[local]" accelerate 2>&1 | tail -3
cd "$CODE" || exit 1
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
VARIANTS="dflash,self-keep1,self-keep2,self-keepconfidence"
for spec in gsm8k:80 mt-bench:80 math500:80 humaneval:80; do
  name=${spec%%:*}; count=${spec##*:}
  python nexus/bench_redraft.py --dataset "$name" --max-samples "$count" --variants "$VARIANTS" \
    --out "$RESULTS_DIR/$name" > /dev/null 2> "$RESULTS_DIR/$name.err"
  echo "exit $?" >> "$RESULTS_DIR/$name.txt"
  grep -E "Error|Traceback" -A12 "$RESULTS_DIR/$name.err" | tail -c 3000 > "$RESULTS_DIR/$name.errors.txt"
  rm -f "$RESULTS_DIR/$name.err"
done
