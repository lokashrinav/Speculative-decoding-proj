#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
# 1) Three-way run (target-only, DFlash, DFlash + Qwen3-0.6B secondary) with token
#    mismatches reported instead of aborting the run.
# 2) Target-only vs DFlash baselines on three more datasets.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
module load Python3/3.12.3 >/dev/null 2>&1 || true
rm -rf "$CODE" && mkdir -p "$(dirname "$CODE")" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
python -m pip install -q -e "$CODE[local]" accelerate 2>&1 | tail -3
cd "$CODE" || exit 1
COMMON="--model Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 --block-size 16 --max-new-tokens 256 --temperature 0 --reasoning off"
run() { # name, extra args...
  name=$1; shift
  python -m dflash.cli benchmark transformers $COMMON "$@" > "$RESULTS_DIR/$name.txt" 2> "$RESULTS_DIR/$name.err"
  echo "exit $?" >> "$RESULTS_DIR/$name.txt"
  grep -E "WARNING|Error|Traceback" -A6 "$RESULTS_DIR/$name.err" | tail -c 3000 > "$RESULTS_DIR/$name.warnings.txt"
  rm -f "$RESULTS_DIR/$name.err"
}
DFLASH_MISMATCH=warn run secondary_gsm8k_32 --secondary-model Qwen/Qwen3-0.6B --dataset gsm8k --max-samples 32
run dflash_math500_64 --dataset math500 --max-samples 64
run dflash_humaneval_64 --dataset humaneval --max-samples 64
run dflash_mtbench_40 --dataset mt-bench --max-samples 40
