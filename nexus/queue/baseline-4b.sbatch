#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# First real end-to-end measurement: target-only vs DFlash on Qwen3-4B (A5000),
# then a short three-way run that adds Frank's token-only secondary drafter.
set -x
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME"
export HF_HOME="$HOME/hf-cache"
export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false
cd "$HOME" || exit 1
module load Python3/3.12.3 2>&1 || true
python3 --version
rm -rf "$CODE" && mkdir -p "$(dirname "$CODE")" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
[ -x "$ENV/bin/python" ] || python3 -m venv "$ENV" || exit 1
source "$ENV/bin/activate"
python -m pip install -q -e "$CODE[local]" accelerate 2>&1 | tail -5
python - <<'PY' > "$RESULTS_DIR/env.txt" 2>&1
import torch, transformers, sys
print(sys.version); print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 2**20, "MiB")
PY
nvidia-smi --query-gpu=name,memory.total --format=csv >> "$RESULTS_DIR/env.txt"
du -sh "$ENV" >> "$RESULTS_DIR/env.txt" 2>&1
cd "$CODE" || exit 1
COMMON="--model Qwen/Qwen3-4B --draft z-lab/Qwen3-4B-DFlash-b16 --block-size 16 --max-new-tokens 256 --temperature 0 --reasoning off"
python -m dflash.cli benchmark transformers $COMMON --dataset gsm8k --max-samples 64 \
  > "$RESULTS_DIR/dflash_gsm8k_64.txt" 2> "$RESULTS_DIR/dflash_gsm8k_64.err"
echo "exit $?" >> "$RESULTS_DIR/dflash_gsm8k_64.txt"; tail -c 4000 "$RESULTS_DIR/dflash_gsm8k_64.err" > "$RESULTS_DIR/dflash_gsm8k_64.err.tail"; rm -f "$RESULTS_DIR/dflash_gsm8k_64.err"
python -m dflash.cli benchmark transformers $COMMON --secondary-model Qwen/Qwen3-0.6B --dataset gsm8k --max-samples 16 \
  > "$RESULTS_DIR/secondary_gsm8k_16.txt" 2> "$RESULTS_DIR/secondary_gsm8k_16.err"
echo "exit $?" >> "$RESULTS_DIR/secondary_gsm8k_16.txt"; tail -c 4000 "$RESULTS_DIR/secondary_gsm8k_16.err" > "$RESULTS_DIR/secondary_gsm8k_16.err.tail"; rm -f "$RESULTS_DIR/secondary_gsm8k_16.err"
du -sh "$HF_HOME" "$ENV" >> "$RESULTS_DIR/env.txt" 2>&1
quota -s >> "$RESULTS_DIR/env.txt" 2>&1
