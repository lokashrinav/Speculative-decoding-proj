#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=07:00:00
# More training data for the left-to-right editor: about 77,000 more prompts (chat, math,
# code; none from the evaluation sets) answered by Qwen3-4B itself, in a second folder (v2).
# Eight jobs each generate one eighth. Prompts already in v1 are dropped when training.
SHARD=7
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" "$HOME/specdec-data/v2" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/gen_data.py --total 81000 --seed 1 --shard "$SHARD" --num-shards 8 --batch-size 48 \
  --out-dir "$HOME/specdec-data/v2" --status "$RESULTS_DIR/status.json" --max-hours 4 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "loaded|FAILED|Error|Traceback|time budget" -A8 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 3000 > "$RESULTS_DIR/messages.txt"
du -sh "$HOME/specdec-data/v2" "$HOME/hf-cache" "$ENV" > "$RESULTS_DIR/disk.txt" 2>&1
quota -s >> "$RESULTS_DIR/disk.txt" 2>&1
