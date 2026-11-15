#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=13:00:00
# Week 1 of the hindsight-memory plan, step 2 (experiments E0 to E3). Qwen3-4B answers one
# quarter of the long-answer prompts in thinking mode, decoded with plain DFlash (the drafts
# are never changed). While it decodes, a measuring probe records what a memory of the
# drafter's own earlier drafts in the same answer would have known at each rejected token
# (D1), next to token-only memories (D3), by position in the answer (D2), with label
# delays (D4) and the gain of a perfectly used memory (D5). The answers are saved for the
# replays of weeks 2 and 3. Four jobs each take one quarter of the prompts.
SHARD=2
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
DATA="$HOME/specdec-data/long1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" "$DATA" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/probe_memory.py --prompts-file "$DATA/prompts.json" --shard "$SHARD" --num-shards 4 \
  --max-new-tokens 16384 --out-dir "$DATA" --results-dir "$RESULTS_DIR" --max-hours 11.5 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
du -sh "$DATA" > "$RESULTS_DIR/disk.txt" 2>&1
