#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
# Week 1 of the hindsight-memory plan, step 1: build the list of about 600 development
# prompts that lead to long answers (AIME problems from before 2024, the hardest MATH
# training problems, long-writing requests, chat prompts). None of them is in a benchmark
# used for final results. The list is written once so that the four decoding jobs cut the
# same list. No GPU is needed.
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
python nexus/probe_memory.py --only-write-prompts --total 600 --seed 0 \
  --prompts-file "$DATA/prompts.json" --results-dir "$RESULTS_DIR" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "FAILED|Error|Traceback|failed to load" -A12 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
ls -la "$DATA" > "$RESULTS_DIR/disk.txt" 2>&1
quota -s >> "$RESULTS_DIR/disk.txt" 2>&1
