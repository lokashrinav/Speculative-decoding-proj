#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=07:00:00
#SBATCH --dependency=afterany:8419295:8419296:8419297:8419298
# Week 1 of the hindsight-memory plan, step 3. Starts when the four decoding jobs have ended.
# The saved long answers are replayed round by round without the target's check (about five
# times cheaper than decoding), with a richer measurement than the first pass:
#   - the trained left-to-right editor runs on every draft and is only scored, so the numbers
#     also say what a memory could add on top of the editor;
#   - the simple rules also look at how sure the drafter was of its own token;
#   - everything recorded per round is saved, one file per answer, for fitting gates later.
# Its counts for the plain drafts should agree with the decoding jobs' counts (a cross-check).
SHARD=2
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
CKPT="$HOME/specdec-ckpt"
DATA="$HOME/specdec-data/long1"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" "$DATA/rounds1" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/replay_memory.py --data-dir "$DATA" --out-dir "$DATA/rounds1" --results-dir "$RESULTS_DIR" \
  --editor "$CKPT/pair-v2-bs-d7-slow.pt" --table "$CKPT/bigram-v2-pair.pt" \
  --shard "$SHARD" --num-shards 4 --max-hours 6 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
du -sh "$DATA" "$DATA/rounds1" > "$RESULTS_DIR/disk.txt" 2>&1
