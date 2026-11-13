#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:40:00
# How much could a left-to-right editor gain at best on our data? Nothing is trained:
# coverage of the target's token among the first pass's top candidates, oracle editors,
# what the second drafter does, and a counted (not learned) bigram editor.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/editor_headroom.py --data-dir "$HOME/specdec-data/v1" --teacher "$HOME/specdec-ckpt/v1-edit" \
  --out "$RESULTS_DIR/headroom.json" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 3000 > "$RESULTS_DIR/messages.txt"
