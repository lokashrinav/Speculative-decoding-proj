#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
# Second version of the draft editor, part A: what should the editor be rewarded for?
# Five editors train side by side on exactly the same batches. The edited draft
# replaces the first draft. The best editor is then timed in real generation.
#   plain-ce  the first version again (same layout, same reward), as the baseline
#   ce        first version's reward, new layout (inputs rescaled, reads the drafter's confidence)
#   ce-mix    the same, and while training, confident draft tokens are shown as the true token
#   tv        a token counts by how likely it is to be accepted
#   eal       reward = expected number of tokens in a row the target accepts
VARIANTS="plain-ce:plain=1,loss=ce;ce:loss=ce;ce-mix:loss=ce,mix=0.5;tv:loss=tv;eal:loss=eal"
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
CKPT="$HOME/specdec-ckpt"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" "$CKPT" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/train_editor2.py --data-dir "$HOME/specdec-data/v1" --variants "$VARIANTS" \
  --out-dir "$CKPT" --prefix "editor-v2-" --status "$RESULTS_DIR/status.json" \
  --also-evaluate "$CKPT/editor-v1-main.pt,$CKPT/editor-v1-tokens.pt,$CKPT/editor-v1-keepfirst.pt,$CKPT/editor-v1-teacher.pt" \
  --steps 12000 --train-hours 2 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.train.err"
echo "train exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.train.err" | tail -c 3000 > "$RESULTS_DIR/messages.txt"
BEST=$(python nexus/best_editor.py "$RESULTS_DIR/status.json")
echo "best editor: ${BEST:-none beat the first draft}" >> "$RESULTS_DIR/exit.txt"
[ -n "$BEST" ] || exit 0
for name in gsm8k mt-bench math500 humaneval; do
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants "dflash,editor:$BEST" \
    --analysis-samples 40 --out "$RESULTS_DIR/$name" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.$name.err"
  echo "bench $name exit $?" >> "$RESULTS_DIR/exit.txt"
  grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.$name.err" | tail -c 2000 >> "$RESULTS_DIR/messages.txt"
done
