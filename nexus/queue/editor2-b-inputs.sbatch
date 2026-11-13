#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
# Second version of the draft editor, part B: what should the editor read, and how big
# should it be? Six editors train side by side on the same batches as part A. All are
# rewarded for the number of tokens in a row the target accepts, unless noted.
#   pce         rewarded for every token up to the first likely miss
#   eal-tokens  reads only the draft's tokens (and the drafter's confidence)
#   eal-states  reads only the drafter's states (and confidence), not the draft's tokens
#   eal-big     four layers of width 512 instead of two of width 256
#   eal-kl      also imitates the trained second drafter
#   eal-slow    learning rate 0.0003 instead of 0.001
VARIANTS="pce:loss=pce;eal-tokens:states=0;eal-states:tokens=none;eal-big:width=512,layers=4;eal-kl:kl=0.5;eal-slow:lr=0.0003"
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
  --teacher "$CKPT/v1-edit" --out-dir "$CKPT" --prefix "editor-v2-" --status "$RESULTS_DIR/status.json" \
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
