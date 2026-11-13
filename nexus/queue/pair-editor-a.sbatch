#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=05:00:00
# The left-to-right editor with a trained judge. For each draft position it learns to
# pick the target's token among the first pass's top candidates, told the true token
# just before. Eleven versions train side by side on the same batches; the best one
# is then timed in real generation. The edited draft replaces the first draft.
#   base           top 16 candidates, reads the drafter's state and the token just chosen
#   k8, k4         fewer candidates
#   tokens         does not read the drafter's state (a learned token-pair judge only)
#   bigram         starts from the counted token-pair table and learns on top of it
#   bigram-tokens  the same without the drafter's state
#   soft           also matches how the target itself ranks the candidates
#   bigram-soft    both
#   slow, decay7, big   lower learning rate, flatter position weights, wider editor
VARIANTS="base;k8:k=8;k4:k=4;tokens:states=0;bigram:bigram=0.25;bigram-tokens:bigram=0.25,states=0;soft:soft=1;bigram-soft:bigram=0.25,soft=1;slow:lr=0.0002;decay7:decay=7;big:rank=512"
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
CKPT="$HOME/specdec-ckpt"
TABLE="$CKPT/bigram-v1-pair.pt"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
mkdir -p "$HOME/specdec-code" "$CKPT" && \
  git clone -q --depth 1 -b nexus-jobs https://github.com/lokashrinav/Speculative-decoding-proj.git "$CODE" || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/train_pair_editor.py --data-dir "$HOME/specdec-data/v1" --variants "$VARIANTS" \
  --out-dir "$CKPT" --prefix "pair-v1-" --table "$TABLE" --status "$RESULTS_DIR/status.json" \
  --steps 12000 --train-hours 2 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.train.err"
echo "train exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.train.err" | tail -c 3000 > "$RESULTS_DIR/messages.txt"
BEST=$(python nexus/best_editor.py "$RESULTS_DIR/status.json")
echo "best editor: ${BEST:-none beat the first draft}" >> "$RESULTS_DIR/exit.txt"
[ -n "$BEST" ] || exit 0
for name in gsm8k mt-bench math500 humaneval; do
  python nexus/bench_redraft.py --dataset "$name" --max-samples 80 --variants "dflash,pair:$BEST:$TABLE" \
    --analysis-samples 40 --out "$RESULTS_DIR/$name" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.$name.err"
  echo "bench $name exit $?" >> "$RESULTS_DIR/exit.txt"
  grep -E "Error|Traceback" -A12 "$HOME/specdec-slurm/$JOB_NAME.$name.err" | tail -c 2000 >> "$RESULTS_DIR/messages.txt"
done
