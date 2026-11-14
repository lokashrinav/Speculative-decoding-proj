#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
# The trained left-to-right editor, longer run. Halfway through the first run the best
# recipes were: start from the counted token-pair table, also match how the target ranks
# the candidates, flatter position weights, lower learning rate. This run trains eight
# combinations of those for 30,000 steps (the first run had 12,000), then times the best
# one in real generation. The edited draft replaces the first draft.
VARIANTS="bs:bigram=0.25,soft=1;bs-d7:bigram=0.25,soft=1,decay=7;bs-slow:bigram=0.25,soft=1,lr=0.0002;bs-d7-slow:bigram=0.25,soft=1,decay=7,lr=0.0002;bs-w5:bigram=0.5,soft=1;bs-s2:bigram=0.25,soft=2;bs-big:bigram=0.25,soft=1,rank=512;s-d7:soft=1,decay=7"
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
  --out-dir "$CKPT" --prefix "pair-v1b-" --table "$TABLE" --status "$RESULTS_DIR/status.json" \
  --steps 30000 --validate-every 5000 --train-hours 3 > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.train.err"
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
