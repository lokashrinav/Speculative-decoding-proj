#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=10:00:00
# The two-layer restart on eight times the text (nexus/train_second.py): on from warm-opd-s2.pt (two layers that read
# the tokens, states written into the block), on long1's training answers mixed with the new thinking-mode text
# (~/specdec-data/think1). The predictor stays as it is. Draft-OPD. Saved as warm-opd-s2x.pt.
ENV="$HOME/specdec-env"
CODE="$HOME/specdec-code/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/train_second.py --draft opd --data-dir "$HOME/specdec-data/long1" --more-data "$HOME/specdec-data/think1" \
  --results-dir "$RESULTS_DIR" --start-from opd-s2 --states-in-block --save opd-s2x --freeze-predictor \
  --no-states 0.3 --real-share 0.2 --max-hours 8.5 --check-answers 12 \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
