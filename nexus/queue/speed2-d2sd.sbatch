#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
# (speed1 died loading the saved drafter outside the benchmark script; fixed. speed2 is the same job.)
# The D2SD second drafter made here (drafter-opd-d2sd, loop6-d2sd) against the matched baselines in real decoding, the
# same text in every row: plain; the drafter run again at the four likely breaks (D2SD's "control reuse", +redraft4);
# the trained second drafter at the four likely breaks (branches4@drafter-opd-d2sd); DDTree; the count table and the
# corrector recorded as CUDA graphs. Acceptance and full round time: does a learned branch continuation earn its pass?
ENV="$HOME/specdec-env"
CODE="/tmp/specdec-code-$USER/$JOB_NAME-$SLURM_JOB_ID"
export HF_HOME="$HOME/hf-cache" TOKENIZERS_PARALLELISM=false
module load Python3/3.12.3 >/dev/null 2>&1 || true
# The code is copied from the runner's folder as committed there (under 1 MB), not cloned again.
mkdir -p "$CODE" && ( set -o pipefail; git -C "$HOME/specdec-runner" archive HEAD dflash nexus tests \
  | tar -x -C "$CODE" --exclude='nexus/results' --exclude='nexus/logs' --exclude='nexus/status' ) || exit 1
source "$ENV/bin/activate" || exit 1
cd "$CODE" || exit 1
export PYTHONPATH="$CODE"
git -C "$HOME/specdec-runner" rev-parse HEAD > "$RESULTS_DIR/code_commit.txt"
python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --rank 16 --min-tokens 2000 --max-answers 8 --max-hours 3.7 \
  --count-data "$HOME/specdec-data/think1" --settings "plain,plain+redraft4,plain+branches4@drafter-opd-d2sd,plain+tree63,plain+gctree63,plain+gptree63@opd-p2s" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

