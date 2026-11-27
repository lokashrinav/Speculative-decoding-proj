#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=03:00:00
# A 127-node corrector tree against 63, recorded as CUDA graphs, where the check is nearly flat (C11: 128 rows cost about
# 1 ms more than 64 at up to 4k of text, 7.5 ms more at 8k): on 5 prompts each of the four benchmarks (short text) and on
# 4 long answers. Both the tree build's cost and the check's are in the round time.
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
( python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --rank 16 --bench gsm8k,math500,humaneval,mt-bench --per-bench 5 --reasoning off --max-answers 20 --max-hours 1.3 \
    --count-data "$HOME/specdec-data/think1" --settings "plain,plain+tree63,plain+gptree63@opd-p2s,plain+gptree127@opd-p2s" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && mkdir -p "$RESULTS_DIR/long" \
  && python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/long" --part validation --rank 16 --min-tokens 2000 --max-answers 4 --max-hours 1.4 \
    --count-data "$HOME/specdec-data/think1" --settings "plain,plain+tree63,plain+gptree63@opd-p2s,plain+gptree127@opd-p2s" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

