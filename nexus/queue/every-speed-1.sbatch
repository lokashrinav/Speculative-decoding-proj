#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
# Where the milliseconds of the tree the second drafter scores go (real decoding, 3 validation answers, Draft-OPD).
# In the first speed test its part of a round took 11.1 ms where the tree with counts takes 3.2 ms, which made it
# 7% slower although it holds 3% more tokens. Since then the scoring takes the counts' lifts from the tree's own
# lookup and reads the GPU back once. The x in a name counts the milliseconds of each part (first tree, rows,
# pass, scores, choice); the same tree without x shows the total without the waits the timing adds.
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
python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --rank 16 \
  --min-tokens 2000 --max-answers 3 --max-hours 1.2 --count-data "$HOME/specdec-data/think1" \
  --settings "plain,plain+ctree63,plain+cstree63x@opd-e1c,plain+cstree63@opd-e1c,plain+cstree63t1x@opd-e1c,plain+cstree63f40x@opd-e1c,plain+stree63x@opd-e1" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"
