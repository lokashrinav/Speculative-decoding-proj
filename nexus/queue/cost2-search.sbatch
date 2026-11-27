#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:30:00
# (cost1 died recording the graph: a constant was copied from the CPU inside it; fixed. cost2 is the same job.)
# The cost of the search: the tree with counts and the pair scores searched in numpy on the CPU (ctree63, ptree63) against
# the same search in tensors of fixed shape on the GPU (qctree63, qptree63; dflash/redraft.py: CountTree.search_tensors),
# and that search recorded as one CUDA graph and replayed every round (gctree63, gptree63).
# The trees are the same, so tokens per round should agree and only the milliseconds per round differ. Published
# correctors take under 1 ms a round; ours took about 4.3 ms more than DDTree (next3-speed).
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
python nexus/live_speed.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --rank 16 --min-tokens 2000 --max-answers 4 --max-hours 2.2 \
  --count-data "$HOME/specdec-data/think1" --settings "plain,plain+tree63,plain+ctree63,plain+qctree63,plain+gctree63,plain+ptree63@opd-p2s,plain+qptree63@opd-p2s,plain+gptree63@opd-p2s" \
  > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err"
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

