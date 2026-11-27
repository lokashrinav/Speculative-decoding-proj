#!/bin/bash -l
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
# (oracle1 died at once on a shadowed import in the method names; fixed. oracle2 is the same job.)
# The room for branching at the drafter's four likely breaks, an oracle (dflash/redraft.py: OracleBranches): DDTree's
# tree plus, at each likely break, the second choice followed by the ANSWER's own next tokens, i.e. perfect continuations
# whenever the alternative token is right. otree63: 63 nodes plus up to 4 x 4 branch nodes; otree63f47: a 47-node first
# tree so the whole stays near 63; t15: continuations to the block's end. The pair scores for reference. Traced on
# DDTree's rounds (what the branches add over DDTree, round by round) and replayed on their own rounds. A feasibility
# bound for "branching inside the drafter", not a prediction: the real continuations will be worse than the answer's.
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
( mkdir -p "$RESULTS_DIR/trace" \
  && python nexus/trace_trees.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR/trace" --part validation --max-answers 8 --max-hours 0.8 \
    --settings "tree63,otree63,otree63f47,otree63t15,otree63f31t15,ptree63@opd-p2s" > /dev/null 2> "$HOME/specdec-slurm/$JOB_NAME.err" \
  && python nexus/redraft_long.py --draft opd --data-dir "$HOME/specdec-data/long1" --results-dir "$RESULTS_DIR" --part validation --max-answers 8 --max-hours 1.0 \
    --settings "otree63,otree63f47,otree63t15,otree63f31t15" > /dev/null 2>> "$HOME/specdec-slurm/$JOB_NAME.err" )
echo "exit $?" > "$RESULTS_DIR/exit.txt"
grep -E "Error|Traceback" -A14 "$HOME/specdec-slurm/$JOB_NAME.err" | tail -c 4000 > "$RESULTS_DIR/messages.txt"

