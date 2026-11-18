"""Where do DDTree's rounds end, and what would have carried them further?

    python nexus/tree_ends.py --data-dir ~/specdec-data/long1 --results-dir nexus/results/NAME

Everything tried on top of DDTree so far (a second run of the drafter, blocks from what the last
check left, the table of what the target already said) adds few tokens to its rounds. This counts
what there is to add. Saved answers are replayed with DDTree in place of the first draft, and in
every round the target's walk through the tree is followed to the node where it ends. A round ends
in one of these ways:

    block end        the walk reached the end of the block (15 draft tokens): only a longer draft helps
    leaf             it ended at a node that has no children in the tree: nothing was there after an
                     accepted token. On the trunk (the path of first choices) or off it (after an
                     alternative token: the place of a second run, which writes tokens after one), and
                     with the answer's next token being the drafter's FIRST choice at that place (the
                     tree simply stopped: its budget went elsewhere) or another token
    not among them   the node has children, and the answer's token is not one of them: by the rank of
                     that token among the drafter's choices there (2-4: a larger tree has it; 5-10; 11
                     and more: the drafter does not see it, only something else can)

For each way it also counts what the first pass would give AFTER the missing token if that one token
were put in place: how many of the answer's next tokens are the drafter's first choices at their
places (the swap's old rest), which is what a table or a correction earns beyond its one token.
"""
import argparse
import os
import time

import numpy
import torch

from dflash.adapt import DraftTrainer
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.redraft import DraftTree
from dflash.replay import replay_decode, target_features
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers

WIDTH = 16
COLUMNS = ("answer", "where", "depth", "on_trunk", "children", "rank", "after_it", "nodes", "trunk_nodes")
WAYS = ("block end", "leaf on the trunk, the first choice was next", "leaf on the trunk, another token was next",
        "leaf off the trunk, the first choice was next", "leaf off the trunk, another token was next",
        "not among the children: rank 2-4", "not among the children: rank 5-10", "not among the children: rank 11 and more")


class Ends:
    """The replay's plug-in: DDTree, and for every round where the answer leaves the tree."""

    replace = True

    def __init__(self, budget, width=WIDTH):
        self.tree, self.width, self.rows = DraftTree(budget), width, []

    def begin(self, ids, prompt_length, repetition_from, answer):
        self.ids, self.prompt_length, self.repetition_from, self.answer = ids, prompt_length, repetition_from, answer

    def start(self, drafter):
        self.tree.start(drafter)

    def __call__(self, s):
        rows = self.tree(s)
        width = s.block_ids.shape[1]
        where = s.start - self.prompt_length
        outside = self.repetition_from is not None and where >= self.repetition_from
        truth = self.ids[0, s.start + 1 : s.start + width].tolist()
        if width != self.width or len(truth) != width - 1 or outside:
            return rows
        logits = s.draft_logits[0].float()
        top = logits.log_softmax(dim=-1).topk(min(self.tree.budget, logits.shape[-1]), dim=-1)     # as DraftTree builds it
        best, chosen = top.indices.tolist(), set(self.tree.paths(top.values.tolist()))
        path = ()
        while len(path) < width - 1:
            token = truth[len(path)]
            rank = best[len(path)].index(token) if token in best[len(path)] else None
            if rank is None or path + (rank,) not in chosen:
                break
            path = path + (rank,)
        depth = len(path)
        record = {"answer": self.answer, "where": where, "depth": depth, "on_trunk": int(not any(path)), "children": -1,
                  "rank": -1, "after_it": -1, "nodes": len(chosen), "trunk_nodes": sum(1 for node in chosen if not any(node))}
        if depth < width - 1:
            record["children"] = sum(1 for node in chosen if len(node) == depth + 1 and node[:depth] == path)
            token = truth[depth]
            record["rank"] = min(int((logits[depth] > logits[depth, token]).sum()) + 1, 999)      # 1: the first choice
            first = logits.argmax(dim=-1).tolist()
            after = 0
            while depth + 1 + after < width - 1 and first[depth + 1 + after] == truth[depth + 1 + after]:
                after += 1
            record["after_it"] = after
        self.rows.append([record[name] for name in COLUMNS])
        return rows


def way(depth, on_trunk, children, rank, width=WIDTH):
    """The name (one of WAYS) of how a round ended."""
    if depth >= width - 1:
        return WAYS[0]
    if children == 0:
        if on_trunk:
            return WAYS[1] if rank == 1 else WAYS[2]
        return WAYS[3] if rank == 1 else WAYS[4]
    return WAYS[5] if rank <= 4 else WAYS[6] if rank <= 10 else WAYS[7]     # (a node with children has its first choice)


def summary(table, width=WIDTH):
    """Shares of the ways a round ends, with the tokens the round had by then, the tokens the first
    pass would add after the missing token, and for leaves how deep they were."""
    table = numpy.asarray(table).reshape(-1, len(COLUMNS))
    out = {"rounds": int(len(table))}
    if not len(table):
        return out
    column = {name: table[:, number] for number, name in enumerate(COLUMNS)}
    names = numpy.asarray([way(*row, width=width) for row in zip(column["depth"], column["on_trunk"], column["children"], column["rank"])])
    out["draft_tokens_accepted_per_round"] = round(float(column["depth"].mean()), 3)
    out["nodes_per_tree"] = round(float(column["nodes"].mean()), 1)
    out["nodes_on_the_path_of_first_choices"] = round(float(column["trunk_nodes"].mean()), 2)
    out["ways"] = {}
    for name in WAYS:
        mask = names == name
        if not mask.any():
            continue
        entry = {"share": round(float(mask.mean()), 4), "draft_tokens_accepted_by_then": round(float(column["depth"][mask].mean()), 2)}
        if name != WAYS[0]:
            entry["first_pass_tokens_after_the_missing_one"] = round(float(column["after_it"][mask].mean()), 3)
            # what one right token there is worth with the first pass's own tokens after it, per round of ALL rounds
            entry["tokens_per_round_if_that_token_were_there"] = round(float(((column["after_it"][mask] + 1).sum()) / len(table)), 3)
        out["ways"][name] = entry
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation")
    parser.add_argument("--budgets", default="63,31,127", help="DDTree's node budgets, each replayed over the same answers")
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--max-answers", type=int, default=30)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir)
               if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    first = DraftTrainer(draft, [], rate=0.0)
    probes = {int(budget): Ends(int(budget)) for budget in args.budgets.split(",")}
    records, started = [], time.time()

    def report(done):
        status = {"done": done, "answers": len(records), "part": args.part, "hours": round((time.time() - started) / 3600, 3),
                  "draft": args.draft, "columns": COLUMNS, "answers_read": records,
                  "by_budget": {str(budget): summary(probe.rows) for budget, probe in probes.items()}}
        write_json(status_path, status, indent=1)
        if done:
            for budget, probe in probes.items():
                numpy.savez_compressed(os.path.join(args.results_dir, f"rounds-tree{budget}.npz"),
                                       rounds=numpy.asarray(probe.rows, dtype=numpy.int32), columns=numpy.asarray(COLUMNS))

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or len(records) >= args.max_answers:
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2:
            continue
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        repetition_from = None if ended else loop_start(generated)
        features = target_features(target, draft, ids)
        rounds = {}
        for budget, probe in probes.items():
            first.begin()
            probe.begin(ids, prompt_length, repetition_from, len(records))
            result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=probe,
                                   block_size=WIDTH)
            rounds[str(budget)] = int(result.rounds)
        del features
        records.append({"source": source, "origin": [name, index], "tokens": len(generated), "rounds": rounds})
        report(False)
    report(True)


if __name__ == "__main__":
    main()
