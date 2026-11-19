"""Where the answer's own path leaves a tree of drafts, and why: several trees on the SAME rounds.

    python nexus/trace_trees.py --draft opd --data-dir ~/specdec-data/long1 --count-data ~/specdec-data/think1 \
        --settings tree63,ctree63,ptree63@opd-p1,stree63@opd-e1,stree63p16@opd-e1 --results-dir nexus/results/NAME

Replay (nexus/redraft_long.py) gives every tree its tokens per round, each tree on its own rounds. That says
which tree is better, not why: the second drafter is right more often than the counts at the place after a
branch leaves the first run's choices (49% against 39%), and its tree still gets fewer tokens (8.21 against
8.31). Here the answer is decoded once, with the FIRST tree of --settings, and in every round all trees are
built from the same first run. For each tree the round is recorded: how many of its tokens are the answer's (as
the target would accept them), and why the answer's path is not in the tree at the next place:

    block end                 the path reached the end of the block
    not a candidate           the answer's token is not among the candidates the tree's scores are over
    no correction left        (trees of a second drafter) the node is one no pass had a row for: nothing was
                              proposed after it
    chain token wrong         (the same trees) after it only the one token of a chain was proposed, and it is
                              not the answer's
    ranked first, not kept    the tree's scores put the answer's token first among the node's children, and the
                              tree's nodes went elsewhere
    ranked 2 to 4, ranked 5 or lower   the scores put other tokens before it

and where that is: before, at or after the first place at which the first run's own first choice is wrong (the
first fork).

Two kinds of numbers are kept apart.

  * Of the trees AS BUILT (nothing of the answer is used to build them): the tokens per round on these rounds,
    the reasons above, and round by round which tree got further.
  * "Following the answer": a ceiling, not a method. For each tree's scores, at which position they put the
    answer's token among the children of the answer's own path, at every depth (for a second drafter: with a row
    for exactly those nodes, made in one more pass). From it: how far a path would get if at every node the b
    best children were kept (b = 1, 2, 3, 4, 8), which is what the scores are worth where the tree's
    construction does not limit them and it is known which branch matters; and, for a stop, how many tokens
    would follow if it were mended (the place itself and the first choices after it; after a token that is not
    a candidate nothing more is looked at).

Written: trace.json (the tables) and rounds.json (per round and tree: the tokens and the reason).
"""
import argparse
import os
import time

import torch

from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.redraft import CountTree, DraftTree, SecondTree
from dflash.replay import replay_decode, target_features
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter, follow_counts, make_method, with_counts
from nexus.replay_memory import load_answers

CAUSES = ("block end", "not a candidate", "no correction left", "chain token wrong", "ranked first, not kept",
          "ranked 2 to 4", "ranked 5 or lower")
WHERE = ("before the first fork", "at the first fork", "after the first fork")
KEPT = (1, 2, 3, 4, 8)


def _leading(tokens: list, truth: list) -> int:
    count = 0
    for token, true in zip(tokens, truth):
        if token != true:
            break
        count += 1
    return count


def _position(values, index: int) -> int:
    """Where candidate `index` stands when the values are put in falling order (0: first); of equal values the lower index first."""
    mine = values[index]
    return int(sum(1 for at, value in enumerate(values) if value > mine or (value == mine and at < index)))


class Trace:
    """Builds every tree of `methods` (name: tree) in every round and records where the answer's path leaves it. The
    first tree is the one that is returned, so the rounds are its rounds. ids: the answer being replayed ([1, length]),
    set before each answer. block: the drafter's block; rounds with a shorter one (an answer's end) are not recorded."""

    replace = True

    def __init__(self, methods: dict, block: int = 16):
        self.methods, self.block = methods, block
        for method in methods.values():
            if not isinstance(method, (DraftTree, SecondTree)):
                raise ValueError("only trees of drafts can be traced")
            method.keep_last = True
        self.ids = None
        # per round: (tokens, reason, where, forks passed, would, mended, beyond the 32, reach, kept ...)
        self.rounds = {name: [] for name in methods}
        self.first_run = []                             # per round: (first choices right in a row, candidates of 32 right in a row)

    def start(self, drafter) -> None:
        for method in self.methods.values():
            method.start(drafter)

    def close(self) -> None:
        for method in self.methods.values():
            if hasattr(method, "close"):
                method.close()

    def __call__(self, s):
        width = s.block_ids.shape[1]
        truth = self.ids[0, s.start + 1 : s.start + width].tolist()
        whole = width == self.block and len(truth) == width - 1        # (an answer's last rounds are shorter: left out)
        ranked, result = None, None
        if whole:
            logits = s.draft_logits[0].float()
            ranked = logits.log_softmax(dim=-1).topk(min(128, logits.shape[-1]), dim=-1).indices.tolist()
            plain = _leading([row[0] for row in ranked], truth)
            among = 0
            while among < width - 1 and truth[among] in ranked[among][:32]:
                among += 1
            self.first_run.append((plain, among))
        for name, method in self.methods.items():
            rows = method(s)
            if result is None:
                result = rows
            if whole:
                self.rounds[name].append(self.record(method, rows, truth, ranked, plain))
        return result

    @staticmethod
    def following(method, ranks: list, reach: int) -> list:
        """Where the tree's scores put the answer's token among the children of the answer's own path, for the places
        0 .. reach - 1 (0: first)."""
        if isinstance(method, SecondTree):
            given = method.corrected([tuple(ranks[:place]) for place in range(1, reach)])
            return ranks[: min(reach, 1)] + [given[tuple(ranks[:place])][1].index(ranks[place]) for place in range(1, reach)]
        if isinstance(method, CountTree):
            return [_position(method.children(place, ranks[place - 2] if place >= 2 else 0, ranks[place - 1] if place else 0,
                                              not any(ranks[:place])), ranks[place]) for place in range(reach)]
        return ranks[:reach]

    def record(self, method, rows, truth: list, ranked: list, plain: int):
        places = len(truth)
        got = max(_leading(row[1:], truth) for row in rows.tolist())
        if isinstance(method, SecondTree):
            lists, many = method.last["tokens"], method.top
        elif isinstance(method, CountTree):
            lists, many = method.seen["tokens"], len(method.seen["tokens"][0])
        else:
            lists, many = ranked, len(ranked[0])       # DDTree: any token it can pick is a candidate
        ranks = [row.index(token) if token in row[:many] else None for row, token in zip(lists, truth)]
        reach = 0
        while reach < places and ranks[reach] is not None:
            reach += 1
        positions = self.following(method, ranks, reach)
        would = -1
        if got >= places:
            cause = 0
        elif ranks[got] is None:
            cause = 1
        else:
            position = positions[got]
            if isinstance(method, SecondTree) and got:
                node, last = tuple(ranks[:got]), method.last
                if node in last["corrected"]:
                    position = last["final_order"][node][1].index(ranks[got])
                elif node in last["final_order"]:
                    position = 0 if last["final_order"][node][1][0] == ranks[got] else "chain"
                else:
                    position = "none"
            if position in ("none", "chain"):
                cause, would = (2 if position == "none" else 3), positions[got]
            else:
                cause = 4 if position == 0 else 5 if position < 4 else 6
        mended = 0
        if cause:
            mended = 1
            while got + mended < reach and positions[got + mended] == 0:
                mended += 1
        where = 0 if got < plain else 1 if got == plain else 2
        forks = sum(1 for place in range(got) if truth[place] != ranked[place][0])
        kept = [_leading([int(position < most) for position in positions], [1] * reach) for most in KEPT]
        beyond = int(got < places and truth[got] not in ranked[got][:32])      # the same for every tree: not among the first run's 32
        return (got, cause, where, forks, would, mended, beyond, reach, *kept)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def summarize(trace: Trace) -> dict:
    names = list(trace.rounds)
    out = {"rounds": len(trace.first_run), "decoded_with": names[0],
           "first_run": {"tokens_per_round_of_first_choices": round(_mean(row[0] for row in trace.first_run) + 1, 3),
                         "tokens_per_round_if_any_of_32_candidates_counted": round(_mean(row[1] for row in trace.first_run) + 1, 3)},
           "trees": {}}
    for name in names:
        rows = trace.rounds[name]
        count = max(len(rows), 1)
        tree = {"tokens_per_round": round(_mean(row[0] for row in rows) + 1, 3),
                "following_the_answer": {f"best_{most}_kept": round(_mean(row[8 + at] for row in rows) + 1, 3) for at, most in enumerate(KEPT)},
                "candidates_reach": round(_mean(row[7] for row in rows) + 1, 3),
                "stops_at_a_token_not_among_the_first_runs_32": round(_mean(row[6] for row in rows), 4),
                "tokens_past_the_first_fork": {"none (stops before or at it)": round(sum(1 for row in rows if row[2] < 2) / count, 4),
                                               "share_of_rounds_past_it": round(sum(1 for row in rows if row[2] == 2) / count, 4)},
                "why_the_path_stops": {}}
        for index, cause in enumerate(CAUSES):
            mine = [row for row in rows if row[1] == index]
            if not mine:
                continue
            entry = {"share_of_rounds": round(len(mine) / count, 4),
                     "tokens_of_the_answer_before_it": round(_mean(row[0] for row in mine), 2),
                     "would_follow_if_mended": round(_mean(row[5] for row in mine), 2),
                     "tokens_per_round_if_mended": round(sum(row[5] for row in mine) / count, 3),
                     "where": {WHERE[at]: round(sum(1 for row in mine if row[2] == at) / len(mine), 3) for at in range(3)},
                     "forks_passed_before_it": round(_mean(row[3] for row in mine), 2),
                     "token_not_among_the_first_runs_32": round(_mean(row[6] for row in mine), 3)}
            if index in (2, 3):                        # what the same scores would have said, had the node had a row
                entry["with_a_row_the_answers_token_would_be"] = {
                    "first": round(sum(1 for row in mine if row[4] == 0) / len(mine), 3),
                    "2 to 4": round(sum(1 for row in mine if 1 <= row[4] < 4) / len(mine), 3),
                    "5 or lower": round(sum(1 for row in mine if row[4] >= 4) / len(mine), 3)}
            tree["why_the_path_stops"][cause] = entry
        tree["against"] = {}
        for other in names:
            if other == name:
                continue
            theirs = trace.rounds[other]
            ahead = [mine[0] - yours[0] for mine, yours in zip(rows, theirs)]
            tree["against"][other] = {
                "rounds_further": round(sum(1 for value in ahead if value > 0) / count, 4),
                "rounds_less_far": round(sum(1 for value in ahead if value < 0) / count, 4),
                "tokens_gained_per_round": round(sum(value for value in ahead if value > 0) / count, 3),
                "tokens_lost_per_round": round(-sum(value for value in ahead if value < 0) / count, 3),
                # why this tree stopped in the rounds where the other got further
                "stopped_by_where_less_far": {CAUSES[index]: round(sum(1 for value, row in zip(ahead, rows) if value < 0 and row[1] == index)
                                                                   / max(sum(1 for value in ahead if value < 0), 1), 3)
                                              for index in range(len(CAUSES))
                                              if any(value < 0 and row[1] == index for value, row in zip(ahead, rows))}}
        counts = getattr(trace.methods[name], "counts", None)
        if isinstance(counts, dict) and counts.get("rounds"):
            tree["a_round"] = {key: round(value / counts["rounds"], 2) for key, value in counts.items() if key != "rounds"}
        out["trees"][name] = tree
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--settings", required=True, help="trees by their replay names, comma-separated; the first decodes")
    parser.add_argument("--part", default="validation")
    parser.add_argument("--count-data", action="append", default=[])
    parser.add_argument("--ckpt-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--max-answers", type=int, default=0)
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens")
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir) if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_in_this_job": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    names = args.settings.split(",")
    if any(with_counts(name) for name in names):
        follow_counts(args.data_dir, args.count_data, target.config.vocab_size, device)
    trace = Trace({name: make_method(name, args.ckpt_dir, like=draft) for name in names})
    started, done = time.time(), 0

    def report(finished):
        write_json(status_path, {"done": finished, "answers": done, "answers_in_this_job": len(answers), "draft": args.draft,
                                 "settings": args.settings, "hours": round((time.time() - started) / 3600, 3)}, indent=1)
        write_json(os.path.join(args.results_dir, "trace.json"), summarize(trace), indent=1)
        write_json(os.path.join(args.results_dir, "rounds.json"),
                   {"reasons": CAUSES, "first_run": {"first_choices": [row[0] for row in trace.first_run],
                                                     "of_32_candidates": [row[1] for row in trace.first_run]},
                    "trees": {name: {"tokens": [row[0] for row in rows], "reason": [row[1] for row in rows]}
                              for name, rows in trace.rounds.items()}}, separators=(",", ":"))

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and done >= args.max_answers):
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2 or (args.longest and len(generated) > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        trace.ids = ids
        replay_decode(draft, target, ids, prompt_length, ended=generated[-1] in stop_set,
                      features=target_features(target, draft, ids), improver=trace, block_size=16)
        done += 1
        report(False)
    trace.close()
    report(True)


if __name__ == "__main__":
    main()
