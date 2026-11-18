"""What a second run of the drafter needs in order to write well AFTER the break.

    python nexus/restart_probe.py --data-dir ~/specdec-data/long1 --results-dir nexus/results/NAME

The study of why a second draft helps (docs/WHY_A_SECOND_DRAFT_HELPS.md) found that with the right
token at the break the first draft's old tokens give 0.55 more tokens, a second pass that rewrites
them inside the block 0.64, and the drafter in the next round, with the target's hidden states,
2.03. So the one thing a second run can add that a swapped token or a tree cannot is a RESTART: the
drafter starts a new block at the token after the break, as it does in the next round.

At the time of a second run the target has not read this round's tokens yet. For a restart after
`n - 1` accepted draft tokens the drafter is therefore missing the target's hidden states of `n`
tokens: the block's first token and the accepted draft tokens (the token it restarts from needs
none, as in any round). n is small: the first draft breaks early.

This probe replays saved answers with the first draft alone and, in every round with a break,
restarts the drafter from the RIGHT token at the break with the missing states replaced in several
ways, all in one drafter pass. For each it counts how many of the restart's 15 tokens the target
accepts:

    real              the target's real states (the limit: this is the next round's first draft)
    none              nothing in their place; the block keeps its true positions (a gap)
    none_shift        nothing, and the block is placed right after the text the drafter has states for
    last              the last real state, repeated
    stale             the real states of the n tokens before them
    mean              the mean of the last 256 real states, repeated
    only_nearest      the real state of the token right before the restart only
    all_but_nearest   the real states of all but that token
    noisy25 ... 200   the real states plus noise of 25% ... 200% of each dimension's spread
    predicted         what a trained predictor writes for them (--predictor NAME; dflash/stand_in.py)

and for comparison, after the right token: old_rest (the first draft's old tokens) and new_rest
(rewritten inside the block by a second pass). second_choice_right says whether the drafter's
second choice at the break is the right token: where it is, a restart from the second choice is
the restart measured here. "covered" says whether the break is one of the four likely breaks.

none/last/stale/mean need no training: if they keep most of what `real` gives, a second run can
restart without any predictor of the target's states. The noisy rows say how exact a predictor of
those states has to be. Rounds are saved as tables (rounds-N.npz) like nexus/why_redraft.py.
"""
import argparse
import os
import time

import numpy
import torch

from dflash.adapt import DraftTrainer, adapter_notes, attach_adapters, load_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.redraft import SelfRedraft, break_scores, restarts
from dflash.replay import _matches, replay_decode, target_features
from dflash.stand_in import load_predictor, predicted_states, project
from dflash.sharp import QueryScale
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers

WIDTH = 16
NOISE = (25, 50, 100, 200)
ARMS = ("real", "none", "none_shift", "last", "stale", "mean", "only_nearest", "all_but_nearest") \
    + tuple(f"noisy{level}" for level in NOISE)
COLUMNS = ("answer", "where", "broke_after", "second_choice_right", "covered", "confidence", "old_rest", "new_rest") + ARMS
# --reanchor adds a second kind of restart, which needs no guess of the right token: a new block that
# starts at the LAST token the first draft got right (not at the token after it), with the states of
# the tokens before that one. It writes the token at the break again, now with those states, and goes
# on from there. Counted from the token at the break, so that it compares with
# second_choice_right * (1 + real), what a restart from the second choice gets. "rank" is the place of
# the right token among the drafter's choices at the break (1: its second choice, 2: its third, ...).
REANCHOR = ("reanchor", "reanchor_noisy100")
# --more: stand-ins that are REAL states of the target, only not the ones of these tokens (the predictors
# failed where noise did not: the states must be real more than they must be exact).
#   shallowN / deepN: the real states of the missing tokens in the drafter's first N / last N target
#     layers only, the other layers from the last state the target really has. shallow: what a check that
#     is stopped after a few layers could hand over ("split the check").
#   echo_states: for every missing token, the real state of an EARLIER place in the answer where the
#     target read the same token: before the same next token and after the same token (level 4), before
#     the same next token (3), after the same token (2), or anywhere (1). echo_nearest: only for the
#     last missing token. echo_level: the level found for that last token (0: none).
#   snap (with --predictor): the predicted vector is only used to look up the most similar real state.
MORE = ("shallow1", "shallow2", "shallow3", "shallow4", "deep1", "deep2", "echo_states", "echo_nearest")
# Is the rest of the first draft right, only one or two places off? (No drafter pass: tokens are compared.) A
# drafter that writes all tokens at once can drop or double a token, and then everything after it is right but
# shifted, which neither a swap nor a tree can use: they change tokens in place.
#   shift_del1 / shift_del2: the wrong token (and the one after it) left out: the draft's later tokens moved
#     one / two places EARLIER, counted from the token at the break on. Needs no guess of the right token.
#   shift_ins1: after the right token, the wrong token and the rest as they were: moved one place LATER (one
#     token was missing). Counted after the right token, as old_rest is.
SHIFT = ("shift_del1", "shift_del2", "shift_ins1")
MOST_ROUNDS_PER_FILE = 12000


class Restart:
    """The replay's plug-in: makes no second draft, measures the restarts of every round with a break."""

    def __init__(self, k=4, width=WIDTH, predictor=None, reanchor=False, more=False):
        self.inner, self.k, self.rows, self.width, self.predictor = SelfRedraft(), k, [], width, predictor
        self.arms = ARMS + (("predicted",) if predictor is not None else ())
        self.columns = COLUMNS + (("predicted",) if predictor is not None else ())
        self.reanchor = () if not reanchor else REANCHOR + (("reanchor_predicted",) if predictor is not None else ())
        if reanchor:
            self.columns = self.columns + self.reanchor + ("rank",)
        self.more = () if not more else MORE + (("snap",) if predictor is not None else ())
        if more:
            self.arms = self.arms + self.more
            self.columns = self.columns + self.more + ("echo_level",) + SHIFT

    def begin(self, ids, prompt_length, features, repetition_from, answer, seed=0):
        self.ids, self.prompt_length, self.features = ids, prompt_length, features
        self.repetition_from, self.answer = repetition_from, answer
        self.generator = torch.Generator().manual_seed(1000 * seed + answer)
        self.spread = features[0, prompt_length:].float().std(dim=0)
        self.tokens, self.index, self.indexed, self.projected = ids[0].tolist(), {}, 1, None

    def earlier_place(self, place):
        """(an earlier place where the target read the token of `place`, how alike its surroundings are)"""
        before, token, following = self.tokens[place - 1], self.tokens[place], self.tokens[place + 1]
        for level, key in ((4, (before, token, following)), (3, ("then", token, following)), (2, ("after", before, token)),
                           (1, ("alone", token))):
            found = self.index.get(key)
            if found is not None:
                return found, level
        return None, 0

    def start(self, first_drafter):
        self.inner.start(first_drafter)

    def __call__(self, s):
        width, full = s.block_ids.shape[1], self.width
        where = s.start - self.prompt_length
        outside = self.repetition_from is not None and where >= self.repetition_from
        truth = self.ids[:, s.start + 1 : s.start + width]
        if width != full or truth.shape[1] != full - 1 or outside:
            return None
        broke = _matches(s.block_ids[:, 1:], truth)
        restart = s.start + broke + 1                              # the position of the token the first draft got wrong
        if broke == full - 1 or restart + full > self.ids.shape[1]:
            return None
        device = s.block_ids.device
        token = truth[0, broke]
        after = self.ids[:, restart + 1 : restart + full]         # what the answer has after that token
        first = s.draft_logits[0].float()
        confidence = first.softmax(dim=-1).amax(dim=-1)
        likely = break_scores(confidence)[: full - 1].topk(self.k).indices.tolist()
        record = {name: -1 for name in self.columns}
        record.update(answer=self.answer, where=where, broke_after=broke, confidence=round(1000 * float(confidence[broke])),
                      second_choice_right=int(first[broke].topk(2).indices[1] == token), covered=int(broke in likely))

        # After the right token: the first draft's old tokens, and a second pass inside the block.
        correct = s.block_ids.clone()
        correct[0, broke + 1] = token
        rest = slice(broke + 2, full)
        record["old_rest"] = _matches(correct[:, rest], after)
        made, _ = self.inner.redraft_together(s, [broke + 1], correct)
        record["new_rest"] = _matches(made[:, rest], after)
        if "shift_del1" in record:
            from_the_break = self.ids[:, restart : restart + full]
            record["shift_del1"] = _matches(s.block_ids[:, broke + 2 :], from_the_break)
            record["shift_del2"] = _matches(s.block_ids[:, broke + 3 :], from_the_break)
            record["shift_ins1"] = _matches(s.block_ids[:, broke + 1 :], from_the_break[:, 1:])

        missing = broke + 1                                        # tokens of this round the target has not read
        states = self.features[:, s.start : s.start + missing]
        places = torch.arange(s.start, s.start + missing, device=device)
        before = self.features[:, max(s.start - 256, 0) : s.start]

        def noisy(level):
            noise = torch.randn(states.shape, generator=self.generator).to(device) * self.spread * (level / 100)
            return (states.float() + noise).to(states.dtype)

        stand_ins = {
            "real": (states, places),
            "none": None,
            "none_shift": None,
            "last": (before[:, -1:].repeat(1, missing, 1), places),
            "stale": (self.features[:, s.start - missing : s.start], places),
            "mean": (before.float().mean(dim=1, keepdim=True).to(states.dtype).repeat(1, missing, 1), places),
            "only_nearest": (states[:, -1:], places[-1:]),
            "all_but_nearest": (states[:, :-1], places[:-1]) if missing > 1 else None,
        }
        stand_ins.update({f"noisy{level}": (noisy(level), places) for level in NOISE})
        if self.predictor is not None:
            stand_ins["predicted"] = (predicted_states(self.predictor, s, missing, coming=token), places, True)
        if self.more:
            size = states.shape[-1] // len(s.first_drafter.target_layer_ids)      # one target layer's part of a state
            old = before[:, -1:].repeat(1, missing, 1)
            for count in (1, 2, 3, 4):
                mixed = old.clone()
                mixed[..., : count * size] = states[..., : count * size]
                stand_ins[f"shallow{count}"] = (mixed, places)
            for count in (1, 2):
                mixed = old.clone()
                mixed[..., -count * size :] = states[..., -count * size :]
                stand_ins[f"deep{count}"] = (mixed, places)
            while self.indexed < s.start:                           # places the target has read, with the token after each
                at = self.indexed
                before_it, it, after_it = self.tokens[at - 1], self.tokens[at], self.tokens[at + 1]
                for key in ((before_it, it, after_it), ("then", it, after_it), ("after", before_it, it), ("alone", it)):
                    self.index[key] = at
                self.indexed += 1
            found = [self.earlier_place(place) for place in range(s.start, s.start + missing)]
            hits = [(number, place) for number, (place, _) in enumerate(found) if place is not None]
            stand_ins["echo_states"] = None
            if hits:
                stand_ins["echo_states"] = (self.features[:, [place for _, place in hits]], places[[number for number, _ in hits]])
            stand_ins["echo_nearest"] = (self.features[:, [found[-1][0]]], places[-1:]) if found[-1][0] is not None else None
            record["echo_level"] = found[-1][1]
            if "snap" in self.more:
                if self.projected is None:
                    self.projected = torch.nn.functional.normalize(project(s.first_drafter, self.features)[0].float(), dim=-1)
                asked = torch.nn.functional.normalize(stand_ins["predicted"][0][0].float(), dim=-1)
                nearest = (asked @ self.projected[: s.start].T).argmax(dim=-1)
                stand_ins["snap"] = (self.features[:, nearest], places)
        begin = [s.start if name == "none_shift" else restart for name in self.arms]
        anchors, given = [token] * len(self.arms), [stand_ins[name] for name in self.arms]
        if self.reanchor:
            # A block from the last right token (the round's first token when the draft broke at once):
            # only the states of the tokens BEFORE that one are missing.
            earlier = {"reanchor": (states[:, :-1], places[:-1]), "reanchor_noisy100": (noisy(100)[:, :-1], places[:-1])}
            if self.predictor is not None:
                earlier["reanchor_predicted"] = (stand_ins["predicted"][0][:, :-1], places[:-1], True)
            for name in self.reanchor:
                anchors.append(s.block_ids[0, broke])
                begin.append(restart - 1)
                given.append(earlier[name] if missing > 1 else None)
            record["rank"] = min(int((first[broke] > first[broke, token]).sum()), 99)
        tokens = restarts(s, anchors, begin, given, full)
        for number, name in enumerate(self.arms):
            record[name] = _matches(tokens[number : number + 1], after)
        from_the_break = self.ids[:, restart : restart + full - 1]
        for number, name in enumerate(self.reanchor, start=len(self.arms)):
            record[name] = _matches(tokens[number : number + 1], from_the_break)
        self.rows.append([record[name] for name in self.columns])
        return None


def summary(table, columns=COLUMNS):
    """Tokens accepted after the right token at the break, for each way of restarting: over all
    rounds with a break and by how many of the target's states were missing."""
    COLUMNS = tuple(columns)
    ARMS = tuple(name for name in COLUMNS[COLUMNS.index("new_rest") + 1 :] if name not in ("rank", "echo_level"))
    table = numpy.asarray(table).reshape(-1, len(COLUMNS))
    out = {"rounds": int(len(table))}
    if not len(table):
        return out
    missing = table[:, COLUMNS.index("broke_after")] + 1
    groups = (("all", missing > 0), ("1 missing", missing == 1), ("2 missing", missing == 2),
              ("3-4 missing", (missing >= 3) & (missing <= 4)), ("5 or more missing", missing >= 5))
    for label, mask in groups:
        if mask.any():
            out[label] = {"rounds": int(mask.sum()),
                          **{name: round(float(table[mask, COLUMNS.index(name)].mean()), 3) for name in ("old_rest", "new_rest") + ARMS}}
    out["second_choice_right"] = round(float(table[:, COLUMNS.index("second_choice_right")].mean()), 3)
    if "echo_level" in COLUMNS:
        # By how alike the earlier place is whose state stands in for the last missing token's.
        level = table[:, COLUMNS.index("echo_level")]
        names = {0: "no earlier place", 1: "the same token", 2: "after the same token", 3: "before the same next token",
                 4: "both"}
        out["by_earlier_place"] = {
            names[value]: {"rounds": int((level == value).sum()),
                           **{name: round(float(table[level == value, COLUMNS.index(name)].mean()), 3)
                              for name in ("real", "none", "only_nearest", "echo_nearest", "echo_states") if name in COLUMNS}}
            for value in range(5) if (level == value).any()}
    if "shift_del1" in COLUMNS:
        # Tokens accepted from the token at the break on, with no second run of the drafter: the swap (the
        # second choice, then the old tokens in place), the swap with the old tokens also one place later,
        # the old tokens one or two places earlier (no guess), and the best of all of these.
        column = {name: table[:, COLUMNS.index(name)] for name in COLUMNS}
        for label, mask in groups:
            if not mask.any():
                continue
            right = column["second_choice_right"][mask]
            swap = right * (1 + column["old_rest"][mask])
            later = right * (1 + numpy.maximum(column["old_rest"][mask], column["shift_ins1"][mask]))
            earlier = numpy.maximum(column["shift_del1"][mask], column["shift_del2"][mask])
            out[label]["shifted_first_draft"] = {
                "the_swap": round(float(swap.mean()), 3), "the_swap_or_one_place_later": round(float(later.mean()), 3),
                "one_or_two_places_earlier": round(float(earlier.mean()), 3),
                "the_best_of_all": round(float(numpy.maximum(later, earlier).mean()), 3),
                "earlier_gets_two_or_more": round(float((earlier >= 2).mean()), 3),
                "later_beats_in_place": round(float((column["shift_ins1"][mask] > column["old_rest"][mask]).mean()), 3)}
    if "reanchor" in COLUMNS:
        # Tokens accepted from the token at the break on: a restart from the second choice (nothing
        # unless that choice is right), a block from the last right token, and the better of the two.
        column = {name: table[:, COLUMNS.index(name)] for name in COLUMNS}
        for label, mask in groups:
            if not mask.any():
                continue
            compared = {}
            for states, again in (("real", "reanchor"), ("noisy100", "reanchor_noisy100"), ("predicted", "reanchor_predicted")):
                if states in column and again in column:
                    second = column["second_choice_right"][mask] * (1 + column[states][mask])
                    compared[states] = {"from_the_second_choice": round(float(second.mean()), 3),
                                        "from_the_last_right_token": round(float(column[again][mask].mean()), 3),
                                        "the_better_of_the_two": round(float(numpy.maximum(second, column[again][mask]).mean()), 3),
                                        "the_last_right_token_gets_the_break_right": round(float((column[again][mask] > 0).mean()), 3)}
            out[label]["from_the_break_on"] = compared
        rank = column["rank"]
        out["right_token_is_the_drafters"] = {"second_choice": round(float((rank == 1).mean()), 3),
                                              "third_choice": round(float((rank == 2).mean()), 3),
                                              "fourth_choice": round(float((rank == 3).mean()), 3),
                                              "fifth_to_tenth": round(float(((rank >= 4) & (rank <= 9)).mean()), 3)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation", help="train, validation, test, or all (every answer in the folder)")
    parser.add_argument("--fixed", default="", help="start from the saved add-on weights lora-<name>.pt")
    parser.add_argument("--sharp", action="store_true", help="sharper attention on long texts (SharpDraft)")
    parser.add_argument("--predictor", default="", help="a saved predictor of the missing states: restart-<name>.pt in --adapter-dir")
    parser.add_argument("--reanchor", action="store_true", help="also restart from the last right token (see REANCHOR above)")
    parser.add_argument("--more", action="store_true", help="also stand-ins that are real states of other places or layers (see MORE above)")
    parser.add_argument("--rank", type=int, default=16, help="of the saved add-on weights")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens (for a quick check)")
    parser.add_argument("--max-answers", type=int, default=0, help="stop after this many answers (for a quick check)")
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir)
               if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    answers = answers[args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_in_this_job": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    adapters = attach_adapters(draft, rank=args.rank) if args.fixed else []
    for adapter in adapters:
        adapter.fast = True
    notes, state = {}, None
    if args.fixed:
        path = os.path.join(args.adapter_dir, f"lora-{args.fixed}.pt")
        state, notes = load_adapters(path, adapters), adapter_notes(path)
    first = DraftTrainer(draft, adapters, rate=0.0, init=state, sharpen=QueryScale() if args.sharp else None)
    predictor, predictor_notes = None, {}
    if args.predictor:
        predictor, predictor_notes = load_predictor(os.path.join(args.adapter_dir, f"restart-{args.predictor}.pt"), device)
    probe = Restart(predictor=predictor, reanchor=args.reanchor, more=args.more)
    records, written, started = [], [0, 0], time.time()          # written: files, rounds in them

    def report(done):
        fresh = probe.rows[written[1] :]
        if fresh and (done or len(fresh) >= MOST_ROUNDS_PER_FILE):
            numpy.savez_compressed(os.path.join(args.results_dir, f"rounds-{written[0]}.npz"),
                                   rounds=numpy.asarray(fresh, dtype=numpy.int32), columns=numpy.asarray(probe.columns))
            written[0] += 1
            written[1] += len(fresh)
        status = {"done": done, "answers": len(records), "answers_in_this_job": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "draft": args.draft, "fixed": args.fixed,
                  "sharp": args.sharp, "saved_adapters_used": notes, "columns": probe.columns,
                  "target_layers_the_drafter_reads": list(draft.target_layer_ids), "files": written[0],
                  "predictor": args.predictor, "predictor_notes": predictor_notes,
                  "answers_read": records, "summary": summary(probe.rows, probe.columns)}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and len(records) >= args.max_answers):
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2 or (args.longest and len(generated) > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        repetition_from = None if ended else loop_start(generated)
        features = target_features(target, draft, ids)
        first.begin()
        probe.begin(ids, prompt_length, features, repetition_from, len(records))
        result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=probe,
                               block_size=WIDTH)
        probe.features = None
        del features
        records.append({"source": source, "origin": [name, index], "tokens": len(generated), "rounds": int(result.rounds),
                        "ended_by_itself": ended, "repetition_from": repetition_from})
        report(False)
    report(True)


if __name__ == "__main__":
    main()
