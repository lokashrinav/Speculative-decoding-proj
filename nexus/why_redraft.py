"""Why does a second draft help? Every candidate reason, tried on the same rounds.

Each saved answer is replayed with the first draft alone, so the rounds are plain decoding's own.
In every round the first draft breaks somewhere: the target accepts its first b tokens and rejects
token b + 1 (known here from the saved answer). At that same round, second drafts of many kinds are
made, and each is scored by how many tokens the target would accept from it. Because every kind
is tried on the same rounds, their differences are differences in cause, not in luck.

Second drafts from the TRUE break (so that placement plays no part). With the same, unchanged drafter:
  again          shown its first draft up to the break, it fills in the rest
  first_token    shown only the first token of its draft (the second draft we tried in October)
  random         the same pass, but shown random tokens of the answer instead of its draft
  elsewhere      the same pass, but shown a piece of text from another place in the answer
                 (random and elsewhere give the drafter a second look without any true information:
                 what they gain is the gain of merely guessing again)
  switch         NO second pass: the first draft with the drafter's second choice at the break
  early          the second choice at the break is shown, and a pass fills in the rest knowing it
  late           the pass of "again", taking its best token other than the first draft's at the break
  right_token_old_rest   the RIGHT token at the break, then the first draft's old tokens (no pass)
  right_token_new_rest   the right token at the break is shown, and a pass fills in the rest
  next_round     what the drafter drafts after the target has read the right token (plain decoding's
                 next round): the same continuation as the two above, but from the target's states
With saved second drafters (trained add-ons), if given:
  second, second_late, second_random   as again, late and random, by the drafter trained to read a prefix
  second_blind   that drafter shown nothing of the first draft
  control        the add-on trained without ever seeing a draft, shown nothing
  edit           the add-on trained to correct a whole first draft, shown all of it
Second drafts from the four most LIKELY breaks, as decoding would place them (best of the four):
  four, four_random, four_switch, four_early, four_late

Also noted for every break: how sure the drafter and the target were, where the right token stood in
the drafter's ranking (first pass, and each second pass), and the kind of error.

The rounds are saved as tables (rounds-N.npz, COLUMNS) and a sample of rounds as text
(examples-N.json) to read. nexus/why_report.py turns the tables into the comparison.
"""

import argparse
import json
import os
import time

import numpy
import torch

from dflash.adapt import DraftTrainer, adapter_notes, attach_adapters, load_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.redraft import SelfRedraft, break_scores
from dflash.replay import _matches, replay_decode, target_features
from dflash.sharp import QueryScale
from nexus import bench_redraft
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers

ARMS = ("again", "first_token", "random", "elsewhere", "switch", "early", "late", "right_token_old_rest",
        "right_token_new_rest", "next_round", "second", "second_late", "second_random", "second_blind", "control", "edit",
        "four", "four_random", "four_switch", "four_early", "four_late")
RANKS = ("rank_first", "rank_again", "rank_random", "rank_elsewhere", "rank_second", "rank_second_random",
         "rank_second_blind", "rank_control")
COLUMNS = ("answer", "where", "broke_after", "confidence", "target_confidence", "error", "p_first", "p_again",
           "covered", "covered_with_zero") + RANKS + ARMS
MOST_ROUNDS_PER_FILE = 12000
WIDTH = 16


class Why:
    """The plug-in that makes all the second drafts of a round. It never changes the decoding."""

    def __init__(self, second=None, control=None, tokenizer=None, examples_every=0, k=4, edit=None):
        self.inner = SelfRedraft(keep=1)
        self.second = None if second is None else SelfRedraft(model=second, keep=1)
        self.control = None if control is None else SelfRedraft(model=control, keep=1)
        self.edit = None if edit is None else SelfRedraft(model=edit, keep=1, show_rest=True)
        self.tokenizer, self.examples_every, self.k = tokenizer, examples_every, k
        self.rows, self.examples, self.breaks_seen = [], [], 0

    def begin(self, ids, prompt_length, predictions, repetition_from, answer, seed=0):
        self.ids, self.prompt_length, self.predictions = ids, prompt_length, predictions
        self.repetition_from, self.answer = repetition_from, answer
        self.generator = torch.Generator().manual_seed(1000 * seed + answer)
        self.last = None            # (row, round start, tokens accepted) of the round before this one

    def start(self, first_drafter):
        self.inner.start(first_drafter)
        for other in (self.second, self.control, self.edit):
            if other is not None:
                other.start(first_drafter)

    def _skip(self, s):
        for other in (self.second, self.control, self.edit):
            if other is not None:
                other.skip(s)

    def _text_from_the_answer(self, count, device, together):
        """`count` tokens of the answer: single ones picked at random, or one piece of it."""
        low, high = self.prompt_length, self.ids.shape[1]
        if together:
            begin = int(torch.randint(low, max(high - count, low + 1), (1,), generator=self.generator))
            return self.ids[0, begin : begin + count]
        return self.ids[0, torch.randint(low, high, (count,), generator=self.generator).to(device)]

    def __call__(self, s):
        width = s.block_ids.shape[1]
        truth = self.ids[:, s.start + 1 : s.start + width]
        where = s.start - self.prompt_length
        outside = self.repetition_from is not None and where >= self.repetition_from
        if width != WIDTH or truth.shape[1] != WIDTH - 1 or outside:
            self._skip(s)
            self.last = None
            return None
        device = s.block_ids.device
        draft, right = s.block_ids, truth[0]
        broke = _matches(draft[:, 1:], truth)                    # tokens accepted; token broke + 1 is rejected
        if self.last is not None and s.start == self.last[1] + self.last[2] + 1:
            row, _, before = self.last
            self.rows[row][COLUMNS.index("next_round")] = before + 1 + min(broke, WIDTH - 2 - before)
        record = {name: -1 for name in COLUMNS}
        record.update(answer=self.answer, where=where, broke_after=broke)
        self.rows.append(None)
        number = len(self.rows) - 1
        if broke == WIDTH - 1:                                   # nothing to fix: the whole draft is accepted
            self._skip(s)
            self.rows[number] = [record[name] for name in COLUMNS]
            self.last = (number, s.start, broke)
            return None

        def accepted(row):
            return _matches(row.reshape(1, -1)[:, 1:], truth)

        def rank(logits):
            return min(int((logits > logits[token]).sum()) + 1, 30000)

        def with_shown(tokens, upto):
            block = draft.clone()
            block[0, 1 : 1 + upto] = tokens
            return block

        def filled(logits, keep, other_at_break=False):
            """The first draft up to `keep`, then this pass's tokens (its best other token right after
            `keep` if asked)."""
            out = draft.clone()
            out[0, keep + 1 :] = logits.argmax(dim=-1)[keep:]
            if other_at_break:
                best = logits[keep].topk(2).indices
                out[0, keep + 1] = best[1] if best[0] == draft[0, keep + 1] else best[0]
            return out

        token = right[broke]                                     # the right token at the break
        first_logits = s.draft_logits[0].float()
        probabilities = first_logits.softmax(dim=-1)
        confidence = probabilities.amax(dim=-1)
        second_choice = first_logits.topk(2, dim=-1).indices[:, 1]
        wrong = draft[0, broke + 1]
        record.update(
            confidence=round(1000 * float(confidence[broke])), p_first=round(1000 * float(probabilities[broke, token])),
            rank_first=rank(first_logits[broke]),
            error=1 if wrong == draft[0, broke] else 2 if broke + 1 < WIDTH - 1 and wrong == right[broke + 1] else 0)
        if self.predictions is not None:
            record["target_confidence"] = round(1000 * float(self.predictions[1][s.start + broke, 0].exp()))
        scores = break_scores(confidence)
        likely = (scores[1 : WIDTH - 1].topk(self.k).indices + 1).tolist()          # the drafter itself needs one token
        likely_zero = scores[: WIDTH - 1].topk(self.k).indices.tolist()             # with a switch, none
        record.update(covered=int(broke in likely), covered_with_zero=int(broke in likely_zero))

        # One pass of the unchanged drafter makes every second draft of this round, one per row.
        rows, blocks, keeps = {}, [], []

        def ask(name, block, keep):
            if name not in rows:
                rows[name] = len(keeps)
                blocks.append(block)
                keeps.append(keep)

        def switched(place):
            block = draft.clone()
            block[0, place + 1] = second_choice[place]
            return block

        for place in sorted(set(likely) | {i for i in likely_zero if i > 0} | ({broke, 1} if broke else set())):
            ask(("plain", place), draft, place)
        for place in likely + ([broke] if broke else []):
            ask(("random", place), with_shown(self._text_from_the_answer(place, device, False), place), place)
        if broke:
            ask(("elsewhere", broke), with_shown(self._text_from_the_answer(broke, device, True), broke), broke)
        for place in set(likely_zero) | {broke}:
            ask(("early", place), switched(place), place + 1)
        correct = draft.clone()
        correct[0, broke + 1] = token
        ask("right", correct, broke + 1)
        made, logits = self.inner.redraft_together(s, keeps, torch.cat(blocks, dim=0))
        logits = logits.float()

        def plain(place, other_at_break=False):
            return filled(logits[rows[("plain", place)]], place, other_at_break)

        def late(place):
            return plain(place, True) if place else made[rows[("early", 0)]]

        if broke:
            again = logits[rows[("plain", broke)]]
            record.update(
                again=accepted(plain(broke)), first_token=accepted(plain(1)), rank_again=rank(again[broke]),
                p_again=round(1000 * float(again[broke].softmax(dim=-1)[token])),
                random=accepted(filled(logits[rows[("random", broke)]], broke)),
                rank_random=rank(logits[rows[("random", broke)]][broke]),
                elsewhere=accepted(filled(logits[rows[("elsewhere", broke)]], broke)),
                rank_elsewhere=rank(logits[rows[("elsewhere", broke)]][broke]))
        record.update(
            switch=accepted(switched(broke)), early=accepted(made[rows[("early", broke)]]), late=accepted(late(broke)),
            right_token_old_rest=accepted(correct), right_token_new_rest=accepted(made[rows["right"]]),
            four=max(accepted(plain(place)) for place in likely),
            four_random=max(accepted(filled(logits[rows[("random", place)]], place)) for place in likely),
            four_switch=max(accepted(switched(place)) for place in likely_zero),
            four_early=max(accepted(made[rows[("early", place)]]) for place in likely_zero),
            four_late=max(accepted(late(place)) for place in likely_zero))

        shown = {"first": draft[0], "right": truth[0], "early": made[rows[("early", broke)]],
                 "right_token_new_rest": made[rows["right"]]}
        if broke:
            shown["again"] = plain(broke)[0]
            shown["random"] = filled(logits[rows[("random", broke)]], broke)[0]
        if self.second is not None:
            noise = with_shown(self._text_from_the_answer(broke, device, False), broke) if broke else draft
            _, theirs = self.second.redraft_together(s, [broke, broke, 0], torch.cat((draft, noise, draft), dim=0))
            theirs = theirs.float()
            blind = filled(theirs[2], 0)
            record.update(
                second=accepted(filled(theirs[0], broke)), second_late=accepted(filled(theirs[0], broke, True)),
                second_random=accepted(filled(theirs[1], broke)), second_blind=accepted(blind),
                rank_second=rank(theirs[0][broke]), rank_second_random=rank(theirs[1][broke]),
                rank_second_blind=rank(theirs[2][broke]))
            shown["second"], shown["second_blind"] = filled(theirs[0], broke)[0], blind[0]
        if self.control is not None:
            _, theirs = self.control.redraft_together(s, [0], draft)
            theirs = theirs.float()
            record.update(control=accepted(filled(theirs[0], 0)), rank_control=rank(theirs[0][broke]))
            shown["control"] = filled(theirs[0], 0)[0]
        if self.edit is not None:
            edited = self.edit.redraft(s, 1)
            record.update(edit=accepted(edited))
            shown["edit"] = edited[0]

        self.rows[number] = [record[name] for name in COLUMNS]
        self.last = (number, s.start, broke)
        self.breaks_seen += 1
        if self.examples_every and self.tokenizer is not None and self.breaks_seen % self.examples_every == 0:
            def words(tokens):
                return [self.tokenizer.decode([int(value)]) for value in tokens.reshape(-1)[-(WIDTH - 1):]]

            self.examples.append({
                "answer": self.answer, "where": where, "broke_after": broke,
                "text_before": self.tokenizer.decode(self.ids[0, max(s.start - 40, 0) : s.start + 1].tolist()),
                "confidence": [round(float(value), 2) for value in confidence],
                "rank_of_right_token": record["rank_first"], "target_confidence": record["target_confidence"] / 1000,
                "likely_breaks": likely_zero, "drafts": {name: words(tokens) for name, tokens in shown.items()},
                "accepted": {name: record[name] for name in ARMS if record[name] >= 0}})
        return None


def summary(table):
    """For a quick look: of the rounds in which the first draft broke, how often each kind of second
    draft got past the break and how many tokens it added per such round."""
    table = numpy.asarray(table).reshape(-1, len(COLUMNS))
    broke = table[:, COLUMNS.index("broke_after")]
    rounds = table[broke < WIDTH - 1]
    out = {"rounds": int(len(table)), "rounds_in_which_the_first_draft_broke": int(len(rounds)),
           "tokens_accepted_from_the_first_draft": float(broke.mean()) if len(table) else 0.0, "second_drafts": {}}
    for name in ARMS:
        column, before = rounds[:, COLUMNS.index(name)], rounds[:, COLUMNS.index("broke_after")]
        tried = column >= 0
        if tried.any():
            gain = numpy.maximum(column[tried] - before[tried], 0)
            out["second_drafts"][name] = {
                "rounds_tried": int(tried.sum()), "got_past_the_break": float((gain > 0).mean()),
                "tokens_added_per_round_tried": float(gain.mean()),
                "tokens_added_per_round_with_a_break": float(gain.sum() / len(rounds))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation", help="train, validation, test, or all (every answer in the folder)")
    parser.add_argument("--second", default="", help="a saved second drafter trained to read a prefix (in --ckpt-dir)")
    parser.add_argument("--control", default="", help="a saved second drafter trained without seeing a draft")
    parser.add_argument("--edit", default="", help="a saved second drafter trained to correct a whole first draft")
    parser.add_argument("--fixed", default="", help="start from the saved add-on weights lora-<name>.pt")
    parser.add_argument("--sharp", action="store_true", help="sharper attention on long texts (SharpDraft)")
    parser.add_argument("--rank", type=int, default=16, help="of the saved add-on weights")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--ckpt-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--examples-every", type=int, default=100, help="save every Nth break as text (0: none)")
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
    bench_redraft._draft_class = type(draft)

    def saved(name):
        path = os.path.expanduser(name)
        model = bench_redraft.load_second(path if os.path.isdir(path) else os.path.join(args.ckpt_dir, name))
        return model.to(draft.fc.weight)

    second = saved(args.second) if args.second else None
    control = saved(args.control) if args.control else None
    edit = saved(args.edit) if args.edit else None
    adapters = attach_adapters(draft, rank=args.rank) if args.fixed else []
    for adapter in adapters:
        adapter.fast = True
    notes = {}
    state = None
    if args.fixed:
        path = os.path.join(args.adapter_dir, f"lora-{args.fixed}.pt")
        state, notes = load_adapters(path, adapters), adapter_notes(path)
    first = DraftTrainer(draft, adapters, rate=0.0, init=state, sharpen=QueryScale() if args.sharp else None)
    probe = Why(second, control, tokenizer, args.examples_every, edit=edit)
    records, written, started = [], [0, 0], time.time()          # written: files, rounds in them

    def report(done):
        fresh = probe.rows[written[1] :]
        if fresh and (done or len(fresh) >= MOST_ROUNDS_PER_FILE):
            # The last round's "next round" is not known yet: it stays in memory until the following save.
            keep_back = 0 if done else 1
            chunk = fresh[: len(fresh) - keep_back]
            numpy.savez_compressed(os.path.join(args.results_dir, f"rounds-{written[0]}.npz"),
                                   rounds=numpy.asarray(chunk, dtype=numpy.int32), columns=numpy.asarray(COLUMNS))
            write_json(os.path.join(args.results_dir, f"examples-{written[0]}.json"), probe.examples, separators=(",", ":"))
            probe.examples = []
            written[0] += 1
            written[1] += len(chunk)
        status = {"done": done, "answers": len(records), "answers_in_this_job": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "draft": args.draft, "second": args.second,
                  "control": args.control, "edit": args.edit, "fixed": args.fixed, "sharp": args.sharp, "saved_adapters_used": notes,
                  "columns": COLUMNS, "files": written[0], "answers_read": records, "summary": summary(probe.rows)}
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
        features, predictions = target_features(target, draft, ids, top=2)
        first.begin()
        probe.begin(ids, prompt_length, predictions, repetition_from, len(records))
        result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=probe,
                               block_size=WIDTH)
        del features, predictions
        records.append({"source": source, "origin": [name, index], "tokens": len(generated), "rounds": int(result.rounds),
                        "ended_by_itself": ended, "repetition_from": repetition_from})
        report(False)
    report(True)


if __name__ == "__main__":
    main()
