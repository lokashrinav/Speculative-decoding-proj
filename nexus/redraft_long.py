"""Running the drafter more than once in a round, on long saved answers.

Each saved answer is replayed once per method (dflash/replay.py: only the drafter runs, and a draft
counts as far as it equals the answer the target wrote). The first draft is always kept. A method
adds one or more redrafts that use the first draft, the target checks them all together, and the
draft it accepts furthest counts. So a method can only add accepted tokens to a round; what it
costs is its extra drafter pass, which is timed here.

Methods (comma-separated):
  plain                  the first draft alone
  self-keepN             the same drafter once more, shown its first N draft tokens
  self-keepconfidence    ... shown its draft up to the most likely break
  branchesK              K redrafts in one pass, from the K most likely breaks of the first draft
                         (how D2SD places them, arXiv 2606.04446)
  branchesKl, branchesKe the same, and each redraft is made to differ from the first draft right at
                         its break: l(ate) takes its best other token there, e(arly) is shown the first
                         draft's second choice there before the pass (dflash/redraft.py: Branches)
  branchesKs             NO second pass: the first draft with its second choice at each of the K likely
                         breaks and nothing else changed. The control for branchesKe: what the
                         redraft pass adds on top of the switch (truebreaks: the same at the true break)
  branchesKstN           the same, but the target checks only the switched token and the N tokens
                         after it (a shorter check; any branchesK method can end in tN)
  chainD                 D drafts in all, each redraft from the likely break of the one before it
  truebreak, truebreakl, truebreake, truebreaks
                         one redraft from where the target really rejects the first draft. Decoding
                         cannot know that place: this is the most that finding the break could give.
  ckpt:NAME:MODE         a saved second drafter (nexus/train_redraft.py), MODE keep0, keep1 or edit
A method that redrafts can end in @NAME: the redrafts are then made by the saved second drafter
NAME (a folder in --ckpt-dir) instead of by the first drafter itself.
To any method +sharp and +fixed@SAVED can be added. They change the first drafter for the whole
answer: sharper attention on long texts (SharpDraft), and the saved add-on weights lora-SAVED.pt
(trained ahead of time across other answers). A redraft by the first drafter then has them too.

Besides tokens per round, the time of a round is put together from measured parts, so that
methods can be compared in real time before any of them is built into decoding: the target's
check and the first draft as measured by text length on this GPU type (nexus/cost_check.py),
plus the extra drafter pass as timed in this run. It assumes that checking the redrafts next to
the first draft costs no more than checking the first draft alone.
"""

import argparse
import os
import re
import time

import torch

from dflash.adapt import DraftTrainer, adapter_notes, attach_adapters, load_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.model import _crop_to, _draft_value, _make_cache, _output_head, _raw_input_embeddings, extract_context_feature
from dflash.redraft import Branches, Chain, SelfRedraft, lowest_break, redrafts_from
from dflash.replay import _checked_matches, _clock, _matches, replay_decode, target_features
from dflash.sharp import QueryScale
from nexus import bench_redraft
from nexus.adapt_drafter import GROUPS, by_position, settings_for, summarize
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.replay_memory import load_answers

DEFAULT = "plain,self-keep1,self-keepconfidence,branches4,branches4l,branches4e,chain3,truebreak"

# Milliseconds on an RTX A5000 by how many tokens the text already has (nexus/results/adapt-amount-0):
# the target's check of one block of 16, and one pass of the public drafter.
MEASURED_AT = (1000, 2000, 4000, 8000, 16000, 24000, 32000)
CHECK_MS = (48.2, 48.3, 48.6, 55.2, 97.7, 137.0, 176.8)
DRAFT_MS = (8.11, 8.20, 8.30, 8.29, 8.56, 9.49, 11.04)
COUNTS = ("rounds", "first_draft_broke", "break_was_among_the_redrafts", "redraft_got_past_the_break",
          "a_redraft_got_further", "tokens_added")


def measured(table, tokens):
    """A measured time at a text length, in a straight line between the lengths it was measured at."""
    if tokens <= MEASURED_AT[0]:
        return table[0]
    for number in range(1, len(MEASURED_AT)):
        if tokens <= MEASURED_AT[number]:
            low, high = MEASURED_AT[number - 1], MEASURED_AT[number]
            return table[number - 1] + (table[number] - table[number - 1]) * (tokens - low) / (high - low)
    return table[-1]


def check_and_draft_ms(acceptance_lengths, prompt_length, repetition_from):
    """Milliseconds of the target's checks and the first drafts of an answer's rounds, for each
    position group, on rounds outside endless repetition (the rounds by_position counts)."""
    table = [0.0 for _ in GROUPS]
    start = prompt_length
    for gained in acceptance_lengths:
        where = start - prompt_length
        if repetition_from is None or where < repetition_from:
            cost = measured(CHECK_MS, start) + measured(DRAFT_MS, start)
            for number, (_, low, high) in enumerate(GROUPS):
                if low <= where < high:
                    table[number] += cost
        start += gained
    return table


class TrueBreak:
    """One redraft from where the target really rejects the first draft, which is read from the
    saved answer. Decoding cannot do this: it shows the most that finding the break could give."""

    def __init__(self, model=None, differ=None):
        self.inner = SelfRedraft(model=model, keep=1)
        self.differ = differ
        self.answer = None      # (1, length): the prompt and the answer, set before the replay
        self.kept = []

    def start(self, first_drafter):
        self.inner.start(first_drafter)
        self.kept = []

    def __call__(self, s):
        width = s.block_ids.shape[1]
        own = _matches(s.block_ids[:, 1:], self.answer[:, s.start + 1 : s.start + width])
        if not lowest_break(self.inner, self.differ) <= own <= width - 2:
            self.kept = []
            self.inner.skip(s)
            return None
        self.kept = [own]
        return redrafts_from(self.inner, s, self.kept, self.differ)


def make_method(name, ckpt_dir, like=None):
    """The plug-in of a method name (None for "plain"). like: the first drafter, whose number
    type and device a saved second drafter takes."""
    def saved(folder):
        path = os.path.expanduser(folder)
        model = bench_redraft.load_second(path if os.path.isdir(path) else os.path.join(ckpt_dir, folder))
        return model if like is None else model.to(like.fc.weight)     # not an add-on's weights: they are 32-bit

    if name == "plain":
        return None
    if name.startswith("ckpt:"):
        _, folder, mode = name.split(":")
        model = saved(folder)
        if mode == "edit":
            return SelfRedraft(model=model, keep=1, show_rest=True)
        return SelfRedraft(model=model, keep=int(mode.removeprefix("keep")))
    body, _, second = name.partition("@")
    model = saved(second) if second else None
    found = re.fullmatch(r"self-keep(\d+|confidence)", body)
    if found:
        return SelfRedraft(model=model, keep=found[1] if found[1] == "confidence" else int(found[1]))
    kinds = {"": None, "l": "late", "e": "early", "s": "switch"}
    found = re.fullmatch(r"branches(\d+)([les]?)(?:t(\d+))?", body)
    if found:
        return Branches(k=int(found[1]), model=model, differ=kinds[found[2]], tail=None if found[3] is None else int(found[3]))
    found = re.fullmatch(r"chain(\d+)", body)
    if found:
        return Chain(depth=int(found[1]), model=model)
    found = re.fullmatch(r"truebreak([les]?)", body)
    if found:
        return TrueBreak(model=model, differ=kinds[found[1]])
    raise ValueError(f"cannot read the method {name}")


def parse_setting(setting):
    """METHOD, then +sharp and +fixed@SAVED in any order."""
    method, *extras = setting.split("+")
    options = {"method": method, "sharpen": None, "fixed": ""}
    for extra in extras:
        if extra == "sharp":
            options["sharpen"] = QueryScale()
        elif extra.startswith("fixed@") and len(extra) > 6:
            options["fixed"] = extra[6:]
        else:
            raise ValueError(f"cannot read the setting {setting}")
    return options


class Watched:
    """Runs a method's plug-in, times it, and notes how its redrafts did against the saved answer.

    For each position group it adds up the seconds of the extra drafter work and, on full blocks,
    COUNTS: rounds; rounds in which the first draft was not accepted whole; of those, rounds in which
    the true break was one of the breaks the method redrafted from; of those, rounds in which that
    redraft got the next token right; rounds in which any redraft got further than the first draft;
    and the tokens that added."""

    def __init__(self, improver, ids, prompt_length, repetition_from, block=16):
        self.improver = improver
        self.alternative_tail = getattr(improver, "alternative_tail", None)     # the replay asks the plug-in it is given
        self.ids, self.prompt_length, self.repetition_from, self.block = ids, prompt_length, repetition_from, block
        self.seconds = [0.0 for _ in GROUPS]
        self.counts = [[0 for _ in COUNTS] for _ in GROUPS]
        if hasattr(improver, "answer"):
            improver.answer = ids

    def start(self, first_drafter):
        self.improver.start(first_drafter)

    def __call__(self, s):
        clock = _clock(s.block_ids.device)
        drafts = self.improver(s)
        seconds = _clock(s.block_ids.device) - clock
        where = s.start - self.prompt_length
        if self.repetition_from is not None and where >= self.repetition_from:
            return drafts
        width = s.block_ids.shape[1]
        add = [0 for _ in COUNTS]
        if width == self.block:
            truth = self.ids[:, s.start + 1 : s.start + width]
            own = _matches(s.block_ids[:, 1:], truth)
            got = [] if drafts is None else [_checked_matches(drafts[row : row + 1], s.block_ids, truth, self.alternative_tail)
                                             for row in range(drafts.shape[0])]
            kept = list(getattr(self.improver, "kept", []))
            add[0] = 1
            add[1] = int(own < width - 1)
            if add[1] and own in kept and len(got) >= len(kept):
                add[2] = 1
                add[3] = int(got[kept.index(own)] > own)
            best = max(got, default=own)
            add[4] = int(best > own)
            add[5] = max(best - own, 0)
        for number, (_, low, high) in enumerate(GROUPS):
            if low <= where < high:
                self.seconds[number] += seconds
                for index, value in enumerate(add):
                    self.counts[number][index] += value
        return drafts


def in_real_time(records, names):
    """For every method and position group: the tokens per second the measured parts add up to, how
    much faster that is than the first draft alone, the extra drafter work per round, and how the
    redrafts did (see Watched)."""
    out = {}
    for name in names:
        rows = {}
        for number, (group, _, _) in enumerate(GROUPS):
            mine = [record["settings"][name] for record in records if name in record["settings"]]
            rounds = sum(entry["table"][number][0] for entry in mine)
            tokens = sum(entry["table"][number][1] for entry in mine)
            if not rounds:
                continue
            base_ms = sum(entry["check_and_draft_ms"][number] for entry in mine)
            extra_ms = 1000 * sum(entry.get("redraft_seconds", [0.0] * len(GROUPS))[number] for entry in mine)
            row = {"tokens_per_second": 1000 * tokens / (base_ms + extra_ms),
                   "extra_drafter_ms_per_round": extra_ms / rounds,
                   "check_and_first_draft_ms_per_round": base_ms / rounds}
            plain = [record["settings"]["plain"] for record in records
                     if "plain" in record["settings"] and name in record["settings"]]
            plain_ms = sum(entry["check_and_draft_ms"][number] for entry in plain)
            if plain_ms:
                plain_speed = 1000 * sum(entry["table"][number][1] for entry in plain) / plain_ms
                row["faster_than_plain"] = row["tokens_per_second"] / plain_speed - 1
            counts = [sum(entry["breaks"][number][index] for entry in mine if "breaks" in entry) for index in range(len(COUNTS))]
            if counts[0]:
                row["redrafts"] = {
                    "rounds_counted": counts[0],
                    "first_draft_broke": counts[1] / counts[0],
                    "the_break_was_redrafted_from": counts[2] / max(counts[1], 1),
                    "that_redraft_got_past_the_break": counts[3] / max(counts[2], 1),
                    "a_redraft_got_further": counts[4] / counts[0],
                    "tokens_added_when_it_did": counts[5] / max(counts[4], 1)}
            rows[group] = row
        out[name] = rows
    return out


@torch.inference_mode()
def time_the_passes(target, draft, ids, features, lengths, repeats=24, skip=4, block=16):
    """Milliseconds of what a round is made of, with the text at several lengths: the first draft,
    one redraft, several redrafts in one pass (attending directly, and through the library's masked
    call), and the target's check of 16 tokens and of more (redrafts are checked in the same pass
    as the first draft, so the check then reads more tokens)."""
    from types import SimpleNamespace

    from nexus.cost_check import middle

    device = ids.device
    head = _output_head(target)
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    positions = torch.arange(ids.shape[1] + block, device=device)[None]
    cache, target_cache, held, rows = _make_cache(draft.config), _make_cache(target.config), 0, []

    def timed(work):
        times = []
        for _ in range(repeats):
            start = _clock(device)
            work()
            times.append(_clock(device) - start)
        return middle(times, skip)

    for length in sorted(lengths):
        if length + block > ids.shape[1]:
            break
        block_ids = torch.full((1, block), draft.mask_token_id, dtype=torch.long, device=device)
        block_ids[:, 0] = ids[:, length]
        noise = _raw_input_embeddings(target, block_ids, scale)

        def first(new=features[:, :0], begin=length):
            states = draft(target_hidden=new, noise_embedding=noise, position_ids=positions[:, begin : length + block],
                           past_key_values=cache, use_cache=True)[:, -block:, :]
            logits = draft.compute_logits(states[:, 1:, :], head)
            _crop_to(cache, length)
            return logits

        first(features[:, held:length], held)                        # the drafter reads the text up to here
        for begin in range(held, length, 1024):                      # ... and so does the target
            end = min(begin + 1024, length)
            target(ids[:, begin:end], position_ids=positions[:, begin:end], past_key_values=target_cache,
                   use_cache=True, logits_to_keep=1)
        held = length
        logits = first()
        block_ids[:, 1:] = logits.argmax(dim=-1)
        state = SimpleNamespace(block_ids=block_ids, draft_logits=logits, start=length, position_ids=positions,
                                context_features=features[:, length:length], accepted_ids=ids[:, :length],
                                first_drafter=draft, draft_cache=cache, target=target)
        inner = SelfRedraft(keep=1)
        inner.start(draft)
        row = {"tokens_before": length, "first_draft": timed(first), "one_redraft": timed(lambda: inner.redraft(state, 4))}
        for count in (2, 4, 8, 15):
            keeps = list(range(min(count, block - 1)))
            row[f"{count}_redrafts_in_one_pass"] = timed(lambda: inner.redraft_together(state, keeps))
            row[f"{count}_redrafts_in_one_pass_library_mask"] = timed(
                lambda: inner.redraft_together(state, keeps, no_copies=False))
        for tokens in (block, 2 * block, 3 * block, 5 * block, 8 * block):
            if length + tokens > ids.shape[1]:
                break

            def check():
                output = target(ids[:, length : length + tokens], position_ids=positions[:, length : length + tokens],
                                past_key_values=target_cache, use_cache=True, output_hidden_states=True)
                output.logits.argmax(dim=-1)
                extract_context_feature(output.hidden_states, draft.target_layer_ids)
                _crop_to(target_cache, length)

            row[f"check_of_{tokens}_tokens"] = timed(check)
        rows.append(row)
    return rows


def find_drafter(name):
    """"opd" is the better drafter (Draft-OPD). It is kept on the node's own disk, never in the home folder."""
    if name != "opd":
        return name
    from huggingface_hub import snapshot_download

    for base in ("/scratch1", "/scratch0", "/tmp"):
        folder = os.path.join(base, os.environ.get("USER", "user"), "opd-4b-thinking")
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError:
            continue
        snapshot_download("bingyang-lei/Qwen3-4B-Thinking-Draft-OPD", local_dir=folder)
        return folder
    raise RuntimeError("no disk on this node for the better drafter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation", help="train, validation, test, or all (every answer in the folder)")
    parser.add_argument("--settings", default=DEFAULT)
    parser.add_argument("--rank", type=int, default=16, help="of the saved add-on weights (+fixed@SAVED)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"), help="where lora-<name>.pt files are")
    parser.add_argument("--ckpt-dir", default=os.path.expanduser("~/specdec-ckpt"), help="where saved second drafters are")
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens (for a quick check)")
    parser.add_argument("--max-answers", type=int, default=0, help="stop after this many answers (for a quick check)")
    parser.add_argument("--time-passes", default="", help="text lengths, comma-separated: before the replay, time "
                        "the drafter's passes and the target's check at these lengths on the longest answer")
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
    settings, settings_from = settings_for(args.results_dir, args.settings)
    names = settings.split(",")
    parsed = {name: parse_setting(name) for name in names}
    # The saved add-on weights need the adapters on the drafter. They are attached only if a method
    # asks for them, so that the other runs time the drafter as it is.
    adapters = attach_adapters(draft, rank=args.rank) if any(options["fixed"] for options in parsed.values()) else []
    for adapter in adapters:
        adapter.fast = True
    saved, saved_notes, methods = {}, {}, {}
    for name, options in parsed.items():
        fixed = options["fixed"]
        if fixed and fixed not in saved:
            path = os.path.join(args.adapter_dir, f"lora-{fixed}.pt")
            saved[fixed] = load_adapters(path, adapters)
            saved_notes[fixed] = adapter_notes(path)
        first = DraftTrainer(draft, adapters, rate=0.0, init=saved.get(fixed), sharpen=options["sharpen"])
        methods[name] = (first, make_method(options["method"], args.ckpt_dir, like=draft))
    seconds = {name: {} for name in names}
    timed_here = []
    if args.time_passes and answers:
        for first, _ in methods.values():
            first.begin()                     # the drafter as the first method has it
            break
        longest = max(answers, key=lambda answer: len(answer[2]))
        ids = longest[2].long().to(device)[None]
        timed_here = time_the_passes(target, draft, ids, target_features(target, draft, ids),
                                     [int(length) for length in args.time_passes.split(",")])
        write_json(os.path.join(args.results_dir, "timing.json"), {"milliseconds": timed_here, "tokens": ids.shape[1]}, indent=1)
        del ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    records, started = [], time.time()

    def report(done):
        status = {"done": done, "answers": len(records), "answers_in_this_job": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "draft": args.draft,
                  "settings": settings, "settings_from": settings_from, "saved_adapters_used": saved_notes,
                  "groups": [name for name, _, _ in GROUPS], "by_setting": summarize(records, names),
                  "in_real_time": in_real_time(records, names),
                  "measured_ms": {"tokens_before": MEASURED_AT, "target_check": CHECK_MS, "first_draft": DRAFT_MS},
                  "timed_in_this_job_ms": timed_here,
                  "replay_ms_per_round": {name: {key: 1000 * value / max(seconds[name].get("rounds", 0), 1)
                                                 for key, value in seconds[name].items() if key != "rounds"}
                                          for name in names}}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"),
                   {"groups": status["groups"], "counts": COUNTS, "answers": records}, separators=(",", ":"))

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
        record = {"source": source, "origin": [name, index], "tokens": len(generated), "ended_by_itself": ended,
                  "repetition_from": repetition_from, "settings": {}}
        for setting, (first, improver) in methods.items():
            first.begin()            # the first drafter's own state: saved add-on weights, sharper attention
            watched = None if improver is None else Watched(improver, ids, prompt_length, repetition_from)
            clock = time.time()
            result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=watched,
                                   block_size=16, timers=seconds[setting])
            entry = {"rounds": int(result.rounds), "seconds": round(time.time() - clock, 1),
                     "table": by_position(result.acceptance_lengths, prompt_length, repetition_from),
                     "check_and_draft_ms": [round(value, 1) for value in
                                            check_and_draft_ms(result.acceptance_lengths, prompt_length, repetition_from)]}
            if watched is not None:
                entry["redraft_seconds"] = [round(value, 4) for value in watched.seconds]
                entry["breaks"] = watched.counts
            record["settings"][setting] = entry
        del features
        records.append(record)
        report(False)
    report(True)


if __name__ == "__main__":
    main()
