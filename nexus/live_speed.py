"""Real tokens per second for each way of drafting, with the big model in the loop.

Replay counts rounds. A round with learning in it takes longer, so what counts in the end is
time. This script decodes the same prompts for real, once per setting, and measures tokens per
second: over the whole answer and by how far into the answer a round began (0-2k, 2k+, 8k+ ...).

Settings are written as in nexus/adapt_drafter.py, for example
    plain, plain:sharp, immediate:3e-4:onepass, rejected:3e-4:onepass:only=last1:trigger=2
A setting whose drafter does not learn can end in +swapK or +swapKtN, for example
    plain+swap4t3, frozen@off-a+swap4t3
The target then also checks, in the same pass, the first draft with the drafter's second choice
at each of its K most likely breaks (dflash/redraft.py: Branches, "switch"). The drafter is not
run again. With tN only the switched token and the N tokens after it are checked.
+redraftK is the comparison that does run the drafter again: the same second choices at the same
breaks, and one more drafter pass that rewrites what follows each of them (Branches, "early").
+treeB is DDTree with B nodes in place of the first draft; +restart4pxst7@NAME runs the drafter again
from its likely breaks with the states of the saved predictor NAME; +tree63_restart4pxt7@NAME is both
(dflash/redraft.py: DraftTree, Restarts, TreeWithRestarts).
A setting with :window=N (plain:window=3072, frozen:window=3072@off-a+swap4t3) lets the drafter see
only the last N tokens of the text.

The same text for every setting. A pass with alternatives has more rows than plain decoding's pass,
and in 16-bit arithmetic that rounds the target's numbers differently: about 1 row in 100 has two
nearly tied best tokens, and there the first choice can be the other one (nexus/time_check.py).
Left to itself such a setting soon writes another answer than plain decoding, of another length,
and its speed cannot be compared. So the settings with alternatives FOLLOW plain decoding's answer
to the same prompt (decoded first): every pass is computed and timed as usual, but what the target
says after a token is read from that answer. "followed" gives the rows that decided and how many
of them the pass itself decided otherwise. --free turns this off.
Decoding is lossless whatever the setting, so every setting should write the same tokens; this
is checked against the plain drafter's output and the saved answer. The order of the settings
changes from prompt to prompt, so that no setting always runs first.
"""

import argparse
import os
import re
import time

import torch

from dflash.adapt import OnePassTrainer, attach_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.redraft import Branches, dflash_generate_redraft
from nexus.adapt_drafter import GROUPS, build_trainers, settings_for
from nexus.fit_gate import split_of
from nexus.live_check import first_difference
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers


def swap_of(setting, ckpt_dir=None, like=None):
    """'frozen@off-a+swap4t3' -> ('frozen@off-a', the plug-in of the swap); (setting, None) without one.
    '+redraft4' gives the plug-in that runs the drafter again from the same switched tokens.
    '+tree63', '+tree63_restart4pxt7@NAME', '+restart4pxst7@NAME': DDTree, restarts with the states of
    the saved predictor NAME, or both in one check (the method names of nexus/redraft_long.py)."""
    found = re.fullmatch(r"(.+)\+(swap|redraft)(\d+)(?:t(\d+))?", setting)
    if found:
        return found[1], Branches(k=int(found[3]), differ="switch" if found[2] == "swap" else "early",
                                  tail=None if found[4] is None else int(found[4]))
    found = re.fullmatch(r"(.+?)\+((?:(?:b?tree|restart|echo)\d+|reanchor).*)", setting)
    if found:
        from nexus.redraft_long import make_method

        return found[1], make_method(found[2], ckpt_dir, like)
    return setting, None


def by_position(round_clock):
    """[rounds, tokens, seconds] for each position group, from the clock of every round."""
    table = [[0, 0, 0.0] for _ in GROUPS]
    before = 0.0
    for where, produced, now in round_clock:
        for number, (_, low, high) in enumerate(GROUPS):
            if low <= where < high:
                table[number][0] += 1
                table[number][1] += produced
                table[number][2] += now - before
        before = now
    return table


def summarize(records, names):
    out = {}
    for name in names:
        rows = {}
        for number, (group, _, _) in enumerate(GROUPS):
            cells = [record["settings"][name]["table"][number] for record in records if name in record["settings"]]
            rounds, tokens, seconds = (sum(cell[column] for cell in cells) for column in range(3))
            if not rounds:
                continue
            rows[group] = {"rounds": rounds, "tokens_per_round": tokens / rounds, "tokens_per_second": tokens / seconds,
                           "ms_per_round": 1000 * seconds / rounds}
            plain = [record["settings"]["plain"]["table"][number] for record in records
                     if "plain" in record["settings"] and name in record["settings"]]
            if plain and sum(cell[0] for cell in plain):
                base_rounds, base_tokens, base_seconds = (sum(cell[column] for cell in plain) for column in range(3))
                rows[group]["more_tokens_per_round_than_plain"] = tokens / rounds / (base_tokens / base_rounds) - 1
                rows[group]["more_tokens_per_second_than_plain"] = tokens / seconds / (base_tokens / base_seconds) - 1
        rows["answers_with_the_plain_drafters_tokens"] = sum(
            record["settings"][name]["differs_from_plain_at"] is None for record in records if name in record["settings"])
        out[name] = rows
    return out


def bench_prompts(tokenizer, names, count, reasoning, load=None):
    """The first turn of `count` prompts of each standard benchmark, in the chat template, as the
    saved answers are listed: (benchmark, number, tokens, prompt length, benchmark). The prompts
    are the ones dflash/benchmark.py picks (a fixed shuffle), taken in turn from the benchmarks so
    that a run that stops early has read all of them alike."""
    from dflash.benchmark import _select_dataset, apply_chat_template, load_and_process_dataset

    picked = []
    for name in names:
        rows = _select_dataset((load or load_and_process_dataset)(name), count)
        for number, row in enumerate(rows):
            text = apply_chat_template(tokenizer, [{"role": "user", "content": row["turns"][0]}], reasoning)
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
            picked.append((number, name, tokens))
    picked.sort(key=lambda entry: entry[0])                      # the benchmarks in turn (the sort keeps their order)
    return [(name, number, tokens, tokens.shape[0], name) for number, name, tokens in picked]


def by_source(records, names):
    """Tokens per round and per second of every setting for each source of prompts, with the gain over plain."""
    out = {}
    for source in dict.fromkeys(record["source"] for record in records):
        mine = [record for record in records if record["source"] == source]
        out[source] = {"prompts": len(mine)}
        totals = {name: [sum(record["settings"][name][key] for record in mine if name in record["settings"])
                         for key in ("tokens", "rounds", "seconds")] for name in names}
        for name, (tokens, rounds, seconds) in totals.items():
            if not rounds or not seconds:
                continue
            row = {"tokens_per_round": tokens / rounds, "tokens_per_second": tokens / seconds}
            base_tokens, base_rounds, base_seconds = totals.get("plain", (0, 0, 0))
            if base_rounds and base_seconds and all(name in record["settings"] for record in mine):
                row["more_tokens_per_round_than_plain"] = row["tokens_per_round"] / (base_tokens / base_rounds) - 1
                row["more_tokens_per_second_than_plain"] = row["tokens_per_second"] / (base_tokens / base_seconds) - 1
            out[source][name] = row
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--settings", default="plain,immediate:3e-4:onepass")
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--part", default="validation")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--min-tokens", type=int, default=0, help="only prompts whose saved answer is at least this long")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-answers", type=int, default=12)
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--free", action="store_true", help="settings with alternatives do not follow the saved answer")
    parser.add_argument("--bench", default="", help="prompts of the standard benchmarks instead of the saved answers': "
                        "a comma-separated list of gsm8k, math500, humaneval, mbpp, mt-bench (dflash/benchmark.py)")
    parser.add_argument("--per-bench", type=int, default=40, help="prompts of each benchmark (the same ones every time)")
    parser.add_argument("--reasoning", default="off", help="with --bench: the chat template's thinking, on or off")
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    if args.bench:
        answers = bench_prompts(tokenizer, args.bench.split(","), args.per_bench, args.reasoning)
    else:
        answers = [answer for answer in load_answers(args.data_dir)
                   if (args.part == "all" or split_of([answer[0], answer[1]]) == args.part)
                   and answer[2].shape[0] - answer[3] >= max(args.min_tokens, 2)]
    answers = answers[args.shard :: args.num_shards]
    stops = stop_token_ids(target, tokenizer)
    adapters = attach_adapters(draft, rank=args.rank)
    for adapter in adapters:
        adapter.fast = True                 # the cheaper way to apply the extra weights when nothing is learned
    settings, settings_from = settings_for(args.results_dir, args.settings)
    names = settings.split(",")
    from nexus.redraft_long import token_counts

    token_counts(args.adapter_dir, args.data_dir, target.config.vocab_size)       # for settings with a short list (vN)
    swaps = {name: swap_of(name, args.adapter_dir, draft) for name in names}
    trainers, started_from = build_trainers(",".join(dict.fromkeys(base for base, _ in swaps.values())), draft, adapters,
                                            args.adapter_dir)
    assert names[0] == "plain", "the first setting must be plain: the others are compared with it"
    for name, trainer in trainers.items():
        assert trainer.loss == "ce" or isinstance(trainer, OnePassTrainer), f"{name}: kl needs onepass in real decoding"
    for name, (base, swap) in swaps.items():
        assert swap is None or trainers[base].optimizer is None, f"{name}: the swap goes with a drafter that does not learn"

    def improver_of(name):
        base, swap = swaps[name]
        trainer = trainers[base]
        trainer.begin()
        # A setting that trains nothing needs no call in every round: its state is set by begin().
        return (trainer if trainer.optimizer is not None else swap), int(trainer.context_window or 0)

    def decode(name, prompt, plain_ids=None):
        improver, window = improver_of(name)
        timers, round_clock = {}, []
        follow = plain_ids if swaps[name][1] is not None and not args.free else None
        result = dflash_generate_redraft(draft, target, prompt, args.max_new_tokens, stops, improver=improver,
                                         block_size=16, return_stats=True, timers=timers, round_clock=round_clock,
                                         context_window=window, follow=follow)
        return result, timers, round_clock

    def report(done):
        status = {"done": done, "answers": len(records), "hours": round((time.time() - started) / 3600, 3),
                  "settings": settings, "settings_from": settings_from, "saved_adapters_used": started_from,
                  "groups": [name for name, _, _ in GROUPS], "by_setting": summarize(records, names),
                  "by_source": by_source(records, names), "updates": {}}
        for name, trainer in trainers.items():
            rounds = max(trainer.totals["rounds"], 1)
            status["updates"][name] = {"updates_per_round": trainer.totals["updates"] / rounds,
                                       "ms_per_update": 1000 * trainer.totals["seconds"] / max(trainer.totals["updates"], 1)}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"), {"groups": status["groups"], "answers": records},
                   separators=(",", ":"))

    records, started = [], time.time()
    warmed = False
    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or len(records) >= args.max_answers:
            break
        saved = sequence.tolist()
        prompt = sequence[:prompt_length].long().to(device)[None]
        if not warmed:                                  # the first decoding of a process is slower: not timed
            for setting in names:
                improver, window = improver_of(setting)
                dflash_generate_redraft(draft, target, prompt, 96, stops, block_size=16, return_stats=True,
                                        improver=improver, context_window=window)
            warmed = True
        record = {"source": source, "origin": [name, index], "saved_tokens": len(saved) - prompt_length, "settings": {}}
        outputs, plain_ids = {}, None
        others = names[1:]
        turn = len(records) % max(len(others), 1)
        for setting in names[:1] + others[turn:] + others[:turn]:      # plain first: the others may follow its answer
            result, timers, round_clock = decode(setting, prompt, plain_ids)
            plain_ids = result.output_ids if setting == "plain" else plain_ids
            outputs[setting] = result.output_ids[0].tolist()
            record["settings"][setting] = {
                "tokens": len(outputs[setting]) - prompt_length, "rounds": int(result.rounds),
                "seconds": round(round_clock[-1][2], 2) if round_clock else 0.0, "table": by_position(round_clock),
                "differs_from_saved_at": first_difference(outputs[setting], saved),
                "timers": {key: round(value, 2) for key, value in timers.items() if isinstance(value, float)},
                # rounds and tokens the target read in checks of one draft ("one") and of several ("two")
                "checks": {key: value for key, value in timers.items() if isinstance(value, int)},
                # rounds with alternatives, rounds one of them won, tokens they added, and how often the
                # second reading of tokens accepted on an alternative's rows corrected one
                "alternatives": [int(result.rounds_with_alternative), int(result.alternative_wins),
                                 int(result.alternative_extra_tokens), int(result.corrections)],
                # rows that decided while following the saved answer, and how many the pass decided otherwise
                "followed": [int(result.followed_rows), int(result.followed_against)]}
            counts = getattr(swaps[setting][1], "counts", None)   # a second run's own counts, summed over the answers so far
            # (for a second run in a pass of its own "improve" is its time; blocks written in the first
            # draft's pass add to "draft" instead)
            if counts:
                record["settings"][setting]["second_run"] = dict(counts)
        for setting in names:
            record["settings"][setting]["differs_from_plain_at"] = first_difference(outputs[setting], outputs["plain"])
        records.append(record)
        report(False)
    report(True)


if __name__ == "__main__":
    main()
