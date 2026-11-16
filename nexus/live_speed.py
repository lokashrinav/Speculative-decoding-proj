"""Real tokens per second for each way of drafting, with the big model in the loop.

Replay counts rounds. A round with learning in it takes longer, so what counts in the end is
time. This script decodes the same prompts for real, once per setting, and measures tokens per
second: over the whole answer and by how far into the answer a round began (0-2k, 2k+, 8k+ ...).

Settings are written as in nexus/adapt_drafter.py, for example
    plain, plain:sharp, immediate:3e-4:onepass, rejected:3e-4:onepass:only=last1:trigger=2
Decoding is lossless whatever the setting, so every setting should write the same tokens; this
is checked against the plain drafter's output and the saved answer. The order of the settings
changes from prompt to prompt, so that no setting always runs first.
"""

import argparse
import os
import time

import torch

from dflash.adapt import OnePassTrainer, attach_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.redraft import dflash_generate_redraft
from nexus.adapt_drafter import GROUPS, build_trainers, settings_for
from nexus.fit_gate import split_of
from nexus.live_check import first_difference
from nexus.probe_memory import stepped_aside, write_json
from nexus.replay_memory import load_answers


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
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
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir)
               if (args.part == "all" or split_of([answer[0], answer[1]]) == args.part)
               and answer[2].shape[0] - answer[3] >= max(args.min_tokens, 2)]
    answers = answers[args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stops = stop_token_ids(target, tokenizer)
    adapters = attach_adapters(draft, rank=args.rank)
    settings, settings_from = settings_for(args.results_dir, args.settings)
    trainers, started_from = build_trainers(settings, draft, adapters, args.adapter_dir)
    names = list(trainers)
    assert names[0] == "plain", "the first setting must be plain: the others are compared with it"
    for name, trainer in trainers.items():
        assert trainer.loss == "ce" or isinstance(trainer, OnePassTrainer), f"{name}: kl needs onepass in real decoding"

    def decode(name, prompt):
        trainer = trainers[name]
        trainer.begin()
        # A setting that trains nothing needs no call in every round: its state is set by begin().
        improver = trainer if trainer.optimizer is not None else None
        timers, round_clock = {}, []
        result = dflash_generate_redraft(draft, target, prompt, args.max_new_tokens, stops, improver=improver,
                                         block_size=16, return_stats=True, timers=timers, round_clock=round_clock)
        return result, timers, round_clock

    def report(done):
        status = {"done": done, "answers": len(records), "hours": round((time.time() - started) / 3600, 3),
                  "settings": settings, "settings_from": settings_from, "saved_adapters_used": started_from,
                  "groups": [name for name, _, _ in GROUPS], "by_setting": summarize(records, names), "updates": {}}
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
                trainers[setting].begin()
                dflash_generate_redraft(draft, target, prompt, 96, stops, block_size=16, return_stats=True,
                                        improver=trainers[setting] if trainers[setting].optimizer is not None else None)
            warmed = True
        record = {"source": source, "origin": [name, index], "saved_tokens": len(saved) - prompt_length, "settings": {}}
        outputs = {}
        turn = len(records) % len(names)
        for setting in names[turn:] + names[:turn]:
            result, timers, round_clock = decode(setting, prompt)
            outputs[setting] = result.output_ids[0].tolist()
            record["settings"][setting] = {
                "tokens": len(outputs[setting]) - prompt_length, "rounds": int(result.rounds),
                "seconds": round(round_clock[-1][2], 2) if round_clock else 0.0, "table": by_position(round_clock),
                "differs_from_saved_at": first_difference(outputs[setting], saved),
                "timers": {key: round(value, 2) for key, value in timers.items() if isinstance(value, float)}}
        for setting in names:
            record["settings"][setting]["differs_from_plain_at"] = first_difference(outputs[setting], outputs["plain"])
        records.append(record)
        report(False)
    report(True)


if __name__ == "__main__":
    main()
