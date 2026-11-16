"""Real decoding with the frozen drafter (plan item E10, a first check).

Replay measures rounds on saved answers without running the big model. This script checks the
same thing in real decoding. For each chosen prompt it decodes twice with the big model in
the loop: once with the plain drafter and once with the frozen drafter that was taught on
other answers (saved adapters, nothing is trained here). It records the time, the rounds,
and whether both outputs are the same tokens as each other and as the saved answer.

The adapters are used as they are (not merged into the drafter's weights), so their small
extra cost per draft is part of the measured time.
"""

import argparse
import os
import time

import torch

from dflash.adapt import adapter_notes, attach_adapters, load_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.redraft import dflash_generate_redraft
from nexus.fit_gate import split_of
from nexus.probe_memory import write_json
from nexus.replay_memory import load_answers

KINDS = ("plain", "frozen")


def first_difference(first, second):
    """Index of the first token that differs, or None if one is the start of the other and they are equally long."""
    for number, (a, b) in enumerate(zip(first, second)):
        if a != b:
            return number
    return None if len(first) == len(second) else min(len(first), len(second))


def summarize(records):
    out = {}
    for kind in KINDS:
        tokens = sum(record[kind]["tokens"] for record in records)
        rounds = sum(record[kind]["rounds"] for record in records)
        seconds = sum(record[kind]["seconds"] for record in records)
        if rounds:
            out[kind] = {"tokens_per_round": tokens / rounds, "tokens_per_second": tokens / seconds,
                         "ms_per_round": 1000 * seconds / rounds, "tokens": tokens}
    if "plain" in out and "frozen" in out:
        out["frozen_over_plain"] = {
            "more_tokens_per_round": out["frozen"]["tokens_per_round"] / out["plain"]["tokens_per_round"] - 1,
            "more_tokens_per_second": out["frozen"]["tokens_per_second"] / out["plain"]["tokens_per_second"] - 1}
    out["answers"] = len(records)
    out["same_tokens_plain_and_frozen"] = sum(record["plain_and_frozen_differ_at"] is None for record in records)
    out["plain_same_as_saved_answer"] = sum(record["plain"]["differs_from_saved_at"] is None for record in records)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--adapter", default="off-a", help="name of the saved adapters (file lora-<name>.pt)")
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--part", default="validation")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-answers", type=int, default=12)
    parser.add_argument("--max-hours", type=float, default=1.5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir)
               if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    answers = answers[args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stops = stop_token_ids(target, tokenizer)
    adapters = attach_adapters(draft, rank=args.rank)
    path = os.path.join(args.adapter_dir, f"lora-{args.adapter}.pt")
    learned, notes = load_adapters(path, adapters), adapter_notes(path)

    def decode(kind, prompt):
        for number, adapter in enumerate(adapters):
            adapter.reset(learned[number] if kind == "frozen" else None)
        timers = {}
        if device.type == "cuda":
            torch.cuda.synchronize()
        clock = time.perf_counter()
        result = dflash_generate_redraft(draft, target, prompt, args.max_new_tokens, stops, improver=None, block_size=16,
                                         return_stats=True, timers=timers)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return result, time.perf_counter() - clock, timers

    records, started = [], time.time()
    warmed = False
    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or len(records) >= args.max_answers:
            break
        saved = sequence.tolist()
        if len(saved) - prompt_length < 2:
            continue
        prompt = sequence[:prompt_length].long().to(device)[None]
        if not warmed:                                  # the first decoding of a process is slower: not timed
            dflash_generate_redraft(draft, target, prompt, 64, stops, improver=None, block_size=16, return_stats=True)
            warmed = True
        record = {"source": source, "origin": [name, index], "saved_tokens": len(saved) - prompt_length}
        outputs = {}
        for kind in KINDS:
            result, seconds, timers = decode(kind, prompt)
            outputs[kind] = result.output_ids[0].tolist()
            record[kind] = {"tokens": len(outputs[kind]) - prompt_length, "rounds": int(result.rounds),
                            "seconds": round(seconds, 2), "differs_from_saved_at": first_difference(outputs[kind], saved),
                            "timers": {key: round(value, 2) for key, value in timers.items() if isinstance(value, float)}}
        record["plain_and_frozen_differ_at"] = first_difference(outputs["plain"], outputs["frozen"])
        records.append(record)
        write_json(status_path, {"done": False, "adapter": args.adapter, "adapter_notes": notes,
                                 "hours": round((time.time() - started) / 3600, 3), "summary": summarize(records)}, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"), {"answers": records}, separators=(",", ":"))
    write_json(status_path, {"done": True, "adapter": args.adapter, "adapter_notes": notes,
                             "hours": round((time.time() - started) / 3600, 3), "summary": summarize(records)}, indent=1)


if __name__ == "__main__":
    main()
