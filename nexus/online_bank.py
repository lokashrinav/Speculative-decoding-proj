"""Tries several settings of the editor that is trained during the answer (plan item E14).

Saved answers are replayed round by round with the frozen drafter. A bank of small
editors, one per setting, starts from zero for every answer and is trained while the
answer goes by, on the drafter's own drafts and the tokens the target committed
(dflash/online.py). Each editor's draft is recorded in every round and scored against
the text the target wrote. Decoding itself follows the plain draft, so all settings
are scored at the same rounds and can be compared directly.

Only answers from the validation part are used here (the same fixed split as
nexus/fit_gate.py), so that the test part stays untouched for the final numbers.
"""

import argparse
import os
import time

import torch

from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.online import GROUPS, OnlineBank, Setting, summarize
from dflash.replay import replay_decode, target_features
from nexus.fit_gate import split_of
from nexus.probe_memory import write_json
from nexus.replay_memory import load_answers


def settings_grid():
    grid = []
    for labels in ("hindsight", "immediate"):
        for rate in (1e-4, 3e-4, 1e-3):
            for steps in (1, 4):
                grid.append(Setting(f"{labels}-rate{rate:g}-steps{steps}", rank=32, rate=rate, steps=steps, labels=labels))
    grid.append(Setting("hindsight-rate3e-05-steps1", rank=32, rate=3e-5))
    grid.append(Setting("hindsight-rate0.003-steps1", rank=32, rate=3e-3))
    grid.append(Setting("hindsight-rate0.0003-steps1-rank128", rank=128, rate=3e-4))
    grid.append(Setting("hindsight-rate0.0003-steps1-rank8", rank=8, rate=3e-4))
    return grid


def run(draft, target, stops, answers, bank, args, device, report):
    records = []
    started = time.time()
    stop_set = set(stops or [])
    seconds = {}
    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600:
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2:
            continue
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        clock = time.time()
        features = target_features(target, draft, ids)
        bank.begin()
        result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=bank,
                               block_size=16, timers=seconds)
        del features
        scored = bank.finish(ids, ended=ended)
        records.append({"source": source, "origin": [name, index], "tokens": len(generated), "rounds": int(result.rounds),
                        "ended_by_itself": ended, "repetition_from": scored["loop_start"],
                        "seconds": round(time.time() - clock, 1),
                        "table": [[[[round(value) for value in group] for group in margin] for margin in setting]
                                  for setting in scored["table"].tolist()]})
        if len(records) == 1 or len(records) % args.save_every == 0:
            report(bank, records, started, False, seconds)
    report(bank, records, started, True, seconds)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir) if split_of([answer[0], answer[1]]) == args.part]
    answers = answers[args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_in_this_shard": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stops = stop_token_ids(target, tokenizer)
    bank = OnlineBank(settings_grid(), capacity=32768)

    def report(bank, records, started, done, seconds):
        rounds = max(seconds.get("rounds", 0), 1)
        status = {
            "done": done, "answers": len(records), "answers_in_this_shard": len(answers), "part": args.part,
            "hours": round((time.time() - started) / 3600, 3),
            "settings": [setting.__dict__ for setting in bank.settings], "margins": list(bank.margins),
            "groups": [name for name, _, _ in GROUPS], "table_columns": ["rounds", "tokens", "tokens_gained", "rounds_changed"],
            "milliseconds_per_round": {
                "drafter": 1000 * seconds.get("draft", 0.0) / rounds,
                "training_all_settings": 1000 * bank.update_seconds["train"] / max(bank.update_seconds["rounds"], 1),
                "editing_all_settings": 1000 * bank.update_seconds["edit"] / max(bank.update_seconds["rounds"], 1)},
            "more_tokens": summarize(bank),
        }
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"), {"answers": records}, separators=(",", ":"))

    run(draft, target, stops, answers, bank, args, device, report)


if __name__ == "__main__":
    main()
