"""Trains the drafter during the answer and compares which positions it should learn from
(plan items E9 and E14).

Each saved answer is replayed once per setting. The drafter's adapters start at zero for
every answer and are trained while the answer goes by (dflash/adapt.py). Decoding follows
the adapted drafter's own drafts, so the rounds are the ones real decoding would have,
and the result is simply how many rounds the answer takes.

Settings are written as labels:rate (for example hindsight:1e-4), with :kl added to compare
the drafter with the target's own predictions instead of the committed token alone, and
"plain" is the frozen drafter. More can follow the rate: stride=N trains on every Nth
round only, and only=attn, only=mlp or only=lastK trains part of the adapters (cheaper
steps). A setting can end in @name: the adapters then start every
answer from the saved state lora-<name>.pt (trained across other answers by
nexus/train_adapter.py) instead of from zero, and "frozen@name" uses that state without any
training during the answer. The labels "canvas" and "canvas+hindsight" need what the target
predicted on every draft, so for those settings the target checks each draft again as it
does in decoding (slower: about as slow as decoding itself). The same answers are used for every setting, so they can be compared
answer by answer. Only validation answers are used here unless another part is named.
"""

import argparse
import os
import time

import torch

from dflash.adapt import DraftTrainer, adapter_notes, attach_adapters, choose_adapters, load_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.replay import replay_decode, target_features
from nexus.fit_gate import split_of
from nexus.probe_memory import write_json
from nexus.replay_memory import load_answers

GROUPS = (("0-2k", 0, 2000), ("2k+", 2000, 10**9), ("8k+", 8000, 10**9), ("everything", 0, 10**9))
DEFAULT = "plain,immediate:3e-5,immediate:1e-4,immediate:3e-4,hindsight:3e-5,hindsight:1e-4,hindsight:3e-4,both:1e-4"


def by_position(acceptance_lengths, prompt_length, repetition_from):
    """[rounds, tokens] of an answer for each position group, on rounds outside endless repetition."""
    table = [[0, 0] for _ in GROUPS]
    start = prompt_length
    for gained in acceptance_lengths:
        where = start - prompt_length
        if repetition_from is None or where < repetition_from:
            for number, (_, low, high) in enumerate(GROUPS):
                if low <= where < high:
                    table[number][0] += 1
                    table[number][1] += gained
        start += gained
    return table


def parse_setting(setting):
    """labels:rate, then in any order kl, stride=N (train on every Nth round only) and
    only=attn|mlp|lastK (which adapters are trained), and at the end @name (start from saved adapters).
    "plain" and "frozen" take no rate."""
    body, _, start_from = setting.partition("@")
    parts = body.split(":")
    untrained = parts[0] in ("plain", "frozen")
    options = {"labels": parts[0], "rate": 0.0 if untrained else float(parts[1]), "loss": "ce", "stride": 0,
               "only": "all", "start_from": start_from}
    for part in parts[1 if untrained else 2 :]:
        key, _, value = part.partition("=")
        if part in ("kl", "ce"):
            options["loss"] = part
        elif key == "stride":
            options["stride"] = int(value)
        elif key == "only":
            options["only"] = value
        else:
            raise ValueError(f"cannot read the setting {setting}")
    return options


def settings_for(results_dir, given, folder=os.path.join("nexus", "settings")):
    """The settings of a job can still be chosen after it was queued: the first line of
    nexus/settings/<job name without its number>.txt, if that file exists, replaces the
    ones on the command line. Returns the settings and where they came from."""
    family = os.path.basename(os.path.normpath(results_dir)).rsplit("-", 1)[0]
    path = os.path.join(folder, family + ".txt")
    if os.path.exists(path):
        with open(path) as handle:
            line = handle.readline().strip()
        if line:
            return line, path
    return given, "command line"


def canvas_agreement(used, acceptance_lengths, prompt_length, tokens):
    """How often the target's first choice on a draft was the token the answer really has there.

    used: (round start, the target's first choice at each position of that round's draft).
    Returns [positions up to the first rejection, of those equal to the answer,
             positions after the first rejection, of those equal to the answer]."""
    gained_at, start = {}, prompt_length
    for gained in acceptance_lengths:
        gained_at[start] = gained
        start += gained
    counts = [0, 0, 0, 0]
    for position, labels in used:
        gained = gained_at.get(position)
        if gained is None:
            continue
        for offset, (label, true) in enumerate(zip(labels, tokens[position + 1 : position + 1 + len(labels)])):
            base = 0 if offset < gained else 2
            counts[base] += 1
            counts[base + 1] += int(label == true)
    return counts


def summarize(records, names):
    """Tokens per round of every setting, and how many more tokens per round than the frozen drafter."""
    out = {}
    for name in names:
        rows = {}
        for number, (group, _, _) in enumerate(GROUPS):
            rounds = sum(record["settings"][name]["table"][number][0] for record in records if name in record["settings"])
            tokens = sum(record["settings"][name]["table"][number][1] for record in records if name in record["settings"])
            if not rounds:
                continue
            rows[group] = {"rounds": rounds, "tokens_per_round": tokens / rounds}
            plain = [record["settings"]["plain"]["table"][number] for record in records
                     if "plain" in record["settings"] and name in record["settings"]]
            if plain and sum(row[0] for row in plain):
                base = sum(row[1] for row in plain) / sum(row[0] for row in plain)
                rows[group]["more_tokens_per_round_than_plain"] = tokens / rounds / base - 1
        counts = [sum(record["settings"][name]["canvas"][number] for record in records
                      if "canvas" in record["settings"].get(name, {})) for number in range(4)]
        if counts[0] and counts[2]:
            rows["what_the_target_predicted_on_the_drafts"] = {
                "share_of_positions_after_the_first_rejection": counts[2] / (counts[0] + counts[2]),
                "first_choice_is_the_answers_token_after_the_first_rejection": counts[3] / counts[2],
                "first_choice_is_the_answers_token_up_to_it": counts[1] / counts[0]}
        out[name] = rows
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation", help="train, validation, test, or all (every answer in the folder)")
    parser.add_argument("--settings", default=DEFAULT)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--decay", type=float, default=7.0)
    parser.add_argument("--top", type=int, default=32, help="how many of the target's most likely tokens are kept for kl")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"), help="where lora-<name>.pt files are")
    parser.add_argument("--wait-minutes", type=float, default=120, help="how long to wait for a lora-<name>.pt file")
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens (for a quick check)")
    parser.add_argument("--max-answers", type=int, default=0, help="stop after this many answers (for a quick check)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir)
               if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    answers = answers[args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_in_this_job": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    adapters = attach_adapters(draft, rank=args.rank)
    trainers = {}
    settings, settings_from = settings_for(args.results_dir, args.settings)
    starts, started_from = {}, {}
    for setting in settings.split(","):
        options = parse_setting(setting)
        start_from = options["start_from"]
        plain = options["labels"] in ("plain", "frozen")
        if start_from and start_from not in starts:
            path = os.path.join(args.adapter_dir, f"lora-{start_from}.pt")
            deadline = time.time() + 60 * args.wait_minutes          # the file may still be in the making
            while not os.path.exists(path) and time.time() < deadline:
                time.sleep(30)
            starts[start_from] = load_adapters(path, adapters)
            started_from[start_from] = adapter_notes(path)
        trainers[setting] = DraftTrainer(draft, adapters, labels="hindsight" if plain else options["labels"],
                                         rate=options["rate"], stride=options["stride"] or args.stride, decay=args.decay,
                                         loss=options["loss"], top=args.top, init=starts.get(start_from),
                                         trained=choose_adapters(adapters, options["only"]))
    top = args.top if any(trainer.loss == "kl" for trainer in trainers.values()) else 0
    names = list(trainers)
    seconds = {name: {} for name in names}
    records, started = [], time.time()

    def report(done):
        status = {"done": done, "answers": len(records), "answers_in_this_job": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "rank": args.rank, "stride": args.stride,
                  "settings": settings, "settings_from": settings_from, "saved_adapters_used": started_from,
                  "adapter_parameters": sum(parameter.numel() for adapter in adapters for parameter in (adapter.down, adapter.up)),
                  "groups": [name for name, _, _ in GROUPS], "by_setting": summarize(records, names), "cost": {}}
        for name, trainer in trainers.items():
            rounds = max(seconds[name].get("rounds", 0), 1)
            status["cost"][name] = {
                "drafting_ms_per_round": 1000 * seconds[name].get("draft", 0.0) / rounds,
                "training_ms_per_round": 1000 * trainer.totals["seconds"] / rounds,
                "target_check_ms_per_round": 1000 * seconds[name].get("target", 0.0) / rounds,
                "training_ms_per_update": 1000 * trainer.totals["seconds"] / max(trainer.totals["updates"], 1),
                "updates_per_round": trainer.totals["updates"] / rounds,
                "positions_per_update": trainer.totals["positions"] / max(trainer.totals["updates"], 1)}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"), {"groups": status["groups"], "answers": records},
                   separators=(",", ":"))

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and len(records) >= args.max_answers):
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2 or (args.longest and len(generated) > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        repetition_from = None if ended else loop_start(generated)
        features, predictions = target_features(target, draft, ids, top=top) if top else (target_features(target, draft, ids), None)
        record = {"source": source, "origin": [name, index], "tokens": len(generated), "ended_by_itself": ended,
                  "repetition_from": repetition_from, "settings": {}}
        for setting, trainer in trainers.items():
            trainer.begin()
            trainer.predictions = predictions
            clock = time.time()
            result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=trainer,
                                   block_size=16, timers=seconds[setting])
            record["settings"][setting] = {"rounds": int(result.rounds), "seconds": round(time.time() - clock, 1),
                                           "table": by_position(result.acceptance_lengths, prompt_length, repetition_from)}
            if trainer.canvas_used:
                record["settings"][setting]["canvas"] = canvas_agreement(
                    trainer.canvas_used, result.acceptance_lengths, prompt_length, sequence.tolist())
        del features, predictions
        records.append(record)
        report(False)
    report(True)


if __name__ == "__main__":
    main()
