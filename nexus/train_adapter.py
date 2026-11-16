"""Trains the drafter's adapters across many answers, to see what carries over (an ablation of E14).

The published method starts every answer from the frozen drafter. Here the adapters are
NOT reset between answers: the drafter goes through the training answers one after another
and keeps learning, with the same training as during an answer (dflash/adapt.py).

Before it trains on an answer, that answer is replayed twice without any training:
  plain    the frozen drafter,
  frozen   the drafter with what it has learned from the EARLIER answers only.
The answer has not been seen before, so "frozen" over "plain" is what carries over from
other answers. Then the answer is replayed a third time with training ("online"), which
continues the learning. The adapters are saved every few answers, for use as a starting
state by nexus/adapt_drafter.py (settings ending in @name).
"""

import argparse
import os
import time

import torch

from dflash.adapt import DraftTrainer, adapter_state, attach_adapters, save_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.replay import replay_decode, target_features
from nexus.adapt_drafter import GROUPS, by_position
from nexus.fit_gate import split_of
from nexus.probe_memory import write_json
from nexus.replay_memory import load_answers

KINDS = ("plain", "frozen", "online")


def summarize(records, block=10):
    """Tokens per round over the frozen drafter, for the answers in the order they came, in blocks."""
    out = []
    for begin in list(range(0, len(records), block)) + ["all but the first block"]:
        part = records[block:] if begin == "all but the first block" else records[begin : begin + block]
        if not part:
            continue
        row = {"answers": f"{begin + 1}-{begin + len(part)}" if isinstance(begin, int) else begin}
        for number, (group, _, _) in enumerate(GROUPS):
            sums = {kind: [sum(record[kind][number][column] for record in part) for column in (0, 1)] for kind in KINDS}
            if not sums["plain"][0]:
                continue
            base = sums["plain"][1] / sums["plain"][0]
            row[group] = {"plain_tokens_per_round": base, "rounds": sums["plain"][0]}
            for kind in ("frozen", "online"):
                if sums[kind][0]:
                    row[group][f"{kind}_more_tokens_per_round_than_plain"] = sums[kind][1] / sums[kind][0] / base - 1
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out", required=True, help="file the adapters are saved to")
    parser.add_argument("--part", default="train")
    parser.add_argument("--labels", default="immediate")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--decay", type=float, default=7.0)
    parser.add_argument("--top", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-answers", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=2.5)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir) if split_of([answer[0], answer[1]]) == args.part]
    order = torch.randperm(len(answers), generator=torch.Generator().manual_seed(args.seed)).tolist()
    answers = [answers[number] for number in order]            # kinds of answers are mixed, the same way for every job
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_available": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    adapters = attach_adapters(draft, rank=args.rank)
    trainer = DraftTrainer(draft, adapters, labels=args.labels, rate=args.rate, stride=args.stride, decay=args.decay,
                           loss=args.loss, top=args.top)
    top = args.top if args.loss == "kl" else 0
    records, started, seconds = [], time.time(), {kind: {} for kind in KINDS}

    def report(done):
        status = {"done": done, "answers": len(records), "answers_available": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "labels": args.labels, "loss": args.loss,
                  "rate": args.rate, "rank": args.rank, "stride": args.stride, "updates": trainer.totals["updates"],
                  "groups": [name for name, _, _ in GROUPS], "in_order": summarize(records),
                  "ms_per_round": {kind: 1000 * seconds[kind].get("draft", 0.0) / max(seconds[kind].get("rounds", 0), 1)
                                   for kind in KINDS},
                  "training_ms_per_update": 1000 * trainer.totals["seconds"] / max(trainer.totals["updates"], 1)}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"), {"groups": status["groups"], "answers": records},
                   separators=(",", ":"))

    def save():
        save_adapters(args.out, adapters, answers=len(records), updates=trainer.totals["updates"], labels=args.labels,
                      loss=args.loss, rate=args.rate, stride=args.stride)

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and len(records) >= args.max_answers):
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2:
            continue
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        repetition_from = None if ended else loop_start(generated)
        features, predictions = target_features(target, draft, ids, top=top) if top else (target_features(target, draft, ids), None)
        record = {"source": source, "origin": [name, index], "tokens": len(generated), "ended_by_itself": ended,
                  "repetition_from": repetition_from, "updates_before": trainer.totals["updates"]}

        def replay(kind, improver=None):
            result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=improver,
                                   block_size=16, timers=seconds[kind])
            record[kind] = by_position(result.acceptance_lengths, prompt_length, repetition_from)

        learned = adapter_state(adapters)                 # what the earlier answers taught
        for adapter in adapters:
            adapter.reset()
        replay("plain")
        for adapter, state in zip(adapters, learned):
            adapter.reset(state)
        replay("frozen")
        trainer.begin(keep=True)                          # go on learning; nothing is reset
        trainer.predictions = predictions
        replay("online", trainer)
        del features, predictions
        records.append(record)
        if len(records) % args.save_every == 0:
            save()
        report(False)
    save()
    report(True)


if __name__ == "__main__":
    main()
