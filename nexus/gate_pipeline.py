"""The richer memory measurement and the fitted gates, without storing per-round records.

Three steps (nexus/fit_gate.py explains the gates and what is exact):

  collect  replays the training answers (60% of the saved answers). For each answer the
           memory probe hands back its record of every round in memory; a few hundred cells
           per answer are kept for fitting and written to one small file per job.
  fit      fits the gates on those cells (no language model runs) and saves them.
  score    replays the validation and test answers and scores every gate at every margin
           on them; only small tables are written.

collect and score also write the same status.json and per-answer files as the first
replay design (the left-to-right editor is run on every draft and scored, the simple
rules look at the drafter's confidence), so nexus/gate1.py reads them the same way.
nexus/gate_report.py puts the scores together afterwards.
"""

import argparse
import glob
import os
import time

import torch

from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import MemoryProbe
from dflash.replay import replay_decode, target_features
from nexus import fit_gate
from nexus.probe_memory import distinct_share, status_of, write_json
from nexus.replay_memory import load_answers


def replay_with_probe(draft, target, probe, sequence, prompt_length, stop_set, device, seconds):
    """Replays one answer with the probe attached. Returns (the probe's result, whether the answer ended by itself)."""
    ids = sequence.long().to(device)[None]
    ended = int(sequence[-1]) in stop_set
    features = target_features(target, draft, ids)
    probe.begin()
    result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=probe,
                           block_size=16, timers=seconds)
    del features
    scored = probe.finish(ids, ended=ended)
    if "rounds" in scored:
        scored["rounds"]["repetition_from"] = scored["loop_start"]
    scored["decoded_rounds"] = int(result.rounds)
    return scored, ended


def run(args, draft, target, stops, probe, answers, device, handle, finish):
    """handle(answer, record) is called with each answer's in-memory record; finish(records) at the end."""
    records, seconds = [], {}
    started = time.time()
    stop_set = set(stops or [])
    status_path = os.path.join(args.results_dir, "status.json")

    def report(done):
        status = status_of(probe, records, started, done)
        status["answers_in_this_job"] = len(answers)
        rounds = seconds.get("rounds", 0)
        if rounds:
            status["milliseconds_per_round"] = {"drafter": 1000 * seconds.get("draft", 0.0) / rounds,
                                                "probe_and_editor": 1000 * seconds.get("improve", 0.0) / rounds}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        for part in range(0, len(records), 50):   # in parts, so that every file stays small enough to be pushed
            write_json(os.path.join(args.results_dir, f"answers-{part // 50:02d}.json"),
                       {"columns": probe.columns, "answers": records[part : part + 50]}, separators=(",", ":"))

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600:
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2:
            continue
        clock = time.time()
        scored, ended = replay_with_probe(draft, target, probe, sequence, prompt_length, stop_set, device, seconds)
        record = scored.pop("rounds")
        record.update(source=source, origin=[name, index])
        handle((name, index, source), record)
        del record
        records.append({
            "source": source, "origin": [name, index], "prompt_tokens": prompt_length, "tokens": len(generated),
            "rounds": scored["decoded_rounds"], "ended_by_itself": ended, "repetition_from": scored["loop_start"],
            "distinct_4_token_windows": distinct_share(generated),
            "distinct_4_token_windows_last_quarter": distinct_share(generated[len(generated) * 3 // 4:]),
            "seconds": round(time.time() - clock, 1), "bins": scored["bins"],
        })
        if len(records) == 1 or len(records) % args.save_every == 0:
            report(False)
    finish(records)
    report(True)


def fit_step(args):
    """Fits the gates on the cells the collect jobs kept, and saves them."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.time()
    parts = {}
    files = sorted(glob.glob(os.path.join(args.work_dir, "cells-*.pt")))
    for path in files:
        for base, (x, lp, target) in torch.load(path, map_location="cpu").items():
            parts.setdefault(base, []).append((x, lp, target))
    cells = {base: tuple(torch.cat([part[number] for part in rows]) for number in range(3)) for base, rows in parts.items()}
    status = {"done": False, "cell_files": len(files), "fitted_on_cells": {base: int(value[0].shape[0]) for base, value in cells.items()},
              "inputs": [name for name, _ in fit_gate.feature_names()]}
    if not cells:
        write_json(os.path.join(args.results_dir, "status.json"), status, indent=1)
        raise SystemExit("no cells to fit on: did the collect jobs run?")
    status["share_of_cells_where_the_starting_draft_is_right"] = {
        base: float((x[:, :, 2].float().argmax(-1) == target).float().mean()) for base, (x, lp, target) in cells.items()}
    gates = fit_gate.fit_gates(cells, device, args.epochs)
    fit_gate.save_gates(gates, os.path.join(args.work_dir, "gates.pt"))
    status.update(done=True, gates=sorted(gates), seconds=round(time.time() - started, 1))
    write_json(os.path.join(args.results_dir, "status.json"), status, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("collect", "fit", "score"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir")
    parser.add_argument("--work-dir", required=True, help="where the kept cells and the fitted gates go")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--editor", default=None)
    parser.add_argument("--table", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--cells-per-answer", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    if args.mode == "fit":
        fit_step(args)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wanted = ("train",) if args.mode == "collect" else ("validation", "test")
    answers = [answer for answer in load_answers(args.data_dir) if fit_gate.split_of([answer[0], answer[1]]) in wanted]
    answers = answers[args.shard :: args.num_shards]
    write_json(os.path.join(args.results_dir, "status.json"), {"done": False, "answers": 0, "answers_in_this_job": len(answers)}, indent=1)
    gates = fit_gate.load_gates(os.path.join(args.work_dir, "gates.pt"), device) if args.mode == "score" else None
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stops = stop_token_ids(target, tokenizer)
    editor = None
    if args.editor:
        from dflash.pair_editor import BigramTable, PairImprover, load_pair_editor

        judge = load_pair_editor(args.editor, str(device), torch.bfloat16 if device.type == "cuda" else torch.float32)
        editor = PairImprover(judge, BigramTable.load(args.table, str(device)) if args.table else None)
    probe = MemoryProbe(capacity=args.capacity, editor=editor, dump=True)

    if args.mode == "collect":
        kept = {}
        generator = torch.Generator(device=device).manual_seed(args.shard)

        def handle(answer, record):
            for base, cells in fit_gate.sample_cells(record, device, args.cells_per_answer, generator).items():
                kept.setdefault(base, []).append(cells)

        def finish(records):
            cells = {base: tuple(torch.cat([part[number] for part in rows]) for number in range(3)) for base, rows in kept.items()}
            path = os.path.join(args.work_dir, f"cells-{args.shard}of{args.num_shards}.pt")
            torch.save(cells, path + ".tmp")
            os.replace(path + ".tmp", path)
    else:
        rows = []

        def handle(answer, record):
            name, index, source = answer
            sums = fit_gate.score_answer(record, gates, device)
            rows.append({"origin": [name, index], "source": source, "part": fit_gate.split_of([name, index]),
                         "sums": {key: [[[round(value) for value in group] for group in margin] for margin in table.tolist()]
                                  for key, table in sums.items()}})
            if len(rows) % 20 == 0:
                finish(None)

        def finish(records):
            for part in range(0, len(rows), 20):
                write_json(os.path.join(args.results_dir, f"gate-scores-{part // 20:02d}.json"),
                           {"margins": list(fit_gate.MARGINS), "groups": [name for name, _, _ in fit_gate.GROUPS],
                            "counts": list(fit_gate.COUNTS), "answers": rows[part : part + 20]}, separators=(",", ":"))

    run(args, draft, target, stops, probe, answers, device, handle, finish)


if __name__ == "__main__":
    main()
