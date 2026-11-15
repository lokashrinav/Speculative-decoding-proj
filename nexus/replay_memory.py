"""Scores memories on answers that were already decoded, without decoding them again.

The answers saved by probe_memory.py are replayed round by round (dflash/replay.py):
the drafter drafts exactly as in decoding, and each draft is scored against the text
the target wrote. That costs a fraction of decoding, so the measurement can be
repeated with richer settings:

  - the trained left-to-right editor runs on every draft (its result is only scored),
    so the numbers also say what a memory could add on top of the editor;
  - the simple rules also look at how sure the drafter was of its own token;
  - everything recorded per round is saved (one file per answer), so that gates can be
    fitted afterwards without running any model.

The results folder gets the same files as probe_memory.py (status.json, and the
per-answer records in parts, answers-00.json and so on), so nexus/gate1.py reads
them the same way.
"""

import argparse
import glob
import os
import time

import torch

from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import MemoryProbe
from dflash.replay import replay_decode, target_features
from nexus.probe_memory import distinct_share, status_of, write_json


def load_answers(data_dir):
    """Every saved answer, in a fixed order: (file name, index in the file, tokens, prompt length, source)."""
    answers = []
    for path in sorted(glob.glob(os.path.join(data_dir, "shard*of*.pt"))):
        shard = torch.load(path, map_location="cpu")
        name = os.path.basename(path)
        for index, sequence in enumerate(shard["sequences"]):
            answers.append((name, index, sequence, int(shard["prompt_lengths"][index]), shard["sources"][index]))
    return answers


def run(draft, target, stops, answers, probe, args, device, report):
    records, seconds = [], {}
    started = time.time()
    stop_set = set(stops or [])
    os.makedirs(args.out_dir, exist_ok=True)
    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600:
            break
        ids = sequence.long().to(device)[None]
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2:
            continue
        ended = generated[-1] in stop_set
        clock = time.time()
        features = target_features(target, draft, ids)
        read_seconds = time.time() - clock
        probe.begin()
        result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=probe,
                               block_size=16, timers=seconds)
        del features
        scored = probe.finish(ids, ended=ended)
        if "rounds" in scored:
            scored["rounds"].update(source=source, origin=[name, index], ended=ended, tokens=len(generated),
                                    repetition_from=scored["loop_start"])
            torch.save(scored["rounds"], os.path.join(args.out_dir, f"{name[:-3]}-{index:04d}.pt"))
        records.append({
            "source": source, "origin": [name, index], "prompt_tokens": prompt_length, "tokens": len(generated),
            "rounds": int(result.rounds), "ended_by_itself": ended, "repetition_from": scored["loop_start"],
            "distinct_4_token_windows": distinct_share(generated),
            "distinct_4_token_windows_last_quarter": distinct_share(generated[len(generated) * 3 // 4:]),
            "seconds": round(time.time() - clock, 1), "seconds_reading_the_answer": round(read_seconds, 1),
            "bins": scored["bins"],
        })
        if len(records) == 1 or len(records) % args.save_every == 0:
            report(probe, records, started, False, seconds)
    report(probe, records, started, True, seconds)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--editor", default=None)
    parser.add_argument("--table", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = load_answers(args.data_dir)[args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_in_this_shard": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stops = stop_token_ids(target, tokenizer)
    editor = None
    if args.editor:
        from dflash.pair_editor import BigramTable, PairImprover, load_pair_editor

        on_gpu = device.type == "cuda"
        judge = load_pair_editor(args.editor, str(device), torch.bfloat16 if on_gpu else torch.float32)
        editor = PairImprover(judge, BigramTable.load(args.table, str(device)) if args.table else None)
    probe = MemoryProbe(capacity=args.capacity, editor=editor, dump=True)

    def report(probe, records, started, done, seconds):
        status = status_of(probe, records, started, done)
        status["answers_in_this_shard"] = len(answers)
        rounds = seconds.get("rounds", 0)
        if rounds:
            status["milliseconds_per_round"] = {"drafter": 1000 * seconds.get("draft", 0.0) / rounds,
                                                "probe_and_editor": 1000 * seconds.get("improve", 0.0) / rounds}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        # In parts of 50 answers, so that every file stays small enough to be pushed.
        for part in range(0, len(records), 50):
            write_json(os.path.join(args.results_dir, f"answers-{part // 50:02d}.json"),
                       {"columns": probe.columns, "answers": records[part : part + 50]}, separators=(",", ":"))

    run(draft, target, stops, answers, probe, args, device, report)


if __name__ == "__main__":
    main()
