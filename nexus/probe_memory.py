"""Week 1 of the hindsight-memory plan (experiments E0 to E3).

Long answers are generated with Qwen3-4B in thinking mode, decoded with plain DFlash.
While each answer is decoded, MemoryProbe (dflash/hindsight.py) measures what a memory
of the drafter's own past drafts would have known. The drafts are never changed, so
the answers are the target's own text and are saved for later replay and training.

Prompts are development prompts only, kept apart from every benchmark used for final
results: AIME problems from before 2024, MATH training problems of the hardest level,
long-writing requests, and chat prompts. A source that fails to load is skipped and
reported, and its share is filled from MetaMathQA's MATH-type questions.

The prompt list is built once (--only-write-prompts) so that every shard cuts the same list.

Two files are written to the results folder and refreshed every few answers:
  status.json   the summary tables (by position in the answer) and how long answers are
  answers.json  one compact record per answer, so answers can be left out or resampled later
"""

import argparse
import json
import os
import random
import sys
import time

import torch

from dflash.benchmark import apply_chat_template, load_transformers_models, stop_token_ids
from dflash.hindsight import MemoryProbe, read_cost, summarize
from dflash.redraft import dflash_generate_redraft

MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def field(row, names):
    """The first non-empty text among the row's columns called one of `names` (any letter case)."""
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def mostly_english(text):
    return sum(character.isascii() for character in text) >= 0.95 * max(len(text), 1)


def load_long_prompts(total, seed, notes):
    """(source, prompt) pairs, shuffled. notes collects what loaded and what failed."""
    from datasets import load_dataset

    def aime():
        out = []
        for row in load_dataset("di-zhang-fdu/AIME_1983_2024", split="train"):
            lowered = {str(key).lower(): value for key, value in row.items()}
            year = lowered.get("year")
            if year is None:  # an ID such as "2024-II-5"
                year = str(lowered.get("id", "")).split("-")[0]
            if not str(year).strip().isdigit() or int(year) >= 2024:
                continue      # 2024 and 2025 are kept for final results
            text = field(row, ("question", "problem"))
            out.append(text + MATH_SUFFIX if text else None)
        return out

    def math_level_5():
        out = []
        for subject in MATH_SUBJECTS:
            for row in load_dataset("EleutherAI/hendrycks_math", subject, split="train"):
                if str(row.get("level", "")).strip() == "Level 5":
                    text = field(row, ("problem",))
                    out.append(text + MATH_SUFFIX if text else None)
        return out

    def long_writing():
        out = []
        for row in load_dataset("zai-org/LongWriter-6k", split="train"):
            messages = row.get("messages")
            if isinstance(messages, list):
                users = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
                text = users[0] if users else None
            else:
                text = field(row, ("prompt", "instruction", "query", "question"))
            out.append(text if isinstance(text, str) and mostly_english(text) else None)
        return out

    def stream(name, split, count, pick):
        rows = load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=10000)
        out = []
        for row in rows:
            text = pick(row)
            if text:
                out.append(text)
            if len(out) >= count:
                break
        return out

    plan = [
        ("aime-before-2024", 0.30, aime),
        ("math-train-level5", 0.30, math_level_5),
        ("longwriter", 0.25, long_writing),
        ("ultrachat", 0.15, lambda: stream("HuggingFaceH4/ultrachat_200k", "train_sft", 2 * total,
                                            lambda row: row.get("prompt"))),
    ]
    rng = random.Random(seed)
    chosen = []
    for name, share, loader in plan:
        try:
            prompts = sorted({p.strip() for p in loader() if isinstance(p, str) and p.strip()})
            rng.shuffle(prompts)
            take = prompts[: round(total * share)]
            chosen.extend((name, prompt) for prompt in take)
            notes[name] = f"{len(prompts)} available, {len(take)} used"
        except Exception as error:
            notes[name] = f"FAILED: {type(error).__name__}: {str(error)[:300]}"
    if len(chosen) < total:
        try:
            fill = stream("meta-math/MetaMathQA", "train", 3 * (total - len(chosen)),
                          lambda row: row.get("query") if str(row.get("type", "")).startswith("MATH") else None)
            fill = sorted(set(fill))
            rng.shuffle(fill)
            fill = fill[: total - len(chosen)]
            chosen.extend(("metamath-math", prompt + MATH_SUFFIX) for prompt in fill)
            notes["metamath-math"] = f"{len(fill)} used to fill"
        except Exception as error:
            notes["metamath-math"] = f"FAILED: {type(error).__name__}: {str(error)[:300]}"
    rng.shuffle(chosen)
    return chosen


def load_bench_prompts(notes):
    """(source, prompt) pairs of the math benchmarks the published Test-Time Speculation paper reports on:
    AIME 2024, AIME 2025 and the hardest MATH-500 problems. In a fixed order, AIME first. These are kept
    for final comparisons and are never used to choose anything."""
    from datasets import load_dataset

    def rows_of(names, splits=("train", "test")):
        """Rows of the first of several places a set is published at that loads."""
        problems = []
        for name in names:
            parts, found = (name if isinstance(name, (list, tuple)) else [name]), []
            try:
                for part in parts:
                    for split in splits:
                        try:
                            found.extend(load_dataset(part, split=split))
                            break
                        except ValueError:          # no such split: try the next one
                            continue
                    else:
                        raise ValueError(f"no usable split in {part}")
                return found, "+".join(parts)
            except Exception as error:
                problems.append(f"{'+'.join(parts)}: {type(error).__name__}: {str(error)[:120]}")
        raise RuntimeError("; ".join(problems))

    plan = [
        ("aime-2024", [["MathArena/aime_2024_I", "MathArena/aime_2024_II"]], None),
        ("aime-2025", ["MathArena/aime_2025", ["MathArena/aime_2025_I", "MathArena/aime_2025_II"], "opencompass/AIME2025"], None),
        ("math-500-level5", ["HuggingFaceH4/MATH-500"], lambda row: str(row.get("level", "")).replace("Level", "").strip() == "5"),
    ]
    chosen = []
    for source, names, keep in plan:
        try:
            rows, origin = rows_of(names)
            texts = [field(row, ("problem", "question")) for row in rows if keep is None or keep(row)]
            texts = list(dict.fromkeys(text for text in texts if text))          # no repeats, order kept
            chosen.extend((source, text + MATH_SUFFIX) for text in texts)
            notes[source] = f"{len(texts)} problems from {origin}"
        except Exception as error:
            notes[source] = f"FAILED: {type(error).__name__}: {str(error)[:300]}"
    return chosen


def distinct_share(tokens, n=4):
    """Share of different 4-token windows: low values mean the text repeats itself."""
    windows = [tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0))]
    return round(len(set(windows)) / max(len(windows), 1), 3)


def run(draft, target, encode, stops, prompts, args, save, report):
    """Decode every prompt with the probe attached. save(sequences, lengths, sources) and
    report(probe, answers, started, done, seconds) are called every few answers."""
    probe = MemoryProbe(capacity=args.capacity)
    sequences, lengths, sources, answers = [], [], [], []
    seconds = {}  # time spent drafting, in the probe, and verifying, summed over every round
    started = time.time()
    stop_set = set(stops or [])
    for source, prompt in prompts:
        if time.time() - started > args.max_hours * 3600:
            break
        input_ids = encode(prompt)
        if input_ids.shape[1] > args.max_prompt_tokens:
            continue
        probe.begin()
        clock = time.time()
        result = dflash_generate_redraft(draft, target, input_ids, args.max_new_tokens, stops,
                                         improver=probe, block_size=16, return_stats=True, timers=seconds)
        output = result.output_ids[0].tolist()
        generated = output[input_ids.shape[1]:]
        ended = bool(generated) and generated[-1] in stop_set
        scored = probe.finish(result.output_ids, ended=ended)
        sequences.append(torch.tensor(output, dtype=torch.int32))
        lengths.append(int(input_ids.shape[1]))
        sources.append(source)
        answers.append({
            "source": source, "prompt_tokens": int(input_ids.shape[1]), "tokens": len(generated),
            "rounds": int(result.rounds), "ended_by_itself": ended, "repetition_from": scored["loop_start"],
            "distinct_4_token_windows": distinct_share(generated),
            "distinct_4_token_windows_last_quarter": distinct_share(generated[len(generated) * 3 // 4:]),
            "seconds": round(time.time() - clock, 1), "bins": scored["bins"],
        })
        if len(answers) == 1 or len(answers) % args.save_every == 0:
            save(sequences, lengths, sources)
            report(probe, answers, started, False, seconds)
    save(sequences, lengths, sources)
    report(probe, answers, started, True, seconds)
    return probe, answers


def status_of(probe, answers, started, done, seconds=None):
    tokens = sorted(a["tokens"] for a in answers)
    status = {
        "done": done, "answers": len(answers), "hours": round((time.time() - started) / 3600, 3),
        "tokens_total": sum(tokens),
        "tokens_median": tokens[len(tokens) // 2] if tokens else 0,
        "answers_over_2000_tokens": sum(t > 2000 for t in tokens),
        "answers_over_4000_tokens": sum(t > 4000 for t in tokens),
        "answers_that_did_not_end_by_themselves": sum(not a["ended_by_itself"] for a in answers),
        "answers_that_fell_into_repetition": sum(a["repetition_from"] is not None for a in answers),
        "by_source": {},
    }
    for source in sorted({a["source"] for a in answers}):
        mine = sorted(a["tokens"] for a in answers if a["source"] == source)
        status["by_source"][source] = {
            "answers": len(mine), "tokens_median": mine[len(mine) // 2],
            "fell_into_repetition": sum(a["repetition_from"] is not None for a in answers if a["source"] == source)}
    status["memory_read_cost"] = read_cost(probe)
    rounds = (seconds or {}).get("rounds_one", 0)
    if rounds:
        status["milliseconds_per_round"] = {
            "drafter": 1000 * seconds.get("draft", 0.0) / rounds, "probe": 1000 * seconds.get("improve", 0.0) / rounds,
            "target_check": 1000 * seconds.get("verify_one", 0.0) / rounds}
    status.update(summarize(probe.totals, probe.delays, probe.columns))
    return status


def write_json(path, value, **options):
    with open(path + ".tmp", "w") as handle:
        json.dump(value, handle, **options)
    os.replace(path + ".tmp", path)


def other_work(name, folder):
    """The line of nexus/settings/instead.txt for this job, as [script, arguments...], or None.

    A line reads `job-name: script.py --its arguments`."""
    path = os.path.join(folder, "instead.txt")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        for line in handle:
            job, colon, work = line.split("#")[0].partition(":")
            if colon and job.strip() == name and work.split():
                return [os.path.expanduser(part) for part in work.split()]
    return None


def stepped_aside(results_dir, folder=os.path.join("nexus", "settings"), scripts="nexus"):
    """A queued job cannot be taken back or moved, but its plans can change before it starts.

    If the job's name, or its name without the number at the end, is a line of nexus/settings/skip.txt,
    the job writes that it stepped aside and does nothing else. If nexus/settings/instead.txt gives the
    job other work (a line `job-name: script.py arguments`), it runs that script instead, with the
    results going to the job's own results folder: a job that waits near the front of the queue can
    so be used for something more urgent. Returns whether the job's own work should be left undone."""
    if os.environ.get("SPECDEC_OTHER_WORK"):           # this process is the other work: it does its own job
        return False
    name = os.path.basename(os.path.normpath(results_dir))
    work = other_work(name, folder)
    if work is not None:
        import subprocess

        command = [sys.executable, os.path.join(scripts, work[0]), *work[1:], "--results-dir", results_dir]
        with open(os.path.join(results_dir, "instead.txt"), "w") as handle:
            handle.write(" ".join(command) + "\n")
        code = subprocess.run(command, env={**os.environ, "SPECDEC_OTHER_WORK": "1"}).returncode
        if code != 0:
            raise SystemExit(code)
        return True
    path = os.path.join(folder, "skip.txt")
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        listed = {line.split("#")[0].strip() for line in handle}
    if name not in listed and name.rsplit("-", 1)[0] not in listed:
        return False
    write_json(os.path.join(results_dir, "status.json"),
               {"done": True, "stepped_aside": True, "why": "listed in nexus/settings/skip.txt"}, indent=1)
    return True


def prepare_prompts(args):
    """Build the prompt list once, so that every shard cuts the same list."""
    from transformers import AutoTokenizer

    notes = {}
    prompts = load_bench_prompts(notes) if args.set == "bench" else load_long_prompts(args.total, args.seed, notes)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    kept, sizes, dropped = [], {}, {}
    for source, prompt in prompts:
        text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], args.reasoning)
        size = len(tokenizer.encode(text, add_special_tokens=False))
        if size > args.max_prompt_tokens:
            dropped[source] = dropped.get(source, 0) + 1
            continue
        kept.append([source, prompt])
        sizes.setdefault(source, []).append(size)
    os.makedirs(os.path.dirname(args.prompts_file), exist_ok=True)
    write_json(args.prompts_file, {"notes": notes, "prompts": kept})
    summary = {"loaded": notes, "dropped_for_length": dropped, "prompts": len(kept), "by_source": {}}
    for source, values in sorted(sizes.items()):
        values = sorted(values)
        examples = [prompt[:300] for name, prompt in kept if name == source][:2]
        summary["by_source"][source] = {"prompts": len(values), "prompt_tokens_median": values[len(values) // 2],
                                        "prompt_tokens_longest": values[-1], "examples": examples}
    write_json(os.path.join(args.results_dir, "prompts.json"), summary, indent=1)
    if any(str(note).startswith("FAILED") for note in notes.values()):
        raise SystemExit("a prompt source failed to load: " + json.dumps(notes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--total", type=int, default=600)
    parser.add_argument("--set", default="long", choices=("long", "bench"),
                        help="long: the development prompts; bench: AIME 2024, AIME 2025 and MATH-500 level 5")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reasoning", default="on")
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-prompt-tokens", type=int, default=3000)
    parser.add_argument("--capacity", type=int, default=32768)
    parser.add_argument("--max-hours", type=float, default=9.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--only-write-prompts", action="store_true")
    parser.add_argument("--out-dir")
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    if args.only_write_prompts:
        prepare_prompts(args)
        return
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(args.prompts_file) as handle:
        prepared = json.load(handle)
    notes = prepared["notes"]
    prompts = [tuple(pair) for pair in prepared["prompts"]][args.shard :: args.num_shards]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "prompt_sources": notes,
                             "prompts_in_this_shard": len(prompts)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    stops = stop_token_ids(target, tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    shard_path = os.path.join(args.out_dir, f"shard{args.shard}of{args.num_shards}.pt")

    def encode(prompt):
        text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], args.reasoning)
        return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    def save(sequences, lengths, sources):
        torch.save({"sequences": sequences, "prompt_lengths": lengths, "sources": sources}, shard_path + ".tmp")
        os.replace(shard_path + ".tmp", shard_path)

    def report(probe, answers, started, done, seconds):
        status = status_of(probe, answers, started, done, seconds)
        status["prompt_sources"] = notes
        status["prompts_in_this_shard"] = len(prompts)
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        write_json(os.path.join(args.results_dir, "answers.json"),
                   {"columns": probe.columns, "answers": answers}, separators=(",", ":"))

    run(draft, target, encode, stops, prompts, args, save, report)


if __name__ == "__main__":
    main()
