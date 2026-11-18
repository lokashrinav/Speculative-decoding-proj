"""More text written by the target in thinking mode, many answers at once.

The trained parts of the second run (the layers that restart, the predictor of the states) and the
counts of which token follows which all learn from text the target wrote itself. The long answers of
specdec-data/long1 were decoded one at a time (366 of them are for training). A target pass costs
nearly the same for one answer as for twenty, so this script writes about twenty answers at a time,
plainly (no drafter), cut at --max-new-tokens.

The prompts are of the same kinds as long1's (nexus/probe_memory.py: load_long_prompts), with another
seed, and every prompt that long1 already has is left out, so that none of its validation answers is
written again. The list is built by the first job and saved (prompts.json); each job writes the
answers of its own share of it (--shard of --num-shards). The files have the format of nexus/gen_data.py:
{"sequences": [int32 tensor], "prompt_lengths": [int], "sources": [str]}.
"""

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # the saved keys and values grow every step

import torch  # noqa: E402

from nexus.probe_memory import stepped_aside, write_json  # noqa: E402


def without_known(encoded, known):
    """The prompts whose tokens are not among the known ones. encoded: [(source, tokens)]."""
    known = {tuple(tokens) for tokens in known}
    kept, seen = [], set()
    for source, tokens in encoded:
        key = tuple(tokens)
        if key in known or key in seen:
            continue
        seen.add(key)
        kept.append((source, tokens))
    return kept


def batches_by_memory(encoded, max_new_tokens, row_tokens, largest=64):
    """Prompts of similar length together; a batch has as many rows as fit: rows x (prompt + answer) <= row_tokens."""
    ordered = sorted(encoded, key=lambda item: len(item[1]))
    batches, start = [], 0
    while start < len(ordered):
        count = 1
        while (start + count < len(ordered) and count < largest
               and (count + 1) * (len(ordered[start + count][1]) + max_new_tokens) <= row_tokens):
            count += 1
        batches.append(ordered[start:start + count])
        start += count
    return batches


def cut_at_stop(answer, stops):
    """The answer up to and with its first stop token, and whether it has one."""
    for position, token in enumerate(answer):
        if token in stops:
            return answer[:position + 1], True
    return answer, False


def write_batch(target, batch, pad, stops, max_new_tokens, device):
    """Greedy answers to a batch of prompts: [(tokens of the answer, ended at a stop token)]."""
    width = max(len(tokens) for _, tokens in batch)
    input_ids = torch.full((len(batch), width), pad, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (_, tokens) in enumerate(batch):
        input_ids[row, width - len(tokens):] = torch.tensor(tokens)
        attention[row, width - len(tokens):] = 1
    with torch.inference_mode():
        output = target.generate(
            input_ids.to(device), attention_mask=attention.to(device), do_sample=False, num_beams=1,
            max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=sorted(stops),
        )
    return [cut_at_stop(output[row, width:].tolist(), stops) for row in range(len(batch))]


def write_batch_in_parts(target, batch, pad, stops, max_new_tokens, device):
    """write_batch, in two halves if the GPU's memory does not hold the whole batch."""
    try:
        return write_batch(target, batch, pad, stops, max_new_tokens, device)
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(batch) // 2
        return (write_batch_in_parts(target, batch[:half], pad, stops, max_new_tokens, device)
                + write_batch_in_parts(target, batch[half:], pad, stops, max_new_tokens, device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--reasoning", default="on")
    parser.add_argument("--total", type=int, default=8000, help="prompts across all shards, before the known ones are left out")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--row-tokens", type=int, default=60000,
                        help="rows x (prompt + answer) tokens a batch may hold in the GPU's memory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--known", action="append", default=[],
                        help="a folder of saved answers whose prompts are left out (may be given more than once)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompts-file", default="", help="the list of prompts, built by the first job (default: prompts.json in --out-dir)")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-hours", type=float, default=3.5)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dflash.benchmark import apply_chat_template, stop_token_ids
    from nexus.probe_memory import load_long_prompts
    from nexus.replay_memory import load_answers

    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    status_path = os.path.join(args.results_dir, "status.json")
    out_path = os.path.join(args.out_dir, f"shard{args.shard}of{args.num_shards}.pt")
    write_json(status_path, {"done": False, "answers": 0}, indent=1)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # The list of prompts is built once and saved beside the text, so that every job cuts the same list.
    prompts_file = args.prompts_file or os.path.join(args.out_dir, "prompts.json")
    if os.path.exists(prompts_file):
        with open(prompts_file) as handle:
            saved = json.load(handle)
        notes, prompts = saved["notes"], [tuple(pair) for pair in saved["prompts"]]
    else:
        notes = {}
        prompts = load_long_prompts(args.total, args.seed, notes)
        write_json(prompts_file, {"notes": notes, "total": args.total, "seed": args.seed,
                                  "prompts": [list(pair) for pair in prompts]})
    encoded, too_long = [], 0
    for source, prompt in prompts:
        text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], args.reasoning)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > args.max_prompt_tokens:
            too_long += 1
            continue
        encoded.append((source, tokens))
    known = []
    for folder in args.known:
        known += [sequence[:prompt_length].tolist()
                  for _, _, sequence, prompt_length, _ in load_answers(os.path.expanduser(folder))]
    fresh = without_known(encoded, known)
    batches = batches_by_memory(fresh, args.max_new_tokens, args.row_tokens)
    random.Random(args.seed).shuffle(batches)   # a run that stops early still has every length and source
    batches = batches[args.shard::args.num_shards]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    stops = set(stop_token_ids(target, tokenizer) or [])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    sequences, prompt_lengths, sources, ended = [], [], [], []
    tokens_written, batch_seconds = 0, []

    def save(done):
        torch.save({"sequences": sequences, "prompt_lengths": prompt_lengths, "sources": sources, "ended": ended},
                   out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
        elapsed = time.time() - started
        write_json(status_path, {
            "done": done, "shard": args.shard, "of_shards": args.num_shards, "answers": len(sequences),
            "batches": len(batch_seconds), "batches_of_this_shard": len(batches),
            "tokens_written": tokens_written, "answers_that_ended": sum(ended),
            "hours": round(elapsed / 3600, 3),
            "tokens_per_second": round(tokens_written / max(sum(batch_seconds), 1e-9), 1),
            "per_source": {name: sources.count(name) for name in sorted(set(sources))},
            "prompts": {"loaded": notes, "too_long": too_long, "already_known": len(encoded) - len(fresh),
                        "fresh": len(fresh), "known_folders": args.known},
            "max_new_tokens": args.max_new_tokens, "row_tokens": args.row_tokens, "saved_as": out_path,
        }, indent=1)

    for batch in batches:
        longest = max(batch_seconds, default=0.0)
        if time.time() - started + 1.15 * longest > args.max_hours * 3600:
            print("time budget reached, stopping early", file=sys.stderr)
            break
        clock = time.time()
        answers = write_batch_in_parts(target, batch, pad, stops, args.max_new_tokens, device)
        batch_seconds.append(time.time() - clock)
        for (source, tokens), (answer, stopped) in zip(batch, answers):
            if len(answer) < 4:
                continue
            sequences.append(torch.tensor(tokens + answer, dtype=torch.int32))
            prompt_lengths.append(len(tokens))
            sources.append(source)
            ended.append(bool(stopped))
            tokens_written += len(answer)
        save(False)
    save(True)


if __name__ == "__main__":
    main()
