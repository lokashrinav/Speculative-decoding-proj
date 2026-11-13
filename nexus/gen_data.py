"""Build training data: prompts from public datasets, answers written by the target itself.

The drafter's job is to predict what the target will write, so the answers must
come from the target (greedy, thinking off), not from the original datasets.
None of the evaluation sets (GSM8K test, MATH-500, HumanEval, MT-Bench) is used.

Each shard is saved as a torch file: {"sequences": [int32 tensor], "prompt_lengths": [int], "sources": [str]}.
"""

import argparse
import json
import os
import random
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dflash.benchmark import apply_chat_template


def load_prompts(total, seed):
    """Roughly 40% chat, 30% math, 30% code. A source that fails to load is skipped and reported."""
    def stream(name, split, count, field):
        # Streaming reads only the rows we use, so nothing large lands in the home quota.
        rows = load_dataset(name, split=split, streaming=True).shuffle(seed=seed, buffer_size=10000)
        out = []
        for row in rows:
            out.append(row[field])
            if len(out) >= count:
                break
        return out

    plan = [
        ("ultrachat", 0.4, lambda: stream("HuggingFaceH4/ultrachat_200k", "train_sft", 3 * total, "prompt")),
        ("metamath", 0.3, lambda: stream("meta-math/MetaMathQA", "train", 3 * total, "query")),
        ("codealpaca", 0.3, lambda: [
            r["instruction"] + ("\n\n" + r["input"] if r.get("input") else "")
            for r in load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        ]),
    ]
    rng = random.Random(seed)
    loaded = []
    for name, share, loader in plan:
        try:
            prompts = [p.strip() for p in loader() if p and p.strip()]
            rng.shuffle(prompts)
            loaded.append((name, share, prompts))
            print(f"loaded {name}: {len(prompts)} prompts", file=sys.stderr)
        except Exception as error:  # report and continue with the other sources
            print(f"FAILED to load {name}: {type(error).__name__}: {error}", file=sys.stderr)
    if not loaded:
        raise SystemExit("no prompt source could be loaded")
    share_sum = sum(share for _, share, _ in loaded)
    chosen = []
    for name, share, prompts in loaded:
        count = min(len(prompts), round(total * share / share_sum))
        chosen.extend((name, prompt) for prompt in prompts[:count])
    rng.shuffle(chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--total", type=int, default=21000, help="prompts across all shards")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--status", required=True, help="small text file with progress, for the job runner")
    parser.add_argument("--max-hours", type=float, default=100.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"shard{args.shard}of{args.num_shards}.pt")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = load_prompts(args.total, args.seed)[args.shard :: args.num_shards]

    encoded = []
    for source, prompt in prompts:
        text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], "off")
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= args.max_prompt_tokens:
            encoded.append((source, ids))
    encoded.sort(key=lambda item: len(item[1]))  # similar lengths together, little padding
    batches = [encoded[i : i + args.batch_size] for i in range(0, len(encoded), args.batch_size)]
    random.Random(args.seed).shuffle(batches)  # so a partial run still covers all lengths and sources

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="sdpa", dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    sequences, prompt_lengths, sources = [], [], []
    started = time.time()
    tokens = 0

    def save():
        torch.save({"sequences": sequences, "prompt_lengths": prompt_lengths, "sources": sources}, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
        elapsed = time.time() - started
        with open(args.status, "w") as handle:
            json.dump({
                "shard": args.shard, "sequences": len(sequences), "of": len(encoded),
                "generated_tokens": tokens, "hours": round(elapsed / 3600, 3),
                "tokens_per_second": round(tokens / max(elapsed, 1e-9), 1),
                "per_source": {name: sources.count(name) for name in sorted(set(sources))},
            }, handle)

    for index, batch in enumerate(batches):
        width = max(len(ids) for _, ids in batch)
        input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (_, ids) in enumerate(batch):
            input_ids[row, width - len(ids) :] = torch.tensor(ids)
            attention[row, width - len(ids) :] = 1
        with torch.inference_mode():
            output = target.generate(
                input_ids.to(device), attention_mask=attention.to(device), do_sample=False,
                max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
            )
        for row, (source, ids) in enumerate(batch):
            answer = output[row, width:].tolist()
            for position, token in enumerate(answer):
                if token in eos_ids:
                    answer = answer[: position + 1]  # keep the stop token, drop the padding after it
                    break
            if len(answer) >= 4:
                sequences.append(torch.tensor(ids + answer, dtype=torch.int32))
                prompt_lengths.append(len(ids))
                sources.append(source)
                tokens += len(answer)
        if index % 10 == 0 or index == len(batches) - 1:
            save()
        if time.time() - started > args.max_hours * 3600:
            print("time budget reached, stopping early", file=sys.stderr)
            break
    save()


if __name__ == "__main__":
    main()
