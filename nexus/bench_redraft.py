"""End-to-end speed of ways to improve a draft using the draft itself.

Every variant runs on the same prompts, one after another, on one GPU. The first
draft is always kept; a variant adds an alternative draft and the target checks
both in one pass. Reports real tokens per second, measured the same way as
`dflash benchmark` (one over the mean time per output token).
"""

import argparse
import json
import statistics
import sys

import torch
from tqdm import tqdm

from dflash.benchmark import (
    _select_dataset,
    apply_chat_template,
    load_and_process_dataset,
    load_transformers_models,
    stop_token_ids,
)
from dflash.redraft import Gated, Resample, SelfRedraft, dflash_generate_redraft


def make_improver(name, second_model=None):
    """Variant names:
    dflash                 no second draft
    self-keepN             same drafter, shown the first N draft tokens (N or "confidence")
    second-keepN           a separate DFlash-architecture checkpoint, same idea
    runnerup               same drafter, shown the draft's SECOND choice at slot 1
    randomcond             same drafter, shown a random token at slot 1
    softcond               same drafter, shown the probability-weighted mix at slot 1
    resample               no second pass: resample slots 2.. from the first pass's probabilities
    gated:FILE             self-keep1 behind a confidence gate with weights from FILE (json)
    """
    if name == "dflash":
        return None
    if name == "runnerup":
        return SelfRedraft(keep=1, condition="runner_up")
    if name == "randomcond":
        return SelfRedraft(keep=1, condition="random")
    if name == "softcond":
        return SelfRedraft(keep=1, condition="soft")
    if name == "resample":
        return Resample()
    if name.startswith("gated:"):
        with open(name.split(":", 1)[1]) as handle:
            gate = json.load(handle)
        return Gated(SelfRedraft(keep=1), gate["weights"], gate["bias"], gate.get("threshold", 0.0))
    kind, _, keep = name.partition("-keep")
    keep = keep if keep == "confidence" else int(keep)
    if kind == "self":
        return SelfRedraft(keep=keep)
    if kind == "second":
        return SelfRedraft(model=second_model, keep=keep)
    raise ValueError(f"Unknown variant {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--second-draft", default=None, help="DFlash-architecture checkpoint used by second-* variants")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--reasoning", default="off")
    parser.add_argument("--variants", default="dflash,self-keep1,self-keepconfidence")
    parser.add_argument("--out", required=True)
    parser.add_argument("--analysis-samples", type=int, default=0,
                        help="After the speed run, rerun this many prompts with per-round logging and timers")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target, draft, tokenizer = load_transformers_models(args.model, args.draft, device)
    second = None
    if args.second_draft:
        second = type(draft).from_pretrained(
            args.second_draft, attn_implementation="sdpa", dtype=torch.bfloat16,
        ).to(device).eval()
    variants = args.variants.split(",")
    stops = stop_token_ids(target, tokenizer)
    dataset = _select_dataset(load_and_process_dataset(args.dataset), args.max_samples)

    def generate(name, input_ids, **extra):
        return dflash_generate_redraft(
            draft, target, input_ids, args.max_new_tokens, stops,
            improver=make_improver(name, second), block_size=args.block_size, return_stats=True, **extra,
        )

    def encode(messages):
        text = apply_chat_template(tokenizer, messages, args.reasoning)
        return tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(device)

    warmup = encode([{"role": "user", "content": dataset[0]["turns"][0]}])
    for name in variants:
        generate(name, warmup)

    records = {name: [] for name in variants}
    mismatches = {name: 0 for name in variants}
    turns = 0
    for instance in tqdm(dataset, file=sys.stderr):
        messages = []
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_ids = encode(messages)
            turns += 1
            reference = None
            for name in variants:
                result = generate(name, input_ids)
                if reference is None:
                    reference = result.output_ids
                elif not torch.equal(result.output_ids, reference):
                    mismatches[name] += 1
                records[name].append({
                    "tokens": result.num_output_tokens,
                    "time_per_token": result.time_per_output_token,
                    "rounds": result.rounds,
                    "rounds_with_alternative": result.rounds_with_alternative,
                    "alternative_wins": result.alternative_wins,
                    "alternative_extra_tokens": result.alternative_extra_tokens,
                })
            generated = reference[0, input_ids.shape[1]:]
            messages.append({"role": "assistant", "content": tokenizer.decode(generated, skip_special_tokens=True)})

    summary = {"args": vars(args), "turns": turns, "variants": {}}
    base_tps = None
    lines = [f"dataset={args.dataset} turns={turns} max_new_tokens={args.max_new_tokens} block={args.block_size}"]
    for name in variants:
        rows = records[name]
        tps = 1.0 / statistics.mean(r["time_per_token"] for r in rows)
        tokens = sum(r["tokens"] for r in rows)
        rounds = sum(r["rounds"] for r in rows)
        tried = sum(r["rounds_with_alternative"] for r in rows)
        wins = sum(r["alternative_wins"] for r in rows)
        extra = sum(r["alternative_extra_tokens"] for r in rows)
        base_tps = tps if base_tps is None else base_tps
        entry = {
            "tokens_per_second": tps,
            "speed_vs_first_variant": tps / base_tps,
            "tokens_per_round": tokens / rounds,
            "rounds": rounds,
            "rounds_with_alternative": tried,
            "alternative_wins": wins,
            "alternative_extra_tokens": extra,
            "outputs_differing_from_first_variant": mismatches[name],
        }
        summary["variants"][name] = entry
        lines.append(
            f"{name:22s} {tps:7.2f} tok/s  x{tps / base_tps:5.3f}  tokens/round {tokens / rounds:5.2f}  "
            f"alt tried {tried}/{rounds}  alt won {wins}  extra tokens {extra}  "
            f"outputs differing {mismatches[name]}/{turns}"
        )
    # Second pass over the first prompts with instrumentation switched on. Kept
    # apart from the speed run so the extra bookkeeping cannot distort the timing.
    if args.analysis_samples:
        summary["timing_ms_per_round"] = {}
        for name in variants:
            timers, log = {}, []
            for instance in dataset[: args.analysis_samples]:
                input_ids = encode([{"role": "user", "content": instance["turns"][0]}])
                generate(name, input_ids, round_log=log, timers=timers)
            rounds = timers.get("rounds_one", 0) + timers.get("rounds_two", 0)
            timing = {
                "rounds": rounds,
                "draft": 1000 * timers.get("draft", 0.0) / max(rounds, 1),
                "improve_per_round": 1000 * timers.get("improve", 0.0) / max(rounds, 1),
                "verify_one_draft": 1000 * timers.get("verify_one", 0.0) / max(timers.get("rounds_one", 0), 1),
                "verify_two_drafts": 1000 * timers.get("verify_two", 0.0) / max(timers.get("rounds_two", 0), 1),
                "rows_one_draft": timers.get("rows_one", 0) / max(timers.get("rounds_one", 0), 1),
                "rows_two_drafts": timers.get("rows_two", 0) / max(timers.get("rounds_two", 0), 1),
                "rounds_two_drafts": timers.get("rounds_two", 0),
            }
            summary["timing_ms_per_round"][name] = timing
            lines.append(
                f"timing {name:20s} draft {timing['draft']:.2f} ms  improve {timing['improve_per_round']:.2f} ms  "
                f"verify one draft {timing['verify_one_draft']:.2f} ms ({timing['rows_one_draft']:.1f} rows)  "
                f"verify two drafts {timing['verify_two_drafts']:.2f} ms ({timing['rows_two_drafts']:.1f} rows)"
            )
            if name != "dflash":
                with open(f"{args.out}.rounds.{name.replace(':', '_').replace('/', '_')}.jsonl", "w") as handle:
                    for entry in log:
                        handle.write(json.dumps(entry, separators=(",", ":")) + "\n")

    with open(args.out + ".json", "w") as handle:
        json.dump(summary, handle, indent=1)
    with open(args.out + ".txt", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
