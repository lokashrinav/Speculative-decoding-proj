"""What the target's check of several drafts in one pass costs, and what it does to the numbers.

    python nexus/time_check.py --data-dir ~/specdec-data/long1 --results-dir nexus/results/NAME

Real decoding with alternative drafts (dflash/redraft.py) lays their tokens after the first draft in
the target's one verification pass, with a mask that says who sees whom. Two things were seen in the
first real runs (nexus/results/long-opd-0b, -1b):
  - the pass took about 9 ms longer on long texts, whether it read 32 tokens or 62;
  - the tokens differed from plain decoding's after a few hundred tokens.
This script takes the longest saved answer and, with the target's cache filled to several lengths,

1. TIMES the check in several forms (median of --repeats passes, the cache cut back after each):
     chain16            16 tokens read in order, no mask given (plain decoding)
     chain16_mask       the same with the mask written out (what a given mask costs by itself)
     chain16_float      the same, the mask as numbers (0 and the lowest value) and not true/false
     chain20            4 tokens read again + 16 (the pass after an alternative has won)
     chain32 ... chain256   32, 64, 128 and 256 tokens in order, no mask (what more rows cost:
                        a tree of drafts with that many nodes costs the same, see tree32 and chain32)
     tree32             16 + four alternatives of 4 tokens (the swap with a tail of 3)
     tree32_float       the same, the mask as numbers
     tree54             16 + four alternatives to the end of the block
     build_mask         only writing the mask of tree32

2. COMPARES THE NUMBERS on --blocks blocks per length: the target's scores for a token on the row of
   an alternative against its scores for the same token read in order (as plain decoding and the
   second reading of dflash_generate_redraft read it), and the first draft's rows with and without
   alternatives after them. For each: how many rows are exactly equal, the largest difference, how
   often the first choice differs, and how far apart the two best tokens were where it did.
"""
import argparse
import os
import statistics
import time

import torch

from dflash.benchmark import load_transformers_models
from dflash.model import _make_cache
from nexus.probe_memory import stepped_aside, write_json
from nexus.replay_memory import load_answers

BLOCK = 16
SPLITS = (2, 5, 8, 11)          # where the four alternatives leave the first draft


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def tree(block, others, tail, cached, device):
    """The tokens, positions (relative to the block's start) and mask of one pass: the block, then
    for each split an alternative whose token at the split is others[n] and whose later tokens are
    the block's, up to `tail` tokens after the split (None: to the end)."""
    ids, positions = [block], [torch.arange(BLOCK, device=device)[None]]
    ends = [BLOCK if tail is None else min(BLOCK, split + 1 + tail) for split in SPLITS]
    rows = BLOCK + sum(end - split for split, end in zip(SPLITS, ends))
    visible = torch.ones((rows, cached + rows), dtype=torch.bool, device=device).tril(cached)
    at, where = BLOCK, []
    for number, (split, end) in enumerate(zip(SPLITS, ends)):
        branch = block[:, split:end].clone()
        branch[:, 0] = others[number]
        ids.append(branch)
        positions.append(torch.arange(split, end, device=device)[None])
        visible[at : at + end - split, cached + split : cached + at] = False
        where.append((split, end, at))
        at += end - split
    return torch.cat(ids, dim=1), torch.cat(positions, dim=1), visible[None, None], where


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--lengths", default="1000,2000,4000,8000,12000,16000")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--blocks", type=int, default=40)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = sorted(load_answers(args.data_dir), key=lambda answer: -answer[2].shape[0])
    ids = answers[0][2].long().to(device)[None]
    lengths = [int(length) for length in args.lengths.split(",") if int(length) + BLOCK * (args.blocks + 2) + 256 < ids.shape[1]]
    target, _, _ = load_transformers_models(args.model, args.draft, device)
    import transformers
    status = {"done": False, "answer_tokens": int(ids.shape[1]), "lengths": lengths, "torch": torch.__version__,
              "transformers": transformers.__version__, "attention": getattr(target.config, "_attn_implementation", None),
              "dtype": str(next(target.parameters()).dtype), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "timing_ms": {}, "numbers": {}}
    path = os.path.join(args.results_dir, "status.json")
    write_json(path, status, indent=1)
    lowest = torch.finfo(next(target.parameters()).dtype).min
    positions_all = torch.arange(ids.shape[1], device=device)[None]

    cache, filled = _make_cache(target.config), 0

    def fill(to):
        nonlocal filled
        while filled < to:
            step = min(1024, to - filled)
            target(ids[:, filled : filled + step], position_ids=positions_all[:, filled : filled + step],
                   past_key_values=cache, use_cache=True)
            filled += step

    def check(tokens, positions, mask=None):
        """One verification pass on top of the filled cache; the cache is cut back after it."""
        output = target(tokens, position_ids=filled + positions, past_key_values=cache, use_cache=True,
                        output_hidden_states=True, attention_mask=mask)
        cache.crop(-tokens.shape[1])
        return output.logits

    def in_order(count, device=device):
        return torch.arange(count, device=device)[None]

    def as_numbers(mask, like):
        return torch.zeros(mask.shape, dtype=like.dtype, device=mask.device).masked_fill(~mask, lowest)

    for length in lengths:
        fill(length)
        block = ids[:, length : length + BLOCK]
        others = [int(ids[0, length + split + 1]) for split in SPLITS]      # any other token will do for timing
        like = next(target.parameters())
        short = tree(block, others, 3, length, device)
        long = tree(block, others, None, length, device)
        causal = torch.ones((BLOCK, length + BLOCK), dtype=torch.bool, device=device).tril(length)[None, None]
        forms = {
            "chain16": lambda: check(block, in_order(BLOCK)),
            "chain16_mask": lambda: check(block, in_order(BLOCK), causal),
            "chain16_float": lambda: check(block, in_order(BLOCK), as_numbers(causal, like)),
            "chain20": lambda: check(ids[:, length : length + 20], in_order(20)),
            "chain32": lambda: check(ids[:, length : length + 32], in_order(32)),
            "chain64": lambda: check(ids[:, length : length + 64], in_order(64)),
            "chain128": lambda: check(ids[:, length : length + 128], in_order(128)),
            "chain256": lambda: check(ids[:, length : length + 256], in_order(256)),
            "tree32": lambda: check(short[0], short[1], short[2]),
            "tree32_float": lambda: check(short[0], short[1], as_numbers(short[2], like)),
            "tree54": lambda: check(long[0], long[1], long[2]),
            "build_mask": lambda: tree(block, others, 3, length, device),
        }
        row = {}
        for _ in range(3):                                   # warm-up
            for form in forms.values():
                form()
        for name, form in forms.items():
            times = []
            for _ in range(args.repeats):
                began = clock(device)
                form()
                times.append(1000 * (clock(device) - began))
            row[name] = round(statistics.median(times), 2)
        status["timing_ms"][str(length)] = row

        # The numbers: rows of alternatives against the same tokens read in order.
        tally = {key: {"rows": 0, "equal": 0, "largest_difference": 0.0, "first_choice_differs": 0, "gaps_where_it_differs": []}
                 for key in ("alternative_rows", "first_draft_rows_with_alternatives", "first_draft_rows_with_a_mask")}

        def compare(key, got, reference):
            entry = tally[key]
            got, reference = got.float(), reference.float()
            entry["rows"] += got.shape[0]
            entry["equal"] += int((got == reference).all(dim=-1).sum())
            entry["largest_difference"] = max(entry["largest_difference"], float((got - reference).abs().max()))
            differs = got.argmax(dim=-1) != reference.argmax(dim=-1)
            entry["first_choice_differs"] += int(differs.sum())
            for index in differs.nonzero()[:, 0].tolist():
                best = reference[index].topk(2).values
                entry["gaps_where_it_differs"].append(round(float(best[0] - best[1]), 4))

        for number in range(args.blocks):
            fill(length + number * BLOCK)                    # the text grows by one block at a time
            block = ids[:, filled : filled + BLOCK]
            plain = check(block, in_order(BLOCK))[0]
            # other tokens at the splits: the target's own runner-up after the token before the split
            others = [int(plain[split - 1].topk(2).indices[1]) for split in SPLITS]
            tokens, positions, mask, where = tree(block, others, 3, filled, device)
            together = check(tokens, positions, mask)[0]
            compare("first_draft_rows_with_alternatives", together[:BLOCK], plain)
            causal = torch.ones((BLOCK, filled + BLOCK), dtype=torch.bool, device=device).tril(filled)[None, None]
            compare("first_draft_rows_with_a_mask", check(block, in_order(BLOCK), causal)[0], plain)
            for split, end, at in where:
                chain = torch.cat((block[:, :split], tokens[:, at : at + end - split]), dim=1)
                reference = check(chain, in_order(chain.shape[1]))[0, split:]
                compare("alternative_rows", together[at : at + end - split], reference)
        for entry in tally.values():
            gaps = sorted(entry.pop("gaps_where_it_differs"))
            entry["gaps_where_it_differs"] = {"count": len(gaps), "median": gaps[len(gaps) // 2] if gaps else None,
                                              "largest": gaps[-1] if gaps else None}
        status["numbers"][str(length)] = tally
        write_json(path, status, indent=1)
    status["done"] = True
    write_json(path, status, indent=1)


if __name__ == "__main__":
    main()
