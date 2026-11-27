"""Can the drafter's alternatives be read out EARLY, before its last layers have run?

The question behind it ("branching inside the drafter", November 27): a branch that continues after
the drafter's second choice at a likely break needs that second choice BEFORE the remaining layers
run on the branch. The only place it can come from without a second pass is the drafter's own
states after an earlier layer, read with the final norm and the output head. This probe measures,
without training anything, what those early readouts are worth on the drafter's own rounds.

For every anchor (a round's start along a saved answer, one every `--every` tokens) the first run
is made once with hooks on every layer. For each layer k (states after layer k, then the final
norm and head; the last layer is the drafter's real output) it records, over the block's places:
  - how often the right token is the readout's first choice, in its top 2, in its top 8;
  - the four likely breaks by that readout's own confidences (dflash/redraft.py: break_scores,
    the rule the branch method would use), and for each selected branch (the readout's first
    choices before the break, its second choice at the break) whether the WHOLE prefix matches the
    answer, whether only the alternative token matches, and whether the break is a real break of
    that readout's chain (prefix right, first choice wrong there);
  - the same for the four breaks chosen by the LAST layer's confidences but with the early
    readout's tokens (does the selection or the readout cost more?);
  - the time of the readout (norm and head over a block of 16 places) and of the layers it skips.
Answers only EVALUATE the readouts; they never choose breaks or tokens. Results in --results-dir:
status.json with every number by layer and by place of the block.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dflash.benchmark import load_transformers_models  # noqa: E402
from dflash.model import _draft_value, _output_head, _raw_input_embeddings  # noqa: E402
from dflash.redraft import break_scores  # noqa: E402
from dflash.replay import target_features  # noqa: E402
from nexus.fit_gate import split_of  # noqa: E402
from nexus.probe_memory import stepped_aside, write_json  # noqa: E402
from nexus.redraft_long import find_drafter  # noqa: E402
from nexus.replay_memory import load_answers  # noqa: E402
from nexus.train_every import _positions, _sees  # noqa: E402


def first_run_all_layers(drafter, target, features, ids, starts, low, width):
    """The first runs of the rounds that start at `starts`, with the states after EVERY layer: a list, one
    [anchors, width, size] tensor per layer (the last one before the final norm)."""
    high, anchors = int(starts.max()), starts.shape[0]
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    block = torch.full((anchors, width), drafter.mask_token_id, dtype=ids.dtype, device=ids.device)
    block[:, 0] = ids[0, starts]
    noise = _raw_input_embeddings(target, block, scale).reshape(1, anchors * width, -1)
    sees = _sees(high - low, starts - low, torch.full_like(starts, width), width)
    kept = {}
    hooks = [layer.register_forward_hook(lambda layer, came, output, k=k: kept.__setitem__(k, output))
             for k, layer in enumerate(drafter.layers)]
    try:
        drafter(target_hidden=features[:, low:high], noise_embedding=noise, position_ids=_positions(low, high, starts, width),
                attention_mask=sees[None, None])
    finally:
        for hook in hooks:
            hook.remove()
    out = []
    for k in range(len(drafter.layers)):
        states = kept[k][0] if isinstance(kept[k], tuple) else kept[k]
        out.append(states[:, -anchors * width :, :].reshape(anchors, width, -1))
    return out


class Tally:
    def __init__(self, layers, width, breaks):
        self.layers, self.width, self.breaks = layers, width, breaks
        places = width - 1
        self.by_layer = {k: {"places": 0, "top1": torch.zeros(places), "top2": torch.zeros(places), "top8": torch.zeros(places),
                             "chain": torch.zeros(places),                      # the first choices right up to and including the place
                             "own_breaks": {"branches": 0, "whole_prefix_right": 0, "alternative_right": 0, "real_break": 0,
                                            "useful": 0, "break_place_sum": 0},
                             "last_layers_breaks": {"branches": 0, "whole_prefix_right": 0, "alternative_right": 0, "real_break": 0,
                                                    "useful": 0}}
                         for k in range(layers)}
        self.rounds = 0

    def add(self, k, ranks, chosen_breaks, own):
        """ranks [anchors, places]: the rank of the right token in layer k's readout (0 = first choice). chosen_breaks
        [anchors, breaks]: the places i (0-based, 1 .. places - 1 of the block) after which a break is expected,
        the branch keeps places < i and takes the second choice at i."""
        row = self.by_layer[k]
        anchors = ranks.shape[0]
        right_so_far = (ranks == 0).long().cumprod(dim=1)                      # chain of first choices right through the place
        if own:                                                                # (the ranks are the same for both calls: counted once)
            row["places"] += anchors
            row["top1"] += (ranks == 0).float().sum(dim=0).cpu()
            row["top2"] += (ranks <= 1).float().sum(dim=0).cpu()
            row["top8"] += (ranks <= 7).float().sum(dim=0).cpu()
            row["chain"] += right_so_far.float().sum(dim=0).cpu()
        which = row["own_breaks" if own else "last_layers_breaks"]
        for b in range(chosen_breaks.shape[1]):
            i = chosen_breaks[:, b]                                               # [anchors], a place index 0 .. places - 1
            prefix_right = torch.where(i > 0, right_so_far.gather(1, (i - 1).clamp(min=0)[:, None])[:, 0], torch.ones_like(i))
            alternative_right = ranks.gather(1, i[:, None])[:, 0] == 1
            first_wrong = ranks.gather(1, i[:, None])[:, 0] != 0
            which["branches"] += anchors
            which["whole_prefix_right"] += int((prefix_right.bool() & alternative_right).sum())
            which["alternative_right"] += int(alternative_right.sum())
            which["real_break"] += int((prefix_right.bool() & first_wrong).sum())
            which["useful"] += int((prefix_right.bool() & alternative_right).sum())      # (the same as whole_prefix_right: kept for reading)
            if own:
                which["break_place_sum"] += int(i.sum())

    def summary(self):
        out = {}
        for k, row in self.by_layer.items():
            n = max(row["places"], 1)
            by_place = lambda t: [round(float(v) / n, 4) for v in t]                 # noqa: E731
            entry = {"anchors": row["places"],
                     "top1": round(float(row["top1"].sum()) / (n * self.width - n), 4),
                     "top2": round(float(row["top2"].sum()) / (n * self.width - n), 4),
                     "top8": round(float(row["top8"].sum()) / (n * self.width - n), 4),
                     "tokens_right_in_a_row": round(float(row["chain"].sum()) / n, 3),
                     "top1_by_place": by_place(row["top1"]), "top2_by_place": by_place(row["top2"]),
                     "chain_by_place": by_place(row["chain"])}
            for name in ("own_breaks", "last_layers_breaks"):
                w = row[name]
                b = max(w["branches"], 1)
                entry[name] = {"branches": w["branches"],
                               "share_whole_prefix_right": round(w["whole_prefix_right"] / b, 4),
                               "share_alternative_right": round(w["alternative_right"] / b, 4),
                               "share_real_break_of_this_readouts_chain": round(w["real_break"] / b, 4)}
                if name == "own_breaks":
                    entry[name]["mean_break_place"] = round(w["break_place_sum"] / b, 2)
            out[f"after_layer_{k + 1}" if k + 1 < self.layers else "last_layer"] = entry
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="opd")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation")
    parser.add_argument("--max-answers", type=int, default=24)
    parser.add_argument("--every", type=int, default=16, help="an anchor every this many tokens of the answer")
    parser.add_argument("--max-anchors", type=int, default=256, help="per answer")
    parser.add_argument("--group", type=int, default=32, help="anchors per pass")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--breaks", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=1.5)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target, draft, _ = load_transformers_models(args.model, find_drafter(args.draft), device)
    target.eval(), draft.eval()
    head = _output_head(target)
    width = int(_draft_value(draft.config, "block_size", 16))
    layers = len(draft.layers)
    answers = [a for a in load_answers(args.data_dir) if args.part == "all" or split_of([a[0], a[1]]) == args.part][: args.max_answers]
    # The readouts are grouped by the source of the prompt as well (the four kinds of text).
    tallies = {"all": Tally(layers, width, args.breaks)}
    timing = {"readout_ms": []}
    started, done = time.time(), 0

    def status(final=False):
        out = {"done": final, "answers": done, "hours": round((time.time() - started) / 3600, 3), "layers": layers, "width": width,
               "breaks": args.breaks, "every": args.every, "draft": args.draft,
               "by_layer": tallies["all"].summary(),
               "by_source": {name: t.summary() for name, t in tallies.items() if name != "all"},
               "timing": {"readout_ms_per_block_of_16": round(sum(timing["readout_ms"]) / max(len(timing["readout_ms"]), 1), 3)},
               "how_to_read": "top1/top2/top8: the right token's rank in each layer's readout over the block's places 1..15; "
                              "own_breaks: the four likely breaks by that readout's confidences, share of branches whose whole "
                              "prefix (first choices before the break, second choice at it) is the answer's; last_layers_breaks: "
                              "breaks chosen by the last layer's confidences, tokens from the early readout."}
        write_json(os.path.join(args.results_dir, "status.json"), out)

    with torch.inference_mode():
        for name, index, ids, prompt_length, *rest in answers:
            if time.time() - started > args.max_hours * 3600:
                break
            source = rest[0] if rest else name
            ids = ids.to(device)[None] if ids.dim() == 1 else ids.to(device)
            features = target_features(target, draft, ids, top=0)
            last = ids.shape[1] - width - 1
            if last <= prompt_length:
                continue
            starts = torch.arange(prompt_length, last, args.every, device=device)[: args.max_anchors]
            tallies.setdefault(source, Tally(layers, width, args.breaks))
            for begin in range(0, starts.shape[0], args.group):
                group = starts[begin : begin + args.group]
                low = max(int(group.max()) - args.window, 0)
                states = first_run_all_layers(draft, target, features, ids, group, low, width)
                answer = ids[0][group[:, None] + torch.arange(1, width, device=device)[None, :]]       # [anchors, places]
                readouts = []
                for k in range(layers):
                    logits = draft.compute_logits(draft.norm(states[k][:, 1:]), head).float()           # [anchors, places, vocab]
                    probs = logits.softmax(dim=-1)
                    best = probs.topk(8, dim=-1)
                    rank = (probs > probs.gather(-1, answer[..., None])).sum(dim=-1)                   # rank of the right token
                    confidence = best.values[..., 0]                                                   # of the first choice, c_1 ..
                    scores = torch.stack([break_scores(confidence[a]) for a in range(confidence.shape[0])])     # [anchors, places]
                    breaks = scores.topk(min(args.breaks, scores.shape[1]), dim=-1).indices             # place i: break right after i accepted
                    readouts.append((rank, breaks))
                    del logits, probs
                last_breaks = readouts[-1][1]
                for k, (rank, breaks) in enumerate(readouts):
                    for t in (tallies["all"], tallies[source]):
                        t.add(k, rank, breaks, own=True)
                        t.add(k, rank, last_breaks, own=False)
                # Timing of one block's readout (norm and head over 16 places) and of each layer over a block.
                if device.type == "cuda" and begin == 0:
                    one = states[-1][:1]
                    for _ in range(3):
                        draft.compute_logits(draft.norm(one), head)
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    for _ in range(10):
                        draft.compute_logits(draft.norm(one), head)
                    torch.cuda.synchronize()
                    timing["readout_ms"].append((time.perf_counter() - tick) * 100.0)
            done += 1
            status()
    status(final=True)


if __name__ == "__main__":
    main()
