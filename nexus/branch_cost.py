"""What "branching inside the drafter" would cost, measured before anything is trained.

The method under test (November 27): run the drafter's first k layers on the block; read the places'
candidates out of the states after layer k (final norm and head); pick the four likely breaks by
those candidates' confidences (dflash/redraft.py: break_scores); make a branch for each break: a copy
of the block's states after layer k with, at the places up to the break, the difference between the
embedding of the branch's token (the first choices before the break, the second choice at it) and
the mask's embedding added; run the remaining layers on the block and the four branches together;
read the branches' continuations out. No second pass through the drafter.

This probe times every part on a real round's context, with the same code for every setting, so
that the differences are the costs of the parts (the absolute numbers include projecting the whole
context window, which decoding does once and keeps in a cache; the differences do not depend on it):
  draft              all layers on the block (the first run as it is)
  draft+readout      the first run plus the readout and the choice of breaks at layer k (no branches)
  split@k            the method: k layers on the block, readout, four branches, the rest on five blocks
  draft+second       the first run plus a second whole pass on the four branches (the reference cost)
and the target's check with 64, 76, 80 and 128 rows after the same context (what the branches'
continuations add to the check). Each number is the median of `--repeats` timed calls after warming
up. Results: status.json in --results-dir, by context length, with the token gain each setting would
need to break even against DDTree's round at that context (DDTree = draft + 1.6 ms + the 64-row check).
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dflash.benchmark import load_transformers_models  # noqa: E402
from dflash.model import _crop_to, _draft_value, _make_cache, _output_head, _raw_input_embeddings  # noqa: E402
from dflash.redraft import break_scores  # noqa: E402
from dflash.replay import target_features  # noqa: E402
from nexus.fit_gate import split_of  # noqa: E402
from nexus.probe_memory import stepped_aside, write_json  # noqa: E402
from nexus.redraft_long import find_drafter  # noqa: E402
from nexus.replay_memory import load_answers  # noqa: E402
from nexus.train_every import _positions, _sees  # noqa: E402

TREE_MS = 1.6          # DDTree's build, measured (docs/DIRECTION.md, November 18)


def projected_context(drafter, features, low, high):
    """The target's states of the context as the drafter reads them (after fc and hidden_norm), once."""
    return drafter.hidden_norm(drafter.fc(features[:, low:high]))


def run_layers(drafter, context, hidden, starts, low, high, width, layers):
    """The drafter's layers `layers` (numbers) over the blocks whose states are `hidden` ([1, blocks * width, size]),
    each block at `starts` after the context [low, high). Returns the states after the last of them (no norm)."""
    blocks = starts.shape[0]
    sees = _sees(high - low, starts - low, torch.full_like(starts, width), width)
    empty = context[:, :0]
    out = drafter(target_hidden=empty, context_states=context, noise_embedding=hidden,
                  position_ids=_positions(low, high, starts, width), attention_mask=sees[None, None], layers=layers)
    return out                                                  # (the drafter's forward ends with its norm)


class Stages:
    """The parts of the method, each a function of a round, so that they can be timed alone and together."""

    def __init__(self, drafter, target, features, ids, start, low, width, breaks=4, tail=3):
        self.drafter, self.target, self.width, self.breaks, self.tail = drafter, target, width, breaks, tail
        self.head = _output_head(target)
        self.low, self.high = low, start
        self.context = projected_context(drafter, features, low, start)
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        self.scale = scale
        block = torch.full((1, width), drafter.mask_token_id, dtype=torch.long, device=ids.device)
        block[:, 0] = ids[0, start]
        self.block_tokens = block
        self.noise = _raw_input_embeddings(target, block, scale)                     # [1, width, size]
        self.mask_embedding = _raw_input_embeddings(target, torch.tensor([[drafter.mask_token_id]], device=ids.device), scale)[0, 0]
        self.start = torch.tensor([start], device=ids.device)
        self.layers = list(range(len(drafter.layers)))
        self.first_norm = None

    def _without_norm(self, run):
        """The drafter's forward ends with its norm; the states BEFORE it are needed to go on. The norm is undone by
        running the layers with the norm replaced by identity for the call."""
        norm, self.drafter.norm = self.drafter.norm, torch.nn.Identity()
        try:
            return run()
        finally:
            self.drafter.norm = norm

    def draft(self):
        """The first run: every layer on the block; the block's logits. Returns the final states (after the norm)."""
        return run_layers(self.drafter, self.context, self.noise, self.start, self.low, self.high, self.width, self.layers)

    def first_layers(self, k):
        """Layers 1 .. k on the block, states without the norm ([1, width, size])."""
        return self._without_norm(lambda: run_layers(self.drafter, self.context, self.noise, self.start, self.low, self.high,
                                                     self.width, self.layers[:k]))

    def readout(self, states):
        """Candidates from states ([1, width, size], not normed): the first and second choices and the first choice's
        probability of every place after the first ([width - 1] each)."""
        logits = self.drafter.compute_logits(self.drafter.norm(states[:, 1:]), self.head)[0].float()    # [places, vocab]
        probs = logits.softmax(dim=-1)
        best = probs.topk(2, dim=-1)
        return best.indices[:, 0], best.indices[:, 1], best.values[:, 0]

    def branches_of(self, states, first, second, confidence):
        """The four branches' states after layer k ([1, breaks * width, size]) and their tokens ([breaks, width])."""
        width, device = self.width, states.device
        scores = break_scores(confidence)                                                   # [places]
        breaks = scores.topk(min(self.breaks, scores.shape[0])).indices                     # place index 0.. : break there
        places = torch.arange(width - 1, device=device)[None, :]
        prefix = places < breaks[:, None]
        at = places == breaks[:, None]
        tokens = torch.where(prefix, first[None, :], torch.where(at, second[None, :], torch.full_like(first[None, :], self.drafter.mask_token_id)))
        tokens = torch.cat((self.block_tokens[:, :1].expand(tokens.shape[0], 1), tokens), dim=1)      # [breaks, width]
        embedded = _raw_input_embeddings(self.target, tokens, self.scale)                   # [breaks, width, size]
        revealed = torch.cat((torch.zeros(tokens.shape[0], 1, dtype=torch.bool, device=device), prefix | at), dim=1)
        shift = torch.where(revealed[..., None], embedded - self.mask_embedding, torch.zeros_like(embedded))
        branch_states = states.expand(tokens.shape[0], -1, -1) + shift.to(states.dtype)
        return branch_states.reshape(1, -1, states.shape[-1]), tokens, breaks

    def rest_layers(self, k, states, branch_states):
        """Layers k + 1 .. L on the block and the branches together; the final states of all ([1, 5 * width, size])."""
        hidden = torch.cat((states, branch_states), dim=1)
        starts = self.start.expand(1 + branch_states.shape[1] // self.width)
        return run_layers(self.drafter, self.context, hidden, starts, self.low, self.high, self.width, self.layers[k:])

    def continuation_logits(self, final, breaks):
        """The logits of the block's places and of each branch's `tail` places after its break."""
        width = self.width
        block_logits = self.drafter.compute_logits(final[:, 1:width], self.head)
        rows = final[0, width:].reshape(-1, width, final.shape[-1])                            # [breaks, width, size]
        # The places after the break, as slots of a row (slot 0 is the block's first token, place i is slot i + 1).
        at = (breaks[:, None] + 2 + torch.arange(self.tail, device=final.device)[None, :]).clamp(max=width - 1)
        wanted = rows.gather(1, at[..., None].expand(-1, -1, rows.shape[-1]))
        return block_logits, self.drafter.compute_logits(wanted, self.head)

    def second_pass(self, tokens):
        """The reference: a second whole pass of the drafter on the four branches (their tokens written in)."""
        embedded = _raw_input_embeddings(self.target, tokens, self.scale).reshape(1, -1, self.noise.shape[-1])
        starts = self.start.expand(tokens.shape[0])
        return run_layers(self.drafter, self.context, embedded, starts, self.low, self.high, self.width, self.layers)

    # The settings, each a whole round's drafting as one call.
    def setting_draft(self):
        final = self.draft()
        return self.drafter.compute_logits(final[:, 1:], self.head)

    def setting_draft_readout(self, k):
        states = self.first_layers(k)
        first, second, confidence = self.readout(states)
        _, _, breaks = self.branches_of(states, first, second, confidence)
        final = self._without_norm(lambda: run_layers(self.drafter, self.context, states, self.start, self.low, self.high,
                                                      self.width, self.layers[k:]))
        return self.drafter.compute_logits(self.drafter.norm(final[:, 1:]), self.head), breaks

    def setting_split(self, k):
        states = self.first_layers(k)
        first, second, confidence = self.readout(states)
        branch_states, tokens, breaks = self.branches_of(states, first, second, confidence)
        final = self.rest_layers(k, states, branch_states)
        return self.continuation_logits(final, breaks)

    def setting_draft_second(self):
        final = self.draft()
        logits = self.drafter.compute_logits(final[:, 1:], self.head)[0].float()
        probs = logits.softmax(dim=-1)
        best = probs.topk(2, dim=-1)
        _, tokens, breaks = self.branches_of(final, best.indices[:, 0], best.indices[:, 1], best.values[:, 0])
        again = self.second_pass(tokens)
        return logits, self.drafter.compute_logits(again[:, 1:], self.head)


def timed(call, repeats, device):
    for _ in range(3):
        call()
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        call()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - tick) * 1000.0)
    return round(statistics.median(times), 3)


def check_cost(target, ids, high, rows_list, repeats, device):
    """The target's check of N rows after the context ids[:, :high], by N: ms, median."""
    cache = _make_cache(target.config)
    positions = torch.arange(ids.shape[1], device=device)[None]
    chunk = 1024
    with torch.no_grad():
        for begin in range(0, high, chunk):
            target(ids[:, begin : min(begin + chunk, high)], position_ids=positions[:, begin : min(begin + chunk, high)],
                   past_key_values=cache, use_cache=True)
    out = {}
    for rows in rows_list:
        tokens = ids[:, high : high + rows] if ids.shape[1] >= high + rows else ids[:, high : high + 1].expand(1, rows)
        pos = positions[:, high : high + rows]

        def call():
            target(tokens, position_ids=pos, past_key_values=cache, use_cache=True)
            _crop_to(cache, high)

        out[str(rows)] = timed(call, repeats, device)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="opd")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation")
    parser.add_argument("--contexts", default="1024,4096,8192", help="tokens of text before the round")
    parser.add_argument("--window", type=int, default=2048, help="context the drafter reads (projected without a cache here)")
    parser.add_argument("--splits", default="2,3")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--answers", type=int, default=3, help="rounds taken from this many answers (medians over them)")
    parser.add_argument("--check-rows", default="64,76,80,128")
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target, draft, _ = load_transformers_models(args.model, find_drafter(args.draft), device)
    target.eval(), draft.eval()
    width = int(_draft_value(draft.config, "block_size", 16))
    contexts = [int(c) for c in args.contexts.split(",")]
    splits = [int(k) for k in args.splits.split(",")]
    rows_list = [int(r) for r in args.check_rows.split(",")]
    answers = [a for a in load_answers(args.data_dir) if args.part == "all" or split_of([a[0], a[1]]) == args.part]
    answers = [a for a in answers if a[2].shape[0] - a[3] > max(contexts) + 2 * width][: args.answers]
    results = {"layers": len(draft.layers), "width": width, "window": args.window, "contexts": {}, "how_to_read": (
        "ms by setting, median over rounds of the median of repeats; split@k = k layers on the block, readout, four branches, the "
        "rest on five blocks; draft+second = the first run and a whole second pass on the branches; check_ms by rows; "
        "break_even: the relative gain in tokens a round a setting needs against DDTree (draft + 1.6 ms + the 64-row check)")}
    with torch.inference_mode():
        for context in contexts:
            per = {"draft": [], "draft+second": [], "check": {str(r): [] for r in rows_list}}
            for k in splits:
                per[f"draft+readout@{k}"], per[f"split@{k}"] = [], []
            for name, index, ids, prompt_length, *rest in answers:
                ids = ids.to(device)[None]
                start = prompt_length + context
                features = target_features(target, draft, ids[:, : start + 1], top=0)
                stages = Stages(draft, target, features, ids, start, max(start - args.window, 0), width)
                per["draft"].append(timed(stages.setting_draft, args.repeats, device))
                per["draft+second"].append(timed(stages.setting_draft_second, args.repeats, device))
                for k in splits:
                    per[f"draft+readout@{k}"].append(timed(lambda k=k: stages.setting_draft_readout(k), args.repeats, device))
                    per[f"split@{k}"].append(timed(lambda k=k: stages.setting_split(k), args.repeats, device))
                del stages, features
                for rows, ms in check_cost(target, ids, start, rows_list, args.repeats, device).items():
                    per["check"][rows].append(ms)
            summary = {key: round(statistics.median(v), 3) for key, v in per.items() if key != "check"}
            summary["check_ms"] = {rows: round(statistics.median(v), 3) for rows, v in per["check"].items()}
            ddtree = summary["draft"] + TREE_MS + summary["check_ms"][str(rows_list[0])]
            check_more = summary["check_ms"].get("76", summary["check_ms"][str(rows_list[0])]) - summary["check_ms"][str(rows_list[0])]
            summary["ddtree_round_ms"] = round(ddtree, 3)
            summary["break_even_gain"] = {key: round((summary[key] - summary["draft"] + (max(check_more, 0.0) if "readout" not in key else 0.0)) / ddtree, 4)
                                          for key in summary if key.startswith("split@") or key.startswith("draft+")}
            results["contexts"][str(context)] = summary
            write_json(os.path.join(args.results_dir, "status.json"), {"done": False, **results})
    write_json(os.path.join(args.results_dir, "status.json"), {"done": True, **results})


if __name__ == "__main__":
    main()
