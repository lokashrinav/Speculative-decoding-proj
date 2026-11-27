"""A second pass of the WHOLE drafter, trained the way D-Loop and D2SD train theirs.

    python nexus/train_loop.py --draft opd --data-dir ~/specdec-data/long1 --more-data ~/specdec-data/think1 \
        --recipe dloop --save opd-dloop --rank 256 --max-hours 3 --results-dir nexus/results/NAME

The published second passes retrain a block drafter so that it can fill the END of a block when the
BEGINNING is already filled in with tokens (D-Loop, arXiv 2610.06011: the same drafter does both passes;
D2SD, 2606.04446: a separate second drafter, the VP-Drafter). Their training: for an anchor in an answer
the target wrote, draw a prefix length l from a geometric law (short prefixes most often), fill the
block's places 1 .. l with the answer's true tokens, leave the rest as masks, and train the drafter to
write the masked places, with a cross-entropy that decays with the distance from the first masked place.
D-Loop adds a second view of the same block with NOTHING filled in (the ordinary first pass), supervised
on the places 1 .. l with the ordinary decay, so that one set of weights serves both passes.

Here the drafter's weights are changed through low-rank add-ons on all its linear layers (dflash/adapt.py:
the same add-ons the long-answer experiments used, saved as lora-<name>.pt), which every other script loads
with +fixed@NAME or --fixed NAME. --full trains all of the drafter's weights instead (D-Loop's own way; a
saved drafter is 1.2 GB, which the home folder cannot hold twice).

Recipes:
  dloop   both views, prefix lengths ~ beta^l over 0 .. width-2 (beta 0.7), decays tau_s = gamma = 7, as the paper.
          With l < 2 the prefix view is a whole ordinary block (the paper's fallback, at the same anchor).
  d2sd    the suffix view only (a separate second drafter: the first drafter stays what it is), beta and tau as
          the paper; the first pass is NOT trained, so in use the add-ons are switched on for the second pass only.
          (dflash/redraft.py does not switch add-ons per pass yet: a d2sd layer is measured as a second pass
          with the probe and in replay with the add-ons on for both passes, which is D-Loop's use of it.)

Measured on validation answers after training, at every prefix length: how often the first masked place's
first choice is the answer's token, and how many places in a row are right ("walk"); and the same for the
ordinary first pass, so that a change to it shows. Written to status.json with the training curve.
"""
import argparse
import math
import os
import random
import time

import torch

from dflash.adapt import attach_adapters, load_adapters, save_adapters
from dflash.benchmark import load_transformers_models
from dflash.model import _draft_value, _output_head, _raw_input_embeddings
from dflash.replay import target_features
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers
from nexus.train_every import _positions, _sees
from nexus.train_second import more_answers


def prefix_law(width: int, beta: float) -> torch.Tensor:
    """D-Loop's law of the prefix length: P(l) proportional to beta^l for l in 0 .. width - 2."""
    weights = torch.tensor([beta**length for length in range(width - 1)], dtype=torch.float)
    return weights / weights.sum()


def views(drafter, target, features, ids, starts: torch.Tensor, prefixes: torch.Tensor, low: int, width: int = 16):
    """One pass of the drafter over several blocks of an answer, each with the answer's true tokens at the
    places 1 .. prefixes[n] and masks after them; block n sees the target's states before its own start and its
    whole self. Returns the final states, [blocks, width, size]."""
    high, blocks = int(starts.max()), starts.shape[0]
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    slots = torch.arange(width, device=ids.device)[None, :]
    known = (slots >= 1) & (slots <= prefixes[:, None])
    block = torch.where(known, ids[0][(starts[:, None] + slots).clamp(max=ids.shape[1] - 1)],
                        torch.full((blocks, width), drafter.mask_token_id, dtype=ids.dtype, device=ids.device))
    block[:, 0] = ids[0, starts]
    kind = drafter.fc.weight.dtype                          # (a drafter trained in 32-bit numbers reads 16-bit states)
    entering = _raw_input_embeddings(target, block, scale).reshape(1, blocks * width, -1).to(kind)
    sees = _sees(high - low, starts - low, torch.full_like(starts, width), width)
    final = drafter(target_hidden=features[:, low:high].to(kind), noise_embedding=entering, position_ids=_positions(low, high, starts, width),
                    attention_mask=sees[None, None])[:, -blocks * width :, :]
    return final.reshape(blocks, width, -1)


class TeachLoop:
    """Trains the drafter (its add-ons or all of it) for a second pass with a prefix filled in."""

    def __init__(self, drafter, target, optimizer, recipe="dloop", beta=0.7, tau=7.0, gamma=7.0, group=16, width=16,
                 window=2048, seed=0, mixed=False):
        self.drafter, self.target, self.optimizer, self.mixed = drafter, target, optimizer, mixed
        self.recipe, self.beta, self.tau, self.gamma = recipe, beta, tau, gamma
        self.group, self.width, self.window = group, width, window
        self.head = _output_head(target)
        self.law = prefix_law(width, beta)
        self.generator = torch.Generator().manual_seed(seed)
        self.updates, self.loss_sum, self.loss_count = 0, 0.0, 0
        self.reset()

    def reset(self):
        """The counts of the measurement: per prefix length, how often the first masked place is right and the walk."""
        self.right = torch.zeros(self.width - 1, dtype=torch.float64)
        self.walk = torch.zeros(self.width - 1, dtype=torch.float64)
        self.seen = torch.zeros(self.width - 1, dtype=torch.float64)
        self.plain_right = torch.zeros(self.width - 1, dtype=torch.float64)    # the ordinary pass, by place
        self.plain_walk, self.plain_seen = 0.0, 0

    def progress(self) -> dict:
        seen = self.seen.clamp(min=1)
        out = {"updates": self.updates, "loss": round(self.loss_sum / max(self.loss_count, 1), 4)}
        if self.seen.sum() > 0:
            weight = self.law[: self.width - 1]
            share = (self.right / seen * weight).sum() / weight.sum()
            out.update(second_right_by_prefix=[round(float(value), 4) for value in (self.right / seen)],
                       second_walk_by_prefix=[round(float(value), 3) for value in (self.walk / seen)],
                       second_right=round(float(share), 4),
                       second_right_after_one=round(float(self.right[1] / seen[1]), 4),
                       second_walk_after_one=round(float(self.walk[1] / seen[1]), 3))
        if self.plain_seen:
            out.update(first_right_by_place=[round(float(value / self.plain_seen), 4) for value in self.plain_right],
                       first_walk=round(self.plain_walk / self.plain_seen, 3))
        return out

    def anchors_of(self, ids, prompt_length: int, count: int) -> torch.Tensor:
        """`count` places in the answer at which a block can start with its whole width inside the answer."""
        first, last = prompt_length, ids.shape[1] - self.width
        if last <= first:
            return torch.empty(0, dtype=torch.long)
        return torch.randint(first, last, (count,), generator=self.generator)

    def learn_from(self, ids, features, prompt_length: int, anchors: int, learn=True, every=0):
        """One answer: groups of blocks, each with a prefix length drawn from the law (dloop: two views a block)."""
        device = ids.device
        if every:
            starts_all = torch.arange(prompt_length, ids.shape[1] - self.width, every)
        else:
            starts_all = self.anchors_of(ids, prompt_length, anchors)
        if starts_all.numel() == 0:
            return
        starts_all = starts_all[torch.randperm(starts_all.numel(), generator=self.generator)] if learn else starts_all
        for begin in range(0, starts_all.numel(), self.group):
            starts = starts_all[begin : begin + self.group].to(device)
            prefixes = torch.multinomial(self.law, starts.numel(), replacement=True, generator=self.generator).to(device)
            if learn:
                self.step(ids, features, starts, prefixes)
            else:
                with torch.no_grad():
                    self.count(ids, features, starts, prefixes)

    def _loss(self, states, truth, prefixes, suffix: bool):
        """Cross-entropy over the masked places (suffix view) or the prefix's places (prefix view), with the
        paper's decays, summed; and the sum of the weights."""
        width = self.width
        slots = torch.arange(1, width, device=states.device)[None, :]                       # the places 1 .. width-1
        if suffix:
            weight = torch.where(slots > prefixes[:, None], torch.exp(-(slots - prefixes[:, None] - 1).float() / self.tau), torch.zeros_like(slots, dtype=torch.float))
        else:
            whole = (prefixes < 2)[:, None]                                                 # the fallback: a whole ordinary block
            weight = torch.where((slots <= prefixes[:, None]) | whole, torch.exp(-(slots - 1).float() / self.gamma), torch.zeros_like(slots, dtype=torch.float))
        rows = weight > 0
        logits = self.drafter.compute_logits(states[:, 1:][rows].to(self.head.weight.dtype), self.head).float()
        losses = torch.nn.functional.cross_entropy(logits, truth[:, 1:][rows], reduction="none")
        return (losses * weight[rows]).sum(), weight.sum()

    def step(self, ids, features, starts, prefixes):
        width, device = self.width, ids.device
        truth = ids[0][(starts[:, None] + torch.arange(width, device=device)[None, :]).clamp(max=ids.shape[1] - 1)]
        low = max(0, int(starts.min()) - self.window)
        total, weights = 0.0, 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.mixed):
            suffix_states = views(self.drafter, self.target, features, ids, starts, prefixes, low, width)
            loss, weight = self._loss(suffix_states, truth, prefixes, suffix=True)
            total, weights = total + loss, weights + weight
            del suffix_states
            if self.recipe == "dloop":
                prefix_states = views(self.drafter, self.target, features, ids, starts, torch.zeros_like(prefixes), low, width)
                loss, weight = self._loss(prefix_states, truth, prefixes, suffix=False)
                total, weights = total + loss, weights + weight
                del prefix_states
        total = total / (weights + 1e-6)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        self.updates += 1
        self.loss_sum += float(total.detach())
        self.loss_count += 1

    def count(self, ids, features, starts, prefixes):
        """The measurement: the second pass at the drawn prefix lengths, and the ordinary first pass."""
        width, device = self.width, ids.device
        truth = ids[0][(starts[:, None] + torch.arange(width, device=device)[None, :]).clamp(max=ids.shape[1] - 1)]
        low = max(0, int(starts.min()) - self.window)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.mixed):
            states = views(self.drafter, self.target, features, ids, starts, prefixes, low, width)
            chosen = self.drafter.compute_logits(states.to(self.head.weight.dtype), self.head).argmax(dim=-1)   # [blocks, width]
        right = (chosen[:, 1:] == truth[:, 1:])
        for number in range(starts.numel()):
            length = int(prefixes[number])
            after = right[number, length:]                                                   # places length+1 .. width-1
            self.seen[length] += 1
            self.right[length] += float(after[0])
            walk = 0
            for value in after.tolist():
                if not value:
                    break
                walk += 1
            self.walk[length] += walk
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.mixed):
            plain = views(self.drafter, self.target, features, ids, starts, torch.zeros_like(prefixes), low, width)
            chosen = self.drafter.compute_logits(plain.to(self.head.weight.dtype), self.head).argmax(dim=-1)
        right = (chosen[:, 1:] == truth[:, 1:])
        self.plain_right += right.sum(dim=0).double().cpu()
        self.plain_seen += starts.numel()
        for row in right.tolist():
            walk = 0
            for value in row:
                if not value:
                    break
                walk += 1
            self.plain_walk += walk


def load_saved_drafter(draft, folder: str) -> None:
    """The weights a stopped run saved (safetensors in its folder, as save_pretrained writes them), loaded over the
    drafter's, so that a run goes on from them."""
    import glob

    from safetensors.torch import load_file

    loaded = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        loaded.update(load_file(path))
    if not loaded:
        raise FileNotFoundError(f"no saved weights in {folder}")
    missing, unexpected = draft.load_state_dict(loaded, strict=False)
    if unexpected or any("layers" in name for name in missing):
        raise ValueError(f"the saved drafter does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--more-data", action="append", default=[], help="folders of more answers to learn from")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save", required=True, help="saved as lora-<name>.pt in --adapter-dir")
    parser.add_argument("--start-from", default="", help="go on from the saved add-ons lora-<name>.pt")
    parser.add_argument("--start-from-drafter", default="",
                        help="with --full: go on from the weights saved in drafter-<name>/ (a run that was stopped)")
    parser.add_argument("--recipe", default="dloop", choices=("dloop", "d2sd"))
    parser.add_argument("--rank", type=int, default=256, help="of the add-ons (0 with --full)")
    parser.add_argument("--full", action="store_true", help="train all of the drafter's weights (saved as a drafter folder, 1.2 GB)")
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=7.0, help="decay of the suffix view's loss")
    parser.add_argument("--gamma", type=float, default=7.0, help="decay of the prefix view's loss")
    parser.add_argument("--anchors", type=int, default=128, help="blocks learned from per answer")
    parser.add_argument("--group", type=int, default=16, help="blocks per update")
    parser.add_argument("--window", type=int, default=2048, help="tokens of text the drafter reads")
    parser.add_argument("--rate", type=float, default=2e-4)
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--max-answers", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--longest", type=int, default=0)
    parser.add_argument("--check-answers", type=int, default=12)
    parser.add_argument("--check-every", type=int, default=64)
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    status_path = os.path.join(args.results_dir, "status.json")
    everything = load_answers(args.data_dir)
    answers = [answer for answer in everything if split_of([answer[0], answer[1]]) == "train"]
    checks = [answer for answer in everything if split_of([answer[0], answer[1]]) == "validation"]
    held_out = {tuple(answer[2][: answer[3]].tolist()) for answer in everything if split_of([answer[0], answer[1]]) != "train"}
    answers = answers + more_answers(args.more_data, held_out, set())
    if args.max_answers:
        answers = answers[: args.max_answers]
    write_json(status_path, {"done": False, "answers": 0, "answers_to_learn_from": len(answers)}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    if args.full:
        # All weights trained, in 32-bit numbers (2.1 GB) with the passes in the drafter's 16-bit kind; the saved
        # drafter is written back in 16-bit (1.1 GB).
        saved_dtype = draft.fc.weight.dtype
        if args.start_from_drafter:
            load_saved_drafter(draft, os.path.join(args.adapter_dir, f"drafter-{args.start_from_drafter}"))
        draft.float()
        for parameter in draft.parameters():
            parameter.requires_grad_(True)
        parameters = list(draft.parameters())
        adapters = []
    else:
        adapters = attach_adapters(draft, rank=args.rank)
        if args.start_from:
            for adapter, (down, up) in zip(adapters, load_adapters(os.path.join(args.adapter_dir, f"lora-{args.start_from}.pt"), adapters)):
                with torch.no_grad():
                    adapter.down.copy_(down)
                    adapter.up.copy_(up)
        parameters = [parameter for adapter in adapters for parameter in (adapter.down, adapter.up)]
    optimizer = torch.optim.AdamW(parameters, lr=args.rate, weight_decay=0.0)
    teach = TeachLoop(draft, target, optimizer, recipe=args.recipe, beta=args.beta, tau=args.tau, gamma=args.gamma,
                      group=args.group, window=args.window, seed=args.seed,
                      mixed=args.full and device.type == "cuda" and saved_dtype != torch.float32)
    curve, started = [], time.time()

    def report(done, learned, measured=None):
        write_json(status_path, {"done": done, "answers": learned, "answers_to_learn_from": len(answers), "hours": round((time.time() - started) / 3600, 3),
                                 "draft": args.draft, "recipe": args.recipe, "rank": 0 if args.full else args.rank, "full": args.full,
                                 "beta": args.beta, "tau": args.tau, "gamma": args.gamma, "rate": args.rate, "saved_as": args.save, "started_from": args.start_from_drafter or args.start_from or None,
                                 "curve": curve, "validation": measured}, indent=1)

    def save(learned):
        notes = {"draft": args.draft, "answers": learned, "updates": teach.updates, "trained_by": "train_loop", "recipe": args.recipe,
                 "curve_end": curve[-1] if curve else None}
        if args.full:
            folder = os.path.join(args.adapter_dir, f"drafter-{args.save}")
            copy = {name: value.detach().to(saved_dtype).cpu() for name, value in draft.state_dict().items()}
            draft.save_pretrained(folder, state_dict=copy, safe_serialization=True)
            write_json(os.path.join(folder, "notes.json"), notes, indent=1)
        else:
            save_adapters(os.path.join(args.adapter_dir, f"lora-{args.save}.pt"), adapters, **notes)

    learned, saved_at, order = 0, time.time(), random.Random(args.seed)
    waiting = [entry for _ in range(max(args.passes, 1)) for entry in answers]
    order.shuffle(waiting)
    draft.train()
    for name, index, sequence, prompt_length, source in waiting:
        if time.time() - started > args.max_hours * 3600:
            break
        if len(sequence) - prompt_length < 3 * teach.width or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        with torch.no_grad():
            features = target_features(target, draft, ids).clone()
        teach.learn_from(ids, features, prompt_length, args.anchors)
        del features
        learned += 1
        if learned % 8 == 0:
            curve.append({"answers": learned, **teach.progress()})
        report(False, learned)
        if time.time() - saved_at > 1800:
            save(learned)
            saved_at = time.time()
    draft.eval()
    save(learned)
    teach.reset()
    measured = 0
    for name, index, sequence, prompt_length, source in checks:
        if measured >= args.check_answers:
            break
        if len(sequence) - prompt_length < 3 * teach.width or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        with torch.no_grad():
            features = target_features(target, draft, ids).clone()
        teach.learn_from(ids, features, prompt_length, 0, learn=False, every=args.check_every)
        del features
        measured += 1
    report(True, learned, {"answers": measured, **teach.progress()})


if __name__ == "__main__":
    main()
