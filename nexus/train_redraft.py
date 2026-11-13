"""Train the DFlash drafter to take a second look at its own draft.

The first pass is left exactly as it is. A small low-rank add-on (LoRA) is
trained for the second pass only, so there is no second model: the same
drafter plus a few million extra numbers.

Three variants, which differ only in what the second pass is shown:
  prefix   the first tokens of the draft (taken as correct), the rest masked
  edit     the first tokens of the draft, then the drafter's OWN first-pass
           guesses for the rest, which it has to correct
  control  nothing from the draft at all (tells us how much of any gain comes
           from merely being a second, slightly different drafter)

Labels are always the target model's own tokens. Validation mimics decoding:
keep the first draft, add the second draft, count how far the target would
accept each, and report the best of the two.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from dflash.model import DFlashDraftModel, _draft_value, _output_head, _raw_input_embeddings, extract_context_feature

BLOCK = 16


class LoRALinear(nn.Module):
    active = True  # switched off to get the untouched first pass

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(rank, base.in_features))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        out = self.base(x)
        if LoRALinear.active:
            out = out + (x.to(self.down.dtype) @ self.down.t() @ self.up.t()).to(out.dtype) * self.scale
        return out

    def delta(self):
        return (self.up @ self.down) * self.scale


def add_lora(model, rank, alpha):
    wrapped = {}
    for index, layer in enumerate(model.layers):
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                module = LoRALinear(getattr(parent, name), rank, alpha)
                setattr(parent, name, module)
                prefix = "self_attn" if parent is layer.self_attn else "mlp"
                wrapped[f"layers.{index}.{prefix}.{name}.weight"] = module
    return wrapped


def load_data(data_dir, max_len, seed):
    sequences, prompt_lengths = [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "shard*.pt"))):
        shard = torch.load(path)
        sequences.extend(shard["sequences"])
        prompt_lengths.extend(shard["prompt_lengths"])
    items = [
        (seq[:max_len].long(), plen) for seq, plen in zip(sequences, prompt_lengths)
        if min(len(seq), max_len) - plen >= 4
    ]
    random.Random(seed).shuffle(items)
    return items


def sample_prefix_lengths(count, variant, generator, device):
    if variant == "control":
        return torch.zeros(count, dtype=torch.long, device=device)
    # Short prefixes most often: P(l) proportional to 0.7^(l-1), l = 1..8.
    weights = torch.tensor([0.7 ** k for k in range(8)])
    return (torch.multinomial(weights, count, replacement=True, generator=generator) + 1).to(device)


class Trainer:
    def __init__(self, args, device):
        self.args, self.device = args, device
        on_gpu = device.type == "cuda"
        self.autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=on_gpu)
        self.target = AutoModelForCausalLM.from_pretrained(
            args.model, attn_implementation="sdpa", dtype=torch.bfloat16 if on_gpu else torch.float32,
        ).to(device).eval()
        self.model = DFlashDraftModel.from_pretrained(
            args.draft, attn_implementation="sdpa", dtype=torch.float32,
        ).to(device).eval()
        for parameter in list(self.target.parameters()) + list(self.model.parameters()):
            parameter.requires_grad_(False)
        self.lora = add_lora(self.model, args.rank, args.alpha)
        self.model.to(device)
        self.parameters = [p for m in self.lora.values() for p in (m.down, m.up)]
        self.scale = float(_draft_value(self.model.config, "input_embedding_scale", 1.0))
        self.mask_id = self.model.mask_token_id
        self.head = _output_head(self.target)

    @torch.no_grad()
    def features(self, ids):
        with torch.autocast(**self.autocast):
            output = self.target(ids[None], output_hidden_states=True)
        return extract_context_feature(output.hidden_states, self.model.target_layer_ids)

    def blocks(self, ids, anchors):
        """Token ids of each anchor's block and which slots have a real label."""
        length = ids.shape[0]
        index = anchors[:, None] + torch.arange(BLOCK, device=self.device)[None]
        valid = index < length
        tokens = ids[index.clamp(max=length - 1)]
        tokens = torch.where(valid, tokens, torch.full_like(tokens, self.mask_id))
        return tokens, valid

    def draft(self, features, anchors, visible, lora):
        """One drafter pass for a batch of anchors of the same sequence. Returns logits for slots 1..15."""
        count, length = anchors.shape[0], features.shape[1]
        context_visible = torch.arange(length, device=self.device)[None] < anchors[:, None]
        mask = torch.cat((context_visible, torch.ones((count, BLOCK), dtype=torch.bool, device=self.device)), dim=1)
        mask = mask[:, None, None, :].expand(count, 1, BLOCK, length + BLOCK)
        positions = torch.cat((
            torch.arange(length, device=self.device)[None].expand(count, length),
            anchors[:, None] + torch.arange(BLOCK, device=self.device)[None],
        ), dim=1)
        LoRALinear.active = lora
        try:
            with torch.autocast(**self.autocast):
                noise = _raw_input_embeddings(self.target, visible, self.scale)
                hidden = self.model(
                    target_hidden=features.expand(count, -1, -1), noise_embedding=noise,
                    position_ids=positions, attention_mask=mask,
                )
                return self.model.compute_logits(hidden[:, 1:], self.head)
        finally:
            LoRALinear.active = True

    def second_pass_input(self, tokens, first_draft, prefix):
        """What the second pass sees: the anchor, `prefix` kept tokens, then masks or the first draft."""
        slots = torch.arange(BLOCK, device=self.device)[None]
        kept = slots <= prefix[:, None]
        if self.args.variant == "edit":
            rest = torch.cat((tokens[:, :1], first_draft), dim=1)
        else:
            rest = torch.full_like(tokens, self.mask_id)
        return torch.where(kept, tokens, rest), kept

    def loss(self, ids, prompt_length, generator):
        features = self.features(ids)
        candidates = torch.arange(prompt_length, ids.shape[0] - 1, device=self.device)
        pick = torch.randperm(candidates.shape[0], generator=generator)[: self.args.anchors]
        anchors = candidates[pick.to(self.device)]
        tokens, valid = self.blocks(ids, anchors)
        prefix = sample_prefix_lengths(anchors.shape[0], self.args.variant, generator, self.device)
        first_draft = None
        if self.args.variant == "edit":
            with torch.no_grad():
                first_visible = tokens.clone()
                first_visible[:, 1:] = self.mask_id
                first_draft = self.draft(features, anchors, first_visible, lora=False).argmax(dim=-1)
        visible, kept = self.second_pass_input(tokens, first_draft, prefix)
        logits = self.draft(features, anchors, visible, lora=True).float()
        slots = torch.arange(1, BLOCK, device=self.device)[None]
        learn = valid[:, 1:] & ~kept[:, 1:]
        weight = torch.exp(-(slots - prefix[:, None] - 1).clamp(min=0) / 7.0) * learn
        losses = nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none")
        return (losses * weight.flatten()).sum() / weight.sum().clamp(min=1e-6)

    @torch.no_grad()
    def validate(self, items):
        """Best-of-two on held-out sequences, as in decoding: keep the first draft, add the second."""
        generator = torch.Generator().manual_seed(1234)
        first_total = best_total = untrained_total = rounds = wins = 0
        keep = 0 if self.args.variant == "control" else 1
        for ids, prompt_length in items:
            ids = ids.to(self.device)
            features = self.features(ids)
            candidates = torch.arange(prompt_length, ids.shape[0] - BLOCK, device=self.device)
            if candidates.numel() == 0:
                continue
            anchors = candidates[torch.randperm(candidates.shape[0], generator=generator)[:8].to(self.device)]
            tokens, _ = self.blocks(ids, anchors)
            first_visible = tokens.clone()
            first_visible[:, 1:] = self.mask_id
            first = self.draft(features, anchors, first_visible, lora=False).argmax(dim=-1)
            draft_block = torch.cat((tokens[:, :1], first), dim=1)
            prefix = torch.full((anchors.shape[0],), keep, dtype=torch.long, device=self.device)
            visible, kept = self.second_pass_input(draft_block, first, prefix)

            def accepted(proposal):
                return (proposal == tokens[:, 1:]).long().cumprod(dim=1).sum(dim=1)

            first_accepted = accepted(first)
            results = {}
            for name, lora in (("trained", True), ("untrained", False)):
                if name == "untrained" and keep == 0:
                    results[name] = first_accepted
                    continue
                second = self.draft(features, anchors, visible, lora=lora).argmax(dim=-1)
                second = torch.where(kept[:, 1:], first, second)
                results[name] = torch.maximum(first_accepted, accepted(second))
            rounds += anchors.shape[0]
            first_total += int((first_accepted + 1).sum())
            best_total += int((results["trained"] + 1).sum())
            untrained_total += int((results["untrained"] + 1).sum())
            wins += int((results["trained"] > first_accepted).sum())
        return {
            "rounds": rounds,
            "tokens_per_round_first_draft": first_total / max(rounds, 1),
            "tokens_per_round_with_trained_second_look": best_total / max(rounds, 1),
            "tokens_per_round_with_untrained_second_look": untrained_total / max(rounds, 1),
            "gain_trained": best_total / max(first_total, 1) - 1,
            "gain_untrained": untrained_total / max(first_total, 1) - 1,
            "second_draft_wins": wins / max(rounds, 1),
        }

    def save(self, path):
        """Write a normal drafter checkpoint with the add-on folded into the weights."""
        merged = DFlashDraftModel.from_pretrained(self.args.draft, attn_implementation="sdpa", dtype=torch.float32)
        state = merged.state_dict()
        for name, module in self.lora.items():
            state[name] = state[name] + module.delta().detach().float().cpu()
        merged.load_state_dict(state)
        merged.to(torch.bfloat16).save_pretrained(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--variant", choices=["prefix", "edit", "control"], required=True)
    parser.add_argument("--out", required=True, help="directory for the merged drafter checkpoint")
    parser.add_argument("--status", required=True, help="small json file with progress and validation numbers")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--anchors", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-sequences", type=int, default=300)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument("--validate-every", type=int, default=4000)
    parser.add_argument("--max-hours", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    items = load_data(args.data_dir, args.max_len, args.seed)
    validation, training = items[-args.val_sequences:], items[: -args.val_sequences]
    if args.max_train_sequences:
        training = training[: args.max_train_sequences]
    trainer = Trainer(args, device)
    total_steps = int(len(training) * args.epochs)
    optimizer = torch.optim.AdamW(trainer.parameters, lr=args.lr, weight_decay=0.0)
    warmup = min(200, max(1, total_steps // 20))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup
        else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
    )
    generator = torch.Generator().manual_seed(args.seed)
    status = {
        "variant": args.variant, "training_sequences": len(training), "validation_sequences": len(validation),
        "total_steps": total_steps, "trainable_parameters": sum(p.numel() for p in trainer.parameters),
        "validation": [],
    }

    def report(step, smoothed, started):
        status.update(step=step, loss=smoothed, hours=round((time.time() - started) / 3600, 3))
        with open(args.status, "w") as handle:
            json.dump(status, handle, indent=1)

    def check(step):
        metrics = trainer.validate(validation)
        metrics["step"] = step
        status["validation"].append(metrics)
        print(json.dumps(metrics), file=sys.stderr)

    started = time.time()
    smoothed = None
    check(0)
    report(0, None, started)
    step = 0
    order = list(range(len(training)))
    while step < total_steps:
        random.Random(args.seed + step).shuffle(order)
        for index in order:
            if step >= total_steps:
                break
            ids, prompt_length = training[index]
            loss = trainer.loss(ids.to(device), prompt_length, generator)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.parameters, 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            value = float(loss)
            smoothed = value if smoothed is None else 0.98 * smoothed + 0.02 * value
            if step % 100 == 0:
                report(step, smoothed, started)
            if step % args.validate_every == 0 and step < total_steps:
                check(step)
                trainer.save(args.out)
                report(step, smoothed, started)
            if time.time() - started > args.max_hours * 3600:
                print("time budget reached, stopping early", file=sys.stderr)
                total_steps = step
    check(step)
    trainer.save(args.out)
    report(step, smoothed, started)


if __name__ == "__main__":
    main()
