"""Train the small draft editor (dflash/draft_editor.py).

The drafter and the target stay frozen. For every training block the drafter
writes its first draft once; the editor then reads the first pass's saved states
and the draft's tokens and learns to output what the target really wrote. At
decoding time the edited draft replaces the first draft, and no second drafter
pass is involved.

Two ways to train it:
  target   learn directly from the target's own tokens
  teacher  the same, plus imitate a trained second-pass drafter (a checkpoint
           from train_redraft.py) on the positions that drafter rewrites
"""

import argparse
import json
import math
import random
import sys
import time

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from dflash.draft_editor import DraftEditor, save_editor
from dflash.model import DFlashDraftModel, _draft_value, _output_head, _raw_input_embeddings, extract_context_feature
from train_redraft import BLOCK, load_data


class EditorTrainer:
    def __init__(self, args, device):
        self.args, self.device = args, device
        on_gpu = device.type == "cuda"
        frozen = torch.bfloat16 if on_gpu else torch.float32
        self.autocast = dict(device_type=device.type, dtype=torch.bfloat16, enabled=on_gpu)
        self.target = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="sdpa", dtype=frozen).to(device).eval()
        self.drafter = DFlashDraftModel.from_pretrained(args.draft, attn_implementation="sdpa", dtype=frozen).to(device).eval()
        self.teacher = None
        if args.teacher:
            self.teacher = DFlashDraftModel.from_pretrained(args.teacher, attn_implementation="sdpa", dtype=frozen).to(device).eval()
        for model in (self.target, self.drafter, self.teacher):
            if model is not None:
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
        hidden_size = self.target.get_input_embeddings().weight.shape[1]
        self.editor = DraftEditor(
            hidden_size, rank=args.rank, read_tokens=args.read_tokens,
            read_states=not args.no_states, first_slot=args.first_slot,
        ).to(device)
        self.scale = float(_draft_value(self.drafter.config, "input_embedding_scale", 1.0))
        self.mask_id = self.drafter.mask_token_id
        self.head = _output_head(self.target)

    @torch.no_grad()
    def features(self, ids):
        with torch.autocast(**self.autocast):
            output = self.target(ids[None], output_hidden_states=True)
        return extract_context_feature(output.hidden_states, self.drafter.target_layer_ids)

    def blocks(self, ids, anchors):
        length = ids.shape[0]
        index = anchors[:, None] + torch.arange(BLOCK, device=self.device)[None]
        valid = index < length
        tokens = ids[index.clamp(max=length - 1)]
        return torch.where(valid, tokens, torch.full_like(tokens, self.mask_id)), valid

    @torch.no_grad()
    def drafter_pass(self, model, features, anchors, visible):
        """Final states for the 16 block positions and logits for positions 1..15."""
        count, length = anchors.shape[0], features.shape[1]
        context_visible = torch.arange(length, device=self.device)[None] < anchors[:, None]
        mask = torch.cat((context_visible, torch.ones((count, BLOCK), dtype=torch.bool, device=self.device)), dim=1)
        mask = mask[:, None, None, :].expand(count, 1, BLOCK, length + BLOCK)
        positions = torch.cat((
            torch.arange(length, device=self.device)[None].expand(count, length),
            anchors[:, None] + torch.arange(BLOCK, device=self.device)[None],
        ), dim=1)
        with torch.autocast(**self.autocast):
            noise = _raw_input_embeddings(self.target, visible, self.scale)
            states = model(
                target_hidden=features.expand(count, -1, -1), noise_embedding=noise,
                position_ids=positions, attention_mask=mask,
            )
            return states, model.compute_logits(states[:, 1:], self.head)

    def first_draft(self, features, anchors, tokens):
        visible = tokens.clone()
        visible[:, 1:] = self.mask_id
        states, logits = self.drafter_pass(self.drafter, features, anchors, visible)
        draft = torch.cat((tokens[:, :1], logits.argmax(dim=-1)), dim=1)
        return states, draft

    def edit(self, states, draft):
        """Editor logits for the block positions it may change (first_slot .. 15)."""
        with torch.autocast(**self.autocast):
            embeddings = _raw_input_embeddings(self.target, draft, 1.0)
            corrected = self.editor(states, embeddings)
            return self.drafter.compute_logits(corrected, self.head)

    def edited_draft(self, states, draft):
        """Draft tokens for positions 1..15 after editing."""
        first = self.editor.first_slot
        changed = self.edit(states, draft).argmax(dim=-1)
        return torch.cat((draft[:, 1:first], changed), dim=1)

    def loss(self, ids, prompt_length, generator):
        features = self.features(ids)
        candidates = torch.arange(prompt_length, ids.shape[0] - 2, device=self.device)
        pick = torch.randperm(candidates.shape[0], generator=generator)[: self.args.anchors]
        anchors = candidates[pick.to(self.device)]
        tokens, valid = self.blocks(ids, anchors)
        states, draft = self.first_draft(features, anchors, tokens)
        first = self.editor.first_slot
        first_token_right = draft[:, 1] == tokens[:, 1]
        slots = torch.arange(first, BLOCK, device=self.device)[None]
        weight = torch.exp(-(slots - first) / 7.0) * valid[:, first:]
        if first == 2:
            # The first draft token is kept, so only blocks where it is right can be improved.
            weight = weight * first_token_right[:, None]
        total = weight.sum()
        if total <= 0:
            return None
        logits = self.edit(states, draft).float()
        log_probs = logits.log_softmax(dim=-1)
        cross_entropy = -log_probs.gather(-1, tokens[:, first:, None])[..., 0]
        loss = (cross_entropy * weight).sum() / total
        if self.teacher is not None:
            # The teacher rewrites positions 2..15 given a correct first token.
            later = torch.arange(2, BLOCK, device=self.device)[None]
            teacher_weight = torch.exp(-(later - 2) / 7.0) * valid[:, 2:] * first_token_right[:, None]
            if teacher_weight.sum() > 0:
                _, teacher_logits = self.drafter_pass(self.teacher, features, anchors, draft)
                teacher_log_probs = teacher_logits[:, 1:].float().log_softmax(dim=-1)
                divergence = (teacher_log_probs.exp() * (teacher_log_probs - log_probs[:, 2 - first :])).sum(dim=-1)
                loss = loss + self.args.teacher_weight * (divergence * teacher_weight).sum() / teacher_weight.sum()
        return loss

    @torch.no_grad()
    def validate(self, items):
        """As in decoding: the edited draft replaces the first draft. How far does each get accepted?"""
        generator = torch.Generator().manual_seed(1234)
        totals = dict(first=0, edited=0, best=0, teacher=0, teacher_best=0)
        rounds = better = worse = changed = 0
        self.editor.eval()
        for ids, prompt_length in items:
            ids = ids.to(self.device)
            candidates = torch.arange(prompt_length, ids.shape[0] - BLOCK, device=self.device)
            if candidates.numel() == 0:
                continue
            features = self.features(ids)
            anchors = candidates[torch.randperm(candidates.shape[0], generator=generator)[:8].to(self.device)]
            tokens, _ = self.blocks(ids, anchors)
            states, draft = self.first_draft(features, anchors, tokens)

            def accepted(proposal):
                return (proposal == tokens[:, 1:]).long().cumprod(dim=1).sum(dim=1)

            first = accepted(draft[:, 1:])
            proposal = self.edited_draft(states, draft)
            edited = accepted(proposal)
            rounds += anchors.shape[0]
            totals["first"] += int((first + 1).sum())
            totals["edited"] += int((edited + 1).sum())
            totals["best"] += int((torch.maximum(first, edited) + 1).sum())
            better += int((edited > first).sum())
            worse += int((edited < first).sum())
            changed += int((proposal != draft[:, 1:]).any(dim=1).sum())
            if self.teacher is not None:
                _, teacher_logits = self.drafter_pass(self.teacher, features, anchors, draft)
                taught = accepted(torch.cat((draft[:, 1:2], teacher_logits[:, 1:].argmax(dim=-1)), dim=1))
                totals["teacher"] += int((taught + 1).sum())
                totals["teacher_best"] += int((torch.maximum(first, taught) + 1).sum())
        self.editor.train()
        base = max(totals["first"], 1)
        metrics = {
            "rounds": rounds,
            "tokens_per_round_first_draft": totals["first"] / max(rounds, 1),
            "tokens_per_round_edited_draft": totals["edited"] / max(rounds, 1),
            "gain_edited_replaces_first": totals["edited"] / base - 1,
            "edited_better": better / max(rounds, 1),
            "edited_worse": worse / max(rounds, 1),
            "drafts_changed": changed / max(rounds, 1),
            "gain_if_first_kept_as_backup": totals["best"] / base - 1,
        }
        if self.teacher is not None:
            metrics["gain_teacher_replaces_first"] = totals["teacher"] / base - 1
            metrics["gain_teacher_with_backup"] = totals["teacher_best"] / base - 1
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--teacher", default=None, help="trained second-pass drafter checkpoint to imitate")
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--read-tokens", choices=["all", "first", "none"], default="all")
    parser.add_argument("--no-states", action="store_true", help="the editor reads only the draft's tokens")
    parser.add_argument("--first-slot", type=int, choices=[1, 2], default=1,
                        help="1: the editor may change every draft token; 2: the first draft token is kept")
    parser.add_argument("--out", required=True, help="file for the trained editor")
    parser.add_argument("--status", required=True)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--anchors", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-sequences", type=int, default=300)
    parser.add_argument("--validate-every", type=int, default=4000)
    parser.add_argument("--max-hours", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    items = load_data(args.data_dir, args.max_len, args.seed)
    validation, training = items[-args.val_sequences:], items[: -args.val_sequences]
    trainer = EditorTrainer(args, device)
    parameters = list(trainer.editor.parameters())
    total_steps = int(len(training) * args.epochs)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    warmup = min(200, max(1, total_steps // 20))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup
        else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))),
    )
    generator = torch.Generator().manual_seed(args.seed)
    status = {
        "read_tokens": args.read_tokens, "read_states": not args.no_states, "first_slot": args.first_slot,
        "teacher": bool(args.teacher),
        "training_sequences": len(training), "validation_sequences": len(validation),
        "total_steps": total_steps, "editor_parameters": sum(p.numel() for p in parameters),
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
    trainer.editor.train()
    while step < total_steps:
        random.Random(args.seed + step).shuffle(order)
        for index in order:
            if step >= total_steps:
                break
            ids, prompt_length = training[index]
            loss = trainer.loss(ids.to(device), prompt_length, generator)
            step += 1
            if loss is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                value = float(loss)
                smoothed = value if smoothed is None else 0.98 * smoothed + 0.02 * value
            schedule.step()
            if step % 100 == 0:
                report(step, smoothed, started)
            if step % args.validate_every == 0 and step < total_steps:
                check(step)
                save_editor(trainer.editor, args.out)
                report(step, smoothed, started)
            if time.time() - started > args.max_hours * 3600:
                print("time budget reached, stopping early", file=sys.stderr)
                total_steps = step
    check(step)
    save_editor(trainer.editor, args.out)
    report(step, smoothed, started)


if __name__ == "__main__":
    main()
