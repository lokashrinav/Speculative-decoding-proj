"""Train several draft editors side by side (dflash/draft_editor.py).

Every editor sees exactly the same training blocks: the target and the drafter
run once per step, and each editor then learns from that same first draft. The
editors differ only in what they are rewarded for and what they read, so their
results can be compared directly. At decoding time the edited draft replaces
the first draft, and no second drafter pass is involved.

What an editor can be rewarded for (loss):
  ce    guess every draft token, later tokens counting less (the first version)
  tv    the same, but a token counts by how likely it is to be accepted, so
        tokens that are hopelessly wrong stop dominating
  pce   guess every token up to the first one the edited draft is likely to get
        wrong; tokens after that cannot be accepted, so they are not chased
  eal   expected accepted length: how many tokens in a row the target is
        expected to accept from the edited draft, which is what we want

Variants are written as  name:key=value,key=value;name:...  with keys
  loss=ce|tv|pce|eal
  mix=0.5       while training, draft tokens the first pass was this sure of are shown as the true token
  tokens=all|none   states=1|0     what the editor reads
  plain=1       the first version's layout: inputs not rescaled, no confidence input
  width=256     layers=2           size of the editor
  kl=0.5        also imitate a trained second drafter (needs --teacher)
  lr=0.001
"""

import argparse
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import torch

from dflash.draft_editor import DraftEditor, draft_confidence, keep_unless_sure, load_editor, save_editor
from dflash.model import _raw_input_embeddings
from train_editor import EditorTrainer
from train_redraft import BLOCK, load_data

DEFAULTS = dict(loss="eal", mix=0.0, tokens="all", states=1, plain=0, width=256, layers=2, kl=0.0, lr=1e-3)
LOSSES = ("ce", "tv", "pce", "eal")
MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def parse_variants(text):
    variants = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, rest = part.partition(":")
        options = dict(DEFAULTS)
        for pair in filter(None, rest.split(",")):
            key, value = pair.split("=")
            if key not in DEFAULTS:
                raise ValueError(f"unknown option {key!r} in variant {name!r}")
            options[key] = type(DEFAULTS[key])(value)
        if options["loss"] not in LOSSES:
            raise ValueError(f"unknown loss {options['loss']!r} in variant {name!r}")
        variants.append((name, options))
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("variant names must be different")
    return variants


class Variant:
    def __init__(self, name, options, hidden_size, device):
        self.name, self.options = name, options
        plain = bool(options["plain"])
        width = options["width"]
        self.editor = DraftEditor(
            hidden_size, rank=width, heads=max(1, width // 64), layers=options["layers"],
            read_tokens=options["tokens"], read_states=bool(options["states"]), first_slot=1,
            normalize=not plain, confidence=not plain,
        ).to(device)
        self.parameters = list(self.editor.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=options["lr"], weight_decay=0.0)
        self.smoothed = None
        self.skipped = 0
        self.failed = None


@torch.no_grad()
def first_pass(frozen, features, anchors, ids, need_teacher=False):
    """Everything the editors share for one batch of blocks: the true tokens, the
    drafter's first draft, the states it left behind and how sure it was."""
    tokens, valid = frozen.blocks(ids, anchors)
    visible = tokens.clone()
    visible[:, 1:] = frozen.mask_id
    states, logits = frozen.drafter_pass(frozen.drafter, features, anchors, visible)
    draft = torch.cat((tokens[:, :1], logits.argmax(dim=-1)), dim=1)
    batch = SimpleNamespace(tokens=tokens, valid=valid, states=states, draft=draft,
                            confidence=draft_confidence(logits), teacher=None)
    if need_teacher:
        _, teacher_logits = frozen.drafter_pass(frozen.teacher, features, anchors, draft)
        batch.teacher = teacher_logits[:, 1:].float().log_softmax(dim=-1)  # block positions 2..15
    return batch


def edited_logits(frozen, editor, batch, shown=None):
    """The editor's logits for draft positions 1..15. shown: the tokens it is shown, the draft by default."""
    with torch.autocast(**frozen.autocast):
        embeddings = _raw_input_embeddings(frozen.target, batch.draft if shown is None else shown, 1.0)
        corrected = editor(batch.states, embeddings, batch.confidence if editor.confidence else None)
        return frozen.drafter.compute_logits(corrected, frozen.head)


def reward_loss(kind, log_right, live):
    """What the editor is rewarded for (see the top of this file).
    log_right: (blocks, 15) log chance the editor gives the true token at draft positions 1..15.
    live: 1 where the text has a token at that position, 0 after it has ended."""
    log_right = log_right * live
    slots = torch.arange(log_right.shape[1], device=log_right.device)[None]
    decay = torch.exp(-slots / 7.0) * live
    if kind == "ce":
        return -(log_right * decay).sum() / decay.sum().clamp_min(1e-6)
    if kind == "tv":
        return (((1.0 - log_right.exp()) - 0.1 * log_right) * decay).sum() / decay.sum().clamp_min(1e-6)
    if kind == "pce":
        earlier_right = torch.exp(log_right.cumsum(dim=1) - log_right).detach()
        return -(log_right * earlier_right * live).sum() / log_right.shape[0]
    if kind == "eal":
        accepted = (torch.exp(log_right.cumsum(dim=1)) * live).sum(dim=1)
        return -accepted.mean()
    raise ValueError(f"unknown loss {kind!r}")


def variant_loss(frozen, variant, batch):
    options = variant.options
    tokens, valid = batch.tokens, batch.valid
    shown = None
    if options["mix"] > 0:
        sure = (batch.confidence[..., 0] > options["mix"]) & valid
        shown = torch.where(sure, tokens, batch.draft)
    log_probs = edited_logits(frozen, variant.editor, batch, shown).float().log_softmax(dim=-1)
    log_right = log_probs.gather(-1, tokens[:, 1:, None])[..., 0]
    loss = reward_loss(options["loss"], log_right, valid[:, 1:].float())
    if options["kl"] > 0 and batch.teacher is not None:
        # The teacher rewrites positions 2..15 given a correct first token.
        first_token_right = batch.draft[:, 1] == tokens[:, 1]
        later = torch.arange(2, BLOCK, device=tokens.device)[None]
        weight = torch.exp(-(later - 2) / 7.0) * valid[:, 2:] * first_token_right[:, None]
        if weight.sum() > 0:
            divergence = (batch.teacher.exp() * (batch.teacher - log_probs[:, 1:])).sum(dim=-1)
            loss = loss + options["kl"] * (divergence * weight).sum() / weight.sum()
    return loss


@torch.no_grad()
def validate(frozen, editors, items, anchors_per_sequence):
    """As in decoding: the edited draft replaces the first draft. How many tokens does
    the target accept from each, on answers the editors never trained on?
    Every margin in MARGINS is tried (see keep_unless_sure)."""
    generator = torch.Generator().manual_seed(1234)
    device = frozen.device
    counts = {name: {m: dict(edited=0, better=0, worse=0, changed=0, gained=0, lost=0) for m in MARGINS}
              for name in editors}
    backup = {name: 0 for name in editors}
    rounds = first_total = 0
    for editor in editors.values():
        editor.eval()
    for ids, prompt_length in items:
        ids = ids.to(device)
        candidates = torch.arange(prompt_length, ids.shape[0] - BLOCK, device=device)
        if candidates.numel() == 0:
            continue
        features = frozen.features(ids)
        pick = torch.randperm(candidates.shape[0], generator=generator)[:anchors_per_sequence]
        batch = first_pass(frozen, features, candidates[pick.to(device)], ids)
        truth, draft = batch.tokens[:, 1:], batch.draft[:, 1:]

        def accepted(proposal):
            return (proposal == truth).long().cumprod(dim=1).sum(dim=1)

        first = accepted(draft)
        rounds += first.shape[0]
        first_total += int((first + 1).sum())
        for name, editor in editors.items():
            logits = edited_logits(frozen, editor, batch).float()
            kept = editor.first_slot - 1  # leading draft tokens this editor never changes
            for margin in MARGINS:
                proposal = torch.cat((draft[:, :kept], keep_unless_sure(logits, draft[:, kept:], margin)), dim=1)
                edited = accepted(proposal)
                count = counts[name][margin]
                count["edited"] += int((edited + 1).sum())
                count["better"] += int((edited > first).sum())
                count["worse"] += int((edited < first).sum())
                count["changed"] += int((proposal != draft).any(dim=1).sum())
                count["gained"] += int((edited - first).clamp_min(0).sum())
                count["lost"] += int((first - edited).clamp_min(0).sum())
                if margin == 0:
                    backup[name] += int((torch.maximum(first, edited) + 1).sum())
    for editor in editors.values():
        editor.train()
    rounds = max(rounds, 1)
    base = max(first_total, 1)
    result = {"rounds": rounds, "tokens_per_round_first_draft": first_total / rounds, "editors": {}}
    for name in editors:
        by_margin = {}
        for margin in MARGINS:
            count = counts[name][margin]
            by_margin[str(margin)] = {
                "gain_edited_replaces_first": count["edited"] / base - 1,
                "edited_better": count["better"] / rounds,
                "edited_worse": count["worse"] / rounds,
                "drafts_changed": count["changed"] / rounds,
                "tokens_gained_per_round": count["gained"] / rounds,
                "tokens_lost_per_round": count["lost"] / rounds,
            }
        best = max(MARGINS, key=lambda m: counts[name][m]["edited"])
        result["editors"][name] = {
            "gain_no_margin": by_margin["0.0"]["gain_edited_replaces_first"],
            "best_margin": best,
            "gain_best_margin": by_margin[str(best)]["gain_edited_replaces_first"],
            "gain_if_first_kept_as_backup": backup[name] / base - 1,
            "by_margin": by_margin,
        }
    return result


def learning_rate_factor(step, total_steps, warmup=200):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--teacher", default=None, help="trained second-pass drafter checkpoint, for variants with kl")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out-dir", required=True, help="folder for the trained editors")
    parser.add_argument("--prefix", default="editor-v2-")
    parser.add_argument("--status", required=True)
    parser.add_argument("--also-evaluate", default="", help="comma-separated editor files to score once at the start")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--train-hours", type=float, default=100.0, help="shorten the run to fit this much training time")
    parser.add_argument("--anchors", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-sequences", type=int, default=300)
    parser.add_argument("--val-anchors", type=int, default=24)
    parser.add_argument("--validate-every", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    specs = parse_variants(args.variants)
    need_teacher = any(options["kl"] > 0 for _, options in specs)
    if need_teacher and not args.teacher:
        raise SystemExit("a variant uses kl, so --teacher is needed")
    items = load_data(args.data_dir, args.max_len, args.seed)
    validation, training = items[-args.val_sequences:], items[: -args.val_sequences]
    frozen = EditorTrainer(SimpleNamespace(
        model=args.model, draft=args.draft, teacher=args.teacher if need_teacher else None,
        rank=8, read_tokens="all", no_states=False, first_slot=1,
    ), device)
    hidden_size = frozen.target.get_input_embeddings().weight.shape[1]
    variants = [Variant(name, options, hidden_size, device) for name, options in specs]
    os.makedirs(args.out_dir, exist_ok=True)
    total_steps = args.steps
    status = {
        "variants": {v.name: dict(v.options, editor_parameters=sum(p.numel() for p in v.parameters)) for v in variants},
        "training_sequences": len(training), "validation_sequences": len(validation),
        "total_steps": total_steps, "validation": [],
    }

    def report(step, started):
        status.update(step=step, total_steps=total_steps, hours=round((time.time() - started) / 3600, 3))
        status["loss"] = {v.name: v.smoothed for v in variants}
        status["problems"] = {v.name: v.failed or f"{v.skipped} steps skipped" for v in variants if v.failed or v.skipped}
        with open(args.status + ".tmp", "w") as handle:
            json.dump(status, handle, indent=1)
        os.replace(args.status + ".tmp", args.status)

    def check(step):
        alive = {v.name: v.editor for v in variants if not v.failed}
        metrics = validate(frozen, alive, validation, args.val_anchors)
        metrics["step"] = step
        status["validation"].append(metrics)
        ranked = []
        for v in variants:
            if v.failed:
                continue
            scores = metrics["editors"][v.name]
            path = os.path.join(args.out_dir, f"{args.prefix}{v.name}.pt")
            save_editor(v.editor, path, margin=scores["best_margin"])
            ranked.append((scores["gain_best_margin"], v.name, path, scores["best_margin"]))
        if ranked:
            gain, name, path, margin = max(ranked)
            status["best"] = {"name": name, "path": path, "margin": margin, "gain_edited_replaces_first": gain, "step": step}
        print(json.dumps({"step": step, "first": metrics["tokens_per_round_first_draft"],
                          "gains": {name: round(scores["gain_best_margin"], 4) for name, scores in metrics["editors"].items()}}),
              file=sys.stderr)

    started = time.time()
    report(0, started)
    earlier = {}
    for path in filter(None, args.also_evaluate.split(",")):
        if os.path.exists(path):
            try:
                earlier[os.path.basename(path)] = load_editor(path, device, torch.float32)
            except Exception as error:  # a file another job is writing right now; skip it
                status.setdefault("not_loaded", {})[os.path.basename(path)] = str(error)[:200]
    if earlier:
        status["earlier_editors"] = validate(frozen, earlier, validation, args.val_anchors)
        for editor in earlier.values():
            editor.to("cpu")
        report(0, started)

    generator = torch.Generator().manual_seed(args.seed)
    step = 0
    order = list(range(len(training)))
    training_started = time.time()
    while step < total_steps:
        random.Random(args.seed + step).shuffle(order)
        for index in order:
            if step >= total_steps:
                break
            ids, prompt_length = training[index]
            ids = ids.to(device)
            candidates = torch.arange(prompt_length, ids.shape[0] - 2, device=device)
            step += 1
            if candidates.numel() > 0:
                features = frozen.features(ids)
                pick = torch.randperm(candidates.shape[0], generator=generator)[: args.anchors]
                batch = first_pass(frozen, features, candidates[pick.to(device)], ids, need_teacher)
                factor = learning_rate_factor(step - 1, total_steps)
                for v in variants:
                    if v.failed:
                        continue
                    try:
                        for group in v.optimizer.param_groups:
                            group["lr"] = v.options["lr"] * factor
                        loss = variant_loss(frozen, v, batch)
                        if not torch.isfinite(loss):
                            v.skipped += 1
                            continue
                        v.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(v.parameters, 1.0)
                        v.optimizer.step()
                        value = float(loss.detach())
                        v.smoothed = value if v.smoothed is None else 0.98 * v.smoothed + 0.02 * value
                    except RuntimeError as error:
                        v.failed = str(error)[:300]
                        v.optimizer.zero_grad(set_to_none=True)
            if step == 100:
                # Fit the run into the time allowed, before the learning rate starts to fall.
                seconds = (time.time() - training_started) / 100
                status["seconds_per_step"] = round(seconds, 4)
                total_steps = min(args.steps, max(1000, int(args.train_hours * 3600 / seconds)))
            if step % 100 == 0:
                report(step, started)
            if step % args.validate_every == 0 and step < total_steps:
                check(step)
                report(step, started)
    check(step)
    report(step, started)


if __name__ == "__main__":
    main()
