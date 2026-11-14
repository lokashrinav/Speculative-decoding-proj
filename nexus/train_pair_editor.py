"""Train the learned judge of the left-to-right editor (dflash/pair_editor.py).

The drafter and the target stay frozen. For every training block the drafter writes
its first draft once. The editor then learns, for each draft position, to pick the
target's token among the first pass's top K candidates when it is told the true
token just before that position. At decoding time it goes left to right, each
position told the token it has just chosen, and the edited draft replaces the first
draft. Several editors train side by side on the same batches.

Only positions that can matter are trained: once the target's token is missing from
the top K, the target would stop accepting there, so later positions are skipped.

Variants are written as  name:key=value,key=value;name:...  with keys
  k=16          how many of the first pass's candidates each position may choose from
  states=1|0    whether the editor reads the drafter's state at the position
  bigram=0.25   also add the counted token-pair table with this weight (0 = off)
  smoothing=10  smoothing of that table
  decay=4       later positions count less: weight exp(-(position - 1) / decay)
  soft=1        also match how the target itself ranks the candidates (0 = off)
  rank=256      size of the editor         lr=0.0006
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

from dflash.draft_editor import draft_confidence
from dflash.model import _draft_value, _raw_input_embeddings, extract_context_feature
from dflash.pair_editor import BigramTable, PairEditor, save_pair_editor, walk_rows
from train_editor import EditorTrainer
from train_redraft import BLOCK, load_data

DEFAULTS = dict(k=16, states=1, bigram=0.0, smoothing=10.0, decay=4.0, soft=0.0, rank=256, lr=6e-4)
MARGINS = (0.0, 0.25, 0.5, 1.0, 2.0)
MAX_K = 16


def parse_variants(text):
    variants = []
    for part in filter(None, (piece.strip() for piece in text.split(";"))):
        name, _, rest = part.partition(":")
        options = dict(DEFAULTS)
        for pair in filter(None, rest.split(",")):
            key, value = pair.split("=")
            if key not in DEFAULTS:
                raise ValueError(f"unknown option {key!r} in variant {name!r}")
            options[key] = type(DEFAULTS[key])(value)
        if not 1 < options["k"] <= MAX_K:
            raise ValueError(f"k must be between 2 and {MAX_K}")
        variants.append((name, options))
    if len({name for name, _ in variants}) != len(variants):
        raise ValueError("variant names must be different")
    return variants


class Variant:
    def __init__(self, name, options, hidden_size, device):
        self.name, self.options = name, options
        self.editor = PairEditor(hidden_size, rank=options["rank"], read_states=bool(options["states"]),
                                 top_k=options["k"], bigram_weight=options["bigram"], smoothing=options["smoothing"]).to(device)
        self.parameters = list(self.editor.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=options["lr"], weight_decay=0.0)
        self.smoothed, self.skipped, self.failed = None, 0, None


@torch.no_grad()
def first_pass(frozen, ids, anchors):
    """What every editor shares for one batch of blocks: the true tokens, the first pass's
    states, its top candidates with their logits, and how the target ranks those candidates."""
    with torch.autocast(**frozen.autocast):
        output = frozen.target(ids[None], output_hidden_states=True)
    features = extract_context_feature(output.hidden_states, frozen.drafter.target_layer_ids)
    tokens, valid = frozen.blocks(ids, anchors)
    visible = tokens.clone()
    visible[:, 1:] = frozen.mask_id
    states, logits = frozen.drafter_pass(frozen.drafter, features, anchors, visible)
    logits = logits.float()
    top = logits.topk(min(MAX_K, logits.shape[-1]), dim=-1)
    # The target's logits for block position p come from the sequence position just before it.
    index = (anchors[:, None] + torch.arange(BLOCK - 1, device=ids.device)[None]).clamp(max=ids.shape[0] - 1)
    return SimpleNamespace(
        tokens=tokens, valid=valid, states=states, draft=logits.argmax(dim=-1), confidence=draft_confidence(logits),
        top_values=top.values, top_tokens=top.indices,
        target_scores=output.logits[0][index[..., None], top.indices].float(),
    )


def pair_loss(frozen, variant, batch, table):
    editor, options = variant.editor, variant.options
    k = editor.top_k
    candidates, truth = batch.top_tokens[..., :k], batch.tokens[:, 1:]
    with torch.autocast(**frozen.autocast):
        scores = editor.forced_scores(batch.states, batch.confidence, batch.top_values[..., :k], candidates,
                                      batch.tokens[:, :-1], frozen.embed, frozen.rows, frozen.multiplier, table)
    present = candidates == truth[..., None]
    # Train a position only while the target's token has been among the candidates so far.
    live = (present.any(dim=-1) & batch.valid[:, 1:]).long().cumprod(dim=1).float()
    slots = torch.arange(BLOCK - 1, device=truth.device)[None]
    weight = torch.exp(-slots / options["decay"]) * live
    if weight.sum() <= 0:
        return None
    log_probs = scores.float().log_softmax(dim=-1)
    label = present.float().argmax(dim=-1, keepdim=True)
    loss = -(log_probs.gather(-1, label)[..., 0] * weight).sum() / weight.sum()
    if options["soft"] > 0:
        wanted = batch.target_scores[..., :k].log_softmax(dim=-1)
        divergence = (wanted.exp() * (wanted - log_probs)).sum(dim=-1)
        loss = loss + options["soft"] * (divergence * weight).sum() / weight.sum()
    return loss


@torch.no_grad()
def validate(frozen, variants, items, anchors_per_sequence, table):
    """As in decoding: left to right, each position told the token just chosen; the edited
    draft replaces the first draft. Counted on answers the editors never trained on."""
    generator = torch.Generator().manual_seed(1234)
    device = frozen.device
    counts = {v.name: {m: dict(edited=0, better=0, worse=0, changed=0) for m in MARGINS} for v in variants}
    rounds = first_total = 0
    for v in variants:
        v.editor.eval()
    for ids, prompt_length in items:
        ids = ids.to(device)
        candidates = torch.arange(prompt_length, ids.shape[0] - BLOCK, device=device)
        if candidates.numel() == 0:
            continue
        pick = torch.randperm(candidates.shape[0], generator=generator)[:anchors_per_sequence]
        batch = first_pass(frozen, ids, candidates[pick.to(device)])
        truth = batch.tokens[:, 1:]

        def accepted(proposal):
            return (proposal == truth).long().cumprod(dim=1).sum(dim=1)

        first = accepted(batch.draft)
        rounds += first.shape[0]
        first_total += int((first + 1).sum())
        for v in variants:
            k = v.editor.top_k
            with torch.autocast(**frozen.autocast):
                start, pair = v.editor.scores(batch.states, batch.confidence, batch.top_values[..., :k],
                                              batch.top_tokens[..., :k], batch.tokens[:, 0],
                                              frozen.embed, frozen.rows, frozen.multiplier, table)
            for margin in MARGINS:
                proposal = walk_rows(start.float(), pair.float(), batch.top_tokens[..., :k], margin)
                edited = accepted(proposal)
                count = counts[v.name][margin]
                count["edited"] += int((edited + 1).sum())
                count["better"] += int((edited > first).sum())
                count["worse"] += int((edited < first).sum())
                count["changed"] += int((proposal != batch.draft).any(dim=1).sum())
    for v in variants:
        v.editor.train()
    rounds, base = max(rounds, 1), max(first_total, 1)
    result = {"rounds": rounds, "tokens_per_round_first_draft": first_total / rounds, "editors": {}}
    for v in variants:
        by_margin = {str(m): {
            "gain_edited_replaces_first": counts[v.name][m]["edited"] / base - 1,
            "edited_better": counts[v.name][m]["better"] / rounds,
            "edited_worse": counts[v.name][m]["worse"] / rounds,
            "drafts_changed": counts[v.name][m]["changed"] / rounds,
        } for m in MARGINS}
        best = max(MARGINS, key=lambda m: counts[v.name][m]["edited"])
        result["editors"][v.name] = {
            "gain_no_margin": by_margin["0.0"]["gain_edited_replaces_first"], "best_margin": best,
            "gain_best_margin": by_margin[str(best)]["gain_edited_replaces_first"], "by_margin": by_margin,
        }
    return result


def add_new_answers(items, training, extra_dirs, max_len, seed):
    """Add answers from other data folders to the training set. An answer whose prompt
    already appears in the first folder (training or held-out) is dropped, so nothing
    leaks into the held-out answers. Returns how many were added and dropped."""
    seen = {ids[:length].numpy().tobytes() for ids, length in items}
    added = dropped = 0
    for folder in filter(None, extra_dirs.split(",")):
        for ids, length in load_data(folder, max_len, seed):
            key = ids[:length].numpy().tobytes()
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            training.append((ids, length))
            added += 1
    return added, dropped


def learning_rate_factor(step, total_steps, warmup=200):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--extra-data-dirs", default="", help="comma-separated folders with more answers, for training only")
    parser.add_argument("--variants", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="pair-v1-")
    parser.add_argument("--table", default=None, help="saved BigramTable; counted from the training answers if missing")
    parser.add_argument("--status", required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--train-hours", type=float, default=100.0)
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
    items = load_data(args.data_dir, args.max_len, args.seed)
    validation, training = items[-args.val_sequences:], items[: -args.val_sequences]
    added, dropped = add_new_answers(items, training, args.extra_data_dirs, args.max_len, args.seed)
    frozen = EditorTrainer(SimpleNamespace(model=args.model, draft=args.draft, teacher=None, rank=8,
                                           read_tokens="all", no_states=False, first_slot=1), device)
    frozen.embed = lambda ids: _raw_input_embeddings(frozen.target, ids, 1.0)
    frozen.rows = lambda ids: frozen.head.weight[ids]
    frozen.multiplier = float(_draft_value(frozen.drafter.config, "output_multiplier", 1.0))
    vocabulary = frozen.head.weight.shape[0]
    if args.table and os.path.exists(args.table):
        table = BigramTable.load(args.table, device)
    else:
        table = BigramTable.count(training, vocabulary).to(device)
        if args.table:
            table.save(args.table)
    hidden_size = frozen.target.get_input_embeddings().weight.shape[1]
    variants = [Variant(name, options, hidden_size, device) for name, options in specs]
    os.makedirs(args.out_dir, exist_ok=True)
    total_steps = args.steps
    status = {
        "variants": {v.name: dict(v.options, editor_parameters=sum(p.numel() for p in v.parameters)) for v in variants},
        "training_sequences": len(training), "validation_sequences": len(validation),
        "new_answers_added": added, "new_answers_dropped_as_repeats": dropped,
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
        alive = [v for v in variants if not v.failed]
        metrics = validate(frozen, alive, validation, args.val_anchors, table)
        metrics["step"] = step
        status["validation"].append(metrics)
        ranked = []
        for v in alive:
            scores = metrics["editors"][v.name]
            path = os.path.join(args.out_dir, f"{args.prefix}{v.name}.pt")
            save_pair_editor(v.editor, path, margin=scores["best_margin"])
            ranked.append((scores["gain_best_margin"], v.name, path, scores["best_margin"]))
        if ranked and step > 0:
            gain, name, path, margin = max(ranked)
            status["best"] = {"name": name, "path": path, "margin": margin, "gain_edited_replaces_first": gain, "step": step}
        print(json.dumps({"step": step, "gains": {n: round(s["gain_best_margin"], 4) for n, s in metrics["editors"].items()}}),
              file=sys.stderr)

    started = time.time()
    check(0)
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
                pick = torch.randperm(candidates.shape[0], generator=generator)[: args.anchors]
                chosen = candidates[pick.to(device)]
                while True:
                    try:
                        batch = first_pass(frozen, ids, chosen)
                        break
                    except torch.OutOfMemoryError:
                        # A long answer with many blocks did not fit: use half the blocks for this step.
                        torch.cuda.empty_cache()
                        status["steps_with_fewer_blocks"] = status.get("steps_with_fewer_blocks", 0) + 1
                        if chosen.shape[0] <= 8:
                            raise
                        chosen = chosen[: chosen.shape[0] // 2]
                factor = learning_rate_factor(step - 1, total_steps)
                for v in variants:
                    if v.failed:
                        continue
                    try:
                        for group in v.optimizer.param_groups:
                            group["lr"] = v.options["lr"] * factor
                        loss = pair_loss(frozen, v, batch, table)
                        if loss is None or not torch.isfinite(loss):
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
