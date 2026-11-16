"""What one round of decoding costs, piece by piece, and what learning during the answer adds.

A round is one draft by the drafter and one check by the target. Learning during the answer
adds an update of the drafter's extra weights. That update can be done in two ways:

  two passes   the drafter is run again on the block it drafted, with gradients, and then the
               gradients are worked out and applied (what dflash/adapt.py does today);
  one pass     the draft itself is made with gradients kept, so that after the target's check
               only the gradients have to be worked out and applied. The drafter is not run again.

This script times the target's check, the draft (plain, with the extra weights, and with the
gradients kept), and the update for each choice of extra weights that is trained (all of them,
the last two layers, the last layer, the attention ones). It does so with the drafter looking
back at 1,000 up to 32,000 tokens, because some of these costs grow with the length of the text.

It also times an update that runs at the same time as the next round (the next draft and its
check), on a second stream of the same GPU, as Test-Time Speculation does it, to see how much
of the update's cost that hides.
The text itself does not matter for the timing, so one saved answer is repeated to make it long
enough. Nothing is decoded and nothing is learned here; the numbers go into the estimate of real
tokens per second for each way of learning.
"""

import argparse
import os
import contextlib
import statistics
import threading
import time

import torch
import torch.nn.functional as F

from dflash.adapt import attach_adapters, choose_adapters, detach_adapters
from dflash.benchmark import load_transformers_models
from dflash.model import _crop_to, _draft_value, _make_cache, _output_head, _raw_input_embeddings, extract_context_feature
from dflash.replay import target_features
from nexus.probe_memory import stepped_aside, write_json
from nexus.replay_memory import load_answers

CHOICES = ("all", "last2", "last1", "attn")


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def middle(times, skip):
    """The median in milliseconds, without the first few (slower) repeats."""
    kept = times[skip:] if len(times) > skip else times
    return 1000 * statistics.median(kept)


@torch.inference_mode()
def time_check(target, model, ids, positions, length, block, repeats, skip, device):
    """The target's check of one block when `length` tokens are already in its cache."""
    cache = _make_cache(target.config)
    for begin in range(0, length, 1024):
        end = min(begin + 1024, length)
        target(ids[:, begin:end], position_ids=positions[:, begin:end], past_key_values=cache, use_cache=True,
               logits_to_keep=1)
    times = []
    for _ in range(repeats):
        start = clock(device)
        output = target(ids[:, length : length + block], position_ids=positions[:, length : length + block],
                        past_key_values=cache, use_cache=True, output_hidden_states=True)
        output.logits.argmax(dim=-1)
        extract_context_feature(output.hidden_states, model.target_layer_ids)
        times.append(clock(device) - start)
        _crop_to(cache, length)
    return middle(times, skip)


def one_draft(model, head, context, noise, positions, cache, keep):
    """One draft as decoding makes it: the block is run, its tokens are chosen, the cache is cut back."""
    block = noise.shape[1]
    states = model(target_hidden=context, noise_embedding=noise, position_ids=positions, past_key_values=cache,
                   use_cache=True)[:, -block:, :]
    logits = model.compute_logits(states[:, 1:, :], head)
    logits.argmax(dim=-1)
    _crop_to(cache, keep)
    return logits


def prime(model, head, features, noise, positions, keep):
    """A drafter cache that holds the first `keep` tokens of the text."""
    cache = _make_cache(model.config)
    block = noise.shape[1]
    one_draft(model, head, features[:, :keep], noise, positions[:, : keep + block], cache, keep)
    return cache


def time_drafts(model, head, features, noise, positions, length, fresh, repeats, skip, device):
    """A draft when the drafter already holds `length - fresh` tokens and reads `fresh` new ones."""
    keep = length - fresh
    block = noise.shape[1]
    with torch.inference_mode():
        cache = prime(model, head, features, noise, positions, keep)
        times = []
        for _ in range(repeats):
            start = clock(device)
            one_draft(model, head, features[:, keep:length], noise, positions[:, keep : length + block], cache, keep)
            times.append(clock(device) - start)
    return middle(times, skip)


def time_learning(model, head, adapters, trained, features, noise, positions, labels, length, fresh, repeats, skip,
                  device):
    """A draft that keeps its gradients, and the update made from it once the labels are known.

    Returns (draft with gradients kept, update from that draft) in milliseconds. A two-pass update
    costs about their sum: the drafter is run again with gradients and then updated."""
    keep = length - fresh
    block = noise.shape[1]
    chosen = {id(adapter) for adapter in trained}
    for adapter in adapters:
        adapter.reset()
        adapter.down.requires_grad_(id(adapter) in chosen)
        adapter.up.requires_grad_(id(adapter) in chosen)
    parameters = [parameter for adapter in trained for parameter in (adapter.down, adapter.up)]
    optimizer = torch.optim.Adam(parameters, lr=1e-6)
    with torch.inference_mode():
        cache = prime(model, head, features, noise, positions, keep)
    drafts, updates = [], []
    with torch.inference_mode(False), torch.enable_grad():
        for layer in cache.layers:               # made without gradients above: plain copies can be used with them
            layer.keys, layer.values = layer.keys.clone(), layer.values.clone()
        context = features[:, keep:length].clone()
        steps = positions[:, keep : length + block].clone()
        block_noise = noise.detach().clone()
        wanted = labels.clone()
        weights = torch.exp(-torch.arange(wanted.shape[0], device=device) / 7.0)
        for _ in range(repeats):
            start = clock(device)
            logits = one_draft(model, head, context, block_noise, steps, cache, keep)
            for layer in cache.layers:           # the kept context must not hold on to the graph
                layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
            drafted = clock(device)
            each = F.cross_entropy(logits[0, : wanted.shape[0]].float(), wanted, reduction="none")
            loss = (each * weights).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            done = clock(device)
            drafts.append(drafted - start)
            updates.append(done - drafted)
    for adapter in adapters:
        adapter.reset()
    return middle(drafts, skip), middle(updates, skip)


def time_overlap(target, model, head, adapters, trained, ids, features, noise, positions, labels, length, fresh,
                 repeats, skip, device):
    """A round of decoding with the update of the round before running at the same time.

    The draft that the update comes from is made on a second stream of the GPU, so its gradients
    are worked out on that stream too, while the next draft and its check run on the first one.
    Returns milliseconds: a round alone (draft and check), the update alone, both at the same time."""
    keep = length - fresh
    block = noise.shape[1]
    chosen = {id(adapter) for adapter in trained}
    for adapter in adapters:
        adapter.reset()
        adapter.down.requires_grad_(id(adapter) in chosen)
        adapter.up.requires_grad_(id(adapter) in chosen)
    optimizer = torch.optim.Adam([parameter for adapter in trained for parameter in (adapter.down, adapter.up)], lr=1e-6)
    side = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    def on_side():
        return torch.cuda.stream(side) if side is not None else contextlib.nullcontext()

    with torch.inference_mode():
        target_cache = _make_cache(target.config)
        for begin in range(0, length, 1024):
            end = min(begin + 1024, length)
            target(ids[:, begin:end], position_ids=positions[:, begin:end], past_key_values=target_cache, use_cache=True,
                   logits_to_keep=1)
        plain_cache = prime(model, head, features, noise, positions, keep)      # for the next round's own draft
        cache = prime(model, head, features, noise, positions, keep)            # for the draft that is learned from
    with torch.inference_mode(False):
        for layer in cache.layers:
            layer.keys, layer.values = layer.keys.clone(), layer.values.clone()
        context = features[:, keep:length].clone()
        steps = positions[:, keep : length + block].clone()
        block_noise = noise.detach().clone()
        wanted = labels.clone()
        weights = torch.exp(-torch.arange(wanted.shape[0], device=device) / 7.0)
    clock(device)            # everything above is finished before the second stream reads it

    def draft_with_gradients():
        clock(device)
        with torch.inference_mode(False), torch.enable_grad(), on_side():
            logits = one_draft(model, head, context, block_noise, steps, cache, keep)
            for layer in cache.layers:
                layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
            each = F.cross_entropy(logits[0, : wanted.shape[0]].float(), wanted, reduction="none")
            loss = (each * weights).sum() / weights.sum()
        clock(device)
        return loss

    def update(loss):
        with torch.inference_mode(False), torch.enable_grad(), on_side():
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if side is not None:
                side.synchronize()

    def next_round():
        with torch.inference_mode():
            one_draft(model, head, features[:, keep:length], noise, positions[:, keep : length + block], plain_cache, keep)
            output = target(ids[:, length : length + block], position_ids=positions[:, length : length + block],
                            past_key_values=target_cache, use_cache=True, output_hidden_states=True)
            output.logits.argmax(dim=-1)
            extract_context_feature(output.hidden_states, model.target_layer_ids)
            if device.type == "cuda":
                torch.cuda.current_stream(device).synchronize()
        _crop_to(target_cache, length)

    rounds, updates, together = [], [], []
    for _ in range(repeats):
        loss = draft_with_gradients()
        start = clock(device)
        next_round()
        after_round = clock(device)
        update(loss)
        updates.append(clock(device) - after_round)
        rounds.append(after_round - start)
        loss = draft_with_gradients()
        start = clock(device)
        worker = threading.Thread(target=update, args=(loss,))
        worker.start()
        next_round()
        worker.join()
        together.append(clock(device) - start)
    for adapter in adapters:
        adapter.reset()
    return middle(rounds, skip), middle(updates, skip), middle(together, skip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--lengths", default="1000,2000,4000,8000,16000,24000,32000")
    parser.add_argument("--fresh", type=int, default=4, help="new tokens the drafter reads in a round")
    parser.add_argument("--learned", type=int, default=4, help="positions of the block an update learns from")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--skip", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False}, indent=1)
    lengths = [int(length) for length in args.lengths.split(",")]
    answers = load_answers(args.data_dir)
    sequence = max(answers, key=lambda answer: answer[2].shape[0])[2].long()
    target, model, _ = load_transformers_models(args.model, args.draft, device)
    block = model.block_size
    needed = max(lengths) + block + 1
    ids = sequence.repeat(needed // sequence.shape[0] + 1)[:needed].to(device)[None]      # long enough, by repeating
    positions = torch.arange(needed + block, device=device)[None]
    head = _output_head(target)
    scale = float(_draft_value(model.config, "input_embedding_scale", 1.0))
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    features = target_features(target, model, ids)

    rows = []
    status = {"done": False, "block": block, "fresh_tokens_per_round": args.fresh,
              "positions_learned_from": args.learned, "repeats": args.repeats, "rank": args.rank,
              "text_tokens_before_repeating": int(sequence.shape[0]), "milliseconds": rows}

    def block_at(length):
        block_ids = torch.full((1, block), model.mask_token_id, dtype=torch.long, device=device)
        block_ids[:, 0] = ids[:, length]
        with torch.inference_mode():
            return _raw_input_embeddings(target, block_ids, scale), ids[0, length + 1 : length + 1 + args.learned]

    def without_adapters():
        detach_adapters(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    # First every piece on its own, for every length. These numbers are saved before anything else is tried.
    for length in lengths:
        noise, labels = block_at(length)
        row = {"tokens_before": length}
        row["target_check"] = time_check(target, model, ids, positions, length, block, args.repeats, args.skip, device)
        row["draft_plain"] = time_drafts(model, head, features, noise, positions, length, args.fresh, args.repeats,
                                         args.skip, device)
        adapters = attach_adapters(model, rank=args.rank)
        row["draft_with_extra_weights"] = time_drafts(model, head, features, noise, positions, length, args.fresh,
                                                      args.repeats, args.skip, device)
        for choice in CHOICES:
            trained = choose_adapters(adapters, choice)
            draft_ms, update_ms = time_learning(model, head, adapters, trained, features, noise, positions, labels,
                                                length, args.fresh, args.repeats, args.skip, device)
            row[choice] = {"draft_keeping_gradients": draft_ms, "update_from_that_draft": update_ms,
                           "update_in_two_passes": draft_ms + update_ms,
                           "trained_parameters": sum(parameter.numel() for adapter in trained
                                                     for parameter in (adapter.down, adapter.up))}
        # The same with the cheaper way to apply extra weights that are not being trained (LowRank.fast),
        # as real decoding uses it (nexus/live_speed.py).
        for adapter in adapters:
            adapter.fast = True
        row["draft_with_extra_weights_fast"] = time_drafts(model, head, features, noise, positions, length, args.fresh,
                                                           args.repeats, args.skip, device)
        for choice in CHOICES:
            draft_ms, update_ms = time_learning(model, head, adapters, choose_adapters(adapters, choice), features, noise,
                                                positions, labels, length, args.fresh, args.repeats, args.skip, device)
            row[choice]["fast"] = {"draft_keeping_gradients": draft_ms, "update_from_that_draft": update_ms}
        without_adapters()
        rows.append(row)
        if device.type == "cuda":
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)

    # Then an update that runs at the same time as the next round, on two streams of the GPU. This is the
    # untried part: if it fails, the GPU is of no use to this process any more, so it comes last and stops.
    failed = False
    for row, length in zip(rows, lengths):
        noise, labels = block_at(length)
        adapters = attach_adapters(model, rank=args.rank)
        for adapter in adapters:
            adapter.fast = True
        for choice in ("all", "last2", "last1"):
            if failed:
                break
            try:
                round_ms, update_ms, both_ms = time_overlap(
                    target, model, head, adapters, choose_adapters(adapters, choice), ids, features, noise, positions,
                    labels, length, args.fresh, args.repeats, args.skip, device)
                row[choice]["at_the_same_time_as_the_next_round"] = {
                    "round_alone": round_ms, "update_alone": update_ms, "one_after_the_other": round_ms + update_ms,
                    "at_the_same_time": both_ms, "added_to_the_round": both_ms - round_ms}
            except Exception as error:
                row[choice]["at_the_same_time_as_the_next_round"] = {"failed": f"{type(error).__name__}: {error}"[:400]}
                failed = True
        write_json(status_path, status, indent=1)
        if failed:
            break
        without_adapters()
    status["done"] = True
    write_json(status_path, status, indent=1)


if __name__ == "__main__":
    main()
