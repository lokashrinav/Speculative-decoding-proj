"""Decoding an answer again without the target's check.

Greedy speculative decoding is lossless: whatever the drafter proposes, the answer is
the target's own greedy text. So once an answer has been decoded, any way of drafting
can be scored against that text without checking each draft with the target again: the
tokens a draft would have had accepted are its leading tokens that equal the known
text. Only the drafter (and whatever improves its draft) runs, round by round, exactly
as in decoding. The target runs once per answer, over the finished text, to provide
the hidden states the drafter reads.

replay_decode takes the same plug-ins as dflash_generate_redraft and gives the same
rounds, at a fraction of the cost (the target's check is most of a round).
"""

from types import SimpleNamespace

import torch

from .model import (
    _crop_to,
    _draft_value,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    extract_context_feature,
)


@torch.inference_mode()
def target_features(target, model, ids, chunk=1024, top=0):
    """The target's hidden states that the drafter reads, for every position of a finished answer.

    With top > 0 it also returns what the target itself predicted after every position: the `top`
    most likely next tokens and their log-probabilities, (length, top) each. Row p describes the
    token at position p + 1, given the text up to p."""
    cache = _make_cache(target.config)
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    parts, tokens, scores = [], [], []
    for begin in range(0, ids.shape[1], chunk):
        output = target(
            ids[:, begin : begin + chunk], position_ids=positions[:, begin : begin + chunk],
            past_key_values=cache, use_cache=True, output_hidden_states=True,
            **({} if top else {"logits_to_keep": 1}),
        )
        parts.append(extract_context_feature(output.hidden_states, model.target_layer_ids))
        if top:
            best = output.logits[0].float().log_softmax(dim=-1).topk(top, dim=-1)
            tokens.append(best.indices)
            scores.append(best.values)
    features = torch.cat(parts, dim=1)
    if top:
        return features, (torch.cat(tokens), torch.cat(scores))
    return features


def _checked_matches(alternative, block_ids, truth, tail):
    """Tokens the target accepts from an alternative draft. With a tail, only the first token that
    differs from the first draft and `tail` tokens after it are checked (see dflash_generate_redraft)."""
    got = _matches(alternative[:, 1:], truth)
    if tail is None:
        return got
    differs = (alternative != block_ids)[0].nonzero()
    return got if not differs.numel() else min(got, int(differs[0]) + tail)


@torch.inference_mode()
def replay_decode(model, target, ids, prompt_length, ended=True, features=None, improver=None,
                  block_size=None, timers=None):
    """Replays the decoding of one finished answer.

    ids: (1, length) the prompt followed by the answer the target wrote.
    ended: whether the answer stopped by itself. If it was cut at a length limit instead,
        the last blocks were shorter in decoding, and they are made shorter here too.
    improver: the same plug-in dflash_generate_redraft takes. One that replaces the draft
        has its edited draft scored; any other alternative draft counts if it gets further.
        A plug-in may return several alternative drafts (one per row): the best one counts. If it
        has alternative_tail = N, only the first differing token of each and N tokens after it count.
    Returns the tokens gained in each round, as dflash_generate_redraft reports them.
    """
    length = ids.shape[1]
    block_size = model.block_size if block_size is None else block_size
    device = ids.device
    if features is None:
        features = target_features(target, model, ids)
    position_ids = torch.arange(length + block_size, device=device)[None]
    draft_cache = _make_cache(model.config)
    head = _output_head(target)
    scale = float(_draft_value(model.config, "input_embedding_scale", 1.0))
    if improver is not None:
        improver.start(model)
    # A plug-in may ask for what the target predicted on each draft (the signal Test-Time Speculation
    # trains on). Then the target does check every draft here too, which costs what decoding costs.
    canvas_top = int(getattr(improver, "canvas_top", 0) or 0)
    # A control: let the drafter see only the last `context_window` tokens of the text so far (0 = all).
    # It was trained on short texts, so a long context may itself be what hurts it.
    context_window = int(getattr(improver, "context_window", 0) or 0)
    target_cache = _make_cache(target.config) if canvas_top else None
    target_fed, canvas = 0, None
    # A plug-in that learns from the draft's own pass makes the draft itself, and is told how the
    # check went right after it (dflash.adapt.OnePassTrainer).
    learner = improver if getattr(improver, "learns_from_the_draft", False) else None

    acceptance_lengths = []
    rounds_edited = rounds_with_alternative = alternative_wins = 0
    start, fed = prompt_length, 0
    while start + 1 < length:
        block_len = block_size if ended else min(block_size, length - start)
        block_ids = torch.full((1, block_len), model.mask_token_id, dtype=torch.long, device=device)
        block_ids[:, 0] = ids[:, start]
        alternative = None
        if block_len > 1:
            clock = _clock(device) if timers is not None else None
            noise = _raw_input_embeddings(target, block_ids, scale)
            taught = learner is not None and block_len > 2 and not context_window
            if taught:
                draft_states, draft_logits = learner.draft(
                    model, head, features[:, fed:start], noise, position_ids[:, fed : start + block_len],
                    draft_cache, start)
            else:
                draft_states = model(
                    target_hidden=features[:, fed:start], noise_embedding=noise,
                    position_ids=position_ids[:, fed : start + block_len],
                    past_key_values=draft_cache, use_cache=True,
                )[:, -block_len:, :]
            if taught:
                pass                                         # the learner has cut the cache back already
            elif context_window:
                draft_cache.crop(-block_len)                 # the block's own entries go, as always
                for layer in draft_cache.layers:             # ... and so does context older than the window
                    if layer.keys is not None and layer.keys.shape[-2] > context_window:
                        layer.keys = layer.keys[..., -context_window:, :]
                        layer.values = layer.values[..., -context_window:, :]
            else:
                _crop_to(draft_cache, start)
            context_features, fed = features[:, fed:start], start
            if not taught:
                draft_logits = model.compute_logits(draft_states[:, 1:, :], head)
            block_ids[:, 1:] = draft_logits.argmax(dim=-1)
            if timers is not None:
                now = _clock(device)
                timers["draft"] = timers.get("draft", 0.0) + now - clock
                clock = now
            if improver is not None and block_len > 2:
                alternative = improver(SimpleNamespace(
                    block_ids=block_ids, draft_logits=draft_logits, start=start,
                    position_ids=position_ids, context_features=context_features,
                    accepted_ids=ids[:, :start], first_drafter=model,
                    draft_cache=draft_cache, target=target, draft_states=draft_states,
                    canvas=canvas,
                ))
            if timers is not None and improver is not None:
                timers["improve"] = timers.get("improve", 0.0) + _clock(device) - clock
        if alternative is not None and getattr(improver, "replace", False):
            if alternative.shape == block_ids.shape and bool((alternative[:, :1] == block_ids[:, :1]).all()):
                rounds_edited += int(bool((alternative != block_ids).any()))
                block_ids = alternative
            alternative = None

        if canvas_top and block_len > 1:
            clock = _clock(device) if timers is not None else None
            output = target(
                torch.cat((ids[:, target_fed:start], block_ids), dim=1),
                position_ids=position_ids[:, target_fed : start + block_len],
                past_key_values=target_cache, use_cache=True, logits_to_keep=block_len,
            )
            # Row k is what the target expects after the block's first k + 1 tokens: the token at
            # start + 1 + k if the draft was right so far, and a token of another text if it was not.
            best = output.logits[0, -block_len:-1].float().log_softmax(dim=-1).topk(canvas_top, dim=-1)
            canvas = (start, best.indices, best.values)
            if timers is not None:
                timers["target"] = timers.get("target", 0.0) + _clock(device) - clock

        truth = ids[:, start + 1 : start + block_len]
        accepted = _matches(block_ids[:, 1:], truth)
        own = accepted
        if learner is not None and block_len > 2:
            clock = _clock(device) if timers is not None else None
            learner.learn(start, own, truth[0, : own + 1], canvas)
            if timers is not None:
                timers["learn"] = timers.get("learn", 0.0) + _clock(device) - clock
        if alternative is not None and alternative.shape[1:] == block_ids.shape[1:] \
                and bool((alternative[:, :1] == block_ids[:, :1]).all()) and bool((alternative != block_ids).any()):
            rounds_with_alternative += 1
            other = max(_checked_matches(alternative[row : row + 1], block_ids, truth, getattr(improver, "alternative_tail", None))
                        for row in range(alternative.shape[0]))
            if other > accepted:
                alternative_wins += 1
                accepted = other
        produced = min(accepted + 1, length - start - 1)
        if canvas_top and block_len > 1:
            # The target keeps what it read of the committed text: the start token and the accepted draft tokens.
            target_fed = start + min(produced, own + 1)
            _crop_to(target_cache, target_fed)
        start += produced
        acceptance_lengths.append(produced)
        if timers is not None:
            timers["rounds"] = timers.get("rounds", 0) + 1

    return SimpleNamespace(
        acceptance_lengths=acceptance_lengths, rounds=len(acceptance_lengths),
        num_output_tokens=length - prompt_length, rounds_edited=rounds_edited,
        rounds_with_alternative=rounds_with_alternative, alternative_wins=alternative_wins,
    )


def _matches(proposed, truth):
    """Leading proposed tokens that equal the known text (which may end earlier)."""
    count = min(proposed.shape[1], truth.shape[1])
    if count == 0:
        return 0
    return int((proposed[:, :count] == truth[:, :count]).long().cumprod(dim=1).sum(dim=1)[0].item())


def _clock(device):
    import time

    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()
