"""Keep the first draft, add an improved draft, and let the target check both.

This is the general frame for "use the draft to improve the draft":

    first draft  ->  improver(first draft)  ->  alternative draft
    target verifies [first draft] and [alternative draft] in one forward pass
    whichever the target accepts further is kept

The improver is a plug-in. Anything that maps a draft to another draft fits:
the same drafter re-reading its own draft, a separately trained second drafter,
a small editor, or a rule. The original draft is always kept, so an improver can
only add accepted tokens, never remove them. Greedy decoding only.
"""

from types import SimpleNamespace

import torch
from torch import nn

from .model import (
    DFlash2DraftModel,
    _crop_to,
    _cuda_time,
    _draft_value,
    _make_cache,
    _output_head,
    _raw_input_embeddings,
    extract_context_feature,
)


class SelfRedraft:
    """Improve a draft by showing a drafter the first part of that draft.

    The first ``keep`` draft tokens stay visible, the rest are masked again, and
    a DFlash-architecture drafter fills them in. With ``model=None`` the first
    drafter itself is reused, so there is no second model. Pass another
    checkpoint of the same architecture (for example a drafter trained for this
    job) to use it as the second drafter instead.

    keep: a positive integer, or "confidence" to keep everything before the
    position where the first draft is most likely to be rejected first.
    """

    def __init__(
        self, model: nn.Module | None = None, keep: int | str = "confidence",
        condition: str = "draft", seed: int = 0,
    ):
        if not (keep == "confidence" or (isinstance(keep, int) and keep >= 1)):
            raise ValueError("keep must be a positive integer or 'confidence'")
        if condition not in ("draft", "runner_up", "random", "soft"):
            raise ValueError("condition must be 'draft', 'runner_up', 'random' or 'soft'")
        if condition != "draft" and keep != 1:
            raise ValueError("conditions other than 'draft' change slot 1 only, so they need keep=1")
        self.model = model
        self.keep = keep
        # What the second pass is shown at slot 1 (used to find out which part of
        # the draft carries the useful information):
        #   draft      the first draft's own token
        #   runner_up  the first draft's second choice, which also becomes the alternative's token
        #   random     a random token (the alternative still keeps the draft's token)
        #   soft       the probability-weighted mix of all candidate embeddings
        self.condition = condition
        self.seed = seed
        self.generator = None
        self.cache = None

    def start(self, first_drafter: nn.Module) -> None:
        """Called once per generation."""
        self.cache = None if self.model is None else _make_cache(self.model.config)
        self.generator = torch.Generator().manual_seed(self.seed)

    def _keep_length(self, draft_logits: torch.Tensor, block_len: int) -> int:
        most = block_len - 2  # leave at least one position to redraft
        if self.keep != "confidence":
            return min(int(self.keep), most)
        confidence = draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]  # one per draft token
        survive = confidence.cumprod(dim=0)
        # P(first i draft tokens right, token i+1 wrong), for i = 1 .. most
        boundary = survive[:most] * (1.0 - confidence[1 : most + 1])
        return int(boundary.argmax().item()) + 1

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        s = round_state
        block_len = s.block_ids.shape[1]
        keep = self._keep_length(s.draft_logits, block_len)
        visible = s.block_ids.clone()
        visible[:, keep + 1 :] = s.first_drafter.mask_token_id
        alternative = s.block_ids.clone()
        if self.condition == "runner_up":
            visible[:, 1] = s.draft_logits[:, 0].topk(2, dim=-1).indices[:, 1]
            alternative[:, 1] = visible[:, 1]
        elif self.condition == "random":
            vocab = s.draft_logits.shape[-1]
            visible[:, 1] = torch.randint(vocab, (1,), generator=self.generator).item()
        drafter = s.first_drafter if self.model is None else self.model
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        noise = _raw_input_embeddings(s.target, visible, scale)
        if self.condition == "soft":
            weights = s.target.get_input_embeddings().weight
            probabilities = s.draft_logits[:, 0].float().softmax(dim=-1).to(weights.dtype)
            noise[:, 1] = (probabilities @ weights) * scale
        if self.model is None:
            # The first drafter's cache already holds the context for this round.
            cache, context = s.draft_cache, s.context_features[:, :0]
            positions = s.position_ids[:, s.start : s.start + block_len]
        else:
            cache, context = self.cache, s.context_features
            positions = s.position_ids[:, s.start - context.shape[1] : s.start + block_len]
        hidden = drafter(
            target_hidden=context, noise_embedding=noise, position_ids=positions,
            past_key_values=cache, use_cache=True,
        )[:, 1 - block_len :, :]
        _crop_to(cache, s.start)
        logits = drafter.compute_logits(hidden, _output_head(s.target))
        alternative[:, keep + 1 :] = logits.argmax(dim=-1)[:, keep:]
        return alternative


class Resample:
    """A second draft with no second pass: keep slot 1, sample the other slots
    from the probabilities the first pass already produced."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.generator = None

    def start(self, first_drafter: nn.Module) -> None:
        self.generator = None

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        s = round_state
        if self.generator is None:
            self.generator = torch.Generator(device=s.draft_logits.device).manual_seed(self.seed)
        probabilities = s.draft_logits[0, 1:].float().softmax(dim=-1)
        alternative = s.block_ids.clone()
        alternative[0, 2:] = torch.multinomial(probabilities, 1, generator=self.generator)[:, 0]
        return alternative


def confidence_features(draft_logits: torch.Tensor) -> torch.Tensor:
    """Per-round numbers a gate can use, all from the first draft alone:
    the confidence of each draft token, then for each slot the chance that the
    first rejection happens exactly there."""
    confidence = draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]
    survive = confidence.cumprod(dim=0)
    first_rejection = torch.cat((1.0 - confidence[:1], survive[:-1] * (1.0 - confidence[1:])))
    return torch.cat((confidence, first_rejection))


class Gated:
    """Run the inner improver only in rounds where a linear score of the first
    draft's confidences says the second draft is likely to win."""

    def __init__(self, inner, weights: list[float], bias: float, threshold: float = 0.0):
        self.inner = inner
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.bias = bias
        self.threshold = threshold
        self.fired = 0

    def start(self, first_drafter: nn.Module) -> None:
        self.inner.start(first_drafter)
        self.fired = 0

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        features = confidence_features(round_state.draft_logits)
        if features.shape[0] != self.weights.shape[0]:
            return None  # short final block
        score = float(features @ self.weights.to(features.device)) + self.bias
        if score <= self.threshold:
            return None
        self.fired += 1
        return self.inner(round_state)


def _leading_matches(proposed: torch.Tensor, verified: torch.Tensor) -> int:
    return int((proposed == verified).cumprod(dim=1).sum(dim=1)[0].item())


@torch.inference_mode()
def dflash_generate_redraft(
    model: nn.Module,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: list[int] | None,
    improver=None,
    block_size: int | None = None,
    return_stats: bool = False,
    round_log: list | None = None,
    timers: dict | None = None,
):
    """Greedy DFlash decoding that verifies the first draft and an improved draft together.

    With ``improver=None`` this is ordinary DFlash and produces the same tokens.
    round_log, if given, receives one entry per full round:
    [block length, split (0 if no alternative), tokens accepted from the first
    draft, tokens accepted from the alternative (-1 if none), confidences].
    timers, if given, accumulates seconds spent in each part of a round.
    """
    if isinstance(model, DFlash2DraftModel):
        raise ValueError("Two-draft decoding supports the original DFlash drafter only")
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    device = target.device

    output_ids = torch.full((1, max_length + 1), model.mask_token_id, dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    target_cache = _make_cache(target.config)
    draft_cache = _make_cache(model.config)
    if improver is not None:
        improver.start(model)

    prefill_start = _cuda_time() if return_stats else None
    output = target(
        input_ids, position_ids=position_ids[:, :num_input_tokens],
        past_key_values=target_cache, use_cache=True, logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens] = output.logits[:, -1].argmax(dim=-1)
    if block_size > 1:
        context_features = extract_context_feature(output.hidden_states, model.target_layer_ids)
    _crop_to(target_cache, num_input_tokens)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    stop_tokens = (
        torch.tensor(stop_token_ids, dtype=torch.long, device=device)
        if stop_token_ids is not None else None
    )
    acceptance_lengths = []
    rounds_with_alternative = alternative_wins = alternative_extra_tokens = 0
    start = num_input_tokens
    # Accepted tokens whose target cache entries were computed on the alternative
    # branch. They are fed again at the front of the next verification pass.
    pending = output_ids[:, :0]
    stopped = stop_tokens is not None and torch.isin(output_ids[:, start], stop_tokens).any()

    while start + 1 < max_length and not stopped:
        block_len = min(block_size, max_length - start)
        block_ids = output_ids[:, start : start + block_len].clone()
        alternative = None
        if block_len > 1:
            noise = _raw_input_embeddings(
                target, block_ids, float(_draft_value(model.config, "input_embedding_scale", 1.0)),
            )
            clock = _cuda_time() if timers is not None else None
            hidden = model(
                target_hidden=context_features, noise_embedding=noise,
                position_ids=position_ids[:, start - context_features.shape[1] : start + block_len],
                past_key_values=draft_cache, use_cache=True,
            )[:, 1 - block_len :, :]
            _crop_to(draft_cache, start)
            draft_logits = model.compute_logits(hidden, _output_head(target))
            block_ids[:, 1:] = draft_logits.argmax(dim=-1)
            if timers is not None:
                now = _cuda_time()
                timers["draft"] = timers.get("draft", 0.0) + now - clock
                clock = now
            if improver is not None and block_len > 2:
                alternative = improver(SimpleNamespace(
                    block_ids=block_ids, draft_logits=draft_logits, start=start,
                    position_ids=position_ids, context_features=context_features,
                    accepted_ids=output_ids[:, :start], first_drafter=model,
                    draft_cache=draft_cache, target=target,
                ))

            if timers is not None and improver is not None:
                timers["improve"] = timers.get("improve", 0.0) + _cuda_time() - clock

        # The two drafts share everything before `split`; only the alternative's
        # tokens from `split` onward are added to the verification pass.
        split = block_len
        if alternative is not None:
            differs = (alternative != block_ids)[0].nonzero()
            if differs.numel() and int(differs[0]) >= 1:
                split = int(differs[0])
            else:
                alternative = None
        extra = block_len - split
        num_pending = pending.shape[1]
        verify_ids = [pending, block_ids]
        verify_positions = [position_ids[:, start - num_pending : start + block_len]]
        attention_mask = None
        if alternative is not None:
            rounds_with_alternative += 1
            verify_ids.append(alternative[:, split:])
            verify_positions.append(position_ids[:, start + split : start + block_len])
            cached = start - num_pending
            rows = num_pending + block_len + extra
            visible = torch.ones((rows, cached + rows), dtype=torch.bool, device=device).tril(cached)
            first_extra_row = num_pending + block_len
            # Alternative tokens must not see the first draft's tokens after the split.
            visible[first_extra_row:, cached + num_pending + split : cached + first_extra_row] = False
            attention_mask = visible[None, None]
        clock = _cuda_time() if timers is not None else None
        output = target(
            torch.cat(verify_ids, dim=1), position_ids=torch.cat(verify_positions, dim=1),
            past_key_values=target_cache, use_cache=True,
            output_hidden_states=block_len > 1, attention_mask=attention_mask,
        )
        verified = output.logits.argmax(dim=-1)
        if timers is not None:
            kind = "two" if alternative is not None else "one"
            timers["verify_" + kind] = timers.get("verify_" + kind, 0.0) + _cuda_time() - clock
            timers["rounds_" + kind] = timers.get("rounds_" + kind, 0) + 1
            timers["rows_" + kind] = timers.get("rows_" + kind, 0) + num_pending + block_len + extra

        first_rows = torch.arange(num_pending, num_pending + block_len, device=device)
        accepted = _leading_matches(block_ids[:, 1:], verified[:, first_rows[:-1]])
        first_accepted, alternative_accepted = accepted, -1
        winner_ids, winner_rows, alternative_won = block_ids, first_rows, False
        if alternative is not None:
            alternative_rows = torch.cat((
                first_rows[:split],
                torch.arange(num_pending + block_len, num_pending + block_len + extra, device=device),
            ))
            alternative_accepted = _leading_matches(alternative[:, 1:], verified[:, alternative_rows[:-1]])
            if alternative_accepted > accepted:
                alternative_wins += 1
                alternative_extra_tokens += alternative_accepted - accepted
                accepted, alternative_won = alternative_accepted, True
                winner_ids, winner_rows = alternative, alternative_rows

        if round_log is not None and block_len == block_size:
            confidence = draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]
            round_log.append([
                block_len, split if alternative is not None else 0, first_accepted,
                alternative_accepted, [round(c, 3) for c in confidence.tolist()],
            ])

        bonus = verified[0, winner_rows[accepted]]
        output_ids[:, start : start + accepted + 1] = winner_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = bonus
        produced = min(accepted + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_indices = torch.isin(
                output_ids[0, start + 1 : start + produced + 1], stop_tokens
            ).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                produced = stop_indices[0].item() + 1
                stopped = True

        # Target cache: keep what lies on the accepted path and is stored in order.
        kept_in_cache = min(produced, split) if alternative_won else produced
        _crop_to(target_cache, start + kept_in_cache)
        pending = output_ids[:, start + kept_in_cache : start + produced].clone()
        if block_len > 1:
            features = extract_context_feature(output.hidden_states, model.target_layer_ids)
            context_features = features[:, winner_rows[:produced], :]
        start += produced
        acceptance_lengths.append(produced)

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
        rounds=len(acceptance_lengths),
        rounds_with_alternative=rounds_with_alternative,
        alternative_wins=alternative_wins,
        alternative_extra_tokens=alternative_extra_tokens,
    )
