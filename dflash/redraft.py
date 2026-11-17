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
        condition: str = "draft", seed: int = 0, show_rest: bool = False,
    ):
        # keep=0 shows the second drafter none of the draft. It only makes sense
        # with separate weights (the same weights would repeat the first draft),
        # and serves as the control for "does seeing the draft matter".
        lowest = 1 if model is None else 0
        if not (keep == "confidence" or (isinstance(keep, int) and keep >= lowest)):
            raise ValueError("keep must be 'confidence' or an integer (at least 1 when reusing the first drafter)")
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
        # show_rest: instead of masking the slots after the kept ones, show the
        # first draft's own tokens there, so the pass edits a complete draft.
        self.show_rest = show_rest
        self.seed = seed
        self.generator = None
        self.cache = None
        self.waiting = None     # context a separate second drafter has not read yet (rounds it skipped)
        self.kept = []          # the prefix length of the last round

    def start(self, first_drafter: nn.Module) -> None:
        """Called once per generation."""
        self.cache = None if self.model is None else _make_cache(self.model.config)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.waiting = None

    def skip(self, s: SimpleNamespace) -> None:
        """A round without a redraft. A separate second drafter keeps the round's new context
        for its next pass (the first drafter's own cache has it already)."""
        if self.model is not None:
            self.waiting = s.context_features if self.waiting is None \
                else torch.cat((self.waiting, s.context_features), dim=1)

    def _new_context(self, s: SimpleNamespace) -> torch.Tensor:
        context = s.context_features if self.waiting is None else torch.cat((self.waiting, s.context_features), dim=1)
        self.waiting = None
        return context

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
        keep = self._keep_length(s.draft_logits, s.block_ids.shape[1])
        self.kept = [keep]
        return self.redraft(s, keep)

    def redraft(self, s: SimpleNamespace, keep: int, context_done: bool = False, with_logits: bool = False):
        """One more draft of the round `s` with the first `keep` tokens of s.block_ids shown.

        context_done: a separate second drafter has already read this round's new context
        (an earlier redraft of the same round), so it is not fed again.
        with_logits: also return this pass's predictions for the block's draft positions."""
        block_len = s.block_ids.shape[1]
        visible = s.block_ids.clone()
        if not self.show_rest:
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
            cache = self.cache
            context = s.context_features[:, :0] if context_done else self._new_context(s)
            positions = s.position_ids[:, s.start - context.shape[1] : s.start + block_len]
        hidden = drafter(
            target_hidden=context, noise_embedding=noise, position_ids=positions,
            past_key_values=cache, use_cache=True,
        )[:, 1 - block_len :, :]
        _crop_to(cache, s.start)
        logits = drafter.compute_logits(hidden, _output_head(s.target))
        alternative[:, keep + 1 :] = logits.argmax(dim=-1)[:, keep:]
        return (alternative, logits) if with_logits else alternative

    def redraft_together(self, s: SimpleNamespace, keeps: list[int], blocks: torch.Tensor | None = None,
                         context_done: bool = False, no_copies: bool = False):
        """Several redrafts of the round `s` in ONE pass of the drafter: row n shows the first keeps[n]
        tokens of blocks[n] (of the first draft if no blocks are given). Returns the drafts (one per
        row) and the pass's predictions for them.

        The copies of the block are laid one after the other in a single sequence. Every copy has the
        block's own positions and sees the context and itself, never another copy, so each row is the
        draft redraft() would give, while the context is neither copied nor read more than once.
        no_copies=True attends directly, without the copies of the keys the library's masked call
        makes for every head. Timed on an A5000 it was the slower of the two up to 16,000 tokens of
        text for four redrafts or more (nexus/results/long-opd-0/timing.json), so it is off."""
        block_len, count = s.block_ids.shape[1], len(keeps)
        device = s.block_ids.device
        slots = torch.arange(block_len, device=device)[None]
        redrafted = slots > torch.tensor(keeps, device=device)[:, None]            # (count, block)
        blocks = s.block_ids.repeat(count, 1) if blocks is None else blocks
        visible = blocks.clone()
        visible[redrafted] = s.first_drafter.mask_token_id
        drafter = s.first_drafter if self.model is None else self.model
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        noise = _raw_input_embeddings(s.target, visible.reshape(1, -1), scale)
        if self.model is None:
            cache, context = s.draft_cache, s.context_features[:, :0]
        else:
            cache = self.cache
            context = s.context_features[:, :0] if context_done else self._new_context(s)
        seen = cache.get_seq_length() + context.shape[1]                           # the context's entries
        positions = torch.cat((
            s.position_ids[:, s.start - context.shape[1] : s.start],
            s.position_ids[:, s.start : s.start + block_len].repeat(1, count),
        ), dim=1)
        copy = torch.arange(count, device=device).repeat_interleave(block_len)
        sees = torch.ones((count * block_len, seen + count * block_len), dtype=torch.bool, device=device)
        sees[:, seen:] = copy[:, None] == copy[None, :]
        hidden = drafter(
            target_hidden=context, noise_embedding=noise, position_ids=positions,
            attention_mask=sees[None, None], past_key_values=cache, use_cache=True, no_copies=no_copies,
        )[:, -count * block_len :, :]
        _crop_to(cache, s.start)
        logits = drafter.compute_logits(hidden.reshape(count, block_len, -1)[:, 1:, :], _output_head(s.target))
        drafts = blocks.clone()
        drafts[:, 1:] = torch.where(redrafted[:, 1:], logits.argmax(dim=-1), drafts[:, 1:])
        return drafts, logits


def break_scores(confidence: torch.Tensor, fixed: int = 0) -> torch.Tensor:
    """How likely the target's first rejection comes right after the first i draft tokens, for
    i = fixed, fixed + 1, ... (entry i - fixed), when the first `fixed` tokens are taken as right:

        r(i) = c_(fixed+1) ... c_i (1 - c_(i+1))

    confidence: the drafter's probability of each draft token it chose (c_1 first)."""
    rest = confidence[fixed:]
    survive = torch.cat((rest.new_ones(1), rest.cumprod(dim=0)[:-1]))
    return survive * (1.0 - rest)


def _batches(drafter: nn.Module) -> bool:
    """Whether redraft_together can be used: the mask it passes replaces the layers' own."""
    return not any(layer.self_attn.is_causal or layer.self_attn.sliding_window is not None for layer in drafter.layers)


class Branches:
    """Redraft from the places where the first draft is most likely to break, as D2SD does
    (arXiv 2606.04446).

    The k most likely breaks are taken (break_scores): "the target accepts the first i draft tokens
    and rejects token i + 1". For each of them the first i draft tokens stay visible, the rest are
    masked again, and a drafter fills them in: all k redrafts in one pass. These drafts are checked
    together with the first draft; the one the target accepts furthest counts.

    model=None runs the first drafter again. A second drafter of the same architecture can be given.

    differ: a redraft from i exists for the case that the first draft is wrong at token i + 1, so it
    can be made to say something else there (the first draft's own token is already being checked):
      None     it says what it says (D2SD). The same drafter then needs i >= 1: with nothing kept
               it would only repeat its first draft.
      "late"   at token i + 1 it takes its best token other than the first draft's. The tokens
               after it come from the same pass, so they do not know the switch.
      "early"  the first draft's own second choice for token i + 1 is put there before the pass,
               which then fills in the rest knowing the switch.
      "switch" NO second pass at all: the first draft with its own second choice at token i + 1 and
               everything else as it was drafted. This is what branching on the drafter's second
               choices gives without running it again (a small draft tree): the control that
               shows what the redraft pass itself adds to "early".
    With differ the same drafter can also take i = 0: its second choice for the first token is
    shown and the rest is filled in ("late" does this too: the pass has nothing new to go on there).
    tail: the target checks only the token at each break and this many tokens after it (None: all of
    the draft). The check is then shorter, and most of what a branch wins is in those tokens.
    """

    def __init__(self, k: int = 4, model: nn.Module | None = None, differ: str | None = None, together: bool = True,
                 tail: int | None = None):
        if k < 1:
            raise ValueError("k must be at least 1")
        if differ not in (None, "late", "early", "switch"):
            raise ValueError("differ must be None, 'late', 'early' or 'switch'")
        if tail is not None and tail < 0:
            raise ValueError("tail must be 0 or more")
        self.k = k
        self.differ = differ
        self.together = together
        self.alternative_tail = tail
        self.inner = SelfRedraft(model=model, keep=1)
        self.kept = []          # the breaks of the last round, most likely first

    def start(self, first_drafter: nn.Module) -> None:
        self.inner.start(first_drafter)
        self.kept = []

    def choose(self, s: SimpleNamespace) -> list[int]:
        most = s.block_ids.shape[1] - 2  # leave at least one position to redraft
        confidence = s.draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]
        lowest = lowest_break(self.inner, self.differ)
        scores = break_scores(confidence)[lowest : most + 1]
        return (scores.topk(min(self.k, scores.shape[0])).indices + lowest).tolist()

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        s = round_state
        self.kept = self.choose(s)
        return redrafts_from(self.inner, s, self.kept, self.differ, self.together)


def lowest_break(inner: SelfRedraft, differ: str | None) -> int:
    """The same drafter with nothing kept and nothing switched would repeat its first draft."""
    return 1 if inner.model is None and differ is None else 0


def redrafts_from(inner: SelfRedraft, s: SimpleNamespace, breaks: list[int], differ: str | None = None,
                  together: bool = True) -> torch.Tensor:
    """The redrafts of the round `s` for each of `breaks` (the first i draft tokens right, token i + 1
    wrong), one per row. See Branches for `differ`."""
    count, device = len(breaks), s.block_ids.device
    rows = torch.arange(count, device=device)
    at = torch.tensor(breaks, device=device)
    blocks, keeps = s.block_ids.repeat(count, 1), at.clone()
    early = torch.zeros(count, dtype=torch.bool, device=device)
    if differ in ("early", "switch"):
        early[:] = True
    elif differ == "late" and inner.model is None:
        early = at == 0
    if bool(early.any()):
        second_choice = s.draft_logits[0].topk(2, dim=-1).indices[:, 1]       # of draft tokens 1, 2, ...
        blocks[rows[early], at[early] + 1] = second_choice[at[early]]
        keeps = keeps + early.long()
    if differ == "switch":                      # the drafter is not run again
        inner.skip(s)
        return blocks
    drafter = s.first_drafter if inner.model is None else inner.model
    if together and count > 1 and _batches(drafter):
        drafts, logits = inner.redraft_together(s, keeps.tolist(), blocks)
    else:
        made = [inner.redraft(SimpleNamespace(**{**vars(s), "block_ids": blocks[number : number + 1]}), keep,
                              context_done=number > 0, with_logits=True)
                for number, keep in enumerate(keeps.tolist())]
        drafts, logits = torch.cat([draft for draft, _ in made]), torch.cat([logit for _, logit in made])
    late = ~early if differ == "late" else torch.zeros_like(early)
    if bool(late.any()):
        best = logits[rows, at].topk(2, dim=-1).indices                       # token i + 1 is row i of the predictions
        before = s.block_ids[0, at + 1]
        other = torch.where(best[:, 0] == before, best[:, 1], best[:, 0])
        drafts[rows[late], at[late] + 1] = other[late]
    return drafts


class Chain:
    """One redraft after the other, each from the likely break of the draft before it.

    The second draft keeps the first draft up to its most likely break and fills in the rest. The
    third keeps the second draft up to ITS most likely break (further on), and so on, `depth` drafts
    in all with the first. Every pass reads more of what the passes before it wrote. All the drafts
    are checked together; the one the target accepts furthest counts (replay only)."""

    def __init__(self, depth: int = 3, model: nn.Module | None = None):
        if depth < 2:
            raise ValueError("depth counts the first draft, so it is at least 2")
        self.depth = depth
        self.inner = SelfRedraft(model=model, keep=1)
        self.kept = []          # the break of the first draft that the second draft starts from
        self.levels = []        # how many tokens each redraft of the last round kept

    def start(self, first_drafter: nn.Module) -> None:
        self.inner.start(first_drafter)
        self.kept, self.levels = [], []

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        s = round_state
        most = s.block_ids.shape[1] - 2
        confidence = s.draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]
        lowest = lowest_break(self.inner, None)
        current, fixed, drafts, self.levels = s, 0, [], []
        for level in range(1, self.depth):
            low = lowest if level == 1 else fixed + 1
            if low > most:
                break
            keep = low + int(break_scores(confidence, fixed)[low - fixed : most + 1 - fixed].argmax())
            made, logits = self.inner.redraft(current, keep, context_done=level > 1, with_logits=True)
            confidence = torch.cat((confidence[:keep], logits.float().softmax(dim=-1).amax(dim=-1)[0, keep:]))
            current = SimpleNamespace(**{**vars(s), "block_ids": made})
            fixed = keep
            drafts.append(made)
            self.levels.append(keep)
        self.kept = self.levels[:1]
        if not drafts:
            self.inner.skip(s)
            return None
        return torch.cat(drafts, dim=0)


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
    round_clock: list | None = None,
):
    """Greedy DFlash decoding that verifies the first draft and alternative drafts together.

    With ``improver=None`` this is ordinary DFlash and produces the same tokens.
    The improver may return several alternative drafts (one per row). Each shares the first draft's
    tokens up to the first place it differs (its split), so only its tokens from there on are added
    to the target's one verification pass, where they see the text, the shared tokens and
    themselves. The draft the target accepts furthest counts. If the improver has
    ``alternative_tail = N``, only the token at the split and the N tokens after it are checked
    (a shorter pass: most of what an alternative wins is in those tokens).
    round_log, if given, receives one entry per full round:
    [block length, split of the best alternative (0 if none), tokens accepted from the first
    draft, tokens accepted from the best alternative (-1 if none), confidences].
    timers, if given, accumulates seconds spent in each part of a round.
    round_clock, if given, receives [tokens written before the round, tokens it produced,
    seconds since decoding began] at the end of every round.
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

    decode_start = _cuda_time() if return_stats or round_clock is not None else None
    stop_tokens = (
        torch.tensor(stop_token_ids, dtype=torch.long, device=device)
        if stop_token_ids is not None else None
    )
    acceptance_lengths = []
    rounds_with_alternative = alternative_wins = alternative_extra_tokens = rounds_edited = 0
    start = num_input_tokens
    # Accepted tokens whose target cache entries were computed on the alternative
    # branch. They are fed again at the front of the next verification pass.
    pending = output_ids[:, :0]
    stopped = stop_tokens is not None and torch.isin(output_ids[:, start], stop_tokens).any()
    # A plug-in may ask for what the target predicted on each draft (see dflash/adapt.py).
    canvas_top = int(getattr(improver, "canvas_top", 0) or 0)
    canvas = None
    # A plug-in that learns from the draft's own pass makes the draft itself, and is told how the
    # check went right after it (dflash.adapt.OnePassTrainer).
    learner = improver if getattr(improver, "learns_from_the_draft", False) else None

    while start + 1 < max_length and not stopped:
        block_len = min(block_size, max_length - start)
        block_ids = output_ids[:, start : start + block_len].clone()
        alternative = None
        if block_len > 1:
            noise = _raw_input_embeddings(
                target, block_ids, float(_draft_value(model.config, "input_embedding_scale", 1.0)),
            )
            clock = _cuda_time() if timers is not None else None
            if learner is not None and block_len > 2:
                draft_states, draft_logits = learner.draft(
                    model, _output_head(target), context_features, noise,
                    position_ids[:, start - context_features.shape[1] : start + block_len], draft_cache, start)
            else:
                draft_states = model(
                    target_hidden=context_features, noise_embedding=noise,
                    position_ids=position_ids[:, start - context_features.shape[1] : start + block_len],
                    past_key_values=draft_cache, use_cache=True,
                )[:, -block_len:, :]
                hidden = draft_states[:, 1:, :]
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
                    draft_cache=draft_cache, target=target, draft_states=draft_states,
                    canvas=canvas,
                ))

            if timers is not None and improver is not None:
                timers["improve"] = timers.get("improve", 0.0) + _cuda_time() - clock

        # An improver that replaces the draft: the edited draft is the only one checked.
        if alternative is not None and getattr(improver, "replace", False):
            if alternative.shape == block_ids.shape and bool((alternative[:, :1] == block_ids[:, :1]).all()):
                rounds_edited += int(bool((alternative != block_ids).any()))
                block_ids = alternative
            alternative = None

        # An alternative draft shares everything before its `split` with the first draft; only its
        # tokens from `split` up to `end` are added to the verification pass. [(draft, split, end)]
        branches = []
        if alternative is not None:
            tail = getattr(improver, "alternative_tail", None)
            for row in alternative.unbind(0):
                differs = (row != block_ids[0]).nonzero()
                if differs.numel() and int(differs[0]) >= 1:
                    split = int(differs[0])
                    branches.append((row[None], split, block_len if tail is None else min(block_len, split + 1 + tail)))
        extra = sum(end - split for _, split, end in branches)
        num_pending = pending.shape[1]
        verify_ids = [pending, block_ids]
        verify_positions = [position_ids[:, start - num_pending : start + block_len]]
        attention_mask = None
        if branches:
            rounds_with_alternative += 1
            cached = start - num_pending
            rows = num_pending + block_len + extra
            visible = torch.ones((rows, cached + rows), dtype=torch.bool, device=device).tril(cached)
            branch_row = num_pending + block_len
            for row, split, end in branches:
                verify_ids.append(row[:, split:end])
                verify_positions.append(position_ids[:, start + split : start + end])
                # An alternative's tokens must not see the first draft's tokens after its split, nor
                # the tokens of the alternatives laid out before it.
                visible[branch_row : branch_row + end - split, cached + num_pending + split : cached + branch_row] = False
                branch_row += end - split
            attention_mask = visible[None, None]
        clock = _cuda_time() if timers is not None else None
        output = target(
            torch.cat(verify_ids, dim=1), position_ids=torch.cat(verify_positions, dim=1),
            past_key_values=target_cache, use_cache=True,
            output_hidden_states=block_len > 1, attention_mask=attention_mask,
        )
        verified = output.logits.argmax(dim=-1)
        if canvas_top and block_len > 1:
            best = output.logits[0, num_pending : num_pending + block_len - 1].float().log_softmax(dim=-1)
            best = best.topk(canvas_top, dim=-1)
            canvas = (start, best.indices, best.values)
        if timers is not None:
            kind = "two" if branches else "one"
            timers["verify_" + kind] = timers.get("verify_" + kind, 0.0) + _cuda_time() - clock
            timers["rounds_" + kind] = timers.get("rounds_" + kind, 0) + 1
            timers["rows_" + kind] = timers.get("rows_" + kind, 0) + num_pending + block_len + extra

        first_rows = torch.arange(num_pending, num_pending + block_len, device=device)
        accepted = _leading_matches(block_ids[:, 1:], verified[:, first_rows[:-1]])
        first_accepted, alternative_accepted = accepted, -1
        winner_ids, winner_rows, alternative_won = block_ids, first_rows, False
        winner_split = best_split = 0
        branch_row = num_pending + block_len
        for row, split, end in branches:
            branch_rows = torch.cat((first_rows[:split], torch.arange(branch_row, branch_row + end - split, device=device)))
            branch_row += end - split
            # If all its checked tokens are accepted, the target's token after the last one is the bonus.
            got = _leading_matches(row[:, 1:end], verified[:, branch_rows[:-1]])
            if got > alternative_accepted:
                alternative_accepted, best_split = got, split
            if got > accepted:
                accepted, alternative_won = got, True
                winner_ids, winner_rows, winner_split = row, branch_rows, split
        if alternative_won:
            alternative_wins += 1
            alternative_extra_tokens += accepted - first_accepted

        if round_log is not None and block_len == block_size:
            confidence = draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]
            round_log.append([
                block_len, best_split, first_accepted,
                alternative_accepted, [round(c, 3) for c in confidence.tolist()],
            ])

        bonus = verified[0, winner_rows[accepted]]
        output_ids[:, start : start + accepted + 1] = winner_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = bonus
        if learner is not None and block_len > 2:
            # What the target made of this draft: the accepted tokens, then its own token.
            clock = _cuda_time() if timers is not None else None
            learner.learn(start, first_accepted, output_ids[0, start + 1 : start + first_accepted + 2], canvas)
            if timers is not None:
                timers["learn"] = timers.get("learn", 0.0) + _cuda_time() - clock
        produced = min(accepted + 1, max_length - start - 1)
        if stop_tokens is not None:
            stop_indices = torch.isin(
                output_ids[0, start + 1 : start + produced + 1], stop_tokens
            ).nonzero(as_tuple=True)[0]
            if stop_indices.numel() > 0:
                produced = stop_indices[0].item() + 1
                stopped = True

        # Target cache: keep what lies on the accepted path and is stored in order.
        kept_in_cache = min(produced, winner_split) if alternative_won else produced
        _crop_to(target_cache, start + kept_in_cache)
        pending = output_ids[:, start + kept_in_cache : start + produced].clone()
        if block_len > 1:
            features = extract_context_feature(output.hidden_states, model.target_layer_ids)
            context_features = features[:, winner_rows[:produced], :]
        if round_clock is not None:
            round_clock.append([start - num_input_tokens, produced, _cuda_time() - decode_start])
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
        rounds_edited=rounds_edited,
    )
