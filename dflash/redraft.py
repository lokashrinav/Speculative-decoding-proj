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

import heapq
import math
from types import SimpleNamespace

import numpy
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

    def roll_back(self, tokens: int) -> None:
        """Decoding went back: the last `tokens` positions of context are read again (a token the
        target had accepted on an alternative's rows was corrected). A separate second drafter
        forgets them; the first drafter's own cache is cut back by the decoding loop."""
        if self.cache is None:
            return
        if self.waiting is not None:
            held = self.waiting.shape[1]
            self.waiting = self.waiting[:, : held - tokens] if held > tokens else None
            tokens -= min(held, tokens)
        if tokens > 0:
            self.cache.crop(-tokens)

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
        cache.crop(-block_len)                               # the block's own entries go, the context stays
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
        cache.crop(-count * block_len)                       # the copies' own entries go, the context stays
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

    def roll_back(self, tokens: int) -> None:
        self.inner.roll_back(tokens)


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

    def roll_back(self, tokens: int) -> None:
        self.inner.roll_back(tokens)


def restarts(s: SimpleNamespace, anchors: list, begin: list[int], stand_ins: list, width: int,
             layers=None, entering: torch.Tensor | None = None, timers: dict | None = None,
             lengths: list[int] | None = None) -> torch.Tensor:
    """Several restarts of the first drafter in ONE pass. Restart n starts a block of masks at the
    token anchors[n], placed at position begin[n], with the context the drafter already has (the
    target's states of the text before this round) and stand_ins[n]: (states [1, count, D], their
    positions) or None. The states are the target's (as the drafter is given them in any round),
    or, as (states, positions, True), vectors in the form the drafter reads them after its own
    projection (a predictor's). A restart sees the context, its own stand-ins and itself. Returns
    the tokens it writes after its anchor, one row per restart ([restarts, width - 1]), and leaves
    the drafter's cache as it was.

    layers: the pass runs only these of the drafter's layers (their numbers; None: all). A pass costs
    by the layers it runs, hardly by the restarts it holds, so this is what makes a second run cheap.
    entering: [restarts, width, D], the states the blocks enter the first of those layers with, in
    place of the embeddings of [anchor, masks]: states the FIRST run already had at those places
    after the layers left out here, so that the second run goes on from the first one.
    lengths: restart n has only lengths[n] places (the first run has no states past its block's end);
    the places after them are seen by no one, and what is written there means nothing.
    timers: seconds are added to its "prepare", "pass" and "head" (the clock waits for the GPU)."""
    drafter, device, count = s.first_drafter, s.block_ids.device, len(begin)
    if layers is not None:                                            # (a trained copy of a layer counts as that layer)
        layers = list(layers)
        ran = [layer if isinstance(layer, int) else layer.self_attn.layer_idx for layer in layers]
        if len(set(ran)) != len(ran):
            raise ValueError("a layer can run only once in a pass")
    clock = _cuda_time() if timers is not None else None
    if entering is None:
        block = torch.full((count, width), drafter.mask_token_id, dtype=s.block_ids.dtype, device=device)
        block[:, 0] = torch.as_tensor(anchors, dtype=block.dtype, device=device)
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        noise = _raw_input_embeddings(s.target, block.reshape(1, -1), scale)
    else:
        noise = entering.reshape(1, count * width, -1)
    slots = torch.arange(width, device=device)
    seen = s.draft_cache.get_seq_length()
    given = [entry for entry in stand_ins if entry is not None]
    extra = sum(entry[0].shape[1] for entry in given)
    read = [entry[0] if len(entry) > 2 and entry[2] else drafter.hidden_norm(drafter.fc(entry[0].to(s.context_features.dtype)))
            for entry in given]
    context = torch.cat(read, dim=1) if read else None
    positions = torch.cat([entry[1].to(device) for entry in given] + [start + slots for start in begin])[None]
    sees = torch.zeros((count * width, seen + extra + count * width), dtype=torch.bool, device=device)
    sees[:, :seen] = True
    at = seen
    for number, entry in enumerate(stand_ins):
        rows = slice(number * width, (number + 1) * width)
        own = width if lengths is None else lengths[number]
        sees[rows, seen + extra + number * width : seen + extra + number * width + own] = True
        if entry is not None:
            sees[rows, at : at + entry[0].shape[1]] = True
            at += entry[0].shape[1]
    if timers is not None:
        now = _cuda_time()
        timers["prepare"] = timers.get("prepare", 0.0) + now - clock
        clock = now
    hidden = drafter(
        target_hidden=s.context_features[:, :0], context_states=context, noise_embedding=noise, position_ids=positions,
        attention_mask=sees[None, None], past_key_values=s.draft_cache, use_cache=True, layers=layers,
    )[:, -count * width :, :]
    if layers is None:
        s.draft_cache.crop(-(extra + count * width))
    else:                                                             # only the layers that ran made their entries longer
        for number in ran:
            s.draft_cache.layers[number].crop(-(extra + count * width))
    if timers is not None:
        now = _cuda_time()
        timers["pass"] = timers.get("pass", 0.0) + now - clock
        clock = now
    logits = drafter.compute_logits(hidden.reshape(count, width, -1)[:, 1:, :], _output_head(s.target))
    tokens = logits.argmax(dim=-1)
    if timers is not None:
        timers["head"] = timers.get("head", 0.0) + _cuda_time() - clock
    return tokens


def _gone_on_from(s: SimpleNamespace, second, first_run: torch.Tensor | None, width: int) -> torch.Tensor:
    """What restart layers go on from: the first run's states of the block after the last layer they leave out
    ([width, size]), or, for layers that stand in for the whole drafter, the block as it entered the first run."""
    if second.first > 0:
        return first_run[0, -width:]
    drafter = s.first_drafter
    block = torch.full((1, width), drafter.mask_token_id, dtype=s.block_ids.dtype, device=s.block_ids.device)
    block[0, 0] = s.block_ids[0, 0]
    return _raw_input_embeddings(s.target, block, float(_draft_value(drafter.config, "input_embedding_scale", 1.0)))[0]


def _restart_pass(s: SimpleNamespace, tokens: list[int], begin: list[int], context: torch.Tensor | None, counts: list[int],
                  context_places: list[int], width: int, layers=None, entering: torch.Tensor | None = None,
                  lengths: list[int] | None = None, shortlist=None, also: torch.Tensor | None = None,
                  timers: dict | None = None, top: int | None = None, shared: list[list[int]] | None = None,
                  wanted: tuple[list[int], int] | None = None, states_only: bool = False):
    """The pass of restarts(), prepared without small steps on the GPU: the tokens, the places and who
    sees whom are made as lists and sent over once each. tokens, begin: the restarts' first tokens and
    their positions. context: [states, D], the stand-ins of all restarts in the form the drafter reads
    states, one restart after the other; counts[n]: how many of them are restart n's; context_places:
    their positions. layers, entering, lengths: as in restarts(). shortlist: the best token is looked
    for among a short list of tokens and the tokens `also` (dflash/shortlist.py), not in the whole
    vocabulary. timers: seconds are added to its "prepare", "pass" and "head" (the clock waits for the
    GPU). Returns [restarts, width - 1].
    shared: the stand-ins are given once and shared: restart n sees the rows shared[n] of `context`
    (counts is not read then). top: in place of the best token, the `top` best tokens of every place
    with their log-probabilities are returned: ([restarts, width - 1, top], the same shape).
    wanted: (the first place wanted of each restart, how many places): the output head is asked for
    these places only, and [restarts, places] is returned (places past the block's end: its last)."""
    drafter, device, count = s.first_drafter, s.block_ids.device, len(begin)
    clock = _cuda_time() if timers is not None else None
    ran = None
    if layers is not None:                                            # (a trained copy of a layer counts as that layer)
        ran = [layer if isinstance(layer, int) else layer.self_attn.layer_idx for layer in layers]
    if entering is None:
        masks = [drafter.mask_token_id] * (width - 1)
        block = torch.tensor([[token for first in tokens for token in [first] + masks]], dtype=s.block_ids.dtype, device=device)
        noise = _raw_input_embeddings(s.target, block, float(_draft_value(drafter.config, "input_embedding_scale", 1.0)))
    else:
        noise = entering.reshape(1, count * width, -1)
    seen, extra = s.draft_cache.get_seq_length(), len(context_places)
    positions = torch.tensor([list(context_places) + [start + slot for start in begin for slot in range(width)]], device=device)
    new = numpy.zeros((count * width, extra + count * width), dtype=bool)        # who sees which of the new entries
    at = 0
    for number in range(count):
        rows = slice(number * width, (number + 1) * width)
        own = width if lengths is None else lengths[number]
        new[rows, extra + number * width : extra + number * width + own] = True
        if shared is not None:
            new[rows, shared[number]] = True
            continue
        new[rows, at : at + counts[number]] = True
        at += counts[number]
    sees = torch.cat((torch.ones((count * width, seen), dtype=torch.bool, device=device), torch.from_numpy(new).to(device)), dim=1)
    if timers is not None:
        now = _cuda_time()
        timers["prepare"] = timers.get("prepare", 0.0) + now - clock
        clock = now
    hidden = drafter(
        target_hidden=s.context_features[:, :0], context_states=None if context is None else context[None],
        noise_embedding=noise, position_ids=positions, attention_mask=sees[None, None], past_key_values=s.draft_cache,
        use_cache=True, layers=layers,
    )[:, -count * width :, :]
    if ran is None:
        s.draft_cache.crop(-(extra + count * width))
    else:                                                             # only the layers that ran made their entries longer
        for number in ran:
            s.draft_cache.layers[number].crop(-(extra + count * width))
    if timers is not None:
        now = _cuda_time()
        timers["pass"] = timers.get("pass", 0.0) + now - clock
        clock = now
    if wanted is None:
        states = hidden.reshape(count, width, -1)[:, 1:, :]
    else:
        index = torch.tensor([[min(first + step, width - 1) for step in range(wanted[1])] for first in wanted[0]], device=device)
        states = hidden.reshape(count, width, -1)[torch.arange(count, device=device)[:, None], index]
    if states_only:                                                   # the caller scores a few tokens itself (SecondTree)
        if timers is not None:
            timers["head"] = timers.get("head", 0.0) + _cuda_time() - clock
        return states
    if top is not None and shortlist is not None:
        if _draft_value(drafter.config, "final_logit_softcapping") is not None:
            raise ValueError("a drafter that caps its scores needs the whole output head for probabilities")
        written = shortlist.best_few(states, top, also, float(_draft_value(drafter.config, "output_multiplier", 1.0)))
    elif top is not None:
        best = drafter.compute_logits(states, _output_head(s.target)).float().log_softmax(dim=-1).topk(top, dim=-1)
        written = (best.indices, best.values)
    elif shortlist is None:
        written = drafter.compute_logits(states, _output_head(s.target)).argmax(dim=-1)
    else:
        written = shortlist.best(states, also)
    if timers is not None:
        timers["head"] = timers.get("head", 0.0) + _cuda_time() - clock
    return written


class Restarts:
    """A second run that RESTARTS the drafter after each likely break.

    With the right token at the break, what the first draft has after it is rarely right, and a
    second pass that rewrites it inside the block does little better: the drafter writes well only
    where it has the target's hidden states of the text before (docs/WHY_A_SECOND_DRAFT_HELPS.md).
    So this second run does what the next round does. For each of the k most likely breaks (after
    i accepted draft tokens) it starts a NEW block at the drafter's second choice for token i + 1
    and writes a whole block after it, all restarts in one drafter pass. The target checks the
    first draft and, for each restart, [the first i draft tokens, the second choice, the new tokens].

    At the time of the second run the target has not read this round's tokens, so the restart after
    i tokens lacks the target's states of i + 1 tokens (the block's first token and the i draft
    tokens). stand_in says what the drafter gets in their place:
        "none"   nothing (the new block keeps its true positions)
        "last"   the last state it does have, repeated
        "real"   the target's real states: only possible in the replay of a saved answer, where it
                 is the limit of any stand-in (the harness sets .features to the answer's states)
        "predicted"  what a small trained network writes for them (dflash/stand_in.py; .predictor)
    swap: also return the swap for every break (the second choice followed by the first draft's old
    tokens): it costs no drafter pass, and where a stand-in is poor it is the better of the two.
    tail: the target checks only the second choice and this many tokens after it.
    choices: how many of the drafter's next choices at each break get a restart of their own (1: the
    second choice; 2: the second and the third; ...). All of them still share the one drafter pass,
    whose cost hardly depends on how many blocks it holds, so more choices raise what the pass earns.

    kind: where a restart starts. "choice": at the token after the break, from the drafter's next
    choice there (wasted when that choice is wrong). "earlier": one token earlier, at the last token
    before the break, which the first draft is believed to have right; the new block then writes the
    token at the break again, this time with the states of the tokens before its first token, so no
    guess is needed (and one state fewer is missing). A break right after the round's first token has
    no such restart: it would be the first draft again. "both": the two kinds in the same pass.

    Returned: rows of [first token, ...] that may be longer than the block (-1 after their end): the
    restarts from the second choices in the order of .kept, then those from the third choices, ...
    (with `best`: in the order they were picked), then the earlier ones, then the swaps of the choices."""

    def __init__(self, k: int = 4, stand_in: str = "none", swap: bool = False, tail: int | None = None,
                 choices: int = 1, kind: str = "choice", gate: float | None = None, block: int | None = None,
                 best: int | None = None, stale: bool = False):
        if stand_in not in ("none", "last", "real", "predicted"):
            raise ValueError("stand_in must be 'none', 'last', 'real' or 'predicted'")
        if choices < 1:
            raise ValueError("choices must be at least 1")
        if kind not in ("choice", "earlier", "both"):
            raise ValueError("kind must be 'choice', 'earlier' or 'both'")
        self.k, self.stand_in, self.swap, self.alternative_tail, self.choices = k, stand_in, swap, tail, choices
        self.kind = kind
        # gate: the second run is made only in rounds where the first draft is expected to get at most
        # this many tokens accepted (from the drafter's own confidences: the sum over places of the
        # product of the confidences up to there). Where the draft will likely hold, the pass is saved.
        self.gate, self.rounds_seen, self.rounds_run = gate, 0, 0
        # block: how many tokens a restart's new block has (its first token and the masks after it).
        # None: a whole block. The target checks only `tail` tokens after a restart's first new token,
        # so a block of tail + 2 tokens is all that is used, and the second run then reads and writes a
        # third of the tokens (the drafter was trained on whole blocks: what a short block costs in
        # accepted tokens is measured, not assumed).
        # best: the restarts from the drafter's next choices are picked best first, this many in all,
        # over every place and every one of the next `choices` choices, by the probability that the
        # text goes that way: the product of the confidences of the draft tokens before the place and
        # the choice's own probability (DDTree's score of that node). None: every choice at the k
        # likely breaks. The restarts one token earlier stay at the k likely breaks.
        if block is not None and block < 2:
            raise ValueError("block must be at least 2")
        if best is not None and best < 1:
            raise ValueError("best must be at least 1")
        self.block, self.best = block, best
        # stale: where the last check already read this round's token at its place (on the part of its
        # first path that the target did not accept), the target's state from that check is used: a real
        # state of the target for that token, computed with a context that differs a few tokens back
        # (the rejected ones). It takes the place of the predicted state there, or with stand_in="none"
        # it is the only state given (nothing is trained then). Real decoding only: a replay makes no
        # checks. counts: how often such a state was there, for all missing states and for the one
        # right before a restart's first token, which is the one a restart needs most.
        self.stale = stale
        self.counts = {"restarts": 0, "states_missing": 0, "states_from_the_last_check": 0,
                       "restarts_with_the_nearest_state_from_the_last_check": 0}
        self.features = None        # the answer's states, for stand_in="real"
        self.predictor = None       # a StatePredictor, for stand_in="predicted"
        self.kept = []              # the breaks of the last round, most likely first
        # second: a tiny second drafter (dflash/restart_layers.py: RestartLayers). The restarts then run
        # through its layers only, which stand in for the drafter's last ones, and go on from the first
        # run: a restart's places enter with the states the first run had there before those layers. A
        # pass costs by the layers it runs, so this is the cheap second run. The first run has no states
        # past its block's end, so a restart writes at most up to there.
        self.second = None
        self._hook, self._first_run = None, None
        # common: the tokens most common in answers (their numbers, counted ahead of time). With them the
        # restarts' tokens are looked for in a short list (these, the tokens of the text so far and the first
        # run's choices), not in the whole vocabulary: the output head over all of it takes over a
        # millisecond (dflash/shortlist.py).
        self.common, self._shortlist = None, None
        # timers: a dict to which the seconds of the second run's parts are added ("decide", "states",
        # "prepare", "pass", "head", "rows"; the clock waits for the GPU at each boundary, which itself
        # costs a little: for measuring, not for use).
        self.timers = None

    def close(self) -> None:
        """Stops watching the drafter's first run (only a second run through a second drafter watches it)."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def start(self, first_drafter: nn.Module) -> None:
        self.kept = []
        self.close()
        if self.second is not None and self.second.first > 0:
            # What the drafter's layer before the second drafter's gave in its last pass: at the time of
            # the second run that pass is the round's first draft.
            self._hook = first_drafter.layers[self.second.first - 1].register_forward_hook(
                lambda layer, came, output: setattr(self, "_first_run", output))

    def _from_the_last_check(self, s: SimpleNamespace, count: int):
        """Which of the round's first `count` tokens the last check read at the same place (their
        numbers in the block), and the target's states of them from that check as the drafter reads
        them ([hits, size]); ([], None) without `stale` or when there are none."""
        left = getattr(s, "leftover", None)
        if not self.stale or left is None:
            return [], None
        shift = s.start - left["start"]                              # the block's first token in the leftover
        mine = s.block_ids[0, :count].tolist()
        hits = [at for at in range(count) if 0 <= shift + at < len(left["tokens"]) and left["tokens"][shift + at] == mine[at]]
        if not hits:
            return [], None
        from .stand_in import project

        return hits, project(s.first_drafter, left["features"][None, [shift + at for at in hits]])[0]

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        # A pass of the drafter is a few milliseconds, and so were the hundreds of small steps on the GPU
        # that this method once took to pick the restarts, lay them out and put the rows together. So:
        # what is decided here is decided from a few lists read from the GPU once, and what the pass needs
        # (tokens, places, who sees whom) and the rows are made as lists and sent over once each.
        s = round_state
        timers = self.timers
        clock = _cuda_time() if timers is not None else None
        width = s.block_ids.shape[1]
        first = s.draft_logits[0].float()
        likely = first.softmax(dim=-1)
        confidence = likely.amax(dim=-1)
        self.rounds_seen += 1
        if self.gate is not None and float(confidence.cumprod(dim=0).sum()) > self.gate:
            self.kept = []
            return None                                               # the first draft will likely hold: no second run
        self.rounds_run += 1
        scores = break_scores(confidence)[: width - 1]
        self.kept = scores.topk(min(self.k, scores.shape[0])).indices.tolist()
        if self.stand_in == "real":
            if self.features is None:
                raise ValueError("stand_in='real' needs the answer's states (replay only)")
            # In replay the last block of an answer that ended by itself reaches past the answer's end:
            # breaks out there have no states, and nothing after them can be accepted anyway.
            self.kept = [broke for broke in self.kept if s.start + broke + 1 <= self.features.shape[1]]
            if not self.kept:
                return None
        choices = min(self.choices, first.shape[-1] - 1)
        top = likely.topk(1 + choices, dim=-1)
        ranked = top.indices                                          # [place, the first choice and the next ones]
        device = s.block_ids.device
        starts = []                                                   # (the break, which choice; 0: one token earlier)
        if self.kind != "earlier" and self.best is None:
            starts += [(broke, choice) for choice in range(1, 1 + choices) for broke in self.kept]
        elif self.kind != "earlier":
            holds = torch.cat((confidence.new_ones(1), confidence.cumprod(dim=0)[:-1]))[: width - 1]
            worth = holds[:, None] * top.values[: width - 1, 1:]      # [place, choice]: the text goes that way
            if self.stand_in == "real":                               # (no states past the answer's end)
                worth[max(self.features.shape[1] - s.start, 0) :] = -1.0
            best = worth.flatten().topk(min(self.best, worth.numel()))
            for index, value in zip(best.indices.tolist(), best.values.tolist()):
                if value >= 0:
                    starts.append((index // choices, 1 + index % choices))
        chosen = len(starts)
        if self.kind != "choice":
            starts += [(broke, 0) for broke in self.kept if broke >= 1]
        if not starts:
            return None
        ranked_tokens, block = ranked.tolist(), s.block_ids[0].tolist()
        if timers is not None:
            now = _cuda_time()
            timers["decide"] = timers.get("decide", 0.0) + now - clock
            clock = now
        most = max(broke for broke, _ in starts) + 1                  # the round's tokens that some restart lacks the states of
        hits, read = self._from_the_last_check(s, most)
        # The stand-ins of all restarts come from one table of vectors, in the form the drafter reads
        # states (pool); a restart that lacks the states of the round's first `missing` tokens takes the
        # rows have(missing), which stand at the places of the block's tokens at(missing).
        pool, told = None, False
        have = at = lambda missing: []
        if self.stand_in == "predicted":
            from .stand_in import predicted_states

            pool = predicted_states(self.predictor, s, most)[0]
            if hits:
                pool = pool.clone()
                pool[hits] = read.to(pool.dtype)
            have = at = lambda missing: list(range(missing))
            told = bool(self.predictor.sizes.get("coming"))
        elif self.stale and self.stand_in == "none":                  # only the states the last check left
            pool = read
            have = lambda missing: [number for number, place in enumerate(hits) if place < missing]
            at = lambda missing: [place for place in hits if place < missing]
        elif self.stand_in == "real":
            from .stand_in import project

            pool = project(s.first_drafter, self.features[:, s.start : s.start + most])[0]
            have = at = lambda missing: list(range(missing))
        elif self.stand_in == "last" and s.context_features.shape[1]:
            from .stand_in import project

            pool = project(s.first_drafter, s.context_features[:, -1:])[0]
            have, at = (lambda missing: [0] * missing), (lambda missing: list(range(missing)))
        first_tokens, begin, rows_taken, places_taken, counts, more = [], [], [], [], [], []
        for broke, choice in starts:
            if choice:
                first_tokens.append(ranked_tokens[broke][choice])
                begin.append(s.start + broke + 1)
                missing = broke + 1
            else:                                                     # from the last token before the break: one state fewer
                first_tokens.append(block[broke])
                begin.append(s.start + broke)
                missing = broke
            mine, where = have(missing), at(missing)
            if choice and told:      # the predictor is told which token this restart starts from: vectors of its own
                own = predicted_states(self.predictor, s, missing, coming=first_tokens[-1])[0]
                for number, place in enumerate(hits):
                    if place < missing:
                        own[place] = read[number].to(own.dtype)
                mine = list(range(pool.shape[0] + sum(extra.shape[0] for extra in more), pool.shape[0]
                                  + sum(extra.shape[0] for extra in more) + missing))
                more.append(own)
            rows_taken += mine
            places_taken += [s.start + place for place in where]
            counts.append(len(mine))
            nearest = missing - 1                                     # the token right before the restart's first token
            self.counts["restarts"] += 1
            self.counts["states_missing"] += nearest + 1
            self.counts["states_from_the_last_check"] += sum(1 for place in hits if place <= nearest)
            self.counts["restarts_with_the_nearest_state_from_the_last_check"] += int(nearest in hits)
        context = None
        if rows_taken:
            if more:
                pool = torch.cat([pool] + more, dim=0)
            context = pool[torch.tensor(rows_taken, device=device)]
        reach = width if self.block is None else min(self.block, width)
        lengths, entering = None, None
        wanted, pass_width, pass_begin = None, reach, begin
        if self.second is not None and getattr(self.second, "reveal", False):
            # Through layers that read the tokens: every restart's block is the round's block from its first
            # token on, as far as the restart writes, with the tokens known at the restart written in at their
            # places (the draft tokens before it, and its first token).
            first_place = [broke + 1 if choice else broke for broke, choice in starts]
            lengths = [min(place + reach, width) for place in first_place]
            pass_width, pass_begin = max(lengths), [s.start] * len(starts)
            wanted = ([place + 1 for place in first_place], reach - 1)
            mask_token = s.first_drafter.mask_token_id
            written_in = [block[: place] + [token] + [mask_token] * (pass_width - place - 1)
                          for place, token in zip(first_place, first_tokens)]
            known = [[0 < slot <= place for slot in range(pass_width)] for place in first_place]
            scale = float(_draft_value(s.first_drafter.config, "input_embedding_scale", 1.0))
            given = torch.tensor(written_in + [[mask_token] * pass_width], dtype=s.block_ids.dtype, device=device)
            embedded = _raw_input_embeddings(s.target, given, scale)
            entering = self.second.enter_blocks(_gone_on_from(s, self.second, self._first_run, width)[:pass_width], embedded[:-1],
                                                embedded[-1, 0], torch.tensor(known, device=device))
            if self.second.states_in_block and context is not None:
                # The stand-ins go into the block, at the places of their tokens, not beside it.
                owner = [number for number, many in enumerate(counts) for _ in range(many)]
                entering = self.second.with_states(entering, context, (torch.tensor(owner, device=device),
                                                                       torch.tensor([place - s.start for place in places_taken], device=device)))
                context, counts, places_taken = None, [0] * len(starts), []
            lengths_written = [length - place for length, place in zip(lengths, first_place)]     # as below: its places from its first token on
        elif self.second is not None:
            # Through the tiny second drafter, going on from the first run. A restart's first token is at
            # the block's place broke + 1 (from a choice) or broke (one token earlier); the first run has
            # no states past the block's end, so nothing is written past it.
            first_place = [broke + 1 if choice else broke for broke, choice in starts]
            lengths = [min(width - place, reach) for place in first_place]
            places = torch.tensor([[min(place + slot, width - 1) for slot in range(reach)] for place in first_place], device=device)
            scale = float(_draft_value(s.first_drafter.config, "input_embedding_scale", 1.0))
            given = torch.tensor([first_tokens + [s.first_drafter.mask_token_id]], dtype=s.block_ids.dtype, device=device)
            embedded = _raw_input_embeddings(s.target, given, scale)[0]
            entering = self.second.enter(self._first_run[0, -width:][places], embedded[:-1], embedded[-1])
            lengths_written = lengths
        also = None
        if self.common is not None:
            if self._shortlist is None or self._shortlist.head is not _output_head(s.target):
                from .shortlist import Shortlist

                self._shortlist = Shortlist(_output_head(s.target), self.common)
            also = torch.cat((s.accepted_ids.flatten(), ranked.flatten()))
        if timers is not None:
            now = _cuda_time()
            timers["states"] = timers.get("states", 0.0) + now - clock
        new = _restart_pass(s, first_tokens, pass_begin, context, counts, places_taken, pass_width,
                            layers=None if self.second is None else list(self.second.layers), entering=entering,
                            lengths=lengths, shortlist=self._shortlist if self.common is not None else None, also=also,
                            timers=timers, wanted=wanted).tolist()
        clock = _cuda_time() if timers is not None else None
        rows = [[-1] * (2 * width) for _ in range(len(starts) + (chosen if self.swap else 0))]
        for number, (broke, choice) in enumerate(starts):
            row, written = rows[number], new[number]
            if lengths is not None:                                   # nothing is written past the first block's end
                written = written[: lengths_written[number] - 1] + [-1] * (reach - lengths_written[number])
            row[: broke + 1] = block[: broke + 1]
            if not choice:
                row[broke + 1 : broke + reach] = written
                continue
            row[broke + 1] = ranked_tokens[broke][choice]
            row[broke + 2 : broke + 1 + reach] = written
            if self.swap:
                swapped = rows[len(starts) + number]
                swapped[:width] = block
                swapped[broke + 1] = ranked_tokens[broke][choice]
        rows = torch.tensor(rows, dtype=s.block_ids.dtype, device=device)
        if timers is not None:
            timers["rows"] = timers.get("rows", 0.0) + _cuda_time() - clock
        return rows


class Reanchor:
    """New blocks of the same drafter started from what the LAST check left behind (ReAnchor, Lee et
    al., arXiv 2610.26531; nothing is trained).

    In its last check the target also read the draft tokens after the first one it rejected. Its
    states there are states of text that was not written, and so is the token it chose after each of
    them. But they belong together: a block started at that token, placed right after the rejected
    token it was chosen after, with the target's states of the rejected tokens before it, is what the
    drafter would write next if the text had gone that way. Text often comes back to the rejected
    draft a token or two later; where this round's first draft holds a block's first token, the
    block's tokens are attached after it (at the earliest such place), and the target checks them in
    the same pass as the first draft.

    k: at most this many blocks, the ones started nearest to this round's first token (None: all).
    tail: the target checks only this many of a block's tokens. block: the tokens of a new block
    (first token and masks; None: a whole block). fixed: attach a block only at the place it was
    started at (ReAnchor's ablation; False: at the earliest place the first draft has its token).
    corrections: see __init__ (ours).
    ahead: the blocks are written in the first draft's own pass, as ReAnchor does (they need nothing
    of the first draft); False: in a pass of their own after it, which costs a drafter pass more.
    Returned: rows [first token, ...] with -1 after their end, or None. Real decoding only: a replay
    makes no checks."""

    swap = False

    def __init__(self, k: int | None = None, tail: int | None = None, block: int | None = None, fixed: bool = False,
                 ahead: bool = True, corrections: bool = False, whole: bool = False):
        self.k, self.alternative_tail, self.block, self.fixed = k, tail, block, fixed
        # whole: on top of DDTree, the target also reads the whole first draft (TreeWithRestarts adds it
        # as a row). DDTree leaves out the first draft's late tokens, which are rarely accepted; but
        # what the target makes of them is what the next round's blocks are started from.
        self.wants_first_draft = whole
        self.ahead = ahead          # the blocks are written in the first draft's pass (real decoding does that)
        # corrections (ours, not in ReAnchor): a block's first token is what the target chose after a
        # draft token of the last round. Where this round's first draft has that same token at that
        # same place again, the target has in effect been asked already what comes next there (in a
        # context that differs a few tokens back). If the first draft goes on with another token, the
        # block is attached at its own place WITH its first token in place of the draft's: a break
        # foretold, the target's own token for it, and a block written after that token.
        self.corrections = corrections
        self.kept = []
        self.counts = {"rounds": 0, "rounds_with_blocks": 0, "blocks": 0, "blocks_attached": 0, "blocks_attached_elsewhere": 0,
                       "corrections": 0, "same_token_before_and_the_same_next": 0}

    def start(self, drafter: nn.Module) -> None:
        self.kept = []

    def blocks(self, s: SimpleNamespace):
        """The blocks the last check allows: (first tokens, their places, [(states, places)]),
        nearest first, or None."""
        left = getattr(s, "leftover", None)
        if left is None:
            return None
        device = s.block_ids.device
        skip = max(s.start - left["start"], 0)                      # leftover tokens whose places are text by now
        firsts, places, stand_ins = [], [], []
        for count in range(skip + 1, len(left["tokens"]) + 1):      # after the first `count` leftover tokens
            if self.k is not None and len(firsts) >= self.k:
                break
            firsts.append(left["said"][count - 1])
            places.append(left["start"] + count)
            stand_ins.append((left["features"][None, skip:count], torch.arange(left["start"] + skip, left["start"] + count, device=device)))
        return (firsts, places, stand_ins) if firsts else None

    @staticmethod
    def token_before(s: SimpleNamespace, place: int):
        """The token the last check read right before `place` (the one a block started there was chosen after)."""
        left = s.leftover
        return left["tokens"][place - 1 - left["start"]]

    def __call__(self, s: SimpleNamespace) -> torch.Tensor | None:
        self.counts["rounds"] += 1
        width = s.block_ids.shape[1]
        if getattr(s, "ahead", None) is not None:                   # written in the first draft's pass
            firsts, places, new = s.ahead
        else:
            found = self.blocks(s)
            if found is None:
                return None
            firsts, places, stand_ins = found
            new = restarts(s, firsts, places, stand_ins, width if self.block is None else min(self.block, width))
        first_draft = s.block_ids[0].tolist()
        rows = []
        for number, token in enumerate(firsts):
            nominal = places[number] - s.start
            again = (self.corrections and 2 <= nominal < width
                     and first_draft[nominal - 1] == self.token_before(s, places[number]))
            if again:                                               # the same token before it as in the last check
                self.counts["corrections"] += int(first_draft[nominal] != token)
                self.counts["same_token_before_and_the_same_next"] += int(first_draft[nominal] == token)
                self.counts["blocks_attached"] += 1
                rows.append(first_draft[:nominal] + [token] + new[number].tolist())
                continue
            if self.fixed:
                at = nominal if nominal < width and first_draft[nominal] == token else None
            else:
                at = next((index for index in range(width) if first_draft[index] == token), None)
            if at is None:
                continue
            self.counts["blocks_attached"] += 1
            self.counts["blocks_attached_elsewhere"] += int(at != nominal)
            rows.append(first_draft[: at + 1] + new[number].tolist())
        self.counts["rounds_with_blocks"] += 1
        self.counts["blocks"] += len(firsts)
        if not rows:
            return None
        out = torch.full((len(rows), 2 * width), -1, dtype=s.block_ids.dtype, device=s.block_ids.device)
        for number, row in enumerate(rows):
            out[number, : len(row)] = torch.tensor(row[: 2 * width], dtype=out.dtype, device=out.device)
        return out


class DraftTree:
    """DDTree (Ringel and Romano, arXiv 2604.12989): a tree of drafts from the first pass alone.

    The drafter gives one distribution per position of the block. A node of the tree is a choice of
    token for every position up to its own, and its probability is the product of those tokens'
    probabilities. The tree is the `budget` most probable nodes, found best first with a heap; it is
    the tree from which most tokens would be accepted if the drafter's distributions were right
    and independent of each other. The target checks all nodes in one pass and follows its own
    choices through the tree. The first draft is the path of first choices: it is in the tree only
    as far as its nodes are among the most probable, so the tree takes the place of the first draft
    (a budget of 15 is a block's worth of nodes, placed where they are most likely to be used).

    Returned: one row per leaf: [the block's first token, the tokens on the way to the leaf, -1 ...].
    nodes: how many nodes the last tree had (the target reads one row more, the block's first token).
    Scored in replay only so far."""

    replace = True

    def __init__(self, budget: int):
        if budget < 1:
            raise ValueError("budget must be at least 1")
        self.budget = budget
        self.nodes = 0

    def start(self, drafter: nn.Module) -> None:
        self.nodes = 0

    def paths(self, scores: list[list[float]]) -> list[tuple[int, ...]]:
        """The `budget` most probable nodes, each as the ranks of its tokens (0 = the first choice),
        from scores[position][rank], the log-probabilities in falling order. A node comes after its
        parent and after the sibling of the rank before it."""
        depth, width = len(scores), len(scores[0])
        heap, chosen = [(-scores[0][0], (0,))], []
        while heap and len(chosen) < self.budget:
            negative, ranks = heapq.heappop(heap)
            chosen.append(ranks)
            at, last = len(ranks), ranks[-1]
            if last + 1 < width:                                        # the next sibling
                heapq.heappush(heap, (negative + scores[at - 1][last] - scores[at - 1][last + 1], ranks[:-1] + (last + 1,)))
            if at < depth:                                              # the first child
                heapq.heappush(heap, (negative - scores[at][0], ranks + (0,)))
        return chosen

    def choose(self, s: SimpleNamespace, best, scores: list[list[float]]):
        """The tree's nodes, each as the ranks of its tokens, and the log-probability of each node.
        best: the first run's most probable tokens of every place (topk: values and indices), scores: its values as lists."""
        chosen = self.paths(scores)
        return chosen, {ranks: sum(scores[at][rank] for at, rank in enumerate(ranks)) for ranks in chosen}

    def __call__(self, s: SimpleNamespace) -> torch.Tensor:
        logits = s.draft_logits[0].float()
        best = logits.log_softmax(dim=-1).topk(min(self.budget, logits.shape[-1]), dim=-1)
        tokens = best.indices.tolist()
        chosen, _ = self.choose(s, best, best.values.tolist())
        self.nodes = len(chosen)
        parents = {ranks[:-1] for ranks in chosen}
        leaves = [ranks for ranks in chosen if ranks not in parents]
        # (the rows are made as lists and sent to the GPU once: a row at a time cost over a millisecond a round)
        width, first = s.block_ids.shape[1], int(s.block_ids[0, 0])
        rows = [[first] + [tokens[at][rank] for at, rank in enumerate(ranks)] + [-1] * (width - 1 - len(ranks)) for ranks in leaves]
        return torch.tensor(rows, dtype=s.block_ids.dtype, device=s.block_ids.device)


class CountTree(DraftTree):
    """DDTree with the scores corrected by counts of which token the target writes after which two
    tokens: the scores of DForkTree (Zheng et al., arXiv 2611.12356), written here from the paper.

    DDTree gives a node the product of the first run's probabilities of its tokens. The first run
    wrote every place without knowing which tokens a branch chose before it. Here the probability
    of a token c after the tokens (a, b) of its own branch is the first run's, times
    exp(weight x lift(a, b, c)), where lift says how much more often the target writes c after
    (a, b) than anywhere (dflash/follow.py). The weight is `on_path` while the branch is the path of
    first choices and `off_path` once it has left it (0.3 and 0.6 in the paper). Only the
    `candidates` most probable tokens of a place are scored; what the first run gives to all other
    tokens stays in the sum the scores are divided by, so a branch whose candidates all fit badly
    loses probability. The first place keeps the first run's probabilities.

    The tree is the `budget` most probable nodes under these scores. They depend on the branch, so
    the tree is grown a place at a time: the `budget` best nodes of each depth are kept, and the
    best `budget` of all kept nodes are the tree (a node never scores above its parent, so this is
    exact). Nothing is trained and no pass is added: one lookup in the counts for all places."""

    def __init__(self, budget: int, follow=None, candidates: int = 16, on_path: float = 0.3, off_path: float = 0.6,
                 fast: bool = False, graph: bool = False):
        super().__init__(budget)
        self.follow, self.candidates, self.on_path, self.off_path = follow, candidates, on_path, off_path
        self.table = None                              # the last round's lifts [places - 1, a, b, c], where they were computed
        # pairs: trained scores of pairs of neighbouring tokens (dflash/pair_scores.py), added to the counts' lifts
        # or used without counts (follow None). Set by nexus/redraft_long.py for the trees named ptree / cptree.
        self.pairs = None
        # fast: the search runs on the GPU in tensors of fixed shape (search_tensors) and the rows come back in one
        # copy; the search on the CPU in numpy costs a few milliseconds a round in small operations and copies.
        # graph: the whole of it (the scores' tables, the counts' lookup, the search, the rows) is recorded once as a
        # CUDA graph and replayed every round: the many small operations are then launched as one.
        self.fast, self.graph = fast or graph, graph
        self._graphed = None                           # (key of the shapes, the graph, the static inputs, the static rows)

    def _pair_tables(self, s: SimpleNamespace, tokens: torch.Tensor, values: torch.Tensor):
        """The pair scores of all candidates of all places after the first: [places - 1, before, here],
        [places - 1, two before, here], the factor on "another token" [places - 1], as numpy arrays."""
        pairs, target, drafter = self.pairs, s.target, s.first_drafter
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        first = s.block_ids[0, :1].to(tokens.device).expand(tokens.shape[1])[None]
        embedded = _raw_input_embeddings(target, torch.cat((first, tokens[:-1])), scale)       # of the block's first token and of every place's candidates
        states = self._final_states(s)[2 : tokens.shape[0] + 1]                                 # the first run's states at the block's places 2 ..
        sure_before = sure_two = None
        if pairs.sure_before is not None:
            # How the first run stood to every place's candidates (values: their log-probabilities, most probable
            # first): the log-probability, first choice or not, the first choice's log-probability.
            values = values.float()
            first = torch.zeros_like(values)
            first[:, 0] = 1.0
            sure = torch.stack((values, first, values[:, :1].expand_as(values)), dim=-1)            # [places, candidates, 3]
            given = torch.tensor([0.0, 1.0, 0.0], device=values.device).expand(1, values.shape[1], 3)   # the block's first token
            sure_before, sure_two = sure[:-1], torch.cat((given, sure[:-2]))
        pair, skip, rest = pairs.tables(states, embedded[1:], embedded[:-1], _output_head(target).weight[tokens[1:]], sure_before, sure_two)
        return (pair.double().cpu().numpy(), skip.double().cpu().numpy(), rest.double().exp().cpu().numpy(),
                float(pairs.on_path), float(pairs.off_path))

    def children(self, place: int, before: int, last: int, on_path: bool):
        """For tracing, after a round with keep_last set: the log-probabilities this tree's scores give the candidates
        of `place` (0: the block's first drafted place) after a node that ends in candidate `last` of the place before,
        with candidate `before` at the place before that (any, at place 1), on the path of first choices or off it.
        The same numbers the search of the round used."""
        likely, rest, lift, extra, on_weight, off_weight = self.seen["tables"]
        if place == 0:
            return likely[0]
        raised, others = likely[place], rest[place]
        if lift is not None:
            raised = raised + (on_weight if on_path else off_weight) * lift[place - 1][before, last]
        if extra is not None:
            raised = raised + extra[0][place - 1][last] + extra[1][place - 1][before]
            others = others * extra[2][place - 1]
        return raised - numpy.log(others + numpy.exp(raised).sum())

    @staticmethod
    def _final_states(s: SimpleNamespace) -> torch.Tensor:
        """The first run's final states of all places of the block ([width, size])."""
        states = getattr(s, "draft_states", None)
        if states is None:
            raise ValueError("the pair scores need the first run's states (the decoding loop keeps them as s.draft_states)")
        return states[0] if states.dim() == 3 else states

    def choose(self, s: SimpleNamespace, best, scores: list[list[float]]):
        if self.follow is None and self.pairs is None:
            raise ValueError("the tree with counts needs the counts (dflash/follow.py: FollowCounts)")
        many = min(self.candidates, best.indices.shape[1])
        tokens = best.indices[:, :many]                                # [places, candidates], most probable first
        places = tokens.shape[0]
        likely = numpy.asarray([row[:many] for row in scores], dtype=numpy.float64)
        rest = numpy.clip(1.0 - numpy.exp(likely).sum(axis=-1), 0.0, None)      # what the other tokens of a place have
        lift, extra, on_weight, off_weight = None, None, self.on_path, self.off_path
        if places > 1 and self.pairs is not None:
            *extra, on_weight, off_weight = self._pair_tables(s, tokens, best.values[:, :many])
            if getattr(self, "keep_last", False):      # for tests
                self.last = {"extra": extra, "likely": likely, "rest": rest, "tokens": tokens.tolist(), "start": s.start}
        if places > 1 and self.follow is not None:
            # One lookup for all places: the lift of every candidate c of a place after every candidate b of the
            # place before it and every candidate a of the place before that (for the second place: the block's
            # first token).
            first = s.block_ids[0, :1].to(tokens.device).expand(many)[None]
            a = torch.cat((first, tokens[:-2]))[:, :, None, None]
            lift = self.follow.lift_of_places(a, tokens[:-1][:, None, :, None], tokens[1:][:, None, None, :])
            self.table = lift                                          # (SecondTree scores with the same lookup)
            lift = lift.cpu().numpy().astype(numpy.float64)            # [places - 1, a, b, c]
        if getattr(self, "keep_last", False):          # for tracing (nexus/trace_trees.py): what this round's scores were made of
            self.seen = {"tokens": tokens.tolist(), "tables": (likely, rest, lift, extra, on_weight, off_weight)}
        # The kept nodes of a depth: their score, the candidate they end in, the candidate before it, whether
        # the branch is still the path of first choices, and which kept node of the depth before is their parent.
        order = numpy.argsort(-likely[0], kind="stable")[: self.budget]
        score, last = likely[0][order], order
        before, on_path, parent = numpy.zeros_like(last), last == 0, numpy.full(len(last), -1)
        depths, so_far = [(score, last, parent)], score
        for place in range(1, places):
            # A node never scores above its parent: once `budget` kept nodes score above the best node of this
            # depth, nothing deeper can come into the tree.
            if len(so_far) >= self.budget and score.max() < numpy.partition(so_far, -self.budget)[-self.budget]:
                break
            raised, others = likely[place][None, :], rest[place]
            if lift is not None:
                raised = raised + numpy.where(on_path, on_weight, off_weight)[:, None] * lift[place - 1][before, last]
            if extra is not None:
                raised = raised + extra[0][place - 1][last] + extra[1][place - 1][before]
                others = others * extra[2][place - 1]
            given = raised - numpy.log(others + numpy.exp(raised).sum(axis=-1))[:, None]
            total = (score[:, None] + given).ravel()
            keep = numpy.argsort(-total, kind="stable")[: self.budget]
            parent, chosen_candidate = keep // many, keep % many
            score, before, last = total[keep], last[parent], chosen_candidate
            on_path = on_path[parent] & (chosen_candidate == 0)
            depths.append((score, last, parent))
            so_far = numpy.concatenate((so_far, score))
        # The best `budget` of all kept nodes; of equal scores the shallower first, so a parent comes before its child.
        every = [(-float(value), depth, at) for depth, (scores_here, _, _) in enumerate(depths) for at, value in enumerate(scores_here)]
        every.sort()
        ranks_of = {}

        def ranks(depth, at):
            if (depth, at) not in ranks_of:
                _, last_here, parent_here = depths[depth]
                above = ranks(depth - 1, int(parent_here[at])) if depth else ()
                ranks_of[(depth, at)] = above + (int(last_here[at]),)
            return ranks_of[(depth, at)]

        chosen, worth = [], {}
        for negative, depth, at in every[: self.budget]:
            node = ranks(depth, at)
            chosen.append(node)
            worth[node] = -negative
        return chosen, worth

    def _constant(self, name: str, make):
        """A constant tensor made once per device (a recorded graph may hold no copy from the CPU)."""
        store = self.__dict__.setdefault("_constants", {})
        if name not in store:
            store[name] = make()
        return store[name]

    def _pair_tables_tensors(self, s: SimpleNamespace, tokens: torch.Tensor, values: torch.Tensor):
        """As _pair_tables, but the tables stay on the device, in float32."""
        pairs, target, drafter = self.pairs, s.target, s.first_drafter
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        first = s.block_ids[0, :1].to(tokens.device).expand(tokens.shape[1])[None]
        embedded = _raw_input_embeddings(target, torch.cat((first, tokens[:-1])), scale)
        states = self._final_states(s)[2 : tokens.shape[0] + 1]
        sure_before = sure_two = None
        if pairs.sure_before is not None:
            values = values.float()
            first_choice = torch.zeros_like(values)
            first_choice[:, 0] = 1.0
            sure = torch.stack((values, first_choice, values[:, :1].expand_as(values)), dim=-1)
            given = self._constant(f"given:{values.device}", lambda: torch.tensor([0.0, 1.0, 0.0], device=values.device))
            given = given.expand(1, values.shape[1], 3)
            sure_before, sure_two = sure[:-1], torch.cat((given, sure[:-2]))
        pair, skip, rest = pairs.tables(states, embedded[1:], embedded[:-1], _output_head(target).weight[tokens[1:]], sure_before, sure_two)
        return pair.float(), skip.float(), rest.float().exp(), pairs.on_path.float(), pairs.off_path.float()

    def search_tensors(self, likely: torch.Tensor, rest: torch.Tensor, lift, extra, on_weight, off_weight):
        """The search of `choose` in tensors of fixed shape, on likely's device: likely [places, many] (the first run's
        log-probabilities of every place's candidates, most probable first), rest [places] (what the other tokens of a
        place have), lift [places - 1, a, b, c] or None, extra (pair [places - 1, b, c], skip [places - 1, a, c], the
        factor on the rest [places - 1]) or None, on_weight and off_weight numbers or 0-d tensors. Returns the ranks
        of the `budget` best nodes [budget, places] (-1 past a node's depth, the shallower of equal scores first) and
        their scores [budget]. Every depth is searched (nothing to stop early for): the result is the same."""
        places, many = likely.shape
        budget, device, dtype = self.budget, likely.device, likely.dtype
        kept = min(budget, many)
        minus = torch.finfo(dtype).min
        on_weight, off_weight = (
            w.to(device=device, dtype=dtype) if isinstance(w, torch.Tensor)
            else self._constant(f"weight:{float(w)}:{device}:{dtype}", lambda w=w: torch.tensor(float(w), device=device, dtype=dtype))
            for w in (on_weight, off_weight))
        # The kept nodes of a depth, padded to `budget` (score minus, no rank): score, the candidate they end in, the
        # one before it, whether the branch is still the path of first choices, and the ranks of their tokens so far.
        score = torch.full((budget,), minus, device=device, dtype=dtype)
        score[:kept] = likely[0, :kept]
        last = torch.zeros(budget, dtype=torch.long, device=device)
        last[:kept] = torch.arange(kept, device=device)
        before = torch.zeros_like(last)
        on_path = last == 0
        ranks = torch.full((budget, places), -1, dtype=torch.long, device=device)
        ranks[:kept, 0] = last[:kept]
        all_scores, all_ranks = [score], [ranks]
        for place in range(1, places):
            raised = likely[place][None, :].expand(budget, many)
            others = rest[place]
            if lift is not None:
                raised = raised + torch.where(on_path, on_weight, off_weight)[:, None] * lift[place - 1][before, last]
            if extra is not None:
                raised = raised + extra[0][place - 1][last] + extra[1][place - 1][before]
                others = others * extra[2][place - 1]
            given = raised - torch.log(others + raised.exp().sum(dim=-1))[:, None]
            total = (score[:, None] + given).reshape(-1)
            keep = total.topk(budget).indices
            parent, candidate = keep // many, keep % many
            score, before, last = total[keep], last[parent], candidate
            on_path = on_path[parent] & (candidate == 0)
            ranks = ranks[parent].clone()
            ranks[:, place] = candidate
            all_scores.append(score)
            all_ranks.append(ranks)
        scores = torch.cat(all_scores)                                       # [places * budget], depth-major
        # The best `budget` of all kept nodes; of equal scores the shallower first (a parent before its child).
        order = (-scores).argsort(stable=True)[:budget]
        return torch.cat(all_ranks)[order], scores[order]

    def __call__(self, s: SimpleNamespace) -> torch.Tensor:
        if not self.fast or getattr(self, "keep_last", False):
            return super().__call__(s)
        if self.follow is None and self.pairs is None:
            raise ValueError("the tree with counts needs the counts (dflash/follow.py: FollowCounts)")
        if self.graph and s.draft_logits.is_cuda:
            rows = self._replayed(s)
        else:
            rows = self._rows_in_tensors(s)
        self.nodes = int((rows[:, 1] >= 0).sum())
        return rows

    def _replayed(self, s: SimpleNamespace) -> torch.Tensor:
        """The rows from the recorded graph, recorded first if the shapes are new (a block of another width, say)."""
        states = self._final_states(s)
        key = (tuple(s.draft_logits.shape), tuple(states.shape), tuple(s.block_ids.shape), s.draft_logits.dtype, states.dtype)
        if self._graphed is None or self._graphed[0] != key:
            static = SimpleNamespace(draft_logits=s.draft_logits.clone(), draft_states=states.clone(), block_ids=s.block_ids.clone(),
                                     target=s.target, first_drafter=s.first_drafter)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):                                   # warming up on a side stream, as the recording needs
                    self._rows_in_tensors(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                rows = self._rows_in_tensors(static)
            self._graphed = (key, graph, static, rows)
        _, graph, static, rows = self._graphed
        static.draft_logits.copy_(s.draft_logits)
        static.draft_states.copy_(states)
        static.block_ids.copy_(s.block_ids)
        graph.replay()
        return rows.clone()

    def _rows_in_tensors(self, s: SimpleNamespace) -> torch.Tensor:
        logits = s.draft_logits[0].float()
        many = min(self.candidates, self.budget, logits.shape[-1])
        best = logits.log_softmax(dim=-1).topk(many, dim=-1)
        tokens, likely = best.indices, best.values                         # [places, candidates]
        places = tokens.shape[0]
        rest = (1.0 - likely.exp().sum(dim=-1)).clamp(min=0.0)
        lift, extra, on_weight, off_weight = None, None, self.on_path, self.off_path
        if places > 1 and self.pairs is not None:
            *extra, on_weight, off_weight = self._pair_tables_tensors(s, tokens, likely)
        if places > 1 and self.follow is not None:
            first = s.block_ids[0, :1].to(tokens.device).expand(many)[None]
            a = torch.cat((first, tokens[:-2]))[:, :, None, None]
            lift = self.follow.lift_of_places(a, tokens[:-1][:, None, :, None], tokens[1:][:, None, None, :]).float()
            self.table = lift
        ranks, _ = self.search_tensors(likely, rest, lift, extra, on_weight, off_weight)
        # One row per node (a node's row is the start of its children's rows: the tree read from the rows is the same).
        width = s.block_ids.shape[1]
        chosen = tokens.gather(1, ranks.clamp(min=0).t()).t()                # [budget, places]
        rows = torch.full((ranks.shape[0], width), -1, dtype=s.block_ids.dtype, device=s.block_ids.device)
        rows[:, 0] = s.block_ids[0, 0]
        rows[:, 1 : places + 1] = torch.where(ranks >= 0, chosen, torch.full_like(chosen, -1)).to(rows.dtype)
        return rows


class BranchTree:
    """A tree of drafts in which the second run writes after EVERY node.

    DDTree builds its tree from the first run alone. The first run fills all places of the block at
    once, so what the tree holds after a token is what the first run guessed there without knowing
    that token. Seven in ten of DDTree's rounds end right after an alternative token, where this is
    worst (nexus/tree_ends.py). A restart writes the tokens after a token knowing it. One restart
    costs a pass of the drafter, but a pass costs by the layers it runs and hardly by the restarts it
    holds. So here the second run restarts at every node of a first tree at once:

      1. a first tree of `first` nodes from the first run (DDTree's own rule);
      2. ONE pass in which a new block starts at each of its nodes: the node's token, then `tail`
         places. A block sees the stand-ins of the target's states for the tokens on its node's path
         (stand_in: "predicted", "real" in replay, "none") and itself;
      3. the tree grows to `budget` nodes by what the pass wrote: after node n the `choices` best
         tokens of the first place, each with the log-probability of n's path plus its own, and after
         the best one the best tokens of the next places, as a chain. The most probable ones are
         added, as DDTree adds nodes. A token the first tree already has after n is not added again.

    The target reads `budget` tokens, as with DDTree(budget). second: the restarts run through a tiny
    second drafter that goes on from the first run (dflash/restart_layers.py); without one they run
    through the whole drafter, with blocks of `block` tokens (None: a whole block), which costs a
    whole pass and is the measure of what such a tree can hold. Rows as DraftTree returns them."""

    replace = True

    def __init__(self, budget: int, first: int, tail: int = 2, stand_in: str = "predicted", choices: int = 2,
                 block: int | None = None):
        if not 1 <= first <= budget:
            raise ValueError("the first tree has at least one node and at most the whole budget")
        if stand_in not in ("none", "real", "predicted"):
            raise ValueError("stand_in must be 'none', 'real' or 'predicted'")
        self.budget, self.first, self.tail, self.stand_in, self.choices, self.block = budget, first, tail, stand_in, choices, block
        self.tree = DraftTree(first)                   # the first tree; a CountTree(first, ...) may be put in its place
        self.features, self.predictor, self.second = None, None, None
        self.nodes, self._hook, self._first_run = 0, None, None
        self.common, self._shortlist = None, None      # the second run's tokens from a short list (see Restarts.common)
        self.counts = {"rounds": 0, "nodes_from_the_first_run": 0, "nodes_from_the_second_run": 0,
                       "second_run_tokens_the_first_tree_had": 0}

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def start(self, drafter: nn.Module) -> None:
        self.nodes = 0
        self.close()
        if self.second is not None and self.second.first > 0:
            self._hook = drafter.layers[self.second.first - 1].register_forward_hook(
                lambda layer, came, output: setattr(self, "_first_run", output))

    def __call__(self, s: SimpleNamespace) -> torch.Tensor:
        width, device = s.block_ids.shape[1], s.block_ids.device
        logits = s.draft_logits[0].float()
        best = logits.log_softmax(dim=-1).topk(min(self.first, logits.shape[-1]), dim=-1)
        tokens, scores = best.indices.tolist(), best.values.tolist()
        first_token = int(s.block_ids[0, 0])
        chosen, worth = self.tree.choose(s, best, scores)             # the first tree's nodes, as ranks along their paths
        path = {ranks: [tokens[at][rank] for at, rank in enumerate(ranks)] for ranks in chosen}
        had = {}                                                      # the tokens the first tree has after each node
        for ranks in chosen:
            had.setdefault(ranks[:-1], set()).add(path[ranks][-1])
        if self.stand_in == "real" and self.features is None:
            raise ValueError("stand_in='real' needs the answer's states (replay only)")
        # Restarts at the nodes that have a place after them (and, with real states, lie inside the answer).
        nodes = [ranks for ranks in chosen if len(ranks) < width - 1
                 and (self.stand_in != "real" or s.start + len(ranks) <= self.features.shape[1])]
        self.counts["rounds"] += 1
        self.counts["nodes_from_the_first_run"] += len(chosen)
        rows = self._rows(first_token, width, path, chosen, [])
        if not nodes or self.budget == self.first:
            self.nodes = len(chosen)
            return torch.tensor(rows, dtype=s.block_ids.dtype, device=device)
        # The stand-ins: one for every node on a restart's path. The state of the token BEFORE a node,
        # given that the node's token comes after it, belongs to the node (and to every restart below it).
        context, shared, number = None, None, {}
        on_paths = sorted({ranks[:depth] for ranks in nodes for depth in range(1, len(ranks) + 1)}, key=lambda ranks: (len(ranks), ranks))
        if self.stand_in != "none":
            number = {ranks: at for at, ranks in enumerate(on_paths)}
            places = [len(ranks) - 1 for ranks in on_paths]
            if self.stand_in == "real":
                from .stand_in import project

                context = project(s.first_drafter, self.features[:, s.start : s.start + max(places) + 1])[0][places]
            else:
                from .stand_in import predicted_for

                before = [first_token if len(ranks) == 1 else path[ranks[:-1]][-1] for ranks in on_paths]
                context = predicted_for(self.predictor, s, places, before, [path[ranks][-1] for ranks in on_paths])
            shared = [[number[ranks[:depth]] for depth in range(1, len(ranks) + 1)] for ranks in nodes]
            context_places = [s.start + place for place in places]
        else:
            context_places, shared = [], [[] for _ in nodes]
        anchors = [path[ranks][-1] for ranks in nodes]
        begin = [s.start + len(ranks) for ranks in nodes]
        reach = min(1 + self.tail, width) if self.second is not None else (width if self.block is None else min(self.block, width))
        lengths, entering, wanted, room_of = None, None, None, None
        if self.second is not None and getattr(self.second, "reveal", False):
            # Through layers that read the tokens: a node's block is the round's block from its first token to
            # `tail` places after the node, with the tokens of the node's path written in at their places.
            lengths = [min(len(ranks) + reach, width) for ranks in nodes]
            wide, mask_token = max(lengths), s.first_drafter.mask_token_id
            wanted, begin = ([len(ranks) + 1 for ranks in nodes], reach - 1), [s.start] * len(nodes)
            written_in = [[first_token] + path[ranks] + [mask_token] * (wide - len(ranks) - 1) for ranks in nodes]
            known = [[0 < slot <= len(ranks) for slot in range(wide)] for ranks in nodes]
            scale = float(_draft_value(s.first_drafter.config, "input_embedding_scale", 1.0))
            given = torch.tensor(written_in + [[mask_token] * wide], dtype=s.block_ids.dtype, device=device)
            embedded = _raw_input_embeddings(s.target, given, scale)
            entering = self.second.enter_blocks(_gone_on_from(s, self.second, self._first_run, width)[:wide], embedded[:-1],
                                                embedded[-1, 0], torch.tensor(known, device=device))
            if self.second.states_in_block and context is not None:
                # The stand-ins go into the block, at the places of their tokens, not beside it.
                owner = [at for at, rows in enumerate(shared) for _ in rows]
                rows = [row for taken in shared for row in taken]
                entering = self.second.with_states(entering, context[torch.tensor(rows, device=device)], (
                    torch.tensor(owner, device=device), torch.tensor([context_places[row] - s.start for row in rows], device=device)))
                context, context_places, shared = None, [], [[] for _ in nodes]
            room_of = [length - len(ranks) - 1 for length, ranks in zip(lengths, nodes)]
            reach = wide
        elif self.second is not None:
            lengths = [min(width - len(ranks), reach) for ranks in nodes]
            at = torch.tensor([[min(len(ranks) + slot, width - 1) for slot in range(reach)] for ranks in nodes], device=device)
            scale = float(_draft_value(s.first_drafter.config, "input_embedding_scale", 1.0))
            given = torch.tensor([anchors + [s.first_drafter.mask_token_id]], dtype=s.block_ids.dtype, device=device)
            embedded = _raw_input_embeddings(s.target, given, scale)[0]
            entering = self.second.enter(self._first_run[0, -width:][at], embedded[:-1], embedded[-1])
        also = None
        if self.common is not None:
            if self._shortlist is None or self._shortlist.head is not _output_head(s.target):
                from .shortlist import Shortlist

                self._shortlist = Shortlist(_output_head(s.target), self.common)
            also = torch.cat((s.accepted_ids.flatten(), best.indices[:, : min(16, best.indices.shape[1])].flatten()))
        wrote, likely = _restart_pass(s, anchors, begin, context, [], context_places, reach,
                                      layers=None if self.second is None else list(self.second.layers), entering=entering,
                                      lengths=lengths, top=max(self.choices, 1), shared=shared,
                                      shortlist=self._shortlist if self.common is not None else None, also=also, wanted=wanted)
        wrote, likely = wrote.tolist(), likely.tolist()
        if room_of is None:                                           # how many places after each node the pass wrote
            room_of = [(reach if lengths is None else lengths[at]) - 1 for at in range(len(nodes))]
        # What the pass wrote, most probable first: after node n its `choices` best tokens, and after the
        # best one the best tokens of the next places (a chain: each is offered once the one before is in).
        heap = []
        for at, ranks in enumerate(nodes):
            room = min(self.tail, width - 1 - len(ranks), room_of[at])
            for choice in range(min(self.choices, len(wrote[at][0])) if room > 0 else 0):
                token = wrote[at][0][choice]
                if token in had.get(ranks, ()):
                    self.counts["second_run_tokens_the_first_tree_had"] += 1
                    continue
                heapq.heappush(heap, (-(worth[ranks] + likely[at][0][choice]), at, choice, 1, (token,)))
        added = []
        while heap and len(added) < self.budget - len(chosen):
            negative, at, choice, depth, chain = heapq.heappop(heap)
            added.append((nodes[at], chain))
            room = min(self.tail, width - 1 - len(nodes[at]), room_of[at])
            if choice == 0 and depth < room:                          # the chain goes on after the best token only
                heapq.heappush(heap, (negative - likely[at][depth][0], at, 0, depth + 1, chain + (wrote[at][depth][0],)))
        self.counts["nodes_from_the_second_run"] += len(added)
        self.nodes = len(chosen) + len(added)
        return torch.tensor(self._rows(first_token, width, path, chosen, added), dtype=s.block_ids.dtype, device=device)

    @staticmethod
    def _rows(first_token, width, path, chosen, added):
        """A row for every node of the first tree that nothing follows, and one for every chain."""
        longer = {ranks for ranks, _ in added}
        parents = {ranks[:-1] for ranks in chosen}
        chains = {}
        for ranks, chain in added:                                    # of chains that begin alike, the longest
            key = (ranks, chain[0])
            if len(chain) > len(chains.get(key, ())):
                chains[key] = chain
        rows = [[first_token] + path[ranks] for ranks in chosen if ranks not in parents and ranks not in longer]
        rows += [[first_token] + path[ranks] + list(chain) for (ranks, _), chain in chains.items()]
        return [row + [-1] * (width - len(row)) for row in rows]


class SecondTree(BranchTree):
    """A tree of drafts in which a small second drafter scores EVERY edge after the first place.

    DDTree ranks a node by the product of the first run's probabilities of the tokens on its path, and
    the first run gave each place its probabilities without knowing the tokens before it. Here:

      1. a first tree of `first` nodes (DDTree's rule, or the tree with counts put in as `tree`);
      2. ONE pass of the second drafter (dflash/restart_layers.py, trained by nexus/train_every.py) with a
         row for every node of it: the round's block with the tokens of the node's path written in. The
         row gives, for the place after the node, the probabilities of that place's `top` candidates
         (the first run's most probable tokens there) given the path, as a correction of the first
         run's (CandidateScores), with the counts' lift if `follow` is given; and the same for the next
         places, where only the best token is used (a chain after the node's best child);
      3. the tree that is checked is chosen anew, best first, from all of that: `budget` nodes by the
         product of the corrected probabilities on their paths. A node of the first tree may fall out
         and a token the first tree did not have may come in. The first place keeps the first run's
         probabilities: it knew the block's first token.

    Weaver (arXiv 2607.06763) and DARTree (2608.13524) correct a tree's edges with a new head that walks
    the tree in several steps one after the other; this is one pass of a copy of the drafter's own last
    layer over the whole first tree. Rows as DraftTree returns them."""

    def __init__(self, budget: int, first: int | None = None, tail: int = 2, top: int = 32):
        super().__init__(budget, budget, tail=tail, stand_in="none")
        if first is not None:
            # The first tree may be LARGER than the tree that is checked: rows for more nodes in the one pass, so that
            # a token the pass moves up is more often a node that has a row of its own.
            if first < 1:
                raise ValueError("the first tree has at least one node")
            self.first, self.tree = first, DraftTree(first)
        self.top, self.follow = top, None
        self.lean = True                               # one layer: only the places that are asked for run through it (see below)
        self.timed = False                             # count the milliseconds of each part (waits for the GPU between them)
        # gate: the pass is left out in rounds where the first run is sure of itself: where the product of the
        # probabilities of its first choices at the first `gate_places` places is above `gate` (None: never left
        # out). The tree of such a round is the first tree. A pass costs the same whatever it finds, and what it
        # corrects are the branches off the first run's own path, which matter when that path is likely to break.
        self.gate, self.gate_places = None, 4
        self.keep_last, self.last = False, None        # for tests: what the last round's pass gave every node
        # reveal_last: a row sees only the last N tokens of its node's path (0: all), as a second drafter trained that
        # way was shown them (nexus/train_every.py --reveal-last).
        self.reveal_last = 0
        # passes: after the first pass the nodes of the tree that had no row get one and the tree is chosen anew,
        # up to this many passes in all (1: the one pass). For measuring what the scores are worth at every depth.
        self.passes = 1
        # mix: an offline test of WHAT in the second drafter's scores is worth having, against a cheap scorer (`cheap`,
        # a CountTree with pair scores) on the same candidates of the same nodes. A node's children are scored by
        # the second drafter; the cheap scorer's log-probabilities of the same candidates are computed as well, and:
        #   "teacher_order": the second drafter's ORDER of the candidates with the cheap scorer's sorted values
        #   "cheap_order":   the cheap scorer's order with the second drafter's sorted values
        #   "cheap_both":    the cheap scorer's order and values (the cheap tree, built in this harness: a check)
        #   "teacher_both":  the second drafter's order and values (the harness with nothing swapped)
        # With a mix the chains after a node's best child are not used (the best child may change), so every
        # node of the tree needs a row: use with passes=16. The values are each scorer's log-probabilities over the
        # same `top` candidates, each normalized with its own "another token" mass.
        self.mix, self.cheap = None, None
        self.counts = {"rounds": 0, "nodes_from_the_first_run": 0, "nodes_from_the_second_run": 0, "rows_of_the_pass": 0,
                       "first_tree_nodes_dropped": 0}

    def _lap(self, name: str, since: float) -> float:
        now = _cuda_time()
        self.counts[name] = self.counts.get(name, 0.0) + (now - since) * 1000.0
        return now

    def __call__(self, s: SimpleNamespace) -> torch.Tensor:
        second = self.second
        if second is None or second.scores is None or not second.reveal:
            raise ValueError("this tree needs a second drafter that reads the tokens and scores candidates (nexus/train_every.py)")
        width, device = s.block_ids.shape[1], s.block_ids.device
        logits = s.draft_logits[0].float()
        top = min(self.top, logits.shape[-1])
        best = logits.log_softmax(dim=-1).topk(min(max(self.first, top), logits.shape[-1]), dim=-1)
        tokens, scores = best.indices.tolist(), best.values.tolist()
        first_token = int(s.block_ids[0, 0])
        clock = _cuda_time() if self.timed else None
        chosen, _ = self.tree.choose(s, best, scores)
        nodes = [ranks for ranks in chosen if len(ranks) < width - 1 and max(ranks) < top]
        if self.mix is not None:
            if self.cheap is None or self.cheap.pairs is None:
                raise ValueError("a mix needs the cheap scorer: a CountTree with pair scores (`cheap`)")
            if min(self.cheap.candidates, best.indices.shape[1]) != top:
                raise ValueError("the cheap scorer must score the same candidates as the second drafter (top)")
            self.cheap.keep_last = True
            self.cheap.choose(s, best, scores)                        # its tables of this round, for children()
        self.counts["rounds"] += 1
        if self.timed:
            clock = self._lap("ms_first_tree", clock)
        if self.gate is not None and sum(scores[place][0] for place in range(min(self.gate_places, width - 1))) > math.log(max(self.gate, 1e-30)):
            nodes = []                                 # sure of itself: no pass, the first tree as it is
            self.counts["rounds_without_the_pass"] = self.counts.get("rounds_without_the_pass", 0) + 1
        self._round = SimpleNamespace(s=s, best=best, logits=logits, tokens=tokens, top=top, width=width, first_token=first_token)
        # full: a node's children, best first, as a pass gave them (log-probabilities, candidates); chain: the best
        # token of the places after a node's best child.
        full, chain = {}, {}
        if nodes:
            clock = self._rows_scored(nodes, full, chain, clock)
        full[()] = (scores[0][:top], list(range(top)))
        if self.keep_last:
            self.last = {"order": dict(full), "tokens": tokens, "start": s.start}
        if not nodes and chosen:                       # no pass this round: the first tree is the tree
            final, order = dict.fromkeys(chosen[: self.budget], 0.0), dict(full)
        else:
            final, order = self._chosen_anew(full, chain, scores[0][0], width)
            for _ in range(self.passes - 1):
                # More passes (a measurement of what the second drafter's scores are worth when every node of the
                # tree has them; each pass costs a pass): the nodes of the tree that had no row get one, and the
                # tree is chosen anew. When no node is left without a row the tree is the best-first tree under
                # the second drafter's scores at every depth.
                more = [node for node in final if node not in full and len(node) < width - 1]
                if not more:
                    break
                self._rows_scored(more, full, chain, None)
                self.counts["more_passes"] = self.counts.get("more_passes", 0) + 1
                final, order = self._chosen_anew(full, chain, scores[0][0], width)
        if self.keep_last:
            self.last.update(final_order=order, corrected=set(full) - {()}, nodes=list(final))
        had = set(chosen)
        kept = sum(1 for node in final if node in had)
        self.counts["nodes_from_the_first_run"] += kept
        self.counts["nodes_from_the_second_run"] += len(final) - kept
        self.counts["first_tree_nodes_dropped"] += len(chosen) - kept
        self.nodes = len(final)
        parents = {node[:-1] for node in final}
        rows = [[first_token] + [tokens[place][rank] for place, rank in enumerate(node)] + [-1] * (width - 1 - len(node))
                for node in final if node not in parents]
        rows = torch.tensor(rows, dtype=s.block_ids.dtype, device=device)
        if self.timed:
            self._lap("ms_choice", clock)
        return rows

    def corrected(self, nodes) -> dict:
        """After a round, for tracing (nexus/trace_trees.py): what a pass of the second drafter gives the given nodes
        of this round (each the ranks of its tokens, all among the `top` candidates, none at the block's last place):
        {node: (its children's log-probabilities, best first; the candidates)}. One more pass; nothing else changes."""
        full: dict = {}
        if nodes:
            self._rows_scored(list(nodes), full, {}, None)
            self.counts["rows_of_the_pass"] -= len(nodes)             # (not rows of the tree's own pass)
        return full

    def _chosen_anew(self, full: dict, chain: dict, first_score: float, width: int):
        """The tree, best first, from what the passes gave. An entry of the heap: (-score, a number to keep the
        order, the parent, which of its children). Returns the nodes with their scores, and every node's children
        as they were used (those of `full`, and the chains' single tokens)."""
        order = dict(full)
        final, heap, counter, link = {(): 0.0}, [(-first_score, 0, (), 0)], 1, {}
        while heap and len(final) - 1 < self.budget:
            negative, _, parent, which = heapq.heappop(heap)
            values, indices = order[parent]
            node = parent + (indices[which],)
            final[node] = -negative
            if which + 1 < len(values):                                   # the parent's next child
                heapq.heappush(heap, (-(final[parent] + values[which + 1]), counter, parent, which + 1))
                counter += 1
            if node not in order:
                # Not a node a pass had a row for: after a node's BEST child, and after the chain's tokens, the pass
                # wrote one token a place.
                begun = (parent, 1) if (parent and parent in full and which == 0) else link.get(node)
                if begun is not None and begun in chain and len(node) < width - 1:
                    value, index = chain[begun]
                    order[node] = ([value], [index])
                    link[node + (index,)] = (begun[0], begun[1] + 1)
            if node in order and len(node) < width - 1:
                heapq.heappush(heap, (-(final[node] + order[node][0][0]), counter, node, 0))
                counter += 1
        del final[()]
        return final, order

    def _rows_scored(self, ask: list, full: dict, chain: dict, clock):
        """One pass of the second drafter with a row for every node of `ask` (each the ranks of its tokens). full[node]
        becomes the node's children, best first (log-probabilities, candidates), chain[(node, step)] the best token
        of the places after. clock: the time so far if the parts are timed (returned moved on)."""
        from .restart_layers import candidate_scores

        second, r = self.second, self._round
        s, best, logits, tokens, top, width, first_token = r.s, r.best, r.logits, r.tokens, r.top, r.width, r.first_token
        device = s.block_ids.device
        path = {ranks: [tokens[at][rank] for at, rank in enumerate(ranks)] for ranks in ask}
        tail = min(self.tail, width - 2)
        mask_token = s.first_drafter.mask_token_id
        scale = float(_draft_value(s.first_drafter.config, "input_embedding_scale", 1.0))
        depth = torch.tensor([len(ranks) for ranks in ask], device=device)
        gone = _gone_on_from(s, second, self._first_run, width)
        # (the first place of a node's path whose token its row sees: all of them, or the last `reveal_last`)
        since = (lambda ranks: max(1, len(ranks) - self.reveal_last + 1)) if self.reveal_last else (lambda ranks: 1)
        if self.lean and len(second.layers) == 1 and getattr(second.layers[0], "attention_conv", None) is None:
            # A row's places with tokens written in are only LOOKED AT by the places that are asked for; with one
            # layer nothing reads what the layer would make of them. So they are given as what the layer looks at
            # (its keys and values: its own first norm, then its projections, exactly as for a place of the
            # block), once for every node and shared by the rows below it, and only the places asked for run
            # through the layer: 2 a node in place of the node's whole block. Number for number the same scores,
            # at a fraction of the pass (a pass costs by the places that run through it).
            layer = second.layers[0]
            # What is looked at: every asked node's own last token and, in a later pass, the same of the nodes above
            # an asked node that are not asked themselves (the first pass asks whole trees: nothing is added).
            entry_of = {ranks: number for number, ranks in enumerate(ask)}
            for ranks in ask:
                for upto in range(1, len(ranks)):
                    entry_of.setdefault(ranks[:upto], len(entry_of))
            entries = list(entry_of)
            entry_depth = depth if len(entries) == len(ask) else torch.tensor([len(ranks) for ranks in entries], device=device)
            given = torch.tensor([[tokens[len(ranks) - 1][ranks[-1]] for ranks in entries] + [mask_token]], dtype=s.block_ids.dtype, device=device)
            embedded = _raw_input_embeddings(s.target, given, scale)[0]
            written = second._written(gone[entry_depth], embedded[:-1], embedded[-1])
            # Places as the first run left them: the block's first one; and, for rows that leave tokens out, every place,
            # since such a row looks at the places before its tokens as they were (as in its training).
            plain = width if self.reveal_last else 1
            looked_at = layer.input_layernorm(torch.cat((gone[:plain], written), dim=0))
            asked = (depth[:, None] + 1 + torch.arange(tail, device=device)[None, :]).clamp(max=width - 1)
            if clock is not None:
                clock = self._lap("ms_rows", clock)
            states = _restart_pass(s, [], [s.start + len(ranks) + 1 for ranks in ask], looked_at, [],
                                   [s.start + place for place in range(plain)] + [s.start + len(ranks) for ranks in entries], tail,
                                   layers=list(second.layers), entering=gone[asked],
                                   lengths=[min(tail, width - 1 - len(ranks)) for ranks in ask],
                                   shared=[list(range(since(ranks))) + [plain + entry_of[ranks[:upto]] for upto in range(since(ranks), len(ranks) + 1)]
                                           for ranks in ask],
                                   wanted=([0] * len(ask), tail), states_only=True)                      # [nodes, tail, size]
        else:
            lengths = [min(len(ranks) + 1 + tail, width) for ranks in ask]
            wide = max(lengths)
            written_in = [[first_token] + path[ranks] + [mask_token] * (wide - len(ranks) - 1) for ranks in ask]
            known = [[since(ranks) <= slot <= len(ranks) for slot in range(wide)] for ranks in ask]
            embedded = _raw_input_embeddings(s.target, torch.tensor(written_in + [[mask_token] * wide], dtype=s.block_ids.dtype, device=device), scale)
            entering = second.enter_blocks(gone[:wide], embedded[:-1], embedded[-1, 0], torch.tensor(known, device=device))
            states = _restart_pass(s, [path[ranks][-1] for ranks in ask], [s.start] * len(ask), None, [], [], wide,
                                   layers=list(second.layers), entering=entering, lengths=lengths, shared=[[] for _ in ask],
                                   wanted=([len(ranks) + 1 for ranks in ask], tail), states_only=True)   # [nodes, tail, size]
        self.counts["rows_of_the_pass"] += len(ask)
        if clock is not None:
            clock = self._lap("ms_pass", clock)
        # The scores, all on the GPU and read back once at the end.
        steps = torch.arange(tail, device=device)[None, :]
        at = (depth[:, None] + steps).clamp(max=width - 2)                                       # the places, as rows of `best`
        candidates = best.indices[:, :top][at]                                                   # [nodes, tail, top]
        first_log, first_scores = best.values[:, :top][at], logits.gather(-1, best.indices[:, :top])[at]
        seconds = candidate_scores(s.first_drafter, _output_head(s.target), states, candidates)
        about = torch.tensor([[any(ranks), ranks[-2] if len(ranks) > 1 else 0, ranks[-1],
                               ([first_token] + path[ranks])[-2], path[ranks][-1]] for ranks in ask], device=device)
        off, before, last, two, one = about[:, 0] > 0, about[:, 1], about[:, 2], about[:, 3], about[:, 4]
        follow = self.follow
        # The counts' lifts: from the lookup the tree with counts made this round for the same candidates
        # ([places - 1, the candidate two before, the candidate before, the candidate]), or looked up here.
        table = getattr(self.tree, "table", None) if follow is not None else None
        if table is not None and (table.shape[-1] < top or table.shape[0] < width - 2):
            table = None
        results = []
        for step in range(tail):
            if follow is None:
                lift = None
            elif table is not None:
                lift = table[(depth - 1 + step).clamp(max=table.shape[0] - 1), before, last][:, :top]
            else:
                lift = follow.lift(two[:, None], one[:, None], candidates[:, step])
            likely = second.scores(first_log[:, step], first_scores[:, step], seconds[:, step], states[:, step], lift, off)[:, :top]
            if step == 0:
                ranked = likely.sort(dim=-1, descending=True)
                if self.mix is not None:
                    ranked = self._mixed(ask, likely, ranked)
                results.append((ranked.values, ranked.indices))
                chosen_here = ranked.indices[:, 0]
            else:                                                     # the chain after a node's best child: one token a place
                most = likely.max(dim=-1)
                results.append((most.values, most.indices))
                chosen_here = most.indices
            before, last = last, chosen_here
            two, one = one, candidates[:, step].gather(-1, chosen_here[:, None])[:, 0]
            off = off | (chosen_here != 0)
        inside = (depth[:, None] + steps <= width - 2).tolist()
        for step, (values, indices) in enumerate(results):
            values, indices = values.tolist(), indices.tolist()
            for number, ranks in enumerate(ask):
                if step == 0:
                    full[ranks] = (values[number], indices[number])
                elif inside[number][step] and self.mix is None:
                    chain[(ranks, step)] = (values[number], indices[number])
        if clock is not None:
            clock = self._lap("ms_scores", clock)
        return clock


    def _mixed(self, ask: list, likely: torch.Tensor, ranked):
        """The children's scores with `mix` applied (see __init__): ranked (values, indices) of the second drafter's
        log-probabilities `likely` [nodes, top], and the cheap scorer's of the same candidates."""
        cheap = numpy.stack([self.cheap.children(len(ranks), ranks[-2] if len(ranks) > 1 else 0, ranks[-1],
                                                 not any(ranks)) for ranks in ask])                 # [nodes, top]
        cheap = torch.as_tensor(cheap, device=likely.device, dtype=likely.dtype)
        cheap_ranked = cheap.sort(dim=-1, descending=True)
        if self.mix == "teacher_order":
            return SimpleNamespace(values=cheap_ranked.values, indices=ranked.indices)
        if self.mix == "cheap_order":
            return SimpleNamespace(values=ranked.values, indices=cheap_ranked.indices)
        if self.mix == "cheap_both":
            return cheap_ranked
        if self.mix == "teacher_both":
            return ranked
        raise ValueError(f"unknown mix {self.mix!r}")


class EchoRows:
    """Tails written from what the target has already said in this answer. No second run of the
    drafter, no predicted states, nothing trained.

    A table maps the last few tokens (up to `order`) to the token the target chose after them, the
    most recent choice winning. It is filled from the accepted text and, in real decoding, from
    every token the target read in its checks, also those on branches it then rejected: the target
    said what it would write after each of them (heard()). In replay only the accepted text is
    known, so replay shows less than real decoding can give.

    Each round, the table is followed for up to `reach` tokens (a hit needs at least `least` tokens
    of the key to match) from: the block's first token; the last token before each of the k likely
    breaks (which needs no guess of the right token); the drafter's next `choices` choices at each
    likely break; and the end of the first draft (past the block). Rows that add nothing to the first
    draft are left out. Returned: rows [first token, ...] with -1 after their end, or None."""

    swap = False
    alternative_tail = None

    def __init__(self, k: int = 4, choices: int = 1, order: int = 3, reach: int = 12, least: int = 2):
        if not 1 <= least <= order:
            raise ValueError("least must be between 1 and order")
        self.k, self.choices, self.order, self.reach, self.least = k, choices, order, reach, least
        self.table, self.text, self.kept = {}, [], []
        self.rows = self.tokens = self.entries_heard = 0

    def start(self, drafter: nn.Module) -> None:
        self.table, self.text, self.kept = {}, [], []
        self.rows = self.tokens = self.entries_heard = 0

    def learn(self, before: list[int], said: int) -> None:
        for count in range(self.least, min(self.order, len(before)) + 1):
            self.table[tuple(before[-count:])] = said

    def follow(self, before: list[int], most: int) -> list[int]:
        """The table's tokens after `before`, the longest matching key first."""
        tail, recent = [], list(before[-self.order :])
        while len(tail) < most:
            said = None
            for count in range(min(self.order, len(recent)), self.least - 1, -1):
                said = self.table.get(tuple(recent[-count:]))
                if said is not None:
                    break
            if said is None:
                break
            tail.append(said)
            recent = (recent + [said])[-self.order :]
        return tail

    def heard(self, before_block: list[int], tokens: list[int], parents: list[int], said: list[int]) -> None:
        """After a check in real decoding: for every token the target read (a tree of drafts: node 0
        is the block's first token, parents[node] its parent), what the target chose after it."""
        for node, choice in enumerate(said):
            recent, at = [], node
            while at >= 0 and len(recent) < self.order:
                recent.append(tokens[at])
                at = parents[at]
            if at < 0 and len(recent) < self.order:
                recent.extend(reversed(before_block[-(self.order - len(recent)) :]))
            self.learn(recent[::-1], choice)
        self.entries_heard += len(said)

    def __call__(self, round_state: SimpleNamespace) -> torch.Tensor | None:
        s = round_state
        width = s.block_ids.shape[1]
        new = s.accepted_ids[0, len(self.text) :].tolist() + s.block_ids[0, :1].tolist()   # ... and the block's first token
        for token in new:                                             # the accepted text, as far as it is new
            self.learn(self.text, token)
            self.text.append(token)
        draft = s.block_ids[0].tolist()
        first = s.draft_logits[0].float()
        scores = break_scores(first.softmax(dim=-1).amax(dim=-1))[: width - 1]
        self.kept = scores.topk(min(self.k, scores.shape[0])).indices.tolist()
        ranked = first.topk(1 + min(self.choices, first.shape[-1] - 1), dim=-1).indices.tolist()
        made, seen = [], set()

        def add(kept_draft: int, then: list[int]) -> None:
            """A row: the first draft's first `kept_draft` tokens after the block's first, `then`, the table's tail."""
            head = draft[: kept_draft + 1] + then
            row = head + self.follow(self.text + head[1:], self.reach)
            same = row[:width] == draft[: len(row[:width])]
            if (same and len(row) <= width) or len(row) == len(head) == 1 or tuple(row) in seen:
                return                                                # nothing the first draft does not have
            if then and len(row) == len(head):
                return                                                # a switched token with no tail: the tree has it
            seen.add(tuple(row))
            made.append(row)

        add(0, [])                                                    # from the block's first token
        for broke in self.kept:
            add(broke, [])                                            # from the last token before a likely break
            for choice in range(1, len(ranked[broke])):
                add(broke, [ranked[broke][choice]])                   # from the drafter's next choice at the break
        add(width - 1, [])                                            # past the end of the first draft
        if not made:
            return None
        rows = torch.full((len(made), width + self.reach + 1), -1, dtype=s.block_ids.dtype, device=s.block_ids.device)
        for number, row in enumerate(made):
            rows[number, : len(row)] = torch.tensor(row, dtype=rows.dtype, device=rows.device)
        self.rows += len(made)
        self.tokens += sum(len(row) for row in made)
        return rows


class TreeWithRestarts:
    """DDTree for the block and, in the same check, restarts after the likely breaks of the path of
    first choices: the tree covers other tokens at a break (its tokens after one are the first
    pass's, written without knowing of the change), a restart writes a new block there.

    The swaps of the restarts are left out: the tree has them where they are likely. A restart's
    tail (the target checks only that many tokens after the restart's first new token) is cut here,
    so the rows hold exactly what the target reads. nodes: the tokens the target reads besides the
    block's first token, counted once where rows share their beginning. Scored in replay only."""

    replace = True

    def __init__(self, tree: DraftTree, restarts):
        if getattr(restarts, "swap", False):
            raise ValueError("the tree has the swaps: use the restarts without them")
        self.tree, self.restarts, self.nodes = tree, restarts, 0      # restarts: Restarts, EchoRows, any maker of rows

    @property
    def features(self):
        return self.restarts.features                                 # (no such thing for rows that need no states)

    @features.setter
    def features(self, value):
        self.restarts.features = value

    @property
    def kept(self):
        return getattr(self.restarts, "kept", [])

    @property
    def counts(self):
        return getattr(self.restarts, "counts", None)

    @property
    def ahead(self):                                                  # blocks written in the first draft's pass
        return getattr(self.restarts, "ahead", False)

    @property
    def block(self):
        return getattr(self.restarts, "block", None)

    def blocks(self, s: SimpleNamespace):
        return self.restarts.blocks(s)

    def heard(self, before_block, tokens, parents, said) -> None:
        if hasattr(self.restarts, "heard"):
            self.restarts.heard(before_block, tokens, parents, said)

    def start(self, drafter: nn.Module) -> None:
        self.tree.start(drafter)
        self.restarts.start(drafter)
        self.nodes = 0

    def __call__(self, s: SimpleNamespace) -> torch.Tensor:
        made = self.tree(s)
        width = s.block_ids.shape[1]
        block = None
        rows = made.tolist()                                          # (put together as lists, sent to the GPU once)
        if getattr(self.restarts, "wants_first_draft", False):       # the whole first draft is read too
            block = s.block_ids[0].tolist()
            rows.append(list(block))
        more = self.restarts(s)
        if more is not None:
            more = more.tolist()
            tail = getattr(self.restarts, "alternative_tail", None)
            if tail is not None:
                block = s.block_ids[0].tolist() if block is None else block
                for row in more:
                    split = next((place for place in range(width) if row[place] != block[place]), width)
                    row[split + 1 + tail :] = [-1] * max(len(row) - (split + 1 + tail), 0)
            rows += more
            across = max(len(row) for row in rows)
            rows = [row + [-1] * (across - len(row)) for row in rows]
        children, nodes = {}, 0                                       # the tokens read, counted once where rows share their beginning
        for row in rows:
            node = 0
            for token in row[1:]:
                if token < 0:
                    break
                following = children.get((node, token))
                if following is None:
                    nodes += 1
                    following = children[(node, token)] = nodes
                node = following
        self.nodes = nodes
        if more is None and block is None:
            return made
        return torch.tensor(rows, dtype=made.dtype, device=made.device)


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


def _draft_with_blocks(drafter, target, cache, context_features, noise, position_ids, start, block_len, blocks, reach):
    """The round's first draft and other blocks of the same drafter in ONE pass.

    blocks: (first tokens, their places, [(the target's states it is given besides the text's, their
    places)]) as Reanchor.blocks gives them: the states of block n are the first n + 1 of the last
    block's, so they are laid out once. The first draft sees the text and itself, as in a pass of its
    own; a block sees the text, its states and itself. Returns the first draft's final states
    ([1, block_len, size]) and (first tokens, places, the tokens each block writes after its first
    token: [blocks, reach - 1]); the cache keeps the text's new entries only."""
    firsts, places, stand_ins = blocks
    device, count = noise.device, len(firsts)
    states, state_places = stand_ins[-1]
    extra = states.shape[1]
    assert all(entry[0].shape[1] == number + 1 + extra - count for number, entry in enumerate(stand_ins))
    more = torch.full((count, reach), drafter.mask_token_id, dtype=torch.long, device=device)
    more[:, 0] = torch.as_tensor(firsts, dtype=torch.long, device=device)
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    new = context_features.shape[1]
    seen = cache.get_seq_length() + new
    slots = torch.arange(reach, device=device)
    positions = torch.cat([position_ids[0, start - new : start], state_places.to(device), position_ids[0, start : start + block_len]]
                          + [place + slots for place in places])[None]
    queries = block_len + count * reach
    sees = torch.zeros((queries, seen + extra + queries), dtype=torch.bool, device=device)
    sees[:, :seen] = True
    sees[:block_len, seen + extra : seen + extra + block_len] = True
    for number in range(count):
        rows = slice(block_len + number * reach, block_len + (number + 1) * reach)
        sees[rows, seen : seen + extra - count + number + 1] = True
        sees[rows, seen + extra + block_len + number * reach : seen + extra + block_len + (number + 1) * reach] = True
    hidden = drafter(
        target_hidden=torch.cat((context_features, states.to(context_features.dtype)), dim=1),
        noise_embedding=torch.cat((noise, _raw_input_embeddings(target, more.reshape(1, -1), scale)), dim=1),
        position_ids=positions, attention_mask=sees[None, None], past_key_values=cache, use_cache=True,
    )[:, -queries:, :]
    cache.crop(-(extra + queries))
    logits = drafter.compute_logits(hidden[:, block_len:].reshape(count, reach, -1)[:, 1:, :], _output_head(target))
    return hidden[:, :block_len], (firsts, places, logits.argmax(dim=-1))


def _tree_of_drafts(trunk: list[int], rows: list[list[int]], deepest: int):
    """All drafts of a round as one tree. trunk: the tokens of the draft whose entries the target's
    cache keeps in order (the block's first token first); rows: other drafts, each from the block's
    first token on, -1 after their end. No token lies deeper than `deepest` places after the first.

    Returns tokens, depths, parents, children (a dict token -> node for every node) and how many
    nodes the trunk has. Node 0 is the block's first token; the trunk's nodes come first, in order;
    every other node comes after its parent. Drafts that begin alike share those nodes."""
    tokens, depths, parents, children = [trunk[0]], [0], [-1], [{}]

    def walk(row):
        at = 0
        for depth, token in enumerate(row[1:], start=1):
            if token < 0 or depth > deepest:
                break
            following = children[at].get(token)
            if following is None:
                following = len(tokens)
                tokens.append(token)
                depths.append(depth)
                parents.append(at)
                children.append({})
                children[at][token] = following
            at = following

    walk(trunk)
    on_trunk = len(tokens)
    for row in rows:
        walk(row)
    return tokens, depths, parents, children, on_trunk


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
    context_window: int = 0,
    recheck: bool = False,
    follow: torch.Tensor | None = None,
):
    """Greedy DFlash decoding that verifies the first draft and alternative drafts together.

    With ``improver=None`` this is ordinary DFlash and produces the same tokens.
    The improver may return several alternative drafts (one per row). Each shares the first draft's
    tokens up to the first place it differs (its split), so only its tokens from there on are added
    to the target's one verification pass, where they see the text, the shared tokens and
    themselves. The draft the target accepts furthest counts. If the improver has
    ``alternative_tail = N``, only the token at the split and the N tokens after it are checked
    (a shorter pass: most of what an alternative wins is in those tokens).

    A tree of drafts. A plug-in with ``replace = True`` that returns several rows, or rows that end
    in -1 (dflash.redraft.DraftTree, TreeWithRestarts), gives a tree in place of the first draft, and
    alternatives may reach past the block (Restarts). Then all drafts of the round are put together
    as one tree: tokens that drafts share are read once, every token sees the text and the tokens
    on its own path only, and the target follows its own choices through the tree as far as they are
    in it. The cache keeps the entries of the path it followed as far as they were stored in order
    (the tree's first path); the other accepted tokens are read again at the front of the next pass.

    Which tokens come out. In 16-bit arithmetic the target's numbers for a token depend a little on
    the shape of the pass that computes them: a pass with more rows (alternatives, or tokens read
    again) rounds every row differently from plain decoding's pass of one block. Where the target's
    two best tokens are nearly tied (about 1 row in 100, measured by nexus/time_check.py) the first
    choice can then be the other one, and from there the answer is another answer. Every token is
    still the target's first choice in the pass that checked it; the text is plain decoding's text
    only up to such ties. This is so for any check of a tree of drafts.

    follow: a text (ids, prompt included) that decoding is to write, normally plain decoding's
    answer to the same prompt. Every pass is computed as usual, but what the target "says" after a
    token is read from this text, so that all settings write the same answer and their times can be
    compared. The stats count the rows used and how many of them the pass itself decided otherwise
    (followed_rows, followed_against).

    recheck: tokens accepted on an alternative's rows are read by the target a second time anyway,
    in order, at the front of the next pass (their cache entries have to be stored in order). With
    recheck that second reading decides, and decoding goes back to a token on which it disagrees.
    It makes the tokens those of in-order reading where only the alternatives' rows are off (the
    tests arrange that); on the GPU the second reading is itself a pass of another shape, so it does
    not give plain decoding's text there. Off by default.

    context_window: the drafter sees only the last N tokens of the text so far (0 = all of it). A
    drafter trained on short texts drafts better on a long answer that way, and its pass is shorter.

    round_log, if given, receives one entry per full round:
    [block length, split of the best alternative (0 if none), tokens accepted from the first
    draft, tokens accepted from the best alternative (-1 if none), confidences].
    timers, if given, accumulates seconds spent in each part of a round.
    round_clock, if given, receives [tokens written before the round, tokens it produced,
    seconds since decoding began] at the end of every round (a round that goes back: 0 or fewer).
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
    rounds_with_alternative = alternative_wins = alternative_extra_tokens = rounds_edited = corrections = 0
    followed_rows = followed_against = 0
    start = num_input_tokens
    # Accepted tokens whose target cache entries were computed on the alternative
    # branch. They are fed again at the front of the next verification pass.
    pending = output_ids[:, :0]
    leftover = None        # what the last check read past the accepted text (see where it is set)
    stopped = stop_tokens is not None and torch.isin(output_ids[:, start], stop_tokens).any()
    # A plug-in may ask for what the target predicted on each draft (see dflash/adapt.py).
    canvas_top = int(getattr(improver, "canvas_top", 0) or 0)
    canvas = None
    # A plug-in that learns from the draft's own pass makes the draft itself, and is told how the
    # check went right after it (dflash.adapt.OnePassTrainer).
    learner = improver if getattr(improver, "learns_from_the_draft", False) else None
    if learner is not None and context_window:
        raise ValueError("a drafter that learns during the answer reads all of the text: no context window")

    def go_back(agreed, hidden_states, drafted, asked):
        """The target read the pending tokens in order and, after `agreed` of them, chose another
        token than the one accepted on an alternative's rows. Decoding goes on from its choice.
        drafted: the drafter has already read the context up to `start` in this round.
        asked: the plug-in was called in this round."""
        nonlocal start, pending, context_features, stopped, corrections
        position = start - pending.shape[1] + agreed + 1      # where the target's token goes
        back = start - position + 1                           # context read again: from position - 1 on
        output_ids[:, position] = verified[0, agreed]
        _crop_to(target_cache, position)
        clean = extract_context_feature(hidden_states, model.target_layer_ids)[:, agreed : agreed + 1, :]
        if drafted:
            draft_cache.crop(-back)
            if asked and hasattr(improver, "roll_back"):
                improver.roll_back(back)
            context_features = clean
        else:
            context_features = torch.cat((context_features[:, : context_features.shape[1] - back], clean), dim=1)
        corrections += 1
        acceptance_lengths.append(position - start)
        if round_clock is not None:
            round_clock.append([start - num_input_tokens, position - start, _cuda_time() - decode_start])
        start = position
        pending = output_ids[:, :0]
        stopped = stop_tokens is not None and bool(torch.isin(output_ids[:, start], stop_tokens).any())

    while (start + 1 < max_length and not stopped) or (recheck and pending.shape[1] > 0):
        if not (start + 1 < max_length and not stopped):
            # The answer is complete, but its last tokens were accepted on an alternative's rows:
            # the target reads them in order once, as it would at the front of a next pass.
            output = target(
                pending, position_ids=position_ids[:, start - pending.shape[1] : start],
                past_key_values=target_cache, use_cache=True, output_hidden_states=True,
            )
            verified = output.logits.argmax(dim=-1)
            agreed = _leading_matches(torch.cat((pending[:, 1:], output_ids[:, start : start + 1]), dim=1), verified)
            if agreed < pending.shape[1]:
                go_back(agreed, output.hidden_states, drafted=False, asked=False)
            pending = output_ids[:, :0]
            continue
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
                # A plug-in may have blocks of its own written in the first draft's pass (Reanchor): they
                # are started from what the last check left behind, so they do not wait for the draft.
                ahead = None
                if improver is not None and block_len > 2 and getattr(improver, "ahead", False):
                    ahead = improver.blocks(SimpleNamespace(start=start, block_ids=block_ids, leftover=leftover))
                if ahead is None:
                    draft_states = model(
                        target_hidden=context_features, noise_embedding=noise,
                        position_ids=position_ids[:, start - context_features.shape[1] : start + block_len],
                        past_key_values=draft_cache, use_cache=True,
                    )[:, -block_len:, :]
                    draft_cache.crop(-block_len)             # the block's own entries go
                else:
                    draft_states, ahead = _draft_with_blocks(
                        model, target, draft_cache, context_features, noise, position_ids, start, block_len, ahead,
                        min(getattr(improver, "block", None) or block_len, block_len))
                hidden = draft_states[:, 1:, :]
                if context_window:                           # ... and so does context older than the window
                    for layer in draft_cache.layers:
                        if layer.keys is not None and layer.keys.shape[-2] > context_window:
                            layer.keys = layer.keys[..., -context_window:, :]
                            layer.values = layer.values[..., -context_window:, :]
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
                    canvas=canvas, leftover=leftover, ahead=ahead if learner is None else None,
                ))

            if timers is not None and improver is not None:
                timers["improve"] = timers.get("improve", 0.0) + _cuda_time() - clock

        leftover = None                                        # (it was for this round's second run only)
        # An improver that replaces the draft: the edited draft is the only one checked.
        tree = None
        if alternative is not None and getattr(improver, "replace", False):
            if alternative.shape[0] > 1 or bool((alternative < 0).any()):
                # A tree in place of the first draft. Its first path (kept in order in the cache) is
                # the row that follows the drafter's first choices furthest.
                rows = alternative.tolist()
                chain = block_ids[0].tolist()

                def along(row):
                    count = 0
                    while count < min(len(row), len(chain)) and row[count] == chain[count]:
                        count += 1
                    return count

                first_path = max(rows, key=along)
                tree = ([token for token in first_path if token >= 0], rows)
            elif alternative.shape == block_ids.shape and bool((alternative[:, :1] == block_ids[:, :1]).all()):
                rounds_edited += int(bool((alternative != block_ids).any()))
                block_ids = alternative
            alternative = None
        if alternative is not None and alternative.shape[1] != block_len:
            # Alternatives that reach past the block (restarts): the first draft and they, as a tree.
            tail = getattr(improver, "alternative_tail", None)
            rows, chain = [], block_ids[0].tolist()
            for row in alternative.tolist():
                split = next((at for at in range(block_len) if row[at] != chain[at]), block_len)
                if split >= 1:
                    rows.append(row if tail is None else row[: split + 1 + tail])
            tree = (chain, rows)
            alternative = None
        if tree is not None:
            if recheck or canvas_top or learner is not None:
                raise NotImplementedError("a tree of drafts goes without recheck, canvas and a learning drafter")
            tokens, depths, parents, children, on_trunk = _tree_of_drafts(tree[0], tree[1], max_length - start - 1)
            nodes, num_pending = len(tokens), pending.shape[1]
            cached = start - num_pending
            verify_ids = torch.cat((pending, torch.tensor([tokens], dtype=torch.long, device=device)), dim=1)
            verify_positions = torch.cat((position_ids[:, cached:start], start + torch.tensor([depths], device=device)), dim=1)
            attention_mask = None
            if nodes > on_trunk:                               # not a single path: who sees whom
                import numpy

                sees = numpy.zeros((num_pending + nodes, cached + num_pending + nodes), dtype=bool)
                sees[:, :cached] = True
                sees[:num_pending, cached : cached + num_pending] = numpy.tri(num_pending, dtype=bool)
                sees[num_pending:, cached : cached + num_pending] = True
                own = sees[num_pending:, cached + num_pending :]
                for node in range(nodes):                      # itself and what its parent sees
                    if node:
                        own[node] = own[parents[node]]
                    own[node, node] = True
                attention_mask = torch.from_numpy(sees).to(device)[None, None]
                rounds_with_alternative += 1
            clock = _cuda_time() if timers is not None else None
            output = target(verify_ids, position_ids=verify_positions, past_key_values=target_cache, use_cache=True,
                            output_hidden_states=True, attention_mask=attention_mask)
            verified = own_choice = output.logits.argmax(dim=-1)
            if follow is not None:
                verified = follow.to(device)[:, (verify_positions + 1).clamp(max=follow.shape[1] - 1)[0]]
            if timers is not None:
                kind = "two" if attention_mask is not None else "one"
                timers["verify_" + kind] = timers.get("verify_" + kind, 0.0) + _cuda_time() - clock
                timers["rounds_" + kind] = timers.get("rounds_" + kind, 0) + 1
                timers["rows_" + kind] = timers.get("rows_" + kind, 0) + num_pending + nodes
            if hasattr(improver, "heard"):                        # what the target chose after every token it read
                improver.heard(output_ids[0, max(start - 8, 0) : start].tolist(), tokens, parents,
                               own_choice[0, num_pending:].tolist())
            # The target follows its own choices through the tree as far as they are in it.
            said = verified[0, num_pending:].tolist()
            path = [0]
            while said[path[-1]] in children[path[-1]]:
                path.append(children[path[-1]][said[path[-1]]])
            accepted = len(path) - 1
            in_order = next((at for at, node in enumerate(path) if node != at or node >= on_trunk), len(path))
            if in_order < len(path):
                alternative_wins += 1
                alternative_extra_tokens += len(path) - in_order
            if round_log is not None and block_len == block_size:
                confidence = draft_logits.float().softmax(dim=-1).amax(dim=-1)[0]
                round_log.append([block_len, in_order if in_order < len(path) else 0, in_order - 1,
                                  accepted if in_order < len(path) else -1, [round(c, 3) for c in confidence.tolist()]])
            path_rows = torch.tensor([num_pending + node for node in path], device=device)
            if follow is not None:
                followed_rows += accepted + 1
                followed_against += int((own_choice[0, path_rows] != verified[0, path_rows]).sum())
            output_ids[0, start : start + accepted + 1] = torch.tensor([tokens[node] for node in path], device=device)
            output_ids[0, start + accepted + 1] = said[path[-1]]
            produced = min(accepted + 1, max_length - start - 1)
            if stop_tokens is not None:
                stop_indices = torch.isin(output_ids[0, start + 1 : start + produced + 1], stop_tokens).nonzero(as_tuple=True)[0]
                if stop_indices.numel() > 0:
                    produced = stop_indices[0].item() + 1
                    stopped = True
            kept_in_cache = min(produced, in_order)            # the path's entries that were stored in order
            _crop_to(target_cache, start + kept_in_cache)
            pending = output_ids[:, start + kept_in_cache : start + produced].clone()
            features = extract_context_feature(output.hidden_states, model.target_layer_ids)
            context_features = features[:, path_rows[:produced], :]
            if on_trunk > in_order:
                # What the target read on the first path past the text it accepted: those tokens, the
                # place of the first of them, and its states at them. They are states of text that was
                # not written, but where the next draft has the same token at the same place they are
                # the target's own states of that token, with a context that differs a few tokens back.
                # "said": the token the target chose after each of them.
                leftover = {"start": start + in_order, "tokens": tokens[in_order:on_trunk],
                            "features": features[0, num_pending + in_order : num_pending + on_trunk],
                            "said": own_choice[0, num_pending + in_order : num_pending + on_trunk].tolist()}
            if round_clock is not None:
                round_clock.append([start - num_input_tokens, produced, _cuda_time() - decode_start])
            start += produced
            acceptance_lengths.append(produced)
            continue

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
        verified = own_choice = output.logits.argmax(dim=-1)
        if follow is not None:
            # The token the text to follow has after each row's token (rows past its end: its last token).
            after = (torch.cat(verify_positions, dim=1) + 1).clamp(max=follow.shape[1] - 1)
            verified = follow.to(device)[:, after[0]]
        if canvas_top and block_len > 1:
            best = output.logits[0, num_pending : num_pending + block_len - 1].float().log_softmax(dim=-1)
            best = best.topk(canvas_top, dim=-1)
            canvas = (start, best.indices, best.values)
        if timers is not None:
            kind = "two" if branches else "one"
            timers["verify_" + kind] = timers.get("verify_" + kind, 0.0) + _cuda_time() - clock
            timers["rounds_" + kind] = timers.get("rounds_" + kind, 0) + 1
            timers["rows_" + kind] = timers.get("rows_" + kind, 0) + num_pending + block_len + extra

        if recheck and num_pending:
            # What the target says after each pending token, now read in order, must be the token
            # that was accepted after it (after the last one: this block's first token).
            agreed = _leading_matches(torch.cat((pending[:, 1:], block_ids[:, :1]), dim=1), verified[:, :num_pending])
            if agreed < num_pending:
                go_back(agreed, output.hidden_states, drafted=block_len > 1, asked=improver is not None and block_len > 2)
                continue

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
        if follow is not None:                               # the rows that decided this round
            used = winner_rows[: accepted + 1]
            followed_rows += accepted + 1
            followed_against += int((own_choice[0, used] != verified[0, used]).sum())
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
            if first_accepted + 1 < block_len:                # the first draft's tokens the target read and did not accept
                leftover = {"start": start + first_accepted + 1, "tokens": block_ids[0, first_accepted + 1 :].tolist(),
                            "features": features[0, first_rows[first_accepted + 1 :]],
                            "said": own_choice[0, first_rows[first_accepted + 1 :]].tolist()}
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
        corrections=corrections,
        followed_rows=followed_rows,
        followed_against=followed_against,
    )
