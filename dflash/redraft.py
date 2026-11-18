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
             layers=None, entering: torch.Tensor | None = None, timers: dict | None = None) -> torch.Tensor:
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
    timers: seconds are added to its "prepare", "pass" and "head" (the clock waits for the GPU)."""
    drafter, device, count = s.first_drafter, s.block_ids.device, len(begin)
    if layers is not None and len(set(layers)) != len(layers):
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
        sees[rows, seen + extra + number * width : seen + extra + (number + 1) * width] = True
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
        for number in set(layers):
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

    def start(self, first_drafter: nn.Module) -> None:
        self.kept = []

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
        s = round_state
        width = s.block_ids.shape[1]
        first = s.draft_logits[0].float()
        confidence = first.softmax(dim=-1).amax(dim=-1)
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
        top = first.softmax(dim=-1).topk(1 + choices, dim=-1)
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
            for index in worth.flatten().topk(min(self.best, worth.numel())).indices.tolist():
                if worth[index // choices, index % choices] >= 0:
                    starts.append((index // choices, 1 + index % choices))
        chosen = len(starts)
        if self.kind != "choice":
            starts += [(broke, 0) for broke in self.kept if broke >= 1]
        if not starts:
            return None
        begin, stand_ins, given = [], [], {}
        used = sorted({broke for broke, _ in starts})
        hits, read = self._from_the_last_check(s, used[-1] + 1)
        if self.stand_in == "predicted":
            from .stand_in import predicted_states

            written = predicted_states(self.predictor, s, used[-1] + 1)
            if hits:
                written = written.clone()
                written[0, hits] = read.to(written.dtype)
        for broke in used:
            missing = broke + 1
            places = torch.arange(s.start, s.start + missing, device=device)
            if self.stand_in == "predicted":
                given[broke] = (written[:, :missing], places, True)
            elif self.stale and self.stand_in == "none":              # only the states the last check left
                mine = [number for number, at in enumerate(hits) if at < missing]
                given[broke] = (read[None, mine], places[[hits[number] for number in mine]], True) if mine else None
            elif self.stand_in == "real":
                given[broke] = (self.features[:, s.start : s.start + missing], places)
            elif self.stand_in == "last" and s.context_features.shape[1]:
                given[broke] = (s.context_features[:, -1:].repeat(1, missing, 1), places)
            else:
                given[broke] = None
        first_tokens = []
        told = self.stand_in == "predicted" and bool(self.predictor.sizes.get("coming"))
        for broke, choice in starts:
            entry = given[broke]
            if choice:
                first_tokens.append(ranked[broke, choice])
                begin.append(s.start + broke + 1)
                if told:        # the predictor is told which token this restart starts from
                    mine = predicted_states(self.predictor, s, broke + 1, coming=ranked[broke, choice])
                    for number, at in enumerate(hits):
                        if at <= broke:
                            mine[0, at] = read[number].to(mine.dtype)
                    entry = (mine, entry[1], True)
            else:                                                     # from the last token before the break: one state fewer
                first_tokens.append(s.block_ids[0, broke])
                begin.append(s.start + broke)
                if entry is not None:
                    keep = entry[1] < s.start + broke
                    entry = (entry[0][:, keep], entry[1][keep]) + tuple(entry[2:]) if bool(keep.any()) else None
            nearest = broke if choice else broke - 1                  # the token right before the restart's first token
            self.counts["restarts"] += 1
            self.counts["states_missing"] += nearest + 1
            self.counts["states_from_the_last_check"] += sum(1 for at in hits if at <= nearest)
            self.counts["restarts_with_the_nearest_state_from_the_last_check"] += int(nearest in hits)
            stand_ins.append(entry)
        reach = width if self.block is None else min(self.block, width)
        new = restarts(s, first_tokens, begin, stand_ins, reach)
        rows = torch.full((len(starts) + (chosen if self.swap else 0), 2 * width), -1, dtype=s.block_ids.dtype, device=device)
        for number, (broke, choice) in enumerate(starts):
            rows[number, : broke + 1] = s.block_ids[0, : broke + 1]
            if not choice:
                rows[number, broke + 1 : broke + reach] = new[number]
                continue
            rows[number, broke + 1] = ranked[broke, choice]
            rows[number, broke + 2 : broke + 1 + reach] = new[number]
            if self.swap:
                swapped = rows[len(starts) + number]
                swapped[:width] = s.block_ids[0]
                swapped[broke + 1] = ranked[broke, choice]
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

    def __call__(self, s: SimpleNamespace) -> torch.Tensor:
        logits = s.draft_logits[0].float()
        best = logits.log_softmax(dim=-1).topk(min(self.budget, logits.shape[-1]), dim=-1)
        tokens = best.indices.tolist()
        chosen = self.paths(best.values.tolist())
        self.nodes = len(chosen)
        parents = {ranks[:-1] for ranks in chosen}
        leaves = [ranks for ranks in chosen if ranks not in parents]
        rows = torch.full((len(leaves), s.block_ids.shape[1]), -1, dtype=s.block_ids.dtype, device=s.block_ids.device)
        rows[:, 0] = s.block_ids[0, 0]
        for number, ranks in enumerate(leaves):
            rows[number, 1 : len(ranks) + 1] = torch.tensor([tokens[at][rank] for at, rank in enumerate(ranks)],
                                                           dtype=rows.dtype, device=rows.device)
        return rows


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
        rows = self.tree(s)
        width = s.block_ids.shape[1]
        if getattr(self.restarts, "wants_first_draft", False):       # the whole first draft is read too
            rows = torch.cat((rows, s.block_ids.to(rows.dtype)), dim=0)
        more = self.restarts(s)
        if more is not None:
            tail = getattr(self.restarts, "alternative_tail", None)
            if tail is not None:
                for row in more:
                    differs = (row[:width] != s.block_ids[0]).nonzero()
                    split = int(differs[0]) if differs.numel() else width
                    row[split + 1 + tail :] = -1
            across = max(rows.shape[1], more.shape[1])
            wide = torch.full((rows.shape[0] + more.shape[0], across), -1, dtype=rows.dtype, device=rows.device)
            wide[: rows.shape[0], : rows.shape[1]] = rows
            wide[rows.shape[0] :, : more.shape[1]] = more
            rows = wide
        seen = set()
        for row in rows.tolist():
            for end in range(2, len(row) + 1):
                if row[end - 1] < 0:
                    break
                seen.add(tuple(row[:end]))
        self.nodes = len(seen)
        return rows


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
