"""Training the drafter itself while the answer is written (plan items E9 and E14).

Small low-rank adapters are added to the drafter's layers. They start at zero for every
request, so the first drafts are the frozen drafter's own. During the request they are
trained on the drafter's own drafts: the drafter is run again on a past round's block
and its predictions are compared with the tokens the target committed there.

Which positions of a past round are used is the question this module is built to ask:

  "immediate"  only the positions up to the round's first rejection. Their tokens are
               known as soon as the round has been checked. This is the on-path part of
               what Test-Time Speculation (arXiv 2605.09329) trains on.
  "hindsight"  every position of the round, once the answer has passed them. The tokens
               after the first rejection become known a few rounds later, and they are
               exact, because the drafter's prediction for a position depends only on
               the committed context, not on its own draft tokens.
  "both"       the immediate positions first, and the rest when they become known.
  "canvas"     every position of the round at once, as Test-Time Speculation does: the target
               checked the whole draft, so it predicted something at every position. Up to the
               first rejection that is the same as "immediate". After it, the target was reading
               rejected draft tokens, so its prediction belongs to a text that was never written.
  "canvas+hindsight"  "canvas" first, and then the positions after the first rejection once more
               with their exact tokens when the answer has passed them.
  "rejected"   only the position of the round's first rejection: the one token the target wrote
               itself in that round, its correction of the draft. A round whose whole draft was
               accepted teaches nothing.
  "accepted"   only the positions before the first rejection, where the draft was already right.
               ("immediate" is "accepted" and "rejected" together.)

A round can also be left out by how it went: with trigger = K, only rounds in which at most K
draft tokens were accepted are learned from (an early rejection), which saves the other steps.
With after = N, nothing is learned until the text is N tokens long: early in an answer the
drafter is at its best and there is little to gain. With windows = [(from, to), ...] the
drafter learns only while the text is between those lengths, and keeps what it learned after
each window: a short spell of learning, then drafting at the plain drafter's cost.

What the drafter is compared with can be the committed token alone ("ce"), or what the
target itself predicted at that position: its most likely tokens with their probabilities
("kl", as in Test-Time Speculation). For a position on the answer's path, the target
computes that prediction anyway when it checks the round that commits the position.

Only the parts of the drafter that work on the block are adapted (not the projections
that turn the target's states into keys and values), so the drafter's cache of the
context stays valid while the adapters change.

DraftTrainer is a plug-in for dflash_generate_redraft and replay_decode. It never
returns a draft: the drafter's own drafts change because its adapters do. Decoding
stays lossless, since the target still checks every draft.

OnePassTrainer does the same learning without running the drafter a second time: the
draft itself is made with its gradients kept, and the update follows right after the
target's check of that draft. This is cheaper (no second run), and the next draft
already uses what the last check taught.
"""

import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from .model import _crop_to, _draft_value, _make_cache, _output_head, _raw_input_embeddings
from .sharp import set_query_scale

ADAPTED = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LABELS = ("immediate", "hindsight", "both", "canvas", "canvas+hindsight", "rejected", "accepted")


class LowRank(nn.Module):
    """A frozen linear layer plus a trainable low-rank update that starts at zero."""

    def __init__(self, base, rank, scale=1.0, seed=0):
        super().__init__()
        self.base = base
        generator = torch.Generator().manual_seed(seed)
        self.down = nn.Parameter(torch.randn(rank, base.in_features, generator=generator) / base.in_features ** 0.5)
        self.up = nn.Parameter(torch.zeros(base.out_features, rank))
        self.register_buffer("down_at_start", self.down.detach().clone(), persistent=False)
        self.scale = scale
        # fast: when no gradients are kept for this adapter, work the update out in the layer's own
        # number format, from copies of the two halves that are renewed whenever they change. This
        # saves the conversions of every call (about a quarter of a draft's time with 25 adapters).
        # Off by default, so that replays stay number for number what they were.
        self.fast = False
        self._copies = None          # (versions of the two halves, number format, down, up times scale)

    def forward(self, x):
        out = self.base(x)
        if self.fast and not (torch.is_grad_enabled() and (self.down.requires_grad or self.up.requires_grad)):
            versions = (self.down._version, self.up._version)
            if self._copies is None or self._copies[0] != versions or self._copies[1] != out.dtype:
                with torch.inference_mode(False), torch.no_grad():
                    self._copies = (versions, out.dtype, self.down.detach().to(out.dtype),
                                    (self.up.detach() * self.scale).to(out.dtype))
            return out + F.linear(F.linear(x, self._copies[2]), self._copies[3])
        update = F.linear(F.linear(x.to(self.down.dtype), self.down), self.up) * self.scale
        return out + update.to(out.dtype)

    @torch.no_grad()
    def reset(self, state=None):
        """Back to the frozen layer: both halves of the update return to where they started.
        With a saved state (down, up), back to that state instead."""
        if state is None:
            self.up.zero_()
            self.down.copy_(self.down_at_start)
        else:
            self.down.copy_(state[0])
            self.up.copy_(state[1])


def attach_adapters(model, rank=16, scale=1.0, names=ADAPTED):
    """Freezes the drafter and wraps the chosen linear layers of every block layer. Returns the adapters."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapters = []
    for number, layer in enumerate(model.layers):
        for parent in (layer.self_attn, layer.mlp):
            for name in names:
                module = getattr(parent, name, None)
                if isinstance(module, LowRank):
                    adapters.append(module)
                elif isinstance(module, nn.Linear):
                    wrapped = LowRank(module, rank, scale, seed=1000 * number + len(adapters)).to(module.weight.device)
                    wrapped.where = (number, name)          # which layer and which of its linear layers
                    setattr(parent, name, wrapped)
                    adapters.append(wrapped)
    for adapter in adapters:                 # also when they were attached before and have just been frozen above
        adapter.down.requires_grad_(True)
        adapter.up.requires_grad_(True)
    return adapters


def choose_adapters(adapters, only="all"):
    """The adapters a setting trains: "all", "attn" (attention), "mlp", or "lastK" (the last K layers)."""
    if only in (None, "", "all"):
        return list(adapters)
    if only == "attn":
        return [adapter for adapter in adapters if adapter.where[1] in ("q_proj", "o_proj")]
    if only == "mlp":
        return [adapter for adapter in adapters if adapter.where[1] in ("gate_proj", "up_proj", "down_proj")]
    if only.startswith("last") and only[4:].isdigit():
        layers = sorted({adapter.where[0] for adapter in adapters})[-int(only[4:]) :]
        return [adapter for adapter in adapters if adapter.where[0] in layers]
    raise ValueError(f"unknown choice of adapters: {only}")


def adapter_state(adapters):
    """A copy of what the adapters hold now: (down, up) of each."""
    return [(adapter.down.detach().clone(), adapter.up.detach().clone()) for adapter in adapters]


def save_adapters(path, adapters, **notes):
    """Writes the adapters to a file (16 MB at rank 16 for the Qwen3-4B drafter)."""
    state = [(down.float().cpu(), up.float().cpu()) for down, up in adapter_state(adapters)]
    torch.save({"state": state, "rank": int(adapters[0].down.shape[0]), **notes}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_adapters(path, adapters):
    """Reads saved adapters as a starting state for the attached ones, which they must fit."""
    saved = torch.load(path, map_location="cpu")
    assert len(saved["state"]) == len(adapters), "the saved adapters belong to another drafter"
    state = []
    for adapter, (down, up) in zip(adapters, saved["state"]):
        assert down.shape == adapter.down.shape and up.shape == adapter.up.shape, "the saved adapters have another rank"
        state.append((down.to(adapter.down), up.to(adapter.up)))
    return state


def adapter_notes(path):
    """What a saved adapter file says about how it was made (answers seen, updates, labels, rate ...)."""
    saved = torch.load(path, map_location="cpu")
    return {key: value for key, value in saved.items() if key != "state"}


def detach_adapters(model):
    """Puts the original linear layers back."""
    for layer in model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, module in list(parent.named_children()):
                if isinstance(module, LowRank):
                    setattr(parent, name, module.base)


class DraftTrainer:
    """Trains the drafter's adapters during a request. Call begin() before each answer."""

    replace = False

    def __init__(self, model, adapters, labels="hindsight", rate=1e-4, stride=1, decay=7.0, loss="ce", top=32,
                 init=None, trained=None, trigger=None, sharpen=None, after=0, windows=None):
        assert labels in LABELS and loss in ("ce", "kl")
        # windows: learn only from rounds that begin while the text is between these lengths [(from, to), ...].
        self.windows = [tuple(window) for window in windows] if windows else None
        # after: learn only from rounds that begin once the text (the prompt included) has this many tokens.
        self.after = after
        # trigger: learn only from rounds in which at most this many draft tokens were accepted (None = every round).
        self.trigger = trigger
        # sharpen: a rule for sharper attention on long texts (dflash/sharp.py), switched on for this
        # trainer's requests and off for the others. It needs no training; it can be combined with it.
        self.sharpen = sharpen
        # A control for replay only: the drafter sees just the last `context_window` tokens (0 = all).
        self.context_window = 0
        # trained: the adapters this trainer changes (choose_adapters). The others stay as they start.
        self.trained = list(adapters) if trained is None else list(trained)
        # init: a saved state of the adapters (load_adapters) that every request starts from, instead of zero.
        self.init = init
        # Asks decoding (or replay) for the target's `top` most likely tokens at every position of each draft.
        self.canvas_top = top if labels.startswith("canvas") else 0
        self.model, self.adapters = model, adapters
        self.labels, self.rate, self.stride, self.decay, self.loss = labels, rate, stride, decay, loss
        self.predictions = None      # for "kl": (tokens, log-probabilities) the target predicted after every position
        self.parameters = [parameter for adapter in self.trained for parameter in (adapter.down, adapter.up)]
        self.scale = float(_draft_value(model.config, "input_embedding_scale", 1.0))
        self.totals = {"updates": 0, "positions": 0, "seconds": 0.0, "rounds": 0}
        self.frozen_target = False
        self.begin()

    def start(self, first_drafter):
        pass

    def open_at(self, start):
        """Whether a round that begins when the text has `start` tokens may be learned from."""
        if start < self.after:
            return False
        return self.windows is None or any(low <= start < high for low, high in self.windows)

    def begin(self, keep=False):
        """Forget what was learned: every request starts from the frozen drafter (or from `init`).

        keep: go on from what the earlier answers taught instead. This is not what a request should
        do; it is for training across answers (nexus/train_adapter.py)."""
        set_query_scale(self.model, self.sharpen)
        if not keep:
            for number, adapter in enumerate(self.adapters):
                adapter.reset(None if self.init is None else self.init[number])
            self.optimizer = torch.optim.Adam(self.parameters, lr=self.rate) if self.rate > 0 else None
        # Only the adapters this trainer changes ask for gradients, so a step goes no deeper than it must.
        chosen = {id(adapter) for adapter in self.trained}
        for adapter in self.adapters:
            adapter.down.requires_grad_(id(adapter) in chosen)
            adapter.up.requires_grad_(id(adapter) in chosen)
        self.cache = None            # the drafter's cache of the context, kept separately for training
        self.pieces = deque()        # (first position, target states) for the context not yet dropped
        self.anchors = deque()       # past rounds that still have positions to train on
        self.round = 0
        self.trained_on = []         # (anchor, first offset, labels) of every update on committed tokens, for checking
        self.canvas_used = []        # (anchor, the target's first choices on that draft) of every "canvas" update

    def __call__(self, round_state):
        if self.optimizer is not None:
            with torch.inference_mode(False):
                self._round(round_state)
        self.totals["rounds"] += 1
        return None

    def _round(self, s):
        if self.round == 0 and not self.frozen_target:
            for parameter in s.target.parameters():      # no gradients are wanted for the target's own weights
                parameter.requires_grad_(False)
            self.frozen_target = True
        start = int(s.start)
        width = s.block_ids.shape[1] - 1
        piece = s.context_features.clone()
        self.pieces.append((start - piece.shape[1], piece))
        committed = torch.cat((s.accepted_ids[0], s.block_ids[0, :1])).clone()        # positions 0 .. start
        clock = _clock(piece)

        canvas = getattr(s, "canvas", None)
        waiting = deque()
        for anchor in self.anchors:
            known = min(start - anchor["start"], anchor["width"])    # offsets 0 .. known - 1 have their token now
            fresh = anchor["round"] == self.round - 1
            first, last, soft = anchor["done"], anchor["done"], None
            if fresh:
                # How many of the round's draft tokens were accepted. If that is fewer than the draft had,
                # the next position holds the token the target wrote itself: its correction.
                anchor["accepted"] = min(start - anchor["start"] - 1, anchor["width"])
            accepted = anchor.get("accepted", anchor["width"])
            if self.labels in ("immediate", "both") and fresh:
                last = known
            if self.labels == "rejected" and fresh and accepted < anchor["width"]:
                first, last = accepted, accepted + 1
            if self.labels == "accepted" and fresh:
                first, last = 0, accepted
            if self.labels.startswith("canvas") and fresh and canvas is not None \
                    and canvas[0] == anchor["start"] and canvas[1].shape[0] == anchor["width"]:
                first, last, soft = 0, anchor["width"], canvas
            if soft is None and self.labels in ("hindsight", "both", "canvas+hindsight") and known >= anchor["width"]:
                last = known
            wanted = (self.trigger is None or accepted <= self.trigger) and self.open_at(anchor["start"])
            if last > first and anchor["round"] % self.stride == 0 and wanted:
                self._train(s, anchor, first, last, committed, soft)
                anchor["done"] = known if soft is not None else last
            finished = known >= anchor["width"] or (
                self.labels in ("immediate", "canvas", "rejected", "accepted") and anchor["round"] < self.round)
            if not finished:
                waiting.append(anchor)
        self.anchors = waiting
        if width >= 1:
            self.anchors.append(dict(start=start, token=int(s.block_ids[0, 0]), width=width, round=self.round, done=0))
        # Context older than every waiting round is no longer needed.
        oldest = min((anchor["start"] for anchor in self.anchors), default=start)
        reach = min(oldest, self._cache_length())
        while self.pieces and self.pieces[0][0] + self.pieces[0][1].shape[1] <= reach:
            self.pieces.popleft()
        self.totals["seconds"] += _clock(piece) - clock
        self.round += 1

    def _cache_length(self):
        return 0 if self.cache is None else self.cache.get_seq_length()

    def _context(self, begin, end, like):
        """The target's states for positions begin .. end - 1, from the kept pieces."""
        parts = []
        for first, piece in self.pieces:
            low, high = max(begin, first), min(end, first + piece.shape[1])
            if low < high:
                parts.append(piece[:, low - first : high - first])
        if not parts:
            return like[:, :0]
        found = torch.cat(parts, dim=1)
        assert found.shape[1] == end - begin, "context for a past round is missing"
        return found

    def _train(self, s, anchor, first, last, committed, soft=None):
        """One step on the positions first .. last - 1 of a past round's block.

        soft: what the target predicted on that round's draft, to learn from instead of the committed tokens."""
        model, position = self.model, anchor["start"]
        if self.cache is None:
            self.cache = _make_cache(model.config)
        if self._cache_length() > position:
            _crop_to(self.cache, position)
        fed = self._cache_length()
        context = self._context(fed, position, s.context_features)
        block = torch.full((1, anchor["width"] + 1), model.mask_token_id, dtype=torch.long, device=committed.device)
        block[0, 0] = anchor["token"]
        labels = committed[position + 1 + first : position + 1 + last]
        tokens = scores = None
        if soft is not None:
            tokens, scores = soft[1][first:last].clone(), soft[2][first:last].float().clone()
            labels = tokens[:, 0]
        elif self.loss == "kl":
            # Row p of the predictions describes the token at p + 1, so position q is described by row q - 1.
            rows = slice(position + first, position + last)
            tokens, scores = self.predictions[0][rows].clone(), self.predictions[1][rows].float().clone()
        weights = torch.exp(-torch.arange(first, last, device=labels.device) / self.decay)
        with torch.enable_grad():
            noise = _raw_input_embeddings(s.target, block, self.scale)
            states = model(
                target_hidden=context, noise_embedding=noise,
                position_ids=s.position_ids[:, fed : position + block.shape[1]],
                past_key_values=self.cache, use_cache=True,
            )[:, -block.shape[1] :, :]
            logits = model.compute_logits(states[:, 1 + first : 1 + last, :], _output_head(s.target))[0].float()
            if self.loss == "kl":
                wanted = torch.log_softmax(scores, dim=-1)           # the target's top tokens, made to sum to one
                given = torch.log_softmax(logits, dim=-1).gather(1, tokens)
                each = (wanted.exp() * (wanted - given)).sum(dim=-1)
            else:
                each = F.cross_entropy(logits, labels, reduction="none")
            loss = (each * weights).sum() / weights.sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        _crop_to(self.cache, position)
        for layer in self.cache.layers:          # the kept context must not hold on to the training graph
            layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
        if soft is not None:
            self.canvas_used.append((position, labels.tolist()))
        else:
            self.trained_on.append((position, first, labels.tolist()))
        self.totals["updates"] += 1
        self.totals["positions"] += last - first


def _clock(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter()


class OnePassTrainer(DraftTrainer):
    """Learns from the draft's own pass: no second run of the drafter.

    Decoding (and replay) hand the draft over to draft(), which keeps the gradients of the
    drafter's predictions, and call learn() as soon as the target has checked that draft.
    The update is applied before the next draft is made.

    labels: "immediate" (the accepted tokens and the target's own token), "rejected" (the
    target's own token alone), "accepted" (the accepted tokens alone), or "canvas" (what the
    target predicted at every position of the draft, as Test-Time Speculation uses it).
    loss "kl" compares with the target's predictions instead of the token alone; it needs
    them from the check, like "canvas" does.
    """

    def __init__(self, model, adapters, labels="immediate", **options):
        assert labels in ("immediate", "rejected", "accepted", "canvas")
        super().__init__(model, adapters, labels=labels, **options)
        if self.loss == "kl" or labels == "canvas":
            self.canvas_top = options.get("top", 32)

    @property
    def learns_from_the_draft(self):
        return self.optimizer is not None

    def begin(self, keep=False):
        super().begin(keep)
        self.pending = None          # (round start, the draft's predictions with their gradients)

    def __call__(self, round_state):
        self.totals["rounds"] += 1
        return None

    def draft(self, model, head, context, noise, positions, cache, start):
        """The draft of one round: the drafter's states and predictions for the block.

        On rounds that may be learned from, the pass keeps its gradients until learn() is called."""
        block = noise.shape[1]
        keep = self.round % self.stride == 0 and self.open_at(start)
        self.round += 1
        if not keep:
            self.pending = None
            states = model(target_hidden=context, noise_embedding=noise, position_ids=positions,
                           past_key_values=cache, use_cache=True)[:, -block:, :]
            _crop_to(cache, start)
            return states, model.compute_logits(states[:, 1:, :], head)
        if not self.frozen_target:
            for parameter in head.parameters():          # no gradients are wanted for the target's own weights
                parameter.requires_grad_(False)
            self.frozen_target = True
        with torch.inference_mode(False), torch.enable_grad():
            states = model(target_hidden=context, noise_embedding=noise, position_ids=positions,
                           past_key_values=cache, use_cache=True)[:, -block:, :]
            _crop_to(cache, start)
            for layer in cache.layers:                   # the kept context must not hold on to the graph
                layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
            logits = model.compute_logits(states[:, 1:, :], head)
        self.pending = (start, logits)
        return states.detach(), logits.detach()

    def learn(self, start, accepted, tokens, canvas=None):
        """The target has checked the draft that began at `start`.

        accepted: how many of its tokens were accepted.
        tokens: the tokens the answer has after `start`: the accepted ones, then the target's own.
        canvas: (start, tokens, log-probabilities) the target predicted at every position of the draft."""
        if self.pending is None:
            return
        position, logits = self.pending
        self.pending = None
        assert position == start, "the check is not of the draft that was kept"
        width = logits.shape[1]
        accepted = min(int(accepted), width)
        known = min(accepted + 1, width, int(tokens.shape[0]))       # positions whose token is known now
        first, last = 0, known
        if self.labels == "rejected":
            first, last = (accepted, accepted + 1) if accepted < known else (0, 0)
        elif self.labels == "accepted":
            last = min(accepted, known)
        soft = None
        if self.canvas_top:
            if canvas is None or canvas[0] != start:
                return
            if self.labels == "canvas":
                first, last = 0, min(width, int(canvas[1].shape[0]))
            soft = canvas
        if last <= first or (self.trigger is not None and accepted > self.trigger):
            return
        clock = _clock(logits)
        with torch.inference_mode(False), torch.enable_grad():
            chosen = logits[0, first:last].float()
            weights = torch.exp(-torch.arange(first, last, device=chosen.device) / self.decay)
            if soft is not None:
                wanted = torch.log_softmax(soft[2][first:last].float().clone(), dim=-1)
                given = torch.log_softmax(chosen, dim=-1).gather(1, soft[1][first:last].clone())
                each = (wanted.exp() * (wanted - given)).sum(dim=-1)
                used = soft[1][first:last, 0]
            else:
                used = tokens[first:last].clone()
                each = F.cross_entropy(chosen, used, reduction="none")
            loss = (each * weights).sum() / weights.sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        if self.labels == "canvas":
            self.canvas_used.append((start, used.tolist()))
        else:
            self.trained_on.append((start, first, used.tolist()))
        self.totals["updates"] += 1
        self.totals["positions"] += last - first
        self.totals["seconds"] += _clock(logits) - clock
