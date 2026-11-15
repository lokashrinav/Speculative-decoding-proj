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
"""

import time
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn

from .model import _crop_to, _draft_value, _make_cache, _output_head, _raw_input_embeddings

ADAPTED = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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

    def forward(self, x):
        out = self.base(x)
        update = F.linear(F.linear(x.to(self.down.dtype), self.down), self.up) * self.scale
        return out + update.to(out.dtype)

    @torch.no_grad()
    def reset(self):
        """Back to the frozen layer: both halves of the update return to where they started."""
        self.up.zero_()
        self.down.copy_(self.down_at_start)


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
                    setattr(parent, name, wrapped)
                    adapters.append(wrapped)
    return adapters


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

    def __init__(self, model, adapters, labels="hindsight", rate=1e-4, stride=1, decay=7.0, loss="ce"):
        assert labels in ("immediate", "hindsight", "both") and loss in ("ce", "kl")
        self.model, self.adapters = model, adapters
        self.labels, self.rate, self.stride, self.decay, self.loss = labels, rate, stride, decay, loss
        self.predictions = None      # for "kl": (tokens, log-probabilities) the target predicted after every position
        self.parameters = [parameter for adapter in adapters for parameter in (adapter.down, adapter.up)]
        self.scale = float(_draft_value(model.config, "input_embedding_scale", 1.0))
        self.totals = {"updates": 0, "positions": 0, "seconds": 0.0, "rounds": 0}
        self.frozen_target = False
        self.begin()

    def start(self, first_drafter):
        pass

    def begin(self):
        """Forget what was learned: every request starts from the frozen drafter."""
        for adapter in self.adapters:
            adapter.reset()
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.rate) if self.rate > 0 else None
        self.cache = None            # the drafter's cache of the context, kept separately for training
        self.pieces = deque()        # (first position, target states) for the context not yet dropped
        self.anchors = deque()       # past rounds that still have positions to train on
        self.round = 0
        self.trained_on = []         # (anchor, first offset, labels) of every update, for checking

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

        waiting = deque()
        for anchor in self.anchors:
            known = min(start - anchor["start"], anchor["width"])    # offsets 0 .. known - 1 have their token now
            first, last = anchor["done"], anchor["done"]
            if self.labels in ("immediate", "both") and anchor["round"] == self.round - 1:
                last = known
            if self.labels in ("hindsight", "both") and known >= anchor["width"]:
                last = known
            if last > first and anchor["round"] % self.stride == 0:
                self._train(s, anchor, first, last, committed)
                anchor["done"] = last
            finished = known >= anchor["width"] or (self.labels == "immediate" and anchor["round"] < self.round)
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

    def _train(self, s, anchor, first, last, committed):
        """One step on the positions first .. last - 1 of a past round's block."""
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
                # Row p of the predictions describes the token at p + 1, so position q is described by row q - 1.
                rows = slice(position + first, position + last)
                tokens, scores = self.predictions[0][rows].clone(), self.predictions[1][rows].float().clone()
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
        self.trained_on.append((position, first, labels.tolist()))
        self.totals["updates"] += 1
        self.totals["positions"] += last - first


def _clock(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter()
