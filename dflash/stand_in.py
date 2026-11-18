"""Stand-ins for the target's hidden states of tokens the target has not read yet.

A DFlash-style drafter reads the target's hidden states of the text before the block it drafts,
after its own projection (fc, then hidden_norm): one vector per token. A second run that restarts
the drafter after a break (dflash/redraft.py: Restarts) needs such vectors for this round's first
token and the draft tokens before the break, and the target has not read those yet.

StatePredictor writes them. For each of those tokens it reads what exists at that moment:
  - the drafter's own final state at that token of the block, from the first run;
  - the drafter's final state at the NEXT token of the block: the target's state at a token is
    what the target predicts the next token from, and this is the drafter's counterpart of it;
  - the token itself (its embedding);
  - the last real vector before the round;
  - how far into the block the token is.
It is trained on saved answers, where the real vectors are known (nexus/train_restart.py).

Two things the first predictors did not have (they closed about a third of the gap between no
states and the real ones at real breaks):
  - coming=True: the token that comes AFTER each token. For the draft tokens it is the next draft
    token; for the last one it is the token the restart starts from. The target's state at a token
    is the state the next token is chosen from, and at a break the drafter's own state there points
    at the wrong token, so without this the predicted state contradicts the restart's first token.
  - layers > 0: the missing tokens look at each other, earlier ones only, and (memory > 0) at the
    last few real vectors. A break means the drafter filled its slots without knowing what the
    other slots chose; the restart knows the tokens, and this is where they are put together.
"""
import torch
from torch import nn

from .model import _draft_value, _raw_input_embeddings


def _unit(values: torch.Tensor) -> torch.Tensor:
    return values * values.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).rsqrt()


class Mixer(nn.Module):
    """One layer in which each missing token looks at the missing tokens before it and at itself
    (never at later ones, so a vector does not depend on how far the draft went on), and at the last
    real vectors."""

    def __init__(self, hidden: int, heads: int, remember: bool):
        super().__init__()
        self.before_own, self.own = nn.LayerNorm(hidden), nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.before_recall = nn.LayerNorm(hidden) if remember else None
        self.recall = nn.MultiheadAttention(hidden, heads, batch_first=True) if remember else None
        self.before_wide = nn.LayerNorm(hidden)
        self.wide = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, values: torch.Tensor, memory: torch.Tensor | None) -> torch.Tensor:
        count = values.shape[1]
        later = torch.ones(count, count, dtype=torch.bool, device=values.device).triu(1)
        seen = self.before_own(values)
        values = values + self.own(seen, seen, seen, attn_mask=later, need_weights=False)[0]
        if self.recall is not None and memory is not None and memory.shape[1]:
            values = values + self.recall(self.before_recall(values), memory, memory, need_weights=False)[0]
        return values + self.wide(self.before_wide(values))


class StatePredictor(nn.Module):
    def __init__(self, size: int, inner: int = 512, hidden: int = 1024, block: int = 16, coming: bool = False,
                 layers: int = 0, heads: int = 8, memory: int = 0):
        super().__init__()
        self.sizes = {"size": size, "inner": inner, "hidden": hidden, "block": block, "coming": coming,
                      "layers": layers, "heads": heads, "memory": memory}
        self.from_state = nn.Linear(size, inner)
        self.from_next = nn.Linear(size, inner)
        self.from_token = nn.Linear(size, inner)
        self.from_last = nn.Linear(size, inner)
        self.from_coming = nn.Linear(size, inner) if coming else None
        self.place = nn.Embedding(block, inner)
        self.body = nn.Sequential(nn.Linear((5 if coming else 4) * inner, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.remember = nn.Linear(size, hidden) if layers and memory else None
        self.mix = nn.ModuleList([Mixer(hidden, heads, bool(memory)) for _ in range(layers)])
        self.out = nn.Linear(hidden, size)

    def forward(self, states: torch.Tensor, following: torch.Tensor, tokens: torch.Tensor, last: torch.Tensor,
                places: torch.Tensor, coming: torch.Tensor | None = None, memory: torch.Tensor | None = None) -> torch.Tensor:
        """states, following, tokens: [count, size] (the drafter's final state at each token, at the
        token after it (zeros past the block's end), and the token's embedding); last: [1, size];
        places: [count] (0 = the block's first token); coming: [count, size], the embedding of the
        token after each token; memory: [any, size], the last real vectors. The last two are read
        only by a predictor built for them. Returns [count, size]: what goes through the drafter's
        hidden_norm (see as_context)."""
        parts = [
            self.from_state(_unit(states)) + self.place(places),
            self.from_next(_unit(following)),
            self.from_token(_unit(tokens)),
            self.from_last(_unit(last)).expand(states.shape[0], -1),
        ]
        if self.from_coming is not None:
            parts.append(self.from_coming(_unit(coming)))
        values = self.body(torch.cat(parts, dim=-1))
        if len(self.mix):
            recalled = None
            if self.remember is not None and memory is not None and memory.shape[0]:
                recalled = self.remember(_unit(memory))[None]
            values = values[None]
            for layer in self.mix:
                values = layer(values, recalled)
            values = values[0]
        return self.out(values)


def project(drafter: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """The target's states as the drafter reads them."""
    return drafter.hidden_norm(drafter.fc(features.to(drafter.fc.weight.dtype)))


def as_context(drafter: nn.Module, written: torch.Tensor) -> torch.Tensor:
    """A predictor's output put through the drafter's hidden_norm (in the predictor's own precision,
    so that it can be trained through it): the same kind of vector as project() gives."""
    norm = drafter.hidden_norm
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    return written * (written.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt() * norm.weight.to(written.dtype)


def predictor_inputs(s, count: int, coming=None, memory: int = 0):
    """What the predictor reads for the round's first `count` tokens (the block's first token and
    the draft tokens after it): the drafter's final states there and one token further, the tokens'
    embeddings, the last real vector before the round, the places, the embedding of the token after
    each token, and the last `memory` real vectors of the round's new context (None for 0).
    coming: the token after the last of them, which is the token a restart starts from (None: the
    first draft's own token there)."""
    drafter = s.first_drafter
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    tokens = _raw_input_embeddings(s.target, s.block_ids[:, :count], scale)[0].float()
    last = project(drafter, s.context_features[:, -1:])[0].float()
    after = s.block_ids[0, 1 : count + 1].clone()
    if after.shape[0] < count:                                   # past the block's end: no token there
        after = torch.cat((after, s.block_ids[0, -1:]))
    if coming is not None:
        after[-1] = int(coming)
    after = _raw_input_embeddings(s.target, after[None], scale)[0].float()
    recalled = project(drafter, s.context_features[:, -memory:])[0].float() if memory else None
    return (*state_inputs(s.draft_states[0], count), tokens, last, torch.arange(count, device=tokens.device), after, recalled)


def state_inputs(block_states: torch.Tensor, count: int):
    """From the drafter's final states of a block ([block, size]): the states at its first `count`
    tokens and at the token after each (zeros past the block's end)."""
    states = block_states[:count].float()
    following = torch.zeros_like(states)
    reach = min(count, block_states.shape[0] - 1)
    following[:reach] = block_states[1 : 1 + reach].float()
    return states, following


def predicted_states(predictor: StatePredictor, s, count: int, coming=None) -> torch.Tensor:
    """Stand-ins for the target's states of the round's first `count` tokens, as the drafter reads
    them: [1, count, size]. coming: the token the restart starts from (see predictor_inputs)."""
    inputs = predictor_inputs(s, count, coming, int(predictor.sizes.get("memory", 0)))
    return as_context(s.first_drafter, predictor(*inputs))[None]


def predicted_for(predictor: StatePredictor, s, places: list[int], tokens: list[int], coming: list[int]) -> torch.Tensor:
    """Stand-ins for the target's states of any tokens at places of the block, not only the first
    draft's own: the state at the block's place places[n] if the token there is tokens[n] and the
    token after it is coming[n] (read only by a predictor that is told it). [n, size], as the drafter
    reads states. For a predictor whose tokens do not look at each other: each state depends on its
    own place and tokens alone, which is what lets one call serve every path of a tree."""
    if len(predictor.mix):
        raise ValueError("a predictor whose tokens look at each other writes a path at a time: use predicted_states")
    drafter, device = s.first_drafter, s.block_ids.device
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    block = s.draft_states[0].float()
    at = torch.tensor(places, device=device)
    states = block[at]
    following = block[(at + 1).clamp(max=block.shape[0] - 1)] * (at + 1 < block.shape[0])[:, None]
    embedded = _raw_input_embeddings(s.target, torch.tensor([tokens, coming], device=device), scale).float()
    last = project(drafter, s.context_features[:, -1:])[0].float()
    return as_context(drafter, predictor(states, following, embedded[0], last, at, embedded[1], None))


def save_predictor(path: str, predictor: StatePredictor, notes: dict | None = None) -> None:
    torch.save({"sizes": predictor.sizes, "notes": notes or {},
                "state": {name: value.detach().half().cpu() for name, value in predictor.state_dict().items()}}, path)


def load_predictor(path: str, device) -> tuple[StatePredictor, dict]:
    saved = torch.load(path, map_location="cpu")
    predictor = StatePredictor(**saved["sizes"])
    predictor.load_state_dict({name: value.float() for name, value in saved["state"].items()})
    return predictor.to(device).eval(), saved.get("notes", {})
