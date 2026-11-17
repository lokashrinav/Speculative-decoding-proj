"""Stand-ins for the target's hidden states of tokens the target has not read yet.

A DFlash-style drafter reads the target's hidden states of the text before the block it drafts,
after its own projection (fc, then hidden_norm): one vector per token. A second run that restarts
the drafter after a break (dflash/redraft.py: Restarts) needs such vectors for this round's first
token and the draft tokens before the break, and the target has not read those yet.

StatePredictor writes them. For each of those tokens it reads what exists at that moment:
  - the drafter's own final state at that token of the block, from the first run;
  - the token itself (its embedding);
  - the last real vector before the round;
  - how far into the block the token is.
It is trained on saved answers, where the real vectors are known (nexus/train_restart.py).
"""
import torch
from torch import nn

from .model import _draft_value, _raw_input_embeddings


def _unit(values: torch.Tensor) -> torch.Tensor:
    return values * values.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).rsqrt()


class StatePredictor(nn.Module):
    def __init__(self, size: int, inner: int = 512, hidden: int = 1024, block: int = 16):
        super().__init__()
        self.sizes = {"size": size, "inner": inner, "hidden": hidden, "block": block}
        self.from_state = nn.Linear(size, inner)
        self.from_token = nn.Linear(size, inner)
        self.from_last = nn.Linear(size, inner)
        self.place = nn.Embedding(block, inner)
        self.body = nn.Sequential(nn.Linear(3 * inner, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, size)

    def forward(self, states: torch.Tensor, tokens: torch.Tensor, last: torch.Tensor, places: torch.Tensor) -> torch.Tensor:
        """states, tokens: [count, size]; last: [1, size] or [count, size]; places: [count] (0 = the block's
        first token). Returns [count, size]: what goes through the drafter's hidden_norm (see as_context)."""
        joined = torch.cat((
            self.from_state(_unit(states)) + self.place(places),
            self.from_token(_unit(tokens)),
            self.from_last(_unit(last)).expand(states.shape[0], -1),
        ), dim=-1)
        return self.out(self.body(joined))


def project(drafter: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """The target's states as the drafter reads them."""
    return drafter.hidden_norm(drafter.fc(features.to(drafter.fc.weight.dtype)))


def as_context(drafter: nn.Module, written: torch.Tensor) -> torch.Tensor:
    """A predictor's output put through the drafter's hidden_norm (in the predictor's own precision,
    so that it can be trained through it): the same kind of vector as project() gives."""
    norm = drafter.hidden_norm
    eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
    return written * (written.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt() * norm.weight.to(written.dtype)


def predictor_inputs(s, count: int):
    """What the predictor reads for the round's first `count` tokens (the block's first token and
    the draft tokens after it): the drafter's final states there, the tokens' embeddings, the last
    real vector before the round, and the places."""
    drafter = s.first_drafter
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    tokens = _raw_input_embeddings(s.target, s.block_ids[:, :count], scale)[0].float()
    states = s.draft_states[0, :count].float()
    last = project(drafter, s.context_features[:, -1:])[0].float()
    return states, tokens, last, torch.arange(count, device=states.device)


def predicted_states(predictor: StatePredictor, s, count: int) -> torch.Tensor:
    """Stand-ins for the target's states of the round's first `count` tokens, as the drafter reads
    them: [1, count, size]."""
    return as_context(s.first_drafter, predictor(*predictor_inputs(s, count)))[None]


def save_predictor(path: str, predictor: StatePredictor, notes: dict | None = None) -> None:
    torch.save({"sizes": predictor.sizes, "notes": notes or {},
                "state": {name: value.detach().half().cpu() for name, value in predictor.state_dict().items()}}, path)


def load_predictor(path: str, device) -> tuple[StatePredictor, dict]:
    saved = torch.load(path, map_location="cpu")
    predictor = StatePredictor(**saved["sizes"])
    predictor.load_state_dict({name: value.float() for name, value in saved["state"].items()})
    return predictor.to(device).eval(), saved.get("notes", {})
