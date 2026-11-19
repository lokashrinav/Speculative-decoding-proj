"""A tiny second drafter: the drafter's last layers, trained to restart after a break.

A second run that restarts the whole drafter after a break (dflash/redraft.py: Restarts) costs a
whole drafter pass, and a pass costs by the layers it runs. RestartLayers runs only the LAST layers,
and goes on from the first run instead of starting over:

  - the places after the restart's first token enter with the states the FIRST run had there after
    the layers that are left out. The first run already read the whole context for these places.
    What it did not know is the tokens before them: it fills all places of a block at once.
  - the restart's first token (the drafter's other choice at the break) enters as the first run's
    state at that place with the mask's embedding replaced by the token's, plus a small trained map
    of the token's embedding;
  - the tokens of this round that the target has not read are given as predicted states of the
    target (dflash/stand_in.py), as in the whole restart.

The layers are COPIES of the drafter's last layers with low-rank updates (dflash/adapt.py: LowRank,
on the parts that work on the block; the projections that make keys and values are left alone, so
the drafter's cache of the context is the copies' cache too). The drafter itself is not changed:
its first draft stays number for number what it was. Trained in nexus/train_second.py.
"""
import copy
import os

import torch
from torch import nn

from .adapt import ADAPTED, LowRank


class CandidateScores(nn.Module):
    """A second run's probabilities of a place's CANDIDATES, as a correction of the first run's.

    The first run gave every place of the block a probability for every token. A second run that knows
    the tokens before a place (the branch of a tree the place lies on) does not start over: for the
    place's candidates (the first run's K most probable tokens there) it changes the first run's
    log-probability by how much its own score of the token differs from the first run's score, and,
    where counts of the target's text are given (dflash/follow.py), by their lift, weighted as
    DForkTree weights it (more after a branch has left the first run's first choices). All other
    tokens together keep the first run's probability, moved by a trained number read from the second
    run's state: if the second run finds no candidate likely, "another token" gains, and the branch is
    worth less in a tree. The result is normalized over the K candidates and "another token".

    The published correctors score candidates only as well (Weaver, arXiv 2607.06763: 512 of them;
    UBTree, arXiv 2609.39972: 64), and DForkTree (arXiv 2611.12356, eq. 4) keeps the first run's rest
    in the same way. Nothing here reads the whole vocabulary."""

    def __init__(self, size: int, on_path: float = 0.3, off_path: float = 0.6):
        super().__init__()
        self.rest = nn.Linear(size, 1)
        nn.init.zeros_(self.rest.weight)
        nn.init.zeros_(self.rest.bias)
        self.strength = nn.Parameter(torch.ones(()))                  # of the second run's change of the scores
        self.on_path = nn.Parameter(torch.tensor(float(on_path)))     # of the counts while the branch is the first run's first choices
        self.off_path = nn.Parameter(torch.tensor(float(off_path)))   # ... and after it has left them

    def forward(self, first_log, first_scores, second_scores, state, lift=None, off=None):
        """first_log [..., K]: the first run's log-probabilities of the candidates (over the whole vocabulary);
        first_scores, second_scores [..., K]: the two runs' scores of them before any softmax; state [..., size]:
        the second run's state at the place; lift [..., K]: the counts' lift of each candidate after the two
        tokens before the place (None: no counts); off [...]: whether the branch has left the first run's first
        choices. Returns log-probabilities [..., K + 1]: the candidates, then "another token"."""
        first_log = first_log.float()
        raised = first_log + self.strength * (second_scores.float() - first_scores.float())
        if lift is not None:
            raised = raised + torch.where(off[..., None], self.off_path, self.on_path) * lift.float()
        rest = torch.log1p(-first_log.exp().sum(dim=-1).clamp(max=1.0 - 1e-6)) + self.rest(state.to(self.rest.weight.dtype))[..., 0].float()
        return torch.cat((raised, rest[..., None]), dim=-1).log_softmax(dim=-1)


def candidate_scores(drafter: nn.Module, output_head: nn.Module, states: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """The scores (before any softmax) the output head gives the tokens `candidates` ([..., K]) at `states`
    ([..., size]), as drafter.compute_logits would, without the rest of the vocabulary: [..., K]."""
    if _capped(drafter):
        raise ValueError("a drafter that caps its scores needs the whole output head")
    scores = torch.einsum("...d,...kd->...k", states.to(output_head.weight.dtype), output_head.weight[candidates])
    if getattr(output_head, "bias", None) is not None:
        scores = scores + output_head.bias[candidates]
    return scores * _multiplier(drafter)


def _draft_setting(drafter, name, default=None):
    from .model import _draft_value

    return _draft_value(drafter.config, name, default)


def _capped(drafter) -> bool:
    cap = _draft_setting(drafter, "final_logit_softcapping")
    return cap is not None and float(cap) > 0


def _multiplier(drafter) -> float:
    return float(_draft_setting(drafter, "output_multiplier", 1.0))


class RestartLayers(nn.Module):
    def __init__(self, drafter: nn.Module, count: int = 1, rank: int = 64, seed: int = 0, reveal: bool = False,
                 states_in_block: bool = False, scores: bool = False):
        super().__init__()
        total = len(drafter.layers)
        if not 1 <= count <= total or (count == total and not reveal):
            raise ValueError("the restart layers stand in for some of the drafter's last layers (all of them only if they read the tokens)")
        # count = all of the drafter's layers: nothing of the first run is gone on from. The block enters as the
        # first run's did, as embeddings, with the known tokens in place of their masks: a whole second pass by a
        # reader trained for it, at a whole pass's cost. It is the measure of what fewer layers give up.
        if states_in_block and not reveal:
            raise ValueError("states are written into the block at the places of the tokens they belong to: with reveal")
        # reveal: the restart's block is the round's whole block from its first token on, with the tokens that
        # are known at the restart WRITTEN IN at their places (the draft tokens before the restart and its first
        # token; see enter_block), so the layers read the tokens themselves next to any stand-ins of the target's
        # states. This is the second pass of D-Loop (Chen et al., arXiv 2610.06011: a retrained drafter re-run on
        # the block with a prefix of its draft revealed), at the cost of these layers instead of a whole pass.
        # states_in_block: the stand-ins of the target's states are not given as context beside the block (where
        # the drafter's FIRST layers pick them up; its last layer alone hardly looks there, trained or not), but
        # written into the block itself: the state of a known token is added, through a trained map, at that
        # token's place, next to the token. The block's places look at each other in every layer.
        # scores: the layers do not give probabilities over the whole vocabulary but CORRECT the first run's
        # probabilities of its own candidates at a place (CandidateScores below; trained in nexus/train_every.py).
        self.sizes = {"count": count, "rank": rank, "reveal": bool(reveal), "states_in_block": bool(states_in_block)}
        if scores:                                     # (not among the sizes of layers saved before the option existed)
            self.sizes["scores"] = True
        self.reveal, self.states_in_block = bool(reveal), bool(states_in_block)
        self.first = total - count                     # the first of the drafter's layers a copy stands in for
        self.layers = nn.ModuleList(copy.deepcopy(layer) for layer in drafter.layers[self.first :])
        for parameter in self.layers.parameters():
            parameter.requires_grad_(False)
        self.adapters = []
        for number, layer in enumerate(self.layers):
            for parent in (layer.self_attn, layer.mlp):
                for name in ADAPTED:
                    module = getattr(parent, name, None)
                    if isinstance(module, LowRank):    # the drafter carries add-on weights of its own: the copy starts from the plain layer
                        module = module.base
                    if isinstance(module, nn.Linear):
                        wrapped = LowRank(module, rank, seed=seed + 1000 * number + len(self.adapters)).to(module.weight.device)
                        setattr(parent, name, wrapped)
                        self.adapters.append(wrapped)
        size = drafter.config.hidden_size
        self.token_down = nn.Linear(size, rank, bias=False)
        self.token_up = nn.Linear(rank, size, bias=False)
        nn.init.zeros_(self.token_up.weight)           # at the start the token enters as described above, nothing added
        self.state_down = nn.Linear(size, rank, bias=False) if states_in_block else None
        self.state_up = nn.Linear(rank, size, bias=False) if states_in_block else None
        if states_in_block:
            nn.init.zeros_(self.state_up.weight)
        self.scores = CandidateScores(size) if scores else None
        self.to(drafter.fc.weight.device)

    def float_parameters(self):
        """The trained parameters in 32-bit numbers (as they are while training), whatever they were read as."""
        for parameter in self.trained():
            parameter.data = parameter.data.float()
        return self

    def trained(self):
        """The parameters that are trained (the copies' own weights are frozen)."""
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def enter(self, states: torch.Tensor, token_in: torch.Tensor, mask_in: torch.Tensor) -> torch.Tensor:
        """states: [..., places, size], the first run's states after the last layer left out, at the
        restart's places (the first one: the place of its first token); token_in, mask_in: [..., size],
        the embeddings of that token and of the mask as the drafter is given them. Returns the states
        the places enter the restart layers with."""
        unit = token_in.float() * token_in.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).rsqrt()
        added = self.token_up(self.token_down(unit.to(self.token_down.weight.dtype)))
        first = states[..., :1, :] + (token_in - mask_in).to(states.dtype)[..., None, :] + added.to(states.dtype)[..., None, :]
        return torch.cat((first, states[..., 1:, :]), dim=-2)

    def enter_block(self, states: torch.Tensor, known_in: torch.Tensor, mask_in: torch.Tensor) -> torch.Tensor:
        """For layers that read the tokens (reveal): states: [..., places, size], the first run's states of a
        block from its first token on, after the last layer left out; known_in: [..., known, size], the
        embeddings of the tokens known at the places 1 .. known (the draft tokens before the restart and the
        restart's first token; the block's own first token at place 0 is a real token in the first run already).
        Each known place keeps the first run's state, the mask's embedding in it replaced by the token's, plus
        the trained map of the token's embedding. Returns the states the block enters the restart layers with."""
        known = known_in.shape[-2]
        if not known:
            return states
        written = self._written(states[..., 1 : 1 + known, :], known_in, mask_in[..., None, :])
        return torch.cat((states[..., :1, :], written, states[..., 1 + known :, :]), dim=-2)

    def _written(self, states, tokens_in, mask_in):
        unit = tokens_in.float() * tokens_in.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).rsqrt()
        added = self.token_up(self.token_down(unit.to(self.token_down.weight.dtype)))
        return states + (tokens_in - mask_in).to(states.dtype) + added.to(states.dtype)

    def enter_blocks(self, states: torch.Tensor, tokens_in: torch.Tensor, mask_in: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        """enter_block for several restarts with different numbers of known tokens: states [places, size]
        (the same first-run states for all), tokens_in [restarts, places, size], known [restarts, places]
        (true where a token is written in). Returns [restarts, places, size]."""
        states = states[None].expand(tokens_in.shape[0], -1, -1)
        return torch.where(known[..., None], self._written(states, tokens_in, mask_in), states)

    def with_states(self, entering: torch.Tensor, stand_ins: torch.Tensor, places) -> torch.Tensor:
        """For layers that take the states in the block: entering [..., places, size] (from enter_block or
        enter_blocks) with the stand-ins of the target's states ([n, size], as the drafter reads states) added
        through the trained map at `places`: an index of the block's places ([n]) or, for several restarts, a
        pair (restart numbers [n], places [n])."""
        if not stand_ins.shape[0]:
            return entering
        added = self.state_up(self.state_down(stand_ins.to(self.state_down.weight.dtype))).to(entering.dtype)
        if isinstance(places, tuple):
            return entering.index_put((places[0], places[1]), added, accumulate=True)
        return entering.index_add(-2, places, added)

    @torch.no_grad()
    def merge(self):
        """For use: the low-rank updates are added into the copies' weights, so that a pass through
        them costs what a pass through the drafter's own layers costs. Not to be trained afterwards."""
        for layer in self.layers:
            for parent in (layer.self_attn, layer.mlp):
                for name in ADAPTED:
                    module = getattr(parent, name, None)
                    if isinstance(module, LowRank):
                        base = module.base
                        update = (module.up.float() @ module.down.float()) * module.scale
                        base.weight.add_(update.to(base.weight.dtype))
                        setattr(parent, name, base)
        self.adapters = []
        return self


def save_restart_layers(path: str, layers: RestartLayers, notes: dict | None = None) -> None:
    """Only what was trained is written (about 7 MB at rank 64 for one layer of the Qwen3-4B drafter)."""
    state = {"adapters": [(adapter.down.detach().half().cpu(), adapter.up.detach().half().cpu()) for adapter in layers.adapters],
             "token_down": layers.token_down.weight.detach().half().cpu(),
             "token_up": layers.token_up.weight.detach().half().cpu()}
    if layers.states_in_block:
        state.update(state_down=layers.state_down.weight.detach().half().cpu(), state_up=layers.state_up.weight.detach().half().cpu())
    if layers.scores is not None:
        state["scores"] = {name: value.detach().float().cpu() for name, value in layers.scores.state_dict().items()}
    torch.save({"sizes": layers.sizes, "notes": notes or {}, "state": state}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_restart_layers(path: str, drafter: nn.Module, **more) -> tuple[RestartLayers, dict]:
    """more: options the saved layers did not have yet (states_in_block=True: the layers go on being trained with
    the states written into the block; the new map starts at nothing)."""
    saved = torch.load(path, map_location="cpu")
    layers = RestartLayers(drafter, **{**saved["sizes"], **more})    # (files saved before an option existed: without it)
    assert len(saved["state"]["adapters"]) == len(layers.adapters), "the saved restart layers belong to another drafter"
    with torch.no_grad():
        for adapter, (down, up) in zip(layers.adapters, saved["state"]["adapters"]):
            adapter.down.copy_(down.to(adapter.down))
            adapter.up.copy_(up.to(adapter.up))
        layers.token_down.weight.copy_(saved["state"]["token_down"].to(layers.token_down.weight))
        layers.token_up.weight.copy_(saved["state"]["token_up"].to(layers.token_up.weight))
        if layers.states_in_block and "state_down" in saved["state"]:
            layers.state_down.weight.copy_(saved["state"]["state_down"].to(layers.state_down.weight))
            layers.state_up.weight.copy_(saved["state"]["state_up"].to(layers.state_up.weight))
        if layers.scores is not None and "scores" in saved["state"]:
            layers.scores.load_state_dict({name: value.to(layers.scores.rest.weight.device) for name, value in saved["state"]["scores"].items()})
    return layers.eval(), saved.get("notes", {})
