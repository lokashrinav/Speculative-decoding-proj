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


class RestartLayers(nn.Module):
    def __init__(self, drafter: nn.Module, count: int = 1, rank: int = 64, seed: int = 0, reveal: bool = False):
        super().__init__()
        total = len(drafter.layers)
        if not 1 <= count < total:
            raise ValueError("the restart layers stand in for some of the drafter's last layers, not all of them")
        # reveal: the restart's block is the round's whole block from its first token on, with the tokens that
        # are known at the restart WRITTEN IN at their places (the draft tokens before the restart and its first
        # token; see enter_block), so the layers read the tokens themselves next to any stand-ins of the target's
        # states. This is the second pass of D-Loop (Chen et al., arXiv 2610.06011: a retrained drafter re-run on
        # the block with a prefix of its draft revealed), at the cost of these layers instead of a whole pass.
        self.sizes = {"count": count, "rank": rank, "reveal": bool(reveal)}
        self.reveal = bool(reveal)
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
    torch.save({"sizes": layers.sizes, "notes": notes or {}, "state": state}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_restart_layers(path: str, drafter: nn.Module) -> tuple[RestartLayers, dict]:
    saved = torch.load(path, map_location="cpu")
    layers = RestartLayers(drafter, **saved["sizes"])                # (files saved before `reveal` existed: without it)
    assert len(saved["state"]["adapters"]) == len(layers.adapters), "the saved restart layers belong to another drafter"
    with torch.no_grad():
        for adapter, (down, up) in zip(layers.adapters, saved["state"]["adapters"]):
            adapter.down.copy_(down.to(adapter.down))
            adapter.up.copy_(up.to(adapter.up))
        layers.token_down.weight.copy_(saved["state"]["token_down"].to(layers.token_down.weight))
        layers.token_up.weight.copy_(saved["state"]["token_up"].to(layers.token_up.weight))
    return layers.eval(), saved.get("notes", {})
