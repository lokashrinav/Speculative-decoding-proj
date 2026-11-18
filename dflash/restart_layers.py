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
    def __init__(self, drafter: nn.Module, count: int = 1, rank: int = 64, seed: int = 0):
        super().__init__()
        total = len(drafter.layers)
        if not 1 <= count < total:
            raise ValueError("the restart layers stand in for some of the drafter's last layers, not all of them")
        self.sizes = {"count": count, "rank": rank}
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
    layers = RestartLayers(drafter, **saved["sizes"])
    assert len(saved["state"]["adapters"]) == len(layers.adapters), "the saved restart layers belong to another drafter"
    with torch.no_grad():
        for adapter, (down, up) in zip(layers.adapters, saved["state"]["adapters"]):
            adapter.down.copy_(down.to(adapter.down))
            adapter.up.copy_(up.to(adapter.up))
        layers.token_down.weight.copy_(saved["state"]["token_down"].to(layers.token_down.weight))
        layers.token_up.weight.copy_(saved["state"]["token_up"].to(layers.token_up.weight))
    return layers.eval(), saved.get("notes", {})
