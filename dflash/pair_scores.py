"""The second drafter's knowledge without a pass: scores of pairs of neighbouring tokens.

A second drafter (restart_layers.py) corrects the first run's probabilities at a place once it knows
the tokens before the place. Trained at every place it is right far more often than the first run
after a branch has left the first run's first choices, but its pass through a layer costs more of a
round than the tokens it adds (docs/SECOND_DRAFTER_RESEARCH.md). Most of what "knowing the tokens
before" gives lies in the token right before and the one before that (DSpark's pair table gets
nearly what its recurrent head gets, arXiv 2607.05147). That part needs no pass:

    change(c | a2, a, place) = (P(h) * phi(e_a)) . psi(E_c)  +  (P2(h) * phi2(e_a2)) . psi(E_c)

h is the first run's own state at the place (already computed), e_a and e_a2 the target's frozen
input embeddings of the token before and the token two before, E_c the frozen output-head row of
a candidate c; phi, phi2 and psi are small networks, P and P2 linear maps. For all candidates of
all places these are two small tables per place ([before, here] and [two before, here]), made by a
handful of matrix products, and a tree can be searched with them at every depth, as the tree with
counts is searched with its lifts (dflash/redraft.py: CountTree, `pairs`). The form is the selector
of DFlash 2 and UBTree (arXiv 2609.39972), which are trained with their drafter on 1.3 million
answers; here the drafter is frozen, the scores are trained on top of the counts on about a
thousand answers (nexus/train_pairs.py), with the same loss as the second drafter.
"""
from __future__ import annotations

import torch
from torch import nn


class PairScores(nn.Module):
    def __init__(self, size: int, rank: int = 256, hidden: int = 1024, on_path: float = 0.3, off_path: float = 0.6):
        super().__init__()

        def code(last_zero: bool = False):
            layers = nn.Sequential(nn.Linear(size, hidden, bias=False), nn.SiLU(), nn.Linear(hidden, rank, bias=False))
            if last_zero:
                nn.init.zeros_(layers[2].weight)
            return layers

        self.sizes = {"size": size, "rank": rank, "hidden": hidden}
        self.before, self.two_before = code(), code()     # phi, phi2: of the token before, of the token two before
        self.here = code(last_zero=True)                  # psi: of a candidate; starts at nothing, so the scores start as the first run's
        self.state, self.state_two = nn.Linear(size, rank), nn.Linear(size, rank)
        self.rest = nn.Linear(size, 1)
        nn.init.zeros_(self.rest.weight)
        nn.init.zeros_(self.rest.bias)
        self.on_path = nn.Parameter(torch.tensor(float(on_path)))     # weights of the counts' lift, as in CandidateScores
        self.off_path = nn.Parameter(torch.tensor(float(off_path)))

    def _unit(self, vectors: torch.Tensor) -> torch.Tensor:
        vectors = vectors.float()
        return vectors * torch.rsqrt(vectors.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def tables(self, states, before, two_before, here):
        """states [..., size]: the first run's states at places; before [..., B, size], two_before [..., A, size]: the
        input embeddings of the candidates of the place before and of the place two before; here [..., C, size]: the
        output-head rows of the place's candidates. Returns the change of every candidate's score after every
        candidate before ([..., B, C]) and after every candidate two before ([..., A, C]), and the change of the
        score of "another token" ([...])."""
        states = self._unit(states)
        coded = self.here(self._unit(here))                                              # [..., C, rank]
        pair = torch.einsum("...br,...cr->...bc", self.before(self._unit(before)) * self.state(states)[..., None, :], coded)
        skip = torch.einsum("...ar,...cr->...ac", self.two_before(self._unit(two_before)) * self.state_two(states)[..., None, :], coded)
        return pair, skip, self.rest(states)[..., 0]

    def forward(self, first_log, states, before, two_before, here, lift=None, off=None):
        """For places whose two tokens before are KNOWN (training, and measuring): first_log [..., C]: the first
        run's log-probabilities of the candidates; before, two_before [..., size]: the input embeddings of the two
        tokens; here [..., C, size]; lift [..., C], off [...] as in CandidateScores. Log-probabilities [..., C + 1]."""
        pair, skip, rest = self.tables(states, before[..., None, :], two_before[..., None, :], here)
        raised = first_log.float() + pair[..., 0, :] + skip[..., 0, :]
        if lift is not None:
            raised = raised + torch.where(off[..., None], self.off_path, self.on_path) * lift.float()
        rest = torch.log1p(-first_log.float().exp().sum(dim=-1).clamp(max=1.0 - 1e-6)) + rest
        return torch.cat((raised, rest[..., None]), dim=-1).log_softmax(dim=-1)


def save_pair_scores(path: str, scores: PairScores, notes: dict | None = None) -> None:
    torch.save({"sizes": scores.sizes, "state": {name: value.detach().half().cpu() for name, value in scores.state_dict().items()},
                "notes": notes or {}}, path)


def load_pair_scores(path: str, device) -> tuple[PairScores, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    scores = PairScores(**saved["sizes"])
    scores.load_state_dict({name: value.float() for name, value in saved["state"].items()})
    return scores.to(device).eval(), saved.get("notes", {})
