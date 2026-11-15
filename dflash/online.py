"""An editor that is trained while the answer is being written (plan item E14).

The drafter stays frozen. A small correction on top of it starts at zero for every
request and is trained during the request on the drafter's own drafts: each drafted
position gives one example (the drafter's state there, its candidates, and, once the
answer has passed that position, the token the target committed there). The label
is exact for every drafted position, also for those after a round's first rejection,
because the drafter's state at a position depends only on the committed context.

The correction is a low-rank score for each of the drafter's top candidates:

    score(candidate) = drafter log-probability + (V state) . (A row(candidate))

where row(candidate) is the candidate's row of the target's output layer. V starts
at zero, so the first drafts are the drafter's own. A draft token is changed only if
another candidate's score beats the draft token's score by a margin.

OnlineBank trains several settings side by side on the same stream of examples and
records what each would have drafted in every round. Used that way it changes
nothing in decoding, and finish() scores every setting at the same rounds. With
replace_with set, it returns one setting's edited draft instead, so that decoding
(or its replay) follows that editor.

Two kinds of examples can be told apart:
  "immediate"  positions up to a round's first rejection: their labels are known as
               soon as the round has been checked
  "hindsight"  every drafted position, labelled when the answer reaches it
"""

from collections import defaultdict
from dataclasses import dataclass

import torch

from .hindsight import accepted_length, loop_start

GROUPS = (("0-2k", 0, 2000), ("2k+", 2000, 10**9), ("8k+", 8000, 10**9), ("everything", 0, 10**9))


@dataclass(frozen=True)
class Setting:
    name: str
    rank: int = 32
    rate: float = 1e-3        # Adam learning rate
    steps: int = 1            # gradient steps per round
    labels: str = "hindsight"  # or "immediate"
    decay: float = 0.0        # weight decay, pulling the correction back towards zero


class _Head(torch.nn.Module):
    def __init__(self, hidden, rank, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.state = torch.nn.Parameter(torch.zeros(rank, hidden))
        self.token = torch.nn.Parameter(torch.randn(rank, hidden, generator=generator) / hidden ** 0.5)

    def forward(self, states, rows):
        """states (n, hidden), rows (n, candidates, hidden) -> (n, candidates) score to add."""
        return torch.einsum("nr,nkr->nk", states @ self.state.t(), rows @ self.token.t())


class OnlineBank:
    """Plug-in for dflash_generate_redraft and replay_decode. Call begin() before each answer
    and finish(output_ids, ended) after it."""

    def __init__(self, settings, top_k=16, batch=256, capacity=16384, margins=(0.0, 0.5, 1.0, 2.0),
                 replace_with=None, seed=0):
        self.settings, self.top_k, self.batch, self.capacity = list(settings), top_k, batch, capacity
        self.margins, self.seed = tuple(margins), seed
        self.replace_with = replace_with            # (setting name, margin): decode with that editor's draft
        self.replace = replace_with is not None
        self.totals = None                          # (settings, margins, groups, 4): rounds, tokens, gained, changed
        self.answers = 0
        self.update_seconds = defaultdict(float)
        self.begin()

    def start(self, first_drafter):
        pass

    def begin(self):
        """Forget everything learned: every request starts from the frozen drafter."""
        self.heads = None
        self.pending, self.log = [], []
        self.size = self.pointer = self.round = 0
        self.prompt_length = None
        self.generator = None

    # ---- examples -----------------------------------------------------------------------------

    def _setup(self, states, head_weight):
        device, hidden = states.device, states.shape[-1]
        self.rows = head_weight                      # the target's output layer (not copied)
        self.row_scale = float(head_weight[:4096].float().pow(2).mean().sqrt())
        self.heads, self.optimizers = [], []
        for number, setting in enumerate(self.settings):
            head = _Head(hidden, setting.rank, self.seed + number).to(device)
            self.heads.append(head)
            self.optimizers.append(torch.optim.Adam(head.parameters(), lr=setting.rate, weight_decay=setting.decay))
        self.states = torch.zeros((self.capacity, hidden), dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                  device=device)
        self.candidates = torch.zeros((self.capacity, self.top_k), dtype=torch.long, device=device)
        self.scores = torch.zeros((self.capacity, self.top_k), dtype=torch.float32, device=device)
        self.targets = torch.full((self.capacity,), -1, dtype=torch.long, device=device)
        self.immediate = torch.zeros((self.capacity,), dtype=torch.bool, device=device)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

    @staticmethod
    def _unit(states):
        """States scaled to length one, so that one learning rate suits every model."""
        states = states.float()
        return states / states.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _candidate_rows(self, candidates):
        return self.rows[candidates].float() / self.row_scale

    def _label(self, s, start):
        """Examples whose position the answer has now passed get their label and enter the buffer.
        Returns the buffer slots just filled."""
        waiting, filled, committed = [], [], None
        for batch in self.pending:
            ready = min(batch["count"], start - batch["first"] + 1)
            if ready <= 0:
                waiting.append(batch)
                continue
            if committed is None:
                committed = torch.cat((s.accepted_ids[0], s.block_ids[0, :1]))      # positions 0 .. start
            labels = committed[batch["first"] : batch["first"] + ready]
            hit = batch["candidates"][:ready] == labels[:, None]
            slots = (self.pointer + torch.arange(ready, device=labels.device)) % self.capacity
            self.states[slots] = batch["states"][:ready].to(self.states.dtype)
            self.candidates[slots] = batch["candidates"][:ready]
            self.scores[slots] = batch["scores"][:ready]
            self.targets[slots] = torch.where(hit.any(dim=1), hit.float().argmax(dim=1), torch.full_like(labels, -1))
            self.immediate[slots] = self.round - batch["round"] == 1                # labelled by the very next round
            self.pointer = int((self.pointer + ready) % self.capacity)
            self.size = min(self.size + ready, self.capacity)
            filled.append(slots)
            if ready < batch["count"]:
                waiting.append({key: (value[ready:] if torch.is_tensor(value) else value) for key, value in batch.items()}
                               | {"first": batch["first"] + ready, "count": batch["count"] - ready})
        self.pending = waiting
        return torch.cat(filled) if filled else None

    def _train(self, new_slots):
        """One round of training for every setting: the newly labelled examples plus a random
        sample of older ones. Settings that use the same kind of labels see the same examples."""
        if self.size == 0:
            return
        usable = self.targets[: self.size] >= 0
        pools = {"hindsight": usable.nonzero()[:, 0], "immediate": (usable & self.immediate[: self.size]).nonzero()[:, 0]}
        fresh = {"hindsight": None, "immediate": None}
        if new_slots is not None:
            keep = self.targets[new_slots] >= 0
            fresh["hindsight"] = new_slots[keep]
            fresh["immediate"] = new_slots[keep & self.immediate[new_slots]]
        for labels, pool in pools.items():
            members = [number for number, setting in enumerate(self.settings) if setting.labels == labels]
            if not members or pool.numel() == 0:
                continue
            new = fresh[labels]
            for step in range(max(self.settings[number].steps for number in members)):
                more = max(self.batch - (0 if new is None else new.numel()), 1)
                drawn = pool[torch.randint(0, pool.numel(), (min(more, pool.numel()),), device=pool.device, generator=self.generator)]
                rows = drawn if new is None or new.numel() == 0 else torch.cat((new[: self.batch], drawn))
                states, candidate_rows = self._unit(self.states[rows]), self._candidate_rows(self.candidates[rows])
                scores, targets = self.scores[rows], self.targets[rows]
                for number in members:
                    if self.settings[number].steps <= step:
                        continue
                    with torch.enable_grad():
                        loss = torch.nn.functional.cross_entropy(scores + self.heads[number](states, candidate_rows), targets)
                        self.optimizers[number].zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizers[number].step()

    # ---- one round ----------------------------------------------------------------------------

    def __call__(self, round_state):
        with torch.inference_mode(False), torch.no_grad():
            return self._round(round_state)

    def _round(self, s):
        start = int(s.start)
        if self.prompt_length is None:
            self.prompt_length = start
        states = s.draft_states[0, 1:].detach().clone()
        log_probs = s.draft_logits[0].detach().float().log_softmax(dim=-1)
        best = log_probs.topk(min(self.top_k, log_probs.shape[-1]), dim=-1)
        candidates, scores = best.indices.clone(), best.values.clone()
        if self.heads is None:
            from .model import _output_head

            self._setup(states, _output_head(s.target).weight.detach())
        clock = _clock(states)
        self._train(self._label(s, start))
        self.update_seconds["train"] += _clock(states) - clock
        self.update_seconds["rounds"] += 1

        clock = _clock(states)
        draft = s.block_ids[0, 1:].clone()
        unit, rows = self._unit(states), self._candidate_rows(candidates)
        edited = []
        for head in self.heads:
            total = scores + head(unit, rows)
            pick = total.argmax(dim=-1)
            lead = total.gather(1, pick[:, None])[:, 0] - total[:, 0]      # the draft token is the first candidate
            tokens = candidates.gather(1, pick[:, None])[:, 0]
            edited.append(torch.stack([torch.where(lead > margin, tokens, draft) for margin in self.margins]))
        self.update_seconds["edit"] += _clock(states) - clock
        self.log.append(dict(start=start, draft=draft, edited=torch.stack(edited)))       # (settings, margins, width)
        self.pending.append(dict(states=states, candidates=candidates, scores=scores, first=start + 1,
                                 count=states.shape[0], round=self.round))
        self.round += 1
        if self.replace_with is None:
            return None
        name, margin = self.replace_with
        chosen = edited[[setting.name for setting in self.settings].index(name)][self.margins.index(margin)]
        return torch.cat((s.block_ids[:, :1], chosen[None]), dim=1)

    # ---- scoring a finished answer ------------------------------------------------------------

    @torch.no_grad()
    def finish(self, output_ids, ended=True):
        """Scores every setting and margin at the rounds that were decoded: tokens gained over the
        first draft, by position in the answer, on rounds outside endless repetition.
        Returns this answer's table (settings, margins, groups, 4) and where repetition began."""
        final = output_ids[0]
        shape = (len(self.settings), len(self.margins), len(GROUPS), 4)
        answer = {"loop_start": None, "table": torch.zeros(shape, dtype=torch.float64)}
        if not self.log:
            return answer
        if not ended:
            answer["loop_start"] = loop_start(final[self.prompt_length :].tolist())
        device = final.device
        width = max(entry["draft"].shape[0] for entry in self.log)
        pad = lambda value, fill: torch.nn.functional.pad(value, (0, width - value.shape[-1]), value=fill)
        draft = torch.stack([pad(entry["draft"], -2) for entry in self.log])                    # (rounds, width)
        edited = torch.stack([pad(entry["edited"], -2) for entry in self.log])                  # (rounds, settings, margins, width)
        starts = torch.tensor([entry["start"] for entry in self.log], dtype=torch.long, device=device)
        positions = starts[:, None] + 1 + torch.arange(width, device=device)[None, :]
        valid = (positions < final.shape[0]) & (draft >= 0)
        truth = torch.where(valid, final[positions.clamp(max=final.shape[0] - 1)], torch.full_like(positions, -1))
        plain = accepted_length(draft, truth)
        rounds = draft.shape[0]
        flat = edited.reshape(rounds, -1, width)
        gained = torch.stack([accepted_length(flat[:, column], truth) for column in range(flat.shape[1])], dim=1) - plain[:, None]
        changed = (flat != draft[:, None, :]).any(dim=-1)
        clean = torch.ones(rounds, dtype=torch.bool, device=device)
        if answer["loop_start"] is not None:
            clean = starts < self.prompt_length + answer["loop_start"]
        where = starts - self.prompt_length
        table = torch.zeros((flat.shape[1], len(GROUPS), 4), dtype=torch.float64)
        for number, (_, low, high) in enumerate(GROUPS):
            rows = clean & (where >= low) & (where < high)
            table[:, number, 0] = float(rows.sum())
            table[:, number, 1] = float((plain[rows] + 1).sum())
            table[:, number, 2] = gained[rows].double().sum(dim=0).cpu()
            table[:, number, 3] = changed[rows].double().sum(dim=0).cpu()
        answer["table"] = table.reshape(shape)
        self.totals = answer["table"] if self.totals is None else self.totals + answer["table"]
        self.answers += 1
        return answer


def summarize(bank):
    """More tokens than the first draft for every setting and margin, by position in the answer."""
    out = {}
    if bank.totals is None:
        return out
    for number, setting in enumerate(bank.settings):
        rows = {}
        for column, margin in enumerate(bank.margins):
            rows[str(margin)] = {
                group: {"more_tokens": float(bank.totals[number, column, index, 2] / bank.totals[number, column, index, 1]),
                        "rounds_changed": float(bank.totals[number, column, index, 3] / bank.totals[number, column, index, 0])}
                for index, (group, _, _) in enumerate(GROUPS) if bank.totals[number, column, index, 0] > 0}
        out[setting.name] = rows
    return out


def _clock(tensor):
    import time

    if tensor.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter()
