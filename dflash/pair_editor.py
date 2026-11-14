"""Editing a draft left to right, among the first pass's own top candidates.

The drafter picks every draft token at the same moment, so each token is chosen
without knowing which token was chosen just before it. This editor goes through
the draft left to right. At each position it re-picks among the first pass's top
K candidates, now knowing the token it has just chosen for the position before.
The edited draft replaces the first draft; the target checks only the edited one.

BigramTable is the simplest judge of which candidate fits: how often each token
followed each other token in the target's own answers. It is counted, not learned.
"""

from types import SimpleNamespace

import torch


class BigramTable:
    """Counts of (token, next token) in the target's own answers."""

    def __init__(self, keys, counts, unigram, previous, total, vocabulary):
        self.keys, self.counts, self.unigram, self.previous = keys, counts, unigram, previous
        self.total, self.vocabulary = float(total), int(vocabulary)
        self.prior = (unigram + 1.0) / (self.total + self.vocabulary)

    @classmethod
    def count(cls, sequences, vocabulary):
        """sequences: (token ids, prompt length) pairs; only the answer part is counted."""
        firsts, seconds = [], []
        for ids, prompt_length in sequences:
            start = max(int(prompt_length), 1)
            firsts.append(ids[start - 1 : -1])
            seconds.append(ids[start:])
        first, second = torch.cat(firsts).long(), torch.cat(seconds).long()
        keys, counts = torch.unique(first * vocabulary + second, return_counts=True)
        return cls(keys, counts.float(), torch.bincount(second, minlength=vocabulary).float(),
                   torch.bincount(first, minlength=vocabulary).float(), second.numel(), vocabulary)

    def save(self, path):
        torch.save({"keys": self.keys.cpu(), "counts": self.counts.cpu(), "unigram": self.unigram.cpu(),
                    "previous": self.previous.cpu(), "total": self.total, "vocabulary": self.vocabulary}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        saved = torch.load(path, map_location="cpu")
        return cls(saved["keys"], saved["counts"], saved["unigram"], saved["previous"],
                   saved["total"], saved["vocabulary"]).to(device)

    def to(self, device):
        return BigramTable(self.keys.to(device), self.counts.to(device), self.unigram.to(device),
                           self.previous.to(device), self.total, self.vocabulary)

    def association(self, previous_token, candidates, smoothing):
        """log p(candidate | previous token) - log p(candidate).
        previous_token: (...), candidates: (..., K). 0 when the previous token was never seen."""
        keys = previous_token[..., None] * self.vocabulary + candidates
        index = torch.searchsorted(self.keys, keys).clamp(max=self.keys.numel() - 1)
        pair = torch.where(self.keys[index] == keys, self.counts[index], torch.zeros((), device=keys.device))
        prior = self.prior[candidates]
        conditional = (pair + smoothing * prior) / (self.previous[previous_token][..., None] + smoothing)
        return conditional.log() - prior.log()


def walk(first_scores, pair_scores, top_tokens):
    """Pick one candidate per position, left to right.
    first_scores: (K,) scores of the first position's candidates, given the anchor.
    pair_scores: (positions - 1, K, K); [p, a, b] scores candidate b at position p + 1
        given that candidate a was chosen at position p.
    top_tokens: (positions, K) the candidates' token ids. Returns (positions,) token ids."""
    choice = int(first_scores.argmax())
    picks = [choice]
    for row in pair_scores.argmax(dim=-1).tolist():
        choice = row[choice]
        picks.append(choice)
    index = torch.tensor(picks, device=top_tokens.device)
    return top_tokens.gather(1, index[:, None])[:, 0]


class BigramImprover:
    """Plug-in for dflash_generate_redraft: the left-to-right editor with a BigramTable as judge.
    top_k: how many of the first pass's candidates each position may choose from.
    weight: how much the counts weigh against the first pass's own logits.
    margin: a draft token is changed only if another candidate beats it by this much."""

    def __init__(self, table, top_k=16, weight=0.25, smoothing=10.0, margin=0.0, replace=True):
        self.table, self.top_k, self.weight, self.smoothing = table, top_k, weight, smoothing
        self.margin, self.replace = margin, replace

    def start(self, first_drafter):
        pass

    def __call__(self, round_state):
        s = round_state
        logits = s.draft_logits[0].float()
        positions = logits.shape[0]
        if positions < 1 or s.block_ids.shape[1] != positions + 1:
            return None
        if self.table.keys.device != logits.device:
            self.table = self.table.to(logits.device)
        k = min(self.top_k, logits.shape[-1])
        top = logits.topk(k, dim=-1)
        # Row p holds the tokens that may have been chosen just before position p + 1:
        # the anchor for the first position, the previous position's candidates after that.
        previous = torch.cat((s.block_ids[0, :1].expand(1, k), top.indices[:-1]), dim=0)
        following = top.indices[:, None, :].expand(-1, k, -1)
        pair = top.values[:, None, :] + self.weight * self.table.association(previous, following, self.smoothing)
        pair[..., 0] += self.margin
        first, pair = pair[0, 0], pair[1:]
        edited = s.block_ids.clone()
        edited[0, 1:] = walk(first, pair, top.indices)
        return edited


def top_confidence(top_values):
    """A cheap reading of how sure the first pass was, from the candidates' logits alone:
    how far the best candidate is ahead of the 2nd, 4th, 8th and last one.
    top_values: (..., K) sorted from best to worst. Returns (..., 4)."""
    last = top_values.shape[-1] - 1
    picks = [min(1, last), min(3, last), min(7, last), last]
    return (top_values[..., :1] - top_values[..., picks]) / 5.0


class PairEditor(torch.nn.Module):
    """A learned judge for the left-to-right editor.

    For a draft position, the token chosen just before it, and each candidate, it adds
    a correction to the first pass's logit for that candidate. As in the first editor,
    the correction is a low-rank shift of the drafter's hidden state read through the
    frozen output head; here the shift depends on the token just chosen. It starts at
    zero, so an untrained editor leaves the draft exactly as it was.
    bigram_weight: also add the counted BigramTable association with this weight."""

    def __init__(self, hidden_size, rank=256, block_size=16, read_states=True, top_k=16,
                 bigram_weight=0.0, smoothing=10.0, confidence="full"):
        super().__init__()
        nn = torch.nn
        if confidence not in ("full", "fast"):
            raise ValueError("confidence must be 'full' (draft_confidence) or 'fast' (top_confidence)")
        self.config = dict(hidden_size=hidden_size, rank=rank, block_size=block_size, read_states=read_states,
                           top_k=top_k, bigram_weight=bigram_weight, smoothing=smoothing, confidence=confidence)
        self.confidence_kind = confidence
        self.read_states, self.top_k, self.bigram_weight, self.smoothing = read_states, top_k, bigram_weight, smoothing
        self.margin = 0.0
        self.state_norm = nn.LayerNorm(hidden_size)
        self.state_in = nn.Linear(hidden_size, rank, bias=False)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.token_in = nn.Linear(hidden_size, rank, bias=False)
        self.confidence_in = nn.Linear(4, rank)
        self.position = nn.Parameter(torch.zeros(block_size, rank))
        self.mix = nn.Sequential(nn.LayerNorm(rank), nn.Linear(rank, 2 * rank), nn.GELU(),
                                 nn.Linear(2 * rank, rank), nn.LayerNorm(rank))
        self.up = nn.Parameter(torch.zeros(hidden_size, rank))

    def confidence_input(self, full_confidence, top_values):
        """The confidence this editor was built to read, for the anchor and every position.
        full_confidence: draft_confidence() of the first pass, or None for a 'fast' editor.
        top_values: (rows, positions, K) the candidates' logits."""
        if self.confidence_kind == "full":
            return full_confidence
        fast = top_confidence(top_values)
        return torch.cat((torch.zeros_like(fast[:, :1]), fast), dim=1)

    def context(self, states, previous_embeddings, confidence, first_position):
        """What the editor knows when it decides a position: the drafter's state there, how
        sure the first pass was, and the token chosen just before.
        states: (rows, positions, hidden). previous_embeddings: (rows, positions, hidden), or
        (rows, positions, A, hidden) to try A possible previous tokens at once."""
        x = self.position[first_position : first_position + states.shape[1]] + self.confidence_in(
            confidence.to(self.position.dtype))
        if self.read_states:
            x = x + self.state_in(self.state_norm(states))
        token = self.token_in(self.token_norm(previous_embeddings))
        if token.dim() == x.dim() + 1:
            x = x[:, :, None, :]
        return self.mix(x + token)

    def correction(self, context, candidate_rows):
        """Logit corrections for the candidates. candidate_rows: (rows, positions, K, hidden),
        the frozen output head's rows for the candidates. Returns (rows, positions, K), or
        (rows, positions, A, K) when the context covers A possible previous tokens."""
        shift = context @ self.up.t()
        if shift.dim() == candidate_rows.dim():
            return shift @ candidate_rows.transpose(-1, -2)
        return (candidate_rows @ shift[..., None])[..., 0]

    def forced_scores(self, states, confidence, top_values, top_tokens, previous_tokens, embed, head_rows,
                      multiplier=1.0, table=None):
        """Scores of each position's candidates when the token before it is known
        (for training: the true previous token). states and confidence include the anchor position."""
        context = self.context(states[:, 1:], embed(previous_tokens), confidence[:, 1:], 1)
        scores = top_values + self.correction(context, head_rows(top_tokens)).float() * multiplier
        if table is not None and self.bigram_weight:
            scores = scores + self.bigram_weight * table.association(previous_tokens, top_tokens, self.smoothing)
        return scores

    def scores(self, states, confidence, top_values, top_tokens, anchor, embed, head_rows, multiplier=1.0, table=None):
        """Everything the walk needs. Returns first: (rows, K), the first position's candidates
        given the anchor, and pair: (rows, positions - 1, K, K), see walk()."""
        rows = head_rows(top_tokens)
        context = self.context(states[:, 1:2], embed(anchor)[:, None], confidence[:, 1:2], 1)
        first = top_values[:, 0] + self.correction(context, rows[:, :1])[:, 0].float() * multiplier
        context = self.context(states[:, 2:], embed(top_tokens[:, :-1]), confidence[:, 2:], 2)
        pair = top_values[:, 1:, None, :] + self.correction(context, rows[:, 1:]).float() * multiplier
        if table is not None and self.bigram_weight:
            first = first + self.bigram_weight * table.association(anchor, top_tokens[:, 0], self.smoothing)
            following = top_tokens[:, 1:, None, :].expand(-1, -1, top_tokens.shape[-1], -1)
            pair = pair + self.bigram_weight * table.association(top_tokens[:, :-1], following, self.smoothing)
        return first, pair


def walk_rows(first, pair, top_tokens, margin=0.0):
    """walk() for many blocks at once, on the device. A draft token (candidate 0) is
    changed only if another candidate beats it by more than the margin."""
    choice = (first + margin * torch.nn.functional.one_hot(torch.zeros((), dtype=torch.long, device=first.device),
                                                           first.shape[-1])).argmax(dim=1)
    picks = [choice]
    if pair.shape[1]:
        bonus = torch.zeros(pair.shape[-1], device=pair.device)
        bonus[0] = margin
        best = (pair + bonus).argmax(dim=-1)
        for slot in range(best.shape[1]):
            choice = best[:, slot].gather(1, choice[:, None])[:, 0]
            picks.append(choice)
    return top_tokens.gather(2, torch.stack(picks, dim=1)[..., None])[..., 0]


def save_pair_editor(editor, path, margin=0.0):
    torch.save({"config": editor.config, "margin": float(margin),
                "state": {k: v.detach().cpu() for k, v in editor.state_dict().items()}}, path)


def load_pair_editor(path, device, dtype):
    saved = torch.load(path, map_location="cpu")
    editor = PairEditor(**saved["config"])
    editor.load_state_dict(saved["state"])
    editor.margin = float(saved.get("margin", 0.0))
    return editor.to(device=device, dtype=dtype).eval()


class PairImprover:
    """Plug-in for dflash_generate_redraft: the left-to-right editor with a trained PairEditor
    as judge. The edited draft replaces the first draft.

    Decoding uses a short path: the output head and the token inputs are folded into the
    editor once, the confidence reuses the top candidates, all candidate pairs are scored
    in one pass and the walk itself runs on the CPU. reference() is the plain version."""

    def __init__(self, editor, table=None, margin=None, replace=True):
        self.editor, self.table, self.replace = editor, table, replace
        self.margin = float(editor.margin) if margin is None else float(margin)
        self.folded = None

    def start(self, first_drafter):
        pass

    @torch.no_grad()
    def _fold(self, s):
        from .model import _draft_value, _output_head, _raw_input_embeddings

        editor, head = self.editor, _output_head(s.target)
        dtype, device = editor.up.dtype, editor.up.device
        self.multiplier = float(_draft_value(s.first_drafter.config, "output_multiplier", 1.0))
        # Output head folded into the editor: the logit change of every token per unit of correction.
        self.folded = head.weight.to(dtype) @ editor.up
        # What each possible previous token contributes to the editor's input.
        vocabulary = head.weight.shape[0]
        rows = []
        for begin in range(0, vocabulary, 32768):
            ids = torch.arange(begin, min(begin + 32768, vocabulary), device=device)
            rows.append(editor.token_in(editor.token_norm(_raw_input_embeddings(s.target, ids, 1.0).to(dtype))))
        self.token_rows = torch.cat(rows)
        if self.table is not None:
            self.table = self.table.to(device)

    @torch.no_grad()
    def __call__(self, round_state):
        s, editor = round_state, self.editor
        fast = editor.confidence_kind == "fast"
        # A 'fast' editor never needs the whole vocabulary again after picking the candidates.
        logits = s.draft_logits[0] if fast else s.draft_logits[0].float()
        positions = logits.shape[0]
        if positions < 1 or s.block_ids.shape[1] != positions + 1:
            return None
        if self.folded is None:
            self._fold(s)
        dtype = editor.up.dtype
        k = min(editor.top_k, logits.shape[-1])
        top = logits.topk(k, dim=-1)
        if fast:
            top = SimpleNamespace(values=top.values.float(), indices=top.indices)
            confidence = top_confidence(top.values)
        else:
            # How sure the first pass was (the same four numbers as draft_confidence).
            total = logits.logsumexp(dim=-1)
            sure = (top.values[:, :2] - total[:, None]).exp()
            entropy = total - (logits.softmax(dim=-1) * logits).sum(dim=-1)
            confidence = torch.stack((sure[:, 0], sure[:, 1], sure[:, 0] - sure[:, 1], entropy / 5.0), dim=-1)
        x = editor.position[1 : positions + 1] + editor.confidence_in(confidence.to(dtype))
        if editor.read_states:
            x = x + editor.state_in(editor.state_norm(s.draft_states[0, 1:].to(dtype)))
        previous = torch.cat((s.block_ids[0, :1].expand(1, k), top.indices[:-1]), dim=0)
        context = editor.mix(x[:, None, :] + self.token_rows[previous])
        pair = top.values[:, None, :] + (context @ self.folded[top.indices].transpose(1, 2)).float() * self.multiplier
        if self.table is not None and editor.bigram_weight:
            following = top.indices[:, None, :].expand(-1, k, -1)
            pair = pair + editor.bigram_weight * self.table.association(previous, following, editor.smoothing)
        pair[..., 0] += self.margin
        edited = s.block_ids.clone()
        edited[0, 1:] = walk(pair[0, 0], pair[1:], top.indices)
        return edited

    @torch.no_grad()
    def reference(self, round_state):
        """The same edit computed the plain way, as in training and validation."""
        from .draft_editor import draft_confidence
        from .model import _draft_value, _output_head, _raw_input_embeddings

        s = round_state
        logits = s.draft_logits.float()
        positions = logits.shape[1]
        if positions < 1 or s.block_ids.shape[1] != positions + 1:
            return None
        dtype = s.draft_states.dtype
        top = logits.topk(min(self.editor.top_k, logits.shape[-1]), dim=-1)
        head = _output_head(s.target)
        table = None if self.table is None else self.table.to(logits.device)
        full = draft_confidence(s.draft_logits) if self.editor.confidence_kind == "full" else None
        first, pair = self.editor.scores(
            s.draft_states, self.editor.confidence_input(full, top.values).to(dtype), top.values, top.indices, s.block_ids[:, 0],
            lambda ids: _raw_input_embeddings(s.target, ids, 1.0).to(dtype), lambda ids: head.weight[ids].to(dtype),
            float(_draft_value(s.first_drafter.config, "output_multiplier", 1.0)), table,
        )
        edited = s.block_ids.clone()
        edited[:, 1:] = walk_rows(first, pair, top.indices, self.margin)
        return edited
