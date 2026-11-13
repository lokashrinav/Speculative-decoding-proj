"""Editing a draft left to right, among the first pass's own top candidates.

The drafter picks every draft token at the same moment, so each token is chosen
without knowing which token was chosen just before it. This editor goes through
the draft left to right. At each position it re-picks among the first pass's top
K candidates, now knowing the token it has just chosen for the position before.
The edited draft replaces the first draft; the target checks only the edited one.

BigramTable is the simplest judge of which candidate fits: how often each token
followed each other token in the target's own answers. It is counted, not learned.
"""

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
        table, weight, smoothing = self.table, self.weight, self.smoothing
        first = top.values[0] + weight * table.association(s.block_ids[0, :1], top.indices[:1], smoothing)[0]
        first[0] += self.margin
        if positions > 1:
            following = top.indices[1:, None, :].expand(-1, k, -1)
            pair = top.values[1:, None, :] + weight * table.association(top.indices[:-1], following, smoothing)
            pair[..., 0] += self.margin
        else:
            pair = logits.new_zeros((0, k, k))
        edited = s.block_ids.clone()
        edited[0, 1:] = walk(first, pair, top.indices)
        return edited
