"""Counts of which token the target writes after which two tokens.

DForkTree (arXiv 2611.12356) corrects a tree of drafts with such counts, taken from text the target
wrote itself: after an alternative token, the first run's guesses for the next places were made
without knowing that token, and the counts know which tokens tend to follow it. The correction for a
token c after the tokens (a, b) is how much more often the target wrote c after (a, b) than it writes
c anywhere (the logarithm of that ratio, clipped): `lift`.

The counts are smoothed as DForkTree's released code smooths them: a count of three tokens falls back
on the count of two, and that on the count of one, each with the weight of `strength` occurrences.
"""

import numpy as np
import torch


class FollowCounts:
    def __init__(self, vocabulary, single, pair_keys, pair_counts, before_one, triple_keys, triple_counts,
                 two_keys, two_counts, strength=20.0, clip=10.0):
        self.vocabulary, self.strength, self.clip = int(vocabulary), float(strength), float(clip)
        total = float(single.sum())
        self.tokens = int(total)
        # how likely a token is anywhere (every token counted once more, so that none has probability 0)
        self.anywhere = ((single.double() + 1.0) / (total + self.vocabulary)).float()
        self.pair_keys, self.pair_counts = pair_keys.long(), pair_counts.float()          # (b, c) -> times
        self.before_one = before_one.float()                                              # b -> times b had a token after it
        self.triple_keys, self.triple_counts = triple_keys.long(), triple_counts.float()  # (a, b, c) -> times
        self.two_keys, self.two_counts = two_keys.long(), two_counts.float()              # (a, b) -> times with a token after

    @classmethod
    def count(cls, texts, vocabulary, **more):
        """From texts [(tokens, start)]: every token from `start` on is counted with the two tokens before it
        (which may lie before `start`: the end of the prompt)."""
        firsts, seconds, thirds = [], [], []
        for tokens, start in texts:
            row = np.asarray(tokens, dtype=np.int64)
            start = max(int(start), 2)
            if len(row) <= start:
                continue
            thirds.append(row[start:])
            seconds.append(row[start - 1:-1])
            firsts.append(row[start - 2:-2])
        empty = np.zeros(0, dtype=np.int64)
        a, b, c = (np.concatenate(part) if part else empty for part in (firsts, seconds, thirds))
        size = int(vocabulary)
        pair_keys, pair_counts = np.unique(b * size + c, return_counts=True)
        triple_keys, triple_counts = np.unique((a * size + b) * size + c, return_counts=True)
        two_keys, two_counts = np.unique(a * size + b, return_counts=True)
        tensor = torch.from_numpy
        return cls(size, tensor(np.bincount(c, minlength=size)), tensor(pair_keys), tensor(pair_counts),
                   tensor(np.bincount(b, minlength=size)), tensor(triple_keys), tensor(triple_counts),
                   tensor(two_keys), tensor(two_counts), **more)

    def to(self, device):
        for name in ("anywhere", "pair_keys", "pair_counts", "before_one", "triple_keys", "triple_counts",
                     "two_keys", "two_counts"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sizes(self):
        return {"tokens_counted": self.tokens, "different_pairs": int(self.pair_keys.shape[0]),
                "different_triples": int(self.triple_keys.shape[0])}

    @staticmethod
    def _looked_up(keys, counts, asked):
        """The count of every asked key (0 for a key that never occurred)."""
        if keys.shape[0] == 0:
            return torch.zeros(asked.shape, dtype=torch.float32, device=asked.device)
        at = torch.searchsorted(keys, asked.reshape(-1)).clamp(max=keys.shape[0] - 1).reshape(asked.shape)
        return torch.where(keys[at] == asked, counts[at], torch.zeros((), dtype=counts.dtype, device=counts.device))

    def after_one(self, b, c):
        """How likely c is after b."""
        seen = self._looked_up(self.pair_keys, self.pair_counts, b * self.vocabulary + c)
        return (seen + self.strength * self.anywhere[c]) / (self.before_one[b] + self.strength)

    def after_two(self, a, b, c):
        """How likely c is after (a, b)."""
        seen = self._looked_up(self.triple_keys, self.triple_counts, (a * self.vocabulary + b) * self.vocabulary + c)
        times = self._looked_up(self.two_keys, self.two_counts, a * self.vocabulary + b)
        return (seen + self.strength * self.after_one(b, c)) / (times + self.strength)

    def lift(self, a, b, c):
        """log( how likely c is after (a, b) / how likely c is anywhere ), within +-clip. a, b, c: token tensors
        of shapes that broadcast against each other."""
        a, b, c = torch.broadcast_tensors(a.long(), b.long(), c.long())
        return (self.after_two(a, b, c).log() - self.anywhere[c].log()).clamp(-self.clip, self.clip)

    def _one_table(self):
        """The three tables of counts as one sorted list of keys (triples, then the pairs that had a token after
        them, then pairs), so that a round asks all its questions in one lookup."""
        if getattr(self, "_keys", None) is None or self._keys.device != self.anywhere.device:
            size = self.vocabulary
            self._after_triples, self._after_twos = size ** 3, size ** 3 + size ** 2
            self._keys = torch.cat((self.triple_keys, self.two_keys + self._after_triples, self.pair_keys + self._after_twos))
            self._values = torch.cat((self.triple_counts, self.two_counts, self.pair_counts))
            self._log_anywhere = self.anywhere.log()
        return self._keys, self._values

    def lift_of_places(self, a, b, c):
        """`lift` for a round's tree in few steps on the GPU. a: [places, A, 1, 1], the tokens two places back;
        b: [places, 1, B, 1], the tokens one place back; c: [places, 1, 1, C], the candidates. Gives [places, A, B, C]."""
        keys, values = self._one_table()
        if keys.shape[0] == 0:
            return self.lift(a, b, c)
        size, a, b, c = self.vocabulary, a.long(), b.long(), c.long()
        two = a * size + b                                   # [places, A, B, 1]
        pair = b * size + c                                  # [places, 1, B, C]
        triple = two * size + c                              # [places, A, B, C]
        asked = torch.cat((triple.reshape(-1), (two + self._after_triples).reshape(-1), (pair + self._after_twos).reshape(-1)))
        at = torch.searchsorted(keys, asked).clamp_(max=keys.shape[0] - 1)
        found = torch.where(keys[at] == asked, values[at], torch.zeros((), dtype=values.dtype, device=values.device))
        first, second = triple.numel(), triple.numel() + two.numel()
        seen_three, times_two, seen_pair = found[:first].view(triple.shape), found[first:second].view(two.shape), found[second:].view(pair.shape)
        after_one = (seen_pair + self.strength * self.anywhere[c]) / (self.before_one[b] + self.strength)
        after_two = (seen_three + self.strength * after_one) / (times_two + self.strength)
        return (after_two.log() - self._log_anywhere[c]).clamp_(-self.clip, self.clip)
