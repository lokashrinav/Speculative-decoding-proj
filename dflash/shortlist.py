"""The output head over a short list of tokens.

The drafter's output head is the target's: one row of weights for each of the vocabulary's 152,000
tokens. Finding the best token for a few dozen places reads all of them, over a millisecond on the
GPU however few the places are. A second run that is to cost a few milliseconds cannot spend that.
Most tokens are never the right one, so it looks only at a short list: the tokens most common in
answers (counted ahead of time on other answers), and, added for each call, the tokens of the text
so far and the first run's own choices, which is where rare tokens that do come up are found
(names, numbers, words of the prompt). Vocabulary pruning of the draft's head is FR-Spec's
(Zhao et al., arXiv 2502.14856); the tokens of the text itself are added here.

The choice among the listed tokens is exact. A token outside the list cannot be written, so the
second run is wrong where the right token is not listed; decoding stays lossless, the target checks.
"""
import os

import torch


class Shortlist:
    def __init__(self, head, tokens: torch.Tensor):
        """head: the output head (a linear layer without bias); tokens: the listed tokens' numbers."""
        if getattr(head, "bias", None) is not None:
            raise ValueError("the short list is made for an output head without bias")
        self.head = head
        self.tokens = tokens.to(head.weight.device).long()
        self.weight = head.weight[self.tokens].contiguous()                  # [listed, D]
        self.listed = torch.zeros(head.weight.shape[0], dtype=torch.bool, device=head.weight.device)
        self.listed[self.tokens] = True

    def best(self, states: torch.Tensor, also: torch.Tensor | None = None) -> torch.Tensor:
        """The best token at each place ([..., D] -> [...]) among the listed tokens and the tokens
        `also` (any shape; tokens already listed are left out). A drafter that scales or caps its
        scores does not change which token is best."""
        scores = states @ self.weight.T
        if also is not None:
            more = also.flatten()
            more = more[~self.listed[more]]
            if more.numel():
                scores = torch.cat((scores, states @ self.head.weight[more].T), dim=-1)
                return torch.cat((self.tokens, more))[scores.argmax(dim=-1)]
        return self.tokens[scores.argmax(dim=-1)]


def count_tokens(sequences, vocabulary: int) -> torch.Tensor:
    """How often each token occurs in the given token sequences: [vocabulary]."""
    counts = torch.zeros(vocabulary, dtype=torch.long)
    for sequence in sequences:
        counts += torch.bincount(sequence.long().flatten().cpu(), minlength=vocabulary)[:vocabulary]
    return counts


def common_tokens(path: str, most: int) -> torch.Tensor:
    """The `most` most common tokens of a saved count (see save_counts)."""
    counts = torch.load(path, map_location="cpu")["counts"]
    return counts.topk(min(most, counts.shape[0])).indices


def save_counts(path: str, counts: torch.Tensor, notes: dict | None = None) -> None:
    torch.save({"counts": counts.to(torch.int32), "notes": notes or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)
