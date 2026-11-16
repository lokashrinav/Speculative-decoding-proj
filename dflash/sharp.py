"""Sharper attention for long texts, without any training (the baseline SharpDraft, arXiv 2610.05106).

The drafter's attention is a softmax over every earlier token. The longer the text, the more
tokens share that softmax, and the less of it the few important ones get. SharpDraft makes up
for this by multiplying the drafter's queries by a number that grows with the number of tokens
it can see:

    scale = min(cap, 1 + log(max((keys - top) / (reference - top), 1)) / slope)

with top = 32, slope = 5, cap = e^0.5 and reference = 2048 for the Qwen3 drafters. Up to
`reference` visible tokens nothing changes. The same scale is used in every layer, after the
queries are normalized. Nothing is trained and the drafter's weights and its cache stay as
they are, so decoding is as lossless as before.

A fixed scale (the same at every length) is the paper's own control.
"""

import math


class QueryScale:
    """The number the drafter's queries are multiplied by when it sees `keys` tokens."""

    def __init__(self, reference=2048, top=32, slope=5.0, cap=math.exp(0.5), fixed=None):
        assert reference > top and slope > 0 and cap >= 1.0
        self.reference, self.top, self.slope, self.cap, self.fixed = reference, top, slope, cap, fixed

    def __call__(self, keys):
        if self.fixed is not None:
            return float(self.fixed)
        ratio = max((keys - self.top) / (self.reference - self.top), 1.0)
        return min(self.cap, 1.0 + math.log(ratio) / self.slope)

    def __repr__(self):
        if self.fixed is not None:
            return f"QueryScale(fixed={self.fixed})"
        return f"QueryScale(reference={self.reference}, top={self.top}, slope={self.slope}, cap={self.cap:.4f})"


def set_query_scale(model, rule=None):
    """Switches the rule on in every layer of the drafter (None switches it off)."""
    for layer in model.layers:
        layer.self_attn.query_scale = rule
