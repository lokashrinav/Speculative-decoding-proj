"""What the draft editor is rewarded for, and its keep-unless-sure rule."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nexus"))

from dflash.draft_editor import draft_confidence, keep_unless_sure  # noqa: E402
from train_editor2 import parse_variants, reward_loss  # noqa: E402


def test_expected_accepted_length_counts_tokens_in_a_row():
    live = torch.ones(1, 15)
    certain = torch.zeros(1, 15)
    assert reward_loss("eal", certain, live).item() == -15.0
    # A coin flip on the third token: two certain tokens, then half of the remaining thirteen.
    third = certain.clone()
    third[0, 2] = math.log(0.5)
    assert abs(reward_loss("eal", third, live).item() + (2 + 0.5 * 13)) < 1e-5
    # A hopeless first token makes everything after it worthless, however good it is.
    hopeless = certain.clone()
    hopeless[0, 0] = math.log(1e-6)
    assert reward_loss("eal", hopeless, live).item() > -1e-4


def test_text_that_ended_is_not_counted():
    live = torch.ones(1, 15)
    live[0, 4:] = 0
    garbage_after_end = torch.full((1, 15), -9.0)
    garbage_after_end[0, :4] = 0.0
    assert reward_loss("eal", garbage_after_end, live).item() == -4.0
    for kind in ("ce", "tv", "pce"):
        assert abs(reward_loss(kind, garbage_after_end, live).item()) < 1e-6


def test_prefix_reward_ignores_tokens_after_a_sure_miss():
    live = torch.ones(1, 15)
    log_right = torch.full((1, 15), math.log(0.5))
    log_right[0, 0] = math.log(1e-9)
    loss = reward_loss("pce", log_right, live).item()
    assert abs(loss - (-math.log(1e-9))) < 1e-4  # only the first token is chased
    everywhere = reward_loss("ce", log_right, live).item()
    assert everywhere > 0


def test_rewards_have_gradients_that_point_the_right_way():
    live = torch.ones(2, 15)
    for kind in ("ce", "tv", "pce", "eal"):
        log_right = torch.full((2, 15), math.log(0.4), requires_grad=True)
        reward_loss(kind, log_right, live).backward()
        assert (log_right.grad <= 0).all() and log_right.grad[:, 0].abs().min() > 0


def test_keep_unless_sure():
    logits = torch.tensor([[[0.0, 1.0, 3.0], [2.0, 0.0, 2.5], [5.0, 0.0, 0.0]]])
    draft = torch.tensor([[1, 0, 0]])
    assert keep_unless_sure(logits, draft, 0.0).tolist() == [[2, 2, 0]]
    assert keep_unless_sure(logits, draft, 1.0).tolist() == [[2, 0, 0]]  # the 0.5 edge is not enough
    assert keep_unless_sure(logits, draft, 10.0).tolist() == [[1, 0, 0]]


def test_draft_confidence_shape_and_anchor():
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    confidence = draft_confidence(logits)
    assert confidence.shape == (1, 3, 4)
    assert confidence[0, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert confidence[0, 1, 0] > 0.99 and confidence[0, 1, 3] < 0.01
    assert abs(confidence[0, 2, 0].item() - 1 / 3) < 1e-6 and abs(confidence[0, 2, 2].item()) < 1e-6


def test_parse_variants():
    variants = dict(parse_variants("a:loss=ce,mix=0.5;b:width=512,layers=4,states=0;c"))
    assert variants["a"]["loss"] == "ce" and variants["a"]["mix"] == 0.5
    assert variants["b"]["width"] == 512 and variants["b"]["states"] == 0 and variants["b"]["loss"] == "eal"
    assert variants["c"]["lr"] == 1e-3
