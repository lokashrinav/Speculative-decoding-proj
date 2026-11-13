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


def _random_table(vocabulary=40, seed=0):
    from dflash.pair_editor import BigramTable

    generator = torch.Generator().manual_seed(seed)
    sequences = [(torch.randint(0, vocabulary, (60,), generator=generator), 5) for _ in range(30)]
    return BigramTable.count(sequences, vocabulary)


def test_bigram_association_matches_counts():
    table = _random_table()
    previous, candidate = 3, 7
    pair = sum(1 for key in [previous * 40 + candidate] if key in set(table.keys.tolist()))
    count = float(table.counts[table.keys.tolist().index(previous * 40 + candidate)]) if pair else 0.0
    prior = (float(table.unigram[candidate]) + 1.0) / (table.total + 40)
    expected = math.log((count + 10.0 * prior) / (float(table.previous[previous]) + 10.0)) - math.log(prior)
    got = table.association(torch.tensor([previous]), torch.tensor([[candidate, 1]]), 10.0)[0, 0].item()
    assert abs(got - expected) < 1e-5


def test_left_to_right_editor_matches_a_step_by_step_walk():
    """The decoding-time editor scores all candidate pairs at once; it must choose
    exactly what a plain step-by-step walk chooses."""
    from types import SimpleNamespace

    from dflash.pair_editor import BigramImprover

    table = _random_table()
    generator = torch.Generator().manual_seed(1)
    for positions in (1, 2, 15):
        for k, weight, margin in ((4, 0.5, 0.0), (16, 1.0, 0.0), (8, 2.0, 0.7)):
            logits = torch.randn(1, positions, 40, generator=generator) * 2
            block = torch.cat((torch.randint(0, 40, (1, 1), generator=generator), logits.argmax(dim=-1)), dim=1)
            improver = BigramImprover(table, top_k=k, weight=weight, smoothing=10.0, margin=margin)
            edited = improver(SimpleNamespace(draft_logits=logits, block_ids=block))
            previous, expected = block[0, :1], []
            top = logits[0].topk(k, dim=-1)
            for slot in range(positions):
                score = top.values[slot] + weight * table.association(previous, top.indices[slot][None], 10.0)[0]
                score[0] += margin
                previous = top.indices[slot][score.argmax()][None]
                expected.append(int(previous))
            assert edited[0, 1:].tolist() == expected
            assert edited[0, 0] == block[0, 0]
    untouched = BigramImprover(table, weight=0.0)(SimpleNamespace(draft_logits=logits, block_ids=block))
    assert torch.equal(untouched, block)
    careful = BigramImprover(table, weight=5.0, margin=1e9)(SimpleNamespace(draft_logits=logits, block_ids=block))
    assert torch.equal(careful, block)


def test_bigram_table_round_trip(tmp_path):
    from dflash.pair_editor import BigramTable

    table = _random_table()
    table.save(tmp_path / "table.pt")
    again = BigramTable.load(tmp_path / "table.pt")
    previous, candidates = torch.tensor([1, 2]), torch.tensor([[3, 4], [5, 6]])
    assert torch.equal(table.association(previous, candidates, 10.0), again.association(previous, candidates, 10.0))
