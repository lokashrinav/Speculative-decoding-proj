"""Training the drafter's adapters during a request: lossless, exact labels, and the same
in decoding and in replay."""

import time

import pytest
import torch

import dflash.redraft
from dflash.adapt import DraftTrainer, LowRank, attach_adapters, detach_adapters
from dflash.redraft import dflash_generate_redraft
from dflash.replay import replay_decode
from tests.test_redraft import make_models


@pytest.fixture(autouse=True)
def _cpu_clock(monkeypatch):
    monkeypatch.setattr(dflash.redraft, "_cuda_time", time.perf_counter)


def test_adapters_start_as_the_frozen_drafter_and_come_off_again():
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    plain = dflash_generate_redraft(draft, target, prompt, 120, None, return_stats=True)
    adapters = attach_adapters(draft, rank=4)
    assert len(adapters) == 5 * len(draft.layers) and all(isinstance(adapter, LowRank) for adapter in adapters)
    assert not any(parameter.requires_grad for name, parameter in draft.named_parameters() if ".base." in name)
    assert attach_adapters(draft, rank=4) == adapters                       # attaching twice adds nothing
    untrained = dflash_generate_redraft(draft, target, prompt, 120, None, return_stats=True)
    assert torch.equal(untrained.output_ids, plain.output_ids) and untrained.acceptance_lengths == plain.acceptance_lengths
    # A learning rate of zero is plain decoding.
    idle = DraftTrainer(draft, adapters, rate=0.0)
    idle.begin()
    same = dflash_generate_redraft(draft, target, prompt, 120, None, improver=idle, return_stats=True)
    assert same.acceptance_lengths == plain.acceptance_lengths and idle.totals["updates"] == 0
    detach_adapters(draft)
    assert not any(isinstance(module, LowRank) for module in draft.modules())
    back = dflash_generate_redraft(draft, target, prompt, 120, None, return_stats=True)
    assert back.acceptance_lengths == plain.acceptance_lengths


@pytest.mark.parametrize("labels", ["immediate", "hindsight", "both"])
def test_training_the_drafter_is_lossless_uses_exact_labels_and_replays(labels):
    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    budget = 300
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, return_stats=True)
    adapters = attach_adapters(draft, rank=4)
    trainer = DraftTrainer(draft, adapters, labels=labels, rate=3e-3)
    trainer.begin()
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=trainer, return_stats=True)
    assert torch.equal(live.output_ids, plain.output_ids)                    # the answer is still the target's own text
    assert live.acceptance_lengths != plain.acceptance_lengths               # ... but the drafts did change
    final = live.output_ids[0]
    width = draft.block_size - 1
    assert trainer.trained_on, "nothing was trained"
    starts = [prompt.shape[1]]
    for gained in live.acceptance_lengths:
        starts.append(starts[-1] + gained)
    seen = {}
    for position, first, used in trainer.trained_on:
        assert position in starts                                            # a past round's start
        assert final[position + 1 + first : position + 1 + first + len(used)].tolist() == used   # the target's tokens there
        for offset in range(first, first + len(used)):
            assert (position, offset) not in seen                            # no position of a round is trained on twice
            seen[(position, offset)] = True
    lengths = [len(used) for _, _, used in trainer.trained_on]
    if labels == "hindsight":
        assert set(lengths) == {width}                                       # whole blocks only, once they are known
    if labels == "immediate":
        accepted = dict(zip(starts, live.acceptance_lengths))
        assert all(len(used) == min(accepted[position], width) for position, _, used in trainer.trained_on)
    if labels == "both":
        whole = [position for position in starts[:-8] if all((position, offset) in seen for offset in range(width))]
        assert len(whole) >= len(starts) - 12                                # every round ends up fully used
    # A second answer starts from the frozen drafter again, so decoding it again gives the same rounds,
    # and replaying the finished answer follows them too.
    updates = trainer.totals["updates"]
    trainer.begin()
    assert all(float(adapter.up.abs().sum()) == 0.0 for adapter in adapters)
    second = dflash_generate_redraft(draft, target, prompt, budget, None, improver=trainer, return_stats=True)
    assert second.acceptance_lengths == live.acceptance_lengths and trainer.totals["updates"] == 2 * updates
    trainer.begin()
    replayed = replay_decode(draft, target, live.output_ids, prompt.shape[1], ended=False, improver=trainer)
    assert replayed.acceptance_lengths == live.acceptance_lengths
    detach_adapters(draft)


def test_stride_trains_on_every_other_round():
    target, draft = make_models()
    prompt = torch.tensor([[2, 7, 1, 8]])
    adapters = attach_adapters(draft, rank=4)
    every = DraftTrainer(draft, adapters, labels="hindsight", rate=1e-3)
    every.begin()
    dflash_generate_redraft(draft, target, prompt, 200, None, improver=every, return_stats=True)
    second = DraftTrainer(draft, adapters, labels="hindsight", rate=1e-3, stride=2)
    second.begin()
    dflash_generate_redraft(draft, target, prompt, 200, None, improver=second, return_stats=True)
    assert 0.35 * every.totals["updates"] < second.totals["updates"] < 0.65 * every.totals["updates"]
    detach_adapters(draft)
