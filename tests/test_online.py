"""The editor trained during the answer: it must learn from hindsight labels, leave decoding
lossless, and behave the same in decoding and in replay."""

import time
from types import SimpleNamespace

import pytest
import torch

import dflash.redraft
from dflash.online import OnlineBank, Setting, summarize
from dflash.redraft import dflash_generate_redraft
from dflash.replay import replay_decode
from tests.test_redraft import make_models


@pytest.fixture(autouse=True)
def _cpu_clock(monkeypatch):
    monkeypatch.setattr(dflash.redraft, "_cuda_time", time.perf_counter)


def test_online_editor_learns_a_repeated_mistake_from_hindsight_labels():
    """A made-up drafter that is always wrong at one kind of position. The editor sees the same
    mistake again and again, with its label arriving a round later, and learns to fix it."""
    torch.manual_seed(0)
    hidden, vocab, width = 16, 12, 3
    kinds = torch.randn(2, hidden)                                  # two kinds of position, told apart by the state
    target = SimpleNamespace(lm_head=torch.nn.Linear(hidden, vocab, bias=False))
    length = 400
    kind_at = torch.tensor([(position * 7) % 5 < 2 for position in range(length)]).long()   # kind 1 at 40% of positions
    truth = torch.where(kind_at == 1, torch.tensor(7), torch.tensor(5))    # kind 1 wants token 7, the drafter always says 5
    logits = torch.full((vocab,), -8.0)
    logits[5], logits[7], logits[9] = 3.0, 1.0, -1.0
    settings = [Setting("hindsight", rank=4, rate=3e-3, steps=2), Setting("immediate", rank=4, rate=3e-3, steps=2, labels="immediate")]
    bank = OnlineBank(settings, top_k=3, batch=32, capacity=256, margins=(0.0, 1.0))
    bank.begin()
    start, rounds = 4, 0
    while start + 1 + width < length:
        positions = torch.arange(start + 1, start + 1 + width)
        states = torch.cat((torch.zeros(1, hidden), kinds[kind_at[positions]] + 0.05 * torch.randn(width, hidden)))
        block = torch.cat((truth[start : start + 1], torch.full((width,), 5)))
        assert bank(SimpleNamespace(start=start, draft_states=states[None], draft_logits=logits.expand(1, width, vocab),
                                    block_ids=block[None], accepted_ids=truth[None, :start], target=target)) is None
        accepted = int((block[1:] == truth[positions]).long().cumprod(0).sum())
        start += accepted + 1
        rounds += 1
    # Every stored label is the target's token at that position (index among the candidates 5, 7, 9).
    assert bank.size > 100 and set(bank.targets[: bank.size].tolist()) <= {0, 1}
    late = bank.log[-40:]
    for number in (0, 1):
        right = total = 0
        for entry in late:
            wanted = truth[entry["start"] + 1 : entry["start"] + 1 + width]
            right += int((entry["edited"][number, 0] == wanted).sum())
            total += width
        assert right / total > 0.95, (settings[number].name, right / total)       # the plain draft is right only 60% of the time
    answer = bank.finish(truth[None], ended=True)
    table = answer["table"]
    assert table.shape == (2, 2, 4, 4) and answer["loop_start"] is None
    everything = table[:, :, 3]
    assert float(everything[0, 0, 0]) == rounds and float(everything[0, 0, 2]) > 0.5 * float(everything[0, 0, 1])
    report = summarize(bank)
    assert report["hindsight"]["0.0"]["everything"]["more_tokens"] > 0.5
    # Hindsight labels cover every drafted position, immediate ones only up to the first rejection.
    assert int(bank.immediate[: bank.size].sum()) < bank.size
    bank.begin()
    assert bank.heads is None and bank.size == 0 and bank.log == []


@pytest.mark.parametrize("block_size", [4, 3])
def test_online_editor_is_lossless_and_replays_exactly(block_size):
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    plain = dflash_generate_redraft(draft, target, prompt, 80, None, block_size=block_size, return_stats=True)
    settings = [Setting("a", rank=4, rate=1e-2, steps=2), Setting("b", rank=4, rate=1e-2, labels="immediate")]

    # Only measuring: decoding is untouched, and replay records the same drafts.
    watching, again = OnlineBank(settings, top_k=8, batch=16, capacity=64), OnlineBank(settings, top_k=8, batch=16, capacity=64)
    live = dflash_generate_redraft(draft, target, prompt, 80, None, improver=watching, block_size=block_size, return_stats=True)
    assert torch.equal(live.output_ids, plain.output_ids) and live.acceptance_lengths == plain.acceptance_lengths
    replay_decode(draft, target, live.output_ids, prompt.shape[1], ended=False, improver=again, block_size=block_size)
    assert len(watching.log) == len(again.log) > 3
    for a, b in zip(watching.log, again.log):
        assert a["start"] == b["start"] and torch.equal(a["edited"], b["edited"])
    first, second = watching.finish(live.output_ids, ended=False), again.finish(live.output_ids, ended=False)
    assert torch.equal(first["table"], second["table"])
    final = live.output_ids[0]
    stored = watching.targets[: watching.size]
    assert watching.size > 10 and int((stored >= 0).sum()) > 0

    # Decoding with one setting's edited draft: the answer is still the target's own text,
    # and replay follows the same rounds.
    editing = OnlineBank(settings, top_k=8, batch=16, capacity=64, margins=(0.0, 0.5), replace_with=("a", 0.0))
    edited = dflash_generate_redraft(draft, target, prompt, 80, None, improver=editing, block_size=block_size, return_stats=True)
    assert torch.equal(edited.output_ids, plain.output_ids)
    replayed = replay_decode(draft, target, edited.output_ids, prompt.shape[1], ended=False, block_size=block_size,
                             improver=OnlineBank(settings, top_k=8, batch=16, capacity=64, margins=(0.0, 0.5), replace_with=("a", 0.0)))
    assert replayed.acceptance_lengths == edited.acceptance_lengths and replayed.rounds_edited == edited.rounds_edited
