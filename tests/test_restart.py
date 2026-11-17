"""The probe of restarts after the break (nexus/restart_probe.py)."""
import pytest
import torch

from dflash.replay import _matches, replay_decode
from nexus.restart_probe import ARMS, COLUMNS, Restart, restarts, summary
from tests.test_branches import BLOCK, _cpu_clock, answer, make_models  # noqa: F401  (_cpu_clock: a fixture for every test)


class Watch:
    """Restarts the drafter after the break with the target's real states, and keeps every round's
    first draft."""

    def __init__(self, ids, features):
        self.ids, self.features, self.first, self.restarted = ids, features, {}, {}

    def start(self, drafter):
        pass

    def __call__(self, s):
        width = s.block_ids.shape[1]
        self.first[s.start] = s.block_ids[0, 1:].clone()
        truth = self.ids[:, s.start + 1 : s.start + width]
        broke = _matches(s.block_ids[:, 1:], truth)
        restart = s.start + broke + 1
        if width == BLOCK and truth.shape[1] == BLOCK - 1 and broke < BLOCK - 1 and restart + BLOCK <= self.ids.shape[1]:
            before = s.draft_cache.get_seq_length()
            states = self.features[:, s.start : restart]
            places = torch.arange(s.start, restart)
            both = restarts(s, truth[0, broke], [restart, restart], [(states, places), None], BLOCK)
            assert s.draft_cache.get_seq_length() == before                   # the drafter's cache is left as it was
            self.restarted[restart] = both
        return None


def test_a_restart_with_the_targets_states_is_the_next_rounds_first_draft():
    """After the break plain decoding starts its next round at the right token, with the target's
    states of everything before it. A restart there, given those states as stand-ins, must write
    that round's first draft; without them it writes something else."""
    from dflash.replay import target_features

    target, draft = make_models()
    plain, length = answer(target, draft, budget=160)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    watch = Watch(ids, features)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=watch, block_size=BLOCK)
    same = other = 0
    for restart, both in watch.restarted.items():
        if restart in watch.first and watch.first[restart].shape[0] == BLOCK - 1:
            assert torch.equal(both[0], watch.first[restart])
            same += 1
            other += int(not torch.equal(both[1], watch.first[restart]))
    assert same > 10 and other > 3


def test_the_probe_counts_what_the_target_accepts_after_the_break():
    from dflash.replay import target_features

    target, draft = make_models()
    plain, length = answer(target, draft, budget=160)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    probe = Restart(k=3, width=BLOCK)
    probe.begin(ids, length, features, None, answer=0)
    result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=probe, block_size=BLOCK)
    table = torch.tensor(probe.rows)
    column = {name: table[:, COLUMNS.index(name)] for name in COLUMNS}
    assert len(table) > 10 and table.shape[1] == len(COLUMNS)
    # The plug-in returns no draft: the rounds are plain decoding's.
    assert result.acceptance_lengths == replay_decode(draft, target, ids, length, ended=False, block_size=BLOCK).acceptance_lengths
    # With the target's states the restart is the next round's first draft: its count is that round's.
    accepted, at = {}, length
    for produced in result.acceptance_lengths:
        accepted[at] = produced - 1
        at += produced
    checked = 0
    for row in probe.rows:
        record = dict(zip(COLUMNS, row))
        following = length + record["where"] + record["broke_after"] + 1
        if following in accepted and following + BLOCK < ids.shape[1]:
            assert record["real"] == accepted[following]
            checked += 1
    assert checked > 10
    for name in ARMS + ("old_rest", "new_rest"):
        assert int(column[name].min()) >= 0 and int(column[name].max()) <= BLOCK - 1
    first = column["broke_after"] == 0                       # one state missing: the nearest is all there is
    assert first.any() and torch.equal(column["only_nearest"][first], column["real"][first])
    assert torch.equal(column["all_but_nearest"][first], column["none"][first])
    assert not torch.equal(column["noisy200"], column["real"]) or not torch.equal(column["none"], column["real"])
    out = summary(probe.rows)
    assert out["rounds"] == len(table) and abs(out["all"]["real"] - float(column["real"].float().mean())) < 1e-3
