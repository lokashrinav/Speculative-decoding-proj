"""The probe of restarts after the break (nexus/restart_probe.py)."""
import pytest
import torch

from dflash.replay import _matches, replay_decode
from dflash.redraft import restarts
from nexus.restart_probe import ARMS, COLUMNS, Restart, summary
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
            both = restarts(s, [truth[0, broke]] * 2, [restart, restart], [(states, places), None], BLOCK)
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


def test_the_probe_restarts_from_the_last_right_token_too():
    """reanchor: a block from the last token the first draft got right, with the target's states of
    the tokens before it, is the first draft of a round that starts at that token; the right token's
    place among the drafter's choices is kept."""
    from dflash.replay import target_features

    target, draft = make_models()
    plain, length = answer(target, draft, budget=160)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    probe = Restart(k=3, width=BLOCK, reanchor=True)
    probe.begin(ids, length, features, None, answer=0)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=probe, block_size=BLOCK)
    assert probe.columns[: len(COLUMNS)] == COLUMNS and probe.columns[len(COLUMNS) :] == ("reanchor", "reanchor_noisy100", "rank")
    checked = at_once = 0
    for row in probe.rows:
        record = dict(zip(probe.columns, row))
        assert record["rank"] >= 1 and (record["rank"] == 1) == bool(record["second_choice_right"])
        assert 0 <= record["reanchor"] <= BLOCK - 1 and 0 <= record["reanchor_noisy100"] <= BLOCK - 1
        last_right = length + record["where"] + record["broke_after"]
        if record["broke_after"] == 0:
            assert record["reanchor"] == 0                     # the round's own first token: the first draft again
            at_once += 1
        elif checked < 12 and last_right + 1 + BLOCK < ids.shape[1]:
            from_there = replay_decode(draft, target, ids, last_right, ended=False, features=features, block_size=BLOCK)
            assert record["reanchor"] == from_there.acceptance_lengths[0] - 1
            checked += 1
    assert checked > 5 and at_once > 0
    out = summary(probe.rows, probe.columns)
    both = out["all"]["from_the_break_on"]["real"]
    assert both["the_better_of_the_two"] >= max(both["from_the_second_choice"], both["from_the_last_right_token"]) - 1e-9
    assert abs(out["all"]["reanchor"] - both["from_the_last_right_token"]) < 1e-3
    places = out["right_token_is_the_drafters"]
    assert abs(places["second_choice"] - out["second_choice_right"]) < 1e-3 and sum(places.values()) <= 1 + 1e-6
    assert "rank" not in out["all"]
    assert "from_the_break_on" not in summary(probe.rows[:0], probe.columns)


class Seen:
    """Keeps what a plug-in returned in every round, with the round's first draft."""

    def __init__(self, inner):
        self.inner, self.rounds = inner, []
        self.alternative_tail = getattr(inner, "alternative_tail", None)

    def start(self, drafter):
        self.inner.start(drafter)

    def __call__(self, s):
        rows = self.inner(s)
        self.rounds.append((s.start, s.block_ids[0].clone(), s.draft_logits[0].clone(), rows.clone(), list(self.inner.kept)))
        return rows


def test_the_restart_second_run_in_the_replay():
    """Restarts after the likely breaks: the rows are [the first draft up to the break, the second
    choice, a new block's tokens]; the replay counts the longest row that is the text, also past the
    block; with the target's real states a restart after a right second choice is the next round's
    first draft."""
    from dflash.redraft import Branches, Restarts, dflash_generate_redraft
    from dflash.replay import target_features

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    watch = Watch(ids, features)
    chain = replay_decode(draft, target, ids, length, ended=False, features=features, improver=watch, block_size=BLOCK)

    real = Restarts(k=3, stand_in="real")
    real.features = features
    seen = Seen(real)
    result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen, block_size=BLOCK)
    assert sum(result.acceptance_lengths) == sum(chain.acceptance_lengths) and result.rounds < chain.rounds
    at, past_the_block, as_next_round = length, 0, 0
    for (start, block, logits, rows, kept), produced in zip(seen.rounds, result.acceptance_lengths):
        width = block.shape[0]                               # the last rounds of a text have a shorter block
        assert start == at and rows.shape == (len(kept), 2 * width)
        second = logits.topk(2, dim=-1).indices[:, 1]
        truth = ids[0, start + 1 : start + 2 * width].tolist()
        best = 0
        for row, broke in zip(rows.tolist(), kept):
            assert row[: broke + 1] == block[: broke + 1].tolist() and row[broke + 1] == int(second[broke])
            assert all(token >= 0 for token in row[: broke + 1 + width]) and all(token == -1 for token in row[broke + 1 + width :])
            count = 0
            while count < len(truth) and row[1 + count] == truth[count]:
                count += 1
            best = max(best, count)
            following = start + broke + 1
            if width == BLOCK and row[1 : broke + 2] == truth[: broke + 1] and following in watch.first \
                    and watch.first[following].shape[0] == BLOCK - 1:
                # the first draft holds up to this break and the second choice is right: plain decoding's next round starts here
                assert row[broke + 2 : broke + 1 + BLOCK] == watch.first[following].tolist()
                as_next_round += 1
        own = 0
        while own < min(width - 1, len(truth)) and int(block[1 + own]) == truth[own]:
            own += 1
        assert produced == min(max(best, own) + 1, ids.shape[1] - start - 1)
        past_the_block += int(produced > BLOCK)
        at += produced
    assert as_next_round > 3

    # Without the states the restarts write other tokens; the swap's rows can be added; a tail shortens the check.
    bare, repeated = Seen(Restarts(k=3)), Seen(Restarts(k=3, stand_in="last"))
    none = replay_decode(draft, target, ids, length, ended=False, features=features, improver=bare, block_size=BLOCK)
    last = replay_decode(draft, target, ids, length, ended=False, features=features, improver=repeated, block_size=BLOCK)
    with_states = {start: rows for start, _, _, rows, _ in seen.rounds}
    for other in (bare, repeated):
        shared = [(rows, with_states[start]) for start, _, _, rows, _ in other.rounds if start in with_states]
        assert len(shared) > 20 and sum(int(not torch.equal(mine, theirs)) for mine, theirs in shared) > 10
    assert sum(none.acceptance_lengths) == sum(last.acceptance_lengths) == sum(chain.acceptance_lengths)
    swap = replay_decode(draft, target, ids, length, ended=False, improver=Branches(k=3, differ="switch"), block_size=BLOCK)
    both = Seen(Restarts(k=3, swap=True))
    with_swap = replay_decode(draft, target, ids, length, ended=False, features=features, improver=both, block_size=BLOCK)
    assert with_swap.rounds <= min(none.rounds, swap.rounds)
    start, block, logits, rows, kept = both.rounds[0]
    for number, broke in enumerate(kept):
        swapped = block.clone()
        swapped[broke + 1] = logits.topk(2, dim=-1).indices[broke, 1]
        assert rows[len(kept) + number, :BLOCK].tolist() == swapped.tolist() and int(rows[len(kept) + number, BLOCK:].max()) == -1
    checked_short = Restarts(k=3, stand_in="real", tail=1)
    checked_short.features = features
    short = replay_decode(draft, target, ids, length, ended=False, features=features, improver=checked_short, block_size=BLOCK)
    assert chain.rounds >= short.rounds >= result.rounds
    with pytest.raises(NotImplementedError):
        dflash_generate_redraft(draft, target, ids[:, :length], 40, None, improver=Restarts(k=3), block_size=BLOCK)


def test_restarts_from_the_third_choice_too():
    """choices=2: every likely break gets a restart from the drafter's second choice and one from its
    third, in the same drafter pass. The rows of the second choices are those of choices=1, the others
    follow, and with the real states the answer takes no more rounds than with one choice."""
    from dflash.redraft import Restarts
    from dflash.replay import target_features

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    one, two = Restarts(k=3, stand_in="real"), Restarts(k=3, stand_in="real", choices=2, swap=True)
    one.features = two.features = features
    first, both = Seen(one), Seen(two)
    single = replay_decode(draft, target, ids, length, ended=False, features=features, improver=first, block_size=BLOCK)
    double = replay_decode(draft, target, ids, length, ended=False, features=features, improver=both, block_size=BLOCK)
    assert sum(double.acceptance_lengths) == sum(single.acceptance_lengths) and double.rounds <= single.rounds
    of_one = {start: rows for start, _, _, rows, _ in first.rounds}
    compared = 0
    for start, block, logits, rows, kept in both.rounds:
        width, count = block.shape[0], len(kept)
        assert rows.shape == (4 * count, 2 * width)
        ranked = logits.topk(3, dim=-1).indices
        for number, broke in enumerate(kept):
            for choice in (1, 2):
                row = rows[(choice - 1) * count + number]
                assert row[: broke + 1].tolist() == block[: broke + 1].tolist() and int(row[broke + 1]) == int(ranked[broke, choice])
                assert int(row[: broke + 1 + width].min()) >= 0
                swapped = block.clone()
                swapped[broke + 1] = ranked[broke, choice]
                assert rows[(choice + 1) * count + number, :width].tolist() == swapped.tolist()
        if start in of_one:
            assert torch.equal(rows[:count], of_one[start])
            compared += 1
    assert compared > 10
    with pytest.raises(ValueError):
        Restarts(choices=0)


def test_restart_methods_by_name():
    from nexus.redraft_long import make_method

    two = make_method("restart4rc2st7", "/nowhere")
    assert two.k == 4 and two.stand_in == "real" and two.choices == 2 and two.swap and two.alternative_tail == 7
    assert make_method("restart4rs", "/nowhere").choices == 1

    plain = make_method("restart4n", "/nowhere")
    assert type(plain).__name__ == "Restarts" and plain.k == 4 and plain.stand_in == "none" and not plain.swap
    limit = make_method("restart4rs", "/nowhere")
    assert limit.stand_in == "real" and limit.swap and limit.alternative_tail is None
    short = make_method("restart2lt7", "/nowhere")
    assert short.k == 2 and short.stand_in == "last" and short.alternative_tail == 7


def test_the_predictor_of_the_missing_states(tmp_path):
    """The predictor writes vectors of the kind the drafter reads; given as stand-ins, the real
    vectors (projected by hand) restart the drafter exactly as the target's states do; training on
    a replay brings its vectors closer to the real ones; a saved predictor restarts by name."""
    from dflash.redraft import Restarts
    from dflash.replay import target_features
    from dflash.stand_in import StatePredictor, as_context, load_predictor, project, save_predictor
    from nexus.redraft_long import make_method
    from nexus.train_restart import Teach

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    size = draft.config.hidden_size
    raw = torch.randn(5, size)
    assert torch.allclose(as_context(draft, raw).double(), draft.hidden_norm(raw.double()).double(), atol=1e-5)

    class Both:
        """Restarts with the target's states, given as they are and given already projected."""
        seen = 0

        def start(self, drafter):
            pass

        def __call__(self, s):
            if s.block_ids.shape[1] == BLOCK and s.start + 2 * BLOCK < ids.shape[1]:
                states = features[:, s.start : s.start + 2]
                places = torch.arange(s.start, s.start + 2)
                anchor = ids[0, s.start + 2]
                rows = restarts(s, [anchor, anchor], [s.start + 2] * 2, [(states, places), (project(draft, states), places, True)], BLOCK)
                assert torch.equal(rows[0], rows[1])
                Both.seen += 1
            return None

    replay_decode(draft, target, ids, length, ended=False, features=features, improver=Both(), block_size=BLOCK)
    assert Both.seen > 10

    torch.manual_seed(0)
    predictor = StatePredictor(size, inner=32, hidden=64, block=BLOCK)
    teach = Teach(predictor, torch.optim.AdamW(predictor.parameters(), lr=3e-3), every=2, width=BLOCK)
    curve = []
    for _ in range(6):
        teach.begin(ids, features)
        replay_decode(draft, target, ids, length, ended=False, features=features, improver=teach, block_size=BLOCK)
        curve.append(teach.progress())
    assert curve[-1]["loss"] < 0.6 * curve[0]["loss"] and curve[-1]["likeness_to_the_real_vector"] > curve[0]["likeness_to_the_real_vector"]
    assert curve[-1]["loss"] < curve[-1]["loss_of_repeating_the_last_vector"] and teach.steps > 20

    path = tmp_path / "restart-tiny.pt"
    save_predictor(str(path), predictor, {"note": 1})
    loaded, notes = load_predictor(str(path), "cpu")
    assert notes == {"note": 1} and loaded.sizes == predictor.sizes
    method = make_method("restart3ps@tiny", str(tmp_path))
    assert isinstance(method, Restarts) and method.stand_in == "predicted" and method.swap and method.predictor is not None
    chain = replay_decode(draft, target, ids, length, ended=False, features=features, block_size=BLOCK)
    result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=method, block_size=BLOCK)
    assert sum(result.acceptance_lengths) == sum(chain.acceptance_lengths) and result.rounds <= chain.rounds
    probe = Restart(k=3, width=BLOCK, predictor=loaded)
    probe.begin(ids, length, features, None, answer=0)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=probe, block_size=BLOCK)
    assert probe.columns[-1] == "predicted" and len(probe.rows[0]) == len(probe.columns)
    assert "predicted" in summary(probe.rows, probe.columns)["all"]


def test_training_the_predictor_on_the_tokens_the_drafter_writes():
    """The pass used for training (nothing cached) is the pass of real decoding: with the real
    vectors it writes plain decoding's next first draft. Training on tokens runs and changes the
    predictor."""
    from dflash.replay import target_features
    from dflash.stand_in import StatePredictor, project
    from nexus.train_restart import TeachByTokens, first_run_states, restart_logits

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    watch = Watch(ids, features)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=watch, block_size=BLOCK)
    features = features.clone()
    checked = 0
    starts = sorted(watch.first)
    for start, following in zip(starts, starts[1:]):
        count = following - start                            # the round's first token and its accepted draft tokens
        if watch.first[following].shape[0] == BLOCK - 1 and watch.first[start].shape[0] == BLOCK - 1:
            with torch.no_grad():
                real = project(draft, features[:, start:following])[0]
                logits = restart_logits(draft, target, features, ids, start, real, 0, BLOCK)
                states = first_run_states(draft, target, features, ids, start, 0, BLOCK)
            assert torch.equal(logits.argmax(dim=-1), watch.first[following])
            assert states.shape == (BLOCK, draft.config.hidden_size)
            checked += 1
    assert checked > 10

    torch.manual_seed(0)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK)
    before = {name: value.clone() for name, value in predictor.state_dict().items()}
    teach = TeachByTokens(predictor, torch.optim.AdamW(predictor.parameters(), lr=1e-3), draft, target, every=2, window=64, width=BLOCK)
    teach.MISSING = (0.5, 0.3, 0.2)
    ids = ids.clone()                                        # (what real decoding returns cannot be learned through as it is)
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    curve = []
    for _ in range(4):
        teach.learn_from(ids, features, length, most=12)
        curve.append(teach.progress())
    assert teach.steps >= 4 and all(point["loss_by_tokens"] == point["loss_by_tokens"] for point in curve)   # numbers, not NaN
    assert curve[-1]["loss"] < curve[0]["loss"]
    assert any(not torch.equal(before[name], value) for name, value in predictor.state_dict().items())
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(True)
