"""The probe of restarts after the break (nexus/restart_probe.py)."""
import math

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


def test_the_probe_with_real_states_of_other_layers_and_places(tmp_path):
    """--more: restarts with the real states in only some of the layers the drafter reads, with the
    real states of earlier places in the answer where the same token was read, and (with a predictor)
    with the real state most like the predicted one. All layers real is the real state."""
    from dflash.replay import target_features
    from dflash.stand_in import StatePredictor
    from nexus.restart_probe import MORE

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    torch.manual_seed(0)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK).eval()
    probe = Restart(k=3, width=BLOCK, predictor=predictor, more=True)
    probe.begin(ids, length, features, None, answer=0)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=probe, block_size=BLOCK)
    from nexus.restart_probe import SHIFT

    assert probe.columns[-len(MORE) - 2 - len(SHIFT) :] == MORE + ("snap", "echo_level") + SHIFT and len(probe.rows) > 20
    layers = len(draft.target_layer_ids)
    levels = set()
    for row in probe.rows:
        record = dict(zip(probe.columns, row))
        assert record[f"shallow{layers}"] == record["real"] if layers <= 4 else True     # every layer real
        assert all(0 <= record[name] <= BLOCK - 1 for name in MORE + ("snap",)) and 0 <= record["echo_level"] <= 4
        if record["echo_level"] == 0:
            assert record["echo_nearest"] == record["none"]          # no earlier place: nothing is given
        levels.add(record["echo_level"])
    assert len(levels) > 1 and probe.index and probe.indexed > length
    place, level = probe.earlier_place(probe.indexed - 1)
    assert level == 4 or place is None or place < probe.indexed
    out = summary(probe.rows, probe.columns)
    assert "echo_level" not in out["all"] and "snap" in out["all"] and out["by_earlier_place"]
    shifted = out["all"]["shifted_first_draft"]
    assert shifted["the_best_of_all"] >= max(shifted["the_swap_or_one_place_later"], shifted["one_or_two_places_earlier"]) - 1e-9
    assert shifted["the_swap_or_one_place_later"] >= shifted["the_swap"] and all(0 <= dict(zip(probe.columns, row))[name] <= BLOCK for row in probe.rows for name in SHIFT)
    assert sum(group["rounds"] for group in out["by_earlier_place"].values()) == out["all"]["rounds"]


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


class Opening:
    """Keeps the first draft of a replay's first round."""

    def __init__(self):
        self.block = None

    def start(self, drafter):
        pass

    def __call__(self, s):
        if self.block is None:
            self.block = s.block_ids[0].clone()
        return None


def test_restarts_from_the_last_token_before_the_break():
    """kind="earlier": with the real states, the block from the last token before a likely break is
    the first draft of a round that starts at that token. "both" holds the two kinds, and the answer
    takes no more rounds than with either alone."""
    from dflash.redraft import Restarts
    from dflash.replay import target_features

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    watch = Watch(ids, features)
    base = replay_decode(draft, target, ids, length, ended=False, features=features, improver=watch, block_size=BLOCK)
    methods = {kind: Restarts(k=3, stand_in="real", kind=kind) for kind in ("choice", "earlier", "both")}
    seen, rounds = {}, {}
    for kind, method in methods.items():
        method.features = features
        seen[kind] = Seen(method)
        done = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen[kind], block_size=BLOCK)
        assert sum(done.acceptance_lengths) == sum(base.acceptance_lengths)
        rounds[kind] = done.rounds
    assert rounds["both"] <= min(rounds["choice"], rounds["earlier"]) <= base.rounds
    checked = 0
    for start, block, logits, rows, kept in seen["earlier"].rounds:
        width = block.shape[0]
        later = [broke for broke in kept if broke >= 1]
        assert (rows is None and not later) or rows.shape == (len(later), 2 * width)
        for number, broke in enumerate(later):
            row = rows[number]
            assert row[: broke + 1].tolist() == block[: broke + 1].tolist() and int(row[broke + width]) == -1
            if checked < 8 and width == BLOCK and start + broke + BLOCK < ids.shape[1]:
                # A round that starts at that token of the first draft (it need not be the answer's token there).
                changed, opening = ids.clone(), Opening()
                changed[0, start + broke] = block[broke]
                replay_decode(draft, target, changed, start + broke, ended=False, features=features, improver=opening, block_size=BLOCK)
                assert row[broke : broke + width].tolist() == opening.block.tolist()
                checked += 1
    assert checked > 3
    of_choice = {start: rows for start, _, _, rows, _ in seen["choice"].rounds}
    of_earlier = {start: rows for start, _, _, rows, _ in seen["earlier"].rounds}
    compared = 0
    for start, block, logits, rows, kept in seen["both"].rounds:
        if start in of_choice and start in of_earlier:
            expected = of_choice[start] if of_earlier[start] is None else torch.cat((of_choice[start], of_earlier[start]))
            assert torch.equal(rows, expected)
            compared += 1
    assert compared > 10
    with pytest.raises(ValueError):
        Restarts(kind="later")
    # An answer that ended by itself: its last block reaches past its end, where there are no states.
    for kind, method in methods.items():
        closed = replay_decode(draft, target, ids, length, ended=True, features=features, improver=Seen(method), block_size=BLOCK)
        assert sum(closed.acceptance_lengths) >= ids.shape[1] - length - 1
    unset = Restarts(k=3, stand_in="real")
    with pytest.raises(ValueError):
        replay_decode(draft, target, ids, length, ended=False, features=features, improver=Seen(unset), block_size=BLOCK)


def test_an_alternative_that_only_adds_to_the_block_is_checked_from_its_end():
    """A restart that writes the first draft's own tokens again and goes on past the block: with a
    tail, only the first token past the block and `tail` more are checked."""
    from dflash.replay import _checked_matches

    block = torch.tensor([[5, 6, 7, 8]])
    longer = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    truth = torch.tensor([[6, 7, 8, 9, 10, 11, 12]])
    assert _checked_matches(longer, block, truth, None) == 7
    assert _checked_matches(longer, block, truth, 1) == 5 and _checked_matches(longer, block, truth, 0) == 4
    same = torch.tensor([[5, 6, 7, 8, -1, -1, -1, -1]])
    assert _checked_matches(same, block, truth, 1) == 3


def test_a_tree_with_restarts_on_top():
    """DDTree and restarts in the same check: the rows are the tree's and the restarts' (cut to
    their tail), the count of what the target reads has shared beginnings once, and the answer
    takes no more rounds than with the tree alone or the restarts alone."""
    from dflash.redraft import DraftTree, Restarts, TreeWithRestarts
    from dflash.replay import target_features
    from nexus.redraft_long import make_method

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    both = make_method("tree12_restart3rxt2", "/nowhere")
    assert isinstance(both, TreeWithRestarts) and both.replace and both.tree.budget == 12 and both.restarts.kind == "both"
    both.features = features
    assert both.restarts.features is features
    alone = Restarts(k=3, stand_in="real", kind="both")
    alone.features = features
    rounds = {}
    for name, method in (("tree", DraftTree(12)), ("restarts", alone), ("both", both)):
        seen = Seen(method) if name != "tree" else method
        done = replay_decode(draft, target, ids, length, ended=True, features=features, improver=seen, block_size=BLOCK)
        assert sum(done.acceptance_lengths) >= ids.shape[1] - length - 1
        rounds[name] = done.rounds
    assert rounds["both"] <= min(rounds["tree"], rounds["restarts"])

    class Check:
        replace = True
        counted = 0

        def start(self, drafter):
            both.start(drafter)

        def __call__(self, s):
            rows = both(s)
            width = s.block_ids.shape[1]
            tree_rows = DraftTree(12)(s)
            assert torch.equal(rows[: tree_rows.shape[0], :width], tree_rows) and int(rows[: tree_rows.shape[0], width:].max()) == -1
            for row in rows[tree_rows.shape[0] :]:
                differs = (row[:width] != s.block_ids[0]).nonzero()
                split = int(differs[0]) if differs.numel() else width
                assert int((row >= 0).sum()) <= split + 1 + 2                 # the tail of 2
            prefixes = {tuple(row[:end]) for row in rows.tolist() for end in range(2, len(row) + 1) if min(row[:end]) >= 0}
            assert both.nodes == len(prefixes) and both.nodes >= both.tree.nodes
            Check.counted += 1
            return rows

    replay_decode(draft, target, ids, length, ended=False, features=features, improver=Check(), block_size=BLOCK)
    assert Check.counted > 10
    with pytest.raises(ValueError):
        TreeWithRestarts(DraftTree(12), Restarts(k=3, swap=True))


def test_a_tree_of_drafts_in_real_decoding(tmp_path):
    """Real decoding with a tree in place of the first draft (DDTree), with restarts that reach
    past the block, and with both: the answer is plain decoding's, in fewer rounds or as many, the
    same when it follows plain decoding's text, and with a window for the drafter."""
    from dflash.redraft import DraftTree, Restarts, TreeWithRestarts, _tree_of_drafts, dflash_generate_redraft
    from dflash.stand_in import StatePredictor

    tokens, depths, parents, children, on_trunk = _tree_of_drafts([7, 1, 2, 3], [[7, 1, 2, 3, 4, -1], [7, 1, 5, 6, -1, -1], [7, 9, -1, -1, 8, 8]], 4)
    assert tokens == [7, 1, 2, 3, 4, 5, 6, 9] and depths == [0, 1, 2, 3, 4, 2, 3, 1] and parents == [-1, 0, 1, 2, 3, 1, 5, 0]
    assert on_trunk == 4 and children[1] == {2: 2, 5: 5} and children[0] == {1: 1, 9: 7}
    assert _tree_of_drafts([7, 1, 2, 3], [[7, 1, 2, 3, 4, -1]], 3)[0] == [7, 1, 2, 3]            # nothing deeper than 3

    target, draft = make_models()
    plain, length = answer(target, draft, budget=120)
    prompt = plain.output_ids[:, :length]
    torch.manual_seed(0)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK, coming=True, layers=1, heads=4, memory=4).eval()

    def restarts(**how):
        method = Restarts(k=3, stand_in="predicted", **how)
        method.predictor = predictor
        return method

    methods = {
        "tree": lambda: DraftTree(24),
        "small tree": lambda: DraftTree(3),
        "restarts": lambda: restarts(kind="both", swap=True),
        "restarts, short check": lambda: restarts(kind="both", tail=2, choices=2),
        "no states": lambda: Restarts(k=3, stand_in="none", kind="earlier"),
        "tree and restarts": lambda: TreeWithRestarts(DraftTree(24), restarts(kind="both", tail=3)),
    }
    for name, make in methods.items():
        result = dflash_generate_redraft(draft, target, prompt, 120, None, improver=make(), block_size=BLOCK, return_stats=True)
        assert torch.equal(result.output_ids, plain.output_ids), name
        assert result.rounds <= plain.rounds and sum(result.acceptance_lengths) == sum(plain.acceptance_lengths), name
        followed = dflash_generate_redraft(draft, target, prompt, 120, None, improver=make(), block_size=BLOCK, return_stats=True,
                                           follow=plain.output_ids)
        assert torch.equal(followed.output_ids, plain.output_ids) and followed.rounds == result.rounds, name
        assert followed.followed_rows >= plain.output_ids.shape[1] - length - 1 and followed.followed_against == 0, name
        clock, log = [], []
        windowed = dflash_generate_redraft(draft, target, prompt, 120, None, improver=make(), block_size=BLOCK, return_stats=True,
                                           context_window=12, round_clock=clock, round_log=log, timers={})
        chain = dflash_generate_redraft(draft, target, prompt, 120, None, block_size=BLOCK, return_stats=True, context_window=12)
        assert torch.equal(windowed.output_ids, chain.output_ids) and windowed.rounds <= chain.rounds, name
        assert len(clock) == windowed.rounds and sum(entry[1] for entry in clock) == sum(windowed.acceptance_lengths)
    wide = dflash_generate_redraft(draft, target, prompt, 120, None, improver=DraftTree(24), block_size=BLOCK, return_stats=True)
    assert wide.alternative_wins > 0 and wide.rounds < plain.rounds          # the tree is used, not only its first path
    stops = [int(plain.output_ids[0, length + 30])]
    ended = dflash_generate_redraft(draft, target, prompt, 120, stops, block_size=BLOCK, return_stats=True)
    for make in (methods["tree"], methods["tree and restarts"]):
        early = dflash_generate_redraft(draft, target, prompt, 120, stops, improver=make(), block_size=BLOCK, return_stats=True)
        assert torch.equal(early.output_ids, ended.output_ids)
    with pytest.raises(NotImplementedError):
        dflash_generate_redraft(draft, target, prompt, 40, None, improver=DraftTree(24), block_size=BLOCK, recheck=True)


def test_tails_from_what_the_target_already_said():
    """EchoRows: a table from the last tokens to the target's next token, filled from the accepted
    text and, in real decoding, from every token the target read; its tails go into the check as
    rows, alone or with DDTree. The answer is plain decoding's, in no more rounds."""
    from dflash.redraft import DraftTree, EchoRows, TreeWithRestarts, dflash_generate_redraft
    from nexus.live_speed import swap_of
    from nexus.redraft_long import make_method

    echo = EchoRows(k=2, order=3, reach=5)
    for at, token in enumerate([1, 2, 3, 4, 1, 2, 3]):
        echo.learn([1, 2, 3, 4, 1, 2, 3][:at], token)
    assert echo.follow([9, 2, 3], 5) == [4, 1, 2, 3, 4]              # ..2 3 -> 4, 3 4 -> 1, and round again
    assert echo.follow([9, 9], 5) == [] and echo.follow([4, 1], 2) == [2, 3]
    # What the target chose after every node of a tree: node 2 (tokens 7 8 5 after ..3) and node 3 (7 6).
    echo.heard([1, 2, 3], [7, 8, 5, 6], [-1, 0, 1, 0], [10, 11, 12, 13])
    assert echo.table[(3, 7)] == 10 and echo.table[(2, 3, 7)] == 10 and echo.table[(7, 8, 5)] == 12 and echo.table[(3, 7, 6)] == 13
    assert echo.follow([3, 7, 6], 1) == [13] and echo.entries_heard == 4
    with pytest.raises(ValueError):
        EchoRows(least=4, order=3)

    named = make_method("tree12_echo3c2n4r9", "/nowhere")
    assert isinstance(named, TreeWithRestarts) and isinstance(named.restarts, EchoRows)
    assert (named.restarts.k, named.restarts.choices, named.restarts.order, named.restarts.reach) == (3, 2, 4, 9)
    assert isinstance(make_method("echo4", "/nowhere"), EchoRows) and isinstance(swap_of("plain+tree63_echo4")[1], TreeWithRestarts)

    target, draft = make_models()
    plain, length = answer(target, draft, budget=160)
    ids = plain.output_ids
    prompt = ids[:, :length]
    chain = replay_decode(draft, target, ids, length, ended=False, block_size=BLOCK)
    for name in ("echo3", "echo3c2r20l1", "tree12_echo3"):
        replayed = replay_decode(draft, target, ids, length, ended=True, improver=make_method(name, "/nowhere"), block_size=BLOCK)
        assert sum(replayed.acceptance_lengths) >= ids.shape[1] - length - 1 and replayed.rounds <= chain.rounds, name
        method = make_method(name, "/nowhere")
        real = dflash_generate_redraft(draft, target, prompt, 160, None, improver=method, block_size=BLOCK, return_stats=True)
        assert torch.equal(real.output_ids, ids) and real.rounds <= plain.rounds, name
        inner = method.restarts if name.startswith("tree") else method
        assert inner.rows > 0 and inner.entries_heard > 0 and length + 20 < len(inner.text) <= ids.shape[1], name
        followed = dflash_generate_redraft(draft, target, prompt, 160, None, improver=make_method(name, "/nowhere"),
                                           block_size=BLOCK, return_stats=True, follow=ids, context_window=12)
        # (a pass of another shape rounds differently, and a nearly tied choice may then fall the other way)
        assert followed.followed_against <= 2 and torch.equal(followed.output_ids, ids), name
    # The tiny models repeat themselves, so the table's tails are accepted: fewer rounds than the tree alone.
    tree = dflash_generate_redraft(draft, target, prompt, 160, None, improver=DraftTree(12), block_size=BLOCK, return_stats=True)
    both = dflash_generate_redraft(draft, target, prompt, 160, None, improver=make_method("tree12_echo3", "/nowhere"),
                                   block_size=BLOCK, return_stats=True)
    assert both.rounds <= tree.rounds


def test_the_second_run_only_where_the_first_draft_will_likely_break():
    """gate: no second run in rounds where the drafter's own confidences say the draft will hold.
    A gate nothing passes is plain decoding; a gate everything passes is the second run in every round."""
    from dflash.redraft import Restarts, dflash_generate_redraft
    from nexus.redraft_long import make_method

    target, draft = make_models()
    plain, length = answer(target, draft, budget=120)
    prompt = plain.output_ids[:, :length]
    named = make_method("restart3nxt2g5", "/nowhere")
    assert named.gate == 5.0 and named.alternative_tail == 2 and named.kind == "both" and make_method("restart3nx", "/nowhere").gate is None
    runs = {}
    for gate in (None, 100.0, 0.0, 0.5):
        method = Restarts(k=3, stand_in="none", kind="both", gate=gate)
        result = dflash_generate_redraft(draft, target, prompt, 120, None, improver=method, block_size=BLOCK, return_stats=True)
        assert torch.equal(result.output_ids, plain.output_ids)
        runs[gate] = (result.rounds, method.rounds_run, method.rounds_seen)
    assert runs[None][:2] == runs[100.0][:2] and runs[100.0][1] == runs[100.0][2]
    assert runs[0.0][1] == 0 and runs[0.0][0] == plain.rounds
    assert 0 <= runs[0.5][1] <= runs[0.5][2] and runs[None][0] <= runs[0.5][0] <= plain.rounds


def test_short_restarts_picked_best_first():
    """block: a restart's new block has only that many tokens, so the second run reads and writes fewer.
    best: the restarts from the drafter's next choices are picked over all places and choices by the
    probability that the text goes that way (DDTree's score of the node), not only at the k likely breaks."""
    from dflash.redraft import Restarts, dflash_generate_redraft
    from dflash.replay import target_features
    from nexus.redraft_long import make_method

    named = make_method("restart2rxc3r6st3b5g9", "/nowhere")
    assert (named.k, named.kind, named.choices, named.best, named.swap, named.alternative_tail, named.block, named.gate) == (
        2, "both", 3, 6, True, 3, 5, 9.0)
    plain_name = make_method("restart4rxt3", "/nowhere")
    assert plain_name.best is None and plain_name.block is None
    with pytest.raises(ValueError):
        Restarts(block=1)
    with pytest.raises(ValueError):
        Restarts(best=0)

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    whole, short = Restarts(k=3, stand_in="real", kind="both"), Restarts(k=3, stand_in="real", kind="both", block=5)
    picked = Restarts(k=3, stand_in="real", kind="both", choices=3, best=5, block=5, swap=True)
    whole.features = short.features = picked.features = features
    seen_whole, seen_short, seen_picked = Seen(whole), Seen(short), Seen(picked)
    full = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen_whole, block_size=BLOCK)
    cut = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen_short, block_size=BLOCK)
    best = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen_picked, block_size=BLOCK)
    assert sum(full.acceptance_lengths) == sum(cut.acceptance_lengths) == sum(best.acceptance_lengths)
    assert max(cut.rounds, best.rounds) <= plain.rounds

    # A short block: the same first tokens of every row as far as the row goes, nothing after it.
    start, block, logits, rows, kept = seen_short.rounds[0]
    assert seen_whole.rounds[0][0] == start and rows.shape == seen_whole.rounds[0][3].shape
    width, chosen = block.shape[0], len(kept)
    for number, broke in enumerate(kept):
        row = rows[number]
        assert int(row[: broke + 6].min()) >= 0 and int(row[broke + 6 :].max()) == -1      # the choice and 4 new tokens
        assert row[: broke + 2].tolist() == seen_whole.rounds[0][3][number, : broke + 2].tolist()
    for number, broke in enumerate([broke for broke in kept if broke >= 1]):
        row = rows[chosen + number]
        assert int(row[: broke + 5].min()) >= 0 and int(row[broke + 5 :].max()) == -1      # 4 new tokens from the break on

    # Best first: the places and choices with the largest (confidences before the place) x (the choice's probability).
    for start, block, logits, rows, kept in seen_picked.rounds[:20]:
        width = block.shape[0]
        if width < BLOCK:
            continue
        top = logits.float().softmax(dim=-1).topk(4, dim=-1)
        holds = torch.cat((top.values.new_ones(1), top.values[:, 0].cumprod(dim=0)[:-1]))[: width - 1]
        worth = holds[:, None] * top.values[: width - 1, 1:]
        order = worth.flatten().topk(5).indices.tolist()
        earlier = sum(1 for broke in kept if broke >= 1)
        assert rows.shape[0] == 5 + earlier + 5
        for number, index in enumerate(order):
            broke, choice = index // 3, 1 + index % 3
            row = rows[number]
            assert row[: broke + 1].tolist() == block[: broke + 1].tolist() and int(row[broke + 1]) == int(top.indices[broke, choice])
            swapped = block.clone()
            swapped[broke + 1] = top.indices[broke, choice]
            assert rows[5 + earlier + number, :width].tolist() == swapped.tolist()

    # In real decoding the text is the plain drafter's whatever the second run adds.
    prompt = ids[:, :length]
    for method in (Restarts(k=3, kind="both", tail=3, block=5), Restarts(k=3, kind="both", choices=3, best=6, tail=3, block=5, swap=True)):
        result = dflash_generate_redraft(draft, target, prompt, 200, None, improver=method, block_size=BLOCK, return_stats=True)
        assert torch.equal(result.output_ids, plain.output_ids) and result.rounds <= plain.rounds


def test_states_left_over_from_the_last_check():
    """stale: where the last check read this round's token at its place, the target's state from that
    check is given to the second run, in place of a predicted one or as the only state. Given the true
    states of the tokens the first draft has right, the restarts are those with the target's real states."""
    from dflash.redraft import Restarts, dflash_generate_redraft
    from dflash.replay import target_features
    from dflash.stand_in import StatePredictor
    from nexus.redraft_long import make_method

    named = make_method("restart4nxmc2r6st3b5", "/nowhere")
    assert named.stale and named.stand_in == "none" and (named.kind, named.choices, named.best, named.block) == ("both", 2, 6, 5)
    assert not make_method("restart4nxt3b5", "/nowhere").stale

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)

    class TruthAsLeftover:
        """Hands the method a leftover that holds the answer's own tokens and states at the block's
        places, shifted by two tokens to see that places are matched, not numbers."""

        def __init__(self, method):
            self.method, self.rounds = method, []

        def start(self, drafter):
            self.method.start(drafter)

        def __call__(self, s):
            begin = max(s.start - 2, 0)
            end = min(s.start + s.block_ids.shape[1], features.shape[1])
            s.leftover = {"start": begin, "tokens": ids[0, begin:end].tolist(), "features": features[0, begin:end]}
            rows = self.method(s)
            self.rounds.append((s.start, s.block_ids[0].clone(), None if rows is None else rows.clone(), list(self.method.kept)))
            return rows

    real = Restarts(k=3, stand_in="real", kind="both")
    real.features = features
    only = Restarts(k=3, stand_in="none", kind="both", stale=True)
    mixed = Restarts(k=3, stand_in="predicted", kind="both", stale=True)
    torch.manual_seed(0)
    mixed.predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK).eval()
    bare = Restarts(k=3, stand_in="none", kind="both")
    seen = {}
    for name, method in (("real", real), ("only", only), ("mixed", mixed), ("bare", bare)):
        seen[name] = TruthAsLeftover(method)
        replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen[name], block_size=BLOCK)
    same = differs_without = compared = 0
    for round_real, round_only, round_mixed, round_bare in zip(*(seen[name].rounds for name in ("real", "only", "mixed", "bare"))):
        start, block, rows, kept = round_real
        if rows is None or round_only[0] != start or round_only[3] != kept or round_mixed[0] != start or round_bare[0] != start:
            break                                           # the replays part ways once one accepts more
        width = block.shape[0]
        truth = ids[0, start : start + width]
        right = int((block[: truth.shape[0]] == truth).long().cumprod(dim=0).sum())      # the block's first tokens that are right
        for number, broke in enumerate(kept):
            if broke + 1 <= right:                          # every state this restart misses is the answer's
                compared += 1
                same += int(torch.equal(rows[number], round_only[2][number]) and torch.equal(rows[number], round_mixed[2][number]))
                differs_without += int(not torch.equal(rows[number], round_bare[2][number]))
    assert compared > 5 and same == compared and differs_without > 0
    assert only.counts["restarts"] > 0 and 0 < only.counts["states_from_the_last_check"] <= only.counts["states_missing"]
    assert 0 < only.counts["restarts_with_the_nearest_state_from_the_last_check"] <= only.counts["restarts"]
    assert bare.counts["states_from_the_last_check"] == 0 and bare.counts["states_missing"] > 0

    # In real decoding the checks leave their states; the text stays the plain drafter's.
    prompt = ids[:, :length]
    for method in (Restarts(k=3, stand_in="none", kind="both", stale=True, tail=3, block=5),
                   Restarts(k=3, stand_in="none", kind="both", stale=True, choices=2, best=4, tail=3, block=5, gate=4.0)):
        result = dflash_generate_redraft(draft, target, prompt, 200, None, improver=method, block_size=BLOCK, return_stats=True)
        assert torch.equal(result.output_ids, plain.output_ids) and result.rounds <= plain.rounds
        assert method.counts["restarts"] > 0 and method.counts["states_missing"] >= method.counts["states_from_the_last_check"] >= 0


def test_new_blocks_from_what_the_last_check_left():
    """Reanchor: a block started at the token the target chose after a rejected draft token, with the
    target's states of the rejected tokens before it, attached where the first draft has that token."""
    from dflash.redraft import Reanchor, TreeWithRestarts, DraftTree, dflash_generate_redraft
    from dflash.replay import target_features
    from nexus.redraft_long import make_method

    named = make_method("reanchor6t5b7f", "/nowhere")
    assert (named.k, named.alternative_tail, named.block, named.fixed) == (6, 5, 7, True)
    free = make_method("reanchor", "/nowhere")
    assert (free.k, free.alternative_tail, free.block, free.fixed) == (None, None, None, False)
    assert type(make_method("tree31_reanchort3b5", "/nowhere")).__name__ == "TreeWithRestarts"

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)

    class TruthAsLeftover:
        """The leftover as it would be had the last check read the answer itself from this round's
        first token on: then a block started after `count` of those tokens is the first draft of a
        round that starts there."""

        def __init__(self, method):
            self.method, self.rounds = method, []

        def start(self, drafter):
            self.method.start(drafter)

        def __call__(self, s):
            width = s.block_ids.shape[1]
            end = min(s.start + width, ids.shape[1] - 1)
            if end - s.start < 3:
                return None
            s.leftover = {"start": s.start, "tokens": ids[0, s.start : end].tolist(), "features": features[0, s.start : end],
                          "said": ids[0, s.start + 1 : end + 1].tolist()}
            expected = {}
            for count in range(1, end - s.start + 1):
                states = (features[:, s.start : s.start + count], torch.arange(s.start, s.start + count))
                expected[count] = restarts(s, [ids[0, s.start + count]], [s.start + count], [states],
                                           min(self.method.block or width, width))[0].tolist()
            rows = self.method(s)
            self.rounds.append((s.start, s.block_ids[0].tolist(), expected, None if rows is None else rows.tolist()))
            return rows

    seen = TruthAsLeftover(Reanchor())
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen, block_size=BLOCK)
    attached = moved = 0
    for start, first_draft, expected, rows in seen.rounds:
        want = []
        for count in sorted(expected):
            token = int(ids[0, start + count])
            at = next((index for index in range(len(first_draft)) if first_draft[index] == token), None)
            if at is not None:
                want.append((first_draft[: at + 1] + expected[count])[: 2 * len(first_draft)])
                moved += int(at != count)
        attached += len(want)
        got = [] if rows is None else [[token for token in row if token >= 0] for row in rows]
        assert got == want
    assert attached > 20 and seen.method.counts["blocks_attached"] == attached
    assert seen.method.counts["blocks_attached_elsewhere"] == moved and seen.method.counts["blocks"] > attached

    # corrections: the first draft has the same token before a block's place as the last check had, and another
    # token at the place: the block is attached there with its first token in the draft's place. With the answer
    # itself as what the last check read, that is the right token after every break past the round's second token.
    assert make_method("reanchor6ct5b7", "/nowhere").corrections and not make_method("reanchor6t5b7", "/nowhere").corrections
    with_draft = make_method("tree15_reanchorwct5b7", "/nowhere")
    assert with_draft.restarts.wants_first_draft and with_draft.restarts.corrections and with_draft.restarts.block == 7
    assert not make_method("tree15_reanchorct5b7", "/nowhere").restarts.wants_first_draft
    told = TruthAsLeftover(Reanchor(corrections=True))
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=told, block_size=BLOCK)
    corrected = 0
    for start, first_draft, expected, rows in told.rounds:
        got = [] if rows is None else [[token for token in row if token >= 0] for row in rows]
        truth = ids[0, start : start + len(first_draft) + 1].tolist()
        for count in sorted(expected):
            if 2 <= count < len(first_draft) and first_draft[count - 1] == truth[count - 1]:
                row = (first_draft[:count] + [truth[count]] + expected[count])[: 2 * len(first_draft)]
                assert row in got
                corrected += int(first_draft[count] != truth[count])
    assert corrected > 0 and told.method.counts["corrections"] == corrected
    assert told.rounds and len(told.rounds) <= len(seen.rounds)        # the foretold breaks are passed: no more rounds

    fixed = TruthAsLeftover(Reanchor(k=4, block=5, fixed=True))
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=fixed, block_size=BLOCK)
    for start, first_draft, expected, rows in fixed.rounds:
        for row in rows or []:
            row = [token for token in row if token >= 0]
            count = next(c for c in sorted(expected) if row[: c + 1] == first_draft[: c + 1] and row[c + 1 :] == expected[c] and c <= 4)
            assert first_draft[count] == int(ids[0, start + count]) and len(row) == count + 5
    assert fixed.method.counts["blocks_attached_elsewhere"] == 0 and fixed.method.counts["blocks"] <= 4 * fixed.method.counts["rounds_with_blocks"]

    # Real decoding: the checks leave what the blocks are started from, and the text stays the plain drafter's.
    prompt = ids[:, :length]
    # The blocks are written in the first draft's pass (ahead) or in one of their own: the same blocks either way.
    for make in (lambda ahead: Reanchor(ahead=ahead), lambda ahead: Reanchor(k=6, tail=3, block=5, ahead=ahead),
                 lambda ahead: Reanchor(tail=3, block=5, ahead=ahead, corrections=True),
                 lambda ahead: TreeWithRestarts(DraftTree(15), Reanchor(tail=3, block=5, ahead=ahead, corrections=True)),
                 lambda ahead: TreeWithRestarts(DraftTree(15), Reanchor(tail=3, block=5, ahead=ahead, whole=True))):
        ran = []
        for ahead in (True, False):
            method = make(ahead)
            assert bool(method.ahead) == ahead
            timers = {}
            result = dflash_generate_redraft(draft, target, prompt, 200, None, improver=method, block_size=BLOCK,
                                             return_stats=True, timers=timers)
            assert torch.equal(result.output_ids, plain.output_ids) and result.rounds <= plain.rounds
            counts = method.counts
            assert counts["rounds"] > 0 and counts["blocks"] >= counts["blocks_attached"] >= counts["blocks_attached_elsewhere"] >= 0
            assert counts["rounds_with_blocks"] > 0
            ran.append((result.rounds, dict(counts), timers["rows_one"] if "rows_one" in timers else 0, timers.get("rows_two", 0)))
        assert ran[0] == ran[1]
    lean = TreeWithRestarts(DraftTree(15), Reanchor(tail=3, block=5))
    full = TreeWithRestarts(DraftTree(15), Reanchor(tail=3, block=5, whole=True))
    for method in (lean, full):
        result = dflash_generate_redraft(draft, target, prompt, 200, None, improver=method, block_size=BLOCK, return_stats=True)
        assert torch.equal(result.output_ids, plain.output_ids)
    assert full.counts["blocks"] > lean.counts["blocks"]             # the target read more of the first draft


def test_restart_methods_by_name():
    from nexus.redraft_long import make_method

    assert make_method("restart4re", "/nowhere").kind == "earlier" and make_method("restart4r", "/nowhere").kind == "choice"
    both = make_method("restart4rxc2st7", "/nowhere")
    assert both.kind == "both" and both.choices == 2 and both.swap and both.alternative_tail == 7

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


def test_a_predictor_that_is_told_the_coming_token_and_looks_back(tmp_path):
    """coming / layers / memory: a vector depends on the tokens before it and on the token right
    after it, not on anything later, so the vectors of the first tokens are the same however far the
    draft went on and whichever token the restart starts from. It trains both ways, is saved and
    loaded, and runs in the replay and in the probe."""
    from dflash.redraft import Restarts
    from dflash.replay import target_features
    from dflash.stand_in import StatePredictor, load_predictor, predicted_states, save_predictor
    from nexus.redraft_long import make_method
    from nexus.train_restart import Teach, TeachByTokens

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    size = draft.config.hidden_size
    torch.manual_seed(0)
    predictor = StatePredictor(size, inner=32, hidden=64, block=BLOCK, coming=True, layers=2, heads=4, memory=4).eval()
    assert StatePredictor(size, inner=32, hidden=64, block=BLOCK).sizes["coming"] is False

    class Look:
        seen = 0

        def start(self, drafter):
            pass

        def __call__(self, s):
            if s.block_ids.shape[1] == BLOCK and s.context_features.shape[1]:
                whole = predicted_states(predictor, s, 6)
                assert whole.shape == (1, 6, size)
                assert torch.allclose(predicted_states(predictor, s, 4)[:, :3], whole[:, :3], atol=1e-5)
                other = int(s.block_ids[0, 6]) + 1
                told = predicted_states(predictor, s, 6, coming=other)
                assert torch.allclose(told[:, :5], whole[:, :5], atol=1e-5) and not torch.allclose(told[:, 5], whole[:, 5], atol=1e-5)
                same = predicted_states(predictor, s, 6, coming=s.block_ids[0, 6])
                assert torch.allclose(same, whole, atol=1e-6)
                Look.seen += 1
            return None

    replay_decode(draft, target, ids, length, ended=False, features=features, improver=Look(), block_size=BLOCK)
    assert Look.seen > 10

    predictor.train()
    before = {name: value.clone() for name, value in predictor.state_dict().items()}
    by_tokens = TeachByTokens(predictor, torch.optim.AdamW(predictor.parameters(), lr=1e-3), draft, target, every=2, window=64,
                              width=BLOCK, compare_every=1)
    by_tokens.learn_from(ids, features.clone(), length, most=12)
    assert by_tokens.steps >= 2 and by_tokens.progress()["loss_by_tokens"] > 0
    by_vectors = Teach(predictor, torch.optim.AdamW(predictor.parameters(), lr=1e-3), every=2, width=BLOCK)
    by_vectors.begin(ids, features)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=by_vectors, block_size=BLOCK)
    assert by_vectors.steps > 5
    changed = [name for name, value in predictor.state_dict().items() if not torch.equal(before[name], value)]
    assert any(name.startswith("mix.") for name in changed) and any(name.startswith("from_coming") for name in changed)
    assert any(name.startswith("remember") for name in changed)

    path = tmp_path / "restart-told.pt"
    save_predictor(str(path), predictor, {})
    loaded, _ = load_predictor(str(path), "cpu")
    assert loaded.sizes == predictor.sizes and len(loaded.mix) == 2
    chain = replay_decode(draft, target, ids, length, ended=False, features=features, block_size=BLOCK)
    for name in ("restart3pxc2s@told", "restart3pe@told"):
        method = make_method(name, str(tmp_path))
        assert isinstance(method, Restarts) and method.predictor.sizes["coming"]
        result = replay_decode(draft, target, ids, length, ended=True, features=features, improver=method, block_size=BLOCK)
        assert sum(result.acceptance_lengths) >= sum(chain.acceptance_lengths) and result.rounds <= chain.rounds + 1
    probe = Restart(k=3, width=BLOCK, predictor=loaded, reanchor=True)
    probe.begin(ids, length, features, None, answer=0)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=probe, block_size=BLOCK)
    assert "reanchor_predicted" in probe.columns and "predicted" in summary(probe.rows, probe.columns)["all"]


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
    assert teach.where == "breaks" and all("on_the_same_rounds" in point for point in curve)
    # Anywhere, with a given number of missing tokens: the restart one token earlier is taught and compared too.
    other = TeachByTokens(predictor, torch.optim.AdamW(predictor.parameters(), lr=1e-3), draft, target, every=2, window=64,
                          width=BLOCK, compare_every=1, where="random")
    for start in range(length + 2, length + 10):
        other.step(ids, features, start, 3)
    point = other.progress()
    assert point["one_token_earlier"]["rounds"] == 8 and point["on_the_same_rounds"]["rounds"] == 8
    only_after = TeachByTokens(predictor, torch.optim.AdamW(predictor.parameters(), lr=1e-3), draft, target, every=2, window=64,
                               width=BLOCK, compare_every=1, where="random", earlier=False)
    only_after.step(ids, features, length + 2, 3)
    assert "one_token_earlier" not in only_after.progress()
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(True)


def test_where_the_trees_rounds_end():
    """nexus/tree_ends.py: the answer's walk through DDTree, the node where it ends, and why."""
    from dflash.redraft import DraftTree
    from nexus.tree_ends import COLUMNS, WAYS, Ends, summary, way

    assert way(15, 1, -1, -1) == WAYS[0] and way(3, 1, 0, 1) == WAYS[1] and way(3, 1, 0, 7) == WAYS[2]
    assert way(3, 0, 0, 1) == WAYS[3] and way(3, 0, 0, 2) == WAYS[4]
    assert way(0, 1, 3, 4) == WAYS[5] and way(0, 1, 3, 5) == WAYS[6] and way(2, 0, 1, 11) == WAYS[7]

    target, draft = make_models()
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    probe = Ends(15, width=BLOCK)
    probe.begin(ids, length, None, 0)
    with_tree = replay_decode(draft, target, ids, length, ended=False, improver=probe, block_size=BLOCK)
    alone = replay_decode(draft, target, ids, length, ended=False, improver=DraftTree(15), block_size=BLOCK)
    assert with_tree.rounds == alone.rounds and with_tree.acceptance_lengths == alone.acceptance_lengths
    table = [dict(zip(COLUMNS, row)) for row in probe.rows]
    assert len(table) > 20
    # A recorded round accepted exactly the draft tokens the walk counted (and then the target's own token).
    by_start, at = {}, length
    for produced in with_tree.acceptance_lengths:
        by_start[at - length] = produced
        at += produced
    for row in table:
        assert by_start[row["where"]] == row["depth"] + 1 and 1 <= row["nodes"] <= 15 and 1 <= row["trunk_nodes"] <= row["nodes"]
        if row["depth"] < BLOCK - 1:
            assert row["children"] >= 0 and row["rank"] >= 1 and row["after_it"] >= 0
            assert not (row["children"] > 0 and row["rank"] == 1)      # a node with children has its first choice among them
    report = summary(probe.rows, width=BLOCK)
    assert report["rounds"] == len(table) and abs(sum(entry["share"] for entry in report["ways"].values()) - 1) < 1e-3
    assert set(report["ways"]) <= set(WAYS)


class Partly:
    """In every round with a break: restarts through some of the drafter's layers, checked against the whole pass."""

    def __init__(self, ids, features):
        self.ids, self.features, self.checked, self.differs = ids, features, 0, 0

    def start(self, drafter):
        pass

    def __call__(self, s):
        width = s.block_ids.shape[1]
        truth = self.ids[:, s.start + 1 : s.start + width]
        broke = _matches(s.block_ids[:, 1:], truth)
        restart = s.start + broke + 1
        if not (width == BLOCK and truth.shape[1] == BLOCK - 1 and broke < BLOCK - 1 and restart + BLOCK <= self.ids.shape[1]):
            return None
        drafter = s.first_drafter
        given = [(self.features[:, s.start : restart], torch.arange(s.start, restart)), None]
        anchors, begin = [truth[0, broke]] * 2, [restart] * 2
        lengths = [layer.get_seq_length() for layer in s.draft_cache.layers]
        whole = restarts(s, anchors, begin, given, BLOCK)
        # Every layer named: the same pass.
        assert torch.equal(restarts(s, anchors, begin, given, BLOCK, layers=[0, 1, 2]), whole)
        # The last two layers, entered with what the first layer gave in the whole pass: the whole pass.
        kept = {}
        hook = drafter.layers[0].register_forward_hook(lambda layer, came, output: kept.update(first=output))
        restarts(s, anchors, begin, given, BLOCK)
        hook.remove()
        entering = kept["first"].reshape(2, BLOCK, -1)
        assert torch.equal(restarts(s, anchors, begin, given, BLOCK, layers=[1, 2], entering=entering), whole)
        # Fewer layers from the embeddings: another draft (not in every round, but in many).
        self.differs += int(not torch.equal(restarts(s, anchors, begin, given, BLOCK, layers=[2]), whole))
        self.differs += int(not torch.equal(restarts(s, anchors, begin, given, BLOCK, layers=[0, 2]), whole))
        # The cache is left as it was, in the layers that ran and in those that did not.
        assert [layer.get_seq_length() for layer in s.draft_cache.layers] == lengths
        timers = {}
        assert torch.equal(restarts(s, anchors, begin, given, BLOCK, timers=timers), whole)
        assert set(timers) == {"prepare", "pass", "head"} and all(seconds >= 0 for seconds in timers.values())
        with pytest.raises(ValueError):
            restarts(s, anchors, begin, given, BLOCK, layers=[1, 1])
        self.checked += 1
        return None


def test_a_restart_through_some_of_the_drafters_layers():
    from dflash.replay import target_features

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=160)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    partly = Partly(ids, features)
    with_it = replay_decode(draft, target, ids, length, ended=False, features=features, improver=partly, block_size=BLOCK)
    assert partly.checked > 10 and partly.differs > 5
    # Nothing it did changed the decoding.
    assert with_it.acceptance_lengths == replay_decode(draft, target, ids, length, ended=False, features=features,
                                                       block_size=BLOCK).acceptance_lengths


def test_the_probe_of_how_many_layers_a_restart_needs():
    from dflash.replay import target_features
    from nexus import layer_probe
    from nexus.layer_probe import Layers

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=160)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    probe = Layers(draft, width=BLOCK, time_rounds=3, time_every=2, common=torch.arange(32))
    assert [name for name, _, _ in probe.plans] == ["all", "top2", "warm2", "warm2p", "bottom2", "top1", "warm1", "warm1p",
                                                    "bottom1", "without2"]
    assert dict((name, layers) for name, layers, _ in probe.plans) == {
        "all": None, "top2": [1, 2], "warm2": [1, 2], "warm2p": [1, 2], "bottom2": [0, 1], "top1": [2], "warm1": [2],
        "warm1p": [2], "bottom1": [0], "without2": [0, 2]}
    probe.begin(ids, length, features, None, answer=0)
    result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=probe, block_size=BLOCK)
    probe.close()
    assert not any(layer._forward_hooks for layer in draft.layers)
    # The plug-in returns no draft: the rounds are plain decoding's.
    assert result.acceptance_lengths == replay_decode(draft, target, ids, length, ended=False, block_size=BLOCK).acceptance_lengths
    # The whole drafter's restarts are the ones the first probe of restarts counts.
    old = Restart(k=3, width=BLOCK)
    old.begin(ids, length, features, None, answer=0)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=old, block_size=BLOCK)
    table, before = torch.tensor(probe.rows), torch.tensor(old.rows)
    column = {name: table[:, probe.columns.index(name)] for name in probe.columns}
    assert len(table) == len(before) > 10 and table.shape[1] == 5 + 10 * 2
    for mine, theirs in (("where", "where"), ("broke_after", "broke_after"), ("old_rest", "old_rest"), ("all_real", "real"),
                         ("all_none", "none"), ("second_choice_right", "second_choice_right")):
        assert torch.equal(column[mine], before[:, COLUMNS.index(theirs)]), mine
    # A plan that goes on from the first run writes only up to the end of the first block.
    inside = BLOCK - 2 - column["broke_after"]
    for name in ("warm2", "warm2p", "warm1", "warm1p"):
        for kind in ("real", "none"):
            assert bool((column[f"{name}_{kind}"] <= inside).all()) and bool((column[f"{name}_{kind}"] >= 0).all())
    assert any(not torch.equal(column[f"{name}_real"], column["all_real"]) for name in ("top2", "top1", "bottom1", "without2"))
    report = layer_probe.summary(probe.rows, probe.columns, width=BLOCK)
    assert report["rounds"] == len(table) and len(report["all"]["all_real"]) == 3
    assert report["all"]["all_real"][0] == round(float(column["all_real"].float().mean()), 3)
    assert report["all"]["all_real"][1] == round(float(column["all_real"].clamp(max=3).float().mean()), 3)
    assert report["all"]["warm1_real"][0] == report["all"]["warm1_real"][2]
    # The parts of the second run were timed in three rounds.
    assert len(probe.timings) == 3
    times = layer_probe.timing_summary(probe.timings)
    assert times["samples"] == 2
    for name in ("4x16 all: pass", "4x16 all: head", "4x16 all: prepare", "4x16 all: whole", "12x5 top1: pass", "12x16 top2: whole",
                 "USED whole drafter, 8 restarts of 16: whole", "USED whole drafter, 16 restarts of 5: pass",
                 "USED one layer, 16 restarts of 5: whole", "USED one layer, 16 restarts of 5, short list: head",
                 "USED two layers, 16 restarts of 5, short list: decide", "USED one layer, 8 restarts of 16, short list: rows",
                 "first draft made again"):
        assert len(times["all"][name]) == 2 and times["all"][name][0] >= 0, name
    assert "predictor" not in times["all"]
    assert not any(layer._forward_hooks for layer in draft.layers)


def test_the_tiny_second_drafter_and_its_training(tmp_path):
    """The restart layers are copies of the drafter's last layers that go on from the first run. Before
    training they are the drafter's own last layers; training changes only them (and the predictor),
    never the drafter; saved and read again, and with their updates merged in, they write the same."""
    from dflash.replay import target_features
    from dflash.restart_layers import RestartLayers, load_restart_layers, save_restart_layers
    from dflash.stand_in import StatePredictor, project
    from nexus.train_second import ARMS, TeachSecond, first_run, warm_logits

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids.clone()
    features = target_features(target, draft, ids).clone()
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    with pytest.raises(ValueError):
        RestartLayers(draft, count=3)
    with pytest.raises(ValueError):
        RestartLayers(draft, count=4, reveal=True)
    second = RestartLayers(draft, count=1, rank=4)
    assert second.first == 2 and len(second.layers) == 1 and len(second.adapters) == 5
    assert second.layers[0].self_attn.layer_idx == 2 and second.layers[0] is not draft.layers[2]
    trained = sum(value.numel() for value in second.trained())
    assert trained == sum(adapter.down.numel() + adapter.up.numel() for adapter in second.adapters) + 2 * 4 * 16

    # Before training: a pass through the copies is a pass through the drafter's own last layer.
    start = length + 9
    with torch.no_grad():
        final, kept = first_run(draft, target, features, ids, start, 0, BLOCK, keep=1)
        assert final.shape == kept.shape == (BLOCK, 16)
        real = project(draft, features[:, start : start + 3])[0]
        pair = torch.tensor([[int(ids[0, start + 3]), draft.mask_token_id]])
        token_in, mask_in = draft_embeddings(target, pair)[0]
        entering = second.enter(kept[3:], token_in, mask_in)
        assert torch.equal(entering[1:], kept[4:]) and torch.allclose(entering[0], kept[3] + token_in - mask_in)
        mine = warm_logits(draft, second, target, features, ids, start, real, entering)
        mine_before_training = mine.clone()
        positions = torch.cat((torch.arange(0, start + 3), torch.arange(start + 3, start + BLOCK)))[None]
        own = draft(target_hidden=features[:, :start], context_states=real[None], noise_embedding=entering[None],
                    position_ids=positions, layers=[2])
        assert torch.equal(mine, draft.compute_logits(own[:, 1:, :], target.lm_head)[0])
        assert mine.shape == (BLOCK - 3 - 1, 32)

    torch.manual_seed(0)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK)
    before = {name: value.clone() for name, value in predictor.state_dict().items()}
    drafter_before = {name: value.clone() for name, value in draft.state_dict().items()}
    optimizer = torch.optim.AdamW([{"params": second.trained(), "lr": 3e-3}, {"params": list(predictor.parameters()), "lr": 1e-3}])
    teach = TeachSecond(second, predictor, optimizer, draft, target, every=2, window=64, width=BLOCK, compare_every=1)
    # The whole drafter's final states at a restart, put through the output head, are restart_logits.
    from nexus.train_restart import restart_logits
    from nexus.train_second import whole_states

    with torch.no_grad():
        whole = whole_states(draft, target, features, ids, start, real, 0, BLOCK)
        assert torch.equal(draft.compute_logits(whole[None], target.lm_head)[0],
                           restart_logits(draft, target, features, ids, start, real, 0, BLOCK))
    def loss_on_the_same_rounds():
        for at in range(length + 2, length + 14):
            teach.step(ids, features, at, learn=False, count=3)
        return teach.progress()["loss_by_tokens"]

    before_training = loss_on_the_same_rounds()
    for _ in range(8):                                       # learned from these rounds, the loss on them falls
        for at in range(length + 2, length + 14):
            teach.step(ids, features, at, count=3)
    teach.progress()
    assert loss_on_the_same_rounds() < before_training
    curve = []
    for _ in range(5):
        teach.learn_from(ids, features, length, most=12)
        curve.append(teach.progress())
    assert teach.steps >= 5 and all(point["loss_by_tokens"] == point["loss_by_tokens"] for point in curve)   # numbers, not NaN
    assert any(bool(adapter.up.abs().sum() > 0) for adapter in second.adapters)
    assert bool(second.token_up.weight.abs().sum() > 0)
    assert any(not torch.equal(before[name], value) for name, value in predictor.state_dict().items())
    assert all(torch.equal(drafter_before[name], value) for name, value in draft.state_dict().items())    # the drafter is as it was
    compared = curve[-1]["on_the_same_rounds"]
    assert set(compared["tokens_right_inside_the_block"]) == set(ARMS) == set(compared["at_most_3_of_them"])
    assert all(compared["at_most_3_of_them"][name] <= min(compared["tokens_right_inside_the_block"][name], 3) for name in ARMS)
    assert compared["rounds"] > 0 and teach.saved_predictor is not predictor
    # The restart one token earlier is taught and counted wherever the first draft has a right token before the break
    # (here: restarts placed three tokens into the round).
    steps = teach.steps
    for at in range(length + 2, length + 12):
        teach.step(ids, features, at, count=3)
    earlier = teach.progress()["one_token_earlier_from_the_break_on"]
    assert earlier["rounds"] == 10 and teach.steps == steps + 5
    assert set(earlier["tokens_right_inside_the_block"]) == {"second", "whole_predicted", "whole_real"} == set(earlier["at_most_4_of_them"])
    assert all(torch.equal(before[name], value) for name, value in teach.saved_predictor.state_dict().items())

    # With the second aim (the whole drafter's final states) and the real states in a share of the steps.
    aimed = TeachSecond(second, predictor, optimizer, draft, target, every=2, window=64, width=BLOCK, compare_every=1,
                        distill=1.0, real_share=0.5)
    aimed.learn_from(ids, features, length, most=12)
    point = aimed.progress()
    assert aimed.steps >= 2 and point["loss_by_tokens"] == point["loss_by_tokens"] and point["on_the_same_rounds"]["rounds"] > 0
    assert all(torch.equal(drafter_before[name], value) for name, value in draft.state_dict().items())

    # With nothing learned: no weight changes, the same counts are made.
    snapshot = [adapter.up.clone() for adapter in second.adapters]
    steps = teach.steps
    teach.learn_from(ids, features, length, most=6, learn=False)
    assert teach.steps == steps and all(torch.equal(old, adapter.up) for old, adapter in zip(snapshot, second.adapters))
    assert teach.progress()["on_the_same_rounds"]["rounds"] > 0

    # A frozen predictor stays as it is.
    frozen = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK)
    kept_as = {name: value.clone() for name, value in frozen.state_dict().items()}
    other = RestartLayers(draft, count=2, rank=2)
    alone = TeachSecond(other, frozen, torch.optim.AdamW(other.trained(), lr=3e-3), draft, target, every=2, window=64, width=BLOCK,
                        compare_every=2, train_predictor=False, earlier=False)
    alone.learn_from(ids, features, length, most=8)
    assert alone.steps >= 1 and alone.saved_predictor is frozen and "one_token_earlier_from_the_break_on" not in alone.progress()
    assert all(torch.equal(kept_as[name], value) for name, value in frozen.state_dict().items())
    assert other.first == 1 and len(other.layers) == 2

    # Layers that read the tokens: the whole block enters, with the known tokens written in at their places; before
    # training that is the drafter's own last layer on those states. Trained with no states in a share of the steps.
    from nexus.train_second import second_states

    reader = RestartLayers(draft, count=1, rank=4, reveal=True)
    assert reader.reveal and reader.sizes["reveal"] and not second.reveal
    with torch.no_grad():
        known_in = draft_embeddings(target, ids[:, start + 1 : start + 4])[0]
        block = reader.enter_block(kept, known_in, mask_in)
        assert block.shape == kept.shape and torch.equal(block[0], kept[0]) and torch.equal(block[4:], kept[4:])
        assert torch.allclose(block[1:4], kept[1:4] + known_in - mask_in)
        assert torch.equal(reader.enter_block(kept, known_in[:0], mask_in), kept)
        mine = second_states(draft, reader, target, features, ids, start, 3, real, kept)
        positions = torch.cat((torch.arange(0, start + 3), torch.arange(start, start + BLOCK)))[None]
        own = draft(target_hidden=features[:, :start], context_states=real[None], noise_embedding=block[None],
                    position_ids=positions, layers=[2])
        assert mine.shape == (BLOCK - 3 - 1, 16) and torch.equal(mine, own[0, 4:])
        # ... and one token earlier: two tokens written in, the states of two tokens, written from the break on
        before = second_states(draft, reader, target, features, ids, start, 3, real[:2], kept, earlier=True)
        assert before.shape == (BLOCK - 3, 16)
        # the layers that do not read the tokens, through the same function, as before
        assert torch.equal(draft.compute_logits(second_states(draft, RestartLayers(draft, count=1, rank=4), target, features, ids,
                                                              start, 3, real, kept)[None], target.lm_head)[0], mine_before_training)
    reading = TeachSecond(reader, predictor, torch.optim.AdamW(reader.trained() + list(predictor.parameters()), lr=3e-3), draft,
                          target, every=2, window=64, width=BLOCK, compare_every=1, distill=1.0, real_share=0.2, none_share=0.4)
    for _ in range(3):
        reading.learn_from(ids, features, length, most=12)
    for at in range(length + 2, length + 8):
        reading.step(ids, features, at, count=3)
    point = reading.progress()
    assert reading.steps >= 4 and point["loss_by_tokens"] == point["loss_by_tokens"]
    assert "second_none" in point["on_the_same_rounds"]["tokens_right_inside_the_block"]
    assert point["one_token_earlier_from_the_break_on"]["rounds"] >= 6
    assert any(bool(adapter.up.abs().sum() > 0) for adapter in reader.adapters)
    save_restart_layers(str(tmp_path / "warm-reader.pt"), reader)
    assert load_restart_layers(str(tmp_path / "warm-reader.pt"), draft)[0].reveal
    # Layers that stand in for the WHOLE drafter (a whole second pass by a trained reader): before training, with
    # the round's first token alone known and the real states, they write the whole drafter's restart from it ...
    whole = RestartLayers(draft, count=3, rank=4, reveal=True)
    assert whole.first == 0 and len(whole.layers) == 3
    with torch.no_grad():
        _, entered = first_run(draft, target, features, ids, start, 0, BLOCK, keep=-1)
        assert entered.shape == (BLOCK, 16) and torch.equal(entered[1], mask_in.to(entered.dtype))
        # ... one token earlier than a restart after one token is the first draft of the round itself
        first_draft = second_states(draft, whole, target, features, ids, start, 1, real[:0], entered, earlier=False)
        assert first_draft.shape == (BLOCK - 2, 16)
    wholly = TeachSecond(whole, predictor, torch.optim.AdamW(whole.trained() + list(predictor.parameters()), lr=3e-3), draft,
                         target, every=2, window=64, width=BLOCK, compare_every=1)
    for at in range(length + 2, length + 10):
        wholly.step(ids, features, at, count=3)
    point = wholly.progress()
    assert wholly.steps == 4 and point["on_the_same_rounds"]["rounds"] == 8
    # ... and layers that take the states inside the block: going on from the saved reader, the new map starts at nothing
    inside = load_restart_layers(str(tmp_path / "warm-reader.pt"), draft, states_in_block=True)[0].float_parameters().train()
    assert inside.states_in_block and float(inside.state_up.weight.abs().sum()) == 0.0
    with torch.no_grad():
        with_them = second_states(draft, inside, target, features, ids, start, 3, real, kept)
        without = second_states(draft, inside, target, features, ids, start, 3, real[:0], kept)
        assert torch.equal(with_them, without)
    taking = TeachSecond(inside, predictor, torch.optim.AdamW(inside.trained() + list(predictor.parameters()), lr=3e-3), draft,
                         target, every=2, window=64, width=BLOCK, compare_every=1, real_share=0.3, none_share=0.2)
    for at in range(length + 2, length + 14):
        taking.step(ids, features, at, count=3)
    assert taking.steps == 6 and bool(inside.state_up.weight.abs().sum() > 0)
    with torch.no_grad():
        assert not torch.equal(second_states(draft, inside, target, features, ids, start, 3, real, kept),
                               second_states(draft, inside, target, features, ids, start, 3, real[:0], kept))
    save_restart_layers(str(tmp_path / "warm-inside.pt"), inside)
    again_inside = load_restart_layers(str(tmp_path / "warm-inside.pt"), draft)[0]
    assert again_inside.states_in_block and torch.allclose(again_inside.state_up.weight, inside.state_up.weight, atol=1e-2)

    # Saved and read again, and merged for use: the same tokens' scores.
    path = str(tmp_path / "warm-test.pt")
    save_restart_layers(path, second, {"answers": 1})
    again, notes = load_restart_layers(path, draft)
    assert notes == {"answers": 1} and again.sizes == second.sizes
    with torch.no_grad():
        entering = second.enter(kept[3:], token_in, mask_in)
        trained_logits = warm_logits(draft, second, target, features, ids, start, real, entering)
        assert not torch.allclose(trained_logits, mine_before_training)
        read = warm_logits(draft, again, target, features, ids, start, real, again.enter(kept[3:], token_in, mask_in))
        assert torch.allclose(read, trained_logits, atol=1e-2)                     # (saved in 16-bit numbers)
        merged = again.merge()
        assert not merged.adapters and isinstance(merged.layers[0].mlp.up_proj, torch.nn.Linear)
        assert torch.allclose(warm_logits(draft, merged, target, features, ids, start, real, merged.enter(kept[3:], token_in, mask_in)),
                              read, atol=1e-6)
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(True)


def draft_embeddings(target, tokens):
    from dflash.model import _raw_input_embeddings

    return _raw_input_embeddings(target, tokens, 1.0)


class SeenWithFirstRun(Seen):
    """Also keeps the round state and what the drafter's layer `layer` gave in the first draft's pass."""

    def __init__(self, inner, drafter, layer):
        super().__init__(inner)
        self.given, self.checked = None, 0
        self.hook = drafter.layers[layer].register_forward_hook(lambda module, came, output: setattr(self, "given", output))

    def __call__(self, s):
        first_run = self.given[0].clone()
        rows = self.inner(s)
        self.check(s, first_run, rows)
        self.rounds.append((s.start, s.block_ids[0].clone(), s.draft_logits[0].clone(), rows.clone(), list(self.inner.kept)))
        return rows

    def check(self, s, first_run, rows):
        """Each restart is the pass through the drafter's own last layer from the first run's states."""
        width = s.block_ids.shape[1]
        second = s.draft_logits[0].topk(2, dim=-1).indices[:, 1]
        mask_in = draft_embeddings(s.target, torch.tensor([[s.first_drafter.mask_token_id]]))[0, 0]
        for number, broke in enumerate(self.inner.kept):
            places = width - (broke + 1)
            assert rows[number, : broke + 1].tolist() == s.block_ids[0, : broke + 1].tolist()
            assert int(rows[number, broke + 1]) == int(second[broke])
            assert bool((rows[number, width:] == -1).all())                  # nothing past the first block's end
            assert bool((rows[number, broke + 2 : width] >= 0).all())
            if places < 2:
                continue
            entering = first_run[broke + 1 : width].clone()
            entering[0] = entering[0] + draft_embeddings(s.target, second[broke].reshape(1, 1))[0, 0] - mask_in
            stand_in = None
            if self.inner.stand_in == "real":
                stand_in = (self.inner.features[:, s.start : s.start + broke + 1], torch.arange(s.start, s.start + broke + 1))
            alone = restarts(s, [second[broke]], [s.start + broke + 1], [stand_in], places, layers=[len(s.first_drafter.layers) - 1],
                             entering=entering[None])
            assert rows[number, broke + 2 : width].tolist() == alone[0].tolist()
            self.checked += 1


def test_the_second_run_through_the_tiny_second_drafter():
    from dflash.redraft import DraftTree, Restarts, TreeWithRestarts, dflash_generate_redraft
    from dflash.replay import target_features
    from dflash.restart_layers import RestartLayers
    from dflash.stand_in import StatePredictor

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    chain = replay_decode(draft, target, ids, length, ended=False, features=features, block_size=BLOCK)
    second = RestartLayers(draft, count=1, rank=4)

    # Before training the tiny second drafter is the drafter's own last layer: every restart is checked against it.
    for stand_in in ("real", "none"):
        method = Restarts(k=3, stand_in=stand_in)
        method.features, method.second = features, second
        seen = SeenWithFirstRun(method, draft, layer=1)
        result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen, block_size=BLOCK)
        seen.hook.remove()
        assert seen.checked > 30 and sum(result.acceptance_lengths) == sum(chain.acceptance_lengths)
        assert result.rounds <= chain.rounds
        method.start(draft)
        method.start(draft)
        assert len(draft.layers[1]._forward_hooks) == 1                      # one hook, however often it is started
        method.close()
        assert not draft.layers[1]._forward_hooks

    # Trained (here: changed at random), with short blocks, the restart one token earlier, the swap and predicted
    # states, alone and on a tree: real decoding writes the same text.
    torch.manual_seed(5)
    with torch.no_grad():
        for adapter in second.adapters:
            adapter.up.normal_(std=0.3)
        second.token_up.weight.normal_(std=0.3)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    for kind, block, on_a_tree in (("choice", None, False), ("both", 4, False), ("choice", 4, True)):
        method = Restarts(k=3, stand_in="predicted", swap=not on_a_tree, tail=2, kind=kind, block=block, choices=2)
        method.predictor, method.second = predictor, second
        improver = TreeWithRestarts(DraftTree(15), method) if on_a_tree else method
        with_it = dflash_generate_redraft(draft, target, prompt, 200, None, improver=improver, block_size=BLOCK, return_stats=True)
        assert torch.equal(with_it.output_ids, ids)
        assert len(with_it.acceptance_lengths) <= len(plain.acceptance_lengths)
        seen = Seen(method)
        replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen, block_size=BLOCK)
        reach = BLOCK if block is None else block
        for start, first, logits, rows, kept in seen.rounds:
            width = first.shape[0]
            for row in rows.tolist():
                assert all(token == -1 for token in row[width:])
        method.close()
    # The merged copies write what the copies with their updates write.
    method = Restarts(k=3, stand_in="none")
    method.second = second
    before = Seen(method)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=before, block_size=BLOCK)
    method.close()
    method = Restarts(k=3, stand_in="none")
    method.second = second.merge()
    after = Seen(method)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=after, block_size=BLOCK)
    method.close()
    assert len(before.rounds) == len(after.rounds) > 20
    assert sum(int(torch.equal(mine[3], theirs[3])) for mine, theirs in zip(before.rounds, after.rounds)) > 0.9 * len(before.rounds)
    assert not draft.layers[1]._forward_hooks


def test_the_output_head_over_a_short_list_of_tokens(tmp_path):
    """Among the listed tokens the short list picks what the whole head picks; a token of the text or one
    of the first run's choices is found although it is not listed; with every token listed the second run
    writes exactly what it writes with the whole head."""
    from dflash.redraft import Restarts, dflash_generate_redraft
    from dflash.replay import target_features
    from dflash.shortlist import Shortlist, common_tokens, count_tokens, save_counts
    from nexus.redraft_long import make_method, short_list

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    head = target.lm_head
    torch.manual_seed(1)
    states = torch.randn(3, 5, 16, dtype=torch.float64)
    scores = head(states)
    whole = Shortlist(head, torch.arange(32))
    assert torch.equal(whole.best(states), scores.argmax(dim=-1))
    few = Shortlist(head, torch.tensor([3, 7, 11, 30]))
    assert torch.equal(few.best(states), torch.tensor([3, 7, 11, 30])[scores[..., [3, 7, 11, 30]].argmax(dim=-1)])
    best = scores.argmax(dim=-1)
    assert torch.equal(few.best(states, also=torch.arange(32).reshape(4, 8)), best)       # every other token added for the call
    also = torch.tensor([7, 5, 5, 20])                                                    # listed ones and doubles do no harm
    allowed = torch.tensor([3, 7, 11, 30, 5, 20])
    assert torch.equal(few.best(states, also=also), allowed[scores[..., allowed].argmax(dim=-1)])
    assert torch.equal(few.best(states, also=torch.tensor([3, 30])), few.best(states))
    # The best few with their log-probabilities among the listed tokens; with every token listed: the whole head's.
    tokens, likely = whole.best_few(states, 3)
    all_of_it = scores.float().log_softmax(dim=-1).topk(3, dim=-1)
    assert torch.equal(tokens, all_of_it.indices) and torch.allclose(likely, all_of_it.values, atol=1e-5)
    tokens, likely = few.best_few(states, 2, also=also)
    among = scores[..., allowed].float().log_softmax(dim=-1).topk(2, dim=-1)
    assert torch.equal(tokens, allowed[among.indices]) and tokens.shape == (3, 5, 2)
    assert few.best_few(states, 9)[0].shape == (3, 5, 4)

    counts = count_tokens([ids[0, length:], ids[0, length:]], 32)
    assert int(counts.sum()) == 2 * (ids.shape[1] - length) and counts.shape == (32,)
    save_counts(str(tmp_path / "token-counts.pt"), counts, {"answers": 2})
    assert common_tokens(str(tmp_path / "token-counts.pt"), 4).tolist() == counts.topk(4).indices.tolist()
    assert short_list(str(tmp_path), None) is None and short_list(str(tmp_path), 1).shape == (32,)

    features = target_features(target, draft, ids)
    kept = {}
    for name, common in (("whole", None), ("all listed", torch.arange(32)), ("few", counts.topk(6).indices)):
        method = Restarts(k=3, stand_in="none", swap=True, tail=2)
        method.common = common
        seen = Seen(method)
        replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen, block_size=BLOCK)
        kept[name] = [rows for _, _, _, rows, _ in seen.rounds]
        # Real decoding writes the same text with any list.
        with_it = dflash_generate_redraft(draft, target, torch.tensor([[1, 2, 3, 4, 5]]), 200, None, improver=method,
                                          block_size=BLOCK, return_stats=True)
        assert torch.equal(with_it.output_ids, ids)
    assert all(torch.equal(mine, theirs) for mine, theirs in zip(kept["whole"], kept["all listed"]))
    assert any(not torch.equal(mine, theirs) for mine, theirs in zip(kept["whole"], kept["few"]))
    # By name: v1 is a short list of the 1024 most common tokens (here: all 32).
    named = make_method("restart3nst2v1", str(tmp_path))
    assert named.common.shape == (32,) and named.swap and named.alternative_tail == 2 and named.gate is None
    assert make_method("restart3nst2v1g6", str(tmp_path)).gate == 6.0 and make_method("restart3nst2", str(tmp_path)).common is None
    # The parts of the second run are timed when asked.
    method = Restarts(k=3, stand_in="none", swap=True, tail=2)
    method.timers = {}
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=method, block_size=BLOCK)
    assert set(method.timers) == {"decide", "states", "prepare", "pass", "head", "rows"}
    assert all(seconds >= 0 for seconds in method.timers.values())


def test_a_tree_in_which_the_second_run_writes_after_every_node():
    from dflash.redraft import BranchTree, DraftTree, dflash_generate_redraft
    from dflash.replay import target_features
    from dflash.restart_layers import RestartLayers
    from dflash.stand_in import StatePredictor, predicted_for, predicted_states
    from nexus.redraft_long import make_method

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    chain = replay_decode(draft, target, ids, length, ended=False, features=features, block_size=BLOCK)
    with pytest.raises(ValueError):
        BranchTree(8, 12)

    class Kept(Seen):
        def __call__(self, s):
            rows = self.inner(s)
            self.rounds.append((s, rows.clone(), self.inner.nodes))
            return rows

    # With the whole budget in the first tree it is DDTree.
    same, tree = Kept(BranchTree(12, 12, stand_in="none")), Kept(DraftTree(12))
    first = replay_decode(draft, target, ids, length, ended=False, features=features, improver=same, block_size=BLOCK)
    second = replay_decode(draft, target, ids, length, ended=False, features=features, improver=tree, block_size=BLOCK)
    assert first.acceptance_lengths == second.acceptance_lengths
    for (_, mine, nodes), (_, theirs, their_nodes) in zip(same.rounds, tree.rounds):
        assert sorted(mine.tolist()) == sorted(theirs.tolist()) and nodes == their_nodes

    def prefixes(rows):
        found = set()
        for row in rows.tolist():
            for end in range(2, len(row) + 1):
                if row[end - 1] < 0:
                    break
                found.add(tuple(row[:end]))
        return found

    # A first tree of 8 nodes grown to 14 by what the second run writes: the first tree's nodes are all there,
    # the tree has at most 14 nodes, and with the target's real states it gets at least as far as the first tree.
    small = Kept(DraftTree(8))
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=small, block_size=BLOCK)
    grown_tokens = {}
    for stand_in in ("real", "none"):
        method = BranchTree(14, 8, tail=2, stand_in=stand_in)
        method.features = features
        kept = Kept(method)
        result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=kept, block_size=BLOCK)
        grew = 0
        for (s, rows, nodes), (_, base, base_nodes) in zip(kept.rounds, small.rounds):
            if rows.shape[1] != BLOCK:
                continue
            assert prefixes(base) <= prefixes(rows) and len(prefixes(rows)) == nodes <= 14 and base_nodes <= 8
            assert bool((rows[:, 0] == rows[0, 0]).all())
            grew += int(nodes > base_nodes)
        assert grew > 20 and method.counts["nodes_from_the_second_run"] > 100
        assert method.counts["rounds"] == len(kept.rounds)
        grown_tokens[stand_in] = sum(result.acceptance_lengths)
        assert sum(result.acceptance_lengths) == sum(chain.acceptance_lengths) and result.rounds <= chain.rounds
        if stand_in == "real":
            with_real = result.rounds
    small_result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=DraftTree(8), block_size=BLOCK)
    assert with_real <= small_result.rounds

    # Predicted states for any path: for the first draft's own tokens they are the ones predicted_states writes.
    torch.manual_seed(2)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK, coming=True)
    s = kept.rounds[3][0]
    block = s.block_ids[0].tolist()
    with torch.inference_mode():
        own = predicted_states(predictor, s, 4, coming=block[4])[0]
        again = predicted_for(predictor, s, [0, 1, 2, 3], block[:4], block[1:5])
        other = predicted_for(predictor, s, [0, 1, 2, 3], block[:4], block[1:4] + [(block[4] + 1) % 30])
    assert torch.allclose(own, again, atol=1e-6)
    assert torch.allclose(other[:3], again[:3], atol=1e-6) and not torch.allclose(other[3], again[3], atol=1e-6)

    # In real decoding, with predicted states, through the whole drafter and through a tiny second drafter: the same text.
    layers = RestartLayers(draft, count=1, rank=4)
    with torch.no_grad():
        for adapter in layers.adapters:
            adapter.up.normal_(std=0.3)
    for through in (None, layers):
        method = BranchTree(14, 8, tail=2, stand_in="predicted", block=4)
        method.predictor, method.second = predictor, through
        with_it = dflash_generate_redraft(draft, target, torch.tensor([[1, 2, 3, 4, 5]]), 200, None, improver=method,
                                          block_size=BLOCK, return_stats=True)
        assert torch.equal(with_it.output_ids, ids) and len(with_it.acceptance_lengths) <= len(plain.acceptance_lengths)
        assert method.counts["nodes_from_the_second_run"] > 0
        method.close()
    assert not any(layer._forward_hooks for layer in draft.layers)
    # By name.
    named = make_method("btree63f40t2c3b8r", "/nowhere")
    assert (named.budget, named.first, named.tail, named.choices, named.block, named.stand_in) == (63, 40, 2, 3, 8, "real")
    assert make_method("btree31f20t1n", "/nowhere").choices == 2 and named.common is None
    # With a short list that holds every token the tree is the same tree.
    listed = BranchTree(14, 8, tail=2, stand_in="none", block=4)
    listed.common = torch.arange(32)
    bare = BranchTree(14, 8, tail=2, stand_in="none", block=4)
    one, two = Kept(listed), Kept(bare)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=one, block_size=BLOCK)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=two, block_size=BLOCK)
    assert sum(int(sorted(mine.tolist()) == sorted(theirs.tolist())) for (_, mine, _), (_, theirs, _) in zip(one.rounds, two.rounds)) \
        > 0.9 * len(one.rounds)


def test_the_second_run_through_layers_that_read_the_tokens():
    """Restarts and the tree with restarts at every node, through restart layers that read the tokens: before
    training each restart is the drafter's own last layer on the round's block with the known tokens written in;
    trained (here: changed at random), real decoding still writes the same text."""
    from dflash.redraft import BranchTree, DraftTree, Restarts, TreeWithRestarts, dflash_generate_redraft
    from dflash.replay import target_features
    from dflash.restart_layers import RestartLayers
    from dflash.stand_in import StatePredictor

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    chain = replay_decode(draft, target, ids, length, ended=False, features=features, block_size=BLOCK)
    reader = RestartLayers(draft, count=1, rank=4, reveal=True)

    class Checked(Seen):
        """Every restart of every round against a pass made by hand."""

        def __init__(self, inner, drafter):
            super().__init__(inner)
            self.given, self.checked = None, 0
            self.hook = drafter.layers[1].register_forward_hook(lambda module, came, output: setattr(self, "given", output))

        def __call__(self, s):
            first_run = self.given[0].clone()
            rows = self.inner(s)
            width, block = s.block_ids.shape[1], s.block_ids[0]
            second = s.draft_logits[0].topk(2, dim=-1).indices[:, 1]
            mask_in = draft_embeddings(s.target, torch.tensor([[s.first_drafter.mask_token_id]]))[0, 0]
            for number, broke in enumerate(self.inner.kept):
                place = broke + 1                                  # the restart's first token: the second choice there
                if place >= width - 1:
                    continue
                known = torch.cat((block[1:place], second[broke : broke + 1]))
                entering = first_run.clone()
                entering[1 : place + 1] += draft_embeddings(s.target, known[None])[0] - mask_in
                given = None
                if self.inner.stand_in == "real":
                    given = self.inner.features[:, s.start : s.start + place]
                positions = torch.cat((torch.arange(s.start, s.start + (place if given is not None else 0)),
                                       torch.arange(s.start, s.start + width)))[None]
                seen = s.draft_cache.get_seq_length()
                hidden = s.first_drafter(target_hidden=s.context_features[:, :0] if given is None else given,
                                         noise_embedding=entering[None], position_ids=positions, past_key_values=s.draft_cache,
                                         use_cache=True, layers=[2])
                s.draft_cache.layers[2].crop(-(s.draft_cache.layers[2].get_seq_length() - seen))
                by_hand = s.first_drafter.compute_logits(hidden[:, -width:][:, place + 1 :], s.target.lm_head)[0].argmax(dim=-1)
                assert rows[number, : place].tolist() == block[:place].tolist() and int(rows[number, place]) == int(second[broke])
                assert rows[number, place + 1 : width].tolist() == by_hand.tolist()
                assert bool((rows[number, width:] == -1).all())
                self.checked += 1
            self.rounds.append((s.start, block.clone(), s.draft_logits[0].clone(), rows.clone(), list(self.inner.kept)))
            return rows

    for stand_in in ("real", "none"):
        method = Restarts(k=3, stand_in=stand_in)
        method.features, method.second = features, reader
        checked = Checked(method, draft)
        result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=checked, block_size=BLOCK)
        checked.hook.remove()
        method.close()
        assert checked.checked > 30 and sum(result.acceptance_lengths) == sum(chain.acceptance_lengths)
        assert result.rounds <= chain.rounds

    # Layers that take the states inside the block: before training the states add nothing (the rows are the rows
    # without states); once the map is trained (here: set at random) they change what is written.
    with pytest.raises(ValueError):
        RestartLayers(draft, count=1, rank=4, states_in_block=True)
    inside = RestartLayers(draft, count=1, rank=4, reveal=True, states_in_block=True)
    assert inside.states_in_block and inside.sizes["states_in_block"] and not reader.states_in_block
    kept_rows = {}
    for name, stand_in, noise in (("none", "none", 0.0), ("real, untrained map", "real", 0.0), ("real, a map", "real", 0.5)):
        if noise:
            torch.manual_seed(11)
            with torch.no_grad():
                inside.state_up.weight.normal_(std=noise)
        for make in (lambda: Restarts(k=3, stand_in=stand_in), lambda: BranchTree(14, 8, tail=2, stand_in=stand_in)):
            method = make()
            method.features, method.second = features, inside
            seen = Seen(method) if isinstance(method, Restarts) else None
            result = replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen or method,
                                   block_size=BLOCK)
            method.close()
            assert sum(result.acceptance_lengths) == sum(chain.acceptance_lengths)
            if seen is not None:
                kept_rows[name] = [rows for _, _, _, rows, _ in seen.rounds]
    assert all(torch.equal(mine, theirs) for mine, theirs in zip(kept_rows["none"], kept_rows["real, untrained map"]))
    assert any(not torch.equal(mine, theirs) for mine, theirs in zip(kept_rows["none"], kept_rows["real, a map"]))

    torch.manual_seed(7)
    with torch.no_grad():
        for adapter in reader.adapters:
            adapter.up.normal_(std=0.3)
        reader.token_up.weight.normal_(std=0.3)
        for adapter in inside.adapters:
            adapter.up.normal_(std=0.3)
    predictor = StatePredictor(draft.config.hidden_size, inner=32, hidden=64, block=BLOCK)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    methods = []
    whole = RestartLayers(draft, count=3, rank=4, reveal=True, states_in_block=True)      # a whole second pass by a reader
    with torch.no_grad():
        for adapter in whole.adapters:
            adapter.up.normal_(std=0.3)
        whole.state_up.weight.normal_(std=0.3)
    for through in (inside, whole):
        for make in (lambda: Restarts(k=3, stand_in="predicted", swap=True, tail=2, kind="both", block=4, choices=2),
                     lambda: BranchTree(14, 8, tail=2, stand_in="predicted")):
            method = make()
            method.predictor, method.second = predictor, through
            methods.append((method, method))
    for kind, block, on_a_tree in (("choice", None, False), ("both", 4, False), ("choice", 4, True)):
        method = Restarts(k=3, stand_in="predicted", swap=not on_a_tree, tail=2, kind=kind, block=block, choices=2)
        method.predictor, method.second = predictor, reader
        methods.append((TreeWithRestarts(DraftTree(15), method) if on_a_tree else method, method))
    for stand_in in ("predicted", "none"):
        tree = BranchTree(14, 8, tail=2, stand_in=stand_in)
        tree.predictor, tree.second = predictor, reader
        methods.append((tree, tree))
    for improver, method in methods:
        with_it = dflash_generate_redraft(draft, target, prompt, 200, None, improver=improver, block_size=BLOCK, return_stats=True)
        assert torch.equal(with_it.output_ids, ids) and len(with_it.acceptance_lengths) <= len(plain.acceptance_lengths)
        seen = Seen(method) if isinstance(method, Restarts) else None
        if seen is not None:                                   # nothing is written past the first block's end
            replay_decode(draft, target, ids, length, ended=False, features=features, improver=seen, block_size=BLOCK)
            assert all(token == -1 for _, first, _, rows, _ in seen.rounds for row in rows.tolist() for token in row[first.shape[0] :])
        else:
            assert method.counts["nodes_from_the_second_run"] > 0
        method.close()
    assert not any(layer._forward_hooks for layer in draft.layers)


def test_text_written_many_answers_at_once():
    """nexus/gen_text.py: a batch of prompts gets the answers each prompt gets alone, known prompts are left out,
    and a batch holds as many rows as the memory allows."""
    from nexus.gen_text import batches_by_memory, cut_at_stop, without_known, write_batch

    target, _ = make_models(seed=5)
    prompts = [("a", [3, 4, 5]), ("b", [6, 7, 8, 9, 10, 11]), ("c", [12]), ("a", [3, 4, 5]), ("d", [13, 14])]
    fresh = without_known(prompts, known=[[12], [1, 2]])
    assert fresh == [("a", [3, 4, 5]), ("b", [6, 7, 8, 9, 10, 11]), ("d", [13, 14])]

    batches = batches_by_memory(fresh, max_new_tokens=6, row_tokens=20)
    assert [[len(tokens) for _, tokens in batch] for batch in batches] == [[2, 3], [6]]     # 2 x (3 + 6) <= 20 < 3 x (6 + 6)
    assert batches_by_memory(fresh, max_new_tokens=6, row_tokens=1000, largest=2)[0] == [("d", [13, 14]), ("a", [3, 4, 5])]

    assert cut_at_stop([4, 5, 30, 0, 0], {30}) == ([4, 5, 30], True)
    assert cut_at_stop([4, 5, 6], {30}) == ([4, 5, 6], False)

    together = write_batch(target, fresh, pad=0, stops={30}, max_new_tokens=6, device=torch.device("cpu"))
    for (_, tokens), (answer, stopped) in zip(fresh, together):
        alone = target.generate(torch.tensor([tokens]), do_sample=False, num_beams=1, max_new_tokens=6,
                                pad_token_id=0, eos_token_id=[30])[0, len(tokens):].tolist()
        assert answer == cut_at_stop(alone, {30})[0]
        assert stopped == (30 in alone)
        assert 1 <= len(answer) <= 6


def test_training_takes_more_folders_of_text(tmp_path):
    """nexus/train_second.py: answers of other folders are learned from, but none whose prompt is held out, none
    twice, and what is written later is found at the next reading."""
    from nexus.train_second import more_answers

    def write(folder, name, rows):
        folder.mkdir(exist_ok=True)
        torch.save({"sequences": [torch.tensor(prompt + answer, dtype=torch.int32) for prompt, answer in rows],
                    "prompt_lengths": [len(prompt) for prompt, _ in rows], "sources": ["x"] * len(rows)}, folder / name)

    one, two = tmp_path / "one", tmp_path / "two"
    write(one, "shard0of2.pt", [([3, 4], [5, 6, 7]), ([8, 9, 10], [11, 12])])
    write(two, "shard0of1.pt", [([3, 4], [9, 9, 9]), ([13], [14, 15])])
    seen = set()
    first = more_answers([str(one), str(two)], held_out={(8, 9, 10)}, seen=seen)
    assert [answer[2].tolist() for answer in first] == [[3, 4, 5, 6, 7], [3, 4, 9, 9, 9], [13, 14, 15]]
    assert more_answers([str(one), str(two)], held_out={(8, 9, 10)}, seen=seen) == []

    write(one, "shard1of2.pt", [([16, 17], [18, 19])])                                  # written while the training runs
    write(one, "shard0of2.pt", [([3, 4], [5, 6, 7]), ([8, 9, 10], [11, 12]), ([20], [21, 22])])
    later = more_answers([str(one), str(two)], held_out={(8, 9, 10)}, seen=seen)
    assert [answer[2].tolist() for answer in later] == [[20, 21, 22], [16, 17, 18, 19]]


def _round_state(logits, first_token=3, text=(1, 2)):
    from types import SimpleNamespace

    width = logits.shape[0] + 1
    return SimpleNamespace(draft_logits=logits[None], block_ids=torch.tensor([[first_token] + [31] * (width - 1)]),
                           accepted_ids=torch.tensor([list(text)]), start=len(text))


def test_counts_of_which_token_follows_which():
    """dflash/follow.py: counts of one, two and three tokens, smoothed, and the lift of a token after two others."""
    from dflash.follow import FollowCounts

    text = torch.tensor([9, 9, 1, 2, 3, 1, 2, 3, 1, 2, 4])
    counts = FollowCounts.count([(text, 2)], vocabulary=10, strength=2.0, clip=10.0)
    assert counts.tokens == 9 and counts.sizes()["different_triples"] == 6
    anywhere = lambda token: (int((text[2:] == token).sum()) + 1) / (9 + 10)
    assert abs(float(counts.anywhere[3]) - anywhere(3)) < 1e-6
    # after (1, 2): 3 twice, 4 once.   after 2 alone: 3 twice, 4 once.
    one = (2 + 2.0 * anywhere(3)) / (3 + 2.0)
    two = (2 + 2.0 * one) / (3 + 2.0)
    got = counts.after_two(torch.tensor(1), torch.tensor(2), torch.tensor(3))
    assert abs(float(got) - two) < 1e-6
    lift = counts.lift(torch.tensor([1]), torch.tensor([2]), torch.tensor([3, 4, 5]))
    assert lift.shape == (3,) and lift[0] > lift[1] > lift[2]            # seen twice, once, never
    assert abs(float(lift[0]) - math.log(two / anywhere(3))) < 1e-5
    never = counts.lift(torch.tensor([7]), torch.tensor([8]), torch.tensor([3]))    # an unseen pair: what is known of 3 alone
    assert abs(float(never) - math.log(((0 + 2.0 * anywhere(3)) / (0 + 2.0)) / anywhere(3))) < 1e-5
    a, b, c = torch.tensor([1, 9, 3]).view(1, 3, 1, 1), torch.tensor([2, 1]).view(1, 1, 2, 1), torch.tensor([3, 4, 5, 1]).view(1, 1, 1, 4)
    assert torch.allclose(counts.lift_of_places(a, b, c), counts.lift(a, b, c), atol=1e-6)     # all places in one lookup
    assert counts.lift_of_places(a, b, c).shape == (1, 3, 2, 4)
    assert float(FollowCounts.count([(text, 2)], 10, strength=2.0, clip=0.5).lift(
        torch.tensor([1]), torch.tensor([2]), torch.tensor([3]))) == 0.5


def test_tree_with_counts():
    """dflash/redraft.py: CountTree. With weight 0 it is DDTree; with counts, the tokens the target writes after
    a branch's own tokens move up; its nodes are whole paths; BranchTree takes it as its first tree."""
    from dflash.follow import FollowCounts
    from dflash.redraft import BranchTree, CountTree, DraftTree

    torch.manual_seed(0)
    logits = torch.randn(5, 32) * 2.0
    s = _round_state(logits)
    nothing = FollowCounts.count([], vocabulary=32)
    plain, counted = DraftTree(12), CountTree(12, nothing, candidates=12, on_path=0.0, off_path=0.0)
    assert sorted(map(tuple, plain(s).tolist())) == sorted(map(tuple, counted(s).tolist()))
    assert plain.nodes == counted.nodes == 12

    # The first run: at the first place tokens 10 (first choice) and 11; at the second place 20, then 21.
    # The target's own text: after (3, 11) it writes 21, after (3, 10) it writes 20.
    logits = torch.full((3, 32), -8.0)
    logits[0, 10], logits[0, 11] = 3.0, 2.5
    logits[1, 20], logits[1, 21] = 3.0, 2.0
    logits[2, 25] = 3.0
    s = _round_state(logits)
    text = torch.tensor([0, 0] + [3, 11, 21, 3, 10, 20] * 30)
    follow = FollowCounts.count([(text, 2)], vocabulary=32)
    best = logits.log_softmax(-1).topk(8, dim=-1)
    marginal, _ = DraftTree(6).choose(s, best, best.values.tolist())
    assert (1, 0) in marginal and (1, 1) not in marginal              # DDTree: after 11 it offers 20, the first choice
    tree = CountTree(6, follow, candidates=8)
    chosen, worth = tree.choose(s, best, best.values.tolist())
    assert (1, 1) in chosen and (0, 0) in chosen                       # after 11 comes 21, after 10 still 20
    assert worth[(1, 1)] > worth.get((1, 0), -1e9)
    assert all(node[:-1] in chosen for node in chosen if len(node) > 1)             # every node has its parent
    assert all(worth[node] <= worth[node[:-1]] + 1e-9 for node in chosen if len(node) > 1)
    rows = tree(s).tolist()
    assert tree.nodes == 6 and [3, 11, 21] in [row[:3] for row in rows]
    with pytest.raises(ValueError):
        CountTree(6)(s)

    both = BranchTree(8, 6, tail=1, stand_in="none")                   # the first tree of the every-node tree
    both.tree = CountTree(6, follow, candidates=8)
    plain_first, counted_first = BranchTree(6, 6, stand_in="none"), BranchTree(6, 6, stand_in="none")
    counted_first.tree = CountTree(6, follow, candidates=8)
    assert [3, 11, 21] in [row[:3] for row in counted_first(s).tolist()]
    assert [3, 11, 21] not in [row[:3] for row in plain_first(s).tolist()]


def test_names_of_trees_with_counts():
    from dflash.follow import FollowCounts
    from dflash.redraft import BranchTree, CountTree
    from nexus import redraft_long

    assert redraft_long.with_counts("ctree63") and redraft_long.with_counts("cbtree63f40t2n@~opd-r1")
    assert redraft_long.with_counts("plain+ctree63l20_40") and not redraft_long.with_counts("btree63f40t2n")
    assert not redraft_long.with_counts("tree63") and not redraft_long.with_counts("plain+tree63_restart4pxt3@opd-m0")
    redraft_long.FOLLOW["counts"] = FollowCounts.count([], vocabulary=32)
    try:
        tree = redraft_long.make_method("ctree63l20_40m8", "/nowhere")
        assert isinstance(tree, CountTree) and (tree.budget, tree.on_path, tree.off_path, tree.candidates) == (63, 0.2, 0.4, 8)
        assert redraft_long.make_method("ctree31", "/nowhere").off_path == 0.6
        both = redraft_long.make_method("cbtree63f40t2n", "/nowhere")
        assert isinstance(both, BranchTree) and isinstance(both.tree, CountTree) and both.tree.budget == 40
        assert not isinstance(redraft_long.make_method("btree63f40t2n", "/nowhere").tree, CountTree)
        rows_too = redraft_long.make_method("ctree39_echo4r4", "/nowhere")          # the tree with counts and more rows
        assert isinstance(rows_too.tree, CountTree) and rows_too.tree.budget == 39
        assert not isinstance(redraft_long.make_method("tree39_echo4r4", "/nowhere").tree, CountTree)
        assert redraft_long.with_counts("ctree39_echo4r4") and redraft_long.with_counts("plain:window=4096+ctree63_echo4r4")
    finally:
        redraft_long.FOLLOW.clear()


def test_the_second_drafter_trained_at_every_place(tmp_path):
    """nexus/train_every.py: many rounds of an answer in one pass, the answer's own tokens written in up to a
    depth, and the second drafter's correction of the first run's candidates after them."""
    from dflash.follow import FollowCounts
    from dflash.replay import target_features
    from dflash.restart_layers import CandidateScores, RestartLayers, candidate_scores, load_restart_layers, save_restart_layers
    from dflash.model import _output_head
    from nexus.train_every import TeachEvery, first_runs, halves, second_rows
    from nexus.train_second import first_run, second_states

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids.clone()
    features, said = target_features(target, draft, ids, top=6)
    features, said = features.clone(), (said[0].clone(), said[1].clone())
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    second = RestartLayers(draft, count=1, rank=4, reveal=True, scores=True)
    assert second.sizes["scores"] and "scores" not in RestartLayers(draft, count=1, rank=4, reveal=True).sizes

    # Several rounds in one pass are the rounds one by one: each sees the context before its own start only.
    starts = torch.tensor([length + 30, length + 4, length + 17])
    with torch.no_grad():
        final, kept = first_runs(draft, target, features, ids, starts, 0, keep=1, width=BLOCK)
        for at, start in enumerate(starts.tolist()):
            one, one_kept = first_run(draft, target, features, ids, start, 0, BLOCK, keep=1)
            assert torch.allclose(final[at], one, atol=1e-4) and torch.allclose(kept[at], one_kept, atol=1e-4)

        # A row with the whole block in sight is the layers' pass with the tokens written in (train_second.py).
        depths = torch.tensor([3, 1, 6])
        rows, inside = second_rows(draft, second, target, features, ids, starts, depths, kept, 0, reach=BLOCK, width=BLOCK)
        for at, (start, depth) in enumerate(zip(starts.tolist(), depths.tolist())):
            one = second_states(draft, second, target, features, ids, start, depth, features.new_zeros(0, 16), kept[at])
            assert torch.allclose(rows[at, : BLOCK - 1 - depth], one, atol=1e-4)
            assert inside[at].tolist() == [slot < BLOCK - 1 - depth for slot in range(BLOCK)]
        # With a reach of 2 a row sees its places up to two after the tokens, as a node's block in a tree does.
        near, inside = second_rows(draft, second, target, features, ids, starts, depths, kept, 0, reach=2, width=BLOCK)
        assert near.shape == (3, 2, 16) and inside.tolist() == [[depth + 1 + slot <= BLOCK - 1 for slot in range(2)] for depth in depths.tolist()]
        assert not torch.allclose(near[0], rows[0, :2], atol=1e-4)

    # The scores: with nothing changed by the second run they are the first run's own probabilities of the
    # candidates and of "another token"; the counts move a candidate, more so off the first run's path.
    scorer = CandidateScores(16)
    first_log = torch.tensor([[0.5, 0.2, 0.1]]).log()
    plain_scores = scorer(first_log, torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 16))
    assert torch.allclose(plain_scores.exp(), torch.tensor([[0.5, 0.2, 0.1, 0.2]]), atol=1e-5)
    lift = torch.tensor([[0.0, 2.0, 0.0]])
    on = scorer(first_log, torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 16), lift, torch.tensor([False]))
    off = scorer(first_log, torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 16), lift, torch.tensor([True]))
    assert plain_scores[0, 1] < on[0, 1] < off[0, 1] and torch.allclose(on.exp().sum(), torch.tensor(1.0), atol=1e-5)
    better = scorer(first_log, torch.zeros(1, 3), torch.tensor([[0.0, 0.0, 3.0]]), torch.zeros(1, 16))
    assert better[0, 2] > plain_scores[0, 2] and better[0, 3] < plain_scores[0, 3]
    head = _output_head(target)
    some = torch.tensor([[1, 5, 9]])
    assert torch.allclose(candidate_scores(draft, head, final[:1, 2], some), draft.compute_logits(final[:1, 2], head)[0, some[0]][None], atol=1e-4)

    # Training: the loss falls on the rounds it has seen, only the layers and the scores' numbers change.
    kept_as = {name: value.clone() for name, value in draft.state_dict().items()}
    follow = FollowCounts.count([(ids[0], length)], target.config.vocab_size)
    optimizer = torch.optim.AdamW(second.trained(), lr=3e-3)
    teach = TeachEvery(second, optimizer, draft, target, candidates=5, reach=2, group=6, depths=3, window=64, width=BLOCK, warm=1)
    losses = []
    for _ in range(12):
        teach.generator.manual_seed(0)                                # the same rounds and depths every time
        teach.learn_from(ids, features, said, length, 12, follow)
        losses.append(teach.progress()["loss"])
    assert losses[-1] < losses[0] and teach.updates == 24 and teach.rows_seen == 24 * 6 * 3
    assert all(torch.equal(kept_as[name], value) for name, value in draft.state_dict().items())
    teach.learn_from(ids, features, said, length, 0, follow, learn=False, every=7)
    point = teach.progress()
    assert teach.updates == 24 and {"first", "first_counts", "second", "second_counts", "covered", "second_counts_walk", "first_walk"} <= set(point)
    assert 0.0 <= point["second_counts"] <= point["covered"] <= 1.0 and 0.0 <= point["first_walk"] <= BLOCK - 1
    without = TeachEvery(second, None, draft, target, candidates=5, reach=2, group=6, width=BLOCK, window=64)
    without.learn_from(ids, features, said, length, 0, None, learn=False, every=7)
    assert "first_counts" not in without.progress()

    # Saved and read again, the layers and the scores are the same.
    path = str(tmp_path / "warm-every.pt")
    save_restart_layers(path, second, {"trained_by": "train_every"})
    again, notes = load_restart_layers(path, draft)
    assert again.scores is not None and notes["trained_by"] == "train_every"
    assert torch.allclose(again.scores.rest.weight, second.scores.rest.weight) and torch.allclose(again.scores.on_path, second.scores.on_path)
    with torch.no_grad():
        mine, _ = second_rows(draft, second.eval(), target, features, ids, starts, depths, kept, 0, reach=2, width=BLOCK)
        theirs, _ = second_rows(draft, again, target, features, ids, starts, depths, kept, 0, reach=2, width=BLOCK)
    assert torch.allclose(mine, theirs, atol=2e-2)                    # (saved in 16-bit numbers)
    assert halves([("a", 0, torch.tensor([1, 2, 3, 9]), 3, ""), ("a", 1, torch.tensor([1, 2, 4, 9]), 3, "")]) == [0, 1]

    # The two controls. full: the copy is the drafter's layer, all of it trained. fresh: a layer from new weights,
    # whose correction is read through a map that starts at nothing, so that its scores start as the first run's.
    whole = RestartLayers(draft, count=1, reveal=True, scores=True, full=True)
    new = RestartLayers(draft, count=1, reveal=True, scores=True, fresh=True)
    plain_copy = RestartLayers(draft, count=1, rank=4, reveal=True, scores=True)
    assert not whole.adapters and not new.adapters and whole.scores.out is None and new.scores.out is not None
    layer = sum(value.numel() for value in draft.layers[2].parameters())
    assert sum(value.numel() for value in whole.trained()) > layer > sum(value.numel() for value in plain_copy.trained())
    with torch.no_grad():
        as_drafter, _ = second_rows(draft, plain_copy, target, features, ids, starts, depths, kept, 0, reach=2, width=BLOCK)
        same, _ = second_rows(draft, whole, target, features, ids, starts, depths, kept, 0, reach=2, width=BLOCK)
        other, _ = second_rows(draft, new, target, features, ids, starts, depths, kept, 0, reach=2, width=BLOCK)
    assert torch.allclose(same, as_drafter, atol=1e-5) and not torch.allclose(other, as_drafter, atol=1e-2)
    for control in (whole, new):
        teach = TeachEvery(control, torch.optim.AdamW(control.trained(), lr=3e-3), draft, target, candidates=5, reach=2, group=6,
                           depths=3, window=64, width=BLOCK, warm=1)
        losses = []
        for _ in range(8):
            teach.generator.manual_seed(0)
            teach.learn_from(ids, features, said, length, 12, follow)
            losses.append(teach.progress())
        assert losses[-1]["loss"] < losses[0]["loss"]
        if control is new:                            # before anything is learned its scores are the first run's with the counts
            assert abs(losses[0]["second_counts_log"] - losses[0]["first_counts_log"]) < 0.2
    assert all(torch.equal(kept_as[name], value) for name, value in draft.state_dict().items())


def test_a_tree_whose_edges_the_second_drafter_scores(tmp_path):
    """dflash/redraft.py: SecondTree. One pass of the second drafter over a first tree, then the tree chosen
    anew by the corrected probabilities; what the pass gives a node is what the training asked for there."""
    from dflash.follow import FollowCounts
    from dflash.model import _output_head
    from dflash.redraft import CountTree, DraftTree, SecondTree, dflash_generate_redraft
    from dflash.replay import target_features
    from dflash.restart_layers import RestartLayers, candidate_scores, save_restart_layers
    from nexus.redraft_long import FOLLOW, make_method, with_counts
    from nexus.train_every import first_runs, second_rows

    target, draft = make_models(layers=3)
    plain, length = answer(target, draft, budget=200)
    ids = plain.output_ids
    features = target_features(target, draft, ids)
    second = RestartLayers(draft, count=1, rank=4, reveal=True, scores=True).eval()
    with torch.no_grad():
        for adapter in second.adapters:
            adapter.up.normal_(std=0.3)

    class Kept(Seen):
        def __call__(self, s):
            rows = self.inner(s)
            self.rounds.append((s.start, rows.clone(), self.inner.nodes, getattr(self.inner, "last", None)))
            return rows

    def prefixes(rows):
        found = set()
        for row in rows.tolist():
            for end in range(2, len(row) + 1):
                if row[end - 1] < 0:
                    break
                found.add(tuple(row[:end]))
        return found

    with pytest.raises(ValueError):
        replay_decode(draft, target, ids, length, ended=False, features=features, improver=SecondTree(12), block_size=BLOCK)
    # A second drafter whose change counts for nothing leaves the first run's probabilities: the tree is DDTree.
    with torch.no_grad():
        second.scores.strength.zero_()
    neutral = SecondTree(12, top=16)                                  # (every token DDTree may pick is a candidate)
    neutral.second = second
    same, tree = Kept(neutral), Kept(DraftTree(12))
    first = replay_decode(draft, target, ids, length, ended=False, features=features, improver=same, block_size=BLOCK)
    other = replay_decode(draft, target, ids, length, ended=False, features=features, improver=tree, block_size=BLOCK)
    alike = sum(int(prefixes(mine) == prefixes(theirs)) for (_, mine, _, _), (_, theirs, _, _) in zip(same.rounds, tree.rounds))
    assert alike > 0.9 * len(tree.rounds) and abs(sum(first.acceptance_lengths) - sum(other.acceptance_lengths)) <= 2
    assert neutral.counts["rounds"] == len(same.rounds) and neutral.counts["rows_of_the_pass"] > 0
    neutral.close()

    # With its change counted the tree differs, has at most the budget's nodes, and every node's parent is in it.
    with torch.no_grad():
        second.scores.strength.fill_(1.0)
    method = SecondTree(12, 8, tail=2, top=8)
    method.second, method.keep_last = second, True
    kept = Kept(method)
    replay_decode(draft, target, ids, length, ended=False, features=features, improver=kept, block_size=BLOCK)
    assert all(nodes <= 12 and len(prefixes(rows)) == nodes for _, rows, nodes, _ in kept.rounds)
    assert sum(int(prefixes(mine) != prefixes(theirs)) for (_, mine, _, _), (_, theirs, _, _) in zip(kept.rounds, tree.rounds)) > 0
    assert method.counts["nodes_from_the_second_run"] > 0 and method.counts["first_tree_nodes_dropped"] >= 0

    # What the pass gave a node on the answer's own path is what the training computes for that round and depth.
    head, checked = _output_head(target), 0
    for start, rows, nodes, last in kept.rounds:
        for ranks, (values, indices) in last["order"].items():
            depth = len(ranks)
            if checked >= 6:
                break
            wrote = [last["tokens"][place][rank] for place, rank in enumerate(ranks)]
            if not depth or wrote != ids[0, start + 1 : start + 1 + depth].tolist() or start + BLOCK >= ids.shape[1] or len(values) < 8:
                continue
            with torch.no_grad():
                starts, depths = torch.tensor([start]), torch.tensor([depth])
                final, entered = first_runs(draft, target, features, ids, starts, 0, keep=1, width=BLOCK)
                scores = draft.compute_logits(final[:, 1:], head).float()
                best = scores.log_softmax(dim=-1).topk(8, dim=-1)
                states, _ = second_rows(draft, second, target, features, ids, starts, depths, entered, 0, reach=2, width=BLOCK)
                candidates = best.indices[0, depth][None]
                mine = second.scores(best.values[0, depth][None], scores[0, depth].gather(-1, candidates[0])[None],
                                     candidate_scores(draft, head, states[:, 0], candidates), states[:, 0], None,
                                     torch.tensor([any(ranks)]))[0, :8]
            assert torch.allclose(mine.sort(descending=True).values, torch.tensor(values), atol=2e-3)
            checked += 1
    assert checked >= 3
    method.close()

    # In real decoding the text is the plain text; with the counts in the first tree and in the scores too.
    follow = FollowCounts.count([(ids[0], length)], target.config.vocab_size)
    for counted in (False, True):
        method = SecondTree(14, 10, tail=2, top=8)
        method.second = second
        if counted:
            method.tree, method.follow = CountTree(10, follow, candidates=8), follow
        with_it = dflash_generate_redraft(draft, target, torch.tensor([[1, 2, 3, 4, 5]]), 200, None, improver=method,
                                          block_size=BLOCK, return_stats=True)
        assert torch.equal(with_it.output_ids, ids) and len(with_it.acceptance_lengths) <= len(plain.acceptance_lengths)
        method.close()
    assert not any(layer._forward_hooks for layer in draft.layers)

    # By name.
    save_restart_layers(str(tmp_path / "warm-tiny.pt"), second, {"candidates": 8, "reach": 2})
    FOLLOW["counts"] = follow
    try:
        named = make_method("cstree63f40@tiny", str(tmp_path), draft)
        assert (named.budget, named.first, named.tail, named.top) == (63, 40, 2, 8) and named.follow is follow
        assert isinstance(named.tree, CountTree) and named.second.scores is not None and not named.second.adapters
        bare = make_method("stree31t1@tiny", str(tmp_path), draft)
        assert (bare.budget, bare.first, bare.tail) == (31, 31, 1) and bare.follow is None and isinstance(bare.tree, DraftTree)
    finally:
        FOLLOW.clear()
    assert with_counts("cstree63@x") and with_counts("plain+cstree63f40@x") and not with_counts("stree63@x")
