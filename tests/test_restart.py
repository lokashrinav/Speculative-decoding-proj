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
