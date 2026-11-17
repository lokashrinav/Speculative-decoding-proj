"""Running the drafter more than once in a round: redrafts from the likely breaks of the first
draft (as D2SD places them), all in one pass, and one redraft after the other."""

import time
from types import SimpleNamespace

import pytest
import torch

import dflash.redraft
from dflash.redraft import Branches, Chain, SelfRedraft, break_scores, dflash_generate_redraft
from dflash.replay import _matches, replay_decode
from tests.test_redraft import make_models

BLOCK = 8


@pytest.fixture(autouse=True)
def _cpu_clock(monkeypatch):
    monkeypatch.setattr(dflash.redraft, "_cuda_time", time.perf_counter)


class Record:
    """Runs an improver and keeps what every round looked like."""

    def __init__(self, improver):
        self.improver = improver
        self.rounds = []

    def start(self, first_drafter):
        self.improver.start(first_drafter)

    def __call__(self, s):
        drafts = self.improver(s)
        self.rounds.append(SimpleNamespace(start=s.start, first=s.block_ids.clone(), logits=s.draft_logits.clone(),
                                           kept=list(self.improver.kept), drafts=None if drafts is None else drafts.clone()))
        return drafts


def lively(*drafters):
    """A drafter with random weights hardly looks at the tokens it is shown. Stronger attention
    makes its redrafts depend on them, as a trained drafter's do."""
    with torch.no_grad():
        for drafter in drafters:
            for name, parameter in drafter.named_parameters():
                if "self_attn" in name and "norm" not in name:
                    parameter.mul_(8.0)


def answer(target, draft, budget=100):
    lively(draft)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, block_size=BLOCK, return_stats=True)
    return plain, prompt.shape[1]


def test_break_scores_are_the_chance_of_the_first_rejection():
    confidence = torch.tensor([0.9, 0.5, 0.8, 0.25, 0.6], dtype=torch.float64)
    scores = break_scores(confidence)
    by_hand = [0.1, 0.9 * 0.5, 0.9 * 0.5 * 0.2, 0.9 * 0.5 * 0.8 * 0.75, 0.9 * 0.5 * 0.8 * 0.25 * 0.4]
    assert torch.allclose(scores, torch.tensor(by_hand, dtype=torch.float64))
    # Every outcome but "all five accepted" is one of the breaks.
    assert abs(float(scores.sum()) + float(confidence.prod()) - 1.0) < 1e-12
    # With the first two tokens taken as right, only what comes after them counts.
    later = break_scores(confidence, fixed=2)
    assert torch.allclose(later, torch.tensor([0.2, 0.8 * 0.75, 0.8 * 0.25 * 0.4], dtype=torch.float64))


@pytest.mark.parametrize("second", [False, True])
def test_redrafts_in_one_pass_are_the_redrafts_made_one_by_one(second):
    models = make_models(second=second)
    target, draft = models[0], models[1]
    other = models[2] if second else None
    if second:
        lively(other)
    plain, length = answer(target, draft)
    one, many = Record(Branches(k=3, model=other, together=False)), Record(Branches(k=3, model=other, together=True))
    slow = replay_decode(draft, target, plain.output_ids, length, ended=False, improver=one, block_size=BLOCK)
    fast = replay_decode(draft, target, plain.output_ids, length, ended=False, improver=many, block_size=BLOCK)
    assert slow.acceptance_lengths == fast.acceptance_lengths and len(one.rounds) == len(many.rounds) > 5
    changed = 0
    for a, b in zip(one.rounds, many.rounds):
        assert a.start == b.start and a.kept == b.kept and torch.equal(a.drafts, b.drafts)
        changed += int(bool((a.drafts != a.first).any()))
    assert changed > 5                                                          # the redrafts are not just the first draft
    # The predictions agree too, not only the chosen tokens.
    inner = SelfRedraft(model=other, keep=1)
    inner.start(draft)
    seen = []

    class Compare:
        def start(self, first_drafter):
            inner.start(first_drafter)

        def __call__(self, s):
            keeps = [1, 3, 2] if s.block_ids.shape[1] >= 5 else [1]
            together, logits = inner.redraft_together(s, keeps)
            for number, keep in enumerate(keeps):
                single, single_logits = inner.redraft(s, keep, context_done=True, with_logits=True)
                assert torch.equal(single, together[number : number + 1])
                assert torch.allclose(single_logits[0], logits[number], atol=1e-9)
                seen.append(keep)
            return None

    replay_decode(draft, target, plain.output_ids, length, ended=False, improver=Compare(), block_size=BLOCK)
    assert len(seen) > 10


def test_branches_keep_the_first_draft_up_to_each_likely_break_and_the_best_draft_counts():
    target, draft = make_models()
    plain, length = answer(target, draft, budget=140)
    ids = plain.output_ids
    watch = Record(Branches(k=3))
    result = replay_decode(draft, target, ids, length, ended=False, improver=watch, block_size=BLOCK)
    assert result.num_output_tokens == plain.num_output_tokens
    assert sum(result.acceptance_lengths) == sum(plain.acceptance_lengths)
    by_start = {round_.start: round_ for round_ in watch.rounds}
    start, checked, wins = length, 0, 0
    for gained in result.acceptance_lengths:
        round_ = by_start.get(start)
        if round_ is not None:
            width = round_.first.shape[1]
            confidence = round_.logits.softmax(dim=-1).amax(dim=-1)[0]
            scores = break_scores(confidence)[1 : width - 1]                      # the same drafter: breaks 1 .. width - 2
            assert round_.kept == (scores.topk(min(3, scores.shape[0])).indices + 1).tolist()
            assert round_.drafts.shape == (len(round_.kept), width)
            truth = ids[:, start + 1 : start + width]
            own = _matches(round_.first[:, 1:], truth)
            best = own
            for row, keep in zip(round_.drafts, round_.kept):
                assert torch.equal(row[: keep + 1], round_.first[0, : keep + 1])   # the kept tokens are the first draft's
                best = max(best, _matches(row[None, 1:], truth))
            assert gained == min(best + 1, ids.shape[1] - start - 1)
            wins += int(best > own)
            checked += 1
        start += gained
    assert checked > 5 and wins == result.alternative_wins
    # One branch is the single redraft from the most likely break.
    single = replay_decode(draft, target, ids, length, ended=False, improver=Branches(k=1), block_size=BLOCK)
    known = replay_decode(draft, target, ids, length, ended=False, improver=SelfRedraft(keep="confidence"), block_size=BLOCK)
    assert single.acceptance_lengths == known.acceptance_lengths


@pytest.mark.parametrize("differ", ["late", "early"])
@pytest.mark.parametrize("second", [False, True])
def test_a_redraft_can_be_made_to_differ_from_the_first_draft_at_its_break(differ, second):
    models = make_models(second=second)
    target, draft = models[0], models[1]
    other = models[2] if second else None
    if second:
        lively(other)
    plain, length = answer(target, draft, budget=120)
    watch = Record(Branches(k=3, model=other, differ=differ))
    again = Record(Branches(k=3, model=other, differ=differ, together=False))
    result = replay_decode(draft, target, plain.output_ids, length, ended=False, improver=watch, block_size=BLOCK)
    slow = replay_decode(draft, target, plain.output_ids, length, ended=False, improver=again, block_size=BLOCK)
    assert result.acceptance_lengths == slow.acceptance_lengths
    zero = 0
    for round_, other_round in zip(watch.rounds, again.rounds):
        assert torch.equal(round_.drafts, other_round.drafts)
        second_choice = round_.logits[0].topk(2, dim=-1).indices[:, 1]
        for row, at in zip(round_.drafts, round_.kept):
            assert torch.equal(row[: at + 1], round_.first[0, : at + 1])           # the first draft up to the break
            assert int(row[at + 1]) != int(round_.first[0, at + 1])                # ... and another token right after it
            if differ == "early" or (at == 0 and not second):
                assert int(row[at + 1]) == int(second_choice[at])                  # the first draft's own second choice
            zero += int(at == 0)
    assert zero > 0                                                                # the break before the first token is used
    with pytest.raises(ValueError):
        Branches(differ="sometimes")


def test_one_redraft_after_the_other_reads_further_each_time():
    target, draft = make_models()
    plain, length = answer(target, draft, budget=140)
    ids = plain.output_ids
    two = replay_decode(draft, target, ids, length, ended=False, improver=Chain(depth=2), block_size=BLOCK)
    known = replay_decode(draft, target, ids, length, ended=False, improver=SelfRedraft(keep="confidence"), block_size=BLOCK)
    assert two.acceptance_lengths == known.acceptance_lengths                     # two drafts: the single redraft

    class Watch(Chain):
        seen = []

        def __call__(self, s):
            drafts = super().__call__(s)
            Watch.seen.append((s.block_ids.clone(), drafts.clone(), list(self.levels)))
            return drafts

    result = replay_decode(draft, target, ids, length, ended=False, improver=Watch(depth=4), block_size=BLOCK)
    assert sum(result.acceptance_lengths) == sum(plain.acceptance_lengths)
    deeper = 0
    for first, drafts, levels in Watch.seen:
        assert drafts.shape[0] == len(levels) <= 3
        assert levels == sorted(set(levels))                                      # every level keeps more than the one before
        before = first[0]
        for row, keep in zip(drafts, levels):
            assert torch.equal(row[: keep + 1], before[: keep + 1])               # ... of the draft made just before it
            before = row
        deeper += int(len(levels) > 1)
    assert deeper > 0
    with pytest.raises(ValueError):
        Chain(depth=1)


def test_a_second_drafter_reads_the_context_of_rounds_it_skipped():
    target, draft, other = make_models(second=True)
    lively(other)
    plain, length = answer(target, draft, budget=120)
    ids = plain.output_ids

    class Sometimes:
        """Redrafts in every third round only."""

        def __init__(self):
            self.inner = SelfRedraft(model=other, keep=1)
            self.calls = 0
            self.made = {}

        def start(self, first_drafter):
            self.inner.start(first_drafter)

        def __call__(self, s):
            self.calls += 1
            if self.calls % 3:
                self.inner.skip(s)
                return None
            self.made[s.start] = self.inner.redraft(s, 1)
            return None

    class Always(Sometimes):
        def __call__(self, s):
            self.made[s.start] = self.inner.redraft(s, 1)
            return None

    some, every = Sometimes(), Always()
    replay_decode(draft, target, ids, length, ended=False, improver=some, block_size=BLOCK)
    replay_decode(draft, target, ids, length, ended=False, improver=every, block_size=BLOCK)
    assert 2 < len(some.made) < len(every.made)
    for start, made in some.made.items():
        assert torch.equal(made, every.made[start])
    assert len({tuple(made[0].tolist()) for made in every.made.values()}) > 5


def test_of_several_alternative_drafts_the_best_one_counts():
    from tests.test_redraft import OracleImprover

    target, draft = make_models()
    plain, length = answer(target, draft, budget=120)
    ids = plain.output_ids

    class Three:
        """A draft of sevens, the true continuation up to its fourth token, and up to its second."""

        def __init__(self):
            self.good, self.worse = OracleImprover(ids, wrong_from=4), OracleImprover(ids, wrong_from=2)

        def start(self, first_drafter):
            pass

        def __call__(self, s):
            good, worse = self.good(s), self.worse(s)
            if good is None:
                return None
            sevens = s.block_ids.clone()
            sevens[:, 2:] = 7
            return torch.cat((sevens, good, worse), dim=0)

    three = replay_decode(draft, target, ids, length, ended=False, improver=Three(), block_size=BLOCK)
    best = replay_decode(draft, target, ids, length, ended=False, improver=OracleImprover(ids, wrong_from=4), block_size=BLOCK)
    assert three.acceptance_lengths == best.acceptance_lengths and three.alternative_wins == best.alternative_wins > 5
    assert three.rounds < plain.rounds


def test_several_alternative_drafts_are_for_replay_only():
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    with pytest.raises(NotImplementedError):
        dflash_generate_redraft(draft, target, prompt, 60, None, improver=Branches(k=2), block_size=BLOCK)
    # One branch is one alternative draft, which decoding does check: the same rounds as the replay.
    live = dflash_generate_redraft(draft, target, prompt, 60, None, improver=Branches(k=1), block_size=BLOCK,
                                   return_stats=True)
    again = replay_decode(draft, target, live.output_ids, prompt.shape[1], ended=False, improver=Branches(k=1), block_size=BLOCK)
    assert again.acceptance_lengths == live.acceptance_lengths


def test_times_between_the_measured_lengths_and_the_cost_of_an_answers_rounds():
    from nexus.adapt_drafter import GROUPS
    from nexus.redraft_long import CHECK_MS, DRAFT_MS, check_and_draft_ms, measured

    assert measured(CHECK_MS, 10) == CHECK_MS[0] and measured(CHECK_MS, 10**6) == CHECK_MS[-1]
    assert measured(CHECK_MS, 16000) == CHECK_MS[4]
    assert abs(measured(CHECK_MS, 12000) - (CHECK_MS[3] + CHECK_MS[4]) / 2) < 1e-9
    # Three rounds from a prompt of 100 tokens: at 100, 104 and 2,204 tokens of text.
    table = check_and_draft_ms([4, 2100, 5], 100, None)
    names = [name for name, _, _ in GROUPS]
    short = 2 * (CHECK_MS[0] + DRAFT_MS[0])
    longer = measured(CHECK_MS, 2204) + measured(DRAFT_MS, 2204)
    assert abs(table[names.index("0-2k")] - short) < 1e-9 and abs(table[names.index("2k+")] - longer) < 1e-9
    assert abs(table[names.index("everything")] - short - longer) < 1e-9 and table[names.index("8k+")] == 0
    # Rounds inside endless repetition are left out, as in the count of rounds.
    assert check_and_draft_ms([4, 2100, 5], 100, 2000)[names.index("everything")] == short


def test_method_names_of_the_long_answer_runs():
    from nexus.redraft_long import TrueBreak, make_method, parse_setting

    assert make_method("plain", "/nowhere") is None
    keep = make_method("self-keep3", "/nowhere")
    assert isinstance(keep, SelfRedraft) and keep.keep == 3 and keep.model is None
    assert make_method("self-keepconfidence", "/nowhere").keep == "confidence"
    plain, late, early = (make_method(name, "/nowhere") for name in ("branches4", "branches2l", "branches15e"))
    assert isinstance(plain, Branches) and (plain.k, plain.differ) == (4, None)
    assert (late.k, late.differ) == (2, "late") and (early.k, early.differ) == (15, "early")
    chain = make_method("chain3", "/nowhere")
    assert isinstance(chain, Chain) and chain.depth == 3
    assert isinstance(make_method("truebreak", "/nowhere"), TrueBreak) and make_method("truebreake", "/nowhere").differ == "early"
    for wrong in ("branches", "chain", "self-keep", "truebreakx", "second"):
        with pytest.raises(ValueError):
            make_method(wrong, "/nowhere")
    options = parse_setting("branches4e+fixed@off-a+sharp")
    assert options["method"] == "branches4e" and options["fixed"] == "off-a" and options["sharpen"] is not None
    assert parse_setting("branches4@v1-prefix")["method"] == "branches4@v1-prefix"
    assert parse_setting("plain") == {"method": "plain", "sharpen": None, "fixed": ""}
    for wrong in ("plain+fixed@", "plain+blunt"):
        with pytest.raises(ValueError):
            parse_setting(wrong)


@pytest.mark.parametrize("differ", [None, "early"])
def test_redrafts_are_counted_against_the_saved_answer(differ):
    from nexus.adapt_drafter import GROUPS, by_position, summarize
    from nexus.redraft_long import COUNTS, TrueBreak, Watched, check_and_draft_ms, in_real_time

    target, draft = make_models()
    plain, length = answer(target, draft, budget=140)
    ids = plain.output_ids
    everything = [name for name, _, _ in GROUPS].index("everything")

    def run(improver):
        watched = None if improver is None else Watched(improver, ids, length, None, block=BLOCK)
        result = replay_decode(draft, target, ids, length, ended=False, improver=watched, block_size=BLOCK)
        entry = {"rounds": result.rounds, "table": by_position(result.acceptance_lengths, length, None),
                 "check_and_draft_ms": check_and_draft_ms(result.acceptance_lengths, length, None)}
        if watched is not None:
            entry["redraft_seconds"], entry["breaks"] = watched.seconds, watched.counts
        return result, entry, watched

    _, plain_entry, _ = run(None)
    result, true_entry, watched = run(TrueBreak(differ=differ))
    counts = dict(zip(COUNTS, watched.counts[everything]))
    lowest = 0 if differ else 1
    # Replayed by hand: where the first draft of every full round breaks.
    first = Record(Branches(k=1, differ=differ))
    again = replay_decode(draft, target, ids, length, ended=False, improver=first, block_size=BLOCK)
    assert counts["rounds"] > 20 and counts["first_draft_broke"] <= counts["rounds"]
    # The true break is always the one redrafted from, whenever a redraft from it is possible.
    assert 0 < counts["break_was_among_the_redrafts"] <= counts["first_draft_broke"]
    if lowest == 0:
        assert counts["break_was_among_the_redrafts"] == counts["first_draft_broke"]
    assert counts["redraft_got_past_the_break"] == counts["a_redraft_got_further"] == result.alternative_wins
    assert counts["tokens_added"] >= counts["a_redraft_got_further"]
    if differ:
        assert counts["a_redraft_got_further"] > 0                   # the second choice is sometimes the right token
        assert result.rounds < plain_entry["rounds"]
    assert watched.seconds[everything] > 0 and again.rounds >= result.rounds - 5

    records = [{"settings": {"plain": plain_entry, "truebreak": true_entry}}]
    tokens = summarize(records, ["plain", "truebreak"])
    speed = in_real_time(records, ["plain", "truebreak"])
    row = speed["truebreak"]["everything"]
    assert speed["plain"]["everything"]["faster_than_plain"] == 0 and "redrafts" not in speed["plain"]["everything"]
    assert row["extra_drafter_ms_per_round"] > 0 and row["redrafts"]["rounds_counted"] == counts["rounds"]
    assert abs(row["redrafts"]["that_redraft_got_past_the_break"]
               - counts["redraft_got_past_the_break"] / counts["break_was_among_the_redrafts"]) < 1e-12
    # More tokens per round, minus the time of the extra pass.
    more = tokens["truebreak"]["everything"]["more_tokens_per_round_than_plain"]
    assert row["faster_than_plain"] <= more + 1e-9
    slower = true_entry["redraft_seconds"][everything] * 1000 / true_entry["check_and_draft_ms"][everything]
    assert abs((1 + row["faster_than_plain"]) * (1 + slower) - (1 + more)) < 1e-6


def test_masked_attention_without_copies_is_the_librarys_attention():
    from dflash.model import _masked_attention

    torch.manual_seed(0)
    batch, heads, key_heads, queries, keys, size = 1, 6, 2, 5, 11, 4
    q = torch.randn(batch, heads, queries, size, dtype=torch.float64)
    k = torch.randn(batch, key_heads, keys, size, dtype=torch.float64)
    v = torch.randn(batch, key_heads, keys, size, dtype=torch.float64)
    visible = torch.rand(queries, keys) > 0.4
    visible[:, 0] = True                                                        # every query sees something
    scaling = size**-0.5
    ours = _masked_attention(q, k, v, visible[None, None], scaling)
    per = heads // key_heads
    theirs = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(per, dim=1), v.repeat_interleave(per, dim=1), attn_mask=visible[None, None], scale=scaling)
    assert ours.shape == theirs.shape and torch.allclose(ours, theirs, atol=1e-12)
    # A hidden key changes nothing for the queries that cannot see it.
    changed = k.clone()
    changed[:, :, 3] += 5.0
    blind = ~visible[:, 3]
    again = _masked_attention(q, changed, v, visible[None, None], scaling)
    assert blind.any() and torch.allclose(again[:, :, blind], ours[:, :, blind], atol=1e-12)
    assert not torch.allclose(again[:, :, ~blind], ours[:, :, ~blind], atol=1e-6)


def test_the_switch_alone_is_the_first_draft_with_its_second_choice_at_the_break():
    """The control without a second pass: nothing but one token of the first draft changes."""
    from nexus.redraft_long import make_method

    target, draft = make_models()
    plain, length = answer(target, draft, budget=120)
    ids = plain.output_ids
    watch, early = Record(Branches(k=3, differ="switch")), Record(Branches(k=3, differ="early"))
    passes = []
    forward = draft.forward

    def counted(*args, **kwargs):
        passes.append(1)
        return forward(*args, **kwargs)

    draft.forward = counted
    try:
        result = replay_decode(draft, target, ids, length, ended=False, improver=watch, block_size=BLOCK)
        assert len(passes) == result.rounds                                    # one pass per round: the first draft
    finally:
        draft.forward = forward
    replay_decode(draft, target, ids, length, ended=False, improver=early, block_size=BLOCK)
    assert result.alternative_wins > 0
    for round_ in watch.rounds:
        second_choice = round_.logits[0].topk(2, dim=-1).indices[:, 1]
        for row, at in zip(round_.drafts, round_.kept):
            expected = round_.first[0].clone()
            expected[at + 1] = second_choice[at]
            assert torch.equal(row, expected)
    # The redraft that knows the switch starts from the same rows and may change only what follows the switch.
    first = {round_.start: round_ for round_ in early.rounds}
    compared = 0
    for round_ in watch.rounds:
        other = first.get(round_.start)
        if other is None or other.kept != round_.kept:
            continue
        for row, redrafted, at in zip(round_.drafts, other.drafts, round_.kept):
            assert torch.equal(row[: at + 2], redrafted[: at + 2])
            compared += 1
    assert compared > 5
    switch = make_method("branches4s", "/nowhere")
    assert isinstance(switch, Branches) and switch.differ == "switch" and make_method("truebreaks", "/nowhere").differ == "switch"


def made_up_answer(target, draft, accept, block=16):
    """A text on which the drafter's first draft is accepted for exactly accept[0], accept[1], ...
    tokens, round after round: its own draft is copied that far and then left."""
    from dflash.model import _draft_value, _output_head, _raw_input_embeddings
    from dflash.replay import target_features

    ids = torch.tensor([[1, 2, 3, 4, 5, 9]])
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    for count in accept:
        anchor = ids.shape[1] - 1
        features = target_features(target, draft, ids)
        tokens = torch.full((1, block), draft.mask_token_id, dtype=torch.long)
        tokens[0, 0] = ids[0, anchor]
        with torch.no_grad():
            states = draft(target_hidden=features[:, :anchor], noise_embedding=_raw_input_embeddings(target, tokens, scale),
                           position_ids=torch.arange(anchor + block)[None])[:, -block:, :]
            guess = draft.compute_logits(states[:, 1:, :], _output_head(target)).argmax(dim=-1)[0]
        if count == block - 1:
            new = torch.cat((guess, torch.tensor([7])))
        else:
            new = torch.cat((guess[:count], (guess[count : count + 1] + 3) % 30))
        ids = torch.cat((ids, new[None]), dim=1)
    return ids, 5


def test_every_kind_of_second_draft_is_scored_on_the_same_rounds(tmp_path):
    from dflash.redraft import redrafts_from
    from nexus.why_redraft import ARMS, COLUMNS, Why, summary

    target, draft, other = make_models(second=True)
    lively(draft, other)
    accept = [2, 0, 5, 1, 15, 3, 0, 7, 4]
    ids, length = made_up_answer(target, draft, accept)
    ids = torch.cat((ids, torch.randint(0, 30, (1, 20), generator=torch.Generator().manual_seed(1))), dim=1)

    class Reference:
        """The same second drafts, made one by one with the parts tested above."""

        def __init__(self):
            self.inner, self.rounds = SelfRedraft(keep=1), {}

        def start(self, first_drafter):
            self.inner.start(first_drafter)

        def __call__(self, s):
            if s.block_ids.shape[1] != 16 or s.start + 16 > ids.shape[1]:
                return None
            truth = ids[:, s.start + 1 : s.start + 16]

            def best(rows):
                return max(_matches(rows[n : n + 1, 1:], truth) for n in range(rows.shape[0]))

            broke = _matches(s.block_ids[:, 1:], truth)
            out = {"broke_after": broke}
            if broke < 15:
                if broke:
                    out["again"] = best(self.inner.redraft(s, broke))
                    out["first_token"] = best(self.inner.redraft(s, 1))
                for name in ("switch", "early", "late"):
                    out[name] = best(redrafts_from(self.inner, s, [broke], name, together=False))
                for name, differ in (("four", None), ("four_switch", "switch"), ("four_early", "early"), ("four_late", "late")):
                    branches = Branches(k=4, differ=differ, together=False)
                    branches.inner = self.inner
                    out[name] = best(branches(s))
            self.rounds[s.start] = out
            return None

    probe, reference = Why(second=other, control=other, edit=other), Reference()
    probe.begin(ids, length, None, None, answer=0)
    plain = replay_decode(draft, target, ids, length, ended=False, block_size=16)
    watched = replay_decode(draft, target, ids, length, ended=False, improver=probe, block_size=16)
    replay_decode(draft, target, ids, length, ended=False, improver=reference, block_size=16)
    assert watched.acceptance_lengths == plain.acceptance_lengths                 # the probe changes nothing
    assert plain.acceptance_lengths[: len(accept)] == [count + 1 for count in accept]
    table = [dict(zip(COLUMNS, row)) for row in probe.rows]
    assert [row["broke_after"] for row in table[: len(accept)]] == accept
    starts, start = [], length
    for gained in plain.acceptance_lengths:
        starts.append(start)
        start += gained
    checked = 0
    for number, row in enumerate(table):
        expected = reference.rounds[starts[number]]
        broke = row["broke_after"]
        assert broke == expected["broke_after"] and row["where"] == starts[number] - length
        if broke == 15:
            assert all(row[name] == -1 for name in ARMS if name != "next_round")
        else:
            for name, value in expected.items():
                assert row[name] == value, (number, name)
            assert row["right_token_old_rest"] > broke and row["right_token_new_rest"] > broke
            assert row["switch"] == broke or row["switch"] > broke
            assert row["rank_first"] >= 2 and 0 <= row["confidence"] <= 1000
            assert row["second_blind"] >= 0 and row["control"] >= 0 and row["second"] >= broke and row["edit"] >= 0
            if broke:
                assert row["again"] >= broke and row["random"] >= broke and row["elsewhere"] >= broke
                assert row["second_random"] >= broke and row["rank_again"] >= 1
            else:
                assert row["again"] == row["random"] == row["elsewhere"] == row["first_token"] == -1
                assert row["late"] == row["early"]                         # nothing new to read: the switch is shown
            checked += 1
        if number + 1 < len(table):
            after = table[number + 1]["broke_after"]
            assert row["next_round"] == (broke + 1 + min(after, 14 - broke) if broke < 15 else -1) or broke == 15
    assert checked >= len(accept) - 1 and any(row["again"] > row["broke_after"] or row["early"] > row["broke_after"]
                                                 or row["second"] > row["broke_after"] for row in table)
    brief = summary(probe.rows)
    assert brief["rounds"] == len(table) and brief["rounds_in_which_the_first_draft_broke"] == checked
    assert set(brief["second_drafts"]) == set(ARMS)
    assert brief["second_drafts"]["right_token_old_rest"]["got_past_the_break"] == 1.0
    # Without saved second drafters those columns stay empty.
    alone = Why()
    alone.begin(ids, length, None, None, answer=0)
    replay_decode(draft, target, ids, length, ended=False, improver=alone, block_size=16)
    assert all(row[COLUMNS.index(name)] == -1 for name in ("second", "control", "edit") for row in alone.rows)
    assert [row[COLUMNS.index("early")] for row in alone.rows] == [row["early"] for row in table]
