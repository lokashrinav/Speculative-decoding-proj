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
    assert all(adapter.down.requires_grad and adapter.up.requires_grad for adapter in adapters)
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


def test_training_against_the_targets_own_predictions():
    """"kl" compares the drafter with what the target predicted at each position. Those predictions
    come from reading the finished answer once, and their first choice is the committed token."""
    from dflash.replay import target_features

    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    plain = dflash_generate_redraft(draft, target, prompt, 300, None, return_stats=True)
    ids = plain.output_ids
    features, (tokens, scores) = target_features(target, draft, ids, top=6, chunk=64)
    assert torch.allclose(features, target_features(target, draft, ids), atol=1e-9)
    assert tokens.shape == (ids.shape[1], 6) and scores.shape == tokens.shape
    start = prompt.shape[1]
    assert torch.equal(tokens[start - 1 : -1, 0], ids[0, start:])           # row p describes the token at p + 1
    assert bool((scores[:, 0] >= scores[:, 1]).all()) and bool((scores.exp().sum(-1) <= 1 + 1e-5).all())
    adapters = attach_adapters(draft, rank=4)
    results = {}
    for labels in ("immediate", "hindsight"):
        trainer = DraftTrainer(draft, adapters, labels=labels, rate=3e-3, loss="kl")
        trainer.begin()
        trainer.predictions = (tokens, scores)
        results[labels] = replay_decode(draft, target, ids, start, ended=False, features=features, improver=trainer)
        assert trainer.totals["updates"] > 20 and results[labels].acceptance_lengths != plain.acceptance_lengths
        # The same run again gives the same rounds: nothing is left over from the run before.
        trainer.begin()
        trainer.predictions = (tokens, scores)
        again = replay_decode(draft, target, ids, start, ended=False, features=features, improver=trainer)
        assert again.acceptance_lengths == results[labels].acceptance_lengths
    detach_adapters(draft)


class _Recorder:
    """A plug-in that only keeps what decoding hands it."""

    replace = False

    def __init__(self, top):
        self.canvas_top, self.seen = top, []

    def start(self, first_drafter):
        pass

    def __call__(self, s):
        self.seen.append((int(s.start), s.block_ids[0].clone(), s.canvas))
        return None


def test_the_canvas_is_what_the_target_predicted_on_the_draft():
    """Decoding and replay hand a plug-in the target's predictions on the previous draft. Up to the first
    rejection they are its predictions on the answer itself; after it they belong to the rejected draft."""
    from dflash.replay import target_features

    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    start = prompt.shape[1]
    live_recorder = _Recorder(top=6)
    live = dflash_generate_redraft(draft, target, prompt, 200, None, improver=live_recorder, return_stats=True)
    ids = live.output_ids
    features, (tokens, scores) = target_features(target, draft, ids, top=6, chunk=64)
    recorder = _Recorder(top=6)
    replayed = replay_decode(draft, target, ids, start, ended=False, features=features, improver=recorder)
    assert replayed.acceptance_lengths == live.acceptance_lengths
    assert replay_decode(draft, target, ids, start, ended=False, features=features).acceptance_lengths == live.acceptance_lengths
    assert recorder.seen[0][2] is None and len(recorder.seen) > 20
    off_path = 0
    for (was_at, block, _), (now_at, _, canvas) in zip(recorder.seen, recorder.seen[1:]):
        assert canvas[0] == was_at and canvas[1].shape == (len(block) - 1, 6)
        gained = now_at - was_at                                             # accepted tokens and the target's own next one
        assert torch.equal(canvas[1][:gained, 0], ids[0, was_at + 1 : now_at + 1])
        assert torch.allclose(canvas[2][:gained], scores[was_at:now_at], atol=1e-9)
        # After the first rejection the target read a wrong token, so it no longer describes the answer.
        rest = slice(gained, len(block) - 1)
        off_path += int((canvas[1][rest, 0] != ids[0, was_at + 1 + gained : was_at + len(block)][: len(block) - 1 - gained]).sum())
    assert off_path > 0
    # Live decoding hands over the same thing.
    assert len(live_recorder.seen) == len(recorder.seen)
    for (a_start, a_block, a_canvas), (b_start, b_block, b_canvas) in zip(live_recorder.seen[1:], recorder.seen[1:]):
        assert a_start == b_start and torch.equal(a_block, b_block) and a_canvas[0] == b_canvas[0]
        assert torch.equal(a_canvas[1][:, 0], b_canvas[1][:, 0]) and torch.allclose(a_canvas[2], b_canvas[2], atol=1e-9)


@pytest.mark.parametrize("labels,loss", [("canvas", "kl"), ("canvas", "ce"), ("canvas+hindsight", "kl")])
def test_training_on_the_canvas_as_the_published_method_does(labels, loss):
    from dflash.replay import target_features

    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    start = prompt.shape[1]
    plain = dflash_generate_redraft(draft, target, prompt, 300, None, return_stats=True)
    ids = plain.output_ids
    features, predictions = target_features(target, draft, ids, top=6, chunk=64)
    adapters = attach_adapters(draft, rank=4)
    trainer = DraftTrainer(draft, adapters, labels=labels, rate=3e-3, loss=loss, top=6)
    assert trainer.canvas_top == 6
    trainer.begin()
    trainer.predictions = predictions
    live = dflash_generate_redraft(draft, target, prompt, 300, None, improver=trainer, return_stats=True)
    assert torch.equal(live.output_ids, ids) and live.acceptance_lengths != plain.acceptance_lengths
    starts = [start]
    for gained in live.acceptance_lengths:
        starts.append(starts[-1] + gained)
    gained_at = dict(zip(starts, live.acceptance_lengths))
    width = draft.block_size - 1
    assert len(trainer.canvas_used) > 20
    wrong_labels = 0
    for position, used in trainer.canvas_used:
        gained = gained_at[position]
        assert len(used) == min(width, ids.shape[1] - position - 1)          # the whole draft at once
        assert used[:gained] == ids[0, position + 1 : position + 1 + gained].tolist()
        wrong_labels += sum(a != b for a, b in zip(used[gained:], ids[0, position + 1 + gained :].tolist()))
    assert wrong_labels > 0                    # some of what it learned from was not the answer's text
    if labels == "canvas":
        assert not trainer.trained_on
    else:
        assert trainer.trained_on              # ... and here those positions are learned again, exactly
        for position, first, used in trainer.trained_on:
            assert first == min(gained_at[position], width) and first + len(used) == width
            assert ids[0, position + 1 + first : position + 1 + width].tolist() == used
    used_live = list(trainer.canvas_used)
    trainer.begin()
    trainer.predictions = predictions
    replayed = replay_decode(draft, target, ids, start, ended=False, features=features, improver=trainer)
    assert replayed.acceptance_lengths == live.acceptance_lengths and trainer.canvas_used == used_live
    detach_adapters(draft)


def test_settings_of_a_queued_job_can_be_replaced_by_a_file(tmp_path):
    from nexus.adapt_drafter import settings_for

    folder = tmp_path / "settings"
    folder.mkdir()
    assert settings_for("/x/nexus/results/adapt-kl-2", "plain,immediate:1e-4:kl", str(folder)) == ("plain,immediate:1e-4:kl", "command line")
    (folder / "adapt-kl.txt").write_text("plain,hindsight:1e-3:kl\nsecond line is ignored\n")
    assert settings_for("/x/nexus/results/adapt-kl-2/", "plain,immediate:1e-4:kl", str(folder))[0] == "plain,hindsight:1e-3:kl"
    assert settings_for("/x/nexus/results/adapt-drafter-0", "plain", str(folder)) == ("plain", "command line")
    (folder / "adapt-drafter.txt").write_text("\n")
    assert settings_for("/x/nexus/results/adapt-drafter-0", "plain", str(folder)) == ("plain", "command line")


def test_what_was_learned_can_be_kept_saved_and_used_as_a_starting_state(tmp_path):
    from dflash.adapt import adapter_state, load_adapters, save_adapters

    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    start = prompt.shape[1]
    ids = dflash_generate_redraft(draft, target, prompt, 200, None, return_stats=True).output_ids
    plain = replay_decode(draft, target, ids, start, ended=False)
    adapters = attach_adapters(draft, rank=4)
    trainer = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3)
    trainer.begin()
    once = replay_decode(draft, target, ids, start, ended=False, improver=trainer)
    learned = adapter_state(adapters)
    assert any(float(up.abs().sum()) > 0 for _, up in learned)
    # Going on without a reset: the second pass over the same answer starts from what the first one taught.
    optimizer = trainer.optimizer
    trainer.begin(keep=True)
    assert trainer.optimizer is optimizer and all(torch.equal(a.up, up) for a, (_, up) in zip(adapters, learned))
    twice = replay_decode(draft, target, ids, start, ended=False, improver=trainer)
    assert once.rounds < plain.rounds and twice.rounds < plain.rounds
    assert twice.acceptance_lengths != once.acceptance_lengths            # it did not start from zero this time
    path = str(tmp_path / "lora-test.pt")
    save_adapters(path, adapters, answers=2)
    kept = adapter_state(adapters)
    # A request starts from zero again ...
    trainer.begin()
    assert all(float(adapter.up.abs().sum()) == 0.0 for adapter in adapters)
    assert replay_decode(draft, target, ids, start, ended=False).acceptance_lengths == plain.acceptance_lengths
    # ... unless it is told to start from the saved state, with or without more training.
    state = load_adapters(path, adapters)
    assert all(torch.allclose(a, b) and torch.allclose(c, d) for (a, c), (b, d) in zip(state, kept))
    frozen = DraftTrainer(draft, adapters, rate=0.0, init=state)
    frozen.begin()
    with_saved = replay_decode(draft, target, ids, start, ended=False, improver=frozen)
    assert with_saved.acceptance_lengths != plain.acceptance_lengths and frozen.totals["updates"] == 0
    frozen.begin()
    assert replay_decode(draft, target, ids, start, ended=False, improver=frozen).acceptance_lengths == with_saved.acceptance_lengths
    warm = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3, init=state)
    warm.begin()
    warmed = replay_decode(draft, target, ids, start, ended=False, improver=warm)
    # Its first draft is made before any training, so it is the saved state's draft; after that it learns on.
    assert warmed.acceptance_lengths[0] == with_saved.acceptance_lengths[0] and warm.totals["updates"] > 20
    assert warmed.acceptance_lengths != with_saved.acceptance_lengths
    detach_adapters(draft)
    with pytest.raises(AssertionError):
        load_adapters(path, attach_adapters(draft, rank=2))
    detach_adapters(draft)


def test_a_setting_can_train_part_of_the_adapters_and_skip_rounds():
    from dflash.adapt import choose_adapters
    from nexus.adapt_drafter import parse_setting

    assert parse_setting("plain") == {"labels": "plain", "rate": 0.0, "loss": "ce", "stride": 0, "only": "all", "window": 0,
                                      "start_from": "", "trigger": None, "sharpen": None}
    assert parse_setting("frozen:window=2048@off-a") == {"labels": "frozen", "rate": 0.0, "loss": "ce", "stride": 0,
                                                         "only": "all", "window": 2048, "start_from": "off-a",
                                                         "trigger": None, "sharpen": None}
    with pytest.raises(ValueError):
        parse_setting("immediate:3e-4:window=2048")                       # only for settings that do not train
    assert parse_setting("frozen@off-a")["start_from"] == "off-a" and parse_setting("frozen@off-a")["rate"] == 0.0
    read = parse_setting("immediate:3e-4:kl:stride=5:only=last1@off-b")
    assert read == {"labels": "immediate", "rate": 3e-4, "loss": "kl", "stride": 5, "only": "last1", "window": 0,
                    "start_from": "off-b", "trigger": None, "sharpen": None}
    assert parse_setting("rejected:3e-4:trigger=2")["trigger"] == 2
    with pytest.raises(ValueError):
        parse_setting("plain:trigger=2")                                  # only for settings that train
    assert parse_setting("canvas+hindsight:1e-3:only=mlp")["only"] == "mlp"
    with pytest.raises(ValueError):
        parse_setting("immediate:3e-4:fast")

    from tests.test_redraft import Qwen3Config, decoding

    target, _ = make_models()
    torch.manual_seed(3)
    config = Qwen3Config(num_hidden_layers=2, vocab_size=32, hidden_size=16, intermediate_size=32, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, max_position_embeddings=128, bos_token_id=0, eos_token_id=30,
                         pad_token_id=0, attention_dropout=0.0, tie_word_embeddings=False)
    config.num_target_layers, config.target_layer_ids, config.mask_token_id = 2, [0], 31
    config.block_size, config.is_causal = 4, False
    draft = decoding.DFlashDraftModel(config).eval().double()            # a drafter with two layers
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    ids = dflash_generate_redraft(draft, target, prompt, 200, None, return_stats=True).output_ids
    adapters = attach_adapters(draft, rank=4)
    layers = len(draft.layers)
    assert layers == 2
    assert len(choose_adapters(adapters)) == 5 * layers and len(choose_adapters(adapters, "attn")) == 2 * layers
    assert len(choose_adapters(adapters, "mlp")) == 3 * layers and len(choose_adapters(adapters, "last1")) == 5
    assert choose_adapters(adapters, "last2") == adapters
    with pytest.raises(ValueError):
        choose_adapters(adapters, "some")
    last = choose_adapters(adapters, "last1")
    assert all(adapter.where[0] == 1 for adapter in last)
    trainer = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3, trained=last)
    trainer.begin()
    replay_decode(draft, target, ids, prompt.shape[1], ended=False, improver=trainer)
    moved = [adapter.where for adapter in adapters if float(adapter.up.abs().sum()) > 0]
    assert moved and all(layer == 1 for layer, _ in moved)                 # the first layer's adapters did not move
    assert not any(adapter.down.requires_grad or adapter.up.requires_grad for adapter in adapters if adapter.where[0] == 0)
    everything = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3)
    everything.begin()
    assert all(adapter.up.requires_grad and float(adapter.up.abs().sum()) == 0.0 for adapter in adapters)
    replay_decode(draft, target, ids, prompt.shape[1], ended=False, improver=everything)
    assert any(layer == 0 for layer, _ in [adapter.where for adapter in adapters if float(adapter.up.abs().sum()) > 0])
    detach_adapters(draft)


def test_a_limited_context_window_for_the_drafter_in_replay():
    """A control: the drafter sees only the last few tokens of the text so far."""
    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    start = prompt.shape[1]
    ids = dflash_generate_redraft(draft, target, prompt, 200, None, return_stats=True).output_ids
    plain = replay_decode(draft, target, ids, start, ended=False)
    adapters = attach_adapters(draft, rank=4)
    viewer = DraftTrainer(draft, adapters, rate=0.0)
    viewer.begin()
    viewer.context_window = 10_000                                           # longer than the text: nothing changes
    assert replay_decode(draft, target, ids, start, ended=False, improver=viewer).acceptance_lengths == plain.acceptance_lengths
    viewer.context_window = 6
    short = replay_decode(draft, target, ids, start, ended=False, improver=viewer)
    assert short.acceptance_lengths != plain.acceptance_lengths              # it saw less, so it drafted differently
    assert sum(short.acceptance_lengths) == sum(plain.acceptance_lengths)    # the same text is covered
    detach_adapters(draft)


def test_a_queued_job_can_be_told_to_step_aside(tmp_path):
    import json

    from nexus.probe_memory import stepped_aside

    folder = tmp_path / "settings"
    folder.mkdir()
    results = tmp_path / "results" / "bench-decode-2"
    results.mkdir(parents=True)
    assert not stepped_aside(str(results), str(folder))                       # no list: carry on
    (folder / "skip.txt").write_text("adapt-window   # the whole family\nbench-decode-3\n\n")
    assert not stepped_aside(str(results), str(folder)) and not (results / "status.json").exists()
    other = tmp_path / "results" / "bench-decode-3"
    other.mkdir()
    assert stepped_aside(str(other) + "/", str(folder))                       # named itself
    assert json.loads((other / "status.json").read_text())["stepped_aside"] is True
    family = tmp_path / "results" / "adapt-window-1"
    family.mkdir()
    assert stepped_aside(str(family), str(folder))                            # named by its family
    lone = tmp_path / "results" / "adapt-window"
    lone.mkdir()
    assert stepped_aside(str(lone), str(folder))


def _starts(prompt, acceptance_lengths):
    starts = [prompt.shape[1]]
    for gained in acceptance_lengths:
        starts.append(starts[-1] + gained)
    return starts


@pytest.mark.parametrize("labels", ["rejected", "accepted"])
def test_the_correction_and_the_accepted_tokens_can_be_learned_from_separately(labels):
    """"immediate" has two parts: the token the target wrote itself, and the draft tokens it accepted."""
    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    budget = 300
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, return_stats=True)
    adapters = attach_adapters(draft, rank=4)
    trainer = DraftTrainer(draft, adapters, labels=labels, rate=3e-3)
    trainer.begin()
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=trainer, return_stats=True)
    assert torch.equal(live.output_ids, plain.output_ids)
    final, width = live.output_ids[0], draft.block_size - 1
    starts = _starts(prompt, live.acceptance_lengths)
    gained = dict(zip(starts, live.acceptance_lengths))
    assert trainer.trained_on, "nothing was trained"
    for position, first, used in trainer.trained_on:
        assert final[position + 1 + first : position + 1 + first + len(used)].tolist() == used
        accepted = min(gained[position] - 1, width)
        if labels == "rejected":
            assert accepted < width and first == accepted and len(used) == 1    # the one token the target wrote itself
        else:
            assert first == 0 and len(used) == accepted >= 1                    # the draft tokens that were right
    # Every round that had something of this kind was learned from (the last rounds have no later call).
    trained = {position for position, _, _ in trainer.trained_on}
    for position in starts[:-3]:
        has = gained[position] <= width if labels == "rejected" else gained[position] >= 2
        assert (position in trained) == has
    trainer.begin()
    replayed = replay_decode(draft, target, live.output_ids, prompt.shape[1], ended=False, improver=trainer)
    assert replayed.acceptance_lengths == live.acceptance_lengths
    detach_adapters(draft)


def test_learning_only_from_rounds_that_were_rejected_early():
    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    budget = 300
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, return_stats=True)
    adapters = attach_adapters(draft, rank=4)
    every = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3)
    every.begin()
    dflash_generate_redraft(draft, target, prompt, budget, None, improver=every, return_stats=True)
    early = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3, trigger=0)
    early.begin()
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=early, return_stats=True)
    assert torch.equal(live.output_ids, plain.output_ids)
    gained = dict(zip(_starts(prompt, live.acceptance_lengths), live.acceptance_lengths))
    assert early.trained_on
    # Only rounds whose first draft token was already wrong: the target's own token is all there is to learn from.
    assert all(gained[position] == 1 and len(used) == 1 for position, _, used in early.trained_on)
    assert 0 < early.totals["updates"] < every.totals["updates"]
    early.begin()
    replayed = replay_decode(draft, target, live.output_ids, prompt.shape[1], ended=False, improver=early)
    assert replayed.acceptance_lengths == live.acceptance_lengths
    detach_adapters(draft)


def test_sharper_attention_on_long_texts_needs_no_training():
    """The baseline SharpDraft: the drafter's queries are scaled up as it sees more tokens."""
    import math

    from dflash.sharp import QueryScale, set_query_scale
    from nexus.adapt_drafter import parse_setting

    rule = QueryScale()
    assert rule(100) == 1.0 and rule(2048) == 1.0                              # short texts are left alone
    assert rule(4096) == pytest.approx(1 + math.log((4096 - 32) / (2048 - 32)) / 5)
    assert rule(8192) > rule(4096) and rule(10**9) == pytest.approx(math.exp(0.5))    # grows, up to the cap
    assert QueryScale(fixed=1.3)(5) == 1.3
    assert parse_setting("plain:sharp")["sharpen"].reference == 2048
    read = parse_setting("immediate:3e-4:sharp=4096:trigger=2")
    assert read["sharpen"].reference == 4096 and read["trigger"] == 2 and read["rate"] == 3e-4
    assert parse_setting("frozen:scale=1.35@off-a")["sharpen"].fixed == 1.35

    target, draft = make_models()
    prompt = torch.tensor([[3, 1, 4, 1, 5]])
    start, budget = prompt.shape[1], 110
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, return_stats=True)
    ids = plain.output_ids

    class Watch:
        """Keeps the drafter's predictions of every round."""

        replace = False

        def __init__(self):
            self.rounds = []

        def start(self, first_drafter):
            pass

        def __call__(self, s):
            self.rounds.append((int(s.start), s.draft_logits.clone()))

    before = Watch()
    replay_decode(draft, target, ids, start, ended=False, improver=before)
    late = QueryScale(reference=60, top=8, slope=0.5, cap=3.0)                 # nothing changes until it sees 60 tokens
    set_query_scale(draft, late)
    after = Watch()
    replay_decode(draft, target, ids, start, ended=False, improver=after)
    short = long = 0
    for (position, own), (other_position, other) in zip(before.rounds, after.rounds):
        assert position == other_position
        if position + draft.block_size <= late.reference:
            assert torch.equal(own, other)                                     # a short text is drafted as before
            short += 1
        else:
            assert not torch.equal(own, other)                                 # a long one is not
            long += 1
    assert short >= 5 and long >= 5
    set_query_scale(draft, None)

    # As a setting of a trainer it is switched on for that trainer's answers, with nothing trained.
    adapters = attach_adapters(draft, rank=4)
    strong = QueryScale(fixed=4.0)
    sharp = DraftTrainer(draft, adapters, rate=0.0, sharpen=strong)
    sharp.begin()
    assert all(layer.self_attn.query_scale is strong for layer in draft.layers)
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=sharp, return_stats=True)
    assert torch.equal(live.output_ids, ids)                                   # still the target's own text
    assert live.acceptance_lengths != plain.acceptance_lengths                 # from other drafts
    sharp.begin()
    replayed = replay_decode(draft, target, ids, start, ended=False, improver=sharp)
    assert replayed.acceptance_lengths == live.acceptance_lengths
    assert sharp.totals["updates"] == 0                                        # nothing was trained
    # The same at every length, with a scale of one, is the plain drafter.
    one = DraftTrainer(draft, adapters, rate=0.0, sharpen=QueryScale(fixed=1.0))
    one.begin()
    assert replay_decode(draft, target, ids, start, ended=False, improver=one).acceptance_lengths \
        == plain.acceptance_lengths
    # It can be combined with training during the answer, and decoding and replay still agree.
    both = DraftTrainer(draft, adapters, labels="immediate", rate=3e-3, sharpen=strong)
    both.begin()
    trained = dflash_generate_redraft(draft, target, prompt, budget, None, improver=both, return_stats=True)
    assert torch.equal(trained.output_ids, ids) and both.totals["updates"] > 0
    both.begin()
    assert replay_decode(draft, target, ids, start, ended=False, improver=both).acceptance_lengths \
        == trained.acceptance_lengths
    # A trainer without the rule switches it off again.
    off = DraftTrainer(draft, adapters, rate=0.0)
    off.begin()
    assert all(layer.self_attn.query_scale is None for layer in draft.layers)
    assert replay_decode(draft, target, ids, start, ended=False, improver=off).acceptance_lengths \
        == plain.acceptance_lengths
    detach_adapters(draft)
