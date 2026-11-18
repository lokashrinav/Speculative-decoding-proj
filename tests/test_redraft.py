import time

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import dflash.model as decoding
from dflash.redraft import SelfRedraft, dflash_generate_redraft
from dflash.validation import check_greedy_output


@pytest.fixture(autouse=True)
def cpu_timer(monkeypatch):
    monkeypatch.setattr("dflash.redraft._cuda_time", time.perf_counter)


def make_models(seed=17, second=False, layers=1):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    common = dict(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, bos_token_id=0, eos_token_id=30,
        pad_token_id=0, attention_dropout=0.0, tie_word_embeddings=False,
    )
    target = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=2, **common)).eval().double()

    def drafter():
        config = Qwen3Config(num_hidden_layers=layers, **common)
        config.num_target_layers = 2
        config.target_layer_ids = [0]
        config.mask_token_id = 31
        config.block_size = 4
        config.is_causal = False
        return decoding.DFlashDraftModel(config).eval().double()

    return (target, drafter(), drafter()) if second else (target, drafter())


class OracleImprover:
    """Returns the target's true continuation, so the alternative wins whenever the first draft errs."""

    def __init__(self, reference, wrong_from=None):
        self.reference = reference
        self.wrong_from = wrong_from
        self.calls = 0

    def start(self, first_drafter):
        pass

    def __call__(self, s):
        self.calls += 1
        block_len = s.block_ids.shape[1]
        alternative = self.reference[:, s.start : s.start + block_len].clone()
        if alternative.shape[1] < block_len:
            return None
        if self.wrong_from is not None:
            alternative[:, self.wrong_from :] = (alternative[:, self.wrong_from :] + 1) % 30
        return alternative


class ConstantImprover:
    def start(self, first_drafter):
        pass

    def __call__(self, s):
        alternative = s.block_ids.clone()
        alternative[:, 2:] = 7
        return alternative


def reference_output(target, draft, prompt, budget):
    return decoding.dflash_generate(draft, target, prompt, budget, None, block_size=1)


@pytest.mark.parametrize("budget", [1, 2, 3, 5, 11, 40])
@pytest.mark.parametrize("block_size", [3, 4, 8, 16])
def test_no_improver_matches_ordinary_dflash(budget, block_size):
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3]])
    ordinary = decoding.dflash_generate(draft, target, prompt, budget, None, block_size=block_size)
    ours = dflash_generate_redraft(draft, target, prompt, budget, None, block_size=block_size)
    assert torch.equal(ordinary, ours)


@pytest.mark.parametrize("seed", [17, 3, 99])
@pytest.mark.parametrize("budget", [2, 5, 11, 40])
@pytest.mark.parametrize("block_size", [3, 4, 8, 16])
@pytest.mark.parametrize("keep", [1, 2, "confidence"])
def test_self_redraft_is_lossless(seed, budget, block_size, keep):
    target, draft = make_models(seed)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    ours = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=SelfRedraft(keep=keep), block_size=block_size,
    )
    assert ours.shape[1] == prompt.shape[1] + budget
    check_greedy_output(target, prompt, ours, budget, None)


@pytest.mark.parametrize("budget", [5, 11, 40])
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_separate_second_drafter_is_lossless(budget, block_size):
    target, draft, second = make_models(second=True)
    prompt = torch.tensor([[1, 2, 3]])
    ours = dflash_generate_redraft(
        draft, target, prompt, budget, None,
        improver=SelfRedraft(model=second, keep="confidence"), block_size=block_size,
    )
    check_greedy_output(target, prompt, ours, budget, None)


@pytest.mark.parametrize("seed", [17, 3])
@pytest.mark.parametrize("budget", [5, 11, 40, 61])
@pytest.mark.parametrize("block_size", [3, 4, 8, 16])
@pytest.mark.parametrize("wrong_from", [None, 2, 3, 5])
def test_winning_alternative_is_lossless_and_faster(seed, budget, block_size, wrong_from):
    target, draft = make_models(seed)
    prompt = torch.tensor([[1, 2, 3]])
    reference = reference_output(target, draft, prompt, budget)
    plain = dflash_generate_redraft(
        draft, target, prompt, budget, None, block_size=block_size, return_stats=True,
    )
    oracle = OracleImprover(reference, wrong_from)
    ours = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=oracle, block_size=block_size, return_stats=True,
    )
    assert torch.equal(ours.output_ids, reference)
    check_greedy_output(target, prompt, ours.output_ids, budget, None)
    assert ours.rounds <= plain.rounds
    if wrong_from is None and budget >= 11 and block_size >= 4:
        assert ours.alternative_wins > 0
        assert ours.rounds < plain.rounds


@pytest.mark.parametrize("stop", [[30], [5, 30], list(range(0, 32, 3))])
def test_stop_tokens_match_ordinary_dflash(stop):
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3]])
    reference = reference_output(target, draft, prompt, 40)
    ordinary = decoding.dflash_generate(draft, target, prompt, 40, stop, block_size=8)
    for improver in (SelfRedraft(keep=1), OracleImprover(reference), ConstantImprover()):
        ours = dflash_generate_redraft(draft, target, prompt, 40, stop, improver=improver, block_size=8)
        assert torch.equal(ordinary, ours)


def test_bad_alternative_never_hurts():
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3]])
    plain = dflash_generate_redraft(draft, target, prompt, 40, None, block_size=8, return_stats=True)
    ours = dflash_generate_redraft(
        draft, target, prompt, 40, None, improver=ConstantImprover(), block_size=8, return_stats=True,
    )
    assert torch.equal(plain.output_ids, ours.output_ids)
    assert ours.rounds <= plain.rounds


def test_keep_rule_is_validated():
    with pytest.raises(ValueError):
        SelfRedraft(keep=0)
    with pytest.raises(ValueError):
        SelfRedraft(keep="most")


@pytest.mark.parametrize("seed", [17, 3])
@pytest.mark.parametrize("budget", [5, 11, 40])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("condition", ["runner_up", "random", "soft"])
def test_diagnostic_conditions_are_lossless(seed, budget, block_size, condition):
    target, draft = make_models(seed)
    prompt = torch.tensor([[1, 2, 3, 4]])
    ours = dflash_generate_redraft(
        draft, target, prompt, budget, None,
        improver=SelfRedraft(keep=1, condition=condition), block_size=block_size,
    )
    check_greedy_output(target, prompt, ours, budget, None)


@pytest.mark.parametrize("budget", [5, 11, 40])
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_resample_and_gate_are_lossless_and_logged(budget, block_size):
    from dflash.redraft import Gated, Resample, confidence_features

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4]])
    log, timers = [], {}
    ours = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=Resample(seed=1),
        block_size=block_size, round_log=log, timers=timers,
    )
    check_greedy_output(target, prompt, ours, budget, None)
    for block_len, split, first, alternative, confidence in log:
        assert block_len == block_size and len(confidence) == block_size - 1
        assert 0 <= first < block_len and -1 <= alternative < block_len
    assert timers["draft"] > 0 and timers.get("verify_one", 0) + timers.get("verify_two", 0) > 0

    width = 2 * (block_size - 1)
    for bias, expect_fire in ((100.0, True), (-100.0, False)):
        gate = Gated(SelfRedraft(keep=1), [0.0] * width, bias)
        gated = dflash_generate_redraft(
            draft, target, prompt, budget, None, improver=gate, block_size=block_size,
        )
        check_greedy_output(target, prompt, gated, budget, None)
        if budget > block_size:
            assert (gate.fired > 0) == expect_fire
    features = confidence_features(torch.randn(1, block_size - 1, 32))
    assert features.shape == (width,)
    assert abs(float(features[block_size - 1 :].sum() + features[: block_size - 1].prod()) - 1.0) < 1e-5


def test_conditions_are_validated():
    with pytest.raises(ValueError):
        SelfRedraft(keep=2, condition="soft")
    with pytest.raises(ValueError):
        SelfRedraft(keep=1, condition="other")


@pytest.mark.parametrize("budget", [5, 11, 40])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("options", [dict(keep=0), dict(keep=1, show_rest=True), dict(keep="confidence", show_rest=True)])
def test_second_drafter_options_are_lossless(budget, block_size, options):
    target, draft, second = make_models(second=True)
    prompt = torch.tensor([[1, 2, 3]])
    ours = dflash_generate_redraft(
        draft, target, prompt, budget, None,
        improver=SelfRedraft(model=second, **options), block_size=block_size,
    )
    check_greedy_output(target, prompt, ours, budget, None)


def test_keep_zero_needs_separate_weights():
    with pytest.raises(ValueError):
        SelfRedraft(keep=0)


EDITOR_SETTINGS = [
    dict(read_tokens="all", first_slot=1),
    dict(read_tokens="all", first_slot=2),
    dict(read_tokens="first", first_slot=2),
    dict(read_tokens="none", first_slot=1),
    dict(read_tokens="all", read_states=False, first_slot=1),
    dict(read_tokens="all", first_slot=1, normalize=True, confidence=True),
    dict(read_tokens="none", first_slot=1, normalize=True, confidence=True),
]


@pytest.mark.parametrize("budget", [5, 11, 40])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("settings", EDITOR_SETTINGS)
@pytest.mark.parametrize("replace", [True, False])
def test_editor_is_lossless_and_starts_as_identity(budget, block_size, settings, replace):
    from dflash.draft_editor import DraftEditor, EditorImprover

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4]])
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, block_size=block_size, return_stats=True)
    editor = DraftEditor(16, rank=8, block_size=16, heads=2, **settings).double().eval()
    untrained = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=EditorImprover(editor, replace=replace),
        block_size=block_size, return_stats=True,
    )
    # A fresh editor adds nothing, so the draft is left exactly as it was.
    assert untrained.rounds_with_alternative == 0 and untrained.rounds_edited == 0
    assert torch.equal(untrained.output_ids, plain.output_ids)
    assert untrained.rounds == plain.rounds
    torch.manual_seed(5)
    torch.nn.init.normal_(editor.up, std=1.0)
    edited = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=EditorImprover(editor, replace=replace),
        block_size=block_size, return_stats=True,
    )
    check_greedy_output(target, prompt, edited.output_ids, budget, None)
    if replace:
        assert edited.rounds_with_alternative == 0


def test_replacing_with_the_true_continuation_accepts_whole_blocks():
    """A perfect editor in replace mode: every block is accepted in full."""
    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3]])
    reference = reference_output(target, draft, prompt, 40)
    oracle = OracleImprover(reference)
    oracle.replace = True
    result = dflash_generate_redraft(
        draft, target, prompt, 40, None, improver=oracle, block_size=8, return_stats=True,
    )
    assert torch.equal(result.output_ids, reference)
    assert result.rounds_with_alternative == 0
    assert result.rounds <= 40 // 8 + 2


def test_editor_shortcut_matches_full_output_head():
    """Reusing the first pass's logits plus a low-rank correction equals rerunning the output head."""
    from types import SimpleNamespace

    from dflash.draft_editor import DraftEditor, EditorImprover
    from dflash.model import _output_head, _raw_input_embeddings

    target, draft = make_models()
    torch.manual_seed(2)
    for first_slot in (1, 2):
        editor = DraftEditor(16, rank=8, block_size=16, heads=2, first_slot=first_slot).double().eval()
        torch.nn.init.normal_(editor.up, std=1.0)
        states = torch.randn(1, 8, 16, dtype=torch.float64)
        block = torch.randint(1, 30, (1, 8))
        head = _output_head(target)
        state = SimpleNamespace(
            block_ids=block, draft_states=states, target=target, first_drafter=draft,
            draft_logits=draft.compute_logits(states[:, 1:], head),
        )
        embeddings = _raw_input_embeddings(target, block, 1.0)
        full = draft.compute_logits(editor(states, embeddings), head)
        improver = EditorImprover(editor)
        short = improver._logits(state, editor.coefficients(states, embeddings))
        assert improver.folded is not None
        assert torch.allclose(full, short, atol=1e-9)


def test_editor_with_confidence_shortcut_and_margin():
    """The rescaled editor that reads the first pass's confidence: same shortcut, and a
    large margin leaves the draft exactly as it was."""
    from types import SimpleNamespace

    from dflash.draft_editor import DraftEditor, EditorImprover, draft_confidence
    from dflash.model import _output_head, _raw_input_embeddings

    target, draft = make_models()
    torch.manual_seed(3)
    editor = DraftEditor(16, rank=8, block_size=16, heads=2, normalize=True, confidence=True).double().eval()
    torch.nn.init.normal_(editor.up, std=1.0)
    states = torch.randn(1, 8, 16, dtype=torch.float64)
    block = torch.randint(1, 30, (1, 8))
    head = _output_head(target)
    logits = draft.compute_logits(states[:, 1:], head)
    block[:, 1:] = logits.argmax(dim=-1)
    state = SimpleNamespace(block_ids=block, draft_states=states, target=target, first_drafter=draft, draft_logits=logits)
    embeddings = _raw_input_embeddings(target, block, 1.0)
    confidence = draft_confidence(logits).double()
    full = draft.compute_logits(editor(states, embeddings, confidence), head)
    improver = EditorImprover(editor)
    short = improver._logits(state, editor.coefficients(states, embeddings, confidence))
    assert torch.allclose(full, short, atol=1e-9)
    assert torch.equal(improver(state)[:, 1:], full.argmax(dim=-1))
    assert not torch.equal(improver(state), block)
    assert torch.equal(EditorImprover(editor, margin=1e9)(state), block)

    prompt = torch.tensor([[1, 2, 3, 4]])
    plain = dflash_generate_redraft(draft, target, prompt, 40, None, return_stats=True)
    careful = dflash_generate_redraft(
        draft, target, prompt, 40, None, improver=EditorImprover(editor, margin=1e9), return_stats=True,
    )
    assert careful.rounds_edited == 0 and careful.rounds == plain.rounds
    assert torch.equal(careful.output_ids, plain.output_ids)


@pytest.mark.parametrize("budget", [5, 11, 40])
@pytest.mark.parametrize("block_size", [4, 16])
def test_left_to_right_bigram_editor_is_lossless(budget, block_size):
    """Whatever the left-to-right editor picks, the output is still the target's own greedy text."""
    from dflash.pair_editor import BigramImprover, BigramTable

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4]])
    vocabulary = target.get_input_embeddings().weight.shape[0]
    generator = torch.Generator().manual_seed(4)
    table = BigramTable.count([(torch.randint(0, vocabulary, (80,), generator=generator), 4) for _ in range(20)], vocabulary)
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, block_size=block_size, return_stats=True)
    edited = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=BigramImprover(table, top_k=8, weight=3.0),
        block_size=block_size, return_stats=True,
    )
    check_greedy_output(target, prompt, edited.output_ids, budget, None)
    assert edited.rounds_with_alternative == 0
    same = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=BigramImprover(table, weight=0.0),
        block_size=block_size, return_stats=True,
    )
    assert same.rounds_edited == 0 and same.rounds == plain.rounds
    assert torch.equal(same.output_ids, plain.output_ids)


@pytest.mark.parametrize("budget", [5, 40])
@pytest.mark.parametrize("block_size", [4, 16])
@pytest.mark.parametrize("bigram_weight", [0.0, 0.5])
def test_left_to_right_pair_editor_is_lossless_and_starts_as_identity(budget, block_size, bigram_weight):
    from dflash.pair_editor import BigramTable, PairEditor, PairImprover

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4]])
    vocabulary = target.get_input_embeddings().weight.shape[0]
    generator = torch.Generator().manual_seed(4)
    table = BigramTable.count([(torch.randint(0, vocabulary, (80,), generator=generator), 4) for _ in range(20)], vocabulary)
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, block_size=block_size, return_stats=True)
    editor = PairEditor(16, rank=8, top_k=8, bigram_weight=bigram_weight).double().eval()
    untrained = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=PairImprover(editor, table), block_size=block_size, return_stats=True,
    )
    check_greedy_output(target, prompt, untrained.output_ids, budget, None)
    if bigram_weight == 0:
        assert untrained.rounds_edited == 0 and untrained.rounds == plain.rounds
        assert torch.equal(untrained.output_ids, plain.output_ids)
    torch.manual_seed(6)
    torch.nn.init.normal_(editor.up, std=1.0)
    edited = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=PairImprover(editor, table), block_size=block_size, return_stats=True,
    )
    check_greedy_output(target, prompt, edited.output_ids, budget, None)
    assert edited.rounds_with_alternative == 0
    careful = dflash_generate_redraft(
        draft, target, prompt, budget, None, improver=PairImprover(editor, table, margin=1e9),
        block_size=block_size, return_stats=True,
    )
    assert careful.rounds_edited == 0 and torch.equal(careful.output_ids, plain.output_ids)


def test_pair_editor_short_decoding_path_matches_the_plain_one():
    """Folding the output head and the token inputs into the editor must not change any pick."""
    from types import SimpleNamespace

    from dflash.model import _output_head
    from dflash.pair_editor import BigramTable, PairEditor, PairImprover

    target, draft = make_models()
    vocabulary = target.get_input_embeddings().weight.shape[0]
    generator = torch.Generator().manual_seed(8)
    table = BigramTable.count([(torch.randint(0, vocabulary, (80,), generator=generator), 4) for _ in range(20)], vocabulary)
    head = _output_head(target)
    for bigram_weight, read_states, k, margin, kind in (
        (0.0, True, 8, 0.0, "full"), (0.5, True, 16, 0.3, "full"), (0.5, False, 4, 0.0, "full"),
        (0.0, True, 16, 0.0, "fast"), (0.5, True, 8, 0.2, "fast"),
    ):
        torch.manual_seed(9)
        editor = PairEditor(16, rank=8, top_k=k, bigram_weight=bigram_weight, read_states=read_states,
                            confidence=kind).double().eval()
        torch.nn.init.normal_(editor.up, std=1.0)
        improver = PairImprover(editor, table, margin=margin)
        changed = 0
        for positions in (1, 2, 7, 15):
            states = torch.randn(1, positions + 1, 16, dtype=torch.float64, generator=generator)
            logits = draft.compute_logits(states[:, 1:], head)
            block = torch.cat((torch.randint(1, 30, (1, 1), generator=generator), logits.argmax(dim=-1)), dim=1)
            state = SimpleNamespace(block_ids=block, draft_states=states, target=target, first_drafter=draft, draft_logits=logits)
            fast, plain = improver(state), improver.reference(state)
            assert torch.equal(fast, plain)
            changed += int((fast != block).any())
        assert changed > 0


def test_hindsight_probe_labels_are_exact_and_drafts_are_untouched():
    """The memory probe must not change decoding, every stored label must be the token the
    answer really has at that position, and its counts must agree with the decoding loop."""
    from dflash.hindsight import MemoryProbe, accepted_length, summarize

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4]])
    budget = 120
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, return_stats=True)
    probe = MemoryProbe(capacity=64, neighbours=4)
    probe.begin()
    probed = dflash_generate_redraft(draft, target, prompt, budget, None, improver=probe, return_stats=True)
    assert torch.equal(probed.output_ids, plain.output_ids) and probed.rounds == plain.rounds
    final = probed.output_ids[0]
    stored = probe.positions[: probe.size]
    assert probe.size > 0 and int(stored.max()) < final.shape[0]
    assert torch.equal(probe.labels[: probe.size], final[stored])          # hindsight labels are exact
    assert probe.suffix.tokens == final[: probe.last_start + 1].tolist()   # the token memory saw the committed text only
    # The probe's reading of "accepted" is the decoding loop's, round by round.
    checked = 0
    for number, entry in enumerate(probe.log):
        first = entry["start"] + 1
        if first + entry["draft"].shape[0] < final.shape[0]:
            truth = final[first : first + entry["draft"].shape[0]]
            assert int(accepted_length(entry["draft"][None], truth[None])) + 1 == probed.acceptance_lengths[number]
            checked += 1
    assert checked >= 3
    answer = probe.finish(probed.output_ids, ended=False)
    names = {name: number for number, name in enumerate(probe.columns)}
    row = answer["bins"]["0"]
    assert list(answer["bins"]) == ["0"] and answer["loop_start"] is None
    assert row[names["rounds"]] == len(probe.log) == probe.round
    assert row[names["raw|rescue"]] <= row[names["raw|label_is_target"]] <= row[names["first_rejections"]] <= row[names["rounds"]]
    assert row[names["tokens"]] <= final.shape[0] - prompt.shape[1] + 1
    assert torch.equal(probe.totals["all"], probe.totals["clean"])
    table = summarize(probe.totals, probe.delays)
    first = table["clean"]["0-1k"]
    assert abs(first["tokens_per_round"] - row[names["tokens"]] / row[names["rounds"]]) < 1e-9
    assert table["clean"]["everything"]["rounds"] == first["rounds"] and "2k+" not in table["clean"]
    assert len(first["rules"]["raw"]) == 36 and len(first["rules"]["suffix"]) == 16 and "with_editor" not in first
    # A second answer starts with an empty memory.
    probe.begin()
    assert probe.size == 0 and probe.pending == [] and probe.log == [] and probe.suffix.tokens == []


def test_hindsight_scoring_on_a_hand_made_answer():
    """Rescues, breaks and gained tokens for an answer small enough to check by hand."""
    from dflash.hindsight import MemoryProbe, summarize

    final = (100 + torch.arange(2600))[None]          # the target's token at position p is 100 + p
    probe = MemoryProbe()
    probe.begin()
    probe.prompt_length = 2

    def entry(start, draft, raw, centered, previous=None, suffix=None, memory_size=0, sure=0.5, edited=None):
        draft = torch.tensor(draft)
        truth = 100 + start + 1 + torch.arange(len(draft))
        top = torch.stack((draft, truth, torch.full_like(draft, 555), torch.full_like(draft, 888)), dim=1)
        as_tensors = lambda rows: (torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows]),
                                   torch.tensor([r[2] for r in rows]))
        silent = [(-1, 0.0, 0.0)] * len(draft)
        nothing = [(-1, 0.0, 0, 0)] * len(draft)
        # The drafter gives its own token the probability `sure`, the other candidates share the rest.
        log_probs = torch.tensor([[sure] + [(1 - sure) / 3] * 3] * len(draft)).log()
        made = dict(start=start, draft=draft, top=top, top_log_probs=log_probs, memory_size=memory_size, raw=as_tensors(raw or silent),
                    centered=as_tensors(centered or silent), previous=previous or nothing, suffix=suffix or nothing)
        if edited is not None:
            made["edited"] = torch.tensor(edited)
        return made

    quiet = (-1, 0.0, 0.0)  # (label, agreement, similarity)
    # Round A starts at position 2: two draft tokens are right, the third (truth 105) is wrong.
    probe.log.append(entry(
        2, [103, 104, 999, 106, 107],
        raw=[quiet, quiet, (105, 1.0, 0.92), quiet, quiet],                       # right where the draft is wrong
        centered=[(555, 1.0, 0.6), quiet, (105, 1.0, 0.6), quiet, quiet],         # also breaks the first token if trusted
        previous=[(-1, 0.0, 0, 0)] * 2 + [(105, 1.0, 3, 1)] + [(-1, 0.0, 0, 0)] * 2,
        suffix=[(-1, 0.0, 0, 0)] * 2 + [(444, 1.0, 2, 3)] + [(-1, 0.0, 0, 0)] * 2))
    # Round B starts at position 5: the whole draft is right, and the raw memory would break its second token.
    probe.log.append(entry(5, [106, 107, 108, 109, 110], raw=[quiet, (888, 1.0, 0.99), quiet, quiet, quiet],
                           centered=None, memory_size=5))
    # Round C is far into the answer (position 2500 of the answer) and its draft is right.
    probe.log.append(entry(2502, [2603, 2604, 2605, 2606, 2607], raw=None, centered=None, memory_size=9))
    answer = probe.finish(final, ended=True)
    names = {name: number for number, name in enumerate(probe.columns)}
    early, late = answer["bins"]["0"], answer["bins"]["2"]
    get = lambda key: early[names[key]]
    assert sorted(answer["bins"]) == ["0", "2"]
    assert (late[names["rounds"]], late[names["tokens"]], late[names["first_rejections"]]) == (1, 6, 0)
    assert (get("rounds"), get("tokens"), get("first_rejections"), get("target_in_top"), get("memory_nonempty")) == (2, 9, 1, 1, 0)
    for variant in ("raw", "centered"):
        assert (get(f"{variant}|label_is_target"), get(f"{variant}|rescue"), get(f"{variant}|oracle_tokens")) == (1, 1, 3)
        assert (get(f"{variant}|rescue_previous_misses"), get(f"{variant}|rescue_suffix_misses")) == (0, 1)
    assert (get("previous|rescue"), get("previous|oracle_tokens"), get("suffix|rescue"), get("suffix|oracle_tokens")) == (1, 3, 0, 0)
    assert (get("raw_or_suffix|oracle_tokens"), get("centered_or_suffix|oracle_tokens")) == (3, 3)
    rule = lambda tag: (get(f"{tag}|rescues"), get(f"{tag}|breaks"), get(f"{tag}|net_tokens"))
    assert rule("raw|0.9|0.9|1.0") == (1, 1, -1)        # +3 in round A, -4 in round B
    assert rule("raw|0.95|0.9|1.0") == (0, 1, -4)       # too strict for round A's neighbour
    assert rule("raw|0.9|0.9|0.6") == (1, 1, -1)        # the drafter gave its tokens 0.5, so the rule still acts
    assert rule("raw|0.9|0.9|0.4") == (0, 0, 0)         # ... and now it is held back everywhere
    assert rule("centered|0.5|0.6|1.0") == (0, 1, -2)   # trusts the wrong label on the first token
    assert rule("centered|0.7|0.0|1.0") == (0, 0, 0)    # never acts
    assert rule("previous|0.8|3|1.0") == (1, 0, 3) and rule("previous|0.9|5|1.0") == (0, 0, 0)
    assert rule("suffix|3|0.0|1.0") == (0, 0, 0)        # its token is not among the candidates
    assert get("editor|tokens") == 0                     # no editor ran
    # The two positions after round A's rejection get their labels two rounds later.
    assert dict(probe.delays) == {2: 2}
    table = summarize(probe.totals, probe.delays)
    assert table["clean"]["2k+"]["rounds"] == 1 and table["clean"]["everything"]["rounds"] == 3
    assert table["clean"]["0-1k"]["D1_rescue_rate"] == {"raw": 1.0, "centered": 1.0}
    assert table["clean"]["0-1k"]["best_rule"]["previous"]["more_tokens"] == 3 / 9
    assert table["D4_rounds_waited_after_first_rejection"]["median"] == 2

    # The same answer with an editor's draft recorded: in round A the editor fixes the third token
    # but breaks the fifth; the memories then have nothing left to fix there. In round B the editor
    # breaks the fourth token (truth 109), which the suffix memory would put right.
    probe = MemoryProbe()
    probe.begin()
    probe.prompt_length = 2
    probe.log.append(entry(2, [103, 104, 999, 106, 107], raw=[quiet, quiet, (105, 1.0, 0.92), quiet, quiet], centered=None,
                           edited=[103, 104, 105, 106, 555]))
    probe.log.append(entry(5, [106, 107, 108, 109, 110], raw=None, centered=None, edited=[106, 107, 108, 888, 110],
                           suffix=[(-1, 0.0, 0, 0)] * 3 + [(109, 1.0, 2, 3)] + [(-1, 0.0, 0, 0)]))
    early = probe.finish(final, ended=True)["bins"]["0"]
    get = lambda key: early[names[key]]
    assert (get("tokens"), get("editor|tokens"), get("editor|first_rejections"), get("editor|target_in_top")) == (9, 9, 2, 2)
    assert (get("after_editor|raw|rescue"), get("after_editor|raw|oracle_tokens")) == (0, 0)
    assert (get("after_editor|suffix|rescue"), get("after_editor|suffix|oracle_tokens")) == (1, 2)
    assert get("after_editor|centered_or_suffix|oracle_tokens") == 2
    row = summarize(probe.totals, probe.delays)["clean"]["0-1k"]["with_editor"]
    assert row["more_tokens_than_plain"] == 0.0 and row["rescue_rate"]["suffix"] == 0.5


def test_hindsight_token_memory_and_repetition_detector():
    from dflash.hindsight import MemoryProbe, SuffixMemory, loop_start

    memory = SuffixMemory(longest=4)
    memory.extend([1, 2, 3, 1, 2, 4, 1, 2])
    assert memory.successor([7, 1, 2], (4, 3, 2, 1)) == (3, 0.5, 2, 2)        # "1 2" was followed by 3 and by 4
    assert memory.successor([3, 1, 2], (4, 3, 2, 1)) == (4, 1.0, 1, 3)        # the longer run decides
    assert memory.successor([3, 1, 2], (1,)) == (3, 0.5, 2, 1)                # previous token only
    assert memory.successor([9], (4, 3, 2, 1)) == (-1, 0.0, 0, 0)
    looping = list(range(1000, 1300)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 60
    assert 280 <= loop_start(looping) <= 320
    assert loop_start(list(range(2000))) is None
    assert loop_start(list(range(1000, 1300)) + [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 12) is None   # too short to call
    # The two readings of the memory agree with a direct computation.
    torch.manual_seed(0)
    probe = MemoryProbe(capacity=32, neighbours=3)
    keys = torch.randn(20, 6) + 4.0
    probe._add(keys, torch.arange(20), torch.arange(20), torch.arange(20) % 15, torch.arange(20) + 100)
    queries = torch.randn(5, 6) + 4.0
    both = probe._similarities(queries)
    cosine = torch.nn.functional.cosine_similarity
    mean = keys.mean(dim=0)
    assert torch.allclose(both["raw"], cosine(queries[:, None], keys[None], dim=-1), atol=1e-5)
    assert torch.allclose(both["centered"], cosine((queries - mean)[:, None], (keys - mean)[None], dim=-1), atol=1e-4)
    label, agreement, similarity, nearest = probe._read(keys[:5])["raw"]
    assert label.tolist() == [0, 1, 2, 3, 4] and bool((similarity > 0.999).all())   # a stored state finds itself
    labels, similarities, positions, offsets, guesses = nearest
    assert labels.shape == (5, 3) and labels[:, 0].tolist() == [0, 1, 2, 3, 4] and offsets[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert guesses[:, 0].tolist() == [100, 101, 102, 103, 104]
    assert bool((similarities[:, 0] >= similarities[:, 1]).all()) and positions[:, 0].tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("budget,block_size", [(40, 4), (57, 4), (70, 3), (33, 2)])
def test_replay_gives_the_same_rounds_as_decoding(budget, block_size):
    """Decoding again without the target's check must reproduce every round: plain, with a
    draft-replacing editor, with a second draft to choose from, and with the memory probe."""
    from dflash.hindsight import MemoryProbe
    from dflash.pair_editor import BigramTable, PairEditor, PairImprover
    from dflash.replay import replay_decode

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    length = prompt.shape[1]
    plain = dflash_generate_redraft(draft, target, prompt, budget, None, block_size=block_size, return_stats=True)
    again = replay_decode(draft, target, plain.output_ids, length, ended=False, block_size=block_size)
    assert again.acceptance_lengths == plain.acceptance_lengths
    assert again.num_output_tokens == plain.num_output_tokens

    # An editor whose edited draft replaces the first draft.
    torch.manual_seed(3)
    generator = torch.Generator().manual_seed(0)
    table = BigramTable.count([(torch.randint(0, 32, (60,), generator=generator), 4) for _ in range(12)], 32)
    editor = PairEditor(16, rank=8, top_k=8, bigram_weight=0.5).double().eval()
    torch.nn.init.normal_(editor.up, std=1.0)
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=PairImprover(editor, table),
                                   block_size=block_size, return_stats=True)
    assert torch.equal(live.output_ids, plain.output_ids)
    again = replay_decode(draft, target, live.output_ids, length, ended=False, improver=PairImprover(editor, table),
                          block_size=block_size)
    assert again.acceptance_lengths == live.acceptance_lengths and again.rounds_edited == live.rounds_edited
    if block_size > 2:
        assert live.rounds_edited > 0

    # A second draft that the target checks next to the first one.
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=OracleImprover(plain.output_ids, wrong_from=3),
                                   block_size=block_size, return_stats=True)
    again = replay_decode(draft, target, live.output_ids, length, ended=False,
                          improver=OracleImprover(plain.output_ids, wrong_from=3), block_size=block_size)
    assert again.acceptance_lengths == live.acceptance_lengths and again.alternative_wins == live.alternative_wins

    # The measuring probe sees the same drafts and builds the same memory.
    first, second = MemoryProbe(capacity=64, neighbours=4), MemoryProbe(capacity=64, neighbours=4)
    live = dflash_generate_redraft(draft, target, prompt, budget, None, improver=first, block_size=block_size, return_stats=True)
    again = replay_decode(draft, target, live.output_ids, length, ended=False, improver=second, block_size=block_size)
    assert again.acceptance_lengths == live.acceptance_lengths and len(first.log) == len(second.log)
    for a, b in zip(first.log, second.log):
        assert a["start"] == b["start"] and torch.equal(a["draft"], b["draft"]) and torch.equal(a["raw"][0], b["raw"][0])
        assert torch.allclose(a["raw"][2], b["raw"][2], atol=1e-6) and torch.equal(a["raw"][3][2], b["raw"][3][2])
    if first.size:
        assert torch.equal(first.labels[: first.size], second.labels[: second.size])
        assert torch.allclose(first.keys[: first.size], second.keys[: second.size], atol=1e-6)
    assert first.finish(live.output_ids, ended=False) == second.finish(live.output_ids, ended=False)


def test_replay_of_an_answer_that_stopped_by_itself():
    from dflash.replay import replay_decode, target_features

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    free = dflash_generate_redraft(draft, target, prompt, 60, None, return_stats=True)
    tested = 0
    for stop in sorted(set(free.output_ids[0, 12:40].tolist())):
        live = dflash_generate_redraft(draft, target, prompt, 60, [stop], return_stats=True)
        if int(live.output_ids[0, -1]) != stop:
            continue
        again = replay_decode(draft, target, live.output_ids, prompt.shape[1], ended=True)
        assert again.acceptance_lengths == live.acceptance_lengths
        tested += 1
    assert tested > 0
    # The target's states are the same whether the finished answer is read in one piece or in parts.
    whole = target_features(target, draft, free.output_ids, chunk=1000)
    parts = target_features(target, draft, free.output_ids, chunk=7)
    assert whole.shape[1] == free.output_ids.shape[1] and torch.allclose(whole, parts, atol=1e-9)


def test_probe_with_an_editor_and_a_dump_of_every_round():
    """With an editor attached the probe still leaves decoding alone, scores the editor's draft at
    the same rounds, and hands back everything it recorded per round."""
    from dflash.hindsight import MemoryProbe, accepted_length
    from dflash.pair_editor import BigramTable, PairEditor, PairImprover

    target, draft = make_models()
    prompt = torch.tensor([[1, 2, 3, 4]])
    plain = dflash_generate_redraft(draft, target, prompt, 90, None, return_stats=True)
    torch.manual_seed(5)
    generator = torch.Generator().manual_seed(0)
    table = BigramTable.count([(torch.randint(0, 32, (60,), generator=generator), 4) for _ in range(12)], 32)
    editor = PairEditor(16, rank=8, top_k=8, bigram_weight=0.5).double().eval()
    torch.nn.init.normal_(editor.up, std=1.0)
    probe = MemoryProbe(capacity=64, neighbours=4, top_k=8, editor=PairImprover(editor, table), dump=True)
    probe.begin()
    probed = dflash_generate_redraft(draft, target, prompt, 90, None, improver=probe, return_stats=True)
    assert torch.equal(probed.output_ids, plain.output_ids) and probed.acceptance_lengths == plain.acceptance_lengths
    answer = probe.finish(probed.output_ids, ended=False)
    rounds = answer["rounds"]
    count, width = rounds["draft"].shape
    assert count == len(probe.log) and width == 3                          # blocks of 4: three drafted tokens
    final = probed.output_ids[0]
    for number in range(count):
        first = int(rounds["starts"][number]) + 1
        known = final[first : first + width]
        assert rounds["truth"][number, : known.shape[0]].tolist() == known.tolist()
        assert bool((rounds["truth"][number, known.shape[0] :] == -1).all())
    assert rounds["top_tokens"].shape == (count, width, 8) and rounds["top_log_probs"].dtype == torch.float16
    drafted = rounds["draft"] >= 0                                           # the last block before the length limit is shorter
    assert int((~drafted).sum()) <= 2
    assert bool((rounds["top_tokens"][:, :, 0] == rounds["draft"])[drafted].all())   # the draft is the drafter's first candidate
    assert rounds["raw_labels"].shape == (count, width, 16) and rounds["centered_offsets"].dtype == torch.int8
    # Token memory per candidate: where its guess is among the candidates, the counts say the same thing.
    counts, runs = rounds["suffix_candidate_counts"], rounds["suffix_run_counts"]
    assert counts.shape == (count, width, 4, 8) and runs.shape == (count, width, 4)
    agreed = 0
    for number, offset in (rounds["suffix_token"] >= 0).nonzero().tolist():
        level = int(rounds["suffix_length"][number, offset]) - 1
        seen = int(rounds["suffix_seen"][number, offset])
        assert int(runs[number, offset, level]) == seen and int(counts[number, offset, level].sum()) <= seen
        where = (rounds["top_tokens"][number, offset] == rounds["suffix_token"][number, offset]).nonzero()
        if where.numel():
            share = float(rounds["suffix_share"][number, offset])
            assert abs(int(counts[number, offset, level, int(where[0])]) - share * seen) < 0.01 * seen + 0.01
            assert int(counts[number, offset, level].max()) == int(counts[number, offset, level, int(where[0])])
            agreed += 1
    assert agreed > 5
    assert int(rounds["raw_labels"][0].max()) == -1                          # nothing is stored before the first round
    # Each neighbour's guess is what the drafter had drafted at that neighbour's position.
    drafted_at = {}
    for number in range(count):
        for offset in range(width):
            if int(rounds["draft"][number, offset]) >= 0:
                drafted_at[(int(rounds["starts"][number]) + 1 + offset, offset)] = int(rounds["draft"][number, offset])
    seen = 0
    for name in ("raw", "centered"):
        positions, offsets, guesses = rounds[f"{name}_positions"], rounds[f"{name}_offsets"], rounds[f"{name}_guesses"]
        for cell in (positions >= 0).nonzero().tolist():
            key = (int(positions[tuple(cell)]), int(offsets[tuple(cell)]))
            assert drafted_at[key] == int(guesses[tuple(cell)])
            assert int(rounds[f"{name}_labels"][tuple(cell)]) == int(final[key[0]])   # and its label is the target's token there
            seen += 1
    assert seen > 50
    assert bool((rounds["edited"] != rounds["draft"])[drafted].any())        # the editor did change some drafts
    names = {name: number for number, name in enumerate(probe.columns)}
    row = answer["bins"]["0"]
    truth, edited = rounds["truth"].long(), rounds["edited"].long()
    assert row[names["editor|tokens"]] == int((accepted_length(edited, truth) + 1).sum())
    assert row[names["tokens"]] == int((accepted_length(rounds["draft"].long(), truth) + 1).sum())
