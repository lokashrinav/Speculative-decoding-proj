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


def make_models(seed=17, second=False):
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
        config = Qwen3Config(num_hidden_layers=1, **common)
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
