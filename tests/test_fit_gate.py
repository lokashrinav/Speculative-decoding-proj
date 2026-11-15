"""Gate fitting on recorded rounds: the scoring walk and the inputs are checked on small cases."""

import glob
import json
import os
import sys
import time
import types

import torch

import nexus.fit_gate as fg


def test_walk_counts_rescues_breaks_and_the_lower_bound():
    # One candidate list for every cell: [starting draft's token, 7, 8]. truth and base per round below.
    truth = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, -1, -1]])
    base = torch.tensor([[1, 2, 9, 4], [1, 2, 3, 4], [1, 9, 3, 9], [1, 2, 5, 5]])
    valid = truth >= 0
    cand = torch.stack((base, torch.full_like(base, 7), torch.full_like(base, 8)), dim=-1)
    cand[0, 2, 1] = 3      # round 0: the right token for the wrong third cell is a candidate
    cand[2, 1, 1] = 2      # round 2: the right token for its first wrong cell
    cand[2, 3, 1] = 4      # ... and for its second wrong cell
    scores = torch.zeros(cand.shape)
    scores[0, 2, 1] = 2.0  # round 0 wants to fix the third cell: a rescue worth 2 tokens (cells 3 and 4)
    scores[1, 1, 2] = 2.0  # round 1 wants to change a right token: a break that loses 3 tokens
    scores[2, 1, 1] = 2.0  # round 2 fixes its first wrong cell (worth 2: cells 2 and 3) ...
    scores[2, 3, 1] = 2.0  # ... and, only when every edit is applied, its last cell too
    scores[3, 2, 1] = 2.0  # round 3 only wants to change a cell past the end of the answer: nothing happens
    out = fg.walk(scores, cand, base, truth, valid, margin=1.0)
    assert out["tokens"].tolist() == [3, 5, 2, 3]
    assert out["rejections"].tolist() == [True, False, True, False]
    assert out["one_edit"].tolist() == [2, -3, 2, 0]
    assert out["all_edits"].tolist() == [2, -3, 3, 0]
    assert out["at_least"].tolist() == [1, -3, 1, 0]       # one sure token per rescue, the exact loss of a break
    assert out["rescues"].tolist() == [True, False, True, False] and out["breaks"].tolist() == [False, True, False, False]
    assert out["changed"].tolist() == [True, True, True, False]
    quiet = fg.walk(scores, cand, base, truth, valid, margin=3.0)   # the margin holds every change back
    assert quiet["one_edit"].tolist() == [0, 0, 0, 0] and not bool(quiet["changed"].any())
    # A starting token that is not among the candidates is never changed.
    cand[0, 2, 0] = 6
    assert fg.walk(scores, cand, base, truth, valid, margin=1.0)["one_edit"].tolist() == [0, -3, 2, 0]


def test_reachable_cells_and_the_fixed_split():
    truth = torch.tensor([[1, 2, 3, 4], [1, 2, -1, -1]])
    base = torch.tensor([[1, 9, 3, 4], [1, 2, 5, 5]])
    assert fg.reachable(base, truth, truth >= 0).tolist() == [[True, True, False, False], [True, True, False, False]]
    parts = [fg.split_of(["shard0of4.pt", index]) for index in range(400)]
    assert 200 < parts.count("train") < 280 and 50 < parts.count("validation") < 110 and 50 < parts.count("test") < 110
    assert fg.split_of(["shard0of4.pt", 7]) == fg.split_of_file("/some/where/shard0of4-0007.pt")
    names = fg.feature_names()
    assert len(names) == 38 and names[2][0] == "is_starting_draft"
    mask = fg.set_mask(("token",), torch.device("cpu"))
    assert int(mask.sum()) == 9 + 12 and int(fg.set_mask((), torch.device("cpu")).sum()) == 9


def test_gate_fitting_runs_end_to_end_on_recorded_rounds(tmp_path, monkeypatch):
    """Tiny models: record rounds with the probe and an editor, then fit and score gates."""
    import dflash.redraft
    import nexus.probe_memory as pm
    import nexus.replay_memory as rm
    from dflash.hindsight import MemoryProbe
    from dflash.pair_editor import BigramTable, PairEditor, PairImprover
    from tests.test_redraft import make_models

    monkeypatch.setattr(dflash.redraft, "_cuda_time", time.perf_counter)
    target, draft = make_models()
    generator = torch.Generator().manual_seed(0)
    sequences, lengths = [], []
    for number in range(14):
        prompt = torch.randint(1, 32, (1, 4 + number % 3), generator=generator)
        result = dflash.redraft.dflash_generate_redraft(draft, target, prompt, 120, None, block_size=8, return_stats=True)
        sequences.append(result.output_ids[0].to(torch.int32))
        lengths.append(prompt.shape[1])
    torch.save({"sequences": sequences, "prompt_lengths": lengths, "sources": ["aime-before-2024", "longwriter"] * 7},
               tmp_path / "shard0of1.pt")
    torch.manual_seed(1)
    table = BigramTable.count([(torch.randint(0, 32, (60,), generator=generator), 4) for _ in range(12)], 32)
    editor = PairEditor(16, rank=8, top_k=8, bigram_weight=0.5).double().eval()
    torch.nn.init.normal_(editor.up, std=1.0)
    probe = MemoryProbe(capacity=512, top_k=16, editor=PairImprover(editor, table), dump=True)
    args = types.SimpleNamespace(out_dir=str(tmp_path / "rounds"), max_hours=1.0, save_every=5)
    monkeypatch.setattr(rm, "replay_decode", lambda *a, **k: __import__("dflash.replay", fromlist=["x"]).replay_decode(
        *a, **{**k, "block_size": 8}))
    rm.run(draft, target, None, rm.load_answers(str(tmp_path)), probe, args, torch.device("cpu"), lambda *a: None)
    files = sorted(glob.glob(str(tmp_path / "rounds" / "*.pt")))
    assert len(files) == 14
    record = torch.load(files[0])
    assert record["source"] == "aime-before-2024" and record["origin"] == ["shard0of1.pt", 0] and record["ended"] is False
    made = fg.build(record, "editor", torch.device("cpu"))
    rounds, width, top = record["top_tokens"].shape
    assert made["x"].shape == (rounds, width, top, 38) and bool(torch.isfinite(made["x"]).all())
    names = [name for name, _ in fg.feature_names()]
    drafted = record["draft"] >= 0
    assert bool((made["x"][..., names.index("is_first_draft")][..., 0] == 1)[drafted].all())   # the draft is the first candidate
    weights = made["x"][..., names.index("raw_weight_nearest_16")].sum(-1)
    assert bool((weights <= 1 + 1e-5).all()) and float(weights.max()) > 0.5                     # a share of the neighbours' weight
    shares = made["x"][..., names.index("followed_run_of_1_share")]
    assert bool((shares >= 0).all()) and bool((shares.sum(-1) <= 1 + 1e-5).all())
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(sys, "argv", ["fit_gate.py", "--rounds-dir", str(tmp_path / "rounds"), "--results-dir", str(results),
                                      "--epochs", "2"])
    fg.main()
    out = json.load(open(results / "status.json"))
    assert out["done"] and sum(out["answers"].values()) == 14 and min(out["answers"].values()) >= 1
    assert set(out["results"]) == {"plain", "editor"} and set(out["results"]["plain"]) == set(fg.SETS)
    for base in fg.BASES:
        for name in fg.SETS:
            row = out["results"][base][name]
            assert row["margin"] in fg.MARGINS and "everything" in row["test"]
            everything = row["test"]["everything"]
            assert everything["more_tokens_at_least"]["value"] <= everything["more_tokens_one_edit"]["value"] + 1e-9 or base == "editor"
    # With only the drafter's own probabilities there is nothing to learn from: the control changes almost nothing.
    assert abs(out["results"]["plain"]["drafter"]["test"]["everything"]["more_tokens_one_edit"]["value"]) < 0.05
