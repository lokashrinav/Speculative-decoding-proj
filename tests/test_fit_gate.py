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
    assert fg.split_of(["shard0of4.pt", 7]) == fg.split_of(["shard0of4.pt", 7])
    names = fg.feature_names()
    assert len(names) == 38 and names[2][0] == "is_starting_draft"
    mask = fg.set_mask(("token",), torch.device("cpu"))
    assert int(mask.sum()) == 9 + 12 and int(fg.set_mask((), torch.device("cpu")).sum()) == 9


def test_gate_pipeline_runs_end_to_end_without_storing_rounds(tmp_path, monkeypatch):
    """Tiny models: collect cells from training answers, fit the gates, score the other answers,
    and put the report together. Only the small files are written."""
    import dflash.redraft
    import nexus.gate_pipeline as gp
    from dflash.hindsight import MemoryProbe
    from dflash.pair_editor import BigramTable, PairEditor, PairImprover
    from dflash.replay import replay_decode
    from tests.test_redraft import make_models

    monkeypatch.setattr(dflash.redraft, "_cuda_time", time.perf_counter)
    target, draft = make_models()
    generator = torch.Generator().manual_seed(0)
    sequences, lengths = [], []
    for number in range(16):
        prompt = torch.randint(1, 32, (1, 4 + number % 3), generator=generator)
        result = dflash.redraft.dflash_generate_redraft(draft, target, prompt, 120, None, block_size=8, return_stats=True)
        sequences.append(result.output_ids[0].to(torch.int32))
        lengths.append(prompt.shape[1])
    data = tmp_path / "data"
    data.mkdir()
    torch.save({"sequences": sequences, "prompt_lengths": lengths, "sources": ["aime-before-2024", "longwriter"] * 8},
               data / "shard0of1.pt")
    torch.manual_seed(1)
    table = BigramTable.count([(torch.randint(0, 32, (60,), generator=generator), 4) for _ in range(12)], 32)
    editor = PairEditor(16, rank=8, top_k=8, bigram_weight=0.5).double().eval()
    torch.nn.init.normal_(editor.up, std=1.0)
    monkeypatch.setattr(gp, "load_transformers_models", lambda *a: (target, draft, None))
    monkeypatch.setattr(gp, "stop_token_ids", lambda *a: None)
    monkeypatch.setattr(gp, "MemoryProbe", lambda **kw: MemoryProbe(capacity=512, top_k=16, editor=PairImprover(editor, table), dump=True))
    monkeypatch.setattr(gp, "replay_decode", lambda *a, **k: replay_decode(*a, **{**k, "block_size": 8}))
    work = tmp_path / "work"

    def step(mode, name, *extra):
        results = tmp_path / name
        results.mkdir()
        monkeypatch.setattr(sys, "argv", ["gate_pipeline.py", "--mode", mode, "--data-dir", str(data), "--work-dir", str(work),
                                          "--results-dir", str(results), "--epochs", "2", *extra])
        gp.main()
        return results

    collected = step("collect", "collect", "--cells-per-answer", "60")
    cells = torch.load(work / "cells-0of1.pt")
    assert set(cells) == {"plain", "editor"}
    x, lp, target_index = cells["plain"]
    assert x.dtype == torch.float16 and x.shape[1:] == (16, 38) and lp.shape == (x.shape[0], 16) and target_index.shape == (x.shape[0],)
    status = json.load(open(collected / "status.json"))
    train = sum(fg.split_of(["shard0of1.pt", index]) == "train" for index in range(16))
    assert status["done"] and status["answers"] == train and 0 < x.shape[0] <= 60 * train
    assert "with_editor" in status["all"]["everything"]                      # the editor's draft was scored too
    assert glob.glob(str(collected / "answers-*.json")) and not glob.glob(str(work / "*rounds*"))

    fitted = json.load(open(step("fit", "fit") / "status.json"))
    assert fitted["done"] and len(fitted["gates"]) == 8 and fitted["fitted_on_cells"]["plain"] == x.shape[0]
    gates = fg.load_gates(str(work / "gates.pt"), torch.device("cpu"))
    assert set(gates) == {f"{base}|{name}" for base in fg.BASES for name in fg.SETS}
    assert float(gates["editor|both"].prior) != 0.0 and int(gates["plain|token"].mask.sum()) == 9 + 12

    scored = step("score", "score")
    files = sorted(glob.glob(str(scored / "gate-scores-*.json")))
    rows = [row for path in files for row in json.load(open(path))["answers"]]
    assert len(rows) == 16 - train and {row["part"] for row in rows} <= {"validation", "test"}
    assert len(rows[0]["sums"]["plain|both"]) == len(fg.MARGINS) and len(rows[0]["sums"]["plain|both"][0]) == len(fg.GROUPS)
    out = fg.report(rows)
    if out["answers"]["validation"] and out["answers"]["test"]:
        assert set(out["results"]) == {"plain", "editor"} and set(out["results"]["plain"]) == set(fg.SETS)
        row = out["results"]["plain"]["hidden"]
        assert row["margin"] in fg.MARGINS and "everything" in row["test"]
        # With only the drafter's own probabilities there is nothing to learn from: the control changes almost nothing.
        assert abs(out["results"]["plain"]["drafter"]["test"]["everything"]["more_tokens_one_edit"]["value"]) < 0.05
