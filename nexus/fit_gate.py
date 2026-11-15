"""Fits small gates on the recorded rounds and scores them on held-out answers (plan item E4).

Each round of each replayed answer was recorded by the memory probe: the draft, the
drafter's 16 candidates with their probabilities, the nearest stored states of the
hidden-state memory, the token memory's counts, an editor's draft, and the target's
tokens. From that, a small network learns to re-pick each draft token among the 16
candidates, and it is scored the way decoding scores a draft: by accepted tokens.

Four sets of inputs are compared, each trained the same way:
  drafter   the drafter's own probabilities only (a control: it should gain nothing)
  token     plus the token memory (which tokens followed the same run of tokens before)
  hidden    plus the hidden-state memory (labels of the nearest earlier drafter states)
  both      plus both memories
and each on two starting drafts: the drafter's own draft ("plain") and the
left-to-right editor's draft ("editor").

A pick is changed only when the network prefers another candidate by a margin; the
margin is chosen on validation answers. Results are for test answers, which are never
used for fitting or for choosing anything, with 95% intervals from resampling whole
answers. Rounds inside endless repetition are left out.

What is exact and what is not:
  - "one edit" changes at most one token per round (the first the gate wants to change).
    On the plain draft its accepted tokens are computed exactly from the record.
  - "all edits" changes every token the gate wants. It is exact for the hidden-state
    memory alone; with the token memory it is approximate, because that memory's counts
    for the few tokens after a change were recorded for the unchanged draft.
  - On the editor's draft, what the editor would draft after a changed token is not in
    the record. "at least" counts one gained token per rescue (a lower bound), and
    "one edit" continues with the editor's recorded tokens (an estimate).
"""

import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch

from dflash.hindsight import accepted_length
from nexus.probe_memory import write_json

SETS = {"drafter": (), "token": ("token",), "hidden": ("hidden",), "both": ("token", "hidden")}
BASES = ("plain", "editor")
MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GROUPS = (("0-2k", 0, 2000), ("2k+", 2000, 10**9), ("8k+", 8000, 10**9), ("everything", 0, 10**9))
MATH = ("aime-before-2024", "math-train-level5", "metamath-math")
COUNTS = ("rounds", "tokens", "rejections", "one_edit", "all_edits", "at_least", "rescues", "breaks", "changed")
TEMPERATURE = 0.05


def split_of(origin):
    """Which part an answer belongs to, fixed by its name: 60% train, 20% validation, 20% test."""
    bucket = int(hashlib.md5(f"{origin[0]}:{origin[1]}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def feature_names():
    """(name, group) of every input, in order. Inputs outside a set's groups are zeroed."""
    names = [("log_probability", "drafter"), ("is_first_draft", "drafter"), ("is_starting_draft", "drafter"), ("rank", "drafter")]
    for variant in ("raw", "centered"):
        names += [(f"{variant}_{part}", "hidden") for part in (
            "weight_nearest_8", "weight_nearest_16", "share_of_16", "best_similarity", "corrects_same_guess", "confirms_same_guess")]
    for length in range(1, 5):
        names += [(f"followed_run_of_{length}_share", "token"), (f"followed_run_of_{length}_count", "token")]
    names += [(part, "drafter") for part in ("entropy", "top_probability", "second_probability", "offset", "position")]
    names += [("memory_size", "hidden")]
    for variant in ("raw", "centered"):
        names += [(f"{variant}_top_similarity", "hidden"), (f"{variant}_similarity_gap", "hidden")]
    names += [(f"run_of_{length}_seen", "token") for length in range(1, 5)]
    return names


def set_mask(groups, device):
    return torch.tensor([group == "drafter" or group in groups for _, group in feature_names()], dtype=torch.float32, device=device)


def build(record, base_name, device):
    """Everything the gate needs for one answer, for every cell (round, offset):
    x (R, W, K, F) inputs, lp (R, W, K) the drafter's log-probabilities, candidates,
    the starting draft, the target's tokens and which cells have one."""
    cand = record["top_tokens"].to(device).long()
    lp = record["top_log_probs"].to(device).float()
    draft = record["draft"].to(device).long()
    truth = record["truth"].to(device).long()
    base = record["edited"].to(device).long() if base_name == "editor" else draft
    valid = (truth >= 0) & (draft >= 0)
    rounds, width, top = cand.shape
    parts = [lp / 10, (cand == draft[..., None]).float(), (cand == base[..., None]).float(),
             (torch.arange(top, device=device) / top).expand(rounds, width, top)]
    for variant in ("raw", "centered"):
        labels = record[f"{variant}_labels"].to(device).long()
        similar = record[f"{variant}_similarities"].to(device).float()
        guesses = record[f"{variant}_guesses"].to(device).long()
        present = labels >= 0
        weights = []
        for nearest in (8, labels.shape[-1]):
            scores = (similar[..., :nearest] / TEMPERATURE).masked_fill(~present[..., :nearest], -1e9)
            weight = torch.softmax(scores, dim=-1) * present[..., :nearest]
            weights.append(torch.nn.functional.pad(weight, (0, labels.shape[-1] - nearest)))
        match = labels[..., None, :] == cand[..., :, None]                     # (R, W, K, N): neighbour n has label k
        wrong_then = present & (guesses != labels)                             # the drafter's guess there was a mistake
        same_guess = present & (guesses == draft[..., None])                   # ... and it was the token it drafts now
        parts += [(match * weights[0][..., None, :]).sum(-1), (match * weights[1][..., None, :]).sum(-1),
                  (match & present[..., None, :]).float().mean(-1), (match * similar[..., None, :]).amax(-1).clamp(min=0),
                  (match * (weights[1] * (wrong_then & same_guess))[..., None, :]).sum(-1),
                  (match * (weights[1] * (~wrong_then & same_guess))[..., None, :]).sum(-1)]
    counts = record["suffix_candidate_counts"].to(device).float()              # (R, W, 4, K)
    runs = record["suffix_run_counts"].to(device).float()                      # (R, W, 4)
    for level in range(counts.shape[2]):
        parts += [counts[:, :, level] / runs[:, :, level, None].clamp(min=1), torch.log1p(counts[:, :, level]) / 5]
    position = (record["starts"].to(device).long()[:, None] - int(record["prompt_length"]) + torch.arange(width, device=device)[None]).clamp(min=0)
    context = [record["entropy"].to(device).float() / 5, lp[..., 0].exp(), lp[..., 1].exp(),
               (torch.arange(width, device=device) / width).expand(rounds, width), torch.log1p(position.float()) / 10,
               (torch.log1p(record["memory_size"].to(device).float()) / 10)[:, None].expand(rounds, width)]
    for variant in ("raw", "centered"):
        similar = record[f"{variant}_similarities"].to(device).float()
        context += [similar[..., 0], similar[..., 0] - similar[..., 7]]
    context += [torch.log1p(runs[:, :, level]) / 5 for level in range(runs.shape[2])]
    x = torch.stack(parts + [value[..., None].expand(rounds, width, top) for value in context], dim=-1)
    return dict(x=x, lp=lp, cand=cand, base=base, truth=truth, valid=valid, position=position[:, 0])


def reachable(base, truth, valid):
    """Cells decoding would reach if nothing were changed: every earlier token of the starting draft is right."""
    right = ((base == truth) & valid).long()
    before = torch.cat((torch.ones_like(right[:, :1]), right[:, :-1]), dim=1).cumprod(dim=1).bool()
    return before & valid


class Gate(torch.nn.Module):
    """Scores each candidate: the drafter's log-probability, a preference for the starting draft, and a small network."""

    def __init__(self, inputs, mask, prior=0.0, hidden=48):
        super().__init__()
        self.register_buffer("mask", mask)
        self.prior = torch.nn.Parameter(torch.tensor(float(prior)))
        self.net = torch.nn.Sequential(torch.nn.Linear(inputs, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.GELU(), torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, lp):
        x = x.float() * self.mask
        return lp.float() + self.prior * x[..., 2] + self.net(x)[..., 0]


def fit(x, lp, target, mask, prior, epochs, device, batch=8192, seed=0):
    """Cross-entropy over the candidates on reachable cells (x is kept in half precision)."""
    torch.manual_seed(seed)
    gate = Gate(x.shape[-1], mask, prior).to(device)
    steps = max(20, epochs * x.shape[0] // batch)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        rows = torch.randint(0, x.shape[0], (min(batch, x.shape[0]),), device=device, generator=generator)
        loss = torch.nn.functional.cross_entropy(gate(x[rows], lp[rows]), target[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    return gate.eval()


@torch.no_grad()
def walk(scores, cand, base, truth, valid, margin):
    """Per round: tokens accepted from the starting draft, and what changing picks would add.
    A pick is changed where another candidate scores higher than the starting draft's token by more than margin."""
    found = cand == base[..., None]
    base_score = torch.where(found.any(-1), (scores * found).sum(-1), torch.full_like(scores[..., 0], float("inf")))
    best_score, best_index = scores.max(dim=-1)
    pick = cand.gather(-1, best_index[..., None])[..., 0]
    change = valid & (pick != base) & (best_score - base_score > margin)
    first_change = change & (change.long().cumsum(dim=1) == 1)
    before = accepted_length(base, truth)
    one_edit = accepted_length(torch.where(first_change, pick, base), truth) - before
    all_edits = accepted_length(torch.where(change, pick, base), truth) - before
    # A lower bound that needs nothing about what follows a changed token: a break loses what
    # the round had after it, a rescue of the first wrong token gains at least that one token.
    where = torch.where(first_change.any(1), first_change.long().argmax(dim=1), torch.full_like(before, base.shape[1]))
    fixed = (where == before) & (one_edit > 0)
    at_least = torch.where(where < before, where - before, fixed.long())
    rejected = before < valid.sum(dim=1)
    return dict(tokens=before + 1, rejections=rejected, one_edit=one_edit, all_edits=all_edits, at_least=at_least,
                rescues=one_edit > 0, breaks=one_edit < 0, changed=change.any(dim=1))


def answer_sums(outcome, position, clean):
    """Sums over an answer's clean rounds, for each position group: (groups, counts)."""
    table = torch.zeros((len(GROUPS), len(COUNTS)), dtype=torch.float64)
    for number, (_, low, high) in enumerate(GROUPS):
        rows = clean & (position >= low) & (position < high)
        table[number, 0] = float(rows.sum())
        for column, name in enumerate(COUNTS[1:], start=1):
            table[number, column] = float(outcome[name][rows].double().sum())
    return table


def with_interval(tables, top, bottom, draws=2000, seed=0):
    """A ratio of two summed counts over answers, with a 95% interval from resampling whole answers.
    tables: (answers, counts) for one position group. top may be a list of (sign, name)."""
    index = {name: number for number, name in enumerate(COUNTS)}
    pairs = top if isinstance(top, list) else [(1, top)]
    above = sum(sign * tables[:, index[name]] for sign, name in pairs)
    below = tables[:, index[bottom]]
    if not len(tables) or not below.sum():
        return {"value": float("nan"), "interval_95": [float("nan")] * 2}
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(tables), (draws, len(tables)))
    values = np.sort(above[picks].sum(axis=1) / np.maximum(below[picks].sum(axis=1), 1e-9))
    return {"value": float(above.sum() / below.sum()),
            "interval_95": [float(values[int(0.025 * draws)]), float(values[int(0.975 * draws) - 1])]}


def describe(tables, sources):
    """The readable result for one gate at one margin: by position group, and math against prose beyond 2,000."""
    out = {}
    for number, (group, _, _) in enumerate(GROUPS):
        part = tables[:, number]
        if not part[:, 0].sum():
            continue
        out[group] = {
            "rounds": int(part[:, 0].sum()),
            "tokens_per_round_before": float(part[:, 1].sum() / part[:, 0].sum()),
            "more_tokens_one_edit": with_interval(part, "one_edit", "tokens"),
            "more_tokens_all_edits": with_interval(part, "all_edits", "tokens"),
            "more_tokens_at_least": with_interval(part, "at_least", "tokens"),
            "rescues_minus_breaks_per_rejection": with_interval(part, [(1, "rescues"), (-1, "breaks")], "rejections"),
            "rescues_per_rejection": float(part[:, 6].sum() / max(part[:, 2].sum(), 1)),
            "breaks_per_rejection": float(part[:, 7].sum() / max(part[:, 2].sum(), 1)),
            "rounds_changed": float(part[:, 8].sum() / part[:, 0].sum()),
        }
    far = [name for name, _, _ in GROUPS].index("2k+")
    for label, wanted in (("math_2k+", True), ("prose_2k+", False)):
        rows = np.array([(source in MATH) == wanted for source in sources], dtype=bool)
        if rows.any() and tables[rows, far, 0].sum():
            out[label] = {"answers": int(rows.sum()),
                          "more_tokens_one_edit": with_interval(tables[rows, far], "one_edit", "tokens"),
                          "more_tokens_at_least": with_interval(tables[rows, far], "at_least", "tokens")}
    return out


def clean_rounds(record, device):
    starts = record["starts"].to(device).long()
    if record.get("repetition_from") is None:
        return torch.ones_like(starts, dtype=torch.bool)
    return starts < int(record["prompt_length"]) + int(record["repetition_from"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-cells", type=int, default=1_500_000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="use only this many answers (for a quick trial)")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    started = time.time()
    paths = sorted(glob.glob(os.path.join(args.rounds_dir, "*.pt")))
    if args.limit:
        paths = paths[:: max(1, len(paths) // args.limit)][: args.limit]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "files": len(paths)}, indent=1)

    # Pass 1: the cells to fit on, from the training answers.
    parts = {path: split_of_file(path) for path in paths}
    train_paths = [path for path in paths if parts[path] == "train"]
    per_answer = max(1, args.max_cells // max(len(train_paths), 1))
    cells = {base: {"x": [], "lp": [], "target": []} for base in BASES}
    generator = torch.Generator(device=device).manual_seed(0)
    for path in train_paths:
        record = torch.load(path, map_location="cpu")
        clean = clean_rounds(record, device)
        for base in BASES:
            if base == "editor" and "edited" not in record:
                continue
            made = build(record, base, device)
            hit = made["cand"] == made["truth"][..., None]
            use = reachable(made["base"], made["truth"], made["valid"]) & hit.any(-1) & clean[:, None]
            rows = use.nonzero()
            if rows.shape[0] > per_answer:
                rows = rows[torch.randperm(rows.shape[0], device=device, generator=generator)[:per_answer]]
            cells[base]["x"].append(made["x"][rows[:, 0], rows[:, 1]].half())
            cells[base]["lp"].append(made["lp"][rows[:, 0], rows[:, 1]])
            cells[base]["target"].append(hit[rows[:, 0], rows[:, 1]].float().argmax(-1))
            del made
    out = {"answers": {part: sum(value == part for value in parts.values()) for part in ("train", "validation", "test")},
           "inputs": [name for name, _ in feature_names()], "fitted_on_cells": {}, "results": {}}
    gates = {}
    for base in BASES:
        if not cells[base]["x"]:
            continue
        x, lp, target = (torch.cat(cells[base][key]) for key in ("x", "lp", "target"))
        out["fitted_on_cells"][base] = int(x.shape[0])
        out["share_of_cells_where_the_starting_draft_is_right"] = out.get("share_of_cells_where_the_starting_draft_is_right", {})
        out["share_of_cells_where_the_starting_draft_is_right"][base] = float((x[:, :, 2].float().argmax(-1) == target).float().mean())
        for name, groups in SETS.items():
            gates[(base, name)] = fit(x, lp, target, set_mask(groups, device), 3.0 if base == "editor" else 0.0, args.epochs, device)
        del x, lp, target
        cells[base] = None
        write_json(status_path, {"done": False, "fitted": base, "hours": round((time.time() - started) / 3600, 3), **out}, indent=1)

    # Pass 2: score every gate at every margin on the validation and test answers.
    sums = {key: {part: [] for part in ("validation", "test")} for key in gates}
    sources = {base: {part: [] for part in ("validation", "test")} for base in BASES}
    for path in paths:
        part = parts[path]
        if part == "train":
            continue
        record = torch.load(path, map_location="cpu")
        clean = clean_rounds(record, device)
        for base in BASES:
            if (base, "both") not in gates or (base == "editor" and "edited" not in record):
                continue
            sources[base][part].append(record.get("source", "unknown"))
            made = build(record, base, device)
            for name in SETS:
                with torch.no_grad():
                    scores = gates[(base, name)](made["x"], made["lp"])
                tables = [answer_sums(walk(scores, made["cand"], made["base"], made["truth"], made["valid"], margin),
                                      made["position"], clean) for margin in MARGINS]
                sums[(base, name)][part].append(torch.stack(tables))       # (margins, groups, counts)
            del made
    everything = [name for name, _, _ in GROUPS].index("everything")
    for (base, name), by_part in sums.items():
        if not by_part["validation"] or not by_part["test"]:
            continue
        validation = torch.stack(by_part["validation"]).numpy()             # (answers, margins, groups, counts)
        test = torch.stack(by_part["test"]).numpy()
        # The margin is chosen on validation answers: most gained tokens (the lower bound on the editor's draft).
        chosen_by = COUNTS.index("at_least" if base == "editor" else "one_edit")
        best = int(validation[:, :, everything, chosen_by].sum(axis=0).argmax())
        out["results"].setdefault(base, {})[name] = {
            "margin": MARGINS[best],
            "validation_more_tokens": float(validation[:, best, everything, chosen_by].sum() / validation[:, best, everything, 1].sum()),
            "test": describe(test[:, best], sources[base]["test"]),
            "test_more_tokens_one_edit_at_every_margin": {
                str(margin): float(test[:, number, everything, 3].sum() / test[:, number, everything, 1].sum())
                for number, margin in enumerate(MARGINS)},
        }
    out["done"] = True
    out["hours"] = round((time.time() - started) / 3600, 3)
    write_json(status_path, out, indent=1)


def split_of_file(path):
    name = os.path.basename(path)[:-3]            # "shard0of4-0012"
    shard, index = name.rsplit("-", 1)
    return split_of([shard + ".pt", int(index)])


if __name__ == "__main__":
    main()
