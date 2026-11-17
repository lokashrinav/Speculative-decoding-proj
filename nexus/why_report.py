"""Reads the tables of nexus/why_redraft.py and says what each kind of second draft adds.

    python nexus/why_report.py nexus/results/tune32-eval-0 nexus/results/tune32-eval-1

Everything is counted on the rounds in which the first draft broke (the target rejected one of its
tokens). For each kind of second draft: how often it got past the break, and how many tokens it
added per such round (where a kind cannot be tried, it adds nothing). The ranges are 95% ranges
from drawing whole answers again and again (2,000 times), so that the rounds of one answer are
not counted as independent. Differences between two kinds are taken round by round.
"""

import argparse
import glob
import json
import os

import numpy

WIDTH = 16
PAIRS = (
    ("again", "random", "reading its own draft, over being shown random tokens"),
    ("again", "elsewhere", "reading its own draft, over being shown other text"),
    ("random", None, "being shown random tokens (guessing again, nothing to read)"),
    ("early", "switch", "the second pass, once the second choice is put at the break"),
    ("late", "switch", "a pass that picks its own other token, over the first pass's second choice"),
    ("right_token_new_rest", "right_token_old_rest", "rewriting the rest, once the RIGHT token is at the break"),
    ("next_round", "right_token_new_rest", "the target's states, over the same tokens shown in the block"),
    ("second", "second_random", "the trained reader reading the draft, over random tokens"),
    ("second", "second_blind", "the trained reader reading the draft, over being shown nothing"),
    ("second_blind", "control", "the reader's weights shown nothing, over the add-on trained blind"),
    ("second", "again", "the trained reader, over the unchanged drafter (both shown the draft)"),
    ("second_late", "second", "the trained reader made to differ at the break"),
    ("four_early", "four_switch", "the second pass at the four likely breaks, once switched"),
    ("four", "four_random", "four redrafts reading the draft, over random tokens"),
)


def load(folders):
    tables, names, columns = [], [], None
    for number, folder in enumerate(folders):
        for path in sorted(glob.glob(os.path.join(folder, "rounds-*.npz"))):
            saved = numpy.load(path)
            columns = [str(name) for name in saved["columns"]]
            rounds = saved["rounds"].astype(numpy.int64)
            rounds[:, columns.index("answer")] += 100000 * number          # an answer is one answer of one job
            tables.append(rounds)
        names.append(folder)
    table = numpy.concatenate(tables)
    return table, columns


class Rounds:
    def __init__(self, table, columns, draws=2000, seed=0):
        self.table, self.columns = table, columns
        self.broke = table[:, columns.index("broke_after")]
        self.answers, self.cluster = numpy.unique(table[:, columns.index("answer")], return_inverse=True)
        generator = numpy.random.default_rng(seed)
        self.picks = generator.integers(0, len(self.answers), size=(draws, len(self.answers)))

    def column(self, name):
        return self.table[:, self.columns.index(name)]

    def gain(self, name):
        """Tokens the kind adds in each round (0 where it cannot be tried or does not get further)."""
        if name is None:
            return numpy.zeros(len(self.table))
        value = self.column(name)
        return numpy.where(value >= 0, numpy.maximum(value - self.broke, 0), 0).astype(float)

    def ratio(self, top, bottom):
        """sum(top) / sum(bottom) with its 95% range over answers drawn again."""
        tops = numpy.bincount(self.cluster, weights=top, minlength=len(self.answers))
        bottoms = numpy.bincount(self.cluster, weights=bottom, minlength=len(self.answers))
        draws = tops[self.picks].sum(axis=1) / numpy.maximum(bottoms[self.picks].sum(axis=1), 1e-9)
        low, high = numpy.percentile(draws, [2.5, 97.5])
        return tops.sum() / max(bottoms.sum(), 1e-9), low, high


def show(value, low, high, scale=1.0, digits=3):
    return f"{scale * value:.{digits}f} [{scale * low:.{digits}f}, {scale * high:.{digits}f}]"


def report(table, columns, where=None, title="all rounds in which the first draft broke"):
    rounds = Rounds(table, columns)
    lines = []
    broke = rounds.broke < WIDTH - 1
    chosen = broke if where is None else broke & where(rounds)
    ones = chosen.astype(float)
    per_round = (rounds.broke + 1).mean()
    lines.append(f"== {title}: {int(chosen.sum())} rounds of {len(rounds.answers)} answers "
                 f"(all rounds: {len(table)}, first draft alone: {per_round:.2f} tokens per round, "
                 f"breaks in {broke.mean():.3f} of rounds)")
    lines.append(f"{'kind of second draft':24s} {'tried in':>8s}  {'got past the break (of all breaks)':>36s}  "
                 f"{'tokens added per break':>30s}  {'per win':>7s}")
    arms = [name for name in columns[columns.index("again"):] if (rounds.column(name)[chosen] >= 0).any()]
    for name in arms:
        gain = rounds.gain(name) * ones
        tried = (rounds.column(name) >= 0) & chosen
        past = rounds.ratio((gain > 0).astype(float), ones)
        added = rounds.ratio(gain, ones)
        lines.append(f"{name:24s} {tried.sum() / max(chosen.sum(), 1):8.3f}  {show(*past):>36s}  {show(*added):>30s}  "
                     f"{gain.sum() / max((gain > 0).sum(), 1):7.2f}")
    lines.append("")
    lines.append("differences, round by round (tokens added per break; a range that leaves out 0 is a real difference)")
    for first, second, meaning in PAIRS:
        if first not in arms or (second is not None and second not in arms):
            continue
        both = chosen & (rounds.column(first) >= 0) & ((rounds.column(second) >= 0) if second else True)
        difference = (rounds.gain(first) - rounds.gain(second)) * both
        value = rounds.ratio(difference, both.astype(float))
        base = rounds.ratio(rounds.gain(first) * both, both.astype(float))[0]
        lines.append(f"  {first} - {second or 'nothing'}: {show(*value)}  of {base:.3f}  ({both.sum()} rounds)  <- {meaning}")
    return lines


def ranks(table, columns):
    rounds = Rounds(table, columns)
    broke = rounds.broke < WIDTH - 1
    lines = ["", "where the right token stood in the drafter's ranking at the break (share of breaks)"]
    edges = ((1, 1, "1st"), (2, 2, "2nd"), (3, 4, "3rd-4th"), (5, 16, "5th-16th"), (17, 10**9, "lower"))
    for name in [name for name in columns if name.startswith("rank_")]:
        value = rounds.column(name)
        seen = broke & (value >= 1)
        if not seen.any():
            continue
        line = f"  {name:20s} ({int(seen.sum()):6d} rounds)"
        for low, high, label in edges:
            line += f"  {label} {((value >= low) & (value <= high) & seen).sum() / seen.sum():.3f}"
        lines.append(line)
    # Which breaks does each kind fix: by where the right token stood after the first pass.
    first = rounds.column("rank_first")
    lines.append("")
    lines.append("share of breaks got past, by where the right token stood after the FIRST pass")
    arms = [name for name in ("again", "random", "switch", "early", "late", "second", "second_blind", "control", "edit")
            if name in columns and (rounds.column(name)[broke] >= 0).any()]
    lines.append("  " + f"{'right token was':16s} {'share':>6s} " + " ".join(f"{name:>13s}" for name in arms))
    for low, high, label in edges[1:]:
        chosen = broke & (first >= low) & (first <= high)
        line = f"  {label:16s} {chosen.sum() / max(broke.sum(), 1):6.3f} "
        for name in arms:
            tried = chosen & (rounds.column(name) >= 0)
            line += f" {(rounds.gain(name)[tried] > 0).mean() if tried.any() else float('nan'):13.3f}"
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--details", action="store_true", help="also by break position, text length, error and confidence")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    table, columns = load(args.folders)
    lines = report(table, columns) + ranks(table, columns)
    if args.details:
        cuts = [
            ("the first token is the one rejected", lambda r: r.broke == 0),
            ("the break is after 1 accepted token", lambda r: r.broke == 1),
            ("the break is after 2-4 accepted tokens", lambda r: (r.broke >= 2) & (r.broke <= 4)),
            ("the break is after 5-14 accepted tokens", lambda r: r.broke >= 5),
            ("text shorter than 2,000 tokens", lambda r: r.column("where") < 2000),
            ("text of 2,000 to 8,000 tokens", lambda r: (r.column("where") >= 2000) & (r.column("where") < 8000)),
            ("text beyond 8,000 tokens", lambda r: r.column("where") >= 8000),
            ("the wrong token repeats the one before it", lambda r: r.column("error") == 1),
            ("the wrong token is the one that comes next (one skipped)", lambda r: r.column("error") == 2),
            ("other errors", lambda r: r.column("error") == 0),
            ("drafter unsure of its wrong token (below 0.5)", lambda r: r.column("confidence") < 500),
            ("drafter sure of its wrong token (0.5 and up)", lambda r: r.column("confidence") >= 500),
            ("target decisive (0.9 and up)", lambda r: r.column("target_confidence") >= 900),
            ("target undecided (below 0.5)", lambda r: (r.column("target_confidence") >= 0) & (r.column("target_confidence") < 500)),
        ]
        for title, where in cuts:
            lines += [""] + report(table, columns, where, title)
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    statuses = [os.path.join(folder, "status.json") for folder in args.folders]
    for path in statuses:
        if os.path.exists(path):
            with open(path) as handle:
                status = json.load(handle)
            print(f"{path}: {status.get('answers')} of {status.get('answers_in_this_job')} answers, done {status.get('done')}")


if __name__ == "__main__":
    main()
