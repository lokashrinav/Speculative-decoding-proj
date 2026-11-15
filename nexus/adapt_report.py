"""Compares the drafter-training settings answer by answer.

    python nexus/adapt_report.py nexus/results/adapt-*/answers.json

Every setting was replayed on the same saved answers, so two settings are compared on the
answers both have. "more" is how many more tokens per round than the frozen drafter of the
same jobs; the brackets are 95% intervals from resampling answers.
"""

import json
import sys

import numpy as np

PAIRS = (("hindsight", "immediate"), ("both", "immediate"), ("hindsight", "canvas"), ("both", "canvas"),
         ("canvas+hindsight", "canvas"), ("canvas", "immediate"))


def load(paths):
    groups, answers = None, {}
    for path in paths:
        data = json.load(open(path))
        groups = data["groups"]
        family = path.replace("\\", "/").split("/")[-2].rsplit("-", 1)[0]
        for answer in data["answers"]:
            entry = answers.setdefault(tuple(answer["origin"]), {"source": answer["source"], "settings": {}})
            for name, row in answer["settings"].items():
                entry["settings"]["plain@" + family if name == "plain" else name] = np.array(row["table"], dtype=float)
                if name != "plain":
                    entry.setdefault("family", {})[name] = family
    return groups, answers


def ratio(first, second, picks=None):
    """Tokens per round of the first over the second, minus one. Arrays are (answers, 2): rounds, tokens."""
    if picks is not None:
        first, second = first[picks].sum(axis=1), second[picks].sum(axis=1)
        return (first[:, 1] / np.maximum(first[:, 0], 1)) / (second[:, 1] / np.maximum(second[:, 0], 1)) - 1
    first, second = first.sum(axis=0), second.sum(axis=0)
    return (first[1] / max(first[0], 1)) / (second[1] / max(second[0], 1)) - 1


def compare(answers, first, second, group, source=None, draws=2000):
    rows = [(entry["settings"][first][group], entry["settings"][second][group]) for entry in answers.values()
            if first in entry["settings"] and second in entry["settings"] and (source is None or entry["source"] == source)]
    rows = [row for row in rows if row[0][0] > 0 and row[1][0] > 0]
    if len(rows) < 2:
        return None
    one, two = np.array([row[0] for row in rows]), np.array([row[1] for row in rows])
    picks = np.random.default_rng(0).integers(0, len(rows), (draws, len(rows)))
    spread = np.sort(ratio(one, two, picks))
    return ratio(one, two), spread[int(0.025 * draws)], spread[int(0.975 * draws) - 1], len(rows), one[:, 1].sum() / one[:, 0].sum()


def show(result):
    if result is None:
        return "      -       "
    return f"{100 * result[0]:+5.1f}% [{100 * result[1]:+5.1f}, {100 * result[2]:+5.1f}]"


def main(paths):
    groups, answers = load(paths)
    names = sorted({name for entry in answers.values() for name in entry["settings"] if not name.startswith("plain@")})
    family = {name: fam for entry in answers.values() for name, fam in entry.get("family", {}).items()}
    sources = sorted({entry["source"] for entry in answers.values()})
    everything = groups.index("everything")
    print(f"{len(answers)} answers. More tokens per round than the frozen drafter:")
    print(f"{'setting':30s} answers  tok/round  " + "  ".join(f"{group:^22s}" for group in groups))
    for name in names:
        results = [compare(answers, name, "plain@" + family[name], number) for number in range(len(groups))]
        total = results[everything]
        if total is None:
            continue
        print(f"{name:30s} {total[3]:5d}   {total[4]:7.2f}   " + "  ".join(f"{show(result):^22s}" for result in results))
    print("\nBy kind of answer (all positions):")
    print(f"{'setting':30s} " + "  ".join(f"{source[:20]:^22s}" for source in sources))
    for name in names:
        print(f"{name:30s} " + "  ".join(f"{show(compare(answers, name, 'plain@' + family[name], everything, source)):^22s}"
                                         for source in sources))
    print("\nOne setting over another, on the same answers (same learning rate and same kind of comparison):")
    print(f"{'first':28s} {'second':28s} " + "  ".join(f"{group:^22s}" for group in groups))
    for first_labels, second_labels in PAIRS:
        for first in names:
            parts = first.split(":")
            if parts[0] != first_labels:
                continue
            second = ":".join([second_labels] + parts[1:])
            if second in names:
                print(f"{first:28s} {second:28s} " + "  ".join(
                    f"{show(compare(answers, first, second, number)):^22s}" for number in range(len(groups))))


if __name__ == "__main__":
    main(sys.argv[1:])
