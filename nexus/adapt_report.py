"""Compares the drafter-training settings answer by answer.

    python nexus/adapt_report.py nexus/results/adapt-*/answers.json

Every setting was replayed on the same saved answers, so two settings are compared on the
answers both have. "more" is how many more tokens per round than the frozen drafter of the
same jobs; the brackets are 95% intervals from resampling answers. Files of several job
families can be given together when they used the same drafter: their settings are then
compared on the answers they share. (Never mix files of two drafters: settings with the same
name would be taken for the same thing.)

    --over FIRST SECOND     also compare these two settings (can be given several times)
"""

import json
import sys

import numpy as np

PAIRS = (("hindsight", "immediate"), ("both", "immediate"), ("hindsight", "canvas"), ("both", "canvas"),
         ("canvas+hindsight", "canvas"), ("canvas", "immediate"), ("rejected", "immediate"), ("accepted", "immediate"),
         ("rejected", "accepted"))


def load(paths):
    """The position groups (of the file that has most), and for every answer its table per setting."""
    groups, loaded = [], []
    for path in paths:
        data = json.load(open(path))
        if len(data["groups"]) > len(groups):
            assert data["groups"][: len(groups)] == groups, "the files group positions differently"
            groups = data["groups"]
        loaded.append((path, data))
    answers = {}
    for path, data in loaded:
        family = path.replace("\\", "/").split("/")[-2].rsplit("-", 1)[0]
        for answer in data["answers"]:
            entry = answers.setdefault(tuple(answer["origin"]), {"source": answer["source"], "settings": {}})
            for name, row in answer["settings"].items():
                table = np.zeros((len(groups), 2))               # an older file has fewer groups: the rest stays empty
                given = np.array(row["table"], dtype=float)
                table[: given.shape[0]] = given
                entry["settings"]["plain@" + family if name == "plain" else name] = table
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
    return (ratio(one, two), spread[int(0.025 * draws)], spread[int(0.975 * draws) - 1], len(rows),
            one[:, 1].sum() / one[:, 0].sum(), two[:, 1].sum() / two[:, 0].sum())


def show(result):
    if result is None:
        return "      -       "
    return f"{100 * result[0]:+5.1f}% [{100 * result[1]:+5.1f}, {100 * result[2]:+5.1f}]"


def without_one_option(name):
    """The settings this one becomes when one of its options is taken away (the labels and the rate stay)."""
    body, at, start_from = name.partition("@")
    parts = body.split(":")
    fixed = 1 if parts[0] in ("plain", "frozen") else 2
    out = []
    for number in range(fixed, len(parts)):
        rest = ":".join(parts[:number] + parts[number + 1 :])
        out.append(rest + at + start_from)
    return out


def main(arguments):
    extra, paths = [], []
    while arguments:
        if arguments[0] == "--over":
            extra.append((arguments[1], arguments[2]))
            arguments = arguments[3:]
        else:
            paths.append(arguments[0])
            arguments = arguments[1:]
    groups, answers = load(paths)
    names = sorted({name for entry in answers.values() for name in entry["settings"] if not name.startswith("plain@")})
    family = {name: fam for entry in answers.values() for name, fam in entry.get("family", {}).items()}
    sources = sorted({entry["source"] for entry in answers.values()})
    everything = groups.index("everything")

    def base(name):
        """A setting's own name, or for "plain" the plain drafter of the jobs the other setting ran in."""
        return name

    print(f"{len(answers)} answers. More tokens per round than the frozen drafter:")
    print(f"{'setting':34s} answers  tok/round  " + "  ".join(f"{group:^22s}" for group in groups))
    for name in names:
        results = [compare(answers, name, "plain@" + family[name], number) for number in range(len(groups))]
        total = results[everything]
        if total is None:
            continue
        print(f"{name:34s} {total[3]:5d}   {total[4]:7.2f}   " + "  ".join(f"{show(result):^22s}" for result in results))

    print("\nTokens per round (on the answers the setting shares with its plain drafter; plain in brackets):")
    print(f"{'setting':34s} " + "  ".join(f"{group:^15s}" for group in groups))
    for name in names:
        results = [compare(answers, name, "plain@" + family[name], number) for number in range(len(groups))]
        if results[everything] is None:
            continue
        print(f"{name:34s} " + "  ".join(
            f"{'-':^15s}" if result is None else f"{result[4]:5.2f} ({result[5]:5.2f}) " for result in results))

    print("\nBy kind of answer (all positions):")
    print(f"{'setting':34s} " + "  ".join(f"{source[:20]:^22s}" for source in sources))
    for name in names:
        print(f"{name:34s} " + "  ".join(f"{show(compare(answers, name, 'plain@' + family[name], everything, source)):^22s}"
                                         for source in sources))

    print("\nOne setting over another, on the same answers:")
    print(f"{'first':34s} {'second':30s} " + "  ".join(f"{group:^22s}" for group in groups))
    done = set()

    def line(first, second):
        other = "plain@" + family[first] if second == "plain" else second
        if (first, second) in done or first not in names or (second != "plain" and second not in names):
            return
        done.add((first, second))
        print(f"{first:34s} {second:30s} " + "  ".join(
            f"{show(compare(answers, first, other, number)):^22s}" for number in range(len(groups))))

    for first, second in extra:
        line(first, second)
    for first_labels, second_labels in PAIRS:      # other labels with everything else the same
        for first in names:
            parts = first.split(":")
            if parts[0] == first_labels:
                line(first, ":".join([second_labels] + parts[1:]))
    for first in names:                            # an option over the same setting without it
        for second in without_one_option(first):
            line(first, second)
    for first in names:                            # a start from saved adapters over the same setting from zero
        if "@" in first:
            line(first, first.split("@")[0])


if __name__ == "__main__":
    main(sys.argv[1:])
