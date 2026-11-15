"""Reads the Week 1 probe results and checks them against Gate 1 of the plan.

The thresholds were fixed before any result existed:

  pass   D1 >= 6% of first rejections beyond position 2,000 of the answer,
         with rescues minus breaks >= 4% for a simple rule,
         D1 >= 1.5 x D3 (the previous-token memory), and
         D5 >= +6% tokens on long answers
  stop   D1 < 2% everywhere, or D1 <= D3, or D5 < +3%

Every number is computed on "clean" rounds (endless repetition left out) and comes
with a 95% interval from resampling whole answers. The rule for "rescues minus
breaks" is chosen on one half of the answers and scored on the other half, so the
choice cannot flatter itself.

Usage: python nexus/gate1.py nexus/results/probe-long-*/answers.json
"""

import json
import sys

import numpy as np

from dflash.hindsight import BIN_NAMES, VARIANTS, rule_list

FAR = (2, 3, 4)  # position bins beyond 2,000 tokens into the answer
FAMILIES = VARIANTS + ("previous", "suffix")


def load(paths):
    columns, answers = None, []
    for path in paths:
        with open(path) as handle:
            data = json.load(handle)
        if columns is None:
            columns = data["columns"]
        assert columns == data["columns"], "the files were written by different versions of the probe"
        answers.extend(data["answers"])
    return columns, answers


def counts(answers, bins, width):
    """One row of summed counts per answer, over the given position bins."""
    table = np.zeros((len(answers), width))
    for row, answer in enumerate(answers):
        for b in bins:
            if str(b) in answer["bins"]:
                table[row] += answer["bins"][str(b)]
    return table


class Counts:
    def __init__(self, table, index):
        self.table, self.index = table, index

    def ratio(self, top, bottom, rows=None):
        sums = self.table.sum(axis=0) if rows is None else self.table[rows].sum(axis=0)
        return self._ratio(sums, top, bottom)

    def _ratio(self, sums, top, bottom):
        above = sum(sign * sums[self.index[key]] for sign, key in top) if isinstance(top, list) else sums[self.index[top]]
        below = sums[self.index[bottom]]
        return float(above / below) if below else float("nan")

    def with_interval(self, top, bottom, draws=2000, seed=0):
        """The ratio and its 95% interval from resampling whole answers."""
        rng = np.random.default_rng(seed)
        size = self.table.shape[0]
        values = [self.ratio(top, bottom, rng.integers(0, size, size)) for _ in range(draws)] if size else []
        values = sorted(value for value in values if value == value)
        if not values:
            return {"value": self.ratio(top, bottom), "interval_95": [float("nan")] * 2}
        return {"value": self.ratio(top, bottom),
                "interval_95": [values[int(0.025 * len(values))], values[min(int(0.975 * len(values)), len(values) - 1)]]}

    def best_rule(self, family):
        tags = ["|".join(str(part) for part in rule) for rule in rule_list() if rule[0] == family]
        sums = self.table.sum(axis=0)
        return max(tags, key=lambda tag: sums[self.index[f"{tag}|net_tokens"]])


def report(columns, answers):
    index = {name: number for number, name in enumerate(columns)}
    lengths = sorted(answer["tokens"] for answer in answers)
    ended = sum(answer["ended_by_itself"] for answer in answers)
    out = {
        "answers": len(answers),
        "answers_that_ended_by_themselves": ended,
        "answers_that_fell_into_repetition": sum(answer["repetition_from"] is not None for answer in answers),
        "tokens_median": lengths[len(lengths) // 2] if lengths else 0,
        "answers_reaching_position_2000": sum(any(str(b) in answer["bins"] for b in FAR) for answer in answers),
        "D2_tokens_per_round_by_position": {},
        "by_position": {},
    }
    groups = [(name, (b,)) for b, name in enumerate(BIN_NAMES)] + [("2k+", FAR), ("everything", tuple(range(len(BIN_NAMES))))]
    for name, bins in groups:
        table = counts(answers, bins, len(columns))
        every = Counts(table, index)
        if len(bins) == 1 and table[:, index["rounds"]].sum():
            out["D2_tokens_per_round_by_position"][name] = every.with_interval("tokens", "rounds")
        if not table[:, index["first_rejections"]].sum():
            continue
        row = {"first_rejections": int(table[:, index["first_rejections"]].sum()),
               "target_in_top_16": every.ratio("target_in_top", "first_rejections")}
        for variant in VARIANTS:
            row[f"D1_{variant}"] = every.with_interval(f"{variant}|rescue", "first_rejections")
            row[f"D1_{variant}_where_suffix_memory_fails"] = every.ratio(f"{variant}|rescue_suffix_misses", "first_rejections")
            row[f"D5_{variant}"] = every.with_interval(f"{variant}|oracle_tokens", "tokens")
            row[f"D1_{variant}_over_D3_previous"] = every.with_interval(f"{variant}|rescue", "previous|rescue")
            row[f"D1_{variant}_over_D3_suffix"] = every.with_interval(f"{variant}|rescue", "suffix|rescue")
        for memory in ("previous", "suffix"):
            row[f"D3_{memory}"] = every.with_interval(f"{memory}|rescue", "first_rejections")
            row[f"D5_{memory}"] = every.ratio(f"{memory}|oracle_tokens", "tokens")
        for pair in ("raw_or_suffix", "centered_or_suffix"):
            row[f"D5_{pair}"] = every.ratio(f"{pair}|oracle_tokens", "tokens")
        # Simple rules: chosen on the even-numbered answers, scored on the odd-numbered ones.
        chosen_on, scored_on = Counts(table[0::2], index), Counts(table[1::2], index)
        for family in FAMILIES:
            tag = chosen_on.best_rule(family)
            row[f"rule_{family}"] = {
                "rule": tag,
                "rescues_minus_breaks_per_rejection": scored_on.with_interval(
                    [(1, f"{tag}|rescues"), (-1, f"{tag}|breaks")], "first_rejections"),
                "more_tokens": scored_on.with_interval(f"{tag}|net_tokens", "tokens"),
            }
        out["by_position"][name] = row

    far = out["by_position"].get("2k+")
    verdict = {"enough_long_answers": ended >= 500 and out["tokens_median"] > 4000}
    if far:
        variant = max(VARIANTS, key=lambda v: far[f"D1_{v}"]["value"])
        d1, d3, d5 = far[f"D1_{variant}"]["value"], far["D3_previous"]["value"], far[f"D5_{variant}"]["value"]
        net = far[f"rule_{variant}"]["rescues_minus_breaks_per_rejection"]["value"]
        everywhere = max(row[f"D1_{v}"]["value"] for row in out["by_position"].values() for v in VARIANTS)
        verdict.update({
            "reading_used": variant,
            "D1_at_least_6_percent": d1 >= 0.06,
            "rescues_minus_breaks_at_least_4_percent": net >= 0.04,
            "D1_at_least_1.5_times_D3": d1 >= 1.5 * d3,
            "D5_at_least_6_percent": d5 >= 0.06,
            "stop_D1_under_2_percent_everywhere": everywhere < 0.02,
            "stop_D1_not_above_D3": d1 <= d3,
            "stop_D5_under_3_percent": d5 < 0.03,
            "stricter_check_D1_at_least_1.5_times_suffix_memory": d1 >= 1.5 * far["D3_suffix"]["value"],
        })
        verdict["passes"] = all(verdict[key] for key in (
            "D1_at_least_6_percent", "rescues_minus_breaks_at_least_4_percent", "D1_at_least_1.5_times_D3", "D5_at_least_6_percent"))
        verdict["stop"] = any(verdict[key] for key in (
            "stop_D1_under_2_percent_everywhere", "stop_D1_not_above_D3", "stop_D5_under_3_percent"))
    out["gate_1"] = verdict
    return out


def main():
    columns, answers = load(sys.argv[1:])
    out = report(columns, answers)
    out["by_source"] = {}
    for source in sorted({answer["source"] for answer in answers}):
        part = report(columns, [answer for answer in answers if answer["source"] == source])
        out["by_source"][source] = {key: part[key] for key in (
            "answers", "answers_that_ended_by_themselves", "answers_that_fell_into_repetition", "tokens_median",
            "answers_reaching_position_2000", "D2_tokens_per_round_by_position")}
        for group in ("2k+", "everything"):
            if group in part["by_position"]:
                row = part["by_position"][group]
                out["by_source"][source][group] = {key: row[key] for key in row if key.startswith(("D1_", "D3_", "D5_", "first_"))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
