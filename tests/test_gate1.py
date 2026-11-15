"""The Week 1 gate is read from per-answer counts; these checks pin down that reading."""

from dflash.hindsight import column_names
from nexus.gate1 import report


def answer(tokens, counts_by_bin, source="s", ended=True, repetition_from=None):
    columns = column_names()
    index = {name: number for number, name in enumerate(columns)}
    bins = {}
    for b, counts in counts_by_bin.items():
        row = [0] * len(columns)
        for key, value in counts.items():
            row[index[key]] = value
        bins[str(b)] = row
    return {"source": source, "tokens": tokens, "ended_by_itself": ended, "repetition_from": repetition_from, "bins": bins}


def test_gate_passes_when_every_threshold_is_met():
    far = {"rounds": 250, "tokens": 1000, "first_rejections": 100, "target_in_top": 90,
           "raw|rescue": 10, "raw|oracle_tokens": 80, "centered|rescue": 7, "centered|oracle_tokens": 50,
           "previous|rescue": 4, "suffix|rescue": 5, "raw|rescue_suffix_misses": 6,
           "raw|0.9|0.9|0.6|rescues": 8, "raw|0.9|0.9|0.6|breaks": 2, "raw|0.9|0.9|0.6|net_tokens": 30,
           "raw|0.8|0.0|1.0|rescues": 10, "raw|0.8|0.0|1.0|breaks": 40, "raw|0.8|0.0|1.0|net_tokens": -90,
           "editor|tokens": 1100, "editor|first_rejections": 80, "after_editor|centered|rescue": 8,
           "after_editor|centered|oracle_tokens": 55}
    early = {"rounds": 100, "tokens": 600, "first_rejections": 50, "raw|rescue": 1, "previous|rescue": 1}
    answers = [answer(5000, {0: early, 2: far}) for _ in range(8)]
    out = report(column_names(), answers)
    row = out["by_position"]["2k+"]
    assert out["answers_reaching_position_2000"] == 8 and row["first_rejections"] == 800
    assert abs(row["D1_raw"]["value"] - 0.10) < 1e-12 and row["D1_raw"]["interval_95"] == [0.1, 0.1]
    assert abs(row["D3_previous"]["value"] - 0.04) < 1e-12 and abs(row["D1_raw_over_D3_previous"]["value"] - 2.5) < 1e-12
    assert abs(row["D5_raw"]["value"] - 0.08) < 1e-12 and abs(row["D1_raw_where_suffix_memory_fails"] - 0.06) < 1e-12
    assert row["rule_raw"]["rule"] == "raw|0.9|0.9|0.6"            # the rule with the most net tokens, not the greediest one
    assert abs(row["rule_raw"]["rescues_minus_breaks_per_rejection"]["value"] - 0.06) < 1e-12
    assert abs(out["D2_tokens_per_round_by_position"]["2k-4k"]["value"] - 4.0) < 1e-12
    assert abs(out["D2_tokens_per_round_by_position"]["0-1k"]["value"] - 6.0) < 1e-12
    edited = row["with_editor"]
    assert abs(edited["more_tokens_than_plain"]["value"] - 0.10) < 1e-12 and edited["first_rejections"] == 640
    assert abs(edited["rescue_rate_centered"]["value"] - 0.10) < 1e-12
    assert abs(edited["more_tokens_if_used_only_when_right_centered"]["value"] - 0.05) < 1e-12
    gate = out["gate_1"]
    assert gate["reading_used"] == "raw" and gate["passes"] and not gate["stop"]
    assert gate["stricter_check_D1_at_least_1.5_times_suffix_memory"]      # 10% against 5%
    assert not gate["enough_long_answers"]                                  # only 8 answers


def test_gate_stops_when_the_memory_is_no_better_than_the_previous_token():
    far = {"rounds": 250, "tokens": 1000, "first_rejections": 100, "raw|rescue": 1, "centered|rescue": 1,
           "raw|oracle_tokens": 5, "centered|oracle_tokens": 5, "previous|rescue": 2}
    out = report(column_names(), [answer(5000, {3: far}) for _ in range(6)])
    gate = out["gate_1"]
    assert gate["stop"] and not gate["passes"]
    assert gate["stop_D1_under_2_percent_everywhere"] and gate["stop_D1_not_above_D3"] and gate["stop_D5_under_3_percent"]


def test_intervals_come_from_resampling_whole_answers():
    good = {"rounds": 10, "tokens": 100, "first_rejections": 10, "raw|rescue": 10}
    bad = {"rounds": 10, "tokens": 100, "first_rejections": 10, "raw|rescue": 0}
    answers = [answer(5000, {2: good}) for _ in range(20)] + [answer(5000, {2: bad}) for _ in range(20)]
    low, high = report(column_names(), answers)["by_position"]["2k+"]["D1_raw"]["interval_95"]
    assert 0.25 < low < 0.45 and 0.55 < high < 0.75     # 20 of 40 answers: about 0.5 plus or minus 0.16
