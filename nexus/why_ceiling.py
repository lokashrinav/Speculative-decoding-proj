"""What the why-probe's rounds say about three practical questions (nexus/why_redraft.py made them).

    python nexus/why_ceiling.py nexus/results/tune32-eval-0 nexus/results/tune32-eval-1

1. How many more tokens per round the swap gives (the first draft with the drafter's second choice
   at the break, no second pass), at the true break and at the four likely breaks.
2. The limit of a second pass that restarts the drafter after the switched token: the same switch,
   followed by what the drafter writes once the target has really read the text up to that token
   (the next round of plain decoding). No predictor of the target's states can do better than this.
3. How much of the swap is left when the target checks only the switched token and a few of the
   tokens after it (a shorter check).

Ranges are 95%, from drawing whole answers again (2,000 draws).
"""
import argparse
import sys

import numpy as np

from nexus.why_report import WIDTH, Rounds, load


def analyse(folders, out):
    table, columns = load(folders)
    rounds = Rounds(table, columns)
    column = rounds.column
    broke = rounds.broke
    breaks = broke < WIDTH - 1
    has_next = column("next_round") >= 0
    per_round = (broke + 1).mean()
    print(f"{len(table)} rounds of {len(rounds.answers)} answers, first draft alone: {per_round:.3f} tokens per round", file=out)

    def line(name, gain, mask=None):
        mask = np.ones(len(table), bool) if mask is None else mask
        value, low, high = rounds.ratio(gain * mask, mask.astype(float))
        base = (broke + 1)[mask].mean()
        print(f"  {name:62s} +{value:.3f} tokens per round [{low:.3f}, {high:.3f}]  = {100 * value / base:+5.1f}% of {base:.2f}", file=out)

    gain = rounds.gain
    switch_right = (column("switch") > broke) & breaks
    after_real = np.where(has_next & breaks, column("next_round") - broke, 0).astype(float)
    covered = column("covered_with_zero") == 1
    print("\nthe swap, and the limit of a restart after the switched token", file=out)
    line("swap at the true break (second choice, old rest)", gain("switch"))
    line("swap at the four likely breaks", gain("four_switch"))
    line("second choice + the rest redrafted from tokens (true break)", gain("early"))
    line("LIMIT: second choice + the target's real states (true break)", after_real * switch_right)
    line("LIMIT: second choice + the target's real states (four breaks)", after_real * switch_right * covered)
    if (column("second_late") >= 0).any():
        reader_right = (column("second_late") > broke) & breaks
        line("the trained reader made to differ (true break)", gain("second_late"))
        line("LIMIT: the reader's token + the target's real states", after_real * reader_right)
    line("LIMIT: the right token always + old rest", gain("right_token_old_rest"))
    line("LIMIT: the right token always + the target's real states", after_real)

    where = column("where")
    print("\nby how much text came before", file=out)
    for name, mask in (("under 2,000 tokens", where < 2000), ("2,000 to 8,000", (where >= 2000) & (where < 8000)), ("beyond 8,000", where >= 8000)):
        if mask.sum() < 500:
            continue
        here = mask & breaks
        print(f" {name}: {int(mask.sum())} rounds; second choice right at {switch_right[here].mean():.3f} of breaks; tokens after "
              f"the right token: old rest {gain('right_token_old_rest')[here].mean() - 1:.2f}, redrafted from tokens "
              f"{gain('right_token_new_rest')[here].mean() - 1:.2f}, with the target's real states {after_real[here & has_next].mean() - 1:.2f}", file=out)
        line("  swap at the four likely breaks", gain("four_switch"), mask)
        line("  LIMIT: second choice + real states (four breaks)", after_real * switch_right * covered, mask)

    print("\nthe swap when the target checks only the switched token and N tokens after it (more tokens per round)", file=out)
    for arm in ("switch", "four_switch"):
        won = gain(arm)
        cells = [f"N={'all' if tail > WIDTH else tail}: {100 * np.minimum(won, 1 + tail).mean() / per_round:+.1f}%" for tail in (0, 1, 2, 3, 4, 99)]
        print(f"  {arm:12s} " + "   ".join(cells), file=out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    out = open(args.out, "w") if args.out else sys.stdout
    analyse(args.folders, out)


if __name__ == "__main__":
    main()
