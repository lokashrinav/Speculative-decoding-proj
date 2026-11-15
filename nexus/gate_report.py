"""Puts the gate scores written by the score jobs together.

The margin of each gate is chosen on the validation answers and the result is given for the
test answers, with 95% intervals from resampling whole answers (see nexus/fit_gate.py).

Usage: python nexus/gate_report.py nexus/results/gate-score-*/gate-scores-*.json
"""

import json
import sys

from nexus import fit_gate


def main():
    rows = []
    for path in sys.argv[1:]:
        with open(path) as handle:
            rows += json.load(handle)["answers"]
    print(json.dumps(fit_gate.report(rows), indent=1))


if __name__ == "__main__":
    main()
