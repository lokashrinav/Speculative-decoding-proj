"""Print the file of the best editor in a train_editor2.py status file, or nothing
if no editor's edited draft beat the first draft on held-out answers."""

import json
import sys

best = json.load(open(sys.argv[1])).get("best") or {}
if best.get("gain_edited_replaces_first", 0) > 0:
    print(best["path"])
