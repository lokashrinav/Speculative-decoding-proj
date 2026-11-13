"""Count which token follows which in the target's own training answers, and save
the table for the left-to-right editor (dflash/pair_editor.py). The held-out
answers used for validation are left out."""

import argparse

from dflash.pair_editor import BigramTable
from train_redraft import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--vocabulary", type=int, required=True)
parser.add_argument("--max-len", type=int, default=1024)
parser.add_argument("--val-sequences", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
items = load_data(args.data_dir, args.max_len, args.seed)
training = items[: -args.val_sequences]
vocabulary = max(args.vocabulary, max(int(ids.max()) for ids, _ in training) + 1)
table = BigramTable.count(training, vocabulary)
table.save(args.out)
print(f"{table.keys.numel()} different token pairs from {int(table.total)} tokens")
