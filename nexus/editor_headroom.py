"""How much could a left-to-right editor gain at best? Nothing is trained here.

On held-out answers the frozen drafter writes its first draft. Then, per block:

  coverage     at the first wrong token, is the target's token among the first
               pass's top K candidates?
  oracle walk  an editor that always picks the target's token when it is among
               the top K, going left to right until it is missing
  one swap     the same, but only the first wrong token is fixed and the rest of
               the draft is kept as it was
  teacher      what the trained second drafter does at the first wrong token
  bigram walk  a real editor with no training at all: left to right, it re-picks
               among the top K, adding how often each candidate followed the token
               it just chose in the training answers (counted, not learned)

The edited draft replaces the first draft in every case.
"""

import argparse
import json
import sys
from types import SimpleNamespace

import torch

from train_editor import EditorTrainer
from train_redraft import BLOCK, load_data

KS = (2, 4, 8, 16, 64)


def bigram_table(training, vocabulary, device):
    """Counts of (token, next token) in the target's own answers."""
    firsts, seconds = [], []
    for ids, prompt_length in training:
        start = max(int(prompt_length), 1)
        firsts.append(ids[start - 1 : -1])
        seconds.append(ids[start:])
    first, second = torch.cat(firsts).to(device), torch.cat(seconds).to(device)
    keys, counts = torch.unique(first * vocabulary + second, return_counts=True)
    return SimpleNamespace(
        keys=keys, counts=counts.float(), vocabulary=vocabulary, total=float(second.numel()),
        unigram=torch.bincount(second, minlength=vocabulary).float(),
        previous=torch.bincount(first, minlength=vocabulary).float(),
    )


def association(table, previous_token, candidates, smoothing):
    """log p(candidate | previous token) - log p(candidate), from the counts.
    previous_token: (rows,), candidates: (rows, K). 0 when the previous token was never seen."""
    keys = previous_token[:, None] * table.vocabulary + candidates
    index = torch.searchsorted(table.keys, keys).clamp(max=table.keys.numel() - 1)
    pair = torch.where(table.keys[index] == keys, table.counts[index], torch.zeros((), device=keys.device))
    prior = (table.unigram[candidates] + 1.0) / (table.total + table.vocabulary)
    conditional = (pair + smoothing * prior) / (table.previous[previous_token][:, None] + smoothing)
    return conditional.log() - prior.log()


def bigram_walk(top_values, top_tokens, anchor, table, k, weight, smoothing):
    """Left to right: at each position re-pick among the top k, given the token just chosen."""
    previous, chosen = anchor, []
    for slot in range(top_values.shape[1]):
        candidates = top_tokens[:, slot, :k]
        score = top_values[:, slot, :k] + weight * association(table, previous, candidates, smoothing)
        previous = candidates.gather(1, score.argmax(dim=1, keepdim=True))[:, 0]
        chosen.append(previous)
    return torch.stack(chosen, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-sequences", type=int, default=300)
    parser.add_argument("--val-anchors", type=int, default=24)
    parser.add_argument("--weights", default="0,0.1,0.25,0.5,1.0")
    parser.add_argument("--smoothings", default="10,100,1000")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    items = load_data(args.data_dir, args.max_len, args.seed)
    validation, training = items[-args.val_sequences:], items[: -args.val_sequences]
    frozen = EditorTrainer(SimpleNamespace(
        model=args.model, draft=args.draft, teacher=args.teacher,
        rank=8, read_tokens="all", no_states=False, first_slot=1,
    ), device)
    vocabulary = frozen.head.weight.shape[0]
    table = bigram_table(training, vocabulary, device)
    walks = [(k, weight, smoothing) for k in (4, 16)
             for weight in map(float, args.weights.split(",")) for smoothing in map(float, args.smoothings.split(","))
             if weight > 0 or smoothing == float(args.smoothings.split(",")[0])]

    generator = torch.Generator().manual_seed(1234)
    total = dict(rounds=0, first=0, wrong=0, teacher_rounds=0, teacher_right=0, teacher_right_top16=0, teacher=0)
    covered = {k: 0 for k in KS}
    oracle = {k: 0 for k in KS}
    swap = {k: 0 for k in KS}
    ranks = torch.zeros(6, dtype=torch.long)  # rank of the target's token at the first wrong position
    walked = {walk: dict(edited=0, better=0, worse=0) for walk in walks}
    with torch.no_grad():
        for ids, prompt_length in validation:
            ids = ids.to(device)
            candidates = torch.arange(prompt_length, ids.shape[0] - BLOCK, device=device)
            if candidates.numel() == 0:
                continue
            features = frozen.features(ids)
            anchors = candidates[torch.randperm(candidates.shape[0], generator=generator)[: args.val_anchors].to(device)]
            tokens, _ = frozen.blocks(ids, anchors)
            visible = tokens.clone()
            visible[:, 1:] = frozen.mask_id
            _, logits = frozen.drafter_pass(frozen.drafter, features, anchors, visible)
            logits = logits.float()
            truth, draft = tokens[:, 1:], logits.argmax(dim=-1)
            rows = truth.shape[0]

            def accepted(proposal):
                return (proposal == truth).long().cumprod(dim=1).sum(dim=1)

            first = accepted(draft)
            rank = (logits > logits.gather(-1, truth[..., None])).sum(dim=-1)  # 0 means the draft's token is right
            has_error = first < truth.shape[1]
            slot = first.clamp(max=truth.shape[1] - 1)
            rank_at_error = rank.gather(1, slot[:, None])[:, 0]
            # Draft tokens right in a row immediately after the first wrong position.
            later_right = torch.zeros(rows, dtype=torch.long, device=device)
            right = (draft == truth).long()
            for row in torch.nonzero(has_error)[:, 0].tolist():
                later_right[row] = right[row, int(first[row]) + 1 :].cumprod(dim=0).sum()
            total["rounds"] += rows
            total["first"] += int((first + 1).sum())
            total["wrong"] += int(has_error.sum())
            for bucket, (low, high) in enumerate(((1, 2), (2, 4), (4, 8), (8, 16), (16, 64), (64, 10**9))):
                ranks[bucket] += int((has_error & (rank_at_error >= low) & (rank_at_error < high)).sum())
            for k in KS:
                covered[k] += int((has_error & (rank_at_error < k)).sum())
                oracle[k] += int(((rank < k).long().cumprod(dim=1).sum(dim=1) + 1).sum())
                fixed = has_error & (rank_at_error < k)
                swap[k] += int((torch.where(fixed, first + 1 + later_right, first) + 1).sum())
            if frozen.teacher is not None:
                shown = torch.cat((tokens[:, :1], draft), dim=1)
                _, teacher_logits = frozen.drafter_pass(frozen.teacher, features, anchors, shown)
                taught = torch.cat((draft[:, :1], teacher_logits[:, 1:].argmax(dim=-1)), dim=1)
                total["teacher"] += int((accepted(taught) + 1).sum())
                judged = has_error & (first >= 1)  # the teacher never changes the first draft token
                right_here = taught.gather(1, slot[:, None])[:, 0] == truth.gather(1, slot[:, None])[:, 0]
                total["teacher_rounds"] += int(judged.sum())
                total["teacher_right"] += int((judged & right_here).sum())
                total["teacher_right_top16"] += int((judged & right_here & (rank_at_error < 16)).sum())
            top = logits.topk(16, dim=-1)
            for walk in walks:
                k, weight, smoothing = walk
                edited = accepted(bigram_walk(top.values, top.indices, tokens[:, 0], table, k, weight, smoothing))
                walked[walk]["edited"] += int((edited + 1).sum())
                walked[walk]["better"] += int((edited > first).sum())
                walked[walk]["worse"] += int((edited < first).sum())

    rounds, base, wrong = max(total["rounds"], 1), max(total["first"], 1), max(total["wrong"], 1)
    result = {
        "rounds": rounds,
        "tokens_per_round_first_draft": total["first"] / rounds,
        "rounds_with_a_wrong_token": total["wrong"] / rounds,
        "target_token_in_top_k_at_first_wrong_token": {str(k): covered[k] / wrong for k in KS},
        "rank_of_target_token_at_first_wrong_token": dict(zip(
            ("2nd", "3rd-4th", "5th-8th", "9th-16th", "17th-64th", "beyond 64th"), (ranks.float() / wrong).tolist())),
        "gain_oracle_walk": {str(k): oracle[k] / base - 1 for k in KS},
        "gain_oracle_one_swap": {str(k): swap[k] / base - 1 for k in KS},
        "bigram_pairs_counted": int(table.keys.numel()),
        "bigram_walk": [
            {"top_k": k, "weight": weight, "smoothing": smoothing,
             "gain_edited_replaces_first": walked[(k, weight, smoothing)]["edited"] / base - 1,
             "edited_better": walked[(k, weight, smoothing)]["better"] / rounds,
             "edited_worse": walked[(k, weight, smoothing)]["worse"] / rounds}
            for k, weight, smoothing in walks
        ],
    }
    if frozen.teacher is not None:
        judged = max(total["teacher_rounds"], 1)
        result["teacher"] = {
            "gain_teacher_replaces_first": total["teacher"] / base - 1,
            "right_at_first_wrong_token": total["teacher_right"] / judged,
            "right_and_in_top_16": total["teacher_right_top16"] / judged,
        }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result), file=sys.stderr)


if __name__ == "__main__":
    main()
