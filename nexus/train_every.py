"""Train the small second drafter the way the published second passes are trained: at every place.

    python nexus/train_every.py --draft opd --data-dir ~/specdec-data/long1 --more-data ~/specdec-data/think1 \
        --results-dir nexus/results/NAME --save opd-e1

The second drafter (dflash/restart_layers.py) is a copy of the drafter's last layers that goes on from
the first run and reads the tokens of a branch. nexus/train_second.py trained it only where the first
draft broke, on the right token, on a few hundred answers; it then added nothing to a tree that
already uses counts (docs/DIRECTION.md). The second passes that work in print are trained otherwise
(docs/SECOND_DRAFTER_RESEARCH.md), and this script follows them:

  - EVERY place, with the TRUE tokens before it (D2SD, arXiv 2606.04446; D-Loop, 2610.06011; Weaver,
    2607.06763). A place of a tree only matters when every token before it was accepted, so the tokens
    before it are the answer's own. For an anchor (a position of a saved answer) the first run drafts
    its block; for a depth d the second drafter is given the block with the answer's tokens written in
    at the places 1 .. d, and is asked for the places d + 1 .. d + reach.
  - MANY anchors per answer (--anchors, 256), a few depths each (--depths, 4, shallow ones most often):
    all rows of a group of anchors run in one pass, which reads the context once.
  - CANDIDATES only: the second drafter does not give probabilities over the vocabulary, it corrects
    the first run's probabilities of its --candidates (32) most probable tokens at the place, and
    "another token" (dflash/restart_layers.py: CandidateScores).
  - ON TOP OF THE COUNTS (dflash/follow.py, as DForkTree uses them, arXiv 2611.12356): with
    --counts, the counts' lift is in the scores while it trains, so it learns what the counts do not
    know. The counts a training answer is scored with never include that answer: the answers are
    split in two halves and each half is scored with the counts of the other.
  - the TARGET'S PROBABILITIES next to its token (--soft, 0.5 of the loss): they come from the pass
    that gives the drafter its context. On the target's own greedy text this is worth little for the
    first choice (arXiv 2610.29013) and helps the ranking of the others.

Counted while training and, at the end, on validation answers at every depth (nothing learned), for
the place right after the true tokens:

    first              the first run alone (what DDTree ranks by)
    first_counts       the first run with the counts (DForkTree's rule)
    second             the second drafter without counts
    second_counts      the second drafter with counts: what a tree would use
    covered            how often the right token is among the candidates at all (the most any arm can reach)

each as the share of places where its first choice is right, for all places, for those on the first
run's own path ("on") and for those after the path has left the first run's first choices ("off":
where a tree's branches are), with the mean log-probability of the right token. "walk" is how many
tokens in a row are right from the block's first place when every place knows the true tokens before
it (the first place is the first run's for all).

Saved: warm-<name>.pt in --adapter-dir (the low-rank updates and the scores' few numbers).
"""
import argparse
import math
import os
import random
import time

import torch

from dflash.benchmark import load_transformers_models
from dflash.follow import FollowCounts
from dflash.model import _draft_value, _output_head, _raw_input_embeddings
from dflash.replay import target_features
from dflash.restart_layers import RestartLayers, candidate_scores, load_restart_layers, save_restart_layers
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers
from nexus.train_second import more_answers

ARMS = ("first", "first_counts", "second", "second_counts")


def _sees(context: int, begins: torch.Tensor, spans: torch.Tensor, width: int) -> torch.Tensor:
    """Who sees whom when several blocks run as one sequence after a shared context: block n sees the first
    begins[n] entries of the context and its own first spans[n] entries. [blocks * width, context + blocks * width]."""
    device, blocks = begins.device, begins.shape[0]
    before = torch.arange(context, device=device)[None, :] < begins[:, None]
    own = torch.eye(blocks, dtype=torch.bool, device=device)[:, :, None] & (torch.arange(width, device=device)[None, None, :] < spans[:, None, None])
    rows = torch.cat((before, own.reshape(blocks, blocks * width)), dim=1)
    return rows[:, None, :].expand(blocks, width, -1).reshape(blocks * width, -1)


def _positions(low: int, high: int, starts: torch.Tensor, width: int) -> torch.Tensor:
    return torch.cat((torch.arange(low, high, device=starts.device), (starts[:, None] + torch.arange(width, device=starts.device)[None, :]).flatten()))[None]


def first_runs(drafter, target, features, ids, starts: torch.Tensor, low: int, keep: int, width: int = 16):
    """The drafter's first runs of the rounds that start at `starts` ([anchors], sorted or not), in one pass that
    reads the target's states of ids[low : max(starts)] once; anchor n sees those before its own start. Returns
    its final states and its states after layer `keep`: [anchors, width, size] each."""
    high, anchors = int(starts.max()), starts.shape[0]
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    block = torch.full((anchors, width), drafter.mask_token_id, dtype=ids.dtype, device=ids.device)
    block[:, 0] = ids[0, starts]
    noise = _raw_input_embeddings(target, block, scale).reshape(1, anchors * width, -1)
    sees = _sees(high - low, starts - low, torch.full_like(starts, width), width)
    kept = {}
    hook = drafter.layers[keep].register_forward_hook(lambda layer, came, output: kept.update(states=output))
    try:
        final = drafter(target_hidden=features[:, low:high], noise_embedding=noise, position_ids=_positions(low, high, starts, width),
                        attention_mask=sees[None, None])[:, -anchors * width :, :]
    finally:
        hook.remove()
    states = kept["states"][0] if isinstance(kept["states"], tuple) else kept["states"]
    return final.reshape(anchors, width, -1), states[:, -anchors * width :, :].reshape(anchors, width, -1)


def second_rows(drafter, second, target, features, ids, starts: torch.Tensor, depths: torch.Tensor, kept: torch.Tensor,
                low: int, reach: int, width: int = 16):
    """The second drafter's states for rows (anchor start, depth): the round's block that starts at starts[r],
    going on from the first run's states `kept` ([rows, width, size], after the last layer left out), with the
    answer's tokens written in at the places 1 .. depths[r]. A row sees the context before its start and its own
    places up to depths[r] + reach, as a node's block does in a tree (dflash/redraft.py). One pass for all rows.
    Returns the states at the places depths[r] + 1 .. depths[r] + reach ([rows, reach, size]; places past the
    block's end: its last place) and which of them lie inside the block ([rows, reach])."""
    device, rows, high = ids.device, starts.shape[0], int(starts.max())
    scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
    slots = torch.arange(width, device=device)[None, :]
    known = (slots >= 1) & (slots <= depths[:, None])
    given = torch.where(known, ids[0][(starts[:, None] + slots).clamp(max=ids.shape[1] - 1)], torch.full_like(slots, drafter.mask_token_id))
    embedded = _raw_input_embeddings(target, given, scale)
    mask_in = _raw_input_embeddings(target, torch.full((1, 1), drafter.mask_token_id, dtype=ids.dtype, device=device), scale)[0, 0]
    entering = torch.where(known[..., None], second._written(kept, embedded, mask_in), kept)
    spans = (depths + 1 + reach).clamp(max=width)
    sees = _sees(high - low, starts - low, spans, width)
    # (a copy whose own weights are trained keeps them in 32-bit numbers next to a 16-bit drafter)
    mixed = (second.full or second.fresh) and device.type == "cuda" and drafter.fc.weight.dtype != torch.float32
    with torch.autocast("cuda", dtype=drafter.fc.weight.dtype, enabled=mixed):
        hidden = drafter(target_hidden=features[:, low:high], noise_embedding=entering.reshape(1, rows * width, -1),
                         position_ids=_positions(low, high, starts, width), attention_mask=sees[None, None],
                         layers=list(second.layers))[:, -rows * width :, :].reshape(rows, width, -1)
    places = depths[:, None] + 1 + torch.arange(reach, device=device)[None, :]
    return hidden[torch.arange(rows, device=device)[:, None], places.clamp(max=width - 1)], places <= width - 1


class TeachEvery:
    """One answer at a time: groups of anchors, the first runs, the rows, the loss; and what is counted."""

    def __init__(self, second, optimizer, drafter, target, candidates=32, reach=2, group=32, depths=4, shallow=0.85,
                 soft=0.5, window=2048, width=16, seed=0, depth_decay=7.0, warm=100):
        self.second, self.optimizer, self.drafter, self.target = second, optimizer, drafter, target
        self.candidates, self.reach, self.group, self.depths, self.shallow = candidates, reach, group, depths, shallow
        self.soft, self.window, self.width, self.depth_decay, self.warm = soft, window, width, depth_decay, warm
        self.generator = torch.Generator().manual_seed(seed)
        self.rates = [group["lr"] for group in optimizer.param_groups] if optimizer is not None else []
        self.updates, self.rows_seen = 0, 0
        self.reset()

    def reset(self):
        self.sums = {}

    def _add(self, name, value, count):
        had = self.sums.get(name, (0.0, 0))
        self.sums[name] = (had[0] + float(value), had[1] + int(count))

    def progress(self):
        """The means of what was counted since the last call."""
        point = {name: round(total / count, 4) for name, (total, count) in sorted(self.sums.items()) if count}
        self.reset()
        return point

    def anchors_of(self, ids, prompt_length, anchors, every=None):
        """Starts of rounds inside the answer: `anchors` random ones, or one every `every` tokens."""
        last = ids.shape[1] - self.width - 1                          # the block and its last token's label lie inside
        if last <= prompt_length:
            return torch.zeros(0, dtype=torch.long)
        if every:
            return torch.arange(prompt_length, last, every)
        count = min(anchors, last - prompt_length)
        return prompt_length + torch.randperm(last - prompt_length, generator=self.generator)[:count].sort().values

    def depths_of(self, anchors: int, all_depths: bool):
        """For every anchor the depths it is asked at: all of 1 .. width - 2, or `depths` of them, shallow ones
        most often (depth d with a weight of shallow ** (d - 1)), without one twice. [anchors, depths]."""
        most = self.width - 2
        if all_depths:
            return torch.arange(1, most + 1)[None, :].expand(anchors, -1)
        weights = torch.tensor([self.shallow ** depth for depth in range(most)]).expand(anchors, -1)
        return torch.multinomial(weights, min(self.depths, most), replacement=False, generator=self.generator) + 1

    def learn_from(self, ids, features, said, prompt_length, anchors, follow=None, learn=True, every=None):
        """ids [1, length]; features: the target's states of the answer; said: (tokens, log-probabilities), the
        target's own most likely next tokens after every position ([length, top] each); follow: the counts this
        answer is scored with (None: none). learn=False: nothing is learned, every depth is asked."""
        starts = self.anchors_of(ids, prompt_length, anchors, every).to(ids.device)
        size = self.group if learn else max(self.group // 4, 1)
        for begin in range(0, starts.shape[0], size):
            self.step(ids, features, said, starts[begin : begin + size], follow, learn)

    def step(self, ids, features, said, starts, follow=None, learn=True):
        drafter, target, second, width, device = self.drafter, self.target, self.second, self.width, ids.device
        head, top, reach = _output_head(target), self.candidates, self.reach
        low = max(int(starts.max()) - self.window, 0)
        anchors = starts.shape[0]
        with torch.no_grad():
            final, kept = first_runs(drafter, target, features, ids, starts, low, second.first - 1, width)
            scores = drafter.compute_logits(final[:, 1:], head).float()                   # [anchors, width - 1, vocabulary]
            best = scores.log_softmax(dim=-1).topk(top, dim=-1)                           # the candidates of every place
            first_scores = scores.gather(-1, best.indices)
            del scores
            answer = ids[0][starts[:, None] + torch.arange(1, width, device=device)[None, :]]      # the tokens of the places 1 .. width - 1
            wrong = (best.indices[..., 0] != answer).long().cumsum(dim=1)                 # how often the first run's choice was wrong up to a place
            depths = self.depths_of(anchors, all_depths=not learn).to(device)             # [anchors, per]
            per = depths.shape[1]
            owner = torch.arange(anchors, device=device)[:, None].expand(-1, per).flatten()
            depths = depths.flatten()
            row_starts = starts[owner]
        with torch.set_grad_enabled(learn):
            states, inside = second_rows(drafter, second, target, features, ids, row_starts, depths, kept[owner], low, reach, width)
            places = (depths[:, None] + 1 + torch.arange(reach, device=device)[None, :]).clamp(max=width - 1)    # [rows, reach]
            at = (owner[:, None], places - 1)
            candidates, first_log, firsts = best.indices[at], best.values[at], first_scores[at]                # [rows, reach, top]
            if second.scores.out is not None:         # a layer from new weights: its correction through its own map, from nothing
                seconds = candidate_scores(drafter, head, second.scores.out(states.to(second.scores.out.weight.dtype)), candidates)
                firsts = torch.zeros_like(firsts)
            else:
                seconds = candidate_scores(drafter, head, states, candidates)
            off = wrong[owner[:, None], places - 2] > 0                                    # a wrong first choice before the place
            label = answer[at]
            position = row_starts[:, None] + places                                        # the answer's position of each place
            lift = None
            if follow is not None:
                lift = follow.lift(ids[0][position - 2][..., None], ids[0][position - 1][..., None], candidates)
            log_with = second.scores(first_log, firsts, seconds, states, lift, off)
            hit = candidates == label[..., None]
            index = torch.where(hit.any(dim=-1), hit.float().argmax(dim=-1), torch.full_like(label, top))
            # What the target itself found likely at the place, over the same outcomes.
            told, told_log = said[0][position - 1], said[1][position - 1]                  # [rows, reach, said]
            match = candidates[..., :, None] == told[..., None, :]
            target_share = (match * told_log.exp()[..., None, :]).sum(dim=-1)
            target_share = torch.cat((target_share, (1.0 - target_share.sum(dim=-1, keepdim=True)).clamp_min(0.0)), dim=-1)
            hard = -log_with.gather(-1, index[..., None])[..., 0]
            soft = -(target_share * log_with).sum(dim=-1)
            weight = inside.float() * torch.exp(-(depths[:, None].float() - 1.0) / self.depth_decay) \
                * (0.5 ** torch.arange(reach, device=device, dtype=torch.float))[None, :]
            loss = (((1.0 - self.soft) * hard + self.soft * soft) * weight).sum() / weight.sum().clamp_min(1e-6)
        if learn:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(second.trained(), 1.0)
            self.updates += 1
            for group, rate in zip(self.optimizer.param_groups, self.rates):
                group["lr"] = rate * min(1.0, self.updates / max(self.warm, 1))
            self.optimizer.step()
            self.rows_seen += int(depths.shape[0])
        with torch.no_grad():
            self._count(loss, first_log, firsts, seconds, states, lift, off, index, inside, log_with, starts, depths, owner, per, learn,
                        best.indices[:, 0, 0] == answer[:, 0])

    def _count(self, loss, first_log, firsts, seconds, states, lift, off, index, inside, log_with, starts, depths, owner, per, learn, first_place):
        """The arms at the place right after the true tokens (the first of the `reach` places)."""
        top, second = self.candidates, self.second
        self._add("loss", float(loss), 1)
        near = inside[:, 0]
        rest = torch.log1p(-first_log.exp().sum(dim=-1).clamp(max=1.0 - 1e-6))[..., None]
        arms = {"first": torch.cat((first_log, rest), dim=-1)}
        if lift is not None:
            fixed = first_log + torch.where(off[..., None], 0.6, 0.3) * lift.float()         # DForkTree's weights, as CountTree has them
            arms["first_counts"] = torch.cat((fixed, rest), dim=-1).log_softmax(dim=-1)
            arms["second_counts"] = log_with
            arms["second"] = second.scores(first_log, firsts, seconds, states, None, off)
        else:
            arms["second"] = log_with
        covered = (index[:, 0] < top)
        kinds = {"": near, "_on": near & ~off[:, 0], "_off": near & off[:, 0]}
        for kind, which in kinds.items():
            count = int(which.sum())
            self._add("covered" + kind, float((covered & which).sum()), count)
            for name, logs in arms.items():
                right = (logs[:, 0, :top].argmax(dim=-1) == index[:, 0]) & covered
                self._add(name + kind, float((right & which).sum()), count)
                self._add(name + kind + "_log", float((logs[:, 0].gather(-1, index[:, :1])[:, 0] * which).sum()), count)
        self._add("off_share", float((near & off[:, 0]).sum()), int(near.sum()))
        if not learn:
            # Every depth was asked: how many tokens in a row are right from the first place, each place knowing the
            # true tokens before it. The first place is the first run's; the place after depth d is row d's.
            anchors = starts.shape[0]
            for name, logs in arms.items():
                right = ((logs[:, 0, :top].argmax(dim=-1) == index[:, 0]) & covered & near).reshape(anchors, per)
                chain = torch.cat((first_place[:, None], right), dim=1).long().cumprod(dim=1).sum(dim=1)
                self._add(name + "_walk", float(chain.sum()), anchors)


def halves(answers):
    """Which half an answer belongs to (by its prompt, so that it is the same in every run)."""
    return [sum(answer[2][: answer[3]].tolist()) % 2 for answer in answers]


def counts_of(answers, vocabulary, device):
    return FollowCounts.count([(answer[2], answer[3]) for answer in answers], vocabulary).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save", required=True, help="saved as warm-<name>.pt in --adapter-dir")
    parser.add_argument("--start-from", default="", help="go on from the saved layers warm-<name>.pt")
    parser.add_argument("--layers", type=int, default=1, help="how many of the drafter's last layers the second drafter has")
    parser.add_argument("--rank", type=int, default=256, help="of the low-rank updates")
    parser.add_argument("--full", action="store_true", help="control: the copy starts as the drafter's layer and all its weights are trained")
    parser.add_argument("--fresh", action="store_true",
                        help="control: the copy starts from NEW weights, all trained (a new layer on the same text); neither control is saved")
    parser.add_argument("--more-data", action="append", default=[], help="folders of further answers to learn from (none of the validation prompts)")
    parser.add_argument("--more-only", action="store_true", help="learn from --more-data only")
    parser.add_argument("--counts", action="store_true", help="train on top of the counts of the training answers")
    parser.add_argument("--candidates", type=int, default=32, help="tokens of a place the second drafter scores")
    parser.add_argument("--reach", type=int, default=2, help="places after the true tokens that are asked for")
    parser.add_argument("--anchors", type=int, default=256, help="rounds per answer")
    parser.add_argument("--depths", type=int, default=4, help="depths asked per round")
    parser.add_argument("--group", type=int, default=32, help="rounds per pass and update")
    parser.add_argument("--soft", type=float, default=0.5, help="share of the loss that is the target's own probabilities")
    parser.add_argument("--said", type=int, default=32, help="the target's most likely tokens kept per position")
    parser.add_argument("--window", type=int, default=2048, help="tokens of text the drafter reads")
    parser.add_argument("--rate", type=float, default=2e-4)
    parser.add_argument("--max-hours", type=float, default=3.0, help="of training; the measurement comes after")
    parser.add_argument("--max-answers", type=int, default=0, help="train on at most this many answers")
    parser.add_argument("--passes", type=int, default=1, help="times it goes through the answers (other rounds of them each time)")
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens")
    parser.add_argument("--check-answers", type=int, default=12, help="validation answers it is measured on")
    parser.add_argument("--check-every", type=int, default=64, help="tokens between the rounds measured on a validation answer")
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    everything = list(load_answers(args.data_dir))
    answers = [] if args.more_only else [answer for answer in everything if split_of([answer[0], answer[1]]) == "train"]
    checks = [answer for answer in everything if split_of([answer[0], answer[1]]) == "validation"]
    held_out = {tuple(answer[2][: answer[3]].tolist()) for answer in everything if split_of([answer[0], answer[1]]) != "train"}
    answers = answers + more_answers(args.more_data, held_out, set())
    if args.max_answers:                                              # the same answers whenever the same number is asked for
        random.Random(1234).shuffle(answers)
        answers = answers[: args.max_answers]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_to_learn_from": len(answers)}, indent=1)
    target, draft, _ = load_transformers_models(args.model, find_drafter(args.draft), device)
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    with torch.inference_mode(False):
        if args.start_from:
            second, _ = load_restart_layers(os.path.join(args.adapter_dir, f"warm-{args.start_from}.pt"), draft, scores=True)
            second = second.float_parameters().train()
            args.layers, args.rank = second.sizes["count"], second.sizes["rank"]
        else:
            second = RestartLayers(draft, count=args.layers, rank=args.rank, seed=args.seed, reveal=True, scores=True,
                                   fresh=args.fresh, full=args.full).float_parameters().train()
        if not second.reveal or second.first <= 0:
            raise ValueError("this training is for layers that read the tokens and go on from the first run")
        optimizer = torch.optim.AdamW([{"params": second.trained(), "lr": args.rate}], weight_decay=0.0)
    teach = TeachEvery(second, optimizer, draft, target, args.candidates, args.reach, args.group, args.depths, soft=args.soft,
                       window=args.window, seed=args.seed)
    # The counts: a training answer is scored with the counts of the OTHER half of the answers, a validation
    # answer with the counts of all training answers (as a tree would have them in use).
    follow, half_of, tokens_counted = None, {}, 0
    if args.counts:
        which = halves(answers)
        half_of = {id(answer): half for answer, half in zip(answers, which)}
        parts = [counts_of([answer for answer, half in zip(answers, which) if half == side], target.config.vocab_size, device) for side in (0, 1)]
        follow = counts_of(answers, target.config.vocab_size, device)
        tokens_counted = sum(len(answer[2]) - answer[3] for answer in answers)
    curve, started = [], time.time()

    def report(done, learned, measured=None):
        write_json(status_path, {
            "done": done, "answers": learned, "answers_to_learn_from": len(answers), "updates": teach.updates, "rows": teach.rows_seen,
            "draft": args.draft, "layers": args.layers, "rank": args.rank, "saved_as": "" if args.fresh or args.full else args.save,
            "started_from": args.start_from, "control": "fresh" if args.fresh else "full" if args.full else "", "rate": args.rate,
            "trained_numbers": sum(value.numel() for value in second.trained()),
            "candidates": args.candidates, "reach": args.reach, "anchors": args.anchors, "depths": args.depths, "soft": args.soft,
            "counts": bool(args.counts), "tokens_counted": tokens_counted, "more_data": args.more_data,
            "scores": {name: round(float(value), 4) for name, value in (("strength", second.scores.strength), ("on_path", second.scores.on_path),
                                                                        ("off_path", second.scores.off_path))},
            "hours": round((time.time() - started) / 3600, 3), "curve": curve[-400:], "validation": measured,
        }, indent=1)

    def save(learned):
        if args.fresh or args.full:                   # the controls are whole layers (about 200 MB): measured, not kept
            return
        notes = {"draft": args.draft, "answers": learned, "updates": teach.updates, "trained_by": "train_every", "counts": bool(args.counts),
                 "candidates": args.candidates, "reach": args.reach, "curve_end": curve[-1] if curve else None}
        save_restart_layers(os.path.join(args.adapter_dir, f"warm-{args.save}.pt"), second, notes)

    def said_and_states(ids):
        features, said = target_features(target, draft, ids, top=args.said)
        return features.clone(), (said[0].clone(), said[1].clone())

    learned, saved_at, order = 0, time.time(), random.Random(args.seed)
    waiting = [entry for _ in range(max(args.passes, 1)) for entry in answers]
    order.shuffle(waiting)
    for answer in waiting:
        name, index, sequence, prompt_length, source = answer
        if time.time() - started > args.max_hours * 3600:
            break
        if len(sequence) - prompt_length < 3 * teach.width or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        second.train()
        ids = sequence.long().to(device)[None]
        features, said = said_and_states(ids)
        teach.learn_from(ids, features, said, prompt_length, args.anchors, parts[1 - half_of[id(answer)]] if args.counts else None)
        del features, said
        learned += 1
        if learned % 8 == 0:
            curve.append({"answers": learned, **teach.progress()})
        report(False, learned)
        if time.time() - saved_at > 1800:                                # what is learned so far outlives a job that is cut off
            save(learned)
            saved_at = time.time()

    second.eval()
    save(learned)
    teach.reset()
    measured = 0
    for name, index, sequence, prompt_length, source in checks:
        if measured >= args.check_answers:
            break
        if len(sequence) - prompt_length < 3 * teach.width or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        features, said = said_and_states(ids)
        teach.learn_from(ids, features, said, prompt_length, 0, follow, learn=False, every=args.check_every)
        del features, said
        measured += 1
    report(True, learned, {"answers": measured, **teach.progress()})


if __name__ == "__main__":
    main()
