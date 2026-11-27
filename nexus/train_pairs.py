"""Train scores of pairs of neighbouring tokens: the second drafter's knowledge without a pass.

    python nexus/train_pairs.py --draft opd --data-dir ~/specdec-data/long1 --more-data ~/specdec-data/think1 \
        --counts --results-dir nexus/results/NAME --save opd-p1

dflash/pair_scores.py says what the scores are and why. They are trained as the second drafter is
(nexus/train_every.py): at every place of a round's block, with the answer's own two tokens before
the place, over the first run's candidates only, on top of the counts (each half of the answers is
scored with the other half's counts), half on the target's token and half on the target's own
probabilities. They need no pass of their own: a round is the first run and a few small products,
so every place of a block is a training example (14 a round), not a few sampled depths.

The same things are counted as in train_every.py, under the same names ("second" is the pair
scores here), so the two can be read side by side: how often the first choice is right at a place
given the true tokens before it, all places / on the first run's path / off it, and "walk".
Saved: pairs-<name>.pt in --adapter-dir.
"""
import argparse
import os
import random
import time

import torch

from dflash.benchmark import load_transformers_models
from dflash.model import _draft_value, _output_head, _raw_input_embeddings
from dflash.pair_scores import PairScores, save_pair_scores
from dflash.replay import target_features
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers
from nexus.train_every import TeachEvery, counts_of, first_runs, halves
from nexus.train_second import more_answers


class TeachPairs(TeachEvery):
    """As TeachEvery, for scores that read the first run's states: all places of every round, no pass."""

    def __init__(self, pairs, optimizer, drafter, target, candidates=32, group=32, soft=0.5, window=2048, width=16, seed=0,
                 depth_decay=7.0, warm=100, teacher=None, teacher_weight=0.0, teacher_places=4):
        super().__init__(None, optimizer, drafter, target, candidates, 1, group, 1, soft=soft, window=window, width=width, seed=seed,
                         depth_decay=depth_decay, warm=warm)
        self.pairs = pairs
        # teacher: a trained second drafter (dflash/restart_layers.py, e.g. warm-opd-e4.pt) whose probabilities of the
        # same candidates at the same places (given the true tokens before) the scores are taught to match, with a
        # KL over the candidates and "another token" (C12: its order alone and its strengths alone transfer nothing;
        # the probabilities do). The loss is (1 - teacher_weight) x the target's labels + teacher_weight x the KL,
        # at `teacher_places` places of each round (a pass of the teacher costs by its rows).
        self.teacher, self.teacher_weight, self.teacher_places = teacher, teacher_weight, teacher_places
        if teacher is not None and not (0.0 <= teacher_weight <= 1.0):
            raise ValueError("teacher_weight is between 0 and 1")

    def _teacher_log(self, teacher, features, ids, starts, kept, low, best, first_scores, wrong, follow):
        """The teacher's log-probabilities of the candidates (and "another token") at `teacher_places` places of each
        round, given the answer's true tokens before them: (owner [rows], column [rows] into the places 2 .. width - 1,
        log-probabilities [rows, top + 1]). One pass of the teacher's layers for all rows."""
        from dflash.restart_layers import candidate_scores
        from nexus.train_every import second_rows

        drafter, target, width, device = self.drafter, self.target, self.width, ids.device
        head = _output_head(target)
        anchors = starts.shape[0]
        per = min(self.teacher_places, width - 2)
        # depth d: the answer's tokens at places 1 .. d are written in; the row predicts place d + 1 (column d - 1).
        depths = torch.stack([torch.randperm(width - 2, generator=self.generator)[:per] + 1 for _ in range(anchors)]).to(device)
        owner = torch.arange(anchors, device=device)[:, None].expand(-1, per).flatten()
        depths = depths.flatten()
        states, _ = second_rows(drafter, teacher, target, features, ids, starts[owner], depths, kept[owner], low, 1, width)
        states = states[:, 0]                                                              # [rows, size]
        column = depths - 1
        candidates = best.indices[owner, depths]                                           # (place d + 1 is row d of best)
        first_log, firsts = best.values[owner, depths], first_scores[owner, depths]
        if teacher.scores.out is not None:
            seconds = candidate_scores(drafter, head, teacher.scores.out(states.to(teacher.scores.out.weight.dtype)), candidates)
            firsts = torch.zeros_like(firsts)
        else:
            seconds = candidate_scores(drafter, head, states, candidates)
        off = wrong[owner, depths - 1] > 0
        lift = None
        if follow is not None:
            position = starts[owner] + depths + 1
            lift = follow.lift(ids[0][position - 2][..., None], ids[0][position - 1][..., None], candidates)
        return owner, column, teacher.scores(first_log, firsts, seconds, states, lift, off).detach()

    def step(self, ids, features, said, starts, follow=None, learn=True):
        drafter, target, pairs, width, device, top = self.drafter, self.target, self.pairs, self.width, ids.device, self.candidates
        head = _output_head(target)
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        low = max(int(starts.max()) - self.window, 0)
        teacher = self.teacher
        with torch.no_grad():
            final, kept = first_runs(drafter, target, features, ids, starts, low, 0 if teacher is None else teacher.first - 1, width)
            answer = ids[0][starts[:, None] + torch.arange(1, width, device=device)[None, :]]        # the tokens of the places 1 .. width - 1
            logits = drafter.compute_logits(final[:, 1:], head).float()
            likely = logits.log_softmax(dim=-1)
            best = likely.topk(top, dim=-1)
            first_scores = logits.gather(-1, best.indices) if teacher is not None else None        # (the teacher reads the raw scores)
            likely_of_answer = likely.gather(-1, answer[..., None])[..., 0]
            del logits
            before_sure = two_sure = None
            if pairs.sure_before is not None:
                # How the first run stood to the answer's token of every place: its log-probability of it, whether it
                # was its first choice, the log-probability of its first choice. The block's first token was given.
                sure = torch.stack((likely.gather(-1, answer[..., None])[..., 0], (best.indices[..., 0] == answer).float(),
                                    best.values[..., 0]), dim=-1)                                    # [anchors, width - 1, 3]
                given = torch.tensor([0.0, 1.0, 0.0], device=device)[None, None].expand(sure.shape[0], 1, 3)
                before_sure, two_sure = sure[:, :-1], torch.cat((given, sure[:, :-2]), dim=1)
            del likely
            wrong = (best.indices[..., 0] != answer).long().cumsum(dim=1)
            # The places 2 .. width - 1: their candidates, the two tokens before them, their state.
            candidates, first_log, states = best.indices[:, 1:], best.values[:, 1:], final[:, 2:]
            before, two_before = answer[:, :-1], torch.cat((ids[0][starts][:, None], answer[:, :-2]), dim=1)
            label, off = answer[:, 1:], wrong[:, :-1] > 0
            branch = None
            if pairs.branch_gate is not None:
                # The branch before each place, from the answer's tokens: a fork is a place where the answer's token
                # was not the first choice (dflash/pair_scores.py: branch_inputs; dflash/redraft.py: branch_numbers).
                places = answer.shape[1]
                forked = best.indices[..., 0] != answer                                      # [anchors, places 1 .. width - 1]
                index = torch.arange(places, device=device)[None, :].expand_as(forked)
                first_at = torch.where(forked, index, torch.full_like(index, places)).cummin(dim=1).values
                last_at = torch.where(forked, index, torch.full_like(index, -1)).cummax(dim=1).values
                gaps = torch.where(forked, likely_of_answer - best.values[..., 0], torch.zeros_like(best.values[..., 0])).cumsum(dim=1)
                # For the place at column c (place c + 2, index c + 1 here) the branch is the places up to index c.
                depth = torch.arange(1, places, device=device, dtype=torch.float)[None, :].expand(forked.shape[0], -1)
                since_first = torch.where(first_at[:, :-1] < places, depth - first_at[:, :-1].float(), torch.zeros_like(depth))
                since_last = torch.where(last_at[:, :-1] >= 0, depth - last_at[:, :-1].float(), torch.zeros_like(depth))
                branch = pairs.branch_inputs(since_first, since_last, wrong[:, :-1], gaps[:, :-1])
            position = starts[:, None] + torch.arange(2, width, device=device)[None, :]
            embedded = _raw_input_embeddings(target, torch.stack((before, two_before)), scale)
            rows = head.weight[candidates]
            lift = None if follow is None else follow.lift(two_before[..., None], before[..., None], candidates)
            hit = candidates == label[..., None]
            index = torch.where(hit.any(dim=-1), hit.float().argmax(dim=-1), torch.full_like(label, top))
            told, told_log = said[0][position - 1], said[1][position - 1]
            match = candidates[..., :, None] == told[..., None, :]
            share = (match * told_log.exp()[..., None, :]).sum(dim=-1)
            share = torch.cat((share, (1.0 - share.sum(dim=-1, keepdim=True)).clamp_min(0.0)), dim=-1)
            weight = torch.exp(-torch.arange(width - 2, device=device, dtype=torch.float) / self.depth_decay)[None, :].expand_as(label)
            taught = None
            if teacher is not None and self.teacher_weight > 0:
                taught = self._teacher_log(teacher, features, ids, starts, kept, low, best, first_scores, wrong, follow)
        with torch.set_grad_enabled(learn):
            log_with = pairs(first_log, states, embedded[0], embedded[1], rows, lift, off, before_sure, two_sure, branch)
            hard = -log_with.gather(-1, index[..., None])[..., 0]
            soft = -(share * log_with).sum(dim=-1)
            loss = (((1.0 - self.soft) * hard + self.soft * soft) * weight).sum() / weight.sum()
            if taught is not None:
                owner, column, teacher_log = taught
                student_log = log_with[owner, column]                                      # [rows, top + 1]
                kl = (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1)         # KL(teacher || student)
                kl = (kl * weight[owner, column]).sum() / weight[owner, column].sum()
                loss = (1.0 - self.teacher_weight) * loss + self.teacher_weight * kl
                self._add("teacher_kl", float(kl.detach()), 1)
        if learn:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pairs.parameters(), 1.0)
            self.updates += 1
            for group, rate in zip(self.optimizer.param_groups, self.rates):
                group["lr"] = rate * min(1.0, self.updates / max(self.warm, 1))
            self.optimizer.step()
            self.rows_seen += int(label.numel())
        with torch.no_grad():
            self._add("loss", float(loss), 1)
            rest = torch.log1p(-first_log.exp().sum(dim=-1).clamp(max=1.0 - 1e-6))[..., None]
            arms = {"first": torch.cat((first_log, rest), dim=-1)}
            if lift is not None:
                fixed = first_log + torch.where(off[..., None], 0.6, 0.3) * lift.float()
                arms["first_counts"] = torch.cat((fixed, rest), dim=-1).log_softmax(dim=-1)
                arms["second_counts"] = log_with
                arms["second"] = pairs(first_log, states, embedded[0], embedded[1], rows, None, off, before_sure, two_sure, branch)
            else:
                arms["second"] = log_with
            covered = index < top
            everything = torch.ones_like(off)
            for kind, which in {"": everything, "_on": ~off, "_off": off}.items():
                count = int(which.sum())
                self._add("covered" + kind, float((covered & which).sum()), count)
                for name, logs in arms.items():
                    right = (logs[..., :top].argmax(dim=-1) == index) & covered
                    self._add(name + kind, float((right & which).sum()), count)
                    self._add(name + kind + "_log", float((logs.gather(-1, index[..., None])[..., 0] * which).sum()), count)
            self._add("off_share", float(off.sum()), int(off.numel()))
            first_place = best.indices[:, 0, 0] == answer[:, 0]
            for name, logs in arms.items():
                right = (logs[..., :top].argmax(dim=-1) == index) & covered
                chain = torch.cat((first_place[:, None], right), dim=1).long().cumprod(dim=1).sum(dim=1)
                self._add(name + "_walk", float(chain.sum()), starts.shape[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save", required=True, help="saved as pairs-<name>.pt in --adapter-dir")
    parser.add_argument("--more-data", action="append", default=[])
    parser.add_argument("--counts", action="store_true", help="train on top of the counts of the training answers")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--sure", action="store_true",
                        help="tell the scores how sure the first run was of the tokens before a place (dflash/pair_scores.py)")
    parser.add_argument("--sure-inputs", default="all",
                        help='which of the three numbers the scores see: "all", or a comma list of logp, top1, best')
    parser.add_argument("--sure-scalar", action="store_true", help="one scale per token instead of one per rank")
    parser.add_argument("--branch", action="store_true", help="a learned gate on the correction from the branch's four numbers")
    parser.add_argument("--teacher", default="", help="a trained second drafter warm-<name>.pt to match (a KL over the candidates)")
    parser.add_argument("--teacher-weight", type=float, default=0.0, help="the KL's share of the loss, 0 to 1 (1: the teacher alone)")
    parser.add_argument("--teacher-places", type=int, default=4, help="places of each round the teacher scores")
    parser.add_argument("--anchors", type=int, default=256, help="rounds per answer")
    parser.add_argument("--group", type=int, default=32, help="rounds per update")
    parser.add_argument("--soft", type=float, default=0.5)
    parser.add_argument("--said", type=int, default=32)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--max-answers", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--check-answers", type=int, default=12)
    parser.add_argument("--check-every", type=int, default=64)
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    everything = list(load_answers(args.data_dir))
    answers = [answer for answer in everything if split_of([answer[0], answer[1]]) == "train"]
    checks = [answer for answer in everything if split_of([answer[0], answer[1]]) == "validation"]
    held_out = {tuple(answer[2][: answer[3]].tolist()) for answer in everything if split_of([answer[0], answer[1]]) != "train"}
    answers = answers + more_answers(args.more_data, held_out, set())
    if args.max_answers:
        random.Random(1234).shuffle(answers)
        answers = answers[: args.max_answers]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_to_learn_from": len(answers)}, indent=1)
    target, draft, _ = load_transformers_models(args.model, find_drafter(args.draft), device)
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    with torch.inference_mode(False):
        pairs = PairScores(draft.fc.weight.shape[0], args.rank, args.hidden, sure=args.sure,
                           sure_inputs=args.sure_inputs, sure_scalar=args.sure_scalar, branch=args.branch).to(device).train()
        optimizer = torch.optim.AdamW(pairs.parameters(), lr=args.rate, weight_decay=0.0)
    teacher = None
    if args.teacher:
        from dflash.restart_layers import load_restart_layers

        teacher, _ = load_restart_layers(os.path.join(args.adapter_dir, f"warm-{args.teacher}.pt"), draft)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    teach = TeachPairs(pairs, optimizer, draft, target, args.candidates, args.group, soft=args.soft, window=args.window, seed=args.seed,
                       teacher=teacher, teacher_weight=args.teacher_weight, teacher_places=args.teacher_places)
    follow, half_of, parts, tokens_counted = None, {}, None, 0
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
            "draft": args.draft, "what": "pair scores", "rank": args.rank, "hidden": args.hidden, "sure": bool(args.sure),
            "sure_inputs": args.sure_inputs, "sure_scalar": bool(args.sure_scalar),
            "branch": bool(args.branch), "teacher": args.teacher or None, "teacher_weight": args.teacher_weight, "teacher_places": args.teacher_places,
            "saved_as": args.save,
            "candidates": args.candidates, "soft": args.soft, "counts": bool(args.counts), "tokens_counted": tokens_counted,
            "trained_numbers": sum(value.numel() for value in pairs.parameters()),
            "scores": {"on_path": round(float(pairs.on_path), 4), "off_path": round(float(pairs.off_path), 4)},
            "hours": round((time.time() - started) / 3600, 3), "curve": curve[-400:], "validation": measured,
        }, indent=1)

    def save(learned):
        save_pair_scores(os.path.join(args.adapter_dir, f"pairs-{args.save}.pt"), pairs,
                         {"draft": args.draft, "answers": learned, "updates": teach.updates, "counts": bool(args.counts), "candidates": args.candidates})

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
        if len(sequence) - prompt_length < 3 * teach.width:
            continue
        pairs.train()
        ids = sequence.long().to(device)[None]
        features, said = said_and_states(ids)
        teach.learn_from(ids, features, said, prompt_length, args.anchors, parts[1 - half_of[id(answer)]] if args.counts else None)
        del features, said
        learned += 1
        if learned % 8 == 0:
            curve.append({"answers": learned, **teach.progress()})
        report(False, learned)
        if time.time() - saved_at > 1800:
            save(learned)
            saved_at = time.time()

    pairs.eval()
    save(learned)
    teach.reset()
    measured = 0
    for name, index, sequence, prompt_length, source in checks:
        if measured >= args.check_answers:
            break
        if len(sequence) - prompt_length < 3 * teach.width:
            continue
        ids = sequence.long().to(device)[None]
        features, said = said_and_states(ids)
        teach.learn_from(ids, features, said, prompt_length, 0, follow, learn=False, every=args.check_every)
        del features, said
        measured += 1
    report(True, learned, {"answers": measured, **teach.progress()})


if __name__ == "__main__":
    main()
