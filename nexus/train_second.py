"""Train the tiny second drafter: the drafter's last layers, restarting after a break from the first run.

    python nexus/train_second.py --draft opd --predictor opd-m0 --data-dir ~/specdec-data/long1 --results-dir nexus/results/NAME --save opd-w1

The second run that restarts the whole drafter after a break costs a whole drafter pass. This
trains dflash/restart_layers.py: copies of the drafter's last --layers layers (1 by default) with
low-rank updates, which go on from the first run. A step plays a round of a saved answer as real
decoding would: the drafter's first run from the round's first token; then, at the place where that
draft really breaks, a restart from the right token through the copies only. The places after the
token enter with the states the first run had there before those layers, and the tokens of the
round the target has not read are given as predicted states (the saved predictor --predictor,
trained on with the layers unless --freeze-predictor). The loss is how unlikely the restart finds
the tokens the answer really has after it (the first ones count most), plus --vector-weight times
the error of the predicted states. The drafter itself is not changed.

On the same rounds, every --compare-every rounds, it counts the tokens that are right from the
restart on (those inside the first block, and at most 3 of them: what the second run lets the
target check) for:

    second             the tiny second drafter with predicted states: what is being trained
    second_real        the same with the target's real states (its limit with a perfect predictor)
    second_none        the same with no states at all (what it gets from the first run and the token alone)
    whole_predicted    the whole drafter restarting with the predicted states: the second run as it is,
                       at the cost of a whole pass (the states of the predictor as it was saved)
    whole_real         the whole drafter with the real states (the next round's first draft: the limit)
    whole_none         the whole drafter with no states
    old_rest           the first draft's old tokens after the right token (no pass at all)

After training the same is counted on validation answers, every round, with nothing learned. Saved:
warm-<name>.pt (the low-rank updates, about 28 MB for one layer at rank 256) and, unless the predictor
is frozen, restart-<name>.pt (the predictor as trained with the layers) in --adapter-dir.
"""
import argparse
import copy
import os
import time

import torch

from dflash.benchmark import load_transformers_models
from dflash.model import _draft_value, _output_head, _raw_input_embeddings
from dflash.replay import target_features
from dflash.restart_layers import RestartLayers, save_restart_layers
from dflash.stand_in import StatePredictor, as_context, load_predictor, project, save_predictor, state_inputs
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers
from nexus.train_restart import first_run_states, restart_logits

ARMS = ("second", "second_real", "second_none", "whole_predicted", "whole_real", "whole_none", "old_rest")


def first_run(drafter, target, features, ids, start, low, width, keep):
    """The drafter's first run of the round that starts at `start`: its final states of the block
    ([width, size]) and its states after layer `keep` ([width, size])."""
    kept = {}
    hook = drafter.layers[keep].register_forward_hook(lambda layer, came, output: kept.update(states=output))
    try:
        final = first_run_states(drafter, target, features, ids, start, low, width)
    finally:
        hook.remove()
    return final, kept["states"][0]


def warm_states(drafter, second, features, ids, start, written, entering, low=0, anchor=None):
    """The restart layers' final states after the token at start + count, going on from the first run:
    `entering` ([places, size]) are the states the places from that token on enter them with, `written`
    ([count, size]) the states of the tokens start .. start + count - 1 as the drafter reads states.
    Returns [places - 1, size]. Nothing is cached, so the layers and a predictor can learn through it.
    anchor: where the restart starts when `written` covers fewer tokens than are missing (none: [0, size])."""
    anchor = start + written.shape[0] if anchor is None else anchor
    positions = torch.cat((torch.arange(low, start + written.shape[0], device=ids.device),
                           torch.arange(anchor, anchor + entering.shape[0], device=ids.device)))[None]
    hidden = drafter(target_hidden=features[:, low:start], context_states=written[None], noise_embedding=entering[None],
                     position_ids=positions, layers=list(second.layers))
    return hidden[0, 1:, :]


def warm_logits(drafter, second, target, features, ids, start, written, entering, low=0, anchor=None):
    """What the restart layers predict there: [places - 1, vocabulary]."""
    return drafter.compute_logits(warm_states(drafter, second, features, ids, start, written, entering, low, anchor)[None],
                                  _output_head(target))[0]


def whole_states(drafter, target, features, ids, start, written, low=0, width=16):
    """The whole drafter's final states when it restarts at start + count with `written` for the tokens
    before (nexus/train_restart.py: restart_logits, before the output head): [width - 1, size]."""
    from nexus.train_restart import _block_at

    anchor = start + written.shape[0]
    positions = torch.cat((torch.arange(low, anchor, device=ids.device),
                           torch.arange(anchor, anchor + width, device=ids.device)))[None]
    hidden = drafter(target_hidden=features[:, low:start], context_states=written[None],
                     noise_embedding=_block_at(drafter, target, ids, anchor, width), position_ids=positions)
    return hidden[0, 1:, :]


def _right(logits, answer):
    """How many tokens are right from the first one on."""
    return int((logits.argmax(dim=-1) == answer).long().cumprod(dim=0).sum())


class TeachSecond:
    """Trains the restart layers (and the predictor) on the tokens they write (see the top of this file)."""

    def __init__(self, second, predictor, optimizer, drafter, target, every=8, window=2048, vector_weight=0.1, decay=0.8,
                 width=16, seed=0, compare_every=4, train_predictor=True, distill=0.0, real_share=0.0, earlier=True):
        self.second, self.predictor, self.optimizer, self.drafter, self.target = second, predictor, optimizer, drafter, target
        self.every, self.window, self.vector_weight, self.width = every, window, vector_weight, width
        self.compare_every, self.train_predictor = compare_every, train_predictor
        # distill: the weight of a second aim: the final states of the WHOLE drafter restarting with the target's
        # real states (the next round's first draft), which the restart layers' final states are pulled towards
        # (one minus their cosine, the first places counting most). A token says one thing about a place, the
        # whole drafter's state there says what it knows.
        # real_share: in this share of the steps the layers are given the target's real states in place of the
        # predicted ones. Before training they do not look at the states at all; with the real ones there is
        # something to find, and the way of reading them carries over to the predicted ones.
        self.distill, self.real_share = distill, real_share
        # earlier: the layers are also taught the restart one token earlier, from the last token the first draft
        # got right (dflash/redraft.py: Restarts, kind "earlier"): it needs no guess of the token at the break
        # and one state fewer, and writes the token at the break itself. In real decoding half of what the second
        # run wins comes from these restarts.
        self.earlier, self.also = earlier, []
        # The predictor as it was saved: the whole drafter's restart is measured with its states, so that the
        # comparison is with the second run as it is today.
        self.saved_predictor = copy.deepcopy(predictor).eval() if train_predictor else predictor
        for parameter in self.saved_predictor.parameters() if train_predictor else ():
            parameter.requires_grad_(False)
        self.weights = decay ** torch.arange(width - 1, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)
        self.rounds = self.steps = self.unbroken = self.at_the_end = 0
        self.recent, self.compared = [], []

    def learn_from(self, ids, features, prompt_length, most=600, learn=True):
        """Random rounds of one saved answer. features: the target's states of the answer (a copy that
        was not made by the replay)."""
        last = ids.shape[1] - 2 * self.width - 1
        if last <= prompt_length:
            return
        for _ in range(min(most, max((ids.shape[1] - prompt_length) // 24, 1))):
            start = int(torch.randint(prompt_length, last, (1,), generator=self.generator))
            self.step(ids, features, start, learn)

    def inputs(self, predictor, block_states, ids, features, start, count, low):
        """What a predictor reads for the round's first `count` tokens, as in real decoding."""
        scale = float(_draft_value(self.drafter.config, "input_embedding_scale", 1.0))
        tokens = _raw_input_embeddings(self.target, ids[:, start : start + count], scale)[0].float()
        coming = _raw_input_embeddings(self.target, ids[:, start + 1 : start + count + 1], scale)[0].float()
        last = project(self.drafter, features[:, start - 1 : start])[0].float()
        most, recalled = int(predictor.sizes.get("memory", 0)), None
        if most:
            held = int(torch.randint(1, most + 1, (1,), generator=self.generator))
            recalled = project(self.drafter, features[:, max(start - held, low) : start])[0].float()
        return (*state_inputs(block_states, count), tokens, last, torch.arange(count, device=ids.device), coming, recalled)

    def step(self, ids, features, start, learn=True, count=None):
        """count: how many of the round's tokens the target has not read at the restart (None: where the first
        draft of the round really breaks, as in use)."""
        drafter, target, width, device = self.drafter, self.target, self.width, ids.device
        low = max(start - self.window, 0)
        with torch.no_grad():
            block_states, kept = first_run(drafter, target, features, ids, start, low, width, self.second.first - 1)
            first = drafter.compute_logits(block_states[None, 1:], _output_head(target))[0]
            wrong = (first.argmax(dim=-1) != ids[0, start + 1 : start + width]).nonzero()
            if count is None and not wrong.numel():
                self.unbroken += 1                    # the first draft is right to its end: nothing to restart
                return
            count = int(wrong[0]) + 1 if count is None else count     # tokens of the round the target has not read at the restart
            places = width - count                    # from the restart's first token to the end of the first block
            if places < 2:
                self.at_the_end += 1                  # the break is at the block's last place: no place to go on from
                return
            given = self.inputs(self.predictor, block_states, ids, features, start, count, low)
            real = project(drafter, features[:, start : start + count])[0].float()
            scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
            trio = torch.stack((ids[0, start + count], torch.as_tensor(drafter.mask_token_id, dtype=ids.dtype, device=device),
                                ids[0, start + count - 1]))
            token_in, mask_in, before_in = _raw_input_embeddings(target, trio[None], scale)[0]
            from_the_break = ids[0, start + count : start + width].clone()             # the tokens from the break on, inside the block
            answer = ids[0, start + count + 1 : start + width].clone()                  # the tokens after the right token, inside the block
            aim = None
            if learn and self.distill:
                aim = whole_states(drafter, target, features, ids, start, real.to(block_states.dtype), low, width)[: places - 1]
            with_real = learn and self.real_share > 0 and float(torch.rand(1, generator=self.generator)) < self.real_share
        with torch.set_grad_enabled(learn):
            written = self.predictor(*given)
            if not (learn and self.train_predictor):
                written = written.detach()
            written = as_context(drafter, written)
            entering = self.second.enter(kept[count:], token_in, mask_in)
            states = warm_states(drafter, self.second, features, ids, start, real if with_real else written, entering, low)
            logits = drafter.compute_logits(states[None], _output_head(target))[0].float()
            each = torch.nn.functional.cross_entropy(logits, answer, reduction="none")
            weights = self.weights[: places - 1].to(device)
            by_tokens = (each * weights).sum() / weights.sum()
            by_vectors = torch.nn.functional.mse_loss(written, real)
            total = by_tokens + self.vector_weight * by_vectors
            if aim is not None:
                apart = 1.0 - torch.nn.functional.cosine_similarity(states.float(), aim.float(), dim=-1)
                total = total + self.distill * (apart * weights).sum() / weights.sum()
            before = None
            if self.earlier and count >= 2:
                # One token earlier: from the last right token, with the states of the tokens before it.
                given_before = (real if with_real else written)[: count - 1]
                before = warm_logits(drafter, self.second, target, features, ids, start, given_before,
                                     self.second.enter(kept[count - 1 :], before_in, mask_in), low).float()
                wider = self.weights[: places].to(device)
                total = total + (torch.nn.functional.cross_entropy(before, from_the_break, reduction="none") * wider).sum() / wider.sum()
            if learn:
                (total / self.every).backward()
        if with_real:           # (the counts below are of the layers as they are used: with the predicted states)
            with torch.no_grad():
                logits = warm_logits(drafter, self.second, target, features, ids, start, written, entering, low).float()
        self.rounds += 1
        if learn and self.rounds % self.every == 0:
            torch.nn.utils.clip_grad_norm_(self.second.trained() + list(self.predictor.parameters()), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.steps += 1
        with torch.no_grad():
            right = _right(logits, answer)
            self.recent.append([float(by_tokens.detach()), float(by_vectors.detach()), float(right)])
            if self.compare_every and self.rounds % self.compare_every == 0:
                counts = {"second": right}
                counts["second_real"] = _right(warm_logits(drafter, self.second, target, features, ids, start,
                                                           real.to(written.dtype), entering, low), answer)
                counts["second_none"] = _right(warm_logits(drafter, self.second, target, features, ids, start,
                                                           real[:0].to(written.dtype), entering, low, anchor=start + count), answer)
                saved = written
                if self.train_predictor:
                    saved = as_context(drafter, self.saved_predictor(*self.inputs(self.saved_predictor, block_states, ids,
                                                                                  features, start, count, low)))
                for name, states in (("whole_predicted", saved), ("whole_real", real), ("whole_none", real[:0])):
                    whole = restart_logits(drafter, target, features, ids, start, states.to(written.dtype), low, width,
                                           anchor=start + count)
                    counts[name] = _right(whole[: places - 1], answer)
                counts["old_rest"] = _right(first[count:], answer)
                self.compared.append([counts[name] for name in ARMS])
                if before is not None:
                    # From the token at the break on: the layers from the last right token (predicted states), and
                    # the whole drafter from there with the saved predictor's states and with the real ones.
                    if with_real:
                        before = warm_logits(drafter, self.second, target, features, ids, start, written[: count - 1],
                                             self.second.enter(kept[count - 1 :], before_in, mask_in), low)
                    others = [restart_logits(drafter, target, features, ids, start, states[: count - 1].to(written.dtype), low, width)
                              [:places] for states in (saved, real)]
                    self.also.append([_right(before, from_the_break)] + [_right(other, from_the_break) for other in others])

    def progress(self):
        """Means over the rounds since the last call."""
        recent, self.recent = self.recent, []
        if not recent:
            return None
        by_tokens, by_vectors, right = (sum(column) / len(recent) for column in zip(*recent))
        point = {"rounds": self.rounds, "steps": self.steps, "loss_by_tokens": round(by_tokens, 4),
                 "error_of_the_predicted_states": round(by_vectors, 5), "tokens_right_after_the_restart": round(right, 3)}
        compared, self.compared = self.compared, []
        if compared:
            table = torch.tensor(compared, dtype=torch.float32)
            point["on_the_same_rounds"] = {
                "rounds": len(compared),
                "tokens_right_inside_the_block": {name: round(float(table[:, at].mean()), 3) for at, name in enumerate(ARMS)},
                "at_most_3_of_them": {name: round(float(table[:, at].clamp(max=3).mean()), 3) for at, name in enumerate(ARMS)}}
        also, self.also = self.also, []
        if also:
            table = torch.tensor(also, dtype=torch.float32)
            names = ("second", "whole_predicted", "whole_real")
            point["one_token_earlier_from_the_break_on"] = {
                "rounds": len(also),
                "tokens_right_inside_the_block": {name: round(float(table[:, at].mean()), 3) for at, name in enumerate(names)},
                "at_most_4_of_them": {name: round(float(table[:, at].clamp(max=4).mean()), 3) for at, name in enumerate(names)}}
        return point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save", required=True, help="saved as warm-<name>.pt (and restart-<name>.pt) in --adapter-dir")
    parser.add_argument("--predictor", default="", help="start from the saved predictor restart-<name>.pt (none: a new one)")
    parser.add_argument("--freeze-predictor", action="store_true", help="train the layers only")
    parser.add_argument("--layers", type=int, default=1, help="how many of the drafter's last layers the second drafter has")
    parser.add_argument("--rank", type=int, default=256, help="of the low-rank updates")
    parser.add_argument("--part", default="train", help="the answers it learns from")
    parser.add_argument("--vector-weight", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=2048, help="tokens of text the drafter reads in a step")
    parser.add_argument("--rate", type=float, default=2e-4)
    parser.add_argument("--predictor-rate", type=float, default=1e-4)
    parser.add_argument("--every", type=int, default=8, help="rounds per update of the weights")
    parser.add_argument("--compare-every", type=int, default=4)
    parser.add_argument("--distill", type=float, default=1.0,
                        help="weight of the pull towards the whole drafter's final states with the real states (0: off)")
    parser.add_argument("--only-after", action="store_true", help="do not also teach the restart one token earlier")
    parser.add_argument("--real-share", type=float, default=0.3,
                        help="share of the steps in which the layers are given the target's real states")
    parser.add_argument("--max-hours", type=float, default=3.0, help="of training; the measurement comes after")
    parser.add_argument("--max-answers", type=int, default=0, help="train on at most this many answers")
    parser.add_argument("--passes", type=int, default=1, help="times it goes through the answers (other rounds of them each time)")
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens")
    parser.add_argument("--check-answers", type=int, default=12, help="validation answers it is measured on afterwards")
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    everything = list(load_answers(args.data_dir))
    answers = [answer for answer in everything if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    checks = [answer for answer in everything if split_of([answer[0], answer[1]]) == "validation"] if args.part != "all" else everything
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0, "answers_to_learn_from": len(answers)}, indent=1)
    target, draft, _ = load_transformers_models(args.model, find_drafter(args.draft), device)
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    from nexus.redraft_long import token_counts

    token_counts(args.adapter_dir, args.data_dir, target.config.vocab_size)       # the count a short list of tokens is made from
    with torch.inference_mode(False):
        second = RestartLayers(draft, count=args.layers, rank=args.rank, seed=args.seed).train()
        if args.predictor:
            predictor, _ = load_predictor(os.path.join(args.adapter_dir, f"restart-{args.predictor}.pt"), device)
            predictor = predictor.float()
        else:
            predictor = StatePredictor(draft.config.hidden_size, block=16).to(device)
        groups = [{"params": second.trained(), "lr": args.rate}]
        if not args.freeze_predictor:
            groups.append({"params": list(predictor.parameters()), "lr": args.predictor_rate})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    teach = TeachSecond(second, predictor, optimizer, draft, target, args.every, args.window, args.vector_weight, seed=args.seed,
                        compare_every=args.compare_every, train_predictor=not args.freeze_predictor, distill=args.distill,
                        real_share=args.real_share, earlier=not args.only_after)
    curve, started = [], time.time()

    def report(done, learned, measured=None):
        status = {"done": done, "answers": learned, "answers_to_learn_from": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "draft": args.draft, "save": args.save,
                  "layers": args.layers, "of_the_drafters_layers": len(draft.layers), "rank": args.rank, "rate": args.rate,
                  "predictor": args.predictor, "predictor_trained_too": not args.freeze_predictor,
                  "predictor_rate": args.predictor_rate, "every": args.every, "vector_weight": args.vector_weight,
                  "distill": args.distill, "real_share": args.real_share, "one_token_earlier_too": not args.only_after,
                  "parameters_trained": sum(value.numel() for value in second.trained()),
                  "rounds_without_a_break": teach.unbroken, "rounds_broken_at_the_last_place": teach.at_the_end,
                  "curve": curve, "on_validation_answers": measured}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)

    def save(learned):
        notes = {"draft": args.draft, "answers": learned, "rounds": teach.rounds, "predictor": args.predictor,
                 "predictor_trained_too": not args.freeze_predictor, "curve_end": curve[-1] if curve else None}
        save_restart_layers(os.path.join(args.adapter_dir, f"warm-{args.save}.pt"), second, notes)
        if not args.freeze_predictor:
            save_predictor(os.path.join(args.adapter_dir, f"restart-{args.save}.pt"), predictor, notes)

    learned, saved_at = 0, time.time()
    for name, index, sequence, prompt_length, source in [entry for _ in range(max(args.passes, 1)) for entry in answers]:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and learned >= args.max_answers):
            break
        if len(sequence) - prompt_length < 2 or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        second.train()
        predictor.train(not args.freeze_predictor)
        ids = sequence.long().to(device)[None]
        features = target_features(target, draft, ids).clone()           # a copy that learning can go through
        teach.learn_from(ids, features, prompt_length)
        del features
        learned += 1
        point = teach.progress()
        if point:
            curve.append({"answers": learned, **point})
        report(False, learned)
        if time.time() - saved_at > 1800:                                # what is learned so far outlives a job that is cut off
            save(learned)
            saved_at = time.time()

    second.eval()
    predictor.eval()
    save(learned)
    teach.compare_every, teach.rounds, measured, point = 1, 0, 0, None
    for name, index, sequence, prompt_length, source in checks:
        if measured >= args.check_answers:
            break
        if len(sequence) - prompt_length < 2 or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        ids = sequence.long().to(device)[None]
        features = target_features(target, draft, ids).clone()
        teach.learn_from(ids, features, prompt_length, learn=False)
        del features
        measured += 1
    point = teach.progress()
    report(True, learned, {"answers": measured, **(point or {})})


if __name__ == "__main__":
    main()
