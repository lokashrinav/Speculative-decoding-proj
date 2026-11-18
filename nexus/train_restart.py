"""Train a small predictor of the target's states that a restart of the drafter does not have yet.

    python nexus/train_restart.py --data-dir ~/specdec-data/long1 --results-dir nexus/results/NAME --save v0

A second run that restarts the drafter after a break (dflash/redraft.py: Restarts) lacks the
target's hidden states of this round's first token and of the draft tokens before the break: the
target has not read them yet. The predictor (dflash/stand_in.py) writes stand-ins for them, in the
form the drafter reads states (after its fc and hidden_norm), from what exists at that moment: the
drafter's own final state at that token, the token, and the last real vector before the round.

Training needs no stored data. Saved answers of the training part are replayed with the first
draft alone. In every round the first token and the accepted draft tokens are examples: their real
vectors are known from the saved answer, and the predictor learns to match them (mean squared
error). The weights are updated every --every rounds. This is --loss vectors.

--loss tokens (the default) trains on what the vectors are for. At a random place of a saved answer a round is
played as real decoding would play it: the drafter's first run from the round's first token, then
a restart some tokens later with the predictor's vectors in place of the target's states of the
tokens in between. The loss is how unlikely the drafter then finds the tokens the answer really
has after the restart (the first ones count most), plus --vector-weight times the error of the
vectors. The drafter itself is not changed. Nothing on the way is cached: each step is two short
drafter passes over the last --window tokens of the text.

Then the restart is measured as in nexus/restart_probe.py on validation answers: how many tokens
the target accepts after the right token at the break when the drafter restarts with the predicted
vectors, next to the real ones, none, the last one repeated, and the others. The predictor is saved
as restart-<name>.pt in --adapter-dir (about 18 MB).
"""
import argparse
import os
import time

import torch

from dflash.adapt import DraftTrainer, adapter_notes, attach_adapters, load_adapters
from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.model import _draft_value, _output_head, _raw_input_embeddings
from dflash.replay import _matches, replay_decode, target_features
from dflash.sharp import QueryScale
from dflash.stand_in import StatePredictor, as_context, predictor_inputs, project, save_predictor, state_inputs
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers
from nexus.restart_probe import Restart, summary


class Teach:
    """The replay's plug-in while training: makes no second draft; in every round it shows the
    predictor the round's first token and accepted draft tokens with their real vectors."""

    def __init__(self, predictor, optimizer, every=8, width=16):
        self.predictor, self.optimizer, self.every, self.width = predictor, optimizer, every, width
        self.rounds = self.examples = self.steps = 0
        self.recent = []            # [loss, loss of repeating the last real vector, likeness to the real vector]

    def begin(self, ids, features):
        self.ids, self.features = ids, features

    def start(self, first_drafter):
        pass

    def __call__(self, s):
        width = s.block_ids.shape[1]
        truth = self.ids[:, s.start + 1 : s.start + width]
        if width != self.width or truth.shape[1] != width - 1:
            return None
        count = _matches(s.block_ids[:, 1:], truth) + 1           # the block's first token and the accepted draft tokens
        coming = self.ids[0, s.start + count] if self.ids.shape[1] > s.start + count else None   # the right token at the break
        inputs = predictor_inputs(s, count, coming, int(self.predictor.sizes.get("memory", 0)))
        real = project(s.first_drafter, self.features[:, s.start : s.start + count])[0].float()
        with torch.inference_mode(False), torch.enable_grad():
            # (copies: what the replay computes cannot be used for learning as it is)
            inputs, real = [None if value is None else value.clone() for value in inputs], real.clone()
            last = inputs[3]
            written = as_context(s.first_drafter, self.predictor(*inputs))
            loss = torch.nn.functional.mse_loss(written, real)
            (loss / self.every).backward()
            self.rounds += 1
            self.examples += count
            if self.rounds % self.every == 0:
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.steps += 1
            with torch.no_grad():
                repeated = torch.nn.functional.mse_loss(last.expand_as(real), real)
                alike = torch.nn.functional.cosine_similarity(written, real, dim=-1).mean()
        self.recent.append([float(loss.detach()), float(repeated), float(alike)])
        return None

    def progress(self):
        """Means over the rounds since the last call."""
        recent, self.recent = self.recent, []
        if not recent:
            return None
        loss, repeated, alike = (sum(column) / len(recent) for column in zip(*recent))
        return {"rounds": self.rounds, "examples": self.examples, "steps": self.steps, "loss": round(loss, 5),
                "loss_of_repeating_the_last_vector": round(repeated, 5), "likeness_to_the_real_vector": round(alike, 4)}


def _block_at(drafter, target, ids, place, width=16):
    """The embeddings of a block of masks that starts with the answer's token at `place`."""
    block = torch.full((1, width), drafter.mask_token_id, dtype=ids.dtype, device=ids.device)
    block[0, 0] = ids[0, place]
    return _raw_input_embeddings(target, block, float(_draft_value(drafter.config, "input_embedding_scale", 1.0)))


def first_run_states(drafter, target, features, ids, start, low=0, width=16):
    """The drafter's final states of the block it drafts from the token at `start`, reading the
    target's states of ids[low:start]: [width, size]."""
    positions = torch.arange(low, start + width, device=ids.device)[None]
    return drafter(target_hidden=features[:, low:start], noise_embedding=_block_at(drafter, target, ids, start, width),
                   position_ids=positions)[0]


def restart_logits(drafter, target, features, ids, start, written, low=0, width=16, anchor=None):
    """What the drafter predicts after the token at start + count when it restarts there, reading the
    target's states of ids[low:start] and `written` ([count, size], as the drafter reads states) for
    the tokens start .. start + count - 1: [width - 1, vocabulary]. Nothing is cached, so a predictor
    can learn through it; with the real vectors it is the first draft of a round that starts there.
    anchor: where the restart starts when `written` covers fewer tokens than are missing (none: [0, size])."""
    anchor = start + written.shape[0] if anchor is None else anchor
    positions = torch.cat((torch.arange(low, start + written.shape[0], device=ids.device),
                           torch.arange(anchor, anchor + width, device=ids.device)))[None]
    hidden = drafter(target_hidden=features[:, low:start], context_states=written[None],
                     noise_embedding=_block_at(drafter, target, ids, anchor, width), position_ids=positions)
    return drafter.compute_logits(hidden[:, 1:, :], _output_head(target))[0]


class TeachByTokens:
    """Trains the predictor on the tokens the drafter writes with its vectors (see the top of this file)."""

    # how many tokens of a round the target has not read at a restart: as often as the first draft breaks there
    MISSING = (0.31, 0.24, 0.14, 0.11, 0.08, 0.05, 0.04, 0.03)

    def __init__(self, predictor, optimizer, drafter, target, every=8, window=2048, vector_weight=0.1, decay=0.8, width=16, seed=0,
                 compare_every=4, where="breaks", earlier=True):
        self.predictor, self.optimizer, self.drafter, self.target = predictor, optimizer, drafter, target
        self.every, self.window, self.vector_weight, self.width = every, window, vector_weight, width
        # where="breaks": the restart is where the first draft of the round really breaks, as in use: the
        # drafter's own states then come from a draft that was right up to there and wrong right after.
        # ("random": any number of tokens, however the first draft did. The first predictor was trained
        # so; it closed 57% of the gap on such rounds and 21% at real breaks.)
        # earlier: the same vectors are also taught for the restart one token earlier (from the last
        # right token, which needs one vector fewer and no guess of the token at the break).
        self.where, self.earlier, self.unbroken = where, earlier, 0
        self.also = []                       # [tokens right from the break on, restarting one token earlier: predicted, real vectors]
        self.compare_every = compare_every   # every so many rounds the same restart is also made with the real vectors and with none
        self.compared = []                   # [tokens right with the predicted vectors, with the real ones, with none]
        self.weights = decay ** torch.arange(width - 1, dtype=torch.float32)
        self.generator = torch.Generator().manual_seed(seed)
        self.rounds = self.examples = self.steps = 0
        self.recent = []

    def learn_from(self, ids, features, prompt_length, most=600):
        """Random rounds of one saved answer. features: the target's states of the answer (a copy that
        was not made by the replay)."""
        last = ids.shape[1] - len(self.MISSING) - self.width - 1
        if last <= prompt_length:
            return
        if self.where == "breaks":
            last = ids.shape[1] - 2 * self.width - 1
            if last <= prompt_length:
                return
        for _ in range(min(most, max((ids.shape[1] - prompt_length) // 24, 1))):
            start = int(torch.randint(prompt_length, last, (1,), generator=self.generator))
            count = None
            if self.where != "breaks":
                count = int(torch.multinomial(torch.tensor(self.MISSING), 1, generator=self.generator)) + 1
            self.step(ids, features, start, count)

    def step(self, ids, features, start, count=None):
        """count: how many of the round's tokens the target has not read at the restart (None: as many
        as the first draft of a round that starts at `start` gets right, and its first token)."""
        low, device = max(start - self.window, 0), ids.device
        with torch.no_grad():
            block_states = first_run_states(self.drafter, self.target, features, ids, start, low, self.width)
            if count is None:
                first = self.drafter.compute_logits(block_states[None, 1:], _output_head(self.target))[0]
                wrong = (first.argmax(dim=-1) != ids[0, start + 1 : start + self.width]).nonzero()
                if not wrong.numel():
                    self.unbroken += 1          # the first draft is right to its end: nothing to restart
                    return
                count = int(wrong[0]) + 1
            scale = float(_draft_value(self.drafter.config, "input_embedding_scale", 1.0))
            tokens = _raw_input_embeddings(self.target, ids[:, start : start + count], scale)[0].float()
            # The token after each of them; after the last one it is the token the restart starts from.
            coming = _raw_input_embeddings(self.target, ids[:, start + 1 : start + count + 1], scale)[0].float()
            last = project(self.drafter, features[:, start - 1 : start])[0].float()
            real = project(self.drafter, features[:, start : start + count])[0].float()
            most = int(self.predictor.sizes.get("memory", 0))
            recalled = None
            if most:        # in use it is the context that is new in the round: as many vectors as the round before gave
                held = int(torch.randint(1, most + 1, (1,), generator=self.generator))
                recalled = project(self.drafter, features[:, max(start - held, low) : start])[0].float()
        written = as_context(self.drafter, self.predictor(*state_inputs(block_states, count), tokens, last,
                                                          torch.arange(count, device=device), coming, recalled))
        logits = restart_logits(self.drafter, self.target, features, ids, start, written, low, self.width).float()
        answer = ids[0, start + count + 1 : start + count + self.width].clone()
        each = torch.nn.functional.cross_entropy(logits, answer, reduction="none")
        weights = self.weights.to(device)
        by_tokens = (each * weights).sum() / weights.sum()
        by_vectors = torch.nn.functional.mse_loss(written, real)
        total = by_tokens + self.vector_weight * by_vectors
        before = None
        if self.earlier and count >= 2:
            # One token earlier: from the last right token, with the vectors of the tokens before it.
            before = restart_logits(self.drafter, self.target, features, ids, start, written[: count - 1], low, self.width).float()
            from_the_break = ids[0, start + count : start + count - 1 + self.width].clone()
            total = total + (torch.nn.functional.cross_entropy(before, from_the_break, reduction="none") * weights).sum() / weights.sum()
        (total / self.every).backward()
        self.rounds += 1
        self.examples += count
        if self.rounds % self.every == 0:
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.steps += 1
        with torch.no_grad():
            right = (logits.argmax(dim=-1) == answer).long().cumprod(dim=0).sum()
            alike = torch.nn.functional.cosine_similarity(written, real, dim=-1).mean()
            if self.compare_every and self.rounds % self.compare_every == 0:
                # The same restart with the real vectors (the limit) and with none: what the predictor is between.
                others = [restart_logits(self.drafter, self.target, features, ids, start, given.to(written.dtype), low, self.width,
                                         anchor=start + count) for given in (real, real[:0])]
                self.compared.append([float(right)] + [float((other.argmax(dim=-1) == answer).long().cumprod(dim=0).sum())
                                                       for other in others])
                if before is not None:
                    limit = restart_logits(self.drafter, self.target, features, ids, start, real[: count - 1].to(written.dtype),
                                           low, self.width)
                    self.also.append([float((given.argmax(dim=-1) == from_the_break).long().cumprod(dim=0).sum())
                                      for given in (before, limit)])
        self.recent.append([float(by_tokens.detach()), float(by_vectors.detach()), float(alike), float(right)])

    def progress(self):
        recent, self.recent = self.recent, []
        if not recent:
            return None
        by_tokens, by_vectors, alike, right = (sum(column) / len(recent) for column in zip(*recent))
        point = {"rounds": self.rounds, "examples": self.examples, "steps": self.steps, "loss_by_tokens": round(by_tokens, 4),
                 "loss": round(by_vectors, 5), "likeness_to_the_real_vector": round(alike, 4),
                 "tokens_right_after_the_restart": round(right, 3)}
        compared, self.compared = self.compared, []
        if compared:
            mine, limit, nothing = (sum(column) / len(compared) for column in zip(*compared))
            point["on_the_same_rounds"] = {"rounds": len(compared), "predicted": round(mine, 3), "real": round(limit, 3),
                                           "none": round(nothing, 3)}
        also, self.also = self.also, []
        if also:
            mine, limit = (sum(column) / len(also) for column in zip(*also))
            point["one_token_earlier"] = {"rounds": len(also), "predicted": round(mine, 3), "real": round(limit, 3)}
        return point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--save", required=True, help="the predictor is saved as restart-<name>.pt in --adapter-dir")
    parser.add_argument("--part", default="train", help="the answers it learns from")
    parser.add_argument("--fixed", default="", help="the drafter starts from the saved add-on weights lora-<name>.pt")
    parser.add_argument("--sharp", action="store_true")
    parser.add_argument("--rank", type=int, default=16, help="of the saved add-on weights")
    parser.add_argument("--loss", default="tokens", choices=("vectors", "tokens"),
                        help="match the real vectors, or the tokens the drafter writes with the predicted ones")
    parser.add_argument("--vector-weight", type=float, default=0.1, help="with --loss tokens: weight of the vectors' error")
    parser.add_argument("--window", type=int, default=2048, help="with --loss tokens: tokens of text the drafter reads in a step")
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--every", type=int, default=8, help="rounds per update of the weights")
    parser.add_argument("--inner", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--max-hours", type=float, default=3.0, help="of training; the measurement comes after")
    parser.add_argument("--max-answers", type=int, default=0, help="train on at most this many answers")
    parser.add_argument("--passes", type=int, default=1, help="times it goes through the answers (other rounds of them each time)")
    parser.add_argument("--where", default="breaks", choices=("breaks", "random"),
                        help="with --loss tokens: restart where the first draft really breaks (as in use), or anywhere")
    parser.add_argument("--only-after", action="store_true",
                        help="with --loss tokens: do not also teach the restart one token earlier")
    parser.add_argument("--start-from", default="", help="continue from the saved predictor restart-<name>.pt")
    parser.add_argument("--coming", action="store_true",
                        help="the predictor is told the token after each token (after the last: the restart's first token)")
    parser.add_argument("--layers", type=int, default=0,
                        help="layers in which the missing tokens look at the ones before them and at the last real vectors")
    parser.add_argument("--memory", type=int, default=8, help="with --layers: how many of the last real vectors it may look at")
    parser.add_argument("--longest", type=int, default=0, help="skip answers longer than this many tokens")
    parser.add_argument("--check-answers", type=int, default=12, help="validation answers the restart is measured on afterwards")
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
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    adapters = attach_adapters(draft, rank=args.rank) if args.fixed else []
    for adapter in adapters:
        adapter.fast = True
    notes, state = {}, None
    if args.fixed:
        path = os.path.join(args.adapter_dir, f"lora-{args.fixed}.pt")
        state, notes = load_adapters(path, adapters), adapter_notes(path)
    first = DraftTrainer(draft, adapters, rate=0.0, init=state, sharpen=QueryScale() if args.sharp else None)
    with torch.inference_mode(False):
        if args.start_from:
            from dflash.stand_in import load_predictor

            predictor, _ = load_predictor(os.path.join(args.adapter_dir, f"restart-{args.start_from}.pt"), device)
            predictor = predictor.float().train()
        else:
            predictor = StatePredictor(draft.config.hidden_size, args.inner, args.hidden, block=16, coming=args.coming,
                                       layers=args.layers, memory=args.memory if args.layers else 0).to(device)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.rate, weight_decay=0.0)
    for parameter in list(draft.parameters()) + list(target.parameters()):
        parameter.requires_grad_(False)
    teach = Teach(predictor, optimizer, args.every) if args.loss == "vectors" else \
        TeachByTokens(predictor, optimizer, draft, target, args.every, args.window, args.vector_weight, seed=args.seed,
                      where=args.where, earlier=not args.only_after)
    curve, started = [], time.time()

    def report(done, learned, measured=None):
        status = {"done": done, "answers": learned, "answers_to_learn_from": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "draft": args.draft, "fixed": args.fixed,
                  "sharp": args.sharp, "saved_adapters_used": notes, "save": args.save, "rate": args.rate, "loss": args.loss,
                  "every": args.every, "parameters": sum(value.numel() for value in predictor.parameters()),
                  "vector_weight": args.vector_weight, "where": args.where, "one_token_earlier_too": not args.only_after,
                  "passes": args.passes, "start_from": args.start_from, "predictor": dict(predictor.sizes),
                  "curve": curve, "restart_after_the_break": measured}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)

    def decode(sequence, prompt_length, plug_in, begin):
        generated = sequence[prompt_length:].tolist()
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        features = target_features(target, draft, ids)
        first.begin()
        begin(ids, features, None if ended else loop_start(generated))
        replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=plug_in, block_size=16)

    learned = 0
    for name, index, sequence, prompt_length, source in [entry for _ in range(max(args.passes, 1)) for entry in answers]:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and learned >= args.max_answers):
            break
        if len(sequence) - prompt_length < 2 or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        predictor.train()
        if args.loss == "vectors":
            decode(sequence, prompt_length, teach, lambda ids, features, loop: teach.begin(ids, features))
        else:
            ids = sequence.long().to(device)[None]
            first.begin()
            features = target_features(target, draft, ids).clone()       # a copy that learning can go through
            teach.learn_from(ids, features, prompt_length)
            del features
        learned += 1
        point = teach.progress()
        if point:
            curve.append({"answers": learned, **point})
        report(False, learned)

    predictor.eval()
    save_predictor(os.path.join(args.adapter_dir, f"restart-{args.save}.pt"), predictor,
                   {"draft": args.draft, "fixed": args.fixed, "answers": learned, "rounds": teach.rounds,
                    "curve_end": curve[-1] if curve else None})
    probe, measured = Restart(predictor=predictor, reanchor=True), 0
    for number, (name, index, sequence, prompt_length, source) in enumerate(checks):
        if measured >= args.check_answers:
            break
        if len(sequence) - prompt_length < 2 or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        decode(sequence, prompt_length, probe,
               lambda ids, features, loop, number=number: probe.begin(ids, prompt_length, features, loop, number))
        probe.features = None
        measured += 1
        report(False, learned, {"answers": measured, **summary(probe.rows, probe.columns)})
    report(True, learned, {"answers": measured, **summary(probe.rows, probe.columns)})


if __name__ == "__main__":
    main()
