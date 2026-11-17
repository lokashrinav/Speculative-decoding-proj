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
error). The weights are updated every --every rounds.

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
from dflash.replay import _matches, replay_decode, target_features
from dflash.sharp import QueryScale
from dflash.stand_in import StatePredictor, as_context, predictor_inputs, project, save_predictor
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
        states, tokens, last, places = predictor_inputs(s, count)
        real = project(s.first_drafter, self.features[:, s.start : s.start + count])[0].float()
        with torch.inference_mode(False), torch.enable_grad():
            # (copies: what the replay computes cannot be used for learning as it is)
            states, tokens, last, real = states.clone(), tokens.clone(), last.clone(), real.clone()
            written = as_context(s.first_drafter, self.predictor(states, tokens, last, places.clone()))
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
        self.recent.append([float(loss), float(repeated), float(alike)])
        return None

    def progress(self):
        """Means over the rounds since the last call."""
        recent, self.recent = self.recent, []
        if not recent:
            return None
        loss, repeated, alike = (sum(column) / len(recent) for column in zip(*recent))
        return {"rounds": self.rounds, "examples": self.examples, "steps": self.steps, "loss": round(loss, 5),
                "loss_of_repeating_the_last_vector": round(repeated, 5), "likeness_to_the_real_vector": round(alike, 4)}


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
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--every", type=int, default=8, help="rounds per update of the weights")
    parser.add_argument("--inner", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--max-hours", type=float, default=3.0, help="of training; the measurement comes after")
    parser.add_argument("--max-answers", type=int, default=0, help="train on at most this many answers")
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
        predictor = StatePredictor(draft.config.hidden_size, args.inner, args.hidden, block=16).to(device)
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.rate, weight_decay=0.0)
    teach = Teach(predictor, optimizer, args.every)
    curve, started = [], time.time()

    def report(done, learned, measured=None):
        status = {"done": done, "answers": learned, "answers_to_learn_from": len(answers), "part": args.part,
                  "hours": round((time.time() - started) / 3600, 3), "draft": args.draft, "fixed": args.fixed,
                  "sharp": args.sharp, "saved_adapters_used": notes, "save": args.save, "rate": args.rate,
                  "every": args.every, "parameters": sum(value.numel() for value in predictor.parameters()),
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
    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or (args.max_answers and learned >= args.max_answers):
            break
        if len(sequence) - prompt_length < 2 or (args.longest and len(sequence) - prompt_length > args.longest):
            continue
        predictor.train()
        decode(sequence, prompt_length, teach, lambda ids, features, loop: teach.begin(ids, features))
        learned += 1
        point = teach.progress()
        if point:
            curve.append({"answers": learned, **point})
        report(False, learned)

    predictor.eval()
    save_predictor(os.path.join(args.adapter_dir, f"restart-{args.save}.pt"), predictor,
                   {"draft": args.draft, "fixed": args.fixed, "answers": learned, "rounds": teach.rounds,
                    "curve_end": curve[-1] if curve else None})
    probe, measured = Restart(predictor=predictor), 0
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
