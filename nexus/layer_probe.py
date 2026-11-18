"""How many of the drafter's layers a restart needs, and what each part of the second run costs.

    python nexus/layer_probe.py --draft opd --predictor opd-m0 --data-dir ~/specdec-data/long1 --results-dir nexus/results/NAME

The second run of the drafter (dflash/redraft.py: Restarts) gives a third more tokens per round and
costs 14 ms a round, of which a bare pass of the drafter is 8: on top of a tree of drafts it does not
pay. A pass costs by the layers it runs and hardly by the restarts it holds. So the second run is
cheap only if it runs FEWER LAYERS. This probe replays saved answers with the first draft alone and,
in every round with a break, restarts the drafter from the RIGHT token at the break with each of
these plans (L: the drafter's number of layers):

    all          the whole drafter, as the second run does now
    topK         only its last K layers: the block of [token, masks] enters layer L - K + 1 as embeddings
    bottomK      only its first K layers (then the output head)
    withoutN     all layers but layer N
    warmK        its last K layers, GOING ON FROM THE FIRST RUN: the masks after the restart's first token
                 enter with the states the first run had at those same places after layer L - K. The first
                 run already read the whole context for these places; it only did not know the tokens
                 before them. The last K layers are then given what it lacked: the right token at the
                 break (as its embedding) and the states of the tokens before it. Nothing is trained.
    warmKp       the same, with the first token's place keeping the first run's state there too, the
                 mask's embedding in it replaced by the token's

each with three kinds of states for the tokens the target has not read: real (the limit, replay
only), predicted (--predictor) and none. A warm plan can only write up to the end of the first
block (the first run has no states beyond it), so the summary gives, for every plan, the tokens the
target accepts after the right token: all of them, at most 3 of them (the second run lets the
target check 3), and those inside the first block (what a warm plan can reach).

It also times the parts of the second run on the GPU, waiting for the GPU at every boundary
(--time-rounds samples, spread over the answers): the predictor, and for the whole drafter and its
last 1, 2 and 3 layers the preparing, the pass and the output head, with 4 restarts of 16 tokens,
12 of 5 and 12 of 16; the second run as real decoding calls it, as a whole; and the first draft's
pass made again. What a whole call takes beyond its parts is time spent in Python.
"""
import argparse
import os
import time

import numpy
import torch

from dflash.benchmark import load_transformers_models, stop_token_ids
from dflash.hindsight import loop_start
from dflash.model import _draft_value, _output_head, _raw_input_embeddings
from dflash.redraft import Restarts, restarts
from dflash.replay import _clock, _matches, replay_decode, target_features
from dflash.stand_in import load_predictor, predicted_states, project
from nexus.fit_gate import split_of
from nexus.probe_memory import stepped_aside, write_json
from nexus.redraft_long import find_drafter
from nexus.replay_memory import load_answers

WIDTH = 16
BASE = ("answer", "where", "broke_after", "second_choice_right", "old_rest")
SHAPES = (("4x16", 4, 16), ("12x5", 12, 5), ("12x16", 12, 16))        # restarts in a pass, tokens in each


def plans(count):
    """(name, the layers that run (None: all), how the block enters them (None: as embeddings))"""
    every = list(range(count))
    out = [("all", None, None)]
    for kept in range(count - 1, 0, -1):
        top = every[count - kept :]
        out += [(f"top{kept}", top, None), (f"warm{kept}", top, "raw"), (f"warm{kept}p", top, "patch"),
                (f"bottom{kept}", every[:kept], None)]
    for left in range(1, count - 1):
        out.append((f"without{left + 1}", every[:left] + every[left + 1 :], None))
    return out


class Layers:
    """The replay's plug-in: makes no second draft, measures the restarts of every round with a break."""

    def __init__(self, drafter, predictor=None, width=WIDTH, time_rounds=0, time_every=10):
        self.width, self.predictor, self.rows = width, predictor, []
        self.plans = plans(len(drafter.layers))
        self.kinds = ("real",) + (("predicted",) if predictor is not None else ()) + ("none",)
        self.columns = BASE + tuple(f"{name}_{kind}" for name, _, _ in self.plans for kind in self.kinds)
        # What each layer gave in the drafter's last pass. At the time the replay calls this plug-in
        # that pass is the round's first draft.
        self.outputs = {}
        self.hooks = [layer.register_forward_hook(self._keep(number)) for number, layer in enumerate(drafter.layers)]
        self.time_rounds, self.time_every, self.breaks, self.timings = time_rounds, time_every, 0, []
        stand_in = "predicted" if predictor is not None else "none"
        self.methods = {"restart4_swap_tail3": Restarts(k=4, stand_in=stand_in, swap=True, tail=3),
                        "restart_best12_of_3_choices_blocks_of_5": Restarts(k=4, stand_in=stand_in, swap=True, tail=3,
                                                                            choices=3, best=12, block=5)}
        for method in self.methods.values():
            method.predictor = predictor
            method.start(drafter)

    def _keep(self, number):
        def keep(layer, given, output):
            self.outputs[number] = output
        return keep

    def close(self):
        for hook in self.hooks:
            hook.remove()

    def begin(self, ids, prompt_length, features, repetition_from, answer):
        self.ids, self.prompt_length, self.features = ids, prompt_length, features
        self.repetition_from, self.answer = repetition_from, answer

    def start(self, first_drafter):
        pass

    def __call__(self, s):
        first_run = dict(self.outputs)                              # taken before any pass made here
        width, full = s.block_ids.shape[1], self.width
        where = s.start - self.prompt_length
        outside = self.repetition_from is not None and where >= self.repetition_from
        truth = self.ids[:, s.start + 1 : s.start + width]
        if width != full or truth.shape[1] != full - 1 or outside:
            return None
        broke = _matches(s.block_ids[:, 1:], truth)
        restart = s.start + broke + 1                               # the position of the token the first draft got wrong
        if broke == full - 1 or restart + full > self.ids.shape[1]:
            return None
        drafter, device = s.first_drafter, s.block_ids.device
        token = truth[0, broke]
        after = self.ids[:, restart + 1 : restart + full]          # what the answer has after that token
        first = s.draft_logits[0].float()
        record = {"answer": self.answer, "where": where, "broke_after": broke,
                  "second_choice_right": int(first[broke].topk(2).indices[1] == token)}
        correct = s.block_ids.clone()
        correct[0, broke + 1] = token
        record["old_rest"] = _matches(correct[:, broke + 2 : full], after[:, : full - broke - 2])

        missing = broke + 1                                         # tokens of this round the target has not read
        places = torch.arange(s.start, s.start + missing, device=device)
        given = {"real": (self.features[:, s.start : s.start + missing], places), "none": None}
        if self.predictor is not None:
            given["predicted"] = (predicted_states(self.predictor, s, missing, coming=token), places, True)
        stand_ins, many = [given[kind] for kind in self.kinds], len(self.kinds)
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        pair = torch.stack((token, torch.as_tensor(drafter.mask_token_id, dtype=token.dtype, device=device)))[None]
        token_in, mask_in = _raw_input_embeddings(s.target, pair, scale)[0]
        inside = full - missing                                     # places from the restart's first token to the first block's end
        for name, layers, enter in self.plans:
            reach, entering = full, None
            if enter is not None:
                if inside < 2:                                      # the break is at the block's last place: no place to go on from
                    record.update({f"{name}_{kind}": 0 for kind in self.kinds})
                    continue
                reach = inside
                states = first_run[layers[0] - 1][0, missing:full].clone()        # after the last layer that is left out
                states[0] = token_in.to(states.dtype) if enter == "raw" else states[0] + (token_in - mask_in).to(states.dtype)
                entering = states[None].expand(many, -1, -1)
            tokens = restarts(s, [token] * many, [restart] * many, stand_ins, reach, layers=layers, entering=entering)
            for number, kind in enumerate(self.kinds):
                record[f"{name}_{kind}"] = _matches(tokens[number : number + 1], after[:, : reach - 1])
        self.rows.append([record[name] for name in self.columns])

        self.breaks += 1
        if len(self.timings) < self.time_rounds and self.breaks % self.time_every == 0:
            self.time_it(s, token, restart, places)
        return None

    def time_it(self, s, token, restart, places):
        """Seconds of each part of a second run in this round (the clock waits for the GPU)."""
        drafter, device, count = s.first_drafter, s.block_ids.device, len(s.first_drafter.layers)
        sample = {"context": int(s.start), "missing": int(places.shape[0])}
        if self.predictor is not None:
            began = _clock(device)
            predicted_states(self.predictor, s, places.shape[0], coming=token)
            sample["predictor"] = _clock(device) - began
        # The states in the form the drafter reads them, as a predictor hands them over.
        given = (project(drafter, self.features[:, s.start : s.start + places.shape[0]]), places, True)
        tops = [("all", None)] + [(f"top{kept}", list(range(count - kept, count))) for kept in (3, 2, 1) if kept < count]
        for label, many, reach in SHAPES:
            for name, layers in tops:
                timers = {}
                began = _clock(device)
                restarts(s, [token] * many, [restart] * many, [given] * many, reach, layers=layers, timers=timers)
                sample[f"{label} {name}: whole"] = _clock(device) - began
                sample.update({f"{label} {name}: {part}": seconds for part, seconds in timers.items()})
        for label, method in self.methods.items():
            began = _clock(device)
            method(s)
            sample[f"second run as used, {label}"] = _clock(device) - began
        # The first draft's pass made again (with no new context to read): pass, cache, output head.
        began = _clock(device)
        block = torch.full_like(s.block_ids, drafter.mask_token_id)
        block[:, 0] = s.block_ids[:, 0]
        scale = float(_draft_value(drafter.config, "input_embedding_scale", 1.0))
        width = block.shape[1]
        hidden = drafter(target_hidden=s.context_features[:, :0], noise_embedding=_raw_input_embeddings(s.target, block, scale),
                         position_ids=s.position_ids[:, s.start : s.start + width], past_key_values=s.draft_cache, use_cache=True)
        s.draft_cache.crop(-width)
        drafter.compute_logits(hidden[:, 1:, :], _output_head(s.target)).argmax(dim=-1)
        sample["first draft made again"] = _clock(device) - began
        self.timings.append(sample)


def summary(table, columns, width=WIDTH):
    """For every plan and kind of states, the tokens the target accepts after the right token at the
    break: [all of them, at most 3 of them, those inside the first block], over all rounds with a break
    and by how many of the target's states were missing."""
    columns = tuple(columns)
    table = numpy.asarray(table).reshape(-1, len(columns))
    out = {"rounds": int(len(table)), "each_entry": ["all tokens", "at most 3", "inside the first block"]}
    if not len(table):
        return out
    broke = table[:, columns.index("broke_after")]
    inside = width - 2 - broke                                     # tokens after the restart's first token, up to the block's end
    missing = broke + 1
    groups = (("all", missing > 0), ("1 missing", missing == 1), ("2 missing", missing == 2),
              ("3-4 missing", (missing >= 3) & (missing <= 4)), ("5 or more missing", missing >= 5))
    for label, mask in groups:
        if not mask.any():
            continue
        part = {"rounds": int(mask.sum())}
        for name in columns[columns.index("old_rest") :]:
            values = table[mask, columns.index(name)]
            part[name] = [round(float(values.mean()), 3), round(float(numpy.minimum(values, 3).mean()), 3),
                          round(float(numpy.minimum(values, inside[mask]).mean()), 3)]
        out[label] = part
    out["second_choice_right"] = round(float(table[:, columns.index("second_choice_right")].mean()), 3)
    return out


def timing_summary(samples):
    """Milliseconds of each part: [mean, median], over all samples but the first (which warms the GPU
    up), and for short and long contexts apart."""
    samples = samples[1:]
    out = {"samples": len(samples)}
    if not samples:
        return out
    out["context_tokens_mean"] = round(float(numpy.mean([sample["context"] for sample in samples])))
    names = [name for name in samples[0] if name not in ("context", "missing")]
    groups = (("all", lambda sample: True), ("context under 4000 tokens", lambda sample: sample["context"] < 4000),
              ("context of 4000 tokens or more", lambda sample: sample["context"] >= 4000))
    for label, wanted in groups:
        part = [sample for sample in samples if wanted(sample)]
        if part:
            out[label] = {"samples": len(part),
                          **{name: [round(1000 * float(numpy.mean([sample[name] for sample in part])), 3),
                                    round(1000 * float(numpy.median([sample[name] for sample in part])), 3)] for name in names}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16", help='a drafter, or "opd" for Draft-OPD')
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--part", default="validation")
    parser.add_argument("--predictor", default="", help="a saved predictor of the missing states: restart-<name>.pt in --adapter-dir")
    parser.add_argument("--adapter-dir", default=os.path.expanduser("~/specdec-ckpt"))
    parser.add_argument("--time-rounds", type=int, default=300, help="rounds in which the parts of the second run are timed")
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--max-answers", type=int, default=30)
    args = parser.parse_args()
    if stepped_aside(args.results_dir):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    answers = [answer for answer in load_answers(args.data_dir)
               if args.part == "all" or split_of([answer[0], answer[1]]) == args.part]
    status_path = os.path.join(args.results_dir, "status.json")
    write_json(status_path, {"done": False, "answers": 0}, indent=1)
    target, draft, tokenizer = load_transformers_models(args.model, find_drafter(args.draft), device)
    stop_set = set(stop_token_ids(target, tokenizer) or [])
    predictor, predictor_notes = None, {}
    if args.predictor:
        predictor, predictor_notes = load_predictor(os.path.join(args.adapter_dir, f"restart-{args.predictor}.pt"), device)
    probe = Layers(draft, predictor=predictor, time_rounds=args.time_rounds)
    records, started = [], time.time()

    def report(done):
        status = {"done": done, "answers": len(records), "part": args.part, "hours": round((time.time() - started) / 3600, 3),
                  "draft": args.draft, "drafter_layers": len(draft.layers), "predictor": args.predictor,
                  "predictor_notes": predictor_notes, "plans": [[name, layers, enter] for name, layers, enter in probe.plans],
                  "columns": probe.columns, "answers_read": records, "summary": summary(probe.rows, probe.columns),
                  "milliseconds": timing_summary(probe.timings)}
        if torch.cuda.is_available():
            status["gpu_memory_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        write_json(status_path, status, indent=1)
        if done:
            numpy.savez_compressed(os.path.join(args.results_dir, "rounds.npz"),
                                   rounds=numpy.asarray(probe.rows, dtype=numpy.int32), columns=numpy.asarray(probe.columns))
            write_json(os.path.join(args.results_dir, "timings.json"), probe.timings)

    for name, index, sequence, prompt_length, source in answers:
        if time.time() - started > args.max_hours * 3600 or len(records) >= args.max_answers:
            break
        generated = sequence[prompt_length:].tolist()
        if len(generated) < 2:
            continue
        ids = sequence.long().to(device)[None]
        ended = generated[-1] in stop_set
        repetition_from = None if ended else loop_start(generated)
        features = target_features(target, draft, ids)
        probe.begin(ids, prompt_length, features, repetition_from, len(records))
        result = replay_decode(draft, target, ids, prompt_length, ended=ended, features=features, improver=probe,
                               block_size=WIDTH)
        probe.features = None
        del features
        records.append({"source": source, "origin": [name, index], "tokens": len(generated), "rounds": int(result.rounds)})
        report(False)
    report(True)


if __name__ == "__main__":
    main()
