"""Hindsight memory of the drafter's own drafts: the first-week probe.

During one answer the drafter's state at every drafted position is stored. Once the
answer has moved past that position, the entry is labelled with the token the target
actually committed there. The label is exact even for positions after the first
rejection, because the drafter's state at a position depends only on the committed
context, not on its own draft tokens.

MemoryProbe plugs into dflash_generate_redraft and never changes a draft. It records
what the memory would have said in every round, and when the answer is finished (so
the target's token at every drafted position is known) it scores that exactly:

  D1  at the first rejected token: was the memory's label the target's token, with
      that token among the first pass's top 16 candidates?
  D2  accepted tokens per round, by position in the answer
  D3  the same question for memories that look only at tokens: the previous token,
      and the longest run of up to four previous tokens seen before in this request
  D4  how many rounds a position after the first rejection waits for its label
  D5  tokens gained if a memory were used exactly when it is right
  net tokens gained or lost by simple rules that use a memory whenever it is similar
      and unanimous enough (a rule loses tokens when it breaks a correct token)

Two readings of the hidden-state memory are scored side by side: "raw" compares the
drafter's states as they are, "centered" first subtracts the average state of this
request, which removes what all states of one answer have in common.

All counts are kept twice: over every round, and over "clean" rounds only, which
leaves out the part of an answer that fell into endless repetition.

Optionally the probe also runs an editor on each draft (for example the trained
left-to-right editor) without using its result, and scores what a memory could still
add to the edited draft. With dump=True, finish() also returns everything recorded
per round (drafts, candidates with their probabilities, the nearest stored states,
the token memories' guesses, the target's tokens), so that gates can be fitted later
without decoding again.
"""

import time
from collections import defaultdict

import torch

BIN_EDGES = (0, 1000, 2000, 4000, 8000)  # by position of the round's start in the answer
BIN_NAMES = ("0-1k", "1k-2k", "2k-4k", "4k-8k", "8k+")
VARIANTS = ("raw", "centered")
MEMORIES = VARIANTS + ("previous", "suffix")
# A hidden-state memory acts when its best neighbour is at least this similar ...
SIMILARITY_STEPS = {"raw": (0.8, 0.9, 0.95), "centered": (0.5, 0.7, 0.8, 0.9)}
AGREEMENT_STEPS = (0.0, 0.6, 0.9)  # ... and its neighbours agree at least this much ...
SURE_STEPS = (1.0, 0.8, 0.6, 0.4)  # ... and the drafter gave its own token at most this probability
PREVIOUS_RULES = ((0.6, 2), (0.8, 3), (0.9, 5))  # (share of the commonest successor, sightings)
SUFFIX_RULES = ((3, 0.0), (2, 0.8), (3, 0.8), (4, 0.8))  # (matched tokens, share)


def rule_list():
    """(memory, first limit, second limit, the drafter's probability limit)"""
    rules = [(variant, similarity, agreement, sure) for variant in VARIANTS
             for similarity in SIMILARITY_STEPS[variant] for agreement in AGREEMENT_STEPS for sure in SURE_STEPS]
    rules += [("previous", share, seen, sure) for share, seen in PREVIOUS_RULES for sure in SURE_STEPS]
    rules += [("suffix", length, share, sure) for length, share in SUFFIX_RULES for sure in SURE_STEPS]
    return rules


def column_names():
    """What each number in a row of counts means (one row per position bin)."""
    names = ["rounds", "tokens", "first_rejections", "target_in_top", "memory_nonempty"]
    for variant in VARIANTS:
        names += [f"{variant}|label_is_target", f"{variant}|rescue", f"{variant}|oracle_tokens",
                  f"{variant}|rescue_previous_misses", f"{variant}|rescue_suffix_misses"]
    for memory in ("previous", "suffix"):
        names += [f"{memory}|rescue", f"{memory}|oracle_tokens"]
    names += ["raw_or_suffix|oracle_tokens", "centered_or_suffix|oracle_tokens"]
    # What is left once an editor has edited the draft (all zero when the probe runs no editor).
    names += ["editor|tokens", "editor|first_rejections", "editor|target_in_top"]
    for memory in MEMORIES:
        names += [f"after_editor|{memory}|rescue", f"after_editor|{memory}|oracle_tokens"]
    names += ["after_editor|centered_or_suffix|oracle_tokens"]
    for rule in rule_list():
        tag = "|".join(str(part) for part in rule)
        names += [f"{tag}|rescues", f"{tag}|breaks", f"{tag}|net_tokens"]
    return names


def accepted_length(tokens, truth):
    """Leading tokens of each row that equal the target's tokens."""
    return (tokens == truth).long().cumprod(dim=1).sum(dim=1)


def loop_start(tokens, window=8, share=0.95, shortest=256):
    """Where an answer that never ended fell into repetition: the earliest index from
    which at least `share` of the tokens only complete a run of `window` tokens that
    already occurred, through to the end. None if there is no such stretch."""
    seen, repeats = set(), []
    for index in range(len(tokens)):
        if index + 1 < window:
            repeats.append(False)
            continue
        run = tuple(tokens[index + 1 - window : index + 1])
        repeats.append(run in seen)
        seen.add(run)
    found, count = None, 0
    for index in range(len(tokens) - 1, -1, -1):
        count += repeats[index]
        length = len(tokens) - index
        if length >= shortest and count >= share * length:
            found = index
    return found


class SuffixMemory:
    """Token-only memory of one request: which token followed each run of 1 to
    `longest` tokens so far (the prompt included)."""

    def __init__(self, longest=4):
        self.longest = longest
        self.tables = [dict() for _ in range(longest)]
        self.tokens = []

    def extend(self, new_tokens):
        for token in new_tokens:
            position = len(self.tokens)
            self.tokens.append(token)
            for length in range(1, min(self.longest, position) + 1):
                counts = self.tables[length - 1].setdefault(tuple(self.tokens[position - length : position]), {})
                counts[token] = counts.get(token, 0) + 1

    def successor(self, context, lengths):
        """The commonest token after the longest of `lengths` trailing runs of
        `context` seen before: (token, share, sightings, run length), or (-1, 0, 0, 0)."""
        for length in lengths:
            if length > len(context):
                continue
            counts = self.tables[length - 1].get(tuple(context[len(context) - length :]))
            if counts:
                best = max(counts, key=counts.get)
                total = sum(counts.values())
                return best, counts[best] / total, total, length
        return -1, 0.0, 0, 0

    def counts(self, context, candidates):
        """For every run length 1..longest: how often each candidate followed that trailing run
        of `context`, and how often the run occurred at all."""
        rows, totals = [], []
        for length in range(1, self.longest + 1):
            counts = self.tables[length - 1].get(tuple(context[len(context) - length :])) if length <= len(context) else None
            rows.append([counts.get(candidate, 0) for candidate in candidates] if counts else [0] * len(candidates))
            totals.append(sum(counts.values()) if counts else 0)
        return rows, totals


class MemoryProbe:
    """Plug-in for dflash_generate_redraft that only measures (it returns no draft).
    Call begin() before each answer and finish(output_ids, ended) after it."""

    replace = False

    def __init__(self, capacity=32768, neighbours=8, temperature=0.05, top_k=16, longest=4, editor=None, dump=False,
                 dump_neighbours=16):
        self.capacity, self.neighbours, self.temperature = capacity, neighbours, temperature
        self.kept = max(neighbours, dump_neighbours) if dump else neighbours  # nearest states handed back per query
        self.top_k, self.longest = top_k, longest
        self.editor, self.dump = editor, dump  # an improver whose draft is only scored, never used
        self.columns = column_names()
        self.totals = {}                 # "all" and "clean": sums over every answer so far, (bins, columns)
        self.delays = defaultdict(int)   # D4: rounds waited -> number of positions
        self.read_seconds = defaultdict(float)  # time spent reading the memory, by how full it was
        self.read_count = defaultdict(int)
        self.begin()

    def start(self, first_drafter):
        pass

    def begin(self):
        """Empty the memory for a new answer."""
        self.keys = None           # drafter states, as they are
        self.size = 0
        self.pointer = 0
        self.added = 0
        self.total = None          # sum of every state stored so far (for the average state)
        self.pending = []          # drafted positions waiting for their label
        self.log = []              # one entry per round, scored in finish()
        self.suffix = SuffixMemory(self.longest)
        self.round = 0
        self.prompt_length = None
        self.last_start = None

    # ---- memory -----------------------------------------------------------------------------

    def _add(self, keys, labels, positions, offsets, guesses):
        if self.keys is None:
            self.keys = torch.zeros((self.capacity, keys.shape[1]), dtype=torch.float32, device=keys.device)
            self.squares = torch.zeros((self.capacity,), dtype=torch.float32, device=keys.device)
            self.labels = torch.full((self.capacity,), -1, dtype=torch.long, device=keys.device)
            self.positions = torch.full((self.capacity,), -1, dtype=torch.long, device=keys.device)
            self.offsets = torch.zeros((self.capacity,), dtype=torch.long, device=keys.device)
            self.guesses = torch.full((self.capacity,), -1, dtype=torch.long, device=keys.device)
            self.total = torch.zeros((keys.shape[1],), dtype=torch.float64, device=keys.device)
        slots = (self.pointer + torch.arange(keys.shape[0], device=keys.device)) % self.capacity
        self.keys[slots] = keys            # once full, the oldest entries are overwritten
        self.squares[slots] = (keys * keys).sum(dim=1)
        self.labels[slots] = labels
        self.positions[slots] = positions
        self.offsets[slots] = offsets      # how far ahead of its round's start the state was drafted
        self.guesses[slots] = guesses      # what the drafter drafted there (a mistake if it differs from the label)
        self.total += keys.double().sum(dim=0)
        self.added += keys.shape[0]
        self.pointer = int((self.pointer + keys.shape[0]) % self.capacity)
        self.size = min(self.size + keys.shape[0], self.capacity)

    def _similarities(self, queries):
        """Cosine similarity of each query to each stored state, for both readings."""
        keys = self.keys[: self.size]
        products = queries @ keys.t()
        key_squares = self.squares[: self.size]
        query_squares = (queries * queries).sum(dim=1)
        mean = (self.total / self.added).float()
        key_mean, query_mean, mean_square = keys @ mean, queries @ mean, mean @ mean
        raw = products / (query_squares[:, None] * key_squares[None, :]).clamp_min(1e-12).sqrt()
        lengths = (query_squares - 2 * query_mean + mean_square)[:, None] * (key_squares - 2 * key_mean + mean_square)[None, :]
        centered = (products - key_mean[None, :] - query_mean[:, None] + mean_square) / lengths.clamp_min(1e-12).sqrt()
        return {"raw": raw, "centered": centered}

    def _read(self, queries):
        """For each query and each reading: the label carrying the most weight among
        the nearest stored states, that weight (agreement), the highest similarity,
        and the nearest states themselves (label, similarity, position, offset, the drafter's guess then)."""
        count, device = queries.shape[0], queries.device
        if self.size == 0:
            empty = torch.full((count, self.kept), -1, dtype=torch.long, device=device)
            nothing = (torch.full((count,), -1, dtype=torch.long, device=device), torch.zeros(count, device=device),
                       torch.zeros(count, device=device), (empty, torch.zeros((count, self.kept), device=device), empty, empty, empty))
            return {variant: nothing for variant in VARIANTS}
        out = {}
        for variant, similarity in self._similarities(queries).items():
            top = similarity.topk(min(self.kept, self.size), dim=1)
            all_labels = self.labels[top.indices]
            # The memory's label is decided by the nearest `neighbours` states only.
            values, labels = top.values[:, : self.neighbours], all_labels[:, : self.neighbours]
            weights = torch.softmax(values / self.temperature, dim=1)
            same = labels[:, :, None] == labels[:, None, :]
            weight_of_label = (same * weights[:, None, :]).sum(dim=-1)
            best = weight_of_label.argmax(dim=1, keepdim=True)
            missing = self.kept - all_labels.shape[1]  # fewer stored states than neighbours asked for
            pad = lambda items, fill: torch.nn.functional.pad(items, (0, missing), value=fill)
            nearest = (pad(all_labels, -1), pad(top.values, 0.0), pad(self.positions[top.indices], -1),
                       pad(self.offsets[top.indices], -1), pad(self.guesses[top.indices], -1))
            out[variant] = (labels.gather(1, best)[:, 0], weight_of_label.gather(1, best)[:, 0], values[:, 0], nearest)
        return out

    # ---- one round --------------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, round_state):
        s = round_state
        start = int(s.start)
        anchor = int(s.block_ids[0, 0])
        if self.prompt_length is None:
            self.prompt_length = start
            fresh = s.accepted_ids[0].tolist() + [anchor]                      # positions 0 .. start
        else:
            fresh = s.accepted_ids[0, self.last_start + 1 :].tolist() + [anchor]  # positions last_start + 1 .. start
        self._label_pending(s, start)
        self.suffix.extend(fresh)

        states = s.draft_states[0, 1:].float()
        draft = s.block_ids[0, 1:].clone()
        log_probs = s.draft_logits[0].float().log_softmax(dim=-1)
        best = log_probs.topk(min(self.top_k, log_probs.shape[-1]), dim=-1)
        entry = dict(start=start, draft=draft, memory_size=self.size, top=best.indices, top_log_probs=best.values,
                     entropy=-(log_probs.exp() * log_probs).sum(dim=-1))
        if self.editor is not None:
            edited = self.editor(s)
            entry["edited"] = draft if edited is None else edited[0, 1:].clone()
        clock = self._clock(states)
        entry.update(self._read(states))
        filled = "under_4k" if self.size < 4096 else "4k_to_16k" if self.size < 16384 else "16k_and_more"
        self.read_seconds[filled] += self._clock(states) - clock
        self.read_count[filled] += 1
        tail = self.suffix.tokens[-self.longest :]
        context = tail + draft.tolist()
        longest_first = range(self.longest, 0, -1)
        entry["previous"] = [self.suffix.successor(context[: len(tail) + offset], (1,)) for offset in range(len(draft))]
        entry["suffix"] = [self.suffix.successor(context[: len(tail) + offset], longest_first) for offset in range(len(draft))]
        if self.dump:  # how often each candidate followed the same run of tokens earlier in this request
            candidates = best.indices.tolist()
            entry["suffix_counts"] = [self.suffix.counts(context[: len(tail) + offset], candidates[offset]) for offset in range(len(draft))]
        self.log.append(entry)
        self.pending.append(dict(keys=states, guesses=draft, first=start + 1, count=states.shape[0], start=start))
        self.last_start = start
        self.round += 1
        return None

    @staticmethod
    def _clock(tensor):
        if tensor.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _label_pending(self, s, start):
        """Give every drafted position the answer has now passed its committed token."""
        waiting = []
        committed = None
        for batch in self.pending:
            ready = min(batch["count"], start - batch["first"] + 1)
            if ready <= 0:
                waiting.append(batch)
                continue
            if committed is None:
                committed = torch.cat((s.accepted_ids[0], s.block_ids[0, :1]))  # positions 0 .. start
            positions = torch.arange(batch["first"], batch["first"] + ready, device=committed.device)
            assert batch["first"] + ready - 1 <= start, "a label was read before the answer reached its position"
            self._add(batch["keys"][:ready], committed[positions], positions, positions - batch["start"] - 1,
                      batch["guesses"][:ready])
            if ready < batch["count"]:
                waiting.append(dict(keys=batch["keys"][ready:], guesses=batch["guesses"][ready:], first=batch["first"] + ready,
                                    count=batch["count"] - ready, start=batch["start"]))
        self.pending = waiting

    # ---- scoring a finished answer ----------------------------------------------------------

    @torch.no_grad()
    def finish(self, output_ids, ended=True):
        """Score the answer now that the target's token at every position is known.
        `ended` says whether the answer stopped by itself (if not, a tail of endless
        repetition is looked for and kept out of the "clean" counts). Returns this
        answer's clean counts per position bin and where the repetition began."""
        final = output_ids[0]
        answer = {"loop_start": None, "bins": {}}
        if not self.log:
            return answer
        if not ended:
            answer["loop_start"] = loop_start(final[self.prompt_length :].tolist())
        device = final.device
        rounds = len(self.log)
        width = max(entry["draft"].shape[0] for entry in self.log)

        def stack(values, fill):
            if all(value.shape[0] == width for value in values):
                return torch.stack(values)
            out = values[0].new_full((len(values), width) + tuple(values[0].shape[1:]), fill)
            for row, value in enumerate(values):
                out[row, : value.shape[0]] = value
            return out

        def from_lists(name, part, fill, dtype):
            rows = [[item[part] for item in entry[name]] + [fill] * (width - len(entry[name])) for entry in self.log]
            return torch.tensor(rows, dtype=dtype, device=device)

        starts = torch.tensor([entry["start"] for entry in self.log], dtype=torch.long, device=device)
        offsets = torch.arange(width, device=device)
        positions = starts[:, None] + 1 + offsets[None, :]
        draft = stack([entry["draft"] for entry in self.log], -2)
        valid = (positions < final.shape[0]) & (draft >= 0)
        truth = torch.where(valid, final[positions.clamp(max=final.shape[0] - 1)], torch.full_like(positions, -1))
        top = stack([entry["top"] for entry in self.log], -3)
        target_in_top = (top == truth[..., None]).any(dim=-1)
        plain = accepted_length(draft, truth)
        rejected = plain < valid.sum(dim=1)
        at = plain.clamp(max=width - 1)[:, None]  # offset of the first rejected token

        def here(values):  # the value at each round's first rejected token
            return values.gather(1, at)[:, 0]

        def gain_of(edited):
            return accepted_length(edited, truth) - plain

        def used_when_right(proposal):
            return torch.where((proposal == truth) & target_in_top, truth, draft)

        proposals, right_here, columns = {}, {}, []
        memory_size = torch.tensor([entry["memory_size"] for entry in self.log], device=device)
        columns += [torch.ones(rounds, device=device), plain + 1, rejected, rejected & here(target_in_top),
                    rejected & (memory_size > 0)]
        proposals["previous"] = from_lists("previous", 0, -1, torch.long)
        proposals["suffix"] = from_lists("suffix", 0, -1, torch.long)
        for variant in VARIANTS:
            proposals[variant] = stack([entry[variant][0] for entry in self.log], -1)
        for name, proposal in proposals.items():
            right_here[name] = rejected & (here(proposal) == here(truth)) & here(target_in_top)
        for variant in VARIANTS:
            columns += [rejected & (here(proposals[variant]) == here(truth)), right_here[variant],
                        gain_of(used_when_right(proposals[variant])),
                        right_here[variant] & ~right_here["previous"], right_here[variant] & ~right_here["suffix"]]
        for memory in ("previous", "suffix"):
            columns += [right_here[memory], gain_of(used_when_right(proposals[memory]))]
        for variant in VARIANTS:
            either = ((proposals[variant] == truth) | (proposals["suffix"] == truth)) & target_in_top
            columns.append(gain_of(torch.where(either, truth, draft)))

        # What a memory could still add once an editor has edited the draft.
        edited = None
        if "edited" in self.log[0]:
            edited = stack([entry["edited"] for entry in self.log], -2)
            after = accepted_length(edited, truth)
            still_rejected = after < valid.sum(dim=1)
            there = after.clamp(max=width - 1)[:, None]
            at_editor = lambda values: values.gather(1, there)[:, 0]
            columns += [after + 1, still_rejected, still_rejected & at_editor(target_in_top)]
            for name in MEMORIES:
                right = (proposals[name] == truth) & target_in_top
                columns += [still_rejected & at_editor(right), accepted_length(torch.where(right, truth, edited), truth) - after]
            either = ((proposals["centered"] == truth) | (proposals["suffix"] == truth)) & target_in_top
            columns.append(accepted_length(torch.where(either, truth, edited), truth) - after)
        else:
            columns += [torch.zeros(rounds, device=device)] * (3 + 2 * len(MEMORIES) + 1)

        measures = {}
        for variant in VARIANTS:
            measures[variant] = (stack([entry[variant][2] for entry in self.log], 0.0),
                                 stack([entry[variant][1] for entry in self.log], 0.0))
        measures["previous"] = (from_lists("previous", 1, 0.0, torch.float32), from_lists("previous", 2, 0, torch.float32))
        measures["suffix"] = (from_lists("suffix", 3, 0, torch.float32), from_lists("suffix", 1, 0.0, torch.float32))
        in_top = {name: (top == proposal[..., None]).any(dim=-1) & (proposal >= 0) for name, proposal in proposals.items()}
        sure = stack([entry["top_log_probs"][:, 0] for entry in self.log], 0.0).exp()  # the drafter's probability of its own token
        for name, first_limit, second_limit, sure_limit in rule_list():
            first, second = measures[name]
            acts = in_top[name] & (first >= first_limit) & (second >= second_limit) & (sure <= sure_limit)
            gain = gain_of(torch.where(acts, proposals[name], draft))
            columns += [gain > 0, gain < 0, gain]

        values = torch.stack([column.double() for column in columns], dim=1)
        assert values.shape[1] == len(self.columns)
        edges = torch.tensor(BIN_EDGES[1:], device=device)
        bins = torch.bucketize(starts - self.prompt_length, edges, right=True)
        member = torch.nn.functional.one_hot(bins, len(BIN_NAMES)).double()
        clean = torch.ones(rounds, dtype=torch.bool, device=device)
        if answer["loop_start"] is not None:
            clean = starts < self.prompt_length + answer["loop_start"]
        sums = {"all": (member.t() @ values).cpu(), "clean": ((member * clean[:, None]).t() @ values).cpu()}
        for kind, table in sums.items():
            self.totals[kind] = table if kind not in self.totals else self.totals[kind] + table
        for index in range(len(BIN_NAMES)):
            if sums["clean"][index, 0] > 0:
                answer["bins"][str(index)] = [int(round(value)) for value in sums["clean"][index].tolist()]

        # D4: a position drafted in round r gets its label in the first later round that starts at or after it.
        labelled_in = torch.searchsorted(starts, positions.reshape(-1).contiguous()).view(rounds, width)
        waits = labelled_in - torch.arange(rounds, device=device)[:, None]
        counted = valid & rejected[:, None] & (offsets[None, :] > plain[:, None]) & (labelled_in < rounds) & clean[:, None]
        for wait, count in enumerate(torch.bincount(waits[counted].clamp(max=10), minlength=11).tolist()):
            if count:
                self.delays[wait] += count

        if self.dump:
            small = lambda values: values.to(torch.int32).cpu()
            half = lambda values: values.to(torch.float16).cpu()
            record = {
                "starts": small(starts), "prompt_length": self.prompt_length, "memory_size": small(memory_size),
                "draft": small(draft), "truth": small(truth), "top_tokens": small(top),
                "top_log_probs": half(stack([entry["top_log_probs"] for entry in self.log], 0.0)),
                "entropy": half(stack([entry["entropy"] for entry in self.log], 0.0)),
            }
            if edited is not None:
                record["edited"] = small(edited)
            for variant in VARIANTS:
                parts = [stack([entry[variant][3][part] for entry in self.log], -1 if part != 1 else 0.0) for part in range(5)]
                record[f"{variant}_labels"], record[f"{variant}_similarities"] = small(parts[0]), half(parts[1])
                record[f"{variant}_positions"], record[f"{variant}_offsets"] = small(parts[2]), parts[3].to(torch.int8).cpu()
                record[f"{variant}_guesses"] = small(parts[4])
            for memory in ("previous", "suffix"):
                record[f"{memory}_token"] = small(proposals[memory])
                record[f"{memory}_share"] = half(from_lists(memory, 1, 0.0, torch.float32))
                record[f"{memory}_seen"] = small(from_lists(memory, 2, 0, torch.long))
            record["suffix_length"] = from_lists("suffix", 3, 0, torch.long).to(torch.int8).cpu()
            blank = ([[0] * top.shape[-1]] * self.longest, [0] * self.longest)
            cells = [entry["suffix_counts"] + [blank] * (width - len(entry["suffix_counts"])) for entry in self.log]
            record["suffix_candidate_counts"] = torch.tensor([[cell[0] for cell in row] for row in cells]).clamp(max=255).to(torch.uint8)
            record["suffix_run_counts"] = torch.tensor([[cell[1] for cell in row] for row in cells], dtype=torch.int32)
            answer["rounds"] = record
        return answer


GROUPS = [(name, [index]) for index, name in enumerate(BIN_NAMES)] + [("2k+", [2, 3, 4]), ("everything", [0, 1, 2, 3, 4])]


def summarize(totals, delays, columns=None):
    """Readable tables from the summed counts: "clean" (endless repetition left out)
    and "all", by position in the answer."""
    columns = columns or column_names()
    index = {name: number for number, name in enumerate(columns)}
    out = {}
    for kind in ("clean", "all"):
        table = totals.get(kind)
        if table is None:
            continue
        rows = {}
        for group, members in GROUPS:
            sums = table[members].sum(dim=0)
            get = lambda key: float(sums[index[key]])
            rounds, rejections, tokens = get("rounds"), get("first_rejections"), get("tokens")
            if not rounds:
                continue
            per_rejection = lambda key: get(key) / max(rejections, 1)
            more_tokens = lambda key: get(key) / max(tokens, 1)
            row = {
                "rounds": int(rounds),
                "tokens_per_round": tokens / rounds,
                "first_rejections": int(rejections),
                "target_token_in_top_16": per_rejection("target_in_top"),
                "D1_rescue_rate": {v: per_rejection(f"{v}|rescue") for v in VARIANTS},
                "D3_rescue_rate": {m: per_rejection(f"{m}|rescue") for m in ("previous", "suffix")},
                "rescued_where_previous_token_memory_fails": {v: per_rejection(f"{v}|rescue_previous_misses") for v in VARIANTS},
                "rescued_where_suffix_memory_fails": {v: per_rejection(f"{v}|rescue_suffix_misses") for v in VARIANTS},
                "D5_more_tokens_if_used_only_when_right": {
                    m: more_tokens(f"{m}|oracle_tokens")
                    for m in MEMORIES + ("raw_or_suffix", "centered_or_suffix")},
                "rules": {},
            }
            if get("editor|tokens"):
                edited_tokens, still = get("editor|tokens"), max(get("editor|first_rejections"), 1)
                row["with_editor"] = {
                    "tokens_per_round": edited_tokens / rounds,
                    "more_tokens_than_plain": edited_tokens / tokens - 1,
                    "first_rejections": int(get("editor|first_rejections")),
                    "target_token_in_top_16": get("editor|target_in_top") / still,
                    "rescue_rate": {m: get(f"after_editor|{m}|rescue") / still for m in MEMORIES},
                    "more_tokens_if_used_only_when_right": {
                        m: get(f"after_editor|{m}|oracle_tokens") / edited_tokens for m in MEMORIES + ("centered_or_suffix",)},
                }
            for rule in rule_list():
                tag = "|".join(str(part) for part in rule)
                row["rules"].setdefault(rule[0], []).append({
                    "limits": list(rule[1:]),
                    "rescues_per_rejection": per_rejection(f"{tag}|rescues"),
                    "breaks_per_rejection": per_rejection(f"{tag}|breaks"),
                    "net_per_rejection": (get(f"{tag}|rescues") - get(f"{tag}|breaks")) / max(rejections, 1),
                    "more_tokens": more_tokens(f"{tag}|net_tokens"),
                })
            row["best_rule"] = {name: max(rules, key=lambda rule: rule["more_tokens"]) for name, rules in row["rules"].items()}
            rows[group] = row
        out[kind] = rows
    waited = sum(delays.values())
    running, median = 0, None
    for wait in sorted(delays):
        running += delays[wait]
        if median is None and running * 2 >= waited:
            median = wait
    out["D4_rounds_waited_after_first_rejection"] = {
        "median": median, "counts": {str(wait): count for wait, count in sorted(delays.items())}}
    return out


def read_cost(probe):
    """Milliseconds per round spent reading both versions of the memory, by how full it was."""
    return {filled: {"rounds": count, "milliseconds_per_round": 1000 * probe.read_seconds[filled] / count}
            for filled, count in probe.read_count.items() if count}
