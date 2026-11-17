# The restart second run: research plan (started November 17, 2026)

## The problem

Say the first draft is a b c d e f z, and z is wrong. A second run of the drafter should give the
target a better continuation: g, the alternative token, and good tokens after it.

Our study showed that the token is the easy part and the tokens after it are the hard part
([WHY_A_SECOND_DRAFT_HELPS.md](WHY_A_SECOND_DRAFT_HELPS.md)). With the right token at the break,
the first draft's old tokens give 0.55 more accepted tokens and a second pass inside the block
gives 0.64. The drafter in the next round gives 2.03, because by then it has the target's hidden
states of a b c d e f.

So the second run we build restarts the drafter at g, as the next round would: g h i j k. At that
moment the target has not read a b c d e f, so the drafter lacks the target's hidden states of
those few tokens. The research question is what to give it in their place.

## What a perfect answer is worth

A restart from the drafter's second choice at its four likely breaks, with the target's REAL hidden
states, is the limit of any stand-in. In accepted tokens per round, over the first draft alone:

| Drafter | The swap (no second run) | Restart with the real states (the limit) |
| --- | --- | --- |
| Public | +17.9% | +34.5% |
| Draft-OPD | +14.8% | +40.1% |
| Public with the add-on weights for long answers | +16.9% | +38.7% |

The swap is measured at 13% (public) and 10% (Draft-OPD) more real tokens per second. A restart
run costs about 13 ms on a round of 62 to 66 ms today. At that cost the limit works out to about
+11% (public) and +17% (Draft-OPD) in real speed. So two things decide whether this pays: how close
a stand-in gets to the real states, and how cheap the second run can be made.

## The experiments

**E1. Does the restart need the states at all?** (`nexus/restart_probe.py`, jobs `restart-pub-0`,
`restart-opd-0`.) In every round with a break the drafter is restarted from the right token with
the missing states real, left out, replaced by crude stand-ins (the last state repeated, older
states, a mean), or made noisy. Decision: if a stand-in that needs no training keeps at least
two thirds of what the real states give for early breaks, the second run needs no predictor and we
go to E4. Otherwise the noisy rows say how exact a predictor must be, and the rows with only the
nearest state or all but the nearest say which states matter.

**E2. The restart second run over whole answers.** (`Restarts` in `dflash/redraft.py`, jobs
`restart-replay-pub-0`, `restart-replay-opd-0`.) Tokens per round and estimated time for the
restart with the real states (the limit), with nothing, and with the last state repeated; alone and
together with the swap's rows; with a short check; on the add-on weights; next to the swap.

**E3. A small predictor of the missing states.** (`nexus/train_restart.py`.) The drafter does not
read the target's hidden states directly. It reads them after its own projection, one vector of
2,560 numbers per token. The predictor writes that vector. For a token of this round it reads
three things that exist at the time of the second run: the drafter's own final state at that token
from the first run, the token itself, and the last real vector before the round. It is trained on
saved answers, where the real vectors are known, first by matching them and then, if that is not
enough, by the tokens the drafter writes with them. Decision: continue if the restart with
predicted states closes at least half of the gap between leaving the states out and the real ones.

**E4. The whole second run in replay.** Restarts at the four likely breaks with the best stand-in
from E1 to E3, with the swap's rows, counted in accepted tokens per round against the swap and
against DDTree at the same number of tokens checked (`tree-replay-pub-2`, `tree-replay-opd-2`).

**E5. Make the second run cheap.** Where do the 13 ms go, and what can go: fewer restarts, a
shorter restart block, fewer drafter layers for the restart, or running it only when the first
draft is likely to break early.

**E6. Real speed.** The restart in real decoding (alternatives that reach past the block in the
target's one check), measured in tokens per second with every setting writing the same text, on
both drafters, alone and with the long-answer fix. The bar is the swap and DDTree.

**E7. Does it hold elsewhere?** Short benchmark prompts, and a second target size if time allows.

## What is ours and what is not

Published: second drafts at likely breaks inside the block (D²SD), stand-in states for drafting
past the end of a block (DLoop, trained), drafting without target states and the loss it causes
(DPara), trees of drafts from the first run (DDTree, GRAFT), and the measurement that a restart
with real states works together with the remark that doing it without them is open (DForkTree).
Not found published: a restart at the break from the alternative token with the missing states
left out, stood in for, or predicted. We search again before claiming it.

## Dates

- November 17 to 19: E1 and E2 (queued), E3 written and queued. The cluster is down on the 19th.
- By November 24: E3 decided, E4 run.
- By December 1: E5 and E6.
- December 6: checkpoint. Is the restart faster than the swap and DDTree in real tokens per second
  on at least one drafter? If not, we write up the study, the limit, and how far the restart got.
