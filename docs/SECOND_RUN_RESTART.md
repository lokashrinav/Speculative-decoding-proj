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
from the first run and at the token after it, the token itself, and the last real vector before the
round. It is trained on saved answers, where the real vectors are known, in two ways: on the tokens
the drafter writes when it restarts with the predicted vectors (jobs `train-restart-pub-0`,
`train-restart-opd-0`), and by matching the real vectors only (`train-restart-pub-1`,
`train-restart-opd-1`). Decision: continue if the restart with
predicted states closes at least half of the gap between leaving the states out and the real ones.

**E4. The whole second run in replay.** Restarts at the four likely breaks with the best stand-in
from E1 to E3, with the swap's rows, counted in accepted tokens per round against the swap and
against DDTree at the same number of tokens checked (`tree-replay-pub-2`, `tree-replay-opd-2`).

**E5. Make the second run earn its pass.** A drafter pass costs 8 ms whatever it holds (measured),
so the pass cannot get much cheaper, but it can earn more and run less: restarts from the third
choice as well in the same pass, a short check, and running it only in rounds where the drafter's
own confidence says a restart is likely to pay.

**E6. Real speed.** The restart in real decoding (alternatives that reach past the block in the
target's one check), measured in tokens per second with every setting writing the same text, on
both drafters, alone and with the long-answer fix. The bar is the swap and DDTree.

**E7. Does it hold elsewhere?** Short benchmark prompts, and a second target size if time allows.

## Results so far

**E1 (November 17, still running: 25 of 100 answers on the public drafter, 68 of 100 on
Draft-OPD).** Tokens the target accepts after the right token at the break, when the drafter
restarts there:

| What the drafter gets for the states it does not have | Public drafter | Draft-OPD |
| --- | --- | --- |
| The real states (the limit) | 2.08 | 3.78 |
| Nothing | 0.59 | 0.95 |
| The last real state, repeated | 0.43 | 0.63 |
| Older real states | 0.42 | 0.62 |
| Only the real state of the token right before the restart | 1.84 | 3.29 |
| All real states but that one | 1.23 | 2.46 |
| The real states with noise of 25% of each dimension's spread | 2.06 | 3.75 |
| ... of 50% | 2.00 | 3.66 |
| ... of 100% | 1.74 | 3.29 |
| ... of 200% | 1.07 | 2.16 |
| For comparison: the first draft's old tokens | 0.56 | 0.65 |
| For comparison: a second pass inside the block | 0.64 | 0.84 |

What this says:

1. The states are needed. Without them, or with a crude stand-in, a restart is no better than
   the first draft's old tokens. A second run with no training does not work.
2. One state carries most of it. With only the real state of the token right before the restart,
   the drafter keeps 87 to 88% of the limit.
3. The states may be rough. Noise as large as half of each dimension's spread costs 3 to 4%, and
   noise as large as the whole spread costs 13 to 16%. A wrong but real-looking state (the last
   one repeated) is much worse than a noisy right one.

So a predictor is needed, it mostly has to get one vector right, and it does not have to be
exact. That is E3.

**What the second run has to earn (from the same rounds, November 17).** The restart costs one more
drafter pass, 9 to 13 ms. In that time plain decoding writes about 1.0 token with Draft-OPD and
about 0.6 with the public drafter. Per round with a break:

| | Public drafter | Draft-OPD |
| --- | --- | --- |
| The break is one of the four likely breaks | 90% | 86% |
| The drafter's second choice there is the right token | 37% | 44% |
| Both | 33% | 38% |
| Tokens the swap adds (no drafter pass) | 0.59 | 0.74 |
| Tokens the restart adds on top of the swap, with the real states | 0.53 | 1.29 |

So even with perfect states the extra pass only pays for itself on the public drafter, and it earns
about 0.3 tokens a round on Draft-OPD. A good predictor is necessary and not enough: the pass also
has to earn more or run less. Three ways, all measurable in replay:

1. More restarts in the same pass. The pass costs nearly the same with more blocks in it, so every
   likely break can also get a restart from the drafter's third choice (`restart4rc2s`).
2. Run the pass only where it is likely to pay. Where the drafter's confidence in the token that
   turns out wrong is between 0.5 and 0.9 (39% of Draft-OPD's breaks), the second choice is right
   60% of the time and the restart adds about 2 tokens. (This looks at the true break, which a
   real rule does not know. A rule that uses only the likely breaks still has to be measured.)
3. A short check. Checking only 7 tokens after the restart's first token keeps 82% of what the
   restart gives on Draft-OPD and 92% on the public drafter.

The weak point is the guess of the right token: the second choice is right in fewer than half of
the breaks, and when it is wrong the whole restart is wasted. So there is a fourth way, which needs
no guess:

4. Restart one token earlier. A new block from the LAST token the first draft got right, with the
   states of the tokens before it, writes the token at the break again, this time with those
   states, and goes on from there. Whether that gets the break right more often than the second
   choice does is measured next to the restart from the second choice on the same rounds
   (`nexus/restart_probe.py --reanchor`; the results are in `nexus/results/tree-replay-opd-3` and
   `tree-replay-pub-3`, two jobs that were waiting for something else). The same run counts how
   often the right token is the drafter's third or fourth choice.

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
