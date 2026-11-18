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

**E1 (November 17, complete: the 100 validation answers; 179,747 rounds with a break on the public
drafter and 117,090 on Draft-OPD; jobs `restart-pub-0`, `restart-opd-0`).** Tokens the target
accepts after the right token at the break, when the drafter restarts there:

| What the drafter gets for the states it does not have | Public drafter | Draft-OPD |
| --- | --- | --- |
| The real states (the limit) | 2.17 | 3.65 |
| Nothing | 0.59 | 0.93 |
| The last real state, repeated | 0.44 | 0.62 |
| Older real states | 0.43 | 0.62 |
| Only the real state of the token right before the restart | 1.91 | 3.18 |
| All real states but that one | 1.28 | 2.38 |
| The real states with noise of 25% of each dimension's spread | 2.14 | 3.62 |
| ... of 50% | 2.08 | 3.53 |
| ... of 100% | 1.82 | 3.18 |
| ... of 200% | 1.11 | 2.10 |
| For comparison: the first draft's old tokens | 0.55 | 0.64 |
| For comparison: a second pass inside the block | 0.64 | 0.83 |

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
   states, and goes on from there. It is measured next to the restart from the second choice on
   the same rounds (`nexus/restart_probe.py --reanchor`; the results are in
   `nexus/results/tree-replay-opd-3` and `tree-replay-pub-3`, two jobs that were waiting for
   something else).

**The restart one token earlier (November 17; 40 answers and 50,964 rounds on Draft-OPD; 12
answers and 31,201 rounds on the public drafter so far).** Tokens the target accepts from the token
at the break on, with the real states:

| | Public drafter | Draft-OPD |
| --- | --- | --- |
| A restart from the second choice (nothing when that choice is wrong) | 1.30 | 2.39 |
| A restart from the last right token | 1.17 | 2.32 |
| The better of the two in each round | 1.91 | 3.44 |
| The restart from the last right token gets the token at the break right | 36% | 49% |
| The right token is the drafter's second choice | 39% | 46% |
| ... its third choice | 16% | 16% |
| ... its fourth choice | 9% | 8% |

The two kinds are worth about the same and they win in different rounds, so both together give 44%
to 46% more than the second choice alone. The earlier restart cannot help when the first draft
breaks at its very first token (it would be the first draft again), and it is the stronger of
the two when the break comes late: after five or more right tokens it gets 4.1 against 3.1 on
Draft-OPD. Both kinds fit in the same one drafter pass (`Restarts(kind="both")`, method names
`restart4rx...`). With noise of 100% on the states the earlier restart keeps 84% (Draft-OPD), the
same as the other kind.

The earlier restart also asks less of a predictor. The other kind needs the state of the last
right token, which is the state the target chooses the token at the break from, and that is exactly
where the drafter was wrong. The earlier restart needs the states only up to the token before, and
there the drafter was right.

**E3, the first predictor (November 17, job `train-restart-opd-0`, Draft-OPD).** A small network
(11 million weights) trained for one pass over 366 answers on the tokens the drafter writes with
its vectors. Measured on 12 validation answers (18,316 rounds with a break), tokens accepted after
the right token at the break: 1.67 with the predicted states, 1.02 with nothing, 4.09 with the real
ones. It closes 21% of the gap, so it fails the rule (half of the gap). The reason is in how it was
taught: restarts at any place in a round, whatever the first draft did there. On such rounds it
closes 57% of the gap (seen in the public drafter's run of the same training, which reports the
same restarts with the real states and with none). In use, a restart comes right after a token
the drafter got wrong, and the drafter's own states, which the predictor reads, are different
there.

**The second predictor (queued on November 17 for the next two free GPUs as jobs `tree-replay-opd-4`
and `tree-replay-pub-4`, which were waiting for something else; saved as `opd-b0` and `pub-b0`).** Taught where the first
draft really breaks, for both kinds of restart, with the vectors' error weighted 2 instead of 0.1,
two passes over the answers. Two more runs (`train-restart-opd-1`, `-pub-1`) teach the vectors
alone; after 8,000 steps their vectors point 0.7 of the way along the real ones (cosine), where
the first predictor's reached 0.08.

**All predictors so far, at real breaks (November 18, 00:40 UTC).** Tokens accepted after the right
token at the break, on 12 validation answers:

| Predictor | Nothing | Predicted | Real | Share of the gap closed |
| --- | --- | --- | --- | --- |
| Draft-OPD, taught on tokens at any place (`opd-v0`) | 1.02 | 1.67 | 4.09 | 21% |
| Draft-OPD, taught the vectors alone (`opd-m0`, cosine 0.77 to the real ones) | 1.02 | 2.20 | 4.09 | 38% |
| Public drafter, taught on tokens at any place (`pub-v0`) | 0.60 | 1.20 | 2.14 | 39% |

For the restart one token earlier, the predicted states give 1.15 tokens from the break on where the
real ones give 2.69 (Draft-OPD, `opd-m0`), and 0.57 against 1.17 (public, `pub-v0`). The second
predictor is still training; on its own training rounds, which are real breaks, it stands at about
30% (Draft-OPD) and 45% (public) of the gap.

So a small network that looks at one token at a time closes about a third of the gap, however it is
taught. Vectors that point 0.77 of the way along the real ones do far worse than real ones with
noise of the same size (87% kept): the predictor is wrong in exactly the directions that matter.
Two things it could not know:

1. Which token the restart starts from. The state of the last right token is the state the target
   chooses the next token from. The predictor built it from the drafter's own state there, which
   points at the wrong token, so the predicted state contradicted the restart's first token.
2. The tokens together. A break happens because the drafter fills its slots without knowing what
   the other slots chose. The restart knows the tokens, but a predictor that looks at one token at a
   time cannot put them together.

**The second predictor's result (November 18, 02:10 UTC).** Taught at real breaks for both kinds:
28% of the gap on Draft-OPD (`opd-b0`: 2.07 against 1.11 with nothing and 4.51 with the real
states, check still running) and 49% on the public drafter (`pub-b0`: 1.40 against 0.59 and 2.24).
The vectors-only predictor of the public drafter (`pub-m0`) also closes 49%. From the break on, the
better of the two kinds of restart gets 2.17 with predicted states where the real ones give 4.32
(Draft-OPD), and 1.32 where they give 1.90 (public). So teaching at real breaks did not change the
picture: about a third on Draft-OPD, about half on the public drafter.

**E2, the ceiling in real speed, and DDTree (November 18, Draft-OPD).** Replay with the measured
costs. Restarts with the target's real states (`restart-replay-opd-3`, 10 of 50 answers so far) and
DDTree (`tree-replay-opd-5`, 50 answers):

| Method | Tokens per round | Extra drafter time | Extra tokens read | Estimated real speed |
| --- | --- | --- | --- | --- |
| The swap | +15.7% | 0.9 ms | 15 | +13.2% |
| Restarts from the second choice, real states, with the swaps | +37.6% | 13.4 ms | 108 | +7.7% |
| Restarts one token earlier, real states | +35.7% | 13.2 ms | 45 | +10.1% |
| Both kinds, real states, with the swaps | +59.0% | 14.8 ms | 153 | +19.7% |
| Both kinds and the third choice, real states, 7 tokens checked | +62.7% | 18.0 ms | 149 | +18.7% |
| DDTree, 31 nodes (no second drafter pass, no states needed) | +27.5% | 0.9 ms | 16 | +24.6% |
| DDTree, 63 nodes | +35.7% | 1.6 ms | 48 | +29.1% |

This is the plain state of things. With the real states the restarts get far more tokens out of a
round than DDTree does (+59% against +36%), but their drafter pass costs 13 to 15 ms of a 62 ms
round, so in real speed they stay below DDTree even at this ceiling, which no predictor reaches.
With today's predictors (a third to a half of the gap) a second run is below the swap.

So on this target a second run can only earn its place on top of DDTree, and only if the tokens it
adds there are worth more than its pass. That is measured next (`tree-restart-opd-0`): DDTree alone,
DDTree with restarts in the same check with the real states (the ceiling of the combination), and
with the states of the predictor `opd-m0`.

**Do restarts add to DDTree? (November 18, `tree-restart-opd-0`, Draft-OPD, 50 answers, replay with
the measured costs.)**

| Method | Tokens per round | Estimated real speed |
| --- | --- | --- |
| DDTree, 31 nodes | +27.5% | +24.7% |
| DDTree, 63 nodes | +35.7% | +29.1% |
| DDTree 31 and restarts of both kinds with the real states, 7 tokens checked | +63.6% | +29.7% |
| DDTree 63 and the same restarts | +67.4% | +29.8% |
| DDTree 63 and restarts also from the third choice | +71.3% | +25.6% |
| DDTree 63 and only the restarts one token earlier | +59.0% | +27.3% |
| DDTree 63 and restarts with the predicted states (`opd-m0`) | +52.2% | +17.7% |
| Restarts with the predicted states and the swaps, no tree | +36.2% | +9.0% |

With the target's real states the restarts nearly double what the tree adds to a round (+67%
against +36%), and the time of their drafter pass (15 ms of a 62 ms round) takes all of it back:
+29.8% against +29.1%. With predicted states the tree is slower with the restarts than without
them. So on this target a second run of the drafter does not pay in real speed, whatever fills
in the hidden states. It would pay where a drafter pass is cheap next to the target's pass (a much
larger target), and then only with hidden states far better than a small predictor gives.

**The third predictor (queued for Draft-OPD on November 18).** `opd-c0` is told the token after each
token; after the last one that is the token the restart starts from (job `train-restart-opd-3`).
`opd-s0` has that and two layers in which the missing tokens look at the ones before them and at
the last 8 real vectors (job `restart-replay-pub-3`, which was waiting for something else). Both
are taught at real breaks for both kinds of restart.

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

**The third predictor's result (November 18, 05:10 UTC).** `opd-s0` (told the coming token, two
layers that look back, the last 8 real vectors; 41 million weights, two passes over the answers)
closes 22% of the gap: 1.71 tokens against 1.02 with nothing and 4.09 with the real states. That is
no better than the simpler predictors (21% to 38% on Draft-OPD). `opd-c0` looks the same during
training. Three versions and two ways of teaching give the same answer: a small network gets
about a third of what the target's states give.

**In real decoding (November 18).** `dflash_generate_redraft` now takes a tree of drafts in the
target's check: DDTree in place of the first draft, restarts that reach past the block, and both.
On the tiny test models every method writes plain decoding's tokens. The jobs `live-tree-opd-0` and
`live-tree-pub-0` measure real tokens per second on the same text: the swap, DDTree with 31 and 63
nodes, DDTree with the context window, and the second run with predicted states, alone and on top
of DDTree.
