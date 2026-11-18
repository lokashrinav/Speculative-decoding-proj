# Where the project stands and where it is going (November 17, 2026)

Shrinav Loka and Frank. Speculative decoding with a DFlash-style block drafter. The target is
Qwen3-4B in thinking mode with greedy decoding. The drafters are the public one
(`z-lab/Qwen3-4B-DFlash-b16`) and a better one (Draft-OPD). All numbers are from our own runs on
RTX A5000 GPUs.

## The goal

A second run of the drafter, in the same round, that makes decoding faster. The research is
centered on one thing: the target's hidden states that the drafter is missing when it runs a second
time.

## What the experiments showed

1. **Why a second draft helps.** It gives the target one more candidate token at the place where
   the first draft breaks, and almost nothing else. The control that proves it is the swap: the
   first draft with the drafter's own second-choice token at the break, with no second run. The swap
   gets past 36.1% of breaks. A real second run that is given the same token gets past the same
   breaks and adds only 0.03 tokens per break.
   ([WHY_A_SECOND_DRAFT_HELPS.md](WHY_A_SECOND_DRAFT_HELPS.md), 181,141 rounds of 100 answers on the
   public drafter and 120,835 on Draft-OPD.)
2. **What is missing after the break is the target's hidden states.** With the right token placed
   at the break, the first draft's old tokens give 0.55 more accepted tokens. A second run that
   rewrites them inside the block gives 0.64. The drafter in the next round, which has the target's
   hidden states, gives 2.03.
3. **A second run costs real time.** It takes about 13 ms on a round of about 62 ms. In replay it
   gave +20.2% accepted tokens against +18.8% for the swap, which works out to no gain in speed.
4. **The swap is a real speedup.** In real decoding, with every setting writing the same text, the
   swap gives 13% more tokens per second on the public drafter and 10% more on Draft-OPD (5 and 7
   long answers so far). It needs no training and no second run.
5. **Long answers.** The public drafter falls off on long answers because it sees more text than it
   was trained on. Showing it only the last 3,072 tokens gives 25% more real tokens per second
   (61% more beyond 8,000 tokens). Small add-on weights taught ahead of time give 42% more. With
   the swap on top these become 42% and 59%.
   ([LEARNING_DURING_THE_ANSWER.md](LEARNING_DURING_THE_ANSWER.md), Results 1 to 9.)
6. **A side finding about 16-bit arithmetic.** When the target checks a tree of drafts in one pass,
   its numbers are rounded differently than in plain decoding. On about 0.6% of tokens its two
   best choices are nearly tied and the first choice flips. So no tree-style check can promise
   plain decoding's text bit for bit in 16 bits.

## What is already published

- The window for long text (arXiv 2610.20865) and sharper attention (SharpDraft).
- Trees of drafts built from the first run alone (DDTree, arXiv 2604.12989; GRAFT, arXiv
  2608.20375). The swap is a small fixed case of such a tree. It is our control and a baseline.
- Second drafts at likely breaks inside the block (D²SD, arXiv 2606.04446) and stand-in states for
  drafting past the end of a block (DLoop, arXiv 2610.07659).
- DForkTree (arXiv 2611.12356) measures that a restart with the target's real states works and
  names doing it without them as open.

We found no paper that splits the gain of a second draft, on the same rounds, into the token at
the break, the second run, and the target's hidden states. We also found no paper that restarts
the drafter at the break without those states.

## What we are building now

A second run that restarts the drafter after the break.

Say the first draft is a b c d e f z and z is wrong. The second run starts the drafter fresh at g,
the alternative token, and it writes new tokens after it: g h i j k. The target checks the first
draft and g h i j k together. The drafter normally reads the target's hidden states for all text
before its starting token. Here the target has not read a b c d e f yet, so those few states are
missing. There are three ways to deal with that: give the drafter nothing in their place, give it
a crude stand-in, or give it predicted hidden states from a small trained network.

## The plan

The full plan for the restart second run, with what each experiment decides, is in
[SECOND_RUN_RESTART.md](SECOND_RUN_RESTART.md). In short:

1. Measure the restart with no training (`nexus/restart_probe.py`, jobs `restart-pub-0` and
   `restart-opd-0`): how many of the new tokens the target accepts with the missing states real,
   left out, stood in for, or made noisy.
2. Decide how to fill the missing states. If leaving them out or a crude stand-in keeps most of the
   gain, the second run needs no training. If not, train something small to predict them.
3. Build the full second run in replay: restarts from the second-choice token at the four most
   likely breaks, all in one drafter run. Count accepted tokens against the swap and against DDTree
   (`tree-replay-pub-1`, `tree-replay-opd-1`).
4. Make the second run cheap and measure real tokens per second on the same text, on both
   drafters, together with the long-answer fix.
5. Write it up: why a second draft helps, the restart second run with its real speed, and the
   long-answer results.

The bar is real tokens per second against the swap and DDTree. If the restart only wins in
accepted tokens, the write-up says so.

### Where the plan stands (November 17, evening)

Step 1 is done and step 2 is decided: the hidden states are needed. With the real ones the target
accepts 2.2 (public drafter) and 3.7 (Draft-OPD) tokens after g; with nothing or a crude stand-in
0.6 and 0.9, no better than the first draft's old tokens. One state, that of f, carries 87% of it,
and the states may be rough. So a small network has to predict them.

Two more things came out of the same rounds.

1. The extra drafter pass costs about as much as a restart from g earns, even with the real states,
   because g is the right token in fewer than half of the breaks. So the pass has to earn more.
2. A second kind of restart does that. It starts at f, the last token the first draft got right,
   and writes the token after f again, now with the hidden states of a b c d e. It needs no guess.
   With the real states it is worth as much as the restart from g, the two win in different rounds,
   and both together give 44 to 46% more than the restart from g alone. Both fit in one pass.

The first predictor closed only 21% of the gap between no states and the real ones, because it was
taught restarts at any place in a round and not where the drafter really breaks. The second one,
taught at real breaks and for both kinds of restart, is the next result.

### Where the plan stands (November 18, morning UTC)

Three versions of the predictor and two ways of teaching it give the same result: about a third of
what the target's real hidden states give on Draft-OPD (21% to 38%), under a half on the public
drafter (39% to 49%).

DDTree, a published method that needs no second run, is the one to beat: +29% in estimated real
speed on Draft-OPD against the swap's +13%. A second run with the real hidden states gets many more
tokens out of a round than DDTree (+60% against +36%), but its extra drafter pass costs 15 ms of a
62 ms round. On top of DDTree it nearly doubles the tokens gained (+67%) and the estimated real
speed stays the same (+29.8% against +29.1%). With predicted hidden states it makes DDTree slower.

So on this 4B target a second run does not pay in real speed. It is a result worth writing down
with its numbers: what a restart needs (the hidden state of f), how rough it may be, why a small
predictor fails, and what the extra pass costs. A second run could pay where a drafter pass is
cheap next to the target's pass, which means a much larger target.

Real decoding now takes DDTree and the restarts. Measured in real tokens per second on the same
text (16 long answers per drafter, public and Draft-OPD): the swap +14% and +11%, DDTree +37% and
+26%, DDTree with the window +65% and +34%, DDTree with the add-on weights +79% (public). The
second run with predicted hidden states: +5% and +7% alone, and +22% and +16% on top of DDTree,
which is less than DDTree alone. The replay's estimates held.

### What the second run cannot be fed (November 18, later)

Three more ways to give the restart its hidden states were measured on the frozen drafters (partial
data, `probe-more-opd-0` and `-pub-0`). Tokens accepted after g on Draft-OPD: nothing 1.0, the
predictor 2.2, the real states 4.1.

- Real states of only the first layers the drafter reads (what a check stopped early could hand
  over): 0.7 to 0.8, worse than nothing. The drafter reads target layers 1, 9, 17, 25 and 33; the
  last two alone give 3.8.
- The real state of an earlier place in the answer where the target read the same tokens: 1.8,
  and 2.4 when the tokens before and after match.
- The real state most like the predicted one: 2.1.

So what a restart needs is made in the target's deep layers at that very place in the text. It
cannot be had early, borrowed or guessed. Feeding a second run is set aside; the ideas now tried
need no second run and no hidden states (see `nexus/settings/live-echo*.txt` and the probe's
`shift_*` columns).

### Where the plan stands (November 18, midday UTC)

**The states, final numbers** (`probe-more-opd-0`, `probe-more-pub-0`, 30 answers each). Tokens
accepted after g, Draft-OPD and the public drafter:

| What the restart is given | Draft-OPD | Public drafter |
| --- | --- | --- |
| Nothing | 0.97 | 0.60 |
| The predictor's states | 2.01 | 1.32 |
| The target's real states | 3.69 | 2.11 |
| Real states of the last two layers the drafter reads (25 and 33) only | 3.48 | 2.00 |
| Real states of the first three layers (1, 9, 17) only | 0.79 | 0.53 |
| The real state of an earlier place with the same tokens | 1.56 | 1.16 |
| The same, where the tokens before and after both match (37% and 47% of breaks) | 2.38 of 4.66 | 1.59 of 2.38 |

**What the second run costs, in real decoding** (`live-tree-opd-0`, 16 answers). A plain round takes
68.7 ms: 10.3 ms for the draft and 57.4 ms for the check. The second run with four restarts of both
kinds took 15.0 ms more for itself and 4.3 ms more in the check (78 more tokens), and the round
gained 38% in tokens: +6.6% in speed. So the tokens in the check are nearly free and the second
drafter pass is what costs. A drafter pass of one block takes 8.2 ms at any length of text
(`long-opd-0/timing.json`), so no second pass can cost less than 12% of a round on this target.
A gate that skips the second run where the first draft looks safe does not help: the tokens a
restart wins are spread evenly over early and late breaks (first answer of `probe-shift-opd-0`:
+29% tokens and +5% speed without the gate, +20% and +4% with it).

**What was published meanwhile, read on November 18.** Three papers reuse what the target computed
on the draft tokens it REJECTED in the last check, and none of them needs a second pass:
ReTrace (arXiv 2608.29748) feeds the drafter's own states of the rejected tokens into the next
draft (trained, +5% speed); Carryover (2609.14717) gives the target's states of the rejected
tokens to the drafter as extra context (trained, +8 to +14%); ReAnchor (2610.26531) starts new
blocks of the frozen drafter from the token the target chose after each rejected token, with its
states of the rejected tokens, in the first draft's own pass, and attaches a block where the first
draft has its first token (nothing trained, +13% on DFlash, +10% on Draft-OPD and others). D3D
(2611.06425) drafts ahead during the check with a drafter that needs no target states. None of the
three is measured with DDTree; Carryover names the combination as open.

**What follows for us.** The cheap form of a second run is one that is ready before the first draft:
its blocks go into the first draft's pass. That is ReAnchor's form. Three things are being measured
(settings in `nexus/settings/live-second-*.txt` and `live-reanchor-*.txt`):

1. Our second run made as cheap as it can be: short new blocks (5 tokens), restarts picked best
   first over places and choices, and the target's state from the last check in place of the
   predicted one wherever the last check read the same token at the same place.
2. ReAnchor in our decoder, alone and on top of DDTree, which has not been shown. Under DDTree
   the target reads little of the first draft, so the next round has little to start blocks from;
   one setting lets it read the whole first draft for that reason.
3. Ours on top of ReAnchor's blocks: corrections. A block's first token is what the target chose
   after a draft token of the last round. Where the new first draft has that same token at that
   same place and goes on with another token, the target has already said what it wants there. The
   block is attached at that place with its own first token in the draft's place: a break foretold,
   the right token for it by the target's own word, and a block written after it.

The table of what the target already said (`live-echo-opd-0`, 9 answers) adds 8% to DDTree's tokens
per round on Draft-OPD and 16% on the public drafter; with tails of 4 tokens it is +2 points of
speed over DDTree on Draft-OPD (`live-echo2-opd-0`, 4 answers).

## The experiments and where their results are

| Experiment | What it asks | Code | Results |
| --- | --- | --- | --- |
| Why a second draft helps | Which part of a second draft's gain is the token at the break, the second run, the target's states | `nexus/why_redraft.py`, `nexus/why_report.py`, `nexus/why_ceiling.py` | `nexus/results/tune32-eval-0..2`, `tune32-opd-0..1` |
| Second drafts over whole answers | Accepted tokens and cost of each kind of second draft | `nexus/redraft_long.py` | `nexus/results/long-opd-0..3`, `long-opd-2b` |
| Real decoding with alternatives | Real tokens per second of the swap, the window and the add-on weights, on the same text | `dflash/redraft.py`, `nexus/live_speed.py` | `nexus/results/live-swap-pub-0`, `live-swap-opd-1` |
| What the larger check costs | Time of the target's check by text length and number of tokens; how often the first choice flips | `nexus/time_check.py` | `nexus/results/time-check-0` |
| Long answers | Learning during the answer, weights taught ahead of time, sharper attention, the window | `nexus/adapt_drafter.py`, `nexus/adapt_report.py` | `LEARNING_DURING_THE_ANSWER.md` |
| The swap against DDTree | Accepted tokens at the same number of tokens checked | `dflash/redraft.py` (`DraftTree`), `nexus/redraft_long.py` | `nexus/results/tree-replay-pub-1`, `tree-replay-opd-1` (queued) |
| The restart | What a second run needs in order to write well after the break | `nexus/restart_probe.py` | `nexus/results/restart-pub-0`, `restart-opd-0` (queued) |
| The restart second run over whole answers | Accepted tokens with the real states (the limit), with nothing, with crude stand-ins | `dflash/redraft.py` (`Restarts`), `nexus/redraft_long.py` | `nexus/results/restart-replay-pub-0`, `restart-replay-opd-0` (queued) |
| A predictor of the missing states | Whether a small trained network can stand in for the target's states at a restart | `dflash/stand_in.py`, `nexus/train_restart.py` | `nexus/results/train-restart-pub-0`, `train-restart-opd-0` (queued) |
| Trained second drafters (October) | Add-ons that read the first draft | `nexus/train_redraft.py`, `nexus/bench_redraft.py` | `nexus/results/train-prefix`, `train-edit`, `train-control`, `diag-a`, `diag-b` |
| The 3M editor (September) | A small network that imitates a second run | `editor-experiment/` | `results/SUMMARY.md` |

## Dates

- November 19, 16:45: cluster maintenance. The job runner has to be started again afterwards.
- December 6: checkpoint. Does the second run beat what exists in real speed? If not, stop new
  method work and write up what the measurements show.
- December 28: everything copied off the cluster. The account is deleted on January 1, 2027.
