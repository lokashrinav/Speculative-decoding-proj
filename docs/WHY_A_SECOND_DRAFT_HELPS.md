# Why does a second draft help?

This file records the study of November 17. The numbers come from `nexus/why_report.py` and
`nexus/why_ceiling.py` on the results of `nexus/why_redraft.py`, and from the whole-answer runs
of `nexus/redraft_long.py`. Results 1 to 8 are complete.

## The question

Our drafter writes 15 tokens at once. The target accepts them up to the first token it
disagrees with. That place is the break. Papers such as D²SD (arXiv 2606.04446) and D-Loop
(arXiv 2610.06011) run the drafter a second time in the same round and report that more tokens
are accepted. Our own runs in October showed the same. The question here is why.

## The short answer

A second draft helps because it hands the target one more candidate token at the break. Almost
all of the gain is that one token. The second run of the drafter itself adds very little on
top of it.

The proof is a control that needs no second run. Take the first draft and put the drafter's
own second-choice token at the break, with the rest of the draft unchanged. We call this the
swap. The target checks the first draft and the swapped copy together. The swap gets past 36.1%
of all breaks. A real second run that is given the same second-choice token and rewrites the
rest gets past exactly the same breaks and adds only 0.03 more tokens per break.

## How it was measured

The target is Qwen3-4B in thinking mode with greedy decoding. Every answer was first decoded
and saved. Each saved answer is then replayed with the first draft alone, so the rounds are the
rounds of plain decoding. In every round, at the same place, many kinds of second draft are made,
and each is scored by how many tokens the target would accept from it. All kinds are tried on
the same rounds, so the differences between them are differences in cause and not in luck.

The answers are the 100 validation answers of `specdec-data/long1` (up to 16,384 tokens each).
The ranges in square brackets are 95% ranges from drawing whole answers again (2,000 draws).

- Public drafter (`z-lab/Qwen3-4B-DFlash-b16`): 181,141 rounds. The first draft alone gives 3.20
  tokens per round and breaks in 99.3% of rounds.
- Better drafter (Draft-OPD): 120,835 rounds. The first draft alone gives 4.80 tokens per round
  and breaks in 97.0% of rounds.
- Public drafter with the add-on weights that repair it on long answers (off-a): 71,361 rounds
  from 50 of the answers. The first draft alone gives 4.15 tokens per round.

## Result 1: every kind of second draft on the same rounds (public drafter)

All second drafts below start from the true break, so that the question of where to place them
plays no part. "Got past" is the share of all breaks at which the target accepts the token at the
break. "Tokens added" is the mean number of extra accepted tokens per round with a break.

| Kind of second draft | Second run? | Got past the break | Tokens added per break |
| --- | --- | --- | --- |
| Same drafter, shown its first draft up to the break | yes | 12.5% | 0.245 |
| Same drafter, shown random tokens instead | yes | 3.9% | 0.066 |
| **The swap: second choice at the break, old rest** | **no** | **36.1%** | **0.646** |
| Second choice at the break, rest rewritten | yes | 36.1% | 0.673 |
| Second run, made to pick a different token at the break | yes | 34.2% | 0.655 |
| October add-on trained to read the draft | yes | 25.8% | 0.602 |
| That add-on, made to pick a different token at the break | yes | 45.4% | 0.941 |
| That add-on, shown nothing of the draft | yes | 12.8% | 0.273 |
| Add-on trained without ever seeing a draft | yes | 9.9% | 0.204 |
| Right token at the break (not possible in practice), old rest | no | 100% | 1.553 |
| Right token at the break, rest rewritten | yes | 100% | 1.642 |
| Right token, then the drafter with the target's real hidden states | next round | 100% | 3.028 |

The same drafter shown its own draft could only be tried when the break came after at least one
accepted token (70.2% of breaks). Its numbers are still per all breaks.

## Result 2: the differences, round by round

Each line is a difference in tokens added per break on the same rounds.

| Comparison | Difference | What it says |
| --- | --- | --- |
| Rewritten rest − swap (same second-choice token) | 0.026 [0.017, 0.036] | The second run adds almost nothing once the token is switched. |
| Different token picked by a second run − swap | 0.008 [−0.006, 0.025] | The second run does not find a better other token than the first run's second choice. |
| Shown its draft − shown random tokens | 0.255 [0.243, 0.269] | A plain second run does read its draft. Most of its small gain comes from that. |
| Shown random tokens − nothing | 0.094 [0.087, 0.101] | Merely guessing again gives a little. |
| Right token, rewritten rest − right token, old rest | 0.089 [0.071, 0.110] | Even with the right token, rewriting the rest from tokens adds little. |
| Real hidden states − right token, rewritten rest | 1.388 [1.309, 1.466] | What is missing after the break is the target's hidden states, not the tokens. |
| Trained add-on shown the draft − shown nothing | 0.328 [0.310, 0.349] | The trained add-on really uses the draft. |
| Trained add-on made to differ − not made to differ | 0.340 [0.328, 0.352] | Forcing a different token at the break is worth as much as the training. |

## Result 3: where the right token stood

At the break, the right token is the drafter's second choice in 38.3% of breaks, third or fourth
in 24.1%, fifth to sixteenth in 24.1%, and lower in 10.3%. (In 3.2% it is tied for first: the
16-bit scores were exactly equal and the other token was taken.)

The swap succeeds at 94.4% of the breaks where the right token was the second choice and at none
of the others. A plain second run of the same drafter gets past 30.0% of the breaks where the
right token was the second choice, because it repeats its first mistake most of the time: after
the second run the right token is its first choice at only 19.1% of the breaks where it was tried.

The October add-on that reads the draft moves the right token to first choice at 27.0% of breaks.
Made to pick a token other than the first draft's, it gets past 45.4% of breaks. That is the one
thing a trained second drafter adds: a better alternative token than the first run's second choice.

## Result 4: what happens after the break

Once the right token is in place, how many more tokens follow?

| What follows the right token | More tokens after it |
| --- | --- |
| The first draft's old tokens | 0.55 |
| A second run that rewrites the rest from the tokens | 0.64 |
| The drafter in the next round, with the target's real hidden states | 2.03 |

A DFlash-style drafter reads the target's hidden states for the checked text. Inside its own
block it has only its own tokens. Showing it the right token inside the block is not the same as
giving it the target's hidden states for that token. This is why a second run cannot repair the
rest of the block, and why the next round is so much better.

For late breaks (after 5 to 14 accepted tokens) rewriting the rest is worse than keeping the old
rest (−0.26 tokens with the right token in place), so a second run there does harm.

## Result 5: the better drafter, and the repaired public drafter

The same picture holds for Draft-OPD and for the public drafter with the add-on weights off-a.

| Kind of second draft | Draft-OPD: got past | Draft-OPD: tokens added | Repaired: got past | Repaired: tokens added |
| --- | --- | --- | --- | --- |
| Same drafter, shown its first draft up to the break | 15.8% | 0.372 | 16.3% | 0.370 |
| Same drafter, shown random tokens instead | 4.0% | 0.077 | 4.4% | 0.080 |
| **The swap** | **43.8%** | **0.828** | **41.8%** | **0.793** |
| Second choice at the break, rest rewritten | 43.8% | 0.912 | 41.8% | 0.901 |
| Second run, made to pick a different token at the break | 34.5% | 0.767 | 39.2% | 0.857 |
| Right token, old rest | 100% | 1.638 | 100% | 1.639 |
| Right token, rest rewritten | 100% | 1.826 | 100% | 1.888 |
| Right token, then the target's real hidden states | 100% | 3.996 | 100% | 3.623 |

Rewritten rest − swap is 0.085 [0.067, 0.102] for Draft-OPD and 0.107 [0.084, 0.131] for the
repaired drafter. That is larger than for the public drafter (0.026), but still about a tenth of
the swap's gain. Real hidden states − rewritten rest is 2.17 [1.94, 2.37] and 1.74 [1.57, 1.89].
The right token is the second choice at 45.4% and 43.6% of breaks.

## Result 6: in practice, without knowing the break

In decoding the true break is not known. D²SD places second drafts at the four most likely breaks,
scored from the drafter's own confidence. The table gives more accepted tokens per round than the
first draft alone, with the same placement.

| Method (four likely breaks) | Public drafter | Draft-OPD | Repaired public drafter |
| --- | --- | --- | --- |
| The swap, no second run | +17.9% | +14.8% | +16.9% |
| Second choice at the break, rest rewritten by a second run | +19.1% | +16.5% | +19.4% |
| Limit of a restart with the target's real hidden states after the switched token | +33.2% | +35.0% | +34.9% |

The last row is the best that any predictor of the target's hidden states could reach. It uses
the real ones.

Whole answers replayed from start to end with each method agree with this (public drafter, the
same 50 answers for every row, plain decoding at 3.31 tokens per round):

| Method | More tokens per round | Extra drafter time per round |
| --- | --- | --- |
| Same drafter again, one redraft | +5.3% | 8.2 ms |
| Same drafter again, redrafts at four likely breaks | +8.4% | 12.8 ms |
| Swap at four likely breaks, no second run | +18.8% | 0.8 ms |
| Second choice at four likely breaks, rest rewritten | +20.2% | 13.2 ms |
| October add-on at four likely breaks | +18.1% | 13.0 ms |
| October add-on at four likely breaks, made to differ | +25.7% | 13.4 ms |

On top of the add-on weights that repair the drafter on long answers (off-a), the gains add up:
the repaired drafter alone gives +25.3% tokens per round, and with the second choice at four
likely breaks +52.3% (+103.5% beyond 8,000 tokens).

These tables count accepted tokens. What they are worth in real time is Result 8.

## Result 7: the swap with a shorter check

The swapped copies add tokens to the target's check. Most of the gain is in the switched token and
the few tokens after it. With the target checking only the switched token and N tokens after it
(four likely breaks):

| N | 0 | 1 | 2 | 3 | 4 | all |
| --- | --- | --- | --- | --- | --- | --- |
| Public drafter: more tokens per round | +10.0% | +13.8% | +15.6% | +16.6% | +17.1% | +17.9% |
| Draft-OPD: more tokens per round | +7.6% | +10.2% | +11.8% | +12.7% | +13.4% | +14.8% |

## Result 8: real decoding

Real decoding now checks the first draft and several alternatives in the target's one pass
(`dflash_generate_redraft`). The first real runs and a timing probe (`nexus/time_check.py`, job
`time-check-0`) showed two things that the replay could not show.

**What the larger check costs.** The target's check on an RTX A5000, by how much text came before
(median of 15 passes, in ms):

| Text so far (tokens) | 1,000 | 4,000 | 8,000 | 12,000 | 16,000 | 24,000 |
| --- | --- | --- | --- | --- | --- | --- |
| The first draft alone (16 tokens) | 48.5 | 48.5 | 54.8 | 73.9 | 97.3 | 135.9 |
| With four alternatives of 4 tokens (32 tokens) | 48.4 | 48.6 | 56.9 | 76.3 | 100.0 | 139.4 |
| With four alternatives to the end of the block (54 tokens) | 49.4 | 49.5 | 59.8 | 79.7 | 103.9 | 144.6 |

The swap with a tail of 3 costs nothing up to 4,000 tokens of text and 2 to 3.5 ms more per round
beyond 8,000. The mask costs nothing: 32 tokens read in order take the same time as the 32 tokens
of the swap, and writing the mask takes 0.4 ms.

**The text is plain decoding's text only up to nearly tied tokens.** Every free-running run with
alternatives wrote a different token from plain decoding somewhere in the first 150 to 550 tokens,
and from there on a different answer, often of a very different length. The probe shows why. On
4,480 rows (640 at each of seven text lengths), compared with the same tokens read in a pass of
plain decoding's shape:

| Rows | Bit-equal | First choice differs |
| --- | --- | --- |
| The first draft's 16 rows, with the mask written out | all | 0 |
| The first draft's 16 rows, with alternatives after them in the pass | none | 17 (0.4%) |
| The rows of the alternatives | none | 39 (0.9%) |

In 16-bit arithmetic a pass with more rows rounds every row differently. Where the first choice
changed, the target's two best tokens were never more than 0.25 apart, which is one or two steps
of the 16-bit scale for numbers of that size. So about one token in a hundred is a coin toss between two tokens, and
any change in the shape of the pass can flip it. This holds for every method that checks a tree of
drafts in one pass, and it means that "the same text as plain decoding" cannot be promised bit for
bit in 16 bits, only "the target's first choice in the pass that checked it".

An attempt to repair this (`recheck`: let the second, in-order reading of the tokens accepted on
an alternative's rows decide) does not give plain decoding's text on the GPU, because that second
reading is itself a pass of another shape (job `live-swap-opd-0`). It is off by default.

**How speed is measured from here on.** Settings that write different text cannot be compared: the
first runs seemed to show the swap 4% slower beyond 2,000 tokens, and that came from the settings
having written answers of different lengths. Now the settings with alternatives follow plain
decoding's answer (`follow` in `dflash_generate_redraft`): every pass is computed and timed as
usual, but what the target says after a token is read from plain decoding's answer, and the rows
the pass itself decided otherwise are counted. The runs `live-swap-pub-0` and `live-swap-opd-1`
measure it that way.

**The result (16 long answers per drafter; every setting wrote plain decoding's tokens in all 16).**
Real tokens per second, and the gain over the plain drafter:

| Setting | Public drafter | beyond 8,000 tokens | Draft-OPD | beyond 8,000 tokens |
| --- | --- | --- | --- | --- |
| Plain | 44.2 | 30.4 | 68.8 | 55.8 |
| The swap (four likely breaks, 3 tokens checked after the switched one) | 50.2 (+13.5%) | +14.3% | 76.0 (+10.4%) | +10.3% |
| The drafter sees only the last 3,072 tokens (4,096 for Draft-OPD) | 54.8 (+24.0%) | +58.7% | 73.8 (+7.2%) | +18.6% |
| That window and the swap | 62.0 (+40.2%) | +79.5% | 81.5 (+18.5%) | +30.8% |
| Add-on weights taught ahead of time | 59.6 (+34.8%) | +71.7% | 73.6 (+7.0%) | +16.6% |
| Those weights and the swap | 67.2 (+52.1%) | +93.2% | 81.6 (+18.6%) | +29.1% |

A round takes 70 ms on these answers with either drafter. The swap adds 3 ms to it and 18% (public)
or 15% (Draft-OPD) to the tokens of a round. The swap's gain adds on top of the window and of the
add-on weights. The window and the add-on weights do not add to each other.

**What is already clean.** Settings without alternatives write plain decoding's text by themselves
(every answer so far), so their real speed is settled:

| Setting, real tokens per second against the plain drafter | All text | Beyond 8,000 tokens |
| --- | --- | --- |
| Public drafter, add-on weights off-a (8 answers) | +34% | +83% |
| Public drafter, sharper attention (8 answers) | +14% | +36% |
| Draft-OPD, add-on weights opd-g (12 answers) | +8% | +17% |
| Draft-OPD, sharper attention (12 answers) | +3% | +7% |
| Draft-OPD, the drafter sees only the last 4,096 tokens (3 answers) | +6% | +15% |

## Result 10: the swap against DDTree

DDTree (arXiv 2604.12989) builds a tree from the first pass alone: the most probable nodes, by the
drafter's own probabilities, checked by the target in one pass. The swap is a small fixed tree of
that kind. Replay on 50 validation answers with Draft-OPD (job `tree-replay-opd-5`), with the
measured cost of the extra tokens the target reads:

| Method | Tokens the target reads besides a block | Tokens per round | Estimated real speed |
| --- | --- | --- | --- |
| Plain | 0 | 4.99 | 80.1 tokens per second |
| The swap (four likely breaks, 3 tokens after each) | 15 | +14.9% | +12.5% |
| DDTree, 15 nodes (a block's worth, placed by probability) | 0 | +16.9% | +15.9% |
| DDTree, 31 nodes | 16 | +27.5% | +24.6% |
| DDTree, 63 nodes | 48 | +35.7% | +29.1% |
| DDTree, 127 nodes | 112 | +42.4% | +28.2% |

DDTree beats the swap clearly at the same number of tokens read (31 nodes against the swap's 15 +
16: +27.5% against +14.9%), and even a tree of only 15 nodes, which costs nothing, beats it. On an
A5000 the best budget is about 63 nodes. Beyond 8,000 tokens of text the extra tokens cost more
and 31 to 63 nodes are equal (+25% to +26%). So the swap is the baseline with no tree search, and
DDTree is the method to beat. (This is a replay with measured costs, not real decoding: DDTree is
not yet built into this repository's real decoding. The public drafter's run is still to do.)

## What the first draft's mistakes look like

I read 60 breaks as text. Four kinds of mistake keep coming back.

1. The drafter repeats the token before the break ("of of", "part part"). This is 12.5% of breaks.
2. The drafter skips a token and writes the one that comes after it. This is 14% of breaks.
3. The text could go on in two ways and the drafter takes the other one ("Let's" where the target
   writes "Let me", "water" where the target writes "air"). The second choice is often right here.
4. The drafter cannot know the token: a digit in a calculation, or a rare word.

In nearly every block the tokens after the first few turn into repeats ("the the the", ",,,,").
Each slot is filled without knowing what the other slots chose, so later slots are a blur of
several possible continuations. A second run with the right token in place still writes the same
blur after it. The next round, with the target's hidden states, is sharp again for a few tokens.

## Conclusion and how sure it is

1. A second draft helps because the target gets one more candidate token at the break. For the
   unchanged drafter this is as sure as a measurement can make it: the swap gets past the same
   breaks with no second run, and the second run adds 0.03 tokens per break on the public drafter
   (range 0.017 to 0.036), 0.09 on Draft-OPD and 0.11 on the repaired public drafter.
2. The second run itself adds almost nothing, because inside the block the drafter has tokens and
   not the target's hidden states. This is well supported: with the right token in place, the
   rewritten rest adds 0.09 tokens and the target's real hidden states add 1.39 more.
3. A trained second drafter adds one real thing: a better alternative token (45.4% against 36.1%).
4. What the swap is worth in real time is not measured yet. Its check costs little (Result 8),
   and the runs that time every setting on the same text are under way.

The limits of the study: one target model, greedy decoding, one kind of text (long thinking
answers), second drafters trained by us in October on short text, and replay in place of live
decoding for Results 1 to 7 (replay is exact for accepted tokens).

## What is already published

- The swap is a small fixed special case of a draft tree (DDTree, arXiv 2604.12989; a learned
  choice of branches: GRAFT, arXiv 2608.20375). We found no paper that tests it by itself. It is
  the control of this study and a baseline that needs no training, not a new method.
- DForkTree (arXiv 2611.12356) already reports that a branch without the target's hidden states is
  drafted badly and that a second pass with a small add-on barely pays. What it does not have is
  the split on the same rounds into the token at the break, the second run, and the target's
  hidden states.

## To reproduce

    python nexus/why_report.py nexus/results/tune32-eval-0 nexus/results/tune32-eval-1 --details
    python nexus/why_ceiling.py nexus/results/tune32-eval-0 nexus/results/tune32-eval-1
    python nexus/why_report.py nexus/results/tune32-eval-2 nexus/results/tune32-opd-0 --details
    python nexus/why_ceiling.py nexus/results/tune32-eval-2 nexus/results/tune32-opd-0
    python nexus/why_report.py nexus/results/tune32-opd-1 --details
    python nexus/why_ceiling.py nexus/results/tune32-opd-1
