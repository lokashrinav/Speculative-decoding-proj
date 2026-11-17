# Why does a second draft help?

This file records the study of November 17. The numbers come from `nexus/why_report.py` and
`nexus/why_ceiling.py` on the results of `nexus/why_redraft.py`, and from the whole-answer runs
of `nexus/redraft_long.py`. The jobs were still running when this was written, so every table
says how many answers it rests on. The file is updated when they finish.

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
swap. The target checks the first draft and the swapped copy together. The swap gets past 36.6%
of all breaks. A real second run that is given the same second-choice token and rewrites the
rest gets past exactly the same breaks and adds only 0.02 more tokens per break.

## How it was measured

The target is Qwen3-4B in thinking mode with greedy decoding. Every answer was first decoded
and saved. Each saved answer is then replayed with the first draft alone, so the rounds are the
rounds of plain decoding. In every round, at the same place, many kinds of second draft are made,
and each is scored by how many tokens the target would accept from it. All kinds are tried on
the same rounds, so the differences between them are differences in cause and not in luck.

The answers are the validation answers of `specdec-data/long1` (up to 16,384 tokens each). The
ranges in square brackets are 95% ranges from drawing whole answers again (2,000 draws).

- Public drafter (`z-lab/Qwen3-4B-DFlash-b16`): 119,551 rounds from 61 of the 100 answers so far.
  The first draft alone gives 3.21 tokens per round and breaks in 99.3% of rounds.
- Better drafter (Draft-OPD): 49,488 rounds from 40 of the 100 answers so far. The first draft
  alone gives 5.03 tokens per round and breaks in 96.2% of rounds.

## Result 1: every kind of second draft on the same rounds (public drafter)

All second drafts below start from the true break, so that the question of where to place them
plays no part. "Got past" is the share of all breaks at which the target accepts the token at the
break. "Tokens added" is the mean number of extra accepted tokens per round with a break.

| Kind of second draft | Second run? | Got past the break | Tokens added per break |
| --- | --- | --- | --- |
| Same drafter, shown its first draft up to the break | yes | 12.4% | 0.245 |
| Same drafter, shown random tokens instead | yes | 4.0% | 0.069 |
| **The swap: second choice at the break, old rest** | **no** | **36.6%** | **0.661** |
| Second choice at the break, rest rewritten | yes | 36.6% | 0.683 |
| Second run, made to pick a different token at the break | yes | 34.4% | 0.659 |
| October add-on trained to read the draft | yes | 25.9% | 0.608 |
| That add-on, made to pick a different token at the break | yes | 45.7% | 0.954 |
| That add-on, shown nothing of the draft | yes | 12.9% | 0.275 |
| Add-on trained without ever seeing a draft | yes | 10.0% | 0.207 |
| Right token at the break (not possible in practice), old rest | no | 100% | 1.567 |
| Right token at the break, rest rewritten | yes | 100% | 1.651 |
| Right token, then the drafter with the target's real hidden states | next round | 100% | 3.035 |

The same drafter shown its own draft could only be tried when the break came after at least one
accepted token (70.3% of breaks). Its numbers are still per all breaks.

## Result 2: the differences, round by round

Each line is a difference in tokens added per break on the same rounds.

| Comparison | Difference | What it says |
| --- | --- | --- |
| Rewritten rest − swap (same second-choice token) | 0.021 [0.011, 0.035] | The second run adds almost nothing once the token is switched. |
| Different token picked by a second run − swap | −0.002 [−0.019, 0.017] | The second run does not find a better other token than the first run's second choice. |
| Shown its draft − shown random tokens | 0.252 [0.236, 0.270] | A plain second run does read its draft. Most of its small gain comes from that. |
| Shown random tokens − nothing | 0.098 [0.088, 0.106] | Merely guessing again gives a little. |
| Right token, rewritten rest − right token, old rest | 0.083 [0.062, 0.112] | Even with the right token, rewriting the rest from tokens adds little. |
| Real hidden states − right token, rewritten rest | 1.386 [1.299, 1.480] | What is missing after the break is the target's hidden states, not the tokens. |
| Trained add-on shown the draft − shown nothing | 0.333 [0.310, 0.359] | The trained add-on really uses the draft. |
| Trained add-on made to differ − not made to differ | 0.346 [0.332, 0.363] | Forcing a different token at the break is worth as much as the training. |

## Result 3: where the right token stood

At the break, the right token is the drafter's second choice in 38.7% of breaks, third or fourth
in 24.3%, fifth to sixteenth in 24.1%, and lower in 9.6%. (In 3.3% it is tied for first: the
16-bit scores were exactly equal and the other token was taken.)

The swap succeeds at 94.4% of the breaks where the right token was the second choice and at none
of the others. A plain second run of the same drafter gets past 29.8% of the breaks where the
right token was the second choice, because it repeats its first mistake most of the time: after
the second run the right token is its first choice at only 19.0% of the breaks where it was tried.

The October add-on that reads the draft moves the right token to first choice at 27.2% of breaks.
Made to pick a token other than the first draft's, it gets past 45.7% of breaks. That is the one
thing a trained second drafter adds: a better alternative token than the first run's second choice.

## Result 4: what happens after the break

Once the right token is in place, how many more tokens follow?

| What follows the right token | More tokens after it |
| --- | --- |
| The first draft's old tokens | 0.57 |
| A second run that rewrites the rest from the tokens | 0.65 |
| The drafter in the next round, with the target's real hidden states | 2.04 |

A DFlash-style drafter reads the target's hidden states for the checked text. Inside its own
block it has only its own tokens. Showing it the right token inside the block is not the same as
giving it the target's hidden states for that token. This is why a second run cannot repair the
rest of the block, and why the next round is so much better.

For late breaks (after 5 to 14 accepted tokens) rewriting the rest is worse than keeping the old
rest (−0.27 tokens with the right token in place), so a second run there does harm.

## Result 5: the better drafter

The same picture holds for Draft-OPD (40 answers so far).

| Kind of second draft | Got past the break | Tokens added per break |
| --- | --- | --- |
| Same drafter, shown its first draft up to the break | 16.2% | 0.384 |
| Same drafter, shown random tokens instead | 4.2% | 0.080 |
| **The swap** | **45.5%** | **0.871** |
| Second choice at the break, rest rewritten | 45.5% | 0.949 |
| Second run, made to pick a different token at the break | 35.4% | 0.789 |
| Right token, old rest | 100% | 1.667 |
| Right token, rest rewritten | 100% | 1.848 |
| Right token, then the target's real hidden states | 100% | 4.097 |

Rewritten rest − swap is 0.078 [0.054, 0.104] here. It is larger than for the public drafter but
still under a tenth of the swap's gain. Real hidden states − rewritten rest is 2.25 [1.93, 2.53].
The right token is the second choice at 47.3% of breaks.

## Result 6: in practice, without knowing the break

In decoding the true break is not known. D²SD places second drafts at the four most likely breaks,
scored from the drafter's own confidence. The table gives more accepted tokens per round than the
first draft alone, with the same placement.

| Method (four likely breaks) | Public drafter | Better drafter |
| --- | --- | --- |
| The swap, no second run | +18.3% | +14.7% |
| Second choice at the break, rest rewritten by a second run | +19.3% | +16.2% |
| Limit of a restart with the target's real hidden states after the switched token | +33.6% | +34.8% |

The last row is the best that any predictor of the target's hidden states could reach. It uses
the real ones.

Whole answers replayed from start to end with each method agree with this (public drafter, 50
answers, plain decoding at 3.31 tokens per round; the first two rows rest on 40 of the answers so far):

| Method | More tokens per round | Extra drafter time per round |
| --- | --- | --- |
| Same drafter again, one redraft | +5.3% | 8.3 ms |
| Same drafter again, redrafts at four likely breaks | +8.4% | 13.0 ms |
| Swap at four likely breaks, no second run | +18.8% | 0.8 ms |
| Second choice at four likely breaks, rest rewritten | +20.2% | 13.2 ms |
| October add-on at four likely breaks | +18.1% | 13.0 ms |
| October add-on at four likely breaks, made to differ | +25.7% | 13.4 ms |

A round of plain decoding takes about 56 ms on short text and about 83 ms beyond 8,000 tokens
(RTX A5000). A second run of 13 ms therefore costs about as much as the tokens it wins: with the
measured times added up, the rewritten rest comes out at 0.996 times the speed of plain decoding,
and the swap at 1.175 times. These speeds are estimates from measured parts, and they leave out
the larger check (see the caveats). The live measurement is the next step.

On top of the add-on weights that repair the drafter on long answers (off-a), the gains add up:
the repaired drafter alone gives +25.3% tokens per round, and with the second choice at four
likely breaks +52.3% (+103.5% beyond 8,000 tokens). The swap on the repaired drafter has not been
run yet.

## Result 7: the swap with a shorter check

The swapped copies add tokens to the target's check. Most of the gain is in the switched token and
the few tokens after it, so the check can be kept short. With the target checking only the
switched token and N tokens after it (four likely breaks, public drafter):

| N | 0 | 1 | 2 | 3 | 4 | all |
| --- | --- | --- | --- | --- | --- | --- |
| More tokens per round | +10.1% | +14.0% | +15.9% | +16.8% | +17.4% | +18.3% |

## What the first draft's mistakes look like

I read 60 breaks as text. Four kinds of mistake keep coming back.

1. The drafter repeats the token before the break ("of of", "part part"). This is 13% of breaks.
2. The drafter skips a token and writes the one that comes after it. This is 15% of breaks.
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
   breaks with no second run, and the second run adds 0.02 tokens per break on the public drafter
   (range 0.011 to 0.035) and 0.08 on the better drafter (range 0.054 to 0.104).
2. The second run itself adds almost nothing, because inside the block the drafter has tokens and
   not the target's hidden states. This is well supported: with the right token in place, the
   rewritten rest adds 0.08 tokens and the target's real hidden states add 1.39 more.
3. A trained second drafter adds one real thing: a better alternative token (45.7% against 36.6%).

The limits of the study: one target model, greedy decoding, one kind of text (long thinking
answers), second drafters trained by us in October on short text, and replay in place of live
decoding (replay is exact for accepted tokens, and speed is estimated).

## Caveats

- The speed estimates count the first draft, the check of 16 tokens and the extra drafter time.
  They do not count the larger check when several candidates are verified together. Measured on
  the A5000, checking 80 tokens in place of 16 costs under 1 ms more up to 4,000 tokens of text,
  11 ms more at 8,000 and 16 ms more at 16,000. Result 7 is the way around this.
- "Four likely breaks" uses the drafter's confidence as D²SD does. A learned choice of where to
  branch already exists (GRAFT, arXiv 2608.20375; DDTree, arXiv 2604.12989). The swap is a small
  fixed special case of such a draft tree. It is the control of this study and not a new method.

## To reproduce

    python nexus/why_report.py nexus/results/tune32-eval-0 nexus/results/tune32-eval-1 --details
    python nexus/why_ceiling.py nexus/results/tune32-eval-0 nexus/results/tune32-eval-1
    python nexus/why_report.py nexus/results/tune32-eval-2 --details
    python nexus/why_ceiling.py nexus/results/tune32-eval-2
