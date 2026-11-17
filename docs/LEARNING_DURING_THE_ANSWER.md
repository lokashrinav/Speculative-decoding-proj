# A drafter that learns from its own draft during the answer: running results

This file is the running record of the main line of the project. It is updated after every
batch of cluster jobs. The numbers come from `nexus/adapt_report.py` on the job results in
`nexus/results/`.

## The question

In speculative decoding a small drafter proposes a block of tokens and the big model (the
target) checks them. Our drafter proposes 15 tokens at once. The target accepts the tokens
up to the first one it disagrees with and writes that one itself. That token is the target's
correction of the draft.

The idea of the main line: after every check, the drafter is trained a little on what the
check just showed (the accepted tokens and the correction), so that it drafts the rest of the
same answer better. The drafter's own weights stay fixed. Only a small set of extra weights on
it is trained (low-rank adapters of rank 16, about 1% of the drafter), and they start from zero
again in every new answer. Decoding stays lossless, because the target still checks every draft.

The basic idea is published (Test-Time Speculation, arXiv 2605.09329). What can be ours is to
show where it really helps, to make it cheap enough to give real speed, and to explain what
the drafter learns.

## Setup

- Target: Qwen/Qwen3-4B, thinking mode, greedy.
- Public drafter: `z-lab/Qwen3-4B-DFlash-b16`. Better drafter: `bingyang-lei/Qwen3-4B-Thinking-Draft-OPD`
  (the same drafter after about 120 GPU-hours of extra training by another group, arXiv 2605.29343).
- Answers: 100 validation answers of up to 16,384 new tokens (old AIME problems, hardest MATH
  training problems, long writing, chat). All settings are replayed on the same saved answers.
- Measure: tokens per round (a round is one draft and one check), overall and by how far into
  the answer the round began. Brackets are 95% intervals from resampling answers.
- Learning step: Adam, learning rate 3e-4, labels "immediate" (the accepted tokens and the
  correction), one step per round unless said otherwise.

## Result 1: learning during the answer against a fix that needs no training (Nov 16)

SharpDraft (arXiv 2610.05106) multiplies the drafter's attention queries by a number that grows
with the length of the text. It needs no training and costs nothing. Its paper reports that it
beats adapters trained during the answer. In tokens per round we find the opposite, on both
drafters, and sharpening adds nothing once the drafter learns.

Public drafter (jobs `sharp-eval-0..3`, 100 answers), tokens per round:

| setting | 0-2k | 2k+ | 8k+ | all | gain over plain |
|---|---|---|---|---|---|
| plain | 3.05 | 3.28 | 2.76 | 3.20 | |
| sharper attention (SharpDraft, reference 2048) | 3.05 | 3.88 | 3.79 | 3.56 | +11.3% [8.5, 13.6] |
| learning during the answer | 3.10 | 4.40 | 4.73 | 3.87 | +20.9% [16.0, 25.1] |
| both | 3.10 | 4.45 | 4.77 | 3.90 | +21.8% [16.7, 26.0] |

Learning over sharper attention: +8.7% [6.7, 10.4] overall, +13.5% beyond 2,000 tokens, +24.5%
beyond 8,000. Sharper attention on top of learning: +0.7% [0.5, 0.9].

Better drafter (jobs `sharp-opd-0..3`, 97 of 100 answers when read), tokens per round:

| setting | 0-2k | 2k+ | 8k+ | all | gain over plain |
|---|---|---|---|---|---|
| plain | 3.78 | 5.54 | 5.29 | 4.81 | |
| sharper attention (SharpDraft) | 3.79 | 5.69 | 5.69 | 4.89 | +1.5% [1.0, 2.1] |
| learning during the answer | 3.84 | 6.00 | 6.31 | 5.07 | +5.3% [3.7, 7.1] |
| both | 3.84 | 5.98 | 6.29 | 5.06 | +5.1% [3.5, 6.9] |

Learning over sharper attention: +3.7% [2.5, 5.3] overall, +11.0% [8.3, 15.4] beyond 8,000 tokens.
Sharper attention on top of learning: -0.2%.

Reading. About half of what the public drafter gains by learning is what the simple rule gives
(sharper attention on long texts). The other half the rule does not give. On the better drafter
both gains are much smaller, and learning is still clearly ahead of the rule, most of all far
into long answers. These are tokens per round. The rule is free and a learning step is not, so
the comparison in real time is still open (Result 3 and the cost table below).

A reference of 4096 instead of 2048 is worse on the public drafter (+8.3%) and the same on the
better one (+1.7%).

## Result 2: the published signal against the simple one, on the better drafter (Nov 16)

Jobs `opd-eval-0..3`, 100 answers. Test-Time Speculation learns from what the target predicted
at all 15 positions of the draft ("canvas"). We learn from the accepted tokens and the
correction ("immediate"). Both give the same: canvas +5.0% [3.5, 6.5], immediate +5.1%
[3.6, 6.8]; beyond 8,000 tokens +18.6% and +18.7% (5.28 to 6.27 tokens per round).

## Result 3: cheaper ways to learn, on the better drafter (Nov 16, 100 answers)

Jobs `cheap-opd-0..3`, read together with `sharp-opd-0..3`. Gains are over the plain drafter.

| variant | all | 2k+ | 8k+ | updates per round |
|---|---|---|---|---|
| full learning (the drafter is run a second time) | +5.2% | +8.2% | +18.9% | 1.00 |
| one pass (the draft keeps its gradients, no second run) | +5.2% | +8.2% | +19.0% | 1.00 |
| the correction alone | +3.7% | +5.8% | +14.6% | 0.94 |
| the accepted tokens alone | -0.8% | +0.0% | +8.1% | 0.77 |
| every 4th round | +2.7% | +4.3% | +10.9% | 0.25 |
| only after an early rejection (at most 2 tokens accepted) | +1.9% | +2.4% | +4.5% | 0.49 |
| only the last layer's extra weights | +1.6% | +2.0% | +3.4% | 1.00 |
| sharper attention, no training (for comparison) | +1.5% | +2.6% | +7.3% | 0 |

Reading. The one-pass update gives exactly what the two-pass update gives (difference -0.0%
[-0.2, 0.1]) and is cheaper. Most of the lesson is in the correction, but the accepted tokens add
to it; the accepted tokens alone even hurt early in an answer and help late. Learning less often
costs gain roughly in step with the saving. Choosing the rounds with an early rejection is worse
than simply taking every 4th round, although it keeps twice as many updates. The last layer alone
is not enough.

Second round (jobs `cheap2-opd-0..3`, 100 answers), each against full one-pass learning on the
same answers, beyond 8,000 tokens: starting to learn only at 2,000 tokens -0.0%, only at 4,000
tokens -0.3% (so what is learned early is of no use later); every 2nd round -2.8%; only the
attention weights -0.5% (they carry the whole gain); only the other weights -7.8%; a three times
larger step on every 4th round is worse than the usual step (-2.0%); sharper attention adds +1.4%
to learning on every 4th round.

## Result 4: what a round costs, by the length of the text (Nov 16)

Job `adapt-amount-0` (it ran `nexus/cost_check.py`), RTX A5000, milliseconds.

| tokens before the block | 1,000 | 4,000 | 8,000 | 16,000 | 24,000 | 32,000 |
|---|---|---|---|---|---|---|
| the target's check of a block | 48.2 | 48.6 | 55.2 | 97.7 | 137.0 | 176.8 |
| a draft, plain drafter | 8.1 | 8.3 | 8.3 | 8.6 | 9.5 | 11.0 |
| a draft with the extra weights attached (cheaper way) | 10.0 | 10.0 | 10.0 | 10.1 | 10.4 | 11.7 |
| a draft that keeps its gradients | 13.9 | 14.0 | 13.9 | 13.6 | 13.8 | 14.0 |
| the update from that draft, all extra weights | 18.0 | 18.1 | 18.1 | 17.2 | 18.2 | 22.3 |
| the same, last two layers / last layer only | 7.4 / 3.8 | 7.4 / 3.8 | 7.4 / 3.6 | 7.4 / 4.2 | 8.2 / 4.9 | 9.8 / 5.7 |
| the update when it runs at the same time as the next round: time added | 5.1 | 5.0 | 4.7 | 8.3 | 14.8 | 21.3 |

Reading. The target's check is flat up to about 4,000 tokens and then grows with the text: it
takes 3.7 times as long at 32,000 tokens as at 1,000. The update of the drafter stays near 18 ms.
So one update after every round makes a round about 39% longer early in an answer, 20% longer at
16,000 tokens and 13% longer at 32,000. Running the update at the same time as the next round (on
a second stream of the same GPU) hides most of its cost while the text is short, and none of it
at 32,000 tokens, where the check already keeps the GPU busy. So that trick helps only where
learning has little to gain; we do not build it for now.

## Result 5: real tokens per second, with the target in the loop (Nov 16-17)

`nexus/live_speed.py` decodes the same prompts once per setting. Prompts: validation prompts
whose saved answers are at least 8,000 tokens long. Every setting wrote exactly the plain
drafter's tokens (decoding is lossless). Gains are tokens per second over the plain drafter, by
how far into the answer the round began; "16k+" is only the last few hundred tokens of each answer.

Better drafter (job `live-speed-0`, 8 answers; plain: 78.4 tokens per second):

| setting | 0-2k | 2k+ | 8k+ | 16k+ | all |
|---|---|---|---|---|---|
| sharper attention (free) | -0.2% | +5.0% | +8.0% | +6.7% | +4.4% |
| extra weights taught ahead of time, fixed | -0.6% | +13.6% | +18.5% | +15.3% | +11.7% |
| learning on every round (one pass) | -26.4% | -10.0% | -0.2% | +27.9% | -12.3% |
| learning on every 4th round | -9.9% | +0.7% | +6.9% | +25.4% | -0.6% |

Public drafter (job `adapt-amount-1`, 6 answers; plain: 40.8 tokens per second):

| setting | 0-2k | 2k+ | 8k+ | 16k+ | all |
|---|---|---|---|---|---|
| sharper attention (free) | -0.0% | +25.3% | +34.6% | +36.0% | +22.2% |
| extra weights taught ahead of time, fixed | +10.1% | +55.4% | +71.6% | +71.1% | +49.3% |
| learning on every round (one pass) | -23.6% | +20.5% | +42.3% | +79.8% | +13.9% |
| learning on every 4th round | -7.8% | +32.9% | +52.6% | +86.3% | +27.3% |

Reading. The measured rounds match the cost table (a round with an update is about 22 ms
longer). On answers of up to 16,384 tokens, learning on every round is slower than the plain
drafter on the better drafter and behind the free rule on the public one; learning on every 4th
round is about level with the free rule. The weights taught ahead of time are the fastest on both
drafters. The one place where learning during the answer is ahead is the last stretch, beyond
16,000 tokens (better drafter: +25 to +28% against +15% for the weights taught ahead of time and
+7% for the free rule; public drafter: +80 to +86% against +71% and +36%). That stretch is a few
hundred tokens per answer here, so it is a hint and not a result.

## Result 6: a short spell of learning does not last (Nov 17)

Jobs `burst-eval-0..3` (public drafter, 97 answers) and `burst-opd-0..3` (better drafter, 73 of
100 answers when read). The drafter learns only while the answer is between two lengths and keeps
what it learned afterwards. Gains beyond 8,000 tokens, over the plain drafter:

| the drafter learns while the answer has ... tokens | public drafter | better drafter |
|---|---|---|
| always (for comparison) | +71.3% | +20.0% |
| never, sharper attention instead (for comparison) | +37.5% | +7.8% |
| 2,000 to 4,000 | +4.0% | -0.1% |
| 4,000 to 5,000 | +14.2% | +0.6% |
| 4,000 to 6,000 | +43.7% | +2.9% |
| 4,000 to 8,000 | +58.3% | +10.4% |
| 8,000 to 10,000 | +55.8% | +10.9% |
| 4,000 to 5,000 and 8,000 to 9,000 | +52.4% | +7.4% |

Reading. What the drafter learns at one length helps at that length and fades as the answer
grows: learning at 2,000 to 4,000 tokens is worth nothing beyond 8,000. On the public drafter two
short spells keep about three quarters of the gain; on the better drafter a spell keeps at most
half of it. So the drafter has to keep learning as the answer grows; one spell is not enough.
(Fixed weights taught ahead of time do work at every length up to the lengths they were taught
on, because they were taught on all of those lengths at once.)

## Result 7: the two controls for teaching ahead of time (Nov 17)

Public drafter, 100 validation answers, fixed weights, tokens per round over the plain drafter.

How many answers are needed (jobs `amount2-0..3`): after 10 training answers +22.3%, after 30
+23.4%, after 100 +26.7%, after 150 +27.5%. Ten answers give most of it.

Does it carry over to other kinds of answers (jobs `kinds2-0..3`): weights taught on writing and
chat only give +24.1% on old AIME problems and +22.2% on hard MATH problems, which they never
saw (weights taught on math only: +36.5% and +33.1%). On writing and chat everything gains
little (+0.3% to +4.8%), because those answers are short. So about two thirds of the gain is not
about the kind of answer at all.

## Where this leaves the main line (Nov 17)

On answers of up to 16,384 tokens, learning during the answer loses in real time to weights that
were taught once (from as few as ten answers, of any kind) and then kept fixed, on both drafters;
on the better drafter it also loses to doing nothing. Everything we tried to make it cheap gives
up gain about in step with the saving. What the drafter learns during an answer is mostly a
repair of its attention for the current length of the text. Where learning during the answer can
still win is beyond the lengths that anything taught ahead of time has seen. The replays of the
benchmark answers of up to 32,768 tokens test that.

## Still to come

- Answers of up to 32,768 new tokens on the benchmark problems (AIME 2024, AIME 2025, hardest
  MATH-500; 194 answers, 2.2 million tokens), replayed with both drafters (`long-eval`,
  `long-opd`): plain, sharper attention, weights taught ahead of time, learning on every round,
  on every 4th round, only once the answer is 8,000 tokens long, and in four short spells. The
  settings were fixed before any of these results existed.
- The long validation set (`val32-decode`: 12 validation answers of up to 32,768 tokens) replayed
  with both drafters (`tune32-opd`, `tune32-eval`): choices about very long answers are made on
  it, among them fixed weights that start to learn at 16,000 tokens.
- Gate on Dec 6: continue only if learning during the answer shows a clear gain in real speed
  on the better drafter; otherwise stop and write up what the measurements show.

## In the back: teaching the extra weights ahead of time

The same extra weights can be trained once on other answers and then kept fixed. This is not
learning from the drafter's own draft during the answer, so it is not the main line, but it is
the strongest control for it. On the public drafter (100 validation answers): fixed weights from
61 training answers (about 1 GPU-hour) give +25.9% [20.8, 30.4], more than learning during the
answer (+21 to 22%), and learning during the answer on top of them adds +1.0% [0.3, 2.2]. On the
better drafter (69 training answers the weights had not seen yet, job `lora-offline-g`): fixed
weights +4.9% (+19.2% beyond 8,000 tokens); with learning during the answer on top of them +5.6%
(+20.1%), so learning adds about +0.7% to them. So most of what the drafter learns during an
answer is not about that answer: it can be learned once, ahead of time, and then costs nothing
while decoding. Two controls say when that works: `kinds2` (weights taught on writing and chat
only, or on math only, tested on every kind) and `amount2` (weights after 10, 30, 100 and 150
training answers). Where teaching ahead of time does not reach (content or lengths it never saw)
is where learning during the answer has to earn its cost.

## Caveats

- Our version of the published method is a re-implementation with low-rank adapters and a
  learning rate we tuned; the paper does not give its learning rate.
- The better drafter's training data includes MATH and AoPS problems, so our math answers may
  be closer to what it was trained on than to what the public drafter was trained on.
- SharpDraft's comparison with learning used a learning rate of 1e-3 and an update every 5
  rounds; we found 1e-3 too high in our setting (+8.3% against +21% at 3e-4 on the public drafter).
