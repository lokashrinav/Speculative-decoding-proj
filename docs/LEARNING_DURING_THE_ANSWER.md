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

## Result 3: cheaper ways to learn, on the better drafter (Nov 16, 21 of 100 answers so far)

Jobs `cheap-opd-0..3`. Each line is compared with full learning on the same answers.

| variant | against full learning | beyond 8,000 tokens | updates per round |
|---|---|---|---|
| one pass (the draft keeps its gradients, no second run of the drafter) | -0.0% [-0.3, 0.1] | +0.0% | 1.00 |
| the correction alone | -1.8% [-2.5, -1.0] | -4.3% | 0.96 |
| the accepted tokens alone | -6.8% [-9.1, -4.8] | -11.4% | 0.79 |
| every 4th round | -3.3% [-5.4, -1.2] | -9.7% | 0.25 |
| only after an early rejection (at most 2 tokens accepted) | -4.4% [-6.1, -2.2] | -13.6% | 0.50 |
| only the last layer's extra weights | -4.8% [-7.4, -2.0] | -15.4% | 1.00 |

Reading so far. The one-pass update loses nothing and is cheaper (about 20 ms instead of 31 ms
on an A5000, where a plain round takes about 68 ms). Most of the lesson is in the correction,
but the accepted tokens add to it. Learning less often costs gain roughly in step with the
saving, and choosing the rounds with an early rejection is worse than simply taking every 4th
round. The last layer alone is not enough.

Still open here: the full 100 answers, the second round (`cheap2-opd`: start learning only once
the answer is long, every 2nd / 8th round, a larger step, attention weights only), the cost of
every piece of a round (`cost-check`, also an update that runs at the same time as the next
check), and real tokens per second with the target in the loop (`live-speed`).

## Still to come

- Answers of up to 32,768 new tokens on the benchmark problems (AIME 2024, AIME 2025, hardest
  MATH-500): `bench3-decode-0..3`, then replays with both drafters (`long-eval`, `long-opd`).
  This is the test of where learning during the answer matters most: far into very long answers.
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
